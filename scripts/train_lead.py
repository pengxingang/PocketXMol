'''
train_energy.py: the loss is that of the orginall diffusion adding the NLL of the bond predictor
we want the bond predictor to lead the reconstruction to maximize predicter bond probabilities
'''

import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import shutil
import argparse
from tqdm import tqdm
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from torch_geometric.loader import DataLoader

from models.maskfill import MolDiff, MolDiffNoEdge
from models.bond_predictor import BondPredictor
from utils.dataset import get_dataset
from utils.transforms import FeaturizeMol, Compose, FeaturizeMaxMol
from utils.misc import *
from utils.train import *
# from apex import amp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/lead/lead_uncertainty.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)

    # Logging
    log_root = args.logdir+'_vscode' if sys.argv[0].startswith('/data') else args.logdir
    log_dir = get_new_log_dir(log_root, prefix=config_name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    for script_dir in ['scripts', 'utils', 'models', 'notebooks']:
        shutil.copytree(script_dir, os.path.join(log_dir, script_dir))


    # Transforms
    if 'max_size' not in config.transform:
        featurizer = FeaturizeMol(config.chem.atomic_numbers, config.chem.mol_bond_types,
                                use_mask_node=config.transform.use_mask_node,
                                use_mask_edge=config.transform.use_mask_edge
                                )
    else: # variable molecule size
        featurizer = FeaturizeMaxMol(config.chem.atomic_numbers, config.chem.mol_bond_types,
                                use_mask_node=config.transform.use_mask_node, use_mask_edge=config.transform.use_mask_edge,
                                max_size=config.transform.max_size,
                                )
    transform = Compose([
        featurizer,
    ])

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config = config.dataset,
        transform = transform,
    )
    train_set, val_set = subsets['train'], subsets['val']
    train_iterator = inf_iterator(DataLoader(
        train_set, 
        batch_size = config.train.batch_size, 
        shuffle = True,
        num_workers = config.train.num_workers,
        pin_memory = config.train.pin_memory,
        follow_batch = featurizer.follow_batch,
        exclude_keys = featurizer.exclude_keys,
    ))
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=featurizer.follow_batch, exclude_keys=featurizer.exclude_keys)

    # Bond predictor
    logger.info('Building bond predictor...')
    ckpt_bond = torch.load(config.bond_predictor, map_location=args.device)
    bond_predictor = BondPredictor(
        config=ckpt_bond['config'].model,
        num_node_types=featurizer.num_node_types,
        num_edge_types=featurizer.num_edge_types
    ).to(args.device)
    bond_predictor.load_state_dict(ckpt_bond['model'])
    bond_predictor.eval()
    
    logger.info('Building model...')
    # if config.model.name == 'diffusion':
    #     model = MolDiff(
    #         config=config.model,
    #         num_node_types=featurizer.num_node_types,
    #         num_edge_types=featurizer.num_edge_types
    #     ).to(args.device)
    # elif config.model.name == 'diffusion_noedge':
    model = MolDiffNoEdge(
        config=config.model,
        num_node_types=featurizer.num_node_types,
        lead_type=config.model.lead_type,
        bond_predictor=bond_predictor,
    ).to(args.device)
    params_to_update = [p for n, p in model.named_parameters() if 'bond_predictor' not in n]
    print('Num of trainable parameters is', np.sum([p.numel() for p in params_to_update if p.requires_grad]))

    # Optimizer and scheduler
    # optimizer = get_optimizer(config.train.optimizer, model)
    cfg_opt = config.train.optimizer
    optimizer = torch.optim.AdamW(params_to_update,
                    lr=cfg_opt.lr, weight_decay=cfg_opt.weight_decay, betas=(cfg_opt.beta1, cfg_opt.beta2))
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=config.train.use_amp)
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    def train(it):
        # model.train()  has been moved to the end of validation function for efficiency
        optimizer.zero_grad(set_to_none=True)
        batch = next(train_iterator).to(args.device)
        
        pos_noise = torch.randn_like(batch.node_pos) * config.train.pos_noise_std
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config.train.use_amp):
            loss_dict = model.get_loss(
                # compose
                node_type = batch.node_type,
                node_pos = batch.node_pos + pos_noise,
                batch_node = batch.node_type_batch,
                halfedge_type = batch.halfedge_type,
                halfedge_index = batch.halfedge_index,
                batch_halfedge = batch.halfedge_type_batch,
                num_mol = batch.num_graphs,
            )
        loss = loss_dict['loss']
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()

        log_info = '[Train] Iter %d | ' % it + ' | '.join([
            '%s: %.6f' % (k, v.item()) for k, v in loss_dict.items()
        ])
        logger.info(log_info)
        for k, v in loss_dict.items():
            writer.add_scalar('train/%s' % k, v.item(), it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad', orig_grad_norm, it)
        writer.flush()

    def validate(it):
        sum_n =  0   # num of loss
        sum_loss_dict = {} 
        if hasattr(model, 'lead_type') and (model.lead_type in ['mul', 'uncertainty']):
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config.train.use_amp):
                    loss_dict = model.get_loss(
                        # compose
                        node_type = batch.node_type,
                        node_pos = batch.node_pos,
                        batch_node = batch.node_type_batch,
                        halfedge_type = batch.halfedge_type,
                        halfedge_index = batch.halfedge_index,
                        batch_halfedge = batch.halfedge_type_batch,
                        num_mol = batch.num_graphs,
                    )
                if len(sum_loss_dict) == 0:
                    sum_loss_dict = {k: v.item() for k, v in loss_dict.items()}
                else:
                    for key in sum_loss_dict.keys():
                        sum_loss_dict[key] += loss_dict[key].item()
                sum_n += 1
        else:
            with torch.no_grad():
                model.eval()
                for batch in tqdm(val_loader, desc='Validate'):
                    batch = batch.to(args.device)
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config.train.use_amp):
                        loss_dict = model.get_loss(
                            # compose
                            node_type = batch.node_type,
                            node_pos = batch.node_pos,
                            batch_node = batch.node_type_batch,
                            halfedge_type = batch.halfedge_type,
                            halfedge_index = batch.halfedge_index,
                            batch_halfedge = batch.halfedge_type_batch,
                            num_mol = batch.num_graphs,
                        )
                    if len(sum_loss_dict) == 0:
                        sum_loss_dict = {k: v.item() for k, v in loss_dict.items()}
                    else:
                        for key in sum_loss_dict.keys():
                            sum_loss_dict[key] += loss_dict[key].item()
                    sum_n += 1
        
        # finish all batches
        avg_loss_dict = {k: v / sum_n for k, v in sum_loss_dict.items()}
        avg_loss = avg_loss_dict['loss']
        # update lr scheduler
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        log_info = '[Validate] Iter %d | ' % it + ' | '.join([
            '%s: %.6f' % (k, v) for k, v in avg_loss_dict.items()
        ])
        logger.info(log_info)
        for k, v in avg_loss_dict.items():
            writer.add_scalar('val/%s' % k, v, it)
        writer.flush()
        return avg_loss

    try:
        model.train()
        for it in range(1, config.train.max_iters+1):
            try:
                train(it)
            except RuntimeError as e:
                logger.error('Runtime Error ' + str(e))
                logger.error('Skipping Iteration %d' % it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                validate(it)
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }, ckpt_path)
                model.train()
    except KeyboardInterrupt:
        logger.info('Terminating...')
        

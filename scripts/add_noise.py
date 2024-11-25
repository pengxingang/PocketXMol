import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import shutil
import argparse
from tqdm import tqdm
import torch
from torch.nn import functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
# from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from torch_geometric.loader import DataLoader

from models.maskfill import MolDiff, MolDiffNoEdge, MolDiffGuide
from models.bond_predictor import BondPredictor
from utils.dataset import get_dataset
from utils.transforms import FeaturizeMol, Compose, FeaturizeMaxMol
from utils.misc import *
from utils.train import *
from models.sample import seperate_outputs_no_traj
# from apex import amp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/add_noise/tomask_early400.yml')
    # parser.add_argument('--config', type=str, default='./configs/add_noise/tomask_absorb.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)

    # Logging
    # log_root = args.logdir+'_vscode' if sys.argv[0].startswith('/data') else args.logdir
    # log_dir = get_new_log_dir(log_root, prefix=config_name)
    # ckpt_dir = os.path.join(log_dir, 'checkpoints')
    # os.makedirs(ckpt_dir, exist_ok=True)
    # logger = get_logger('train', log_dir)
    # writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    # logger.info(args)
    # logger.info(config)
    # shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    # for script_dir in ['scripts', 'utils', 'models', 'notebooks']:
    #     shutil.copytree(script_dir, os.path.join(log_dir, script_dir))


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
    # logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config = config.dataset,
        transform = transform,
    )
    train_set, val_set = subsets['train'], subsets['val']
    train_iterator = DataLoader(
        train_set, 
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = config.train.num_workers,
        pin_memory = config.train.pin_memory,
        follow_batch = featurizer.follow_batch,
        exclude_keys = featurizer.exclude_keys,
    )
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False,
                            follow_batch=featurizer.follow_batch, exclude_keys=featurizer.exclude_keys)

    # Model
    # logger.info('Building model...')
    if config.model.name == 'diffusion':
        model = MolDiff(
            config=config.model,
            num_node_types=featurizer.num_node_types,
            num_edge_types=featurizer.num_edge_types
        ).to(args.device)
    elif config.model.name == 'diffusion_noedge':
        model = MolDiffNoEdge(
            config=config.model,
            num_node_types=featurizer.num_node_types,
            # num_edge_types=featurizer.num_edge_types
        ).to(args.device)
    elif config.model.name == 'diffusion_guide':
        model = MolDiffGuide(
            config=config.model,
            num_node_types=featurizer.num_node_types,
            num_edge_types=featurizer.num_edge_types-1  # for consistency (use_edge_mask is True in config)
        ).to(args.device)
    print('Num of trainable parameters is', np.sum([p.numel() for p in model.parameters() if p.requires_grad]))

    # Bond predictor
    if 'bond_predictor' in config:
        # logger.info('Building bond predictor...')
        ckpt_bond = torch.load(config.bond_predictor, map_location=args.device)
        bond_predictor = BondPredictor(ckpt_bond['config']['model'],
                featurizer.num_node_types,
                featurizer.num_edge_types-1 # note: bond_predictor not use edge mask
        ).to(args.device)
        bond_predictor.load_state_dict(ckpt_bond['model'])
        bond_predictor.eval()
    else:
        bond_predictor = None


    def add_noise(batch, time):
        # model.train()  has been moved to the end of validation function for efficiency
        with torch.no_grad():
            noised_mol = model.add_noise(
                # compose
                node_type = batch.node_type,
                node_pos = batch.node_pos,
                batch_node = batch.node_type_batch,
                halfedge_type = batch.halfedge_type,
                halfedge_index = batch.halfedge_index,
                batch_halfedge = batch.halfedge_type_batch,
                num_mol = batch.num_graphs,
                t=time,
            )
            noised_mol = [n.cpu().numpy() for n in noised_mol]
            output_list = seperate_outputs_no_traj(noised_mol, batch.num_graphs,
                            batch.node_type_batch.cpu().numpy(),
                            batch.halfedge_index.cpu().numpy(),
                            batch.halfedge_type_batch.cpu().numpy())
        return output_list

    def no_noise(batch):
        # model.train()  has been moved to the end of validation function for efficiency
        with torch.no_grad():
            # compose
            node_type = F.one_hot(batch.node_type, num_classes=featurizer.num_node_types).float()
            node_pos = batch.node_pos.float()
            halfedge_type = F.one_hot(batch.halfedge_type, num_classes=featurizer.num_edge_types).float()
            noised_mol = [node_type, node_pos, halfedge_type]
            # batch_node = batch.node_type_batch,
            # halfedge_index = batch.halfedge_index,
            # batch_halfedge = batch.halfedge_type_batch,
            # num_mol = batch.num_graphs,
            # t=time,

            noised_mol = [n.cpu().numpy() for n in noised_mol]
            output_list = seperate_outputs_no_traj(noised_mol, batch.num_graphs,
                            batch.node_type_batch.cpu().numpy(),
                            batch.halfedge_index.cpu().numpy(),
                            batch.halfedge_type_batch.cpu().numpy())
        return output_list



    save_dir_func = lambda x: f'./outputs/add_noise/{config_name}/t_{x}'
    try:
        # no noise
        save_dir = save_dir_func(0)
        os.makedirs(save_dir, exist_ok=True)
        i = 0
        for batch in tqdm(val_loader, total=len(val_loader), desc=f'no noise'):
        # for batch in tqdm(train_iterator, total=len(train_iterator), desc=f'time {time}'):
            try:
                noised_list = no_noise(batch.to(args.device))
                # save
                torch.save(noised_list, os.path.join(save_dir, f'noised_list_{i}.pt'))
                i += 1
            except RuntimeError as e:
                print('Runtime Error ' + str(e))

        for time in np.arange(0, 1000, 100).tolist() + [999]:
        # for time in [999]:
            save_dir = save_dir_func(time+1)
            os.makedirs(save_dir, exist_ok=True)
            i = 0
            for batch in tqdm(val_loader, total=len(val_loader), desc=f'time {time}'):
            # for batch in tqdm(train_iterator, total=len(train_iterator), desc=f'time {time}'):
                try:
                    noised_list = add_noise(batch.to(args.device), time)
                    # save
                    torch.save(noised_list, os.path.join(save_dir, f'noised_list_{i}.pt'))
                    i += 1
                except RuntimeError as e:
                    print('Runtime Error ' + str(e))
    except KeyboardInterrupt:
        print('Terminating...')
        

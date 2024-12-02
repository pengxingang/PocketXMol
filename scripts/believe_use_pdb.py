from copy import deepcopy
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import sys
sys.path.append('.')
import argparse
import torch
import torch.utils.tensorboard
from torch_geometric.loader import DataLoader


from scripts.train_pl import DataModule
from models.maskfill import PMAsymDenoiser
from models.sample import seperate_outputs2, sample_loop3
from utils.transforms import *
from utils.misc import *
from utils.reconstruct import *
from utils.dataset import RegenDataset
from utils.sample_noise import get_sample_noiser
# from process.utils_process import extract_pocket, get_pocmol_data, add_pep_bb_data, get_peptide_info
# from utils.parser import parse_conf_list, PDBProtein
from evaluate.evaluate_mols import get_dir_from_prefix
from process.utils_process import get_input_from_file


def print_pool_status(pool, logger):
    logger.info('[Pool] Succ/Nonstd/Incomp/Bad: %d/%d/%d/%d' % (
        len(pool.succ), len(pool.nonstd), len(pool.incomp), len(pool.bad)
    ))

is_vscode = False
if os.environ.get("TERM_PROGRAM") == "vscode":
    is_vscode = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--exp_name', type=str, default='PDL1_hot135E_free_CCOOH_20231222_230002')
    parser.add_argument('--exp_name', type=str, default='PDL1_gen20_20231224_143110')
    parser.add_argument('--result_root', type=str, default='./outputs_use/PDL1/dock_generated')
    # parser.add_argument('--result_root', type=str, default='outputs_vscode_use/PDL1/pd1hot136E/PDL1/pd1hot136E')
    # parser.add_argument('--result_root', type=str, default='outputs_vscode_use/PDL1/dock_generated/PDL1/dock_generated')
    parser.add_argument('--config', type=str,
        default='./configs/multi_gpus/use/PDL1/belief.yml')
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--batch_size', type=int, default=0)
    args = parser.parse_args()
    
    gen_path = get_dir_from_prefix(args.result_root, args.exp_name)
    gen_name = os.path.basename(gen_path)

    # # Load configs
    config = make_config(args.config)
    seed_all(config.sample.seed)
    config_name = os.path.basename(args.config).replace('.yml', '')
    belief_name = config_name
    ranking_path = os.path.join(gen_path, 'ranking')
    os.makedirs(ranking_path, exist_ok=True)
    # load ckpt and train config
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    cfg_dir = os.path.dirname(config.model.checkpoint).replace('checkpoints', 'train_config')
    train_config = os.listdir(cfg_dir)
    train_config = make_config(os.path.join(cfg_dir, ''.join(train_config)))

    batch_size = config.sample.batch_size if args.batch_size == 0 else args.batch_size
    
    log_dir = gen_path
    logger = get_logger('believe', log_dir)
    logger.info('Load from %s...' % config.model.checkpoint)

    # # Transform
    logger.info('Loading data placeholder...')
    dm = DataModule(train_config)
    featurizer_list = dm.get_featurizers()
    featurizer = featurizer_list[-1]  # for mol decoding
    in_dims = dm.get_in_dims()
    task_trans = get_transforms(config.task.transform, mode='believe')
    noiser = get_sample_noiser(config.noise, in_dims['num_node_types'], in_dims['num_edge_types'],
                               mode='sample',device=args.device, ref_config=train_config.noise)
    transforms = featurizer_list + [task_trans]
    transforms = Compose(transforms)
    follow_batch = sum([getattr(t, 'follow_batch', []) for t in transforms.transforms], [])
    exclude_keys = sum([getattr(t, 'exclude_keys', []) for t in transforms.transforms], [])
    
    # # Data loader
    logger.info('Loading dataset...')    
    # data
    test_set = RegenDataset(gen_path, task=config.task.name, file2input=get_input_from_file, transforms=transforms)
    test_loader = DataLoader(test_set, batch_size, shuffle=False,
                            num_workers = train_config.train.num_workers,
                            pin_memory = train_config.train.pin_memory,
                            follow_batch=follow_batch, exclude_keys=exclude_keys)

    # # Model
    logger.info('Loading diffusion model...')
    if train_config.model.name == 'pm_asym_denoiser':
        model = PMAsymDenoiser(config=train_config.model, **in_dims).to(args.device)
    model.load_state_dict({k[6:]:value for k, value in ckpt['state_dict'].items() if k.startswith('model.')}) # prefix is 'model'
    model.eval()
    
    logger.info('Start sampling for confidence')
    info_keys = ['data_id']
    df_belief_list = []
    for batch in tqdm(test_loader, desc='Making belief...'):
        
        # # prepare batch then sample
        batch = batch.to(args.device)
        # outputs, trajs = sample_loop2(batch, model, noiser, args.device)
        batch, outputs, trajs = sample_loop3(batch, model, noiser, args.device, off_tqdm=True)
        
        # # decode outputs to molecules
        data_list = [{key:batch[key][i] for key in info_keys} for i in range(len(batch))]
        # try:
        generated_list, outputs_list, traj_list_dict = seperate_outputs2(batch, outputs, trajs, off_tqdm=True)
        # except:
        #     continue
        
        # # post process generated data for the batch
        mol_info_list = []
        for output, data in zip(outputs_list, data_list):
            
            # save output
            # output = output['confidence_pos']

            # log info 
            data_id = data['data_id']
            filename = data_id[data_id.index(gen_name)+len(gen_name)+1:]
            info_dict = {
                'filename': filename,
                belief_name: torch.mean(output['confidence_pos']).item()
            }
            df_belief_list.append(info_dict)
            
            
    df_belief = pd.DataFrame(df_belief_list)
    df_belief.to_csv(os.path.join(ranking_path, f'{config_name}.csv'), index=False)
    
    
    
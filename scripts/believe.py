from copy import deepcopy
import os

import sys
sys.path.append('.')
import shutil
import argparse
import gc
import torch
import torch.utils.tensorboard
import numpy as np
from itertools import cycle
from easydict import EasyDict
from tqdm.auto import tqdm
from rdkit import Chem
from torch_geometric.loader import DataLoader
from collections import OrderedDict

from scripts.train_pl import DataModule
from models.maskfill import PMAsymDenoiser
from models.sample import seperate_outputs2, sample_loop3
from utils.transforms import *
from utils.misc import *
from utils.reconstruct import *
from utils.sample_noise import get_sample_noiser
from evaluate.evaluate_mols import get_dir_from_prefix


def print_pool_status(pool, logger):
    logger.info('[Pool] Succ/Incomp/Bad: %d/%d/%d' % (
        len(pool.succ), len(pool.incomp), len(pool.bad)
    ))

is_vscode = False
if os.environ.get("TERM_PROGRAM") == "vscode":
    is_vscode = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='base_pxm')
    parser.add_argument('--result_root', type=str, default='./outputs_test/dock_posebusters')
    parser.add_argument('--config', type=str, default='configs/sample/confidence/tuned_cfd.yml', help='confidence config')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()
    
    gen_path = get_dir_from_prefix(args.result_root, args.exp_name)
    gen_name = os.path.basename(gen_path)
    belief_name = os.path.basename(args.config).split('.')[0]
    save_path = os.path.join(gen_path, f'{belief_name}.csv')
    # # Load config
    be_config = make_config(args.config)
    seed_all(be_config.sample.seed)
    
    # # load config of original sampling
    sample_config_file = [f for f in os.listdir(gen_path) if f.endswith('.yml')]
    assert len(sample_config_file) == 1, 'sample config file is not 1 ge'
    sa_config = make_config(os.path.join(gen_path, sample_config_file[0]))
    
    # load ckpt (of sampling) and train config
    if 'model' in be_config:
        model_ckpt_path = be_config.model.checkpoint
    else:
        model_ckpt_path = sa_config.model.checkpoint  # the same model as sampling
    ckpt = torch.load(model_ckpt_path, map_location=args.device)
    cfg_dir = os.path.dirname(model_ckpt_path).replace('checkpoints', 'train_config')
    train_config = os.listdir(cfg_dir)
    train_config = make_config(os.path.join(cfg_dir, ''.join(train_config)))

    batch_size = sa_config.sample.batch_size if args.batch_size == 0 else args.batch_size
    
    # # Logging
    log_dir = gen_path
    logger = get_logger('believe', log_dir)
    logger.info('Load from %s...' % model_ckpt_path)

    # df_path = os.path.join(log_dir, 'gen_info.csv')

    # # Transform
    logger.info('Loading data placeholder...')
    dm = DataModule(train_config)
    featurizer_list = dm.get_featurizers()
    featurizer = featurizer_list[-1]  # for mol decoding
    in_dims = dm.get_in_dims()
    task_trans = get_transforms(be_config.task.transform, mode='believe')  # use belief task
    noiser = get_sample_noiser(be_config.noise, in_dims['num_node_types'], in_dims['num_edge_types'], # use belief noser
                               mode='sample',device=args.device, ref_config=train_config.noise)
    transforms_list = featurizer_list + [task_trans]
    be_config.task.db = sa_config.task.db

    # # Model
    logger.info('Loading diffusion model...')
    if train_config.model.name == 'pm_asym_denoiser':
        model = PMAsymDenoiser(config=train_config.model, **in_dims).to(args.device)
    model.load_state_dict({k[6:]:value for k, value in ckpt['state_dict'].items() if k.startswith('model.')}) # prefix is 'model'
    model.eval()
    
    df_belief_list = []
    num_repeats = sa_config.sample.num_repeats
    info_keys = ['data_id', 'filename']
    for i_repeat in range(num_repeats):
        
        # # Data loader
        logger.info(f'Loading dataset for repeat {i_repeat}')
        data_cfg = sa_config.data
        overwriter = OverwritePos(config=None,
                        gen_path=gen_path, i_repeat=i_repeat)
        transforms = Compose(transforms_list + [overwriter])
        test_set = TestTaskDataset(data_cfg.dataset, be_config.task,  # use belief task
                                mode='test',
                                split=getattr(data_cfg, 'split', None),
                                transforms=transforms)
        follow_batch = sum([getattr(t, 'follow_batch', []) for t in transforms.transforms], [])
        exclude_keys = sum([getattr(t, 'exclude_keys', []) for t in transforms.transforms], [])
        num_workers = train_config.train.num_workers if args.num_workers == -1 else args.num_workers
        test_loader = DataLoader(test_set, batch_size, shuffle=False,
                                num_workers = num_workers,
                                pin_memory = train_config.train.pin_memory,
                                follow_batch=follow_batch, exclude_keys=exclude_keys)

        # generating molecules
        logger.info(f'Generating molecules. Testset repeat {i_repeat}.')
        for batch in test_loader:

            # # prepare batch then sample
            data_list = batch.to_data_list()
            batch = batch.to(args.device)
            # outputs, trajs = sample_loop2(batch, model, noiser, args.device)
            batch, outputs, trajs = sample_loop3(batch, model, noiser, args.device)
            
            # # decode outputs to molecules
            # data_list = [{key:batch[key][i] for key in info_keys} for i in range(len(batch))]
            # try:
            generated_list, outputs_list, traj_list_dict = seperate_outputs2(batch, outputs, trajs)
            # except:
            #     continue
            
            # # post process generated data for the batch
            mol_info_list = []
            # for output, data in zip(outputs_list, data_list):
            for i_gen in range(len(batch)):
                gen_data = generated_list[i_gen]
                output = outputs_list[i_gen]
                data = data_list[i_gen]
                
                # save output
                data_id = data['data_id']
                filename = data['filename']
                # regen rmsd
                # orig_pos = data['node_pos'].detach().cpu().numpy()
                # regen_pos = gen_data['pos']
                # regen_rmsd = np.sqrt(np.mean(np.sum((orig_pos - regen_pos)**2, axis=-1)))
                # log info 
                info_dict = {
                    'filename': filename,
                    f'{belief_name}': torch.mean(output['confidence_pos']).item(),
                    # f'{belief_name}_rmsd': regen_rmsd,
                    'data_id': data_id,
                    'i_repeat': i_repeat,
                }
                df_belief_list.append(info_dict)
    df_belief = pd.DataFrame(df_belief_list)
    df_belief.to_csv(save_path, index=False)
        
    # print('Done. Results saved at %s' % result_path)
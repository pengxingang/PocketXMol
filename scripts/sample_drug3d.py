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
# from torch_geometric.data import Batch
from easydict import EasyDict
from tqdm.auto import tqdm
from rdkit import Chem
from torch_geometric.loader import DataLoader
from collections import OrderedDict

from scripts.train_pl import DataModule
from models.maskfill import PMAsymDenoiser
from models.sample import seperate_outputs2, sample_loop3, get_cfd_traj
from utils.transforms import *
from utils.misc import *
from utils.reconstruct import *
# from utils.chem import *
from utils.sample_noise import get_sample_noiser

def print_pool_status(pool, logger):
    logger.info('[Pool] Succ/Incomp/Bad: %d/%d/%d' % (
        len(pool.succ), len(pool.incomp), len(pool.bad)
    ))

is_vscode = False
if os.environ.get("TERM_PROGRAM") == "vscode":
    is_vscode = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_task', type=str, default='configs/sample/test/denovo_geom/base.yml', help='task config file')
    parser.add_argument('--config_model', type=str, default='configs/sample/pxm.yml', help='model config file')
    parser.add_argument('--outdir', type=str, default='outputs_test')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    # # Load configs
    config = make_config(args.config_task, args.config_model)
    if args.config_model is not None:
        config_name = os.path.basename(args.config_task).replace('.yml', '') 
        config_name += '_' + os.path.basename(args.config_model).replace('.yml', '')
    else:
        config_name = os.path.basename(args.config_task)[:os.path.basename(args.config_task).rfind('.')]
    seed = config.sample.seed  # + np.sum([ord(s) for s in args.outdir]+[ord(s) for s in args.config_task])
    seed_all(seed)
    # config.sample.complete_seed = seed.item()
    # load ckpt and train config
    ckpt = torch.load(config.model.checkpoint, map_location=args.device, weights_only=False)
    cfg_dir = os.path.dirname(config.model.checkpoint).replace('checkpoints', 'train_config')
    train_config = os.listdir(cfg_dir)
    train_config = make_config(os.path.join(cfg_dir, ''.join(train_config)))

    save_traj_prob = config.sample.save_traj_prob
    batch_size = config.sample.batch_size if args.batch_size == 0 else args.batch_size
    num_mols = getattr(config.sample, 'num_mols', int(1e10))
    num_repeats = getattr(config.sample, 'num_repeats', 1)
    # # Logging
    if is_vscode:  # for debug using vscode
        dir_names= os.path.dirname(args.config_task).split('/')
        is_sample = dir_names.index('sample')
        names = dir_names[is_sample+1:] + os.path.dirname(args.config_task).split('/')[1:]
        log_root = '/'.join(
            [args.outdir.replace('outputs', 'outputs_vscode')] + names
        )
        save_traj_prob = 1.0
        batch_size = 11
        num_mols = 100
        num_repeats = 2
    else:
        log_root = args.outdir
        os.makedirs(log_root, exist_ok=True)
        # remove bad result dir with the same name
        for file in os.listdir(log_root):
            if file.startswith(config_name):
                if not os.path.exists(os.path.join(log_root, file, 'samples_all.pt')):
                    print('Remove bad result dir:', file)
                    # shutil.rmtree(os.path.join(log_root, file))
                else:
                    print('Found existing result dir:', file)
                    # exit()
    log_dir = get_new_log_dir(log_root, prefix=config_name)
    logger = get_logger('sample', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info('Load from %s...' % config.model.checkpoint)
    logger.info(args)
    logger.info(config)
    save_config(config, os.path.join(log_dir, os.path.basename(args.config_task)))
    for script_dir in ['scripts', 'utils', 'models']:
        shutil.copytree(script_dir, os.path.join(log_dir, script_dir))
    sdf_dir = os.path.join(log_dir, 'SDF')
    os.makedirs(sdf_dir, exist_ok=True)
    df_path = os.path.join(log_dir, 'gen_info.csv')

    # # Transform
    logger.info('Loading data placeholder...')
    for samp_trans in config.get('transforms', {}).keys():  # overwirte transform config from sample.yml to train.yml
        if samp_trans in train_config.transforms.keys():
            train_config.transforms.get(samp_trans).update(
                config.transforms.get(samp_trans)
            )
    dm = DataModule(train_config)
    featurizer_list = dm.get_featurizers()
    featurizer = featurizer_list[-1]  # for mol decoding
    in_dims = dm.get_in_dims()
    task_trans = get_transforms(config.task.transform)
    is_ar = config.task.transform.name
    noiser = get_sample_noiser(config.noise, in_dims['num_node_types'], in_dims['num_edge_types'],
                               mode='sample',device=args.device, ref_config=train_config.noise)
    if 'variable_mol_size' in getattr(config, 'transforms', []):
        transforms = featurizer_list + [
            get_transforms(config.transforms.variable_mol_size), task_trans]
    else:
        transforms = featurizer_list + [task_trans]
    addition_transforms = [get_transforms(tr) for tr in config.data.get('transforms', [])]
    transforms = Compose(transforms + addition_transforms)
    follow_batch = sum([getattr(t, 'follow_batch', []) for t in transforms.transforms], [])
    exclude_keys = sum([getattr(t, 'exclude_keys', []) for t in transforms.transforms], [])
    
    # # Data loader
    logger.info('Loading dataset...')
    data_cfg = config.data
    num_workers = train_config.train.num_workers if args.num_workers == -1 else args.num_workers
    test_set = TestTaskDataset(data_cfg.dataset, config.task,
                               mode='test',
                               split=getattr(data_cfg, 'split', None),
                               transforms=transforms)
    test_loader = DataLoader(test_set, batch_size, shuffle=args.shuffle,
                            num_workers = num_workers,
                            pin_memory = train_config.train.pin_memory,
                            follow_batch=follow_batch, exclude_keys=exclude_keys)

    # # Model
    logger.info('Loading diffusion model...')
    if train_config.model.name == 'pm_asym_denoiser':
        model = PMAsymDenoiser(config=train_config.model, **in_dims).to(args.device)
    model.load_state_dict({k[6:]:value for k, value in ckpt['state_dict'].items() if k.startswith('model.')}) # prefix is 'model'
    model.eval()

    pool = EasyDict({
        'succ': [],
        'bad': [],
        'incomp': [],
    })
    info_keys = ['data_id', 'db', 'task', 'key']
    i_saved = 0
    # generating molecules
    logger.info('Start sampling... (n_repeats=%d, n_mols=%d)' % (num_repeats, num_mols))
    for i_repeat in range(num_repeats):
        logger.info(f'Generating molecules. Testset repeat {i_repeat}.')
        
        if ('overwrite_pos_repeat' in config.sample or  # overwrite pos. and different for each repeat. for linker desigin with unknown frag pos
            'overwrite_mol_repeat' in config.sample):  # overwrite mol. for mol optimize round >= 1
            if 'overwrite_pos_repeat' in config.sample:
                overwrite_pos_repeat = config.sample.overwrite_pos_repeat
                overwirter = OverwritePosRepeat(config=overwrite_pos_repeat, i_repeat=i_repeat)
                transforms = Compose(featurizer_list + [overwirter, task_trans] + addition_transforms)
            elif 'overwrite_mol_repeat' in config.sample:
                overwrite_mol_repeat = config.sample.overwrite_mol_repeat
                overwirter = OverwriteMolRepeat(config=overwrite_mol_repeat, i_repeat=i_repeat)
                transforms = Compose([overwirter] + featurizer_list + [task_trans] + addition_transforms)
            test_set = TestTaskDataset(data_cfg.dataset, config.task,
                               mode='test',
                               split=getattr(data_cfg, 'split', None),
                               transforms=transforms)
            test_loader = DataLoader(test_set, batch_size, shuffle=args.shuffle,
                                    num_workers = num_workers,
                                    pin_memory = train_config.train.pin_memory,
                                    follow_batch=follow_batch, exclude_keys=exclude_keys)
        
        for i_batch, batch in enumerate(test_loader):
            if i_saved >= num_mols:
                logger.info('Enough molecules. Stop sampling.')
                break
            
            # # prepare batch then sample
            batch = batch.to(args.device)
            # outputs, trajs = sample_loop2(batch, model, noiser, args.device)
            batch, outputs, trajs = sample_loop3(batch, model, noiser, args.device, is_ar=is_ar)
            
            # # decode outputs to molecules
            data_list = [{key:batch[key][i] for key in info_keys} for i in range(len(batch))]
            # try:
            generated_list, outputs_list, traj_list_dict = seperate_outputs2(batch, outputs, trajs)
            # except:
            #     continue
            
            # # post process generated data for the batch
            mol_info_list = []
            for i_mol in tqdm(range(len(generated_list)), desc='Post process generated mols'):
                # add meta data info
                mol_info = featurizer.decode_output(**generated_list[i_mol]) 
                mol_info.update(data_list[i_mol])  # add data info
                
                # reconstruct mols
                try:
                    rdmol = reconstruct_from_generated_with_edges(mol_info)
                    smiles = Chem.MolToSmiles(rdmol)
                    if '.' in smiles:
                        tag = 'incomp'
                        pool.incomp.append(mol_info)
                        logger.warning('Incomplete molecule: %s' % smiles)
                    else:
                        tag = ''
                        pool.succ.append(mol_info)
                        logger.info('Success: %s' % smiles)
                except MolReconsError:
                    pool.bad.append(mol_info)
                    logger.warning('Reconstruction error encountered.')
                    smiles = ''
                    tag = 'bad'
                    rdmol = create_sdf_string(mol_info)
                
                mol_info.update({
                    'rdmol': rdmol,
                    'smiles': smiles,
                    'tag': tag,
                    'output': outputs_list[i_mol],
                })
                
                # get traj
                p_save_traj = np.random.rand()  # save traj
                if p_save_traj <  save_traj_prob:
                    mol_traj = {}
                    for traj_who in traj_list_dict.keys():
                        traj_this_mol = traj_list_dict[traj_who][i_mol]
                        for t in range(len(traj_this_mol['node'])):
                            mol_this = featurizer.decode_output(
                                    node=traj_this_mol['node'][t],
                                    pos=traj_this_mol['pos'][t],
                                    halfedge=traj_this_mol['halfedge'][t],
                                    halfedge_index=generated_list[i_mol]['halfedge_index'],
                                    pocket_center=generated_list[i_mol]['pocket_center'],
                                )
                            mol_this = create_sdf_string(mol_this)
                            mol_traj.setdefault(traj_who, []).append(mol_this)
                            
                    mol_info['traj'] = mol_traj
                mol_info_list.append(mol_info)

            # # save sdf mols for the batch
            df_info_list = []
            for data_finished in mol_info_list:
                # save mol
                rdmol = data_finished['rdmol']
                tag = data_finished['tag']
                filename = str(i_saved) + (f'-{tag}' if tag else '') + '.sdf'
                if tag != 'bad':
                    Chem.MolToMolFile(rdmol, os.path.join(sdf_dir, filename))
                else:
                    with open(os.path.join(sdf_dir, filename), 'w+') as f:
                        f.write(rdmol)
                # save traj
                if 'traj' in data_finished:
                    for traj_who in data_finished['traj'].keys():
                        sdf_file = '$$$$\n'.join(data_finished['traj'][traj_who])
                        name_traj = filename.replace('.sdf', f'-{traj_who}.sdf')
                        with open(os.path.join(sdf_dir, name_traj), 'w+') as f:
                            f.write(sdf_file)
                i_saved += 1
                
                # save output
                output = data_finished['output']
                cfd_traj = get_cfd_traj(output['confidence_pos_traj'])  # get cfd
                cfd_pos = output['confidence_pos'].detach().cpu().numpy().mean()
                cfd_node = output['confidence_node'].detach().cpu().numpy().mean()
                cfd_edge = output['confidence_halfedge'].detach().cpu().numpy().mean()
                save_output = getattr(config.sample, 'save_output', [])
                if len(save_output) > 0:
                    output = {key: output[key] for key in save_output}
                    torch.save(output, os.path.join(sdf_dir, filename.replace('.sdf', '.pt')))

                # log info 
                info_dict = {
                    key: data_finished[key] for key in info_keys + ['smiles', 'tag']
                }
                info_dict.update({
                    'filename': filename,
                    'i_repeat': i_repeat,
                    'cfd_traj': cfd_traj,
                    'cfd_pos': cfd_pos,
                    'cfd_node': cfd_node,
                    'cfd_edge': cfd_edge,
                })

                df_info_list.append(info_dict)
        
            df_info_batch = pd.DataFrame(df_info_list)
            # # save df
            if os.path.exists(df_path):
                df_info = pd.read_csv(df_path)
                df_info = pd.concat([df_info, df_info_batch], ignore_index=True)
            else:
                df_info = df_info_batch
            df_info.to_csv(df_path, index=False)
            print_pool_status(pool, logger)
            
            # clean up
            del batch, outputs, trajs, mol_info_list[0:len(mol_info_list)]
            with torch.cuda.device(args.device):
                torch.cuda.empty_cache()
            gc.collect()


    # make dummy pool  (save disk space)
    dummy_pool = {key: ['']*len(value) for key, value in pool.items()}
    torch.save(dummy_pool, os.path.join(log_dir, 'samples_all.pt'))
    # torch.save(pool, os.path.join(log_dir, 'samples_all.pt'))

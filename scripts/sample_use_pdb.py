from copy import deepcopy
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
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
from Bio.SeqUtils import seq1
from Bio import PDB


from scripts.train_pl import DataModule
from models.maskfill import PMAsymDenoiser
from models.sample import seperate_outputs2, sample_loop3
from utils.transforms import *
from utils.misc import *
from utils.reconstruct import *
from utils.dataset import UseDataset
from utils.sample_noise import get_sample_noiser
from process.utils_process import extract_pocket, get_pocmol_data, add_pep_bb_data, get_peptide_info, get_input_from_file
from utils.parser import parse_conf_list, PDBProtein
torch.backends.cudnn.benchmark = False

def print_pool_status(pool, logger):
    logger.info('[Pool] Succ/Nonstd/Incomp/Bad: %d/%d/%d/%d' % (
        len(pool.succ), len(pool.nonstd), len(pool.incomp), len(pool.bad)
    ))

is_vscode = False
if os.environ.get("TERM_PROGRAM") == "vscode":
    is_vscode = True
    

def get_input_data(protein_path, ref_ligand_path, input_ligand, 
                   pocket_args={}, pocmol_args={}):
    
    if isinstance(input_ligand, list):
        input_ligand_mol, input_ligand_pep = input_ligand
    else:
        input_ligand_mol, input_ligand_pep = input_ligand, input_ligand
    
    # get pocket
    if ref_ligand_path is None:
        ref_ligand_path = input_ligand_mol
    pocket_pdb = extract_pocket(protein_path, ref_ligand_path, **pocket_args)
    
    # get input ligand
    # pocmol_data = get_pocmol_data(input_ligand_mol, pocket_pdb, **pocmol_args)
    pocmol_data = get_input_from_file(input_ligand_mol, pocket_pdb, **pocmol_args)
    
    # add peptide info
    if input_ligand_pep.endswith('.pdb'):
        pep_info = get_peptide_info(input_ligand_pep)
        assert torch.isclose(pocmol_data['pos_all_confs'][0], pep_info['peptide_pos'], 1e-2).all(), 'mol and pep atoms may not match'
    else:  #! NOTE: might be wrong and useless for some cases, e.g. dock
        pep_info = add_pep_bb_data(pocmol_data)
    pocmol_data.update(pep_info)
    # assert torch.isclose(pocmol_data['pos_all_confs'][0], pep_info['peptide_pos'], 1e-2).all(), 'mol and pep atoms may not match'
    return pocmol_data, pocket_pdb



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config_task', type=str, default='configs/sample/examples/pepdesign_base.yml', help='task config file')
    parser.add_argument('--config_task', type=str, default='configs/sample/examples/dockpep.yml', help='task config file')
    parser.add_argument('--config_model', type=str, default='configs/sample/pxm.yml', help='model config file')
    parser.add_argument('--outdir', type=str, default='./outputs_use')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--shuffle', type=bool, default=False)
    args = parser.parse_args()

    # # Load configs
    config = make_config(args.config_task, args.config_model)
    if args.config_model is not None:
        config_name = os.path.basename(args.config_task).replace('.yml', '') 
        config_name += '_' + os.path.basename(args.config_model).replace('.yml', '')
    else:
        config_name = os.path.basename(args.config_task)[:os.path.basename(args.config_task).rfind('.')]
    seed = config.sample.seed + np.sum([ord(s) for s in args.outdir]+[ord(s) for s in args.config_task])
    seed_all(seed)
    config.sample.complete_seed = seed.item()
    # load ckpt and train config
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    cfg_dir = os.path.dirname(config.model.checkpoint).replace('checkpoints', 'train_config')
    train_config = os.listdir(cfg_dir)
    train_config = make_config(os.path.join(cfg_dir, ''.join(train_config)))

    save_traj_prob = config.sample.save_traj_prob
    batch_size = config.sample.batch_size if args.batch_size == 0 else args.batch_size
    num_mols = getattr(config.sample, 'num_mols', int(1e10))
    num_repeats = getattr(config.sample, 'num_repeats', 1)
    # # Logging
    if is_vscode:
        dir_names= os.path.dirname(args.config_task).split('/')
        is_sample = dir_names.index('configs')
        names = dir_names[is_sample+1:]
        log_root = '/'.join(
            [args.outdir.replace('outputs', 'outputs_vscode')] + names
        )
        save_traj_prob = 1.0
        batch_size = 10
        num_mols = 100
        num_repeats = 1
    else:
        num_mols = config.sample.num_mols
        num_repeats = config.sample.num_repeats
        log_root = args.outdir
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
    pure_sdf_dir = os.path.join(log_dir, os.path.basename(log_dir) +'_SDF')
    os.makedirs(sdf_dir, exist_ok=True)
    os.makedirs(pure_sdf_dir, exist_ok=True)
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
    task_trans = get_transforms(config.task.transform, mode='use')
    noiser = get_sample_noiser(config.noise, in_dims['num_node_types'], in_dims['num_edge_types'],
                               mode='sample',device=args.device, ref_config=train_config.noise)
    if 'variable_sc_size' in getattr(config, 'transforms', []):
        transforms = featurizer_list + [
            get_transforms(config.transforms.variable_sc_size), task_trans]
    else:
        transforms = featurizer_list + [task_trans]
    transforms = Compose(transforms)
    follow_batch = sum([getattr(t, 'follow_batch', []) for t in transforms.transforms], [])
    exclude_keys = sum([getattr(t, 'exclude_keys', []) for t in transforms.transforms], [])
    
    # # Data loader
    logger.info('Loading dataset...')
    data_cfg = config.data
    
    # data
    data, pocket_block = get_input_data(
        data_cfg.protein_path, data_cfg.get('ref_ligand_path'), data_cfg.input_ligand,
        pocket_args=data_cfg.get('pocket_args', {}), pocmol_args=data_cfg.get('pocmol_args', {}))
    test_set = UseDataset(data, n=num_mols, task=config.task.name, transforms=transforms)

    test_loader = DataLoader(test_set, batch_size, shuffle=args.shuffle,
                            num_workers = train_config.train.num_workers,
                            pin_memory = train_config.train.pin_memory,
                            follow_batch=follow_batch, exclude_keys=exclude_keys)
    # save pocket
    with open(os.path.join(pure_sdf_dir, '0_pocket_block.pdb'), 'w') as f:
        f.write(pocket_block)

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
        'nonstd': [],
    })
    info_keys = ['data_id', 'db', 'task', 'key']
    i_saved = 0
    # generating molecules
    logger.info('Start sampling... (n_repeats=%d, n_mols=%d)' % (num_repeats, num_mols))
    
    try:
        for i_repeat in range(num_repeats):
            logger.info(f'Generating molecules. Testset repeat {i_repeat}.')
            for batch in test_loader:
                if i_saved >= num_mols:
                    logger.info('Enough molecules. Stop sampling.')
                    break
                
                # # prepare batch then sample
                batch = batch.to(args.device)
                # outputs, trajs = sample_loop2(batch, model, noiser, args.device)
                batch, outputs, trajs = sample_loop3(batch, model, noiser, args.device)
                
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
                        pdb_struc, rdmol = reconstruct_pdb_from_generated(mol_info, gt_path=data_cfg.input_ligand)
                        aaseq = seq1(''.join(res.resname for res in pdb_struc.get_residues()))
                        if rdmol is None:
                            rdmol = Chem.MolFromSmiles('')
                        smiles = Chem.MolToSmiles(rdmol)
                        if '.' in smiles:
                            tag = 'incomp'
                            pool.incomp.append(mol_info)
                            logger.warning('Incomplete molecule: %s' % aaseq)
                        elif 'X' in aaseq:
                            tag = 'nonstd'
                            pool.nonstd.append(mol_info)
                            logger.warning('Non-standard amino acid: %s' % aaseq)
                        else:  # nb
                            tag = ''
                            pool.succ.append(mol_info)
                            logger.info('Success: %s' % aaseq)
                    except MolReconsError:
                        pool.bad.append(mol_info)
                        logger.warning('Reconstruction error encountered.')
                        smiles = ''
                        aaseq = ''
                        tag = 'bad'
                        rdmol = create_sdf_string(mol_info)
                        pdb_struc = PDB.Structure.Structure('bad')

                    
                    mol_info.update({
                        'pdb_struc': pdb_struc,
                        'aaseq': aaseq,
                        'rdmol': rdmol,
                        'smiles': smiles,
                        'tag': tag,
                        'output': outputs_list[i_mol],
                    })
                    
                    # get traj
                    p_save_traj = np.random.rand()  # save traj
                    gt_struc = None
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

                # # save sdf/pdb mols for the batch
                df_info_list = []
                for data_finished in mol_info_list:
                    # save mol
                    pdb_struc = data_finished['pdb_struc']
                    rdmol = data_finished['rdmol']
                    tag = data_finished['tag']
                    filename = str(i_saved) + (f'-{tag}' if tag else '') + '.pdb'
                    filename_mol = filename.replace('.pdb', '_mol.sdf')
                    # save pdb
                    pdb_io = PDBIO()
                    pdb_io.set_structure(pdb_struc)
                    pdb_io.save(os.path.join(pure_sdf_dir, filename))
                    # save rdmol
                    if tag != 'bad':
                        Chem.MolToMolFile(rdmol, os.path.join(pure_sdf_dir, filename_mol))
                    else:
                        with open(os.path.join(pure_sdf_dir, filename_mol), 'w+') as f:
                            f.write(rdmol)
                    # save gt pdb
                    # db, data_id = data_finished['db'], data_finished['data_id']
                    # gt_pdb_path = f"data/{db}/files/peptides/{data_id}_pep.pdb"
                    # os.system(f"cp {gt_pdb_path} {os.path.join(sdf_dir, filename.replace('.pdb', '_gt.pdb'))}")
                    # save traj
                    if 'traj' in data_finished:
                        for traj_who in data_finished['traj'].keys():
                            sdf_file = '$$$$\n'.join(data_finished['traj'][traj_who])
                            name_traj = filename.replace('.pdb', f'-{traj_who}.sdf')
                            with open(os.path.join(sdf_dir, name_traj), 'w+') as f:
                                f.write(sdf_file)
                    i_saved += 1
                    # save output
                    output = data_finished['output']
                    save_output = getattr(config.sample, 'save_output', ['confidence_pos'])
                    if len(save_output) > 0:
                        output = {key: output[key] for key in save_output}
                        torch.save(output, os.path.join(sdf_dir, filename.replace('.pdb', '.pt')))

                    # log info 
                    info_dict = {
                        key: data_finished[key] for key in info_keys + ['aaseq', 'smiles', 'tag']
                    }
                    info_dict['filename'] = filename
                    info_dict['i_repeat'] = i_repeat
                    df_info_list.append(info_dict)
            
                df_info_batch = pd.DataFrame(df_info_list)
                # save df
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
    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt. Stop sampling.')
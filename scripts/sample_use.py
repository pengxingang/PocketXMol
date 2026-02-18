"""PocketXMol User-Friendly Sampling Script.

This script provides the main entry point for generating molecules using trained
PocketXMol models on user-provided protein and ligand data.

Usage:
    $ python scripts/sample_use.py \
        --config_task configs/sample/examples/dock_smallmol.yml \
        --config_model configs/sample/pxm.yml \
        --outdir outputs_examples \
        --device cuda:0

The script:
    1. Loads model checkpoint and configurations
    2. Processes input protein and ligand files
    3. Runs iterative denoising to generate molecules
    4. Saves results as SDF files with confidence scores
"""

# Standard library imports
import argparse
import gc
import os
import shutil
import sys
from itertools import cycle

# Third-party imports
import numpy as np
import torch
from Bio import PDB
from Bio.SeqUtils import seq1
from easydict import EasyDict
from rdkit import Chem
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

# Local imports
sys.path.append('.')
from models.maskfill import PMAsymDenoiser
from models.sample import get_cfd_traj, sample_loop3, seperate_outputs2
from process.utils_process import (
    add_pep_bb_data,
    extract_pocket,
    get_input_from_file,
    get_peptide_info,
    make_dummy_mol_with_coordinate,
)
from scripts.train_pl import DataModule
from utils.dataset import UseDataset
from utils.misc import *
from utils.reconstruct import *
from utils.sample_noise import get_sample_noiser
from utils.transforms import *


def print_pool_status(pool, logger, is_pep: bool = False) -> None:
    """Print statistics of generation results.

    Args:
        pool: Result pool containing successful and failed generations.
        logger: Logger instance.
        is_pep: Whether generating peptides (affects output format).
    """
    if not is_pep:
        logger.info('[Pool] Succ/Incomp/Bad: %d/%d/%d' % (
            len(pool.succ), len(pool.incomp), len(pool.bad)
        ))
    else:
        logger.info('[Pool] Succ/Nonstd/Incomp/Bad: %d/%d/%d/%d' % (
            len(pool.succ), len(pool.nonstd), len(pool.incomp), len(pool.bad)
        ))


def get_input_data(protein_path,
                   input_ligand=None,
                   is_pep=False,
                   pocket_args={},
                   pocmol_args={}):
    """
    Process input protein and ligand files for generation.
    
    Extracts protein pocket around ligand/reference and prepares molecular data.
    
    Args:
        protein_path: Path to protein PDB file
        input_ligand: Ligand specification (SDF/PDB path or special format like 'pepseq_XXX')
        is_pep: Whether processing peptide
        pocket_args: Pocket extraction parameters (radius, ref_ligand_path, etc.)
        pocmol_args: Additional molecule processing parameters
        
    Returns:
        Tuple of (pocmol_data, pocket_pdb, mol):
            - pocmol_data: Processed pocket-molecule data dict
            - pocket_pdb: Extracted pocket PDB file object
            - mol: RDKit molecule object (or None)
    """
    """
    Process input protein and ligand files for generation.
    
    Extracts protein pocket around ligand/reference and prepares molecular data.
    
    Args:
        protein_path: Path to protein PDB file
        input_ligand: Ligand specification (SDF/PDB path or special format like 'pepseq_XXX')
        is_pep: Whether processing peptide
        pocket_args: Pocket extraction parameters (radius, ref_ligand_path, etc.)
        pocmol_args: Additional molecule processing parameters
        
    Returns:
        Tuple of (pocmol_data, pocket_pdb, mol):
            - pocmol_data: Processed pocket-molecule data dict
            - pocket_pdb: Extracted pocket PDB file object
            - mol: RDKit molecule object (or None)
    """

    # Determine pocket extraction reference
    ref_ligand = pocket_args.get('ref_ligand_path', None)
    pocket_coord = pocket_args.get('pocket_coord', None)
    if ref_ligand is not None:
        pass  # use provided ref_ligand_path
    elif pocket_coord is not None:
        ref_ligand = make_dummy_mol_with_coordinate(pocket_coord)
    else:  # use input_ligand as reference
        print('Neither ref_ligand nor pocket_coord provided for pocket extraction. Using input_ligand as reference.')
        assert input_ligand is not None and (input_ligand.endswith('.sdf') or input_ligand.endswith('.pdb')), \
            'Only SDF/PDB input_ligand can be used for pocket extraction.'
        ref_ligand = input_ligand
    
    # Extract pocket from protein
    pocket_pdb = extract_pocket(protein_path, ref_ligand, 
                            radius=pocket_args.get('radius', 10),
                            criterion=pocket_args.get('criterion', 'center_of_mass'))
    
    # Process input ligand and pocket
    pocmol_data, mol = get_input_from_file(input_ligand, pocket_pdb, return_mol=True, **pocmol_args)
    
    # Add peptide-specific information
    if is_pep:
        if input_ligand.endswith('.pdb'):  # Peptide docking from PDB
            pep_info = get_peptide_info(input_ligand)
            # Verify consistency (sanity check)
            assert torch.isclose(pocmol_data['pos_all_confs'][0], pep_info['peptide_pos'], 1e-2).all(), \
                'Molecule and peptide atoms may not match'
        elif input_ligand.startswith('peplen_'):  # Peptide design
            pep_info = add_pep_bb_data(pocmol_data)
        else:  # pepseq_{xxx} - peptide docking from sequence
            pep_info = {}
        pocmol_data.update(pep_info)
    
    return pocmol_data, pocket_pdb, mol



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_task', type=str, default='configs/sample/examples/dock_pep_know_some.yml', help='task config')
    parser.add_argument('--config_model', type=str, default='configs/sample/pxm.yml', help='model config')
    parser.add_argument('--outdir', type=str, default='./outputs_use')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=0, help='batch size; by default use the value in the config file')
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=-1, help='num_workers for dataloader; by default use the value in the train config file')
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
    ckpt = torch.load(config.model.checkpoint, map_location=args.device, weights_only=False)
    cfg_dir = os.path.dirname(config.model.checkpoint).replace('checkpoints', 'train_config')
    train_config = os.listdir(cfg_dir)
    train_config = make_config(os.path.join(cfg_dir, ''.join(train_config)))

    save_traj_prob = config.sample.save_traj_prob
    batch_size = config.sample.batch_size if args.batch_size == 0 else args.batch_size
    num_mols = config.sample.get('num_mols', 100)
    num_repeats = config.sample.get('num_repeats', 1)

    # # Logging
    log_root = args.outdir
    log_dir = get_new_log_dir(log_root, prefix=config_name)
    logger = get_logger('sample', log_dir)
    # writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info('Load from %s...' % config.model.checkpoint)
    logger.info(args)
    logger.info(config)
    save_config(config, os.path.join(log_dir, os.path.basename(args.config_task)))
    # for script_dir in ['scripts', 'utils', 'models']:
    #     shutil.copytree(script_dir, os.path.join(log_dir, script_dir))
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
    is_ar = config.task.transform.get('name', '')
    noiser = get_sample_noiser(config.noise, in_dims['num_node_types'], in_dims['num_edge_types'],
                               mode='sample',device=args.device, ref_config=train_config.noise)
    if 'variable_mol_size' in getattr(config, 'transforms', []):  # mol design
        transforms = featurizer_list + [
            get_transforms(config.transforms.variable_mol_size), task_trans]
    elif 'variable_sc_size' in getattr(config, 'transforms', []):  # pep design
        transforms = featurizer_list + [
            get_transforms(config.transforms.variable_sc_size), task_trans]
    else:
        transforms = featurizer_list + [task_trans]
    addition_transforms = [get_transforms(tr) for tr in config.data.get('transforms', [])]
    transforms = Compose(transforms + addition_transforms)
    follow_batch = sum([getattr(t, 'follow_batch', []) for t in transforms.transforms], [])
    exclude_keys = sum([getattr(t, 'exclude_keys', []) for t in transforms.transforms], [])
    
    # # Data loader
    logger.info('Loading dataset...')
    data_cfg = config.data
    is_pep = data_cfg.get('is_pep', None)
    if is_pep is None:
        is_pep = data_cfg.input_ligand.endswith('.pdb') or data_cfg.input_ligand.startswith('pep')
    data, pocket_block, in_mol = get_input_data(
        protein_path=data_cfg.protein_path,
        input_ligand=data_cfg.get('input_ligand', None),
        is_pep=is_pep,
        pocket_args=data_cfg.get('pocket_args', {}),
        pocmol_args=data_cfg.get('pocmol_args', {})
    )
    test_set = UseDataset(data, n=num_mols, task=config.task.name, transforms=transforms)

    test_loader = DataLoader(test_set, batch_size, shuffle=args.shuffle,
                            num_workers = train_config.train.num_workers if args.num_workers == -1 else args.num_workers,
                            pin_memory = train_config.train.pin_memory,
                            follow_batch=follow_batch, exclude_keys=exclude_keys)
    # save pocket and mol
    input_pocmol_dir = os.path.join(pure_sdf_dir, '0_inputs')
    os.makedirs(input_pocmol_dir, exist_ok=True)
    with open(os.path.join(input_pocmol_dir, 'pocket_block.pdb'), 'w') as f:
        f.write(pocket_block)
    Chem.MolToMolFile(in_mol, os.path.join(input_pocmol_dir, 'input_mol.sdf'))

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
        **({'nonstd': []} if is_pep else {})
    })
    info_keys = ['data_id', 'db', 'task', 'key']
    i_saved = 0
    # generating molecules
    logger.info('Start sampling... (Total: n_mols=%d)' % (num_mols))
    
    try:
        for i_repeat in range(num_repeats):
            logger.info(f'Generating molecules.')
            for batch in test_loader:
                if i_saved >= num_mols:
                    logger.info('Enough molecules. Stop sampling.')
                    break
                
                # # prepare batch then sample
                batch = batch.to(args.device)
                batch, outputs, trajs = sample_loop3(batch, model, noiser, args.device, is_ar=is_ar)
                
                # # decode outputs to molecules
                data_list = [{key:batch[key][i] for key in info_keys} for i in range(len(batch))]
                generated_list, outputs_list, traj_list_dict = seperate_outputs2(batch, outputs, trajs)
                
                # # post process generated data for the batch
                mol_info_list = []
                for i_mol in tqdm(range(len(generated_list)), desc='Post process generated mols'):
                    # add meta data info
                    mol_info = featurizer.decode_output(**generated_list[i_mol]) 
                    mol_info.update(data_list[i_mol])  # add data info
                    
                    # reconstruct mols
                    try:
                        if not is_pep:
                            with CaptureLogger():
                                rdmol = reconstruct_from_generated_with_edges(mol_info, in_mol=in_mol)
                            smiles = Chem.MolToSmiles(rdmol)
                            if '.' in smiles:
                                tag = 'incomp'
                                pool.incomp.append(mol_info)
                                logger.warning('Incomplete molecule: %s' % smiles)
                            else:
                                tag = ''
                                pool.succ.append(mol_info)
                                logger.info('Success: %s' % smiles)
                        else:
                            with CaptureLogger():
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
                        tag = 'bad'
                        rdmol = create_sdf_string(mol_info)
                        if is_pep:
                            aaseq = ''
                            pdb_struc = PDB.Structure.Structure('bad')
                    
                    mol_info.update({
                        'rdmol': rdmol,
                        'smiles': smiles,
                        'tag': tag,
                        'output': outputs_list[i_mol],
                        **({
                            'pdb_struc': pdb_struc,
                            'aaseq': aaseq,
                        } if is_pep else {})
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

                # # save sdf/pdb mols for the batch
                df_info_list = []
                for data_finished in mol_info_list:
                    # # save generated mol/pdb
                    rdmol = data_finished['rdmol']
                    tag = data_finished['tag']
                    filename_base = str(i_saved) + (f'-{tag}' if tag else '')
                    # save pdb
                    if is_pep:
                        pdb_struc = data_finished['pdb_struc']
                        filename_pdb = filename_base + '.pdb'
                        pdb_io = PDBIO()
                        pdb_io.set_structure(pdb_struc)
                        pdb_io.save(os.path.join(pure_sdf_dir, filename_pdb))
                    # rdmol to sdf
                    filename_sdf = filename_base + ('.sdf' if not is_pep else '_mol.sdf')
                    if tag != 'bad':
                        Chem.MolToMolFile(rdmol, os.path.join(pure_sdf_dir, filename_sdf))
                    else:
                        with open(os.path.join(pure_sdf_dir, filename_sdf), 'w+') as f:
                            f.write(rdmol)
                    # save traj
                    if 'traj' in data_finished:
                        for traj_who in data_finished['traj'].keys():
                            sdf_file = '$$$$\n'.join(data_finished['traj'][traj_who])
                            name_traj = filename_base + f'-{traj_who}.sdf'
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
                        torch.save(output, os.path.join(sdf_dir, filename_base + '.pt'))

                    # log info 
                    info_dict = {
                        key: data_finished[key] for key in info_keys +
                        (['aaseq'] if is_pep else []) + ['smiles', 'tag']
                    }
                    info_dict.update({
                        'filename': filename_sdf if not is_pep else filename_pdb,
                        'i_repeat': i_repeat,
                        'cfd_traj': cfd_traj,
                        'cfd_pos': cfd_pos,
                        'cfd_node': cfd_node,
                        'cfd_edge': cfd_edge,
                    })

                    df_info_list.append(info_dict)
            
                df_info_batch = pd.DataFrame(df_info_list)
                # save df
                if os.path.exists(df_path):
                    df_info = pd.read_csv(df_path)
                    df_info = pd.concat([df_info, df_info_batch], ignore_index=True)
                else:
                    df_info = df_info_batch
                df_info.to_csv(df_path, index=False)
                print_pool_status(pool, logger, is_pep=is_pep)
                
                # clean up
                del batch, outputs, trajs, mol_info_list[0:len(mol_info_list)]
                if args.device != 'cpu':
                    with torch.cuda.device(args.device):
                        torch.cuda.empty_cache()
                gc.collect()


        # make dummy pool  (save disk space)
        dummy_pool = {key: ['']*len(value) for key, value in pool.items()}
        torch.save(dummy_pool, os.path.join(log_dir, 'samples_all.pt'))
    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt. Stop sampling.')


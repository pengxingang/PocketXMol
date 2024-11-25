"""
Depreicated
"""

from copy import deepcopy
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import sys
sys.path.append('.')
import shutil
import argparse
import torch
import torch.utils.tensorboard
import numpy as np
# from torch_geometric.data import Batch
from easydict import EasyDict
from tqdm.auto import tqdm
from rdkit import Chem
from torch_geometric.loader import DataLoader
# from scipy.special import softmax

from models.maskfill import MolDiff, MolDiffNoEdge
from models.bond_predictor import BondPredictor
from models.sample import seperate_outputs
from utils.transforms import *
from utils.misc import *
from utils.reconstruct import *
from utils.chem import *

def print_pool_status(pool, logger):
    logger.info('[Pool] Finished %d | Failed %d' % (
        len(pool.finished), len(pool.failed)
    ))


def data_exists(data, prevs):
    for other in prevs:
        if len(data.logp_history) == len(other.logp_history):
            if (data.ligand_context_element == other.ligand_context_element).all().item() and \
                (data.ligand_context_feature_full == other.ligand_context_feature_full).all().item() and \
                torch.allclose(data.ligand_context_pos, other.ligand_context_pos):
                return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/conf/conf_basic.yml')
    parser.add_argument('--outdir', type=str, default='./outputs/conf')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=0)
    args = parser.parse_args()

    # # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.sample.seed + np.sum([ord(s) for s in args.outdir]))
    # load ckpt and train config
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    train_config = ckpt['config']

    # # Logging
    log_root = args.outdir.replace('outputs', 'outputs_vscode') if sys.argv[0].startswith('/data') else args.outdir
    log_dir = get_new_log_dir(log_root, prefix=config_name)
    logger = get_logger('sample', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    for script_dir in ['scripts', 'utils', 'models', 'notebooks']:
        shutil.copytree(script_dir, os.path.join(log_dir, script_dir))

    # # Transform
    logger.info('Loading data placeholder...')
    if 'max_size' not in train_config.transform:
        featurizer = FeaturizeMol(train_config.chem.atomic_numbers, train_config.chem.mol_bond_types,
                              use_mask_node=train_config.transform.use_mask_node,
                              use_mask_edge=train_config.transform.use_mask_edge,)
        max_size = None
    else:
        max_size = train_config.transform.max_size
        featurizer = FeaturizeMaxMol(train_config.chem.atomic_numbers, train_config.chem.mol_bond_types,
                              use_mask_node=train_config.transform.use_mask_node,
                              use_mask_edge=train_config.transform.use_mask_edge,
                              max_size=max_size)
    add_edge = getattr(config.sample, 'add_edge', None)
    transform = Compose([
        featurizer,
        RdkitConf()
    ])
    
    # # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config = config.dataset,
        transform = transform,
    )
    train_set, val_set = subsets['train'], subsets['val']
    train_iterator = inf_iterator(DataLoader(
        train_set, 
        batch_size = config.sample.batch_size, 
        shuffle = True,
        num_workers = config.sample.num_workers,
        pin_memory = config.sample.pin_memory,
        follow_batch = featurizer.follow_batch,
        exclude_keys = featurizer.exclude_keys,
    ))
    val_loader = DataLoader(val_set, config.sample.batch_size, shuffle=False,
                            follow_batch=featurizer.follow_batch, exclude_keys=featurizer.exclude_keys)

    
    # # Model
    logger.info('Loading diffusion model...')
    if train_config.model.name == 'diffusion':
        model = MolDiff(
                    config=train_config.model,
                    num_node_types=featurizer.num_node_types,
                    num_edge_types=featurizer.num_edge_types
                ).to(args.device)
    elif train_config.model.name == 'diffusion_noedge':
        model = MolDiffNoEdge(
            config=train_config.model,
            num_node_types=featurizer.num_node_types,
            # num_edge_types=featurizer.num_edge_types
            predict_bond=(add_edge=='predictor')
        ).to(args.device)
        train_config.chem.mol_bond_types = []
        num_edge_types_old = featurizer.num_edge_types
        if add_edge in ['edm', 'openbabel']:
            featurizer.num_edge_types = 1  # featurizer will not decode edge
    model.load_state_dict(ckpt['model']) #NOTE
    model.eval()
    
    # Bond predictor
    if 'bond_predictor' in config:
        logger.info('Building bond predictor...')
        ckpt_bond = torch.load(config.bond_predictor, map_location=args.device)
        bond_predictor = BondPredictor(ckpt_bond['config']['model'],
                featurizer.num_node_types,
                featurizer.num_edge_types-1 # note: bond_predictor not use edge mask
        ).to(args.device)
        bond_predictor.load_state_dict(ckpt_bond['model'])
        bond_predictor.eval()
    else:
        bond_predictor = None
    if 'guidance' in config.sample:
        guidance = config.sample.guidance  # tuple: (guidance_type[entropy/uncertainty], guidance_scale)
    else:
        guidance = None

    pool = EasyDict({
        'failed': [],
        'finished': [],
    })
    
    # generating molecules
    def sample_conf(data_loader):
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Sample conformations'):
                n_graphs = batch.num_graphs
                batch_node = batch.node_type_batch
                halfedge_index = batch.halfedge_index
                batch_halfedge = batch.halfedge_type_batch
                
                outputs = model.sample_conf(
                    node_type=batch.node_type.to(args.device),
                    node_pos=batch.node_pos.to(args.device),
                    batch_node=batch_node.to(args.device),
                    halfedge_type=batch.halfedge_type.to(args.device),
                    halfedge_index=halfedge_index.to(args.device),
                    batch_halfedge=batch_halfedge.to(args.device),
                    num_mols=n_graphs,
                    bond_predictor=bond_predictor,
                    guidance=guidance,
                )
                outputs = {key:[v.cpu().numpy() for v in value] for key, value in outputs.items()}
                #TODO check the node/edge type before and after are the same
                # decode outputs to molecules
                try:
                    output_list = seperate_outputs(outputs, n_graphs, 
                                    batch_node.numpy(), halfedge_index.numpy(), batch_halfedge.numpy())
                except:
                    continue
            
                gen_list = []
                for i_mol, output_mol in enumerate(output_list):
                    mol_info = featurizer.decode_output(
                        pred_node=output_mol['pred'][0],
                        pred_pos=output_mol['pred'][1],
                        pred_halfedge=output_mol['pred'][2],
                        halfedge_index=output_mol['halfedge_index'],
                    )  # note: traj is not used
                    try:
                        rdmol = reconstruct_from_generated_with_edges(mol_info, add_edge=add_edge)
                    except MolReconsError:
                        pool.failed.append(mol_info)
                        logger.warning('Reconstruction error encountered.')
                        continue
                    mol_info['rdmol'] = rdmol
                    smiles = Chem.MolToSmiles(rdmol)
                    mol_info['smiles'] = smiles
                    
                    # Check smiles is not changed
                    sm_before = batch.smiles[i_mol]
                    if sm_before != smiles:
                        sm_before = Chem.MolToSmiles(Chem.MolFromSmiles(sm_before), isomericSmiles=False)
                        if sm_before != smiles:
                            print('Smiles changed: %s -> %s' % (sm_before, smiles))
                    if '.' in smiles:
                        logger.warning('Incomplete molecule: %s' % smiles)
                        pool.failed.append(mol_info)
                    else:   # Pass checks
                        logger.info('Success: %s' % smiles)
                        p_save_traj = np.random.rand()  # save traj
                        if p_save_traj <  config.sample.save_traj_prob:
                            traj_info = [featurizer.decode_output(
                                pred_node=output_mol['traj'][0][t],
                                pred_pos=output_mol['traj'][1][t],
                                pred_halfedge=output_mol['traj'][2][t],
                                halfedge_index=output_mol['halfedge_index'],
                            ) for t in range(len(output_mol['traj'][0]))]
                            mol_traj = []
                            for t in range(len(traj_info)):
                                try:
                                    mol_traj.append(reconstruct_from_generated_with_edges(traj_info[t], False, add_edge=add_edge))
                                except MolReconsError:
                                    mol_traj.append(Chem.MolFromSmiles('O'))
                            mol_info['traj'] = mol_traj
                        gen_list.append(mol_info)
                        # pool.finished.append(mol_info)

                # # Save sdf mols
                sdf_dir = log_dir + '_SDF'
                os.makedirs(sdf_dir, exist_ok=True)
                with open(os.path.join(log_dir, 'SMILES.txt'), 'a') as smiles_f:
                    for i, data_finished in enumerate(gen_list):
                        smiles_f.write(data_finished['smiles'] + '\n')
                        rdmol = data_finished['rdmol']
                        Chem.MolToMolFile(rdmol, os.path.join(sdf_dir, '%d.sdf' % (i+len(pool.finished))))

                        if 'traj' in data_finished:
                            with Chem.SDWriter(os.path.join(sdf_dir, 'traj_%d.sdf' % (i+len(pool.finished)))) as w:
                                for m in data_finished['traj']:
                                    try:
                                        w.write(m)
                                    except:
                                        w.write(Chem.MolFromSmiles('O'))
                pool.finished.extend(gen_list)
                print_pool_status(pool, logger)

        torch.save(pool, os.path.join(log_dir, 'samples_all.pt'))
    
    sample_conf(val_loader)

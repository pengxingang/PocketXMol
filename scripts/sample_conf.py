from copy import deepcopy
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
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

from models.maskfill import MolDiff, AsymDiff
from models.bond_predictor import BondPredictor
from models.sample import seperate_outputs, sample_loop
from utils.transforms import *
from utils.misc import *
from utils.reconstruct import *
from utils.chem import *
from utils.train_noise import *

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
    # parser.add_argument('--config', type=str, default='./configs/sample/basic.yml')
    parser.add_argument('--config', type=str, default='./configs/sample/genconf_refine_half_resample.yml')
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
    noiser = GenAndConfController(train_config.noise, featurizer.num_node_types, featurizer.num_edge_types, args.device,
                                  task='conf')  # only predict conf
    # noiser = get_noiser(train_config.noise, featurizer.num_node_types, featurizer.num_edge_types)
    transform = Compose([
        featurizer,
    ])
    # # Data loader
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config = train_config.dataset,
        transform = transform,
    )
    batch_size = args.batch_size if args.batch_size > 0 else config.sample.batch_size
    val_loader = DataLoader(subsets['val'], batch_size, shuffle=False,
                            num_workers = train_config.train.num_workers,
                            pin_memory = train_config.train.pin_memory,
                            follow_batch=featurizer.follow_batch, exclude_keys=featurizer.exclude_keys)
    # iter_loader = iter(val_loader)

    # Model
    logger.info('Loading diffusion model...')
    if train_config.model.name == 'asym_diff':
        model = AsymDiff(
            config=train_config.model,
            num_node_types=featurizer.num_node_types,
            num_edge_types=featurizer.num_edge_types
        ).to(args.device)
    model.load_state_dict(ckpt['model']) #NOTE
    model.eval()

    pool = EasyDict({
        'failed': [],
        'finished': [],
    })
    
    # generating molecules
    for batch in tqdm(val_loader, total=len(val_loader)):
        batch = batch.to(args.device)
        outputs, trajs = sample_loop(batch, model, noiser, args.device)

        # decode outputs to molecules
        try:
            output_list = seperate_outputs(batch, outputs, trajs)
        except:
            continue
        gen_list = []
        for i_mol, output_mol in enumerate(output_list):
            mol_info = featurizer.decode_output(
                pred_node=output_mol['pred']['node'],
                pred_pos=output_mol['pred']['pos'],
                pred_halfedge=output_mol['pred']['halfedge'],
                halfedge_index=output_mol['halfedge_index'],
            )  # note: traj is not used
            try:
                rdmol = reconstruct_from_generated_with_edges(mol_info)
            except MolReconsError:
                pool.failed.append(mol_info)
                logger.warning('Reconstruction error encountered.')
                continue
            mol_info['rdmol'] = rdmol
            smiles = Chem.MolToSmiles(rdmol)
            mol_info['smiles'] = smiles
            if '.' in smiles:
                logger.warning('Incomplete molecule: %s' % smiles)
                pool.failed.append(mol_info)
            else:   # Pass checks
                logger.info('Success: %s' % smiles)
                p_save_traj = np.random.rand()  # save traj
                if p_save_traj <  config.sample.save_traj_prob:
                    traj_info = [featurizer.decode_output(
                        pred_node=output_mol['traj']['node'][t],
                        pred_pos=output_mol['traj']['pos'][t],
                        pred_halfedge=output_mol['traj']['halfedge'][t],
                        halfedge_index=output_mol['halfedge_index'],
                    ) for t in range(len(output_mol['traj']['node']))]
                    mol_traj = []
                    for t in range(len(traj_info)):
                        mol_traj.append(create_sdf_string(traj_info[t]))
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
                    sdf_file = '$$$$\n'.join(data_finished['traj'])
                    with open(os.path.join(sdf_dir, 'traj_%d.sdf' % (i+len(pool.finished))), 'w+') as f:
                        f.write(sdf_file)
        pool.finished.extend(gen_list)
        print_pool_status(pool, logger)

    torch.save(pool, os.path.join(log_dir, 'samples_all.pt'))
    
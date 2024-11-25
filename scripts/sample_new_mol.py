from copy import deepcopy
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import sys
sys.path.append('.')
import shutil
import argparse
import random
import torch
import numpy as np
from torch_geometric.data import Batch
from easydict import EasyDict
from tqdm.auto import tqdm
from rdkit import Chem
# from scipy.special import softmax

from models.maskfill import FragDiff, FragDiffPrev, MolClassifier, IndividualFragDiff
from models.sample import *
from models.sample_grid import *
from utils.transforms import *
from utils.datasets import get_dataset
from utils.misc import *
from utils.data import FOLLOW_BATCH
from utils.reconstruct import *
from utils.chem import *
# import warnings
# warnings.filterwarnings("error")
STATUS_RUNNING = 'running'
STATUS_FINISHED = 'finished'
STATUS_FAILED = 'failed'

def logp_to_rank_prob(logp):
    # return np.ones(len(logp)) / len(logp)

    logp_sum = np.array([np.sum(l) for l in logp])
    prob = np.exp(logp_sum)
    return prob / prob.sum()


@torch.no_grad()
def get_molcls(data_list, model, transform):
    follow_batch = ['real_pos', 'real_bond_feature', ]
    exclude_keys = ['ligand_nbh_list', 'fragment_atom_list', 'ligand_fragmentation', 'protein_atom_name',
                'frag_connect', 'connection_idx', 'protein_center',
                'ligand_context_bond_index'] # for this cls only
    samples = [transform(sample) for sample in data_list]
    batch = Batch.from_data_list(samples, follow_batch=follow_batch, exclude_keys=exclude_keys).to('cuda')
    pred = model(
        batch.real_pos,
        batch.real_atom_feature,
        batch.real_bond_index,
        batch.real_bond_feature,
        batch.real_pos_batch,
        batch.real_bond_feature_batch
    )
    pred = pred.detach().cpu().numpy().flatten()
    pass_list = [data_list[i] for i, p in enumerate(pred) if p > 0]
    return pass_list


@torch.no_grad() 
def get_finished(data_list, model):
    model.eval()
    batch = Batch.from_data_list(data_list, follow_batch=['compose_pos', 'bond_types']).to('cuda')

    ### Predict next fragments
    with torch.no_grad():
        has_focal = model.has_focal(
            compose_pos = batch.compose_pos,
            compose_feature = batch.compose_feature.float(),
            index_ligand = batch.idx_ligand_ctx_in_compose,
            index_protein = batch.idx_protein_in_compose,
            compose_knn_edge_index = batch.compose_knn_edge_index,
            batch_compose = batch.compose_pos_batch,
        )
    
    finished_batch, running_batch = [], []
    for i, has_f in enumerate(has_focal):
        if has_f <= 0:
            finished_batch.append(data_list[i])
        else:
            running_batch.append(data_list[i])

    return finished_batch, running_batch


@torch.no_grad()  # for a protein-ligand
def get_next(data_list, frag_model, threshold):
    frag_model.eval()
    batch = Batch.from_data_list(data_list, follow_batch=['compose_pos', 'bond_types']).to('cuda')

    ### Predict next fragments
    with torch.no_grad():
        generated = frag_model.sample(
            compose_pos = batch.compose_pos,
            compose_feature = batch.compose_feature.float(),
            index_ligand = batch.idx_ligand_ctx_in_compose,
            index_protein = batch.idx_protein_in_compose,
            compose_knn_edge_index = batch.compose_knn_edge_index,
            batch_compose = batch.compose_pos_batch,
            # bonds
            bond_types = batch.bond_types,
            bond_index = batch.bond_index,
            bond_mask = batch.bond_mask,
            batch_bond = batch.bond_types_batch,
        )
    generated = {key:value.cpu() for key, value in generated.items()}
    # generated_list = generated.to('cpu').to_data_list()
    # data = data.to('cpu')

    # if stage == 'finished':  # no focal, predict bonds
    #     data.status = STATUS_FINISHED
    #     return [data]
        # pos_gen, atom_pred, pos_n_atoms = [item.cpu() for item in generated]
    data_next_list = get_next_step(
        data_list,
        generated,
        # transform = transform,
        threshold = threshold
        )
    
    # for i in range(len(data_next_list)):
    #     if is_finished[i]:
    #         data_next_list[i].status = STATUS_FINISHED
    # data_next_list = [transform(data.cpu()) for data in data_next_list]
    data_next_list = [data.cpu() for data in data_next_list]

    return data_next_list


def print_pool_status(pool, logger):
    logger.info('[Pool] Queue %d | Finished %d | Failed %d' % (
        len(pool.queue), len(pool.finished), len(pool.failed)
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
    # parser.add_argument('--config', type=str, default='./configs/sample/diffusion_individual.yml')
    parser.add_argument('--config', type=str, default='./configs/sample/diffusion_bond_long.yml')
    parser.add_argument('--outdir', type=str, default='./outputs/pre_0921/nocls')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_id', type=str, default='3')
    args = parser.parse_args()

    # check exist in out dir
    if os.path.exists(args.outdir):
        all_files = os.listdir(args.outdir)
        file_name = os.path.basename(args.config)[:-4] + '_' + args.data_id + '_'
        # file_name = 'diffusion_bond_long_' + args.data_id + '_'
        files_target = [f for f in all_files if f.startswith(file_name)]
        if len(files_target) > 0:  # # has been sampled before
            for file_name in files_target:
                file_dir = os.path.join(args.outdir, file_name)
                if os.path.exists(os.path.join(file_dir, 'samples_all.pt')):  # # finished
                    print('Already finished!')
                    exit(0)
                else:
                    print('Has been terminated')
                    shutil.rmtree(file_dir)

    # # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.sample.seed)

    # # Get pdb id or data_idx
    data_id = args.data_id
    if len(data_id) == 0:
        if config.data.data_name =='test':
            data_id = config.data.dataset.data_id
        elif config.data.data_name =='new':
            data_id = config.data.new_data.pdb_id
    else:
        data_id = int(data_id)

    # # Logging
    log_dir = get_new_log_dir(args.outdir, prefix='%s_%s' % (config_name, data_id))
    logger = get_logger('sample', log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))    
    sdf_dir = os.path.join(log_dir, 'SDF')  # save dir
    num_finished = 0
    use_molcls = getattr(config.sample, 'use_molcls', False)

    # # Transform
    logger.info('Loading data...')
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    transform = Compose([
        RefineData(),
        protein_featurizer,
        ligand_featurizer,
    ])
    # # Data
    if config.data.data_name == 'test':
        dataset, subsets = get_dataset(
            config = config.data.dataset,
            transform = transform,
        )
        testset = subsets['test']
        base_data = testset[data_id]
    elif config.data.data_name == 'new':
        base_data = get_data_new_mol(config.data.new_data, data_id)
        base_data = transform_data(base_data, transform)

    # # Model (fragment)
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    logger.info('Loading fragment model...')
    max_fragment = ckpt['config'].train.transform.fragmentation.max_fragment
    num_bond_types = 4
    num_elements = len(ligand_featurizer.atomic_numbers)
    num_node_types = num_elements + 1
    if ckpt['config'].model.name == 'diffusion':
        model = FragDiff(
                ckpt['config'].model, 
                protein_atom_feature_dim = protein_featurizer.feature_dim,
                ligand_atom_feature_dim = ligand_featurizer.feature_dim,
                num_node_types = num_node_types,
                num_bond_types = num_bond_types,
                max_fragment = max_fragment,
            ).to(args.device)
    elif ckpt['config'].model.name == 'diff_prev':
        model = FragDiffPrev(
                ckpt['config'].model, 
                protein_atom_feature_dim = protein_featurizer.feature_dim,
                ligand_atom_feature_dim = ligand_featurizer.feature_dim,
                num_node_types = num_node_types,
                num_bond_types = num_bond_types,
                max_fragment = max_fragment,
            ).to(args.device)
    elif ckpt['config'].model.name == 'individual':
        model = IndividualFragDiff(
            ckpt['config'].model, 
            protein_atom_feature_dim = protein_featurizer.feature_dim,
            ligand_atom_feature_dim = ligand_featurizer.feature_dim,
            num_node_types = num_node_types,
            num_bond_types = num_bond_types,
            max_fragment = max_fragment,
        ).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # # Load molcls
    if use_molcls:
        ckpt_molcls = torch.load(config.model.molcls, map_location=args.device)
        config_molcls = ckpt_molcls['config']
        contrastiver = ContextContrastiveBuilder(config_molcls.train.transform.builder,
                                            ligand_featurizer.atomic_numbers, make_fake=False)
        logger.info('Loading molcls model...')
        molcls = MolClassifier(
                config_molcls.model, 
                ligand_atom_feature_dim = contrastiver.atom_feature_dim,
                bond_feature_dim = contrastiver.bond_feature_dim,
            ).to(args.device)
        molcls.load_state_dict(ckpt_molcls['model'])
        molcls.eval()

    pool = EasyDict({
        'queue': [],
        'failed': [],
        'finished': [],
        'duplicate': [],
        'smiles': set(),
    })

    # # Add masked ligand to queue
    mask = LigandMaskAll()  # mask all ligand atoms
    # mask = LigandRandomMask(min_ratio=1., min_num_unmasked=1)  # the initial atoms are not masked
    atom_composer = AtomContextComposer(protein_featurizer.feature_dim,
        ligand_featurizer.feature_dim, model.config.encoder.k)
    masking = mask
    sample_transform = Compose([
        atom_composer,
        BondPlaceholder(max_fragment),
    ]) 
    beam_size = config.sample.beam_size
    for _ in tqdm(range(beam_size), total=beam_size, desc='Init atoms'):
        data = transform_data(deepcopy(base_data), masking)
        # data = transform_data(data, sample_transform)
        # data.status = STATUS_RUNNING
        pool.queue.append(data)

    # # Sampling loop
    logger.info('Start sampling')
    batch_size = config.sample.batch_size
    global_step = 0
    while len(pool.finished) < config.sample.num_samples:
        global_step += 1
        if global_step > 60:
            break
        queue_size = len(pool.queue)

        # # get terminated mols
        running_list = []
        num_batch = int(np.ceil(queue_size / batch_size))
        for i in range(num_batch):
            n_data = batch_size if i < num_batch - 1 else queue_size - i * batch_size
            batch_data = pool.queue[i * batch_size : (i + 1) * batch_size]

            batch_data = [transform_data(d, sample_transform) for d in batch_data]
            if global_step > 1:
                finished_batch, running_batch = get_finished(batch_data, model=model)
            else:
                finished_batch, running_batch = [], batch_data
            running_list.extend(running_batch)
            for data_fini in finished_batch:
                try:
                    # rdmol = reconstruct_from_generated_with_edges(data_next, kekulize=True)
                    try:
                        rdmol = reconstruct_from_generated_with_edges(data_fini)
                    except:
                        continue
                    smiles = Chem.MolToSmiles(rdmol)
                    data_fini.smiles = smiles
                    if '.' in smiles:
                        logger.warning('Incomplete molecule: %s' % smiles)
                        pool.failed.append(data_fini)
                    elif smiles in pool.smiles:
                        logger.warning('Duplicate molecule: %s' % smiles)
                        pool.duplicate.append(data_fini)
                    else:   # Pass checks
                        logger.info('Success: %s' % smiles)
                        data_fini.rdmol = rdmol
                        pool.finished.append(data_fini)
                        pool.smiles.add(smiles)
                        # save sdf
                        os.makedirs(sdf_dir, exist_ok=True)
                        Chem.MolToMolFile(rdmol, os.path.join(sdf_dir, '%d.sdf' % num_finished))
                        num_finished += 1
                except MolReconsError:
                    logger.warning('Reconstruction error encountered.')
                    pool.failed.append(data_fini)
        
        if len(pool.finished) >= config.sample.num_samples:
            break
        
        # # add running
        n_running = len(running_list)
        if n_running < beam_size:
            running_list += [deepcopy(running_list[i]) for i in np.random.choice(
                range(n_running), size=beam_size - n_running, replace=True
            )]

        # # sample candidate new mols for each mol
        queue_tmp = []
        num_batch = int(np.ceil(len(running_list) / batch_size))
        for i in tqdm(range(num_batch)):
            n_data = batch_size if i < num_batch - 1 else queue_size - i * batch_size
            batch_data = running_list[i * batch_size : (i + 1) * batch_size]
            # batch_data = [data for data in batch_data if data.status == STATUS_RUNNING]

            # batch_data = [transform_data(d, sample_transform) for d in batch_data]
            data_next_list = get_next(
                batch_data,
                frag_model = model,
                # transform = sample_transform,
                threshold = config.sample.threshold
            )
            queue_tmp.extend(data_next_list)

        # # prepare for next fragment
        if len(queue_tmp) == 0:
            logger.info('No more candidate molecules in queue.')
            break

        # # check mols
        filter_list = []
        # mol valid check
        for data_run in queue_tmp:
            try:
                mol = reconstruct_from_generated_with_edges(data_run)
                if mol is None:
                    continue
                # if '.' in Chem.MolToSmiles(mol):
                #     continue
                filter_list.append(data_run)
            except:
                continue
        # mol cls check
        if use_molcls:
            pass_list = get_molcls(filter_list, molcls, contrastiver)
            queue_tmp = pass_list
            logger.info(f'Nums before and after the molcls are {len(filter_list)} and {len(pass_list)}.')
        else:
            queue_tmp = filter_list
        if len(queue_tmp) < beam_size:
            index_copy = np.random.choice(range(len(queue_tmp)), beam_size - len(queue_tmp), replace=True)
            queue_tmp += [deepcopy(queue_tmp[i]) for i in index_copy]
        pool.queue = queue_tmp

        print_pool_status(pool, logger)
        # torch.save(pool, os.path.join(log_dir, 'samples_%d.pt' % global_step))


    # # Save sdf mols
    torch.save(pool, os.path.join(log_dir, 'samples_all.pt'))
    with open(os.path.join(log_dir, 'SMILES.txt'), 'a') as smiles_f:
        for i, data_finished in enumerate(pool['finished']):
            smiles_f.write(data_finished.smiles + '\n')
            rdmol = data_finished.rdmol
            # Chem.MolToMolFile(rdmol, os.path.join(sdf_dir, '%d.sdf' % i))

    
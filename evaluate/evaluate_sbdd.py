import sys
import os
import argparse
import pandas as pd
import pickle
from multiprocessing import Pool
from tqdm import tqdm
sys.path.append('.')

from utils.reconstruct import *
from utils.misc import *
from utils.scoring_func import *
from utils.evaluation import *
from utils.dataset import TestTaskDataset
from utils.docking_vina import VinaDockingTask
from process.process_torsional_info import get_mol_from_data
from evaluate.evaluate_mols import evaluate_mol_dict, get_mols_dict_from_gen_path,\
                        get_dir_from_prefix, combine_gt_gen_metrics

def load_mols_from_dataset(data_cfg, task, root_dir='data'):
    test_set = TestTaskDataset(data_cfg.dataset, task,
                               mode='test',
                               split=getattr(data_cfg, 'split', None))

    mols_dict = {d['data_id']:get_mol_from_data(d, root_dir=root_dir) for d in test_set}
    return mols_dict


def gen_name_to_data_id(mols_dict, gen_path):
    df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))
    filename2data_id = df_gen[['filename', 'data_id']]
    filename2data_id['filename'] = filename2data_id['filename'].str.replace('.sdf', '')
    filename2data_id = filename2data_id.set_index('filename').to_dict()['data_id']

    mols_dict_data_id = {}
    for gen_name, mol in mols_dict:
        data_id = filename2data_id[gen_name]
        mols_dict_data_id.setdefault(data_id, []).append(mol)
    return mols_dict_data_id

def get_protein_fn_dict(gen_name_list, gen_path, test_df_path):
    # gen_names to data_id
    df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))
    filename2data_id = df_gen[['filename', 'data_id']].copy()
    filename2data_id['filename'] = filename2data_id['filename'].str.replace('.sdf', '', regex=False)
    filename2data_id = filename2data_id.set_index('filename').to_dict()['data_id']
    
    # data_id to protein_fn
    # df_test = pd.read_csv(test_df_path)
    # data_id_to_protein_fn = df_test.set_index('data_id').to_dict()['protein_fn']

    protein_fn_dict = {
        # gen_name: data_id_to_protein_fn[filename2data_id[gen_name]] for gen_name in gen_name_list
        gen_name: filename2data_id[gen_name]+'_pro.pdb' for gen_name in gen_name_list
    }
    return protein_fn_dict


def dock_single(inputs):
    filename, mol, protein_fn, protein_root, exhaustiveness = inputs
    try:
        if mol is None:
            raise ValueError(f'{filename} is None')
        if mol.HasSubstructMatch(Chem.MolFromSmarts('[#5]')):
            raise ValueError(f'{filename} contains element B')
        vina_task = VinaDockingTask.from_generated_mol(
            mol, protein_fn, protein_root=protein_root
        )
        score_only = vina_task.run(mode='score_only', exhaustiveness=exhaustiveness)[0]
        minimize = vina_task.run(mode='minimize', exhaustiveness=exhaustiveness)[0]
        dock = vina_task.run(mode='dock', exhaustiveness=exhaustiveness)[0]
        # minimize = {'affinity': np.nan, 'pose': ''}
        # dock = {'affinity': np.nan, 'pose': ''}
        vina_scores = {
            'vina_score': score_only['affinity'],
            'vina_min': minimize['affinity'],
            'vina_pose_min': minimize['pose'],
            'vina_dock': dock['affinity'],
            'vina_pose_dock': dock['pose'],
        }
    except Exception as e:
        print(f'Error in {filename}: {e}')
        vina_scores = {
            'vina_score': np.nan,
            'vina_min': np.nan,
            'vina_pose_min': '',
            'vina_dock': np.nan,
            'vina_pose_dock': '',
            
        }
    return {'filename': filename, **vina_scores}

def calc_vina(inputs_list, gen_path, test_df_path,
              protein_root, exhaustiveness):
    # get the dict of generated with keys as data_id

    inputs_list = [(inputs['filename'], inputs['mol'], inputs['protein_fn'], protein_root, exhaustiveness)
                        for inputs in inputs_list]
    with Pool(64) as p:
        results_list = list(tqdm(p.imap_unordered(dock_single, inputs_list), total=len(inputs_list)))
        
    df_vina = pd.DataFrame(results_list)
    df_vina.to_csv(os.path.join(gen_path, 'vina.csv'), index=False)
    return results_list


def add_ref_protein_dict(mols_dict, gen_path, test_df_path, ):
    
    df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))
    filename2data_id = df_gen[['filename', 'data_id']].copy()
    filename2data_id['filename'] = filename2data_id['filename'] # .str.replace('.sdf', '', regex=False)
    filename2data_id = filename2data_id.set_index('filename').to_dict()['data_id']
    
    dock_inputs_list = []
    for gen_name, mol in mols_dict.items():
        data_id = filename2data_id[gen_name]
        protein_fn = data_id+'_pro.pdb'
        dock_inputs_list.append({
            'filename': gen_name,
            'mol': mol,
            'protein_fn': protein_fn,
        })
    return dock_inputs_list


def get_mols_dict_from_baseline(gen_path):
    sdf_path = os.path.join(gen_path, 'SDF')
    mols_dict = {}
    for pocket_path in os.listdir(sdf_path):
        pocket_dir = os.path.join(sdf_path, pocket_path)
        for filenamext in os.listdir(pocket_dir):
            filename = os.path.join(pocket_path, filenamext)
            mol = Chem.MolFromMolFile(os.path.join(sdf_path, filename))
            mols_dict[filename] = mol
    print(f'Loaded {len(mols_dict)} mols from {sdf_path}')
    return mols_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_root', type=str, default='outputs_paper/sbdd_csd')
    parser.add_argument('--exp_name', type=str, default='msel_pocsized')
    parser.add_argument('--test_df_path', type=str, default='data/test/dfs/sbdd_csd.csv')
    parser.add_argument('--protein_root', type=str, default='data/csd/files/proteins')
    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--bl_name', type=str, default='')
    parser.add_argument('--bl_root', type=str, default='./baselines/sbdd/metrics')
    args = parser.parse_args()

    if not args.bl_name:
        mode_list = [
            'per_gen',
            # 'vina',
        ]
        metrics_list = [
            'drug_chem',  # qed, sa, logp, lipinski
            'count_prop',  # n_atoms, n_bonds, n_rings, n_rotatable, weight, n_hacc, n_hdon
            'frags_counts',  # cnt_eleX, cnt_bondX, cnt_ringX(size)
            'local_3d',  # bond length, bond angle, dihedral angle
            'validity',  # validity, connectivity
        ]
    else:
        mode_list = [
            'per_gen',
        ]
        metrics_list = [
            'drug_chem',  # qed, sa, logp, lipinski
            'count_prop',  # n_atoms, n_bonds, n_rings, n_rotatable, weight, n_hacc, n_hdon
            'frags_counts',  # cnt_eleX, cnt_bondX, cnt_ringX(size)
            'local_3d',  # bond length, bond angle, dihedral angle
        ]

    if args.bl_name:
        gen_path = os.path.join(args.bl_root, args.bl_name)
    else:
        gen_path = get_dir_from_prefix(args.result_root, args.exp_name)
    
    if 'per_gen' in mode_list:
        if args.bl_name:
            mols_dict = get_mols_dict_from_baseline(gen_path)
        else:
            mols_dict = get_mols_dict_from_gen_path(gen_path)
        evaluate_mol_dict(mols_dict, metrics_list, gen_path)

    if 'vina' in mode_list:
        assert not args.bl_name, 'bl_name should be None when running vina'
        mols_dict = get_mols_dict_from_gen_path(gen_path)
        inputs_list = add_ref_protein_dict(mols_dict, gen_path, args.test_df_path)
        calc_vina(inputs_list, gen_path, args.test_df_path, args.protein_root, args.exhaustiveness)
        
        
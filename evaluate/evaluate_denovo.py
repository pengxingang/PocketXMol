import sys
import os
import argparse
import pandas as pd
import pickle
from tqdm.auto import tqdm
sys.path.append('.')

from posebusters import PoseBusters

from utils.reconstruct import *
from utils.misc import *
from utils.scoring_func import *
from utils.evaluation import *
from utils.dataset import TestTaskDataset
from process.process_torsional_info import get_mol_from_data
from evaluate.evaluate_mols import evaluate_mol_dict, get_mols_dict_from_gen_path,\
                        get_dir_from_prefix, combine_gt_gen_metrics


def load_mols_from_dataset(data_cfg, task, root_dir='data'):
    test_set = TestTaskDataset(data_cfg.dataset, task,
                               mode='test',
                               split=getattr(data_cfg, 'split', None))
    mols_dict = {d['data_id']+'_mol.sdf': get_mol_from_data(d, root_dir=root_dir) for d in test_set}
    return mols_dict

def get_mols_dict_from_baseline(gen_path):
    sdf_path = os.path.join(gen_path, 'SDF')
    mols_dict = {}
    for filename in os.listdir(sdf_path):
        if filename.endswith('.sdf') and 'traj' not in filename:
            mol = Chem.MolFromMolFile(os.path.join(sdf_path, filename))
            mols_dict[filename] = mol
    print(f'Loaded {len(mols_dict)} mols from {sdf_path}')
    return mols_dict


def calc_buster_one(inputs, buster):
    # filename = list(input_dict.keys())[0]
    # pred_path = list(input_dict.values())[0]
    filename, pred_path = inputs
    
    buster_result = buster.bust(pred_path)
    buster_result = buster_result.astype(float)
    buster_result = buster_result.reset_index()
    buster_result['filename'] = filename
    # buster_result['data_id'] = data_id
    buster_result = buster_result.drop(columns=['file', 'molecule'])

    return buster_result


def get_buster(mols_dict, gen_path):
    buster = PoseBusters(config='mol')  # only check mol
    # remove none mols
    mols_dict = {k: v for k, v in mols_dict.items() if v is not None}
    with Pool(40) as p:
        buster_results = list(tqdm(p.imap_unordered(
            partial(calc_buster_one, buster=buster),
            mols_dict.items()), total=len(mols_dict)))
    
    df_buster = pd.concat(buster_results, ignore_index=True)
    df_buster.to_csv(os.path.join(gen_path, f'buster.csv'), index=False)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_root', type=str, default='outputs_paper/denovo_geom')
    parser.add_argument('--exp_name', type=str, default='msel_default_base')
    parser.add_argument('--bl_name', type=str, default='')
    parser.add_argument('--mol_metric', type=bool, default=False)
    args = parser.parse_args()
    
    metrics_list = [
        'drug_chem',  # qed, sa, logp, lipinski
        'count_prop',  # n_atoms, n_bonds, n_rings, n_rotatable, weight, n_hacc, n_hdon
        'global_3d',  # rmsd_max, rmsd_min, rmsd_median
        'frags_counts',  # cnt_eleX, cnt_bondX, cnt_ringX(size)
        
        'local_3d',  # bond length, bond angle, dihedral angle
        
        'validity',  # validity, connectivity
        'similarity', # sim_with_train, uniqueness, diversity

        'ring_type', # cnt_ring_type_{x}, top_n_freq_ring_type
    ]

    if args.bl_name:  # baseline
        bl_dir = './baselines/denovo/geom'
        if args.bl_name == 'gt':
            gen_path = get_dir_from_prefix(args.result_root, args.exp_name)
            cfg_path = os.path.join(gen_path,
                    [f for f in os.listdir(gen_path) if f.endswith('.yml')][0])
            sample_cfg = load_config(cfg_path)
            mols_dict = load_mols_from_dataset(sample_cfg.data, task=sample_cfg.task)
            gen_path = os.path.join(bl_dir, args.bl_name)
        else:
            gen_path = os.path.join(bl_dir, args.bl_name)
            mols_dict = get_mols_dict_from_baseline(gen_path)
    else: # generated by ours
        gen_path = get_dir_from_prefix(args.result_root, args.exp_name)
        mols_dict = get_mols_dict_from_gen_path(gen_path)

    if args.mol_metric:
        evaluate_mol_dict(mols_dict, metrics_list, gen_path)
    
    # evaluate buster
    get_buster(mols_dict, gen_path)
    
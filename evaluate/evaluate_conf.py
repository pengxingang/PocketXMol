
import pandas as pd
import numpy as np
import os
import copy
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem import rdMolAlign as MA
from multiprocessing import Pool
import argparse
import re
import json
import sys
sys.path.append('.')
from evaluate.evaluate_mols import get_dir_from_prefix

def load_gt(data_path):
    data = pd.read_pickle(data_path)
    data = data.groupby("smi")["mol"].apply(list).reset_index()
    data['mol'] = data['mol'].apply(lambda x: [Chem.RemoveAllHs(mol) for mol in x])
    data['smi'] = data['mol'].apply(lambda x: Chem.MolToSmiles(x[0], isomericSmiles=False))

    gt_dict = data[['smi', 'mol']].set_index('smi').to_dict()['mol']
    return gt_dict


def load_gen(gen_path, test_df_path):
    # transform data_id to smi
    df_test = pd.read_csv(test_df_path)
    data_id_to_smi = df_test[['data_id', 'smiles']].set_index('data_id').to_dict()['smiles']

    # gen df
    df_gen = pd.read_csv(os.path.join(gen_path, "gen_info.csv"))
    sdf_dir = os.path.join(gen_path, "SDF")

    # load generated mol (conf) for each data_id
    gen_dict = {}
    for _, line in tqdm(df_gen.iterrows(), total=len(df_gen), desc='load gen...'):
        data_id = line['data_id']
        filename = line['filename']
        path = os.path.join(sdf_dir, filename)
        if not os.path.exists(path):
            continue
        mol = Chem.MolFromMolFile(path)
        
        smi = data_id_to_smi[data_id]
        if smi in gen_dict:
            gen_dict[smi].append(mol)
        else:
            gen_dict[smi] = [mol]
    
    return gen_dict


def get_rmsd_min(ref_mols, gen_mols, use_ff=False, threshold=0.5):
    rmsd_mat = np.zeros([len(ref_mols), len(gen_mols)], dtype=np.float32)
    for i, gen_mol in enumerate(gen_mols):
        gen_mol_c = copy.deepcopy(gen_mol)
        if use_ff:
            MMFFOptimizeMolecule(gen_mol_c)
        for j, ref_mol in enumerate(ref_mols):
            ref_mol_c = copy.deepcopy(ref_mol)
            rmsd_mat[j, i] = get_best_rmsd(gen_mol_c, ref_mol_c)
    rmsd_mat_min = rmsd_mat.min(-1)
    return (rmsd_mat_min <= threshold).mean(), rmsd_mat_min.mean()


def get_best_rmsd(gen_mol, ref_mol):
    gen_mol = Chem.RemoveAllHs(gen_mol)
    ref_mol = Chem.RemoveAllHs(ref_mol)
    rmsd = MA.GetBestRMS(gen_mol, ref_mol)
    return rmsd


def set_rdmol_positions(rdkit_mol, pos):
    mol = copy.deepcopy(rdkit_mol)
    assert mol.GetConformer(0).GetPositions().shape[0] == pos.shape[0]
    mol = Chem.RemoveAllHs(mol)
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol


def print_results(cov, mat):
    cov_mean, cov_median = np.mean(cov), np.median(cov)
    print("COV_mean: ", cov_mean, ";COV_median: ", cov_median)
    mat_mean, mat_median = np.mean(mat), np.median(mat)
    print("MAT_mean: ", mat_mean, ";MAT_median: ", mat_median)
    return {
        'cov_mean': cov_mean,
        'cov_median': cov_median,
        'mat_mean': mat_mean,
        'mat_median': mat_median,
    }


def single_process(content):
    ref_mols, gen_mols, use_ff, threshold = content
    cov, mat = get_rmsd_min(ref_mols, gen_mols, use_ff, threshold)
    return cov, mat


def process(content):
    try:
        return single_process(content)
    except:
        return None


def fix_inconsistency(gt_dict, gen_dict):
    for smi in tqdm(gen_dict.keys(), desc='fix inconsistency'):
        gen_mols = gen_dict[smi]
        ref_mols = gt_dict[smi]
        
        # if (Chem.MolToSmiles(gen_mols[0], isomericSmiles=False) != 
        #     Chem.MolToSmiles(ref_mols[0], isomericSmiles=False)):
        #     print('fix inconsistency for', smi)
        if True:
            ref_mol = ref_mols[0]
            for i, gen_mol in enumerate(gen_mols):
                gen_mol = Chem.RemoveAllHs(gen_mol)
                assert all([gen_mol.GetAtomWithIdx(idx).GetSymbol() == \
                            ref_mol.GetAtomWithIdx(idx).GetSymbol() 
                        for idx in range(ref_mol.GetNumAtoms())])
                conf = gen_mol.GetConformer(0).GetPositions()
                new_mol = set_rdmol_positions(ref_mol, conf)
                gen_mols[i] = new_mol
        else:
            continue
    return gen_dict


def cal_metrics(gen_path, data_path, 
                test_df_path, use_ff, threshold):

    # load data
    gt_dict = load_gt(data_path)
    gen_dict = load_gen(gen_path, test_df_path)
    
    # gen_dict = fix_inconsistency(gt_dict, gen_dict)
    
    print('num of covered mol graphs of gt', len(set(gt_dict.keys()).intersection(set(gen_dict.keys()))),
          'out of', len(gt_dict.keys()))

    print('num of mol confs in gt:', sum([len(x) for x in gt_dict.values()]))
    print('num of mol confs in gen:', sum([len(x) for x in gen_dict.values()]))
    
    # check num of mol graphs
    # assert len(set(gt_dict.keys()).intersection(set(gen_dict.keys()))) == 188

    # calculate metrics
    cov_list, mat_list = [], []
    index_list = []
    content_list = []
    for smi in gen_dict.keys():
        ref_mols = gt_dict[smi]
        gen_mols = gen_dict[smi]
        if len(gen_mols) != 2 * len(ref_mols):
            # print('Warning: num of generated mols is not twice of num of ref mols')
            raise ValueError('num of generated mols is not twice of num of ref mols')
        content_list.append((ref_mols, gen_mols, use_ff, threshold))

    with Pool(64) as pool:
        for index, inner_output in tqdm(enumerate(pool.imap(process, content_list))):
            if inner_output is None:
                continue
            cov, mat = inner_output
            cov_list.append(cov)
            mat_list.append(mat)
            index_list.append(index)
    metric_dict = print_results(cov_list, mat_list)

    df = pd.DataFrame(zip(cov_list, mat_list), index=index_list, columns=['cov', 'mat'])
    return metric_dict, df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', type=str, default='geom')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--result_root', type=str, default='./outputs')
    args = parser.parse_args()
    
    # generated dir
    gen_path = get_dir_from_prefix(args.result_root, args.exp_name)
    print('Evaluate conf in', gen_path)

    # default params
    data_path = lambda x: f'data/test/conf/rdkit_cluster_data/{x}/test_data_200.pkl'
    if args.db_name == 'geom':
        data_path = data_path('drugs')
        threshold = 1.25
    elif args.db_name == 'qm9':
        data_path = data_path('qm9')
        raise NotImplementedError('check threshold for qm9')
        threshold = 0.5
    else:
        raise ValueError(f'Unknown db_name: {args.db_name}')
    test_df_path = f'data/test/dfs/conf_{args.db_name}.csv'
    use_uff = False

    # calculate metrics
    metric_dict, df = cal_metrics(gen_path, data_path,
                        test_df_path, use_uff, threshold)
                    
    # save
    with open(os.path.join(gen_path, 'metric.txt'), 'w') as f:
        json.dump({k:str(v) for k,v in metric_dict.items()}, f, indent=2)
    df.to_csv(os.path.join(gen_path, 'df_metric.csv'))

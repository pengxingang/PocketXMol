import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import re

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

import sys
sys.path.append('.')
from evaluate.evaluate_mols import get_dir_from_prefix


def get_rmsd(mol_prob, mol_gt):
    """
    Calculate the symm rmsd between two mols. to move them.
    """
    # mol_prob, mol_gt = mol_pair
    mol_prob = deepcopy(mol_prob)
    mol_gt = deepcopy(mol_gt)
    try:
        rmsd = Chem.rdMolAlign.CalcRMS(mol_prob, mol_gt, maxMatches=30000)  # NOT move mol_prob. for docking
    except RuntimeError:
        n_atoms = mol_prob.GetNumAtoms()
        map_list = [[(i, i) for i in range(n_atoms)]]
        rmsd = Chem.rdMolAlign.CalcRMS(mol_prob, mol_gt, map=map_list)
    # rmsd = Chem.rdMolAlign.GetBestRMS(mol_prob, mol_gt)  # move mol_prob to mol_gt
    return rmsd


def set_rdmol_positions(rdkit_mol, pos):
    rdkit_mol = Chem.RemoveAllHs(rdkit_mol)
    assert rdkit_mol.GetConformer(0).GetPositions().shape[0] == pos.shape[0]
    mol = deepcopy(rdkit_mol)
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol


def fix_inconsistency_one_mol(mol_prob, mol_gt):
    assert all([mol_gt.GetAtomWithIdx(idx).GetSymbol() == \
                mol_prob.GetAtomWithIdx(idx).GetSymbol() 
            for idx in range(mol_gt.GetNumAtoms())])
    conf = mol_prob.GetConformer(0).GetPositions()
    new_mol = set_rdmol_positions(mol_gt, conf)
    return new_mol


def evaluate_rmsd_df(df_gen, gen_dir, gt_dir, check_repeats=10):
    """
    Evaluate the rmsd of generated molecules
    """
    data_id_list = df_gen['data_id'].unique()
    print('Find %d generated mols with %d unique data_id' % (len(df_gen), len(data_id_list)))
    if check_repeats > 0:
        assert len(df_gen) / len(data_id_list) == check_repeats, f'Repeat {check_repeats} not match: {len(df_gen)}:{len(data_id_list)}'

    # # load gt mols
    gt_files = {data_id: os.path.join(gt_dir, data_id+'_mol.sdf')
                for data_id in data_id_list}
    gt_mols = {data_id: Chem.MolFromMolFile(gt_files[data_id])
               for data_id in data_id_list}

    # # calc rmsd for each gen mol
    df_gen['rmsd'] = np.nan
    df_gen.reset_index(inplace=True, drop=True)
    for index, line in tqdm(df_gen.iterrows(), total=len(df_gen)):
        # if index % len(data_id_list) == 28:
        #     continue
        data_id = line['data_id']
        filename = line['filename']
        file_path = os.path.join(gen_dir, filename)
        if not os.path.exists(file_path):
            print('Warning: Not found %s' % filename)
            continue
        gen_mol = Chem.MolFromMolFile(file_path)
        if gen_mol is None:
            gen_mol = Chem.MolFromMolFile(os.path.join(gen_dir, filename), sanitize=False)
        if gen_mol is None:
            print(f'Error mol: {filename}')
            continue
        if gen_mol.GetNumAtoms() == 0:
            print('Warning: Empty mol: %s' % filename)
            continue
        # gen_mol = fix_inconsistency_one_mol(gen_mol, gt_mols[data_id])
        rmsd = get_rmsd(gen_mol, gt_mols[data_id])
        df_gen.loc[index, 'rmsd'] = rmsd
        
        # confidence
        # if os.path.exists(os.path.join(gen_dir, filename.replace('.sdf', '.pt'))):
        #     output = torch.load(os.path.join(gen_dir, filename.replace('.sdf', '.pt')))
        #     df_gen.loc[index, 'confidence'] = torch.mean(output['confidence_pos']).item()
    
    return df_gen

    
def get_topk_metrics(df_rmsd, topk):

    oracle = False
    if 'confidence' not in df_rmsd.columns:
        # use real rmsd as -confidence
        df_rmsd['confidence'] = -df_rmsd['rmsd']
        oracle = True

    # get rmsd from confidence topk in each data_id and explode
    df_topk_rmsd = df_rmsd.groupby('data_id').apply(
        lambda x: x.sort_values('confidence', ascending=False).head(topk)
    ).reset_index(drop=True)
    df_topk_rmsd = df_topk_rmsd.groupby('data_id')['rmsd'].apply(
        lambda x: x.sort_values(ascending=True).tolist()).reset_index()
    df_topk_rmsd[[f'rank{i}' for i in range(topk)]] = pd.DataFrame(df_topk_rmsd['rmsd'].tolist())
    df_topk_rmsd = df_topk_rmsd.drop(columns=['rmsd'])
    
    return df_topk_rmsd, oracle


def get_rank_metrics(df_rmsd, ranks):

    df_rmsd = df_rmsd.copy()
    if 'confidence' not in df_rmsd.columns:
        # use real rmsd as -confidence
        df_rmsd['confidence'] = np.random.randn(df_rmsd['rmsd'].shape[0])
        cfd = False
    else:
        cfd = True

    # get the oracle best
    df_metric = df_rmsd.groupby('data_id')[['rmsd']].min()
    df_metric.columns = ['best']

    # get rmsd from confidence topk in each data_id and explode
    df_rank_rmsd = df_rmsd.groupby('data_id').apply(
        lambda x: x.sort_values('confidence', ascending=False)
    ).reset_index(drop=True)
    
    # get confidence-based rank top k
    for rank in ranks:
        df_rank = df_rank_rmsd.groupby('data_id')['rmsd'].apply(
            lambda x: x.values[:rank].min()).reset_index(drop=True)
        df_metric['rank_%d' % rank] = df_rank.values
    df_metric = df_metric.reset_index()
    return df_metric, cfd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='base_pxm')
    parser.add_argument('--result_root', type=str, default='./outputs_test/dock_posebusters')
    parser.add_argument('--check_repeats', type=int, default=0)
    parser.add_argument('--db', type=str, default='')
    parser.add_argument('--sub_dir', type=str, default='')
    parser.add_argument('--use_repeats', type=int, default=100)
    args = parser.parse_args()

    result_root = args.result_root
    exp_name = args.exp_name
    
    # db = exp_name.split('_')[-2]
    if args.db != '':
        db = args.db
    else:
        db = re.findall(r'dock_([a-z]+)_', exp_name)
        if len(db) != 1:
            print('Not found db, use poseboff as default')
            db = 'poseboff'
        else:
            db = db[0]
        assert db in ['pbdock', 'poseb', 'poseboff'], f'Unknown db {db} for docking eval'
    gt_dir = f'data/{db}/files/mols'
    assert os.path.exists(gt_dir), f'gt_dir {gt_dir} does not exist'

    # # generate dir
    gen_path = get_dir_from_prefix(result_root, exp_name)
    
    # # load gen df
    df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))
    
    # # make rmsd df
    if not args.sub_dir:
        rmsd_path = os.path.join(gen_path, 'rmsd.csv')
        sdf_path = os.path.join(gen_path, 'SDF')
    else:
        rmsd_path = os.path.join(gen_path, f'rmsd_{args.sub_dir}.csv')
        sdf_path = os.path.join(gen_path, args.sub_dir)
    # if not os.path.exists(rmsd_path):
    df_gen = evaluate_rmsd_df(df_gen, sdf_path, gt_dir, args.check_repeats)
    df_gen.to_csv(rmsd_path, index=False)
    # else:
    #     df_gen = pd.read_csv(rmsd_path)
    
    
    # rank1 with 
    if os.path.exists(os.path.join(gen_path, 'ranking.csv')):
        df_ranking = pd.read_csv(os.path.join(gen_path, 'ranking.csv'))
        df_rmsd = df_gen[['filename', 'data_id', 'i_repeat', 'rmsd']]
        df_ranking = df_ranking.merge(df_rmsd, on=['filename', 'data_id', 'i_repeat'], how='left')
        df_ranking = df_ranking[df_ranking['i_repeat'] < args.use_repeats]
        df_rank1 = df_ranking.groupby('data_id').apply(lambda x:
            pd.Series({
                'rmsd_self_ranking': x.sort_values('self_ranking', ascending=False)['rmsd'].iloc[0],
                'rmsd_tuned_ranking': x.sort_values('tuned_ranking', ascending=False)['rmsd'].iloc[0]\
                    if 'tuned_ranking' in df_ranking.columns else np.nan,
                'rmsd_oracle_ranking': x['rmsd'].min(),
            }))
        # df_rank1 = df_rank1.merge(df_gen, on=['data_id'], how='left')
        df_rank1.to_csv(os.path.join(gen_path, 'rank1_rmsd.csv'), index=False)
        
        print('Ratio of RMSD < 2A:')
        print((df_rank1< 2).mean(0))

    print('Done')
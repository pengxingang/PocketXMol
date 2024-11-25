import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import re
import subprocess

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from timeout_decorator import timeout

import sys
sys.path.append('.')
from evaluate.evaluate_mols import get_dir_from_prefix
from evaluate.utils_eval import get_dockq


@timeout(20)
def get_rmsd_timeout(mol_prob, mol_gt):
    rmsd = Chem.rdMolAlign.CalcRMS(mol_prob, mol_gt, maxMatches=30000)  # NOT move mol_prob. for docking
    return rmsd

def get_rmsd(mol_prob, mol_gt):
    """
    Calculate the symm rmsd between two mols. to move them.
    """
    # mol_prob, mol_gt = mol_pair
    mol_prob = deepcopy(mol_prob)
    mol_gt = deepcopy(mol_gt)

    # try:
    #     # rmsd = Chem.rdMolAlign.CalcRMS(mol_prob, mol_gt, maxMatches=30000)  # NOT move mol_prob. for docking
    #     rmsd = get_rmsd_timeout(mol_prob, mol_gt)
    # except Exception as e:
    #     if isinstance(e, RuntimeError):
    #         print('matches error. Retry with direct atom mapping')
    #     elif isinstance(e, TimeoutError):
    #         print('timeout. Retry with direct atom mapping')
    #     else:
    #         raise e
    # rmsd use direct atom mapping
    assert mol_prob.GetNumAtoms() == mol_gt.GetNumAtoms(), 'mismatched num of atoms'
    assert all([mol_prob.GetAtomWithIdx(i_atom).GetSymbol() == mol_gt.GetAtomWithIdx(i_atom).GetSymbol()
                for i_atom in range(mol_prob.GetNumAtoms())]), 'mismatched atom element types'
    # try:
    #     rmsd = Chem.rdMolAlign.CalcRMS(mol_prob, mol_gt, maxMatches=30000)
    # except Exception as e:
    atom_map = [[(i, i) for i in range(mol_prob.GetNumAtoms())]]
    rmsd = Chem.rdMolAlign.CalcRMS(mol_prob, mol_gt, map=atom_map)
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


def evaluate_dockq_df(df_gen, gen_dir, gt_dir, rec_dir, check_repeats=10):
    """
    Evaluate the dockq of generated molecules
    """
    data_id_list = df_gen['data_id'].unique()
    print('Find %d generated mols with %d unique data_id' % (len(df_gen), len(data_id_list)))
    if check_repeats > 0:
        assert len(df_gen) / len(data_id_list) == check_repeats, f'Repeat {check_repeats} not match: {len(df_gen)}:{len(data_id_list)}'

    # # load gt mols
    gt_files = {data_id: os.path.join(gt_dir, data_id+'_pep.pdb')
                for data_id in data_id_list}
    rec_files = {data_id: os.path.join(rec_dir, data_id+'_pro.pdb')
                 for data_id in data_id_list}

    # # calc dockq for each gen mol
    df_gen['dockq'] = np.nan
    df_gen['irmsd'] = np.nan
    df_gen['lrmsd'] = np.nan
    df_gen.reset_index(inplace=True, drop=True)
    for index, line in tqdm(df_gen.iterrows(), total=len(df_gen), desc='calc dockq'):

        data_id = line['data_id']
        gen_file = os.path.join(gen_dir, line['filename'])
        if not os.path.exists(gen_file):
            raise ValueError('Not exist: %s' % gen_file)
        
        dockq_dict = get_dockq(gen_file, gt_files[data_id], rec_files[data_id])
        df_gen.loc[index, dockq_dict.keys()] = dockq_dict.values()

    return df_gen


def evaluate_rmsd_df(df_gen, gen_dir, gt_dir, check_repeats=10):
    """
    Evaluate the rmsd of generated molecules
    """
    data_id_list = df_gen['data_id'].unique()
    print('Find %d generated mols with %d unique data_id' % (len(df_gen), len(data_id_list)))
    if check_repeats > 0:
        assert len(df_gen) / len(data_id_list) == check_repeats, f'Repeat {check_repeats} not match: {len(df_gen)}:{len(data_id_list)}'

    # # load gt mols
    gt_files = {data_id: os.path.join(gt_dir, data_id+'_pep.pdb')
                for data_id in data_id_list}
    gt_mols = {data_id: Chem.MolFromPDBFile(gt_files[data_id])
               for data_id in data_id_list}
    
    # # calc rmsd for each gen mol
    df_gen['rmsd'] = np.nan
    df_gen.reset_index(inplace=True, drop=True)
    for index, line in tqdm(df_gen.iterrows(), total=len(df_gen), desc='calc rmsd'):

        data_id = line['data_id']
        filename = line['filename']
        gen_mol = Chem.MolFromPDBFile(os.path.join(gen_dir, filename), sanitize=False)
        if gen_mol is None:
            # raise Exception('fixme: inaccuracte pdb cannot be load by rdkit')
            print('mol is None: %s' % filename)
        else:
            if gen_mol.GetNumAtoms() == 0:
                print('Warning: Empty mol: %s' % filename)
                continue
            # gen_mol = fix_inconsistency_one_mol(gen_mol, gt_mols[data_id])
            rmsd = get_rmsd(gen_mol, gt_mols[data_id])
            df_gen.loc[index, 'rmsd'] = rmsd
        
        # confidence
        if os.path.exists(os.path.join(gen_dir, filename.replace('.pdb', '.pt'))):
            output = torch.load(os.path.join(gen_dir, filename.replace('.pdb', '.pt')))
            df_gen.loc[index, 'confidence'] = torch.mean(output['confidence_pos']).item()
    
    return df_gen


def get_rank_metrics(df_metrics, ranks):

    df_metrics = df_metrics.copy()
    if 'confidence' not in df_metrics.columns:
        # use real rmsd as -confidence
        df_metrics['confidence'] = np.random.randn(df_metrics['rmsd'].shape[0])
        cfd = False
    else:
        cfd = True

    # get the oracle best
    # df_rmsd = df_metrics.groupby('data_id')[['rmsd']].min()
    df_rmsd = df_metrics.groupby('data_id')[['lrmsd']].min()
    df_rmsd.columns = ['best_lrmsd']
    df_dockq = df_metrics.groupby('data_id')[['dockq']].max()
    df_dockq.columns = ['best_dockq']
    df_rank = pd.concat([df_rmsd, df_dockq], axis=1)

    # get rmsd from confidence topk in each data_id and explode
    df_rank_metrics = df_metrics.groupby('data_id').apply(
        lambda x: x.sort_values('confidence', ascending=False)
    ).reset_index(drop=True)
    
    # get confidence-based rank top k
    for rank in ranks:
        # df_rank_rmsd = df_rank_metrics.groupby('data_id')['rmsd'].apply(
        df_rank_rmsd = df_rank_metrics.groupby('data_id')['lrmsd'].apply(
            lambda x: x.iloc[:rank].min()).reset_index(drop=True)
        df_rank['rank_%d_lrmsd' % rank] = df_rank_rmsd.values
        
        df_rand_dockq = df_rank_metrics.groupby('data_id')['dockq'].apply(
            lambda x: x.iloc[:rank].max()).reset_index(drop=True)
        df_rank['rank_%d_dockq' % rank] = df_rand_dockq.values

    df_rank = df_rank.reset_index()
    return df_rank, cfd



def get_rmsd_pymol(move_path='', fix_path=''):  # it works
    if (move_path != '') or (fix_path != ''):
        cmd.delete('all')
        cmd.load(move_path, 'move')
        cmd.load(fix_path, 'fix')
    else:  # cmd should have loaded the two structures
        assert len(cmd.get_chains('move')) > 0, 'move not loaded'
        assert len(cmd.get_chains('fix')) > 0, 'fix not loaded'
    
    sel_move = 'move'
    sel_fix ='fix'
    
    # align (but not move) and get rmsd
    r = cmd.align(f'move and ({sel_move}) and (not element H)',
              f'fix and ({sel_fix}) and (not element H)', cycles=0, transform=0,
              object='alobj')
    sel_move = f'move and ({sel_move}) and alobj'
    sel_fix = f'fix and ({sel_fix}) and alobj'
    
    rmsd_all = cmd.rms_cur(f'{sel_move} and (not element H)',
                f'{sel_fix} and (not element H)',
                matchmaker=-1, cycles=0)
    rmsd_ca = cmd.rms_cur(f'{sel_move} and name CA',
                f'{sel_fix} and name CA',
                matchmaker=-1, cycles=0)
    rmsd_bb = cmd.rms_cur(f'{sel_move} and backbone',
                f'{sel_fix} and backbone',
                matchmaker=-1, cycles=0)
    return {
        'allatoms': rmsd_all,
        'ca': rmsd_ca,
        'bb': rmsd_bb,
    }


def evaluate_rmsd_pdb(df_gen, gen_dir, gt_dir, check_repeats=10):
    """
    Evaluate the rmsd of generated molecules
    """
    data_id_list = df_gen['data_id'].unique()
    print('Find %d generated mols with %d unique data_id' % (len(df_gen), len(data_id_list)))
    if check_repeats > 0:
        assert len(df_gen) / len(data_id_list) == check_repeats, f'Repeat {check_repeats} not match: {len(df_gen)}:{len(data_id_list)}'

    # # load gt paths
    gt_files = {data_id: os.path.join(gt_dir, data_id+'_mol.pdb')
                for data_id in data_id_list}

    # # calc rmsd for each gen mol
    df_gen['rmsd'] = np.nan
    df_gen.reset_index(inplace=True, drop=True)
    for index, line in tqdm(df_gen.iterrows(), total=len(df_gen)):

        data_id = line['data_id']
        filename = line['filename']
        gen_files = os.path.join(gen_dir, filename)
        if not os.path.exists(gen_files):
            raise ValueError('gen_files not exist: %s' % filename)

        # gen_mol = fix_inconsistency_one_mol(gen_mol, gt_mols[data_id])
        rmsd = get_rmsd(gen_files, gt_files[data_id])
        df_gen.loc[index, rmsd.keys()] = rmsd.values()
        
        # confidence
        if os.path.exists(os.path.join(gen_dir, filename.replace('.pdb', '.pt'))):
            output = torch.load(os.path.join(gen_dir, filename.replace('.pdb', '.pt')))
            df_gen.loc[index, 'confidence'] = torch.mean(output['confidence_pos']).item()
    
    return df_gen


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='msel_base_fixendresbb')
    parser.add_argument('--result_root', type=str, default='./outputs_paper/dock_pepbdb')
    parser.add_argument('--gt_dir', type=str, default='data/pepbdb/files/peptides')
    parser.add_argument('--rec_dir', type=str, default='data/pepbdb/files/proteins')
    parser.add_argument('--check_repeats', type=int, default=0)
    args = parser.parse_args()

    result_root = args.result_root
    exp_name = args.exp_name
    
    gt_dir = args.gt_dir
    rec_dir = args.rec_dir
    
    # # generate dir
    gen_path = get_dir_from_prefix(result_root, exp_name)
    print('gen_path:', gen_path)
    pdb_path = os.path.join(gen_path, 'SDF')
    
    # # load gen df
    df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))
    
    # # make rmsd df
    rmsd_path = os.path.join(gen_path, 'rmsd_pdb.csv')
    # if not os.path.exists(rmsd_path):
    if True:
        df_gen = evaluate_rmsd_df(df_gen, pdb_path, gt_dir, args.check_repeats)
        # df_gen = evaluate_rmsd_pdb(df_gen, pdb_path, gt_dir, args.check_repeats)
        df_gen.to_csv(rmsd_path, index=False)
    else:
        df_gen = pd.read_csv(rmsd_path)
        
    # # make dockq df
    dockq_path = os.path.join(gen_path, 'dockq_pdb.csv')
    if not os.path.exists(dockq_path):
    # if True:
        df_gen = evaluate_dockq_df(df_gen, pdb_path, gt_dir, rec_dir, args.check_repeats)
        df_gen.to_csv(dockq_path, index=False)
    else:  # merge rmsd to previous dockq, i.e., recalc rmsd
        df_dockq = pd.read_csv(dockq_path)
        del df_dockq['rmsd']
        df_dockq = df_dockq.merge(df_gen[['filename', 'rmsd']], on='filename', how='left')
        df_dockq.to_csv(dockq_path, index=False)


    print('Done')


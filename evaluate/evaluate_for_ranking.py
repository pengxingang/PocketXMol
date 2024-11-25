import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import re
from multiprocessing import Pool
from functools import partial

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
# from posebusters import check_intermolecular_distance

import sys
sys.path.append('.')
from evaluate.evaluate_mols import get_dir_from_prefix
from utils.buster_tools import check_intermolecular_distance, check_identity
from utils.misc import make_config


def calc_clash(inputs, th=1):
    data_id = inputs['data_id']
    filename = inputs['filename']
    pred_path = inputs['pred_path']
    gt_path = inputs['gt_path']
    protein_path = inputs['protein_path']
    pocket_path = inputs['pocket_path']
    

    # load pocket
    protein = Chem.MolFromPDBFile(protein_path, sanitize=False, proximityBonding=False)
    if pred_path.endswith('.sdf'):
        is_pep = False
        mol = Chem.MolFromMolFile(pred_path, sanitize=False)
    elif pred_path.endswith('.pdb'):
        is_pep = True
        mol = Chem.MolFromPDBFile(pred_path, sanitize=False)

    try:
        clash_results = check_intermolecular_distance(
            mol,
            protein,
            ignore_types={"hydrogens", "organic_cofactors", "inorganic_cofactors", "waters"},
            clash_cutoff=0.75 if not is_pep else 0.65,
        )
        no_clashes = clash_results['results']['no_clashes']
        num_clashes = clash_results['results']['num_pairwise_clashes']
    except Exception as e:
        no_clashes = False
        num_clashes = np.nan
        print(f'Error in check_intermolecular_distance for {data_id} {filename}')
    rel_clashes = num_clashes / mol.GetNumAtoms()
    
    if gt_path.endswith('.sdf'):
        mol_true = Chem.MolFromMolFile(gt_path, sanitize=False)
    elif gt_path.endswith('.pdb'):
        mol_true = Chem.MolFromPDBFile(gt_path, sanitize=False)
    else: # smiles
        mol_true = Chem.MolFromSmiles(gt_path)
    try:
        tet_results = check_identity(
            mol,
            mol_true,
            inchi_options="w",
        )
        stereo = tet_results['results']['stereo']
    except Exception as e:
        stereo = False
        print(f'Error in check_identity for {data_id} {filename}')

    clash_results = {
        'data_id': data_id,
        'filename': filename,
        'no_clashes': no_clashes,
        'num_clashes': num_clashes,
        'rel_clashes': rel_clashes,
        'stereo': stereo,
    }

    return clash_results


def prepare_inputs(df_gen, gen_dir, file_dir, sub_dir='SDF'):
    
    inputs_list = []
    for _, line in (df_gen.iterrows()):
        filename = line['filename']
        data_id = line['data_id']
        
        if not isinstance(file_dir, dict):  # test set
            pred_path = os.path.join(gen_dir, sub_dir, filename)
            if not os.path.exists(pred_path):
                print(f'pred_path {pred_path} not exist. Are you using openmm as sub_dir? Use SDF for this case!')
                pred_path = os.path.join(gen_dir, 'SDF', filename)
            if filename.endswith('.sdf'):
                gt_path = os.path.join(file_dir, 'mols', data_id+'_mol.sdf')
            elif filename.endswith('.pdb'):
                gt_path = os.path.join(file_dir, 'peptides', data_id+'_pep.pdb')
            protein_path = os.path.join(file_dir, 'proteins', data_id+'_pro.pdb')
            pocket_path = os.path.join(file_dir, 'pockets10', data_id+'_pocket.pdb')
        else:
            pred_path = os.path.join(gen_dir, os.path.basename(gen_dir)+'_SDF', filename)
            gt_path = file_dir['mol_path']
            protein_path = file_dir['protein_path']
            pocket_path = ''
        
    
            # continue
        inputs_list.append({
            'data_id': data_id,
            'filename': filename,
            'pred_path': pred_path,
            'gt_path': gt_path,
            'protein_path': protein_path,
            'pocket_path': pocket_path,
        })
    return inputs_list
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='pxm_base')
    parser.add_argument('--result_root', type=str, default='./outputs_test')
    parser.add_argument('--db', type=str, default='')
    parser.add_argument('--sub_dir', type=str, default='')
    parser.add_argument('--gt_mol', type=str, default='config')  # for use
    args = parser.parse_args()

    result_root = args.result_root
    exp_name = args.exp_name
    # # generate dir
    gen_path = get_dir_from_prefix(result_root, exp_name)
    save_path = os.path.join(gen_path, 'aux_ranking.csv')
    if os.path.exists(save_path):
        print(f'Already exists {save_path}, skip')
        exit()
    
    if args.gt_mol == '':  # for test set
        if args.db != '':
            db = args.db
        else:
            db = re.findall(r'dock_([a-z]+)_', exp_name) 
            if len(db) != 1:
                print('Not found db, use poseboff as default')
                db = 'poseboff'
            else:
                db = db[0]
        file_dir = f'data/{db}/files'
        assert os.path.exists(file_dir), f'file_dir {file_dir} does not exist'
    else:
        yml_path = [f for f in os.listdir(gen_path) if f.endswith('.yml')][0]
        sa_config = make_config(os.path.join(gen_path, yml_path))
        file_dir = dict(sa_config.data)

    
    # # load gen df
    df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))
    # df_gen = pd.read_csv(os.path.join(gen_path, 'rank1_rmsd_bel.csv'))
    
    sub_dir = 'SDF' if not args.sub_dir else args.sub_dir
    inputs_list = prepare_inputs(df_gen, gen_path, file_dir, sub_dir=sub_dir)
    
    with Pool(64) as p:
        clash_results = list(tqdm(p.imap_unordered(
            partial(calc_clash),
            inputs_list), total=len(inputs_list)))
    
    df_clash = pd.DataFrame(clash_results)
    df_clash.to_csv(save_path, index=False)
    
    print('Done')
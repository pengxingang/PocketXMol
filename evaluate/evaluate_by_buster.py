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
from posebusters import PoseBusters

import sys
sys.path.append('.')
from evaluate.evaluate_mols import get_dir_from_prefix



def calc_buster_one(inputs, buster):
    data_id = inputs['data_id']
    filename = inputs['filename']
    pred_path = inputs['pred_path']
    gt_path = inputs['gt_path']
    protein_path = inputs['protein_path']
    
    buster_result = buster.bust([pred_path], gt_path, protein_path)
    buster_result = buster_result.reset_index().iloc[0].to_dict()

    buster_result.update({
        'data_id': data_id,
        'filename': filename
    })
    return buster_result


def prepare_inputs(df_gen, gen_dir, file_dir, sub_dir='SDF'):
    
    inputs_list = []
    for _, line in (df_gen.iterrows()):
        filename = line['filename']
        data_id = line['data_id']
        
        pred_path = os.path.join(gen_dir, sub_dir, filename)
        gt_path = os.path.join(file_dir, 'mols', data_id+'_mol.sdf')
        protein_path = os.path.join(file_dir, 'proteins', data_id+'_pro.pdb')
        
        if not os.path.exists(pred_path):
            print(f'pred_path {pred_path} not exist. Are you using openmm as sub_dir? Use SDF for this case!')
            pred_path = os.path.join(gen_dir, 'SDF', filename)
            # continue
        inputs_list.append({
            'data_id': data_id,
            'filename': filename,
            'pred_path': pred_path,
            'gt_path': gt_path,
            'protein_path': protein_path,
        })
    return inputs_list
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='base_pxm')
    parser.add_argument('--result_root', type=str, default='./outputs_test')
    parser.add_argument('--db', type=str, default='')
    parser.add_argument('--sub_dir', type=str, default='')
    args = parser.parse_args()

    result_root = args.result_root
    exp_name = args.exp_name
    
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
    file_dir = f'data/{db}/files'
    assert os.path.exists(file_dir), f'file_dir {file_dir} does not exist'

    # # generate dir
    gen_path = get_dir_from_prefix(result_root, exp_name)
    
    # # load gen df
    # df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))
    df_gen = pd.read_csv(os.path.join(gen_path, 'rank1_rmsd_bel.csv'))
    
    sub_dir = 'SDF' if not args.sub_dir else args.sub_dir
    inputs_list = prepare_inputs(df_gen, gen_path, file_dir, sub_dir=sub_dir)
    
    buster = PoseBusters(config='redock')
    with Pool(40) as p:
        buster_results = list(tqdm(p.imap_unordered(
            partial(calc_buster_one, buster=buster),
            inputs_list), total=len(inputs_list)))
    
    df_buster = pd.DataFrame(buster_results)
    df_buster.to_csv(os.path.join(gen_path, f'buster{args.sub_dir}.csv'), index=False)
    
    print('Done')
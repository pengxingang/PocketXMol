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
    # data_id = inputs['data_id']
    gt_path = inputs['gt_path']
    protein_path = inputs['protein_path']
    pred_paths = inputs['pred_paths']
    filenames = inputs['filenames']
    
    buster_result = buster.bust(pred_paths, gt_path, protein_path)
    buster_result = buster_result.astype(float)
    buster_result = buster_result.reset_index()
    # buster_result['filename'] = buster_result.reset_index()['file'].str.split('/').str[-1]
    buster_result['filename'] = filenames
    # buster_result['data_id'] = data_id
    buster_result = buster_result.drop(columns=['file', 'molecule'])

    return buster_result


def prepare_inputs(df_gen, gen_dir, file_dir, sub_dir='SDF'):
    
    inputs_list = []
    n_load = 0
    for data_id, df_this_pocket in df_gen.groupby('data_id'):
        gt_path = os.path.join(file_dir, 'mols', data_id+'_mol.sdf')
        protein_path = os.path.join(file_dir, 'proteins', data_id+'_pro.pdb')
        this_poc_dict = {
            'data_id': data_id,
            'gt_path': gt_path,
            'protein_path': protein_path,
            'filenames': [],
            'pred_paths': [],
        }
        for _, line in df_this_pocket.iterrows():
            filename = line['filename']
            pred_path = os.path.join(gen_dir, sub_dir, filename)
        
            if not os.path.exists(pred_path):
                print(f'pred_path {pred_path} not exist. Are you using openmm as sub_dir? Use SDF for this case!')
                pred_path = os.path.join(gen_dir, 'SDF', filename)
                # continue
            try:
                mol = Chem.MolFromMolFile(pred_path)
            except:
                # print(f'Error in loading {pred_path}')
                continue
            if mol is None:
                # print(f'mol is none: {pred_path}')
                continue
            this_poc_dict['filenames'].append(filename)
            this_poc_dict['pred_paths'].append(pred_path)
            n_load += 1
        inputs_list.append(this_poc_dict)
    print(f'Loaded {n_load} / {len(df_gen)} mols')
    return inputs_list



def prepare_inputs_from_baseline(df_gen, gen_dir, file_dir):
    
    inputs_list = []
    for index_pocket, df_this_pocket in df_gen.groupby('index_pocket'):
        data_id = df_this_pocket.iloc[0]['data_id']
        gt_path = os.path.join(file_dir, 'mols', data_id+'_mol.sdf')
        protein_path = os.path.join(file_dir, 'proteins', data_id+'_pro.pdb')
        this_poc_dict = {
            'data_id': data_id,
            'gt_path': gt_path,
            'protein_path': protein_path,
            'filenames': [],
            'pred_paths': [],
        }
        for _, line in df_this_pocket.iterrows():
            filename = line['filename']
            pred_path = os.path.join(gen_dir, filename)
        
            # if not os.path.exists(pred_path):
            #     print(f'not exist: {pred_path}')
            #     # pred_path = os.path.join(gen_dir, 'SDF', filename)
            #     continue
            # try:
            #     mol = Chem.MolFromMolFile(pred_path)
            # except:
            #     print(f'Error in loading {pred_path}')
            #     continue
            # if mol is None:
            #     print(f'mol is none: {pred_path}')
            #     continue
            this_poc_dict['pred_paths'].append(pred_path)
            this_poc_dict['filenames'].append(filename)
            
        inputs_list.append(this_poc_dict)
    return inputs_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--result_root', type=str, default='./outputs_test/sbdd_csd')
    parser.add_argument('--bl_name', type=str, default='')
    args = parser.parse_args()

    file_dir = 'data/csd/files'
    
    
    if args.bl_name == '':
        result_root = args.result_root
        exp_name = args.exp_name
        # # generate dir
        gen_path = get_dir_from_prefix(result_root, exp_name)
        # # load gen df
        df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))
        inputs_list = prepare_inputs(df_gen, gen_path, file_dir, sub_dir='SDF')
    else:
        bl_root = 'baselines/sbdd/metrics'
        gen_path = os.path.join(bl_root, args.bl_name)
        df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))
        sdf_path = os.path.join(gen_path, 'SDF')
        inputs_list = prepare_inputs_from_baseline(df_gen, sdf_path, file_dir)
    
    
    buster = PoseBusters(config='dock')  # not redock. this is for sbdd
    with Pool(40) as p:
        buster_results = list(tqdm(p.imap_unordered(
            partial(calc_buster_one, buster=buster),
            inputs_list), total=len(inputs_list)))
    
    df_buster = pd.concat(buster_results, ignore_index=True)
    df_buster.to_csv(os.path.join(gen_path, f'buster.csv'), index=False)
    
    print('Done')
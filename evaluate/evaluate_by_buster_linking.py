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


def calc_buster(sdf_path, buster):

    filenames = [f for f in os.listdir(sdf_path) if f.endswith('.sdf') and
                 '-all.sdf' not in f and '-in.sdf' not in f and
                 '-out.sdf' not in f and '-raw.sdf' not in f]
    paths = [os.path.join(sdf_path, filename) for filename in filenames]
    print(f'Find {len(paths)} in sdf_path')
    
    with Pool(40) as p:
        buster_results = list(tqdm(p.imap_unordered(
            partial(calc_buster_single, buster=buster),
            paths), total=len(paths)))
    
    buster_results = pd.concat(buster_results, ignore_index=True)
    return buster_results

def calc_buster_single(path, buster):
    try:
        buster_result = buster.bust(path)
    except:
        print(f'Error with {path}')
        filename = os.path.basename(path)
        buster_result = pd.DataFrame({'filename': [filename]})
        return buster_result
    buster_result = buster_result.astype(float)
    buster_result = buster_result.reset_index()
    
    filenames = [os.path.basename(path)]
    buster_result['filename'] = filenames
    buster_result = buster_result.drop(columns=['file', 'molecule'])
    return buster_result



def calc_buster_dock(sdf_path, buster):
    assert 'moad' in sdf_path, 'only moad db is supported.'
    df_gen = pd.read_csv(os.path.join(os.path.dirname(sdf_path), 'gen_info.csv'))
    if 'sep_id' not in df_gen:
        df_gen['sep_id'] = df_gen['key'].apply(lambda x: x.split(';')[-1].replace('linking/', ''))

    inputs_list = []
    for sep_id, df_this in df_gen.groupby('sep_id'):
        data_id = df_this['data_id'].values[0]
        filenames = df_this['filename'].values
        paths = [os.path.join(sdf_path, filename) for filename in filenames]
        protein_path = f'./data/moad/files/proteins_fixed/{data_id}_pro_fixed.pdb'
        inputs_list.append({
            'sep_id': sep_id,
            'filenames': filenames,
            'paths': paths,
            'protein_path': protein_path,
        })
    
    with Pool(40) as p:
        buster_results = list(tqdm(p.imap_unordered(
            partial(calc_buster_single_dock, buster=buster),
            inputs_list), total=len(inputs_list)))
    
    buster_results = pd.concat(buster_results, ignore_index=True)
    return buster_results

def calc_buster_single_dock(inputs, buster):
    filenames = inputs['filenames']

    protein_path = inputs['protein_path']
    paths = inputs['paths']
    buster_result = buster.bust(paths, None, protein_path)

    buster_result = buster_result.astype(float)
    buster_result = buster_result.reset_index()

    buster_result['filename'] = filenames
    buster_result = buster_result.drop(columns=['file', 'molecule'])
    return buster_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_root', type=str, default='outputs_test/linking_moad/fixed')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--db', type=str, default='moad')
    parser.add_argument('--bl_name', type=str, default='')
    args = parser.parse_args()

    if args.bl_name == '':
        result_root = args.result_root
        exp_name = args.exp_name
        gen_path = get_dir_from_prefix(result_root, exp_name)
        sdf_path = os.path.join(gen_path, 'SDF')
    else:
        bl_root = 'baselines/linking'
        gen_path = os.path.join(bl_root, args.db, args.bl_name)
        df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))
        sdf_path = os.path.join(gen_path, 'SDF')
    # filenames_list = os.listdir(sdf_path)
    
    if 'moad' not in sdf_path:
        buster = PoseBusters(config='mol')
        df_buster = calc_buster(sdf_path, buster)
    else:
        buster = PoseBusters(config='dock')
        df_buster = calc_buster_dock(sdf_path, buster)
    
    # save
    df_buster.to_csv(os.path.join(gen_path, 'buster.csv'), index=False)
    
    print('Done')

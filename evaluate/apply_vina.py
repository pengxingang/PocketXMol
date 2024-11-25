

import argparse
import os
import io
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import re
from multiprocessing import Pool
from functools import partial 
from rdkit import Chem

import sys
sys.path.append('.')

from utils.docking_vina import VinaDockingTask


def get_dir_from_prefix(result_root, exp_name):
    if '_2023' in exp_name:
        gen_path = os.path.join(result_root, exp_name)
        assert os.path.exists(gen_path), f'path {gen_path} not exist'
        return gen_path

    gen_path_prefix = '^' + exp_name + '_2023[0-9|_]*$'
    gen_path = [x for x in os.listdir(result_root) if re.findall(gen_path_prefix, x)]
    assert len(gen_path) == 1, f'exp {exp_name} is not unique/found in {result_root}: {gen_path}'
    gen_path = os.path.join(result_root, gen_path[0])
    print(f'Get path {gen_path}.')
    return gen_path


def get_vina_robust(inputs, save_dir):
    try:
        get_vina_one_file(inputs, save_dir)
    except Exception as e:
        print('Error:', e)
        return

def get_vina_one_file(inputs, save_dir, exhaustiveness=16):
    protein_path = inputs['protein_path']
    ligand_path = inputs['ligand_path']
    filename = inputs['filename']
    save_path = os.path.join(save_dir, filename + '.pkl')
    # if os.path.exists(save_path):
    #     return

    # # load mol
    mol = Chem.MolFromMolFile(ligand_path)
    if mol is None:
        return
    # contains element B
    if mol.HasSubstructMatch(Chem.MolFromSmarts('[#5]')):
        return
    
    # # calc vina
    vina_task = VinaDockingTask.from_generated_mol(
        mol, os.path.basename(protein_path), os.path.dirname(protein_path), 
    )
    vina_scores = {
        'filename': filename,
        'vina_score': vina_task.run(mode='score_only', exhaustiveness=exhaustiveness)[0]['affinity'],
        'vina_min': vina_task.run(mode='minimize', exhaustiveness=exhaustiveness)[0]['affinity'],
        'vina_dock': vina_task.run(mode='dock', exhaustiveness=exhaustiveness)[0]['affinity'],
    }

    # # save
    with open(save_path, 'wb') as f:
        pickle.dump(vina_scores, f)
        

def prepare_inputs(df_gen, gen_dir, protein_dir):
    data_list = []
    for _, line in df_gen.iterrows():
        data_id = line['data_id']
        filename = line['filename'].replace('.sdf', '')
        
        protein_path = os.path.join(protein_dir, data_id+'_pro_fixed.pdb')
        ligand_path = os.path.join(gen_dir, 'SDF', filename+'.sdf')
        data_list.append({
            'filename': filename,
            'protein_path': protein_path,
            'ligand_path': ligand_path,
        })
    return data_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--result_root', type=str, default='./outputs2')
    parser.add_argument('--db', type=str, default='')
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
    
    
    # # get inputs
    gen_path = get_dir_from_prefix(result_root, exp_name)
    df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))
    protein_dir = f'data/{db}/files/proteins_fixed'
    
    save_dir = os.path.join(gen_path, 'vina')
    os.makedirs(save_dir, exist_ok=True)
    
    df_gen = df_gen.sample(frac=1,)
    inputs_list = prepare_inputs(df_gen, gen_path, protein_dir)
    
    with Pool(8) as f:
        list(tqdm(f.imap_unordered(partial(get_vina_robust, save_dir=save_dir),
                    inputs_list), total=len(inputs_list)))

    print('Done')




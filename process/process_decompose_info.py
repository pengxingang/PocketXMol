import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import argparse

import sys
sys.path.append('.')
from utils.dataset import LMDBDatabase
from utils.fragment import get_delinker_decom, get_difflinker_decom
from utils.graph_decom import decompose_mmpa, decompose_brics
from process.process_torsional_info import get_mol_from_data, get_db_config
from process.unmi.process_mols import get_unmi_raw_db


def get_decompose_info(df, mol_path, save_path, mols_dir, decom_list):
    mol_lmdb = LMDBDatabase(mol_path, readonly=True)
    decom_lmdb = LMDBDatabase(save_path, readonly=False)
    
    i_skip = 0
    if 'unmi' in mols_dir:
        train_txn, val_txn = get_unmi_raw_db()
    else:
        train_txn, val_txn = None, None

    for _, line in tqdm(df.iterrows(), total=len(df)):
        data_id = line['data_id']
        # result = {}

        mol_data = mol_lmdb[data_id]
        if mol_data is None:
            print(f'Skip: {data_id} does not have mol data.')
            continue
        
        # # get mol
        mol = get_mol_from_data(mol_data, mols_dir, train_txn, val_txn)

        # # get decompose info
        result_dict = {}
        if 'brics' in decom_list:
            result_dict['brics'] = decompose_brics(mol)
        if 'mmpa' in decom_list:
            result_dict['mmpa'] = decompose_mmpa(mol)
        
        # # save
        # if len(results_list) == 0:
        #     i_skip += 1
        #     print(f'Skip {i_skip}: {data_id} does not have decompose info.')
        #     continue
        decom_lmdb.add_one(data_id, result_dict)
    decom_lmdb.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', type=str, default='geom')
    args = parser.parse_args()

    decom_list = [
        'brics',
        'mmpa'
    ]

    df_path, mol_path, save_path, mols_dir = get_db_config(args.db_name, save_name='decom')
    df_use = pd.read_csv(df_path)
    
    get_decompose_info(df_use, mol_path, save_path, mols_dir, decom_list)
    print('Done processing decompose info for ', args.db_name)
    
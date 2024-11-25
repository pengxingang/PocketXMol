"""
Process the pocket-molecules files in the database.
Save them to lmdb. This is the basic(first) lmdb of the db.
"""
import os
import sys
import argparse

import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import pickle

sys.path.append('.')
from utils.dataset import LMDBDatabase
from utils.parser import parse_conf_list, PDBProtein
from utils.data import torchify_dict, PocketMolData
from process.utils_process import process_raw

def process(df, mols_dir, lmdb_path, modes):
    db = LMDBDatabase(lmdb_path, readonly=False)

    # data_dict = {}
    bad_data_ids = []
    num_skipped = 0
    for _, line in tqdm(df.iterrows(), total=len(df), desc='Preprocessing data'):
        # mol info
        try:
            data_id = line['data_id']
            mol_path = os.path.join(mols_dir, data_id + '.sdf')
            data = process_raw(data_id, mol_path, modes=modes)
            db.add_one(data_id, data)
        except KeyboardInterrupt:
            break
        except Exception as e:
            bad_data_ids.append(data_id)
            num_skipped += 1
            print('Skipping %d Num: %s' % (num_skipped, data_id))
            print(e)
            continue

    db.close()
    print('Processed %d molecules' % (len(df_use) - num_skipped), 'Skipped %d molecules' % num_skipped)
    return bad_data_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', type=str, default='cremp')
    args = parser.parse_args()
    
    db_name = args.db_name
    if db_name in ['cremp', 'protacdb']:
        # data dir
        mols_dir = f'data_train/{db_name}/mols'
        save_path = f'data_train/{db_name}/lmdb/mols.lmdb'
        # df dir
        df_use = pd.read_csv(f'data_train/{db_name}/dfs/meta_uni.csv')
    elif db_name in ['geom', 'unmi', 'qm9']:
        raise NotImplementedError('see the process_pocmol.py in their corresponding sub directory.')
    else:
        raise NotImplementedError(f'unknown {db_name}')
    
    # process
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    modes = ['mols', 'torsional', 'decompose']
    bad_data_ids = process(df_use, mols_dir, save_path, modes)

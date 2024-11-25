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


def process(df, mols_dir, pro_dir, lmdb_path, kwargs):
    db = LMDBDatabase(lmdb_path, readonly=False)

    # data_dict = {}
    bad_data_ids = []
    num_skipped = 0
    for _, line in tqdm(df.iterrows(), total=len(df), desc='Preprocessing data'):
        # mol info
        try:
            data_id = line['data_id']
            pdbid = line['pdbid']
            mol_path = os.path.join(mols_dir, data_id + '_mol.sdf')
            pro_path = os.path.join(pro_dir, data_id + '_pro.pdb')
            data = process_raw(data_id, mol_path, pro_path, pdbid, **kwargs)
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
    print('Processed %d molecules' % (len(df) - num_skipped), 'Skipped %d molecules' % num_skipped)
    # return bad_data_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', type=str, default='pepbdb')
    args = parser.parse_args()
    
    db_name = args.db_name
    kwargs = {}
    if db_name in ['pepbdb']:
        mols_dir = f'data_train/{db_name}/files/mols'
        pro_dir = f'data_train/{db_name}/files/proteins'
        save_path = f'data_train/{db_name}/lmdb/pocmol10.lmdb'
        df_use = pd.read_csv(f'data_train/{db_name}/dfs/meta_filter.csv')
    else:
        raise NotImplementedError(f'unknown {db_name}')
    
    # process
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    bad_data_ids = process(df_use, mols_dir, pro_dir, save_path, kwargs)

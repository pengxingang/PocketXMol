"""
Process the molecules in the database.
Save them to lmdb
"""
import os
import sys
import lmdb

import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import pickle

sys.path.append('.')
from utils.dataset import LMDBDatabase
from utils.parser import parse_mol_with_confs
from utils.data import Mol3DData, torchify_dict

from process.unmi.preprocess_db import unmi_data_to_rdmol


def get_unmi_raw_db():
    train_db_path = 'data_train/unmi/files/train.lmdb'
    val_db_path = 'data_train/unmi/files/valid.lmdb'
    train_env = lmdb.open(train_db_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=256,)
    train_txn = train_env.begin()
    val_env = lmdb.open(val_db_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=256,)
    val_txn = val_env.begin()
    return train_txn, val_txn

def process(df, lmdb_path):

    # raw db
    train_txn, val_txn = get_unmi_raw_db()

    # save lmdb
    db = LMDBDatabase(lmdb_path, readonly=False)

    # data_dict = {}
    bad_data_ids = []
    num_skipped = 0
    for _, line in tqdm(df.iterrows(), total=len(df), desc='Preprocessing data'):
        # mol info
        data_id = line['data_id']
        smiles = line['smiles']
        
        try:
            # get data
            key = line['orig_key']
            if 'train' in data_id:
                datapoint_pickled = train_txn.get(eval(key))
            else:
                datapoint_pickled = val_txn.get(eval(key))
            data = pickle.loads(datapoint_pickled)

            # get mol
            mol = unmi_data_to_rdmol(data, add_confs=False)
            # load all confs of the mol
            
            # build data
            ligand_dict = parse_mol_with_confs(mol, smiles=smiles, confs=data['coordinates'])
            if ligand_dict['num_confs'] == 0:
                raise ValueError('No conformers found')
            ligand_dict = torchify_dict(ligand_dict)
            data = Mol3DData.from_3dmol_dicts(ligand_dict)

            data.smiles = smiles
            data.data_id = data_id
            
            db.add_one(data_id, data)
        except KeyboardInterrupt:
            break
        except Exception as e:
            bad_data_ids.append(data_id)
            num_skipped += 1
            print(e)
            print('Skipping %d Num: %s, %s' % (num_skipped, data_id, smiles))
            continue

    db.close()
    print('Processed %d molecules' % (len(df_use) - num_skipped), 'Skipped %d molecules' % num_skipped)
    return bad_data_ids


if __name__ == '__main__':
    
    data_dir = './data_train/unmi'
    file_df = 'meta_uni.csv'
    
    # data dir
    # mols_dir = os.path.join(data_dir, 'mols')
    lmdb_dir = os.path.join(data_dir, 'lmdb')
    save_path = os.path.join(lmdb_dir, 'mols.lmdb')
    # df dir
    df_use = pd.read_csv(os.path.join(data_dir, 'dfs', file_df))
    # df_use = df_use[df_use['data_id'].str.contains('valid')]
    
    # process
    os.makedirs(lmdb_dir, exist_ok=True)
    bad_data_ids = process(df_use, save_path)
    
    # log bad
    # df_use['has_processed'] = df_use['data_id'].apply(lambda x: x not in bad_data_ids)
    # df_use.to_csv(os.path.join(data_dir, 'dfs', save_df))
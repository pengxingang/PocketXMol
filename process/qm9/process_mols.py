"""
Process the molecules in the QM9 database.
Save them to lmdb
"""
import os
import sys

import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import pickle

sys.path.append('.')
from utils.dataset import LMDBDatabase
from utils.parser import parse_conf_list
from utils.data import Mol3DData, torchify_dict


def process(df, mols_dir, lmdb_path):
    db = LMDBDatabase(lmdb_path, readonly=False)

    # data_dict = {}
    bad_data_ids = []
    num_skipped = 0
    for _, line in tqdm(df.iterrows(), total=len(df), desc='Preprocessing data'):
        # mol info
        data_id = line['data_id']
        smiles = line['smiles']
        
        try:
            # load all confs of the mol
            suppl = Chem.SDMolSupplier(os.path.join(mols_dir, data_id + '.sdf'))
            confs_list = []
            for i_conf in range(len(suppl)):
                mol = Chem.MolFromMolBlock(suppl.GetItemText(i_conf).replace(
                    "RDKit          3D", "RDKit          2D"
                ))  # removeHs=True is default
                mol = Chem.RemoveAllHs(mol)
                confs_list.append(mol)
            
            # build data
            ligand_dict = parse_conf_list(confs_list, smiles=smiles)
            if ligand_dict['num_confs'] == 0:
                raise ValueError('No conformers found')
            ligand_dict = torchify_dict(ligand_dict)
            data = Mol3DData.from_3dmol_dicts(ligand_dict)

            data.smiles = smiles
            data.data_id = data_id
            
            db.add_one(data_id, data)
        except KeyboardInterrupt:
            break
        except:
            bad_data_ids.append(data_id)
            num_skipped += 1
            print('Skipping %d Num: %s, %s' % (num_skipped, data_id, smiles))
            continue

    db.close()
    print('Processed %d molecules' % (len(df_use) - num_skipped), 'Skipped %d molecules' % num_skipped)
    return bad_data_ids


if __name__ == '__main__':
    
    data_dir = './data_train/qm9'
    file_df = 'meta_uni.csv'
    
    # data dir
    mols_dir = os.path.join(data_dir, 'mols')
    lmdb_dir = os.path.join(data_dir, 'lmdb')
    save_path = os.path.join(lmdb_dir, 'mols.lmdb')
    # df dir
    df_use = pd.read_csv(os.path.join(data_dir, 'dfs', file_df))
    
    # process
    os.makedirs(lmdb_dir, exist_ok=True)
    bad_data_ids = process(df_use, mols_dir, save_path)
    
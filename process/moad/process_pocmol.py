"""
Process the pocket-molecules files in the database.
Save them to lmdb. This is the basic(first) lmdb of the db.
"""
import os
import sys

import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import pickle

sys.path.append('.')
from utils.dataset import LMDBDatabase
from utils.parser import parse_conf_list, PDBProtein
from utils.data import torchify_dict, PocketMolData


def process(df, pockets_dir, mols_dir, lmdb_path):
    db = LMDBDatabase(lmdb_path, readonly=False)

    # data_dict = {}
    bad_data_ids = []
    num_skipped = 0
    for _, line in tqdm(df.iterrows(), total=len(df), desc='Preprocessing data'):
        # mol info
        pdbid = line['pdbid']
        data_id = line['data_id']
        smiles = line['smiles']
        
        try:
            # load mol
            mol = Chem.MolFromMolFile(os.path.join(mols_dir, data_id + '_mol.sdf'))
            # build mol data
            ligand_dict = parse_conf_list([mol], smiles=smiles)
            if ligand_dict['num_confs'] == 0:
                raise ValueError('No conformers found')

            # load pocket
            with open(os.path.join(pockets_dir, data_id + '_pocket.pdb'), 'r') as f:
                pdb_bloack = f.read()
            pocket_dict = PDBProtein(pdb_bloack).to_dict_atom()

            data = PocketMolData.from_pocket_mol_dicts(
                pocket_dict=torchify_dict(pocket_dict),
                mol_dict=torchify_dict(ligand_dict),
            )
            data.pdbid = pdbid
            data.data_id = data_id
            data.smiles = smiles
            
            db.add_one(data_id, data)
        except KeyboardInterrupt:
            break
        except Exception as e:
            bad_data_ids.append(data_id)
            num_skipped += 1
            print('Skipping %d Num: %s, %s' % (num_skipped, data_id, smiles))
            print(e)
            continue

    db.close()
    print('Processed %d molecules' % (len(df_use) - num_skipped), 'Skipped %d molecules' % num_skipped)
    return bad_data_ids


if __name__ == '__main__':
    
    # data dir
    pockets_dir = 'data_train/moad/files/pockets10'
    mols_dir = 'data_train/moad/files/mols'
    save_path = 'data_train/moad/lmdb/pocmol10.lmdb'
    # df dir
    df_use = pd.read_csv('data_train/moad/dfs/meta_uni.csv')
    
    # process
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    bad_data_ids = process(df_use, pockets_dir, mols_dir, save_path)
    

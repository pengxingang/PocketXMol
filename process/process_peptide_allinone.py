
import os
import sys
import argparse

import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import torch
import pickle

sys.path.append('.')
from utils.dataset import LMDBDatabase
from utils.parser import parse_conf_list, PDBProtein, parse_pdb_peptide
from utils.data import torchify_dict, PocketMolData
# from process.utils_process import process_peptide



def process_peptide(ligand_path, data_id=''):
    peptide_dict = parse_pdb_peptide(ligand_path)
    peptide_dict = {'peptide_'+k: v for k, v in peptide_dict.items()}
    peptide_dict = torchify_dict(peptide_dict)
    peptide_dict['data_id'] = data_id

    return peptide_dict



def process(df, ligand_dir, lmdb_path, ref_path):
    db = LMDBDatabase(lmdb_path, readonly=False)
    ref_db = LMDBDatabase(ref_path)

    # data_dict = {}
    num_skipped = 0
    for _, line in tqdm(df.iterrows(), total=len(df), desc='Preprocessing pocket data'):
        # mol info
        try:
            data_id = line['data_id']
            ligand_path = os.path.join(ligand_dir, data_id+'_pep.pdb')
            
            data = process_peptide(ligand_path, data_id)
            # check atom consistency
            pos_ref = ref_db[data_id]['pos_all_confs'][0]
            assert data['peptide_pos'].shape[0] == pos_ref.shape[0], 'Num of atom mismatch'
            assert torch.allclose(data['peptide_pos'], pos_ref), 'Atom mismatch, delta pos: %s' % (data['peptide_pos'] - pos_ref).abs().max()
            db.add_one(data_id, data)
        except KeyboardInterrupt:
            break
        except Exception as e:
            # bad_data_ids.append(data_id)
            num_skipped += 1
            print('Skipping %d Num: %s' % (num_skipped, data_id))
            print(e)
            continue

    db.close()
    print('Processed %d molecules' % (len(df) - num_skipped), 'Skipped %d pockets' % num_skipped)
    # return bad_data_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', type=str, default='pepbdb')
    args = parser.parse_args()
    
    db_name = args.db_name

    ligand_dir = f'data_train/{db_name}/files/peptides'
    save_path = f'data_train/{db_name}/lmdb/peptide.lmdb'
    ref_path = f'data_train/{db_name}/lmdb/pocmol10.lmdb'
    
    if db_name == 'pepbdb':
        df_use = pd.read_csv(f'data_train/{db_name}/dfs/meta_filter.csv')
    else: # apep
        df_use = pd.read_csv(f'data_train/{db_name}/dfs/meta_uni.csv')

    # process
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    process(df_use, ligand_dir, save_path, ref_path)


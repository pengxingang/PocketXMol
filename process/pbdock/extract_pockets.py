from multiprocessing import process
import os
import shutil
import argparse
from rdkit import Chem
import multiprocessing as mp
import pandas as pd
from tqdm.auto import tqdm
from functools import partial
import sys
sys.path.append('.')
from utils.parser import PDBProtein
# from utils.visualize import *


def process_item(data_dict, radius):
    protein_path = data_dict['protein_path']
    mol_path = data_dict['mol_path']
    save_path = data_dict['pocket_path']

    try:
        # load mol
        mol = Chem.MolFromMolFile(mol_path)
        
        # load protein
        with open(protein_path, 'r') as f:
            pdb_block = f.read()
        protein = PDBProtein(pdb_block)

        # find pocket
        selected_pocket = protein.query_residues_ligand(mol, radius)
        pdb_block_pocket = protein.residues_to_pdb_block(selected_pocket)
        
        with open(save_path, 'w') as f:
            f.write(pdb_block_pocket)
            
        # get pocket and receptor info
        pocket_info = protein.get_pocket_info(selected_pocket)
        rec_seqs = protein.get_chain_seqs(pocket_info['cover_chain_ids'].split(';'))
        pocket_info = {'pocket_'+k:v for k, v in pocket_info.items()}
        pocket_info.update({
            'rec_seqs': rec_seqs,
            'data_id': data_dict['data_id']
        })

    except Exception as e:
        print('Exception occured.', protein_path, mol_path, save_path)
        print(e)
        pocket_info = {'data_id': data_dict['data_id']}
        
    return pocket_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', type=str, default='pbdock')
    parser.add_argument('--path_df', type=str, default='./data_train/pbdock/dfs/meta_filter_w_pocket.csv')
    parser.add_argument('--root', type=str, default='./data_train/pbdock/files')
    parser.add_argument('--proteins_dir', type=str, default='proteins')
    parser.add_argument('--mols_dir', type=str, default='mols')
    parser.add_argument('--save_dir', type=str, default='pockets')
    parser.add_argument('--radius', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=32)
    args = parser.parse_args()

    # prepare paths
    df = pd.read_csv(args.path_df)
    proteins_dir = os.path.join(args.root, args.proteins_dir)
    mols_dir = os.path.join(args.root, args.mols_dir)
    save_dir = os.path.join(args.root, args.save_dir + str(args.radius))
    os.makedirs(save_dir, exist_ok=True)

    data_id_list = df['data_id'].values
    print('Found %d protein-ligand pairs.' % len(data_id_list))
    data_list = [{
        'data_id': data_id,
        'protein_path': os.path.join(proteins_dir, f'{data_id}_pro.pdb'),
        'mol_path': os.path.join(mols_dir, f'{data_id}_mol.sdf'),
        'pocket_path': os.path.join(save_dir, f'{data_id}_pocket.pdb')
    } for data_id in data_id_list]

    pool = mp.Pool(args.num_workers)
    result_list = []
    for item_pocket in tqdm(pool.imap_unordered(
            partial(process_item, radius=args.radius), data_list), total=len(data_list)):
        result_list.append(item_pocket)
    pool.close()

    # save pocket_info results
    df_pocket = pd.DataFrame(result_list)
    df_pocket.to_csv(os.path.join(os.path.dirname(args.path_df), 'meta_pocket.csv'), index=False)
    print('Done')
    
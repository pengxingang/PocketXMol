from itertools import product
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import networkx as nx


import sys
sys.path.append('.')
from utils.dataset import LMDBDatabase
from utils.fragment import find_rotatable_bond_mat


def data_id_to_filename(data_id):
    return data_id + '_mol.sdf'

def get_torsional_info(df, mol_path, save_path, mols_dir):
    mol_lmdb = LMDBDatabase(mol_path, readonly=True)
    tor_lmdb = LMDBDatabase(save_path, readonly=False)
    
    for _, line in tqdm(df.iterrows(), total=len(df)):
        data_id = line['data_id']
        result = {}

        mol_data = mol_lmdb[data_id]
        if mol_data is None:
            continue
        bond_index = mol_data['bond_index'].numpy()
        
        # # find rotatable bonds
        mol = Chem.MolFromMolFile(os.path.join(mols_dir, data_id_to_filename(data_id)))
        mol = Chem.RemoveAllHs(mol)
        rot_mat = find_rotatable_bond_mat(mol)
        bond_rotatable = rot_mat[bond_index[0], bond_index[1]]
        rotatable_bond_index = bond_index[:, bond_rotatable==1]
        
        G_base = nx.from_edgelist(bond_index.T)
        tor_twist_pairs = {}
        for rot_bond in rotatable_bond_index.T:
            if rot_bond[0] > rot_bond[1]: # note: bond is symmetric
                continue
            G_break = G_base.copy()
            G_break.remove_edge(*rot_bond)
            connected_components = list(nx.connected_components(G_break))
            if len(connected_components) == 2:
                component_0, component_1 = connected_components
                if rot_bond[0] in component_0:
                    component_0.remove(rot_bond[0])
                    component_1.remove(rot_bond[1])
                    tor_twist_pairs[rot_bond[0], rot_bond[1]] = [
                        component_0, component_1]
                else:
                    component_1.remove(rot_bond[0])
                    component_0.remove(rot_bond[1])
                    tor_twist_pairs[rot_bond[0], rot_bond[1]] = [
                        component_1, component_0]
            else:
                raise ValueError(f'Skip: {data_id} does not have two connected components.')
                continue
        
            
        # # make fixed_dist
        # way 1: initial all fixed (not right if there are multiple components)
        n_atoms = mol.GetNumAtoms()
        fixed_dist = np.ones((n_atoms, n_atoms))
        for tor_edge, twisted_edges in tor_twist_pairs.items():
            for not_fixed_pair in product(twisted_edges[0], twisted_edges[1]):
                fixed_dist[not_fixed_pair[0], not_fixed_pair[1]] = 0
                fixed_dist[not_fixed_pair[1], not_fixed_pair[0]] = 0
                
        # # make nbh info
        path_mat = Chem.GetDistanceMatrix(mol)
        nbh_dict = {}
        for i in range(n_atoms):
            nbh_dict[i] = np.where(path_mat[i] == 1)[0].tolist()

        result = {
            'bond_rotatable': np.array(bond_rotatable, dtype=np.int64),
            'tor_twisted_pairs': tor_twist_pairs,
            'fixed_dist_torsion': fixed_dist,
            
            'tor_bond_mat': rot_mat,
            'path_mat': path_mat,
            'nbh_dict': nbh_dict,
        }
        
        # # add a
        
        tor_lmdb.add_one(data_id, result)
    tor_lmdb.close()


if __name__ == '__main__':
    data_dir = './data/pbdock'
    mol_path = os.path.join(data_dir, 'lmdb', 'pocmol10.lmdb')
    save_path = os.path.join(data_dir, 'lmdb', 'torsion.lmdb')
    mols_dir = os.path.join(data_dir, 'files/mols')

    df_use = pd.read_csv(os.path.join(data_dir, 'dfs', 'meta_filter.csv'))
    
    get_torsional_info(df_use, mol_path, save_path, mols_dir)
    
    
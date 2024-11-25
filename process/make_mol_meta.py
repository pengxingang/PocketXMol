import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from multiprocessing import Pool

import sys
sys.path.append('.')
from utils.scoring_func import get_chem


ELEMENT_LIST = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl',
                'B', 'Br', 'I', 'Se']
BOND_LIST = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, 
             Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]

def make_one_meta(mol_with_id):
    data_id, rdmol = mol_with_id
    result_dict = {
        'data_id': data_id,
        'smiles': '',
        'isomeric_smiles': '',
        'broken': False,
        'n_heavy_atoms': np.nan,
        'pass_element': True,
        'pass_bond': True,
        'qed': np.nan,
        'sa': np.nan,
    }
    try:
        # remove hydrogens
        rdmol_noH = Chem.RemoveAllHs(rdmol)

        # smiles and broken
        sm = Chem.MolToSmiles(rdmol_noH, isomericSmiles=False)
        sm_iso = Chem.MolToSmiles(rdmol_noH, isomericSmiles=True)
        result_dict['smiles'] = sm
        result_dict['isomeric_smiles'] = sm_iso
        if '.' in sm:
            result_dict['broken'] = True
        
        # num of heavy atoms
        n_atoms = rdmol_noH.GetNumAtoms()
        result_dict['n_heavy_atoms'] = n_atoms
        
        # elements 
        for atom in rdmol_noH.GetAtoms():
            if atom.GetSymbol() not in ELEMENT_LIST:
                result_dict['pass_element'] = False
                break
            
        # bonds
        for bond in rdmol_noH.GetBonds():
            if bond.GetBondType() not in BOND_LIST:
                result_dict['pass_bond'] = False
                break
        
        # qed, sa
        qed, sa = get_chem(rdmol_noH)
        result_dict['qed'] = qed
        result_dict['sa'] = sa

        result_dict['error_mol'] = False
    except Exception as e:
        print(e)
        result_dict['error_mol'] = True
    return result_dict


def make_meta(mol_dict, num_workers):
    
    with Pool(num_workers) as pool:
        result_list = list(tqdm(pool.imap(make_one_meta, mol_dict.items()), total=len(mol_dict)))
    return result_list


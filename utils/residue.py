from copy import deepcopy
import pickle
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
import numpy as np
import pandas as pd


# load fgs db
with open('data/ccd/aa_fgs_db.pkl', 'rb') as f:
    aa_fgs_db = pickle.load(f)
    
# load atom_name db
with open('data/ccd/atom_name_db.pkl', 'rb') as f:
    atom_name_db = pickle.load(f)

df_aa_db = pd.read_csv('data/ccd/smiles_aa.csv')


def get_res_name(frag):
    fg = Chem.RDKFingerprint(frag)
    for aa_name, fg_ref in aa_fgs_db.items():
        sim = TanimotoSimilarity(fg, fg_ref)
        if sim == 1:
            ccdid, res_flag = aa_name.split('_')
            line = df_aa_db[df_aa_db['ccdid']==ccdid].iloc[0]
            sm_frag = Chem.MolToSmiles(frag, isomericSmiles=False)
            sm_ref = line['rd_smiles'] if res_flag == 'aa' else line['res_smiles']
            if sm_frag == sm_ref:
                return ccdid, res_flag
    # print("UNK res", Chem.MolToSmiles(frag))
    return 'UNK', 'res'


def set_atom_name(frag, res_name):
    if res_name not in atom_name_db:
        return frag, None
    ref = deepcopy(atom_name_db[res_name])
    ref_mol = ref['mol']
    ref_atom_names = ref['atom_names']

    matches = ref_mol.GetSubstructMatches(frag)
    if len(matches) == 0:
        print('No match found for', res_name)
    atom_match = matches[0]

    atom_names_frag = [ref_atom_names[i] for i in atom_match]
    arg_sort = np.argsort(atom_match).tolist()
    
    new_frag = Chem.RenumberAtoms(frag, arg_sort)
    atom_names_frag = [atom_names_frag[i] for i in arg_sort]
    return new_frag, atom_names_frag
    

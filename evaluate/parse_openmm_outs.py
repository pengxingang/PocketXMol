# depreacted. directly using other env to run apply_openmm.py is ok

import os
import numpy as np
import argparser
import pandas as pd
from tqdm import tqdm

from pymol import cmd
from rdkit import Chem
from rdkit.Geometry import Point3D


def get_mol_from_pdb(inputs):
    lig_path = inputs['input_ligand_path']
    cpx_path = inputs['openmm_cpx_path']
    save_pro_path = inputs['save_protein_path']
    save_lig_path = inputs['save_ligand_path']
    
    # # complex pdb split with pymol
    cmd.delete('all')
    # cmd.read_pdbstr(pdb_block, 'complex')
    cmd.load(cpx_path, 'complex')
    cmd.extract('mol', 'complex and chain 1')
    
    # save protein
    cmd.save(save_pro_path, 'complex')
    # cmd.save(save_mol_path, 'mol')
    
    # save ligand
    rdmol = Chem.MolFromMolFile(lig_path)
    mol_pos = cmd.iterate_state('mol', 'print(x,y,z)')
    n_atoms = rdmol.GetNumAtoms()
    conf = rdmol.GetConformer()
    for i in range(n_atoms):
        x, y, z = mol_pos[i]
        conf.SetAtomPosition(i, Point3D(x,y,z))
    Chem.MolToMolFile(rdmol, save_lig_path)
    

def prepare_inputs(df_gen):
    
    for _, line in df_gen.iterrows():
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_path', type=str, default='')
    args = parser.parse_args()
    
    df_gen = pd.read_csv(os.path.join(args.gen_path, 'gen_info.csv'))
    op
    

"""
ipynb is slow to save mols
"""

from functools import partial
import numpy as np
import argparse
import pandas as pd
import os
from tqdm import tqdm
import lmdb
import pickle
from rdkit import Chem
from rdkit.Geometry import Point3D
from multiprocessing import Pool

# import sys
# sys.path.append('.')
from process.make_mol_meta import make_one_meta


def unmi_data_to_rdmol(data, add_confs=True):
    # get data info
    smi = data['smi']
    coordinates_list = data['coordinates']
    atoms = data['atoms']

    # # make mol
    mol = Chem.MolFromSmiles(smi)
    n_atoms = mol.GetNumAtoms()

    # # check atom consistency
    atom_from_smi = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
    atom_from_data = np.array(atoms)  # usually longer since containing H
    if not all(atom_from_smi == atom_from_data[:n_atoms]):
        print('atom not consistent')

    if add_confs:
        # # add conformers
        for coordinates in coordinates_list:
            conf = Chem.Conformer(n_atoms)
            for i_atom, coor in enumerate(coordinates[:n_atoms]):
                conf.SetAtomPosition(i_atom, Point3D(*[c.item() for c in coor]))
            mol.AddConformer(conf, assignId=True)
        n_confs = mol.GetNumConformers()

    return mol


def fetch_and_meta(key, prefix):
    datapoint_pickled = txn.get(key)
    data = pickle.loads(datapoint_pickled)
    mol = unmi_data_to_rdmol(data)

    data_id = prefix + key.decode('utf-8')
    meta = make_one_meta([data_id, mol])
    meta.update({
        'data_id': data_id,
        'orig_key': key,
        'orig_smiles': data['smi']
    })
    return meta


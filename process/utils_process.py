"""
Process the pocket-molecules files in the database.
Save them to lmdb. This is the basic(first) lmdb of the db.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import torch
from rdkit.Chem import AllChem
import pickle
from Bio.PDB import PDBIO, PDBParser
from io import StringIO
from copy import deepcopy
from Bio.PDB.internal_coords import IC_Chain

from PeptideBuilder import Geometry
import PeptideBuilder

sys.path.append('.')
# from utils.dataset import LMDBDatabase
from utils.parser import parse_conf_list, PDBProtein, parse_pdb_peptide
from utils.data import torchify_dict, PocketMolData, Mol3DData
from process.process_torsional_info import get_torsional_info_mol
from process.process_decompose_info import decompose_brics, decompose_mmpa


def get_pdb_angles(stru, select_chain=None, angle_list=None, max_peptide_bond=None):
    """
    Input: Bio.PDB structure
    Output: dict of angles
    """
    if angle_list is None:
        angle_list = ["omg", "psi", "phi", "chi1", "chi2", "chi3", "chi4", 'chirality']
    

    if isinstance(stru, str):
        parser = PDBParser()
        new_stru = parser.get_structure(None, stru)[0]
    else:
        new_stru = deepcopy(stru)
    # print(new_stru)
    
    if max_peptide_bond is not None:
        IC_Chain.MaxPeptideBond = max_peptide_bond
        new_stru.internal_coord = None
    new_stru.atom_to_internal_coordinates(verbose=True)
    result = {}
    # for chain in new_stru:
    #     for residue in chain:
    for residue in new_stru.get_residues():
        chainid = residue.get_parent().id
        if select_chain is not None and chainid not in select_chain:
            continue
        curr_key = (chainid, residue.id, residue.resname)
        curr_result = {}
        if residue.id[0] != " ":
            print('Not support for pdb containing hetero residues yet.')
            return None
        for key in angle_list:
            tmp_v = residue.internal_coord.pick_angle(key)
            if tmp_v is not None:
                tmp_v = tmp_v.angle
            curr_result[key] = tmp_v
        result[curr_key] = curr_result
    # print(result)
    return result

def get_pdb_chirality(stru):
    if isinstance(stru, str):
        parser = PDBParser()
        new_stru = parser.get_structure(None, stru)[0]
    else:
        new_stru = deepcopy(stru)
    
    ch_list = []
    for residue in new_stru.get_residues():
        ch = calculate_chirality(residue)
        ch_list.append(ch)
    return ch_list

def vector_from_two_atoms(atom1, atom2):
    """Return the vector from atom1 to atom2."""
    return np.array(atom2) - np.array(atom1)

def calculate_chirality(residue):
    """
    Calculate the chirality of the residue.
    """
    if 'CB' not in residue: # Glycine
        return None
    N = residue['N'].get_vector().get_array()
    CA = residue['CA'].get_vector().get_array()
    CB = residue['CB'].get_vector().get_array()
    C = residue['C'].get_vector().get_array()
    
    # Calculate the cross product of the vectors
    cross = np.cross(vector_from_two_atoms(CA, N), vector_from_two_atoms(CA, C))
    # Calculate the dot product of the cross product and the vector from CA to CB
    angle = np.dot(cross, vector_from_two_atoms(CA, CB))
    return angle


def build_peptide(pep, return_pdbblock=True):
    for i, aa in enumerate(pep):
        geo = Geometry.geometry(aa)
        if i == 0:
            structure = PeptideBuilder.initialize_res(geo)
        else:
            PeptideBuilder.add_residue(structure, geo)
    if return_pdbblock:
        out = PDBIO()
        out.set_structure(structure)
        string = StringIO()
        out.save(string)
        pdb = string.getvalue()
        return pdb
    else:
        return structure


def add_pep_bb_data(data):
    num_atoms = data['num_atoms']
    num_res = num_atoms // 4
    peptide_data = {
        'pos': np.zeros([num_atoms, 3]),
        'atom_name': ['N', 'CA', 'C', 'O'] * num_res,
        'res_index': np.repeat(np.arange(num_res), 4),
        # 'atom_to_aa_type': ['X'] * num_atoms,
        'is_backbone': np.ones([num_atoms], dtype=bool),
        'pep_len': num_res,
    }
    peptide_data = {'peptide_'+k: v for k, v in peptide_data.items()}
    peptide_data = torchify_dict(peptide_data)
    return peptide_data


def get_make_mol_from_smiles(smiles, add_3D=True, center=None):
    mol = Chem.MolFromSmiles(smiles)
    if add_3D: # add 3D conformer
        mol = Chem.AddHs(mol)
        confid = AllChem.EmbedMolecule(mol, maxAttempts=5000)
        if confid != 0:
            AllChem.EmbedMolecule(mol, useRandomCoords=True)
        AllChem.UFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
        conf = mol.GetConformer(0).GetPositions()
        if center is not None: # move to center
            conf = conf - np.mean(conf, axis=0) + center
        conf_new = mol.GetConformer()
        for i in range(conf.shape[0]):
            conf_new.SetAtomPosition(i, conf[i])
    return mol


def make_dummy_mol_with_coordinate(pos):
    return get_make_mol_from_smiles('C', add_3D=True, center=[pos])


def get_peptide_info(pep):
    used_keys = ['peptide_pos', 'peptide_atom_name', 'peptide_res_index',
                 'peptide_is_backbone', 'peptide_pep_len']
    if isinstance(pep, str):  # is pep path
        peptide_dict = parse_pdb_peptide(pep)
        peptide_dict = {'peptide_'+k: v for k, v in peptide_dict.items()}
        peptide_dict = torchify_dict(peptide_dict)
    else: # is pocmol data
        peptide_dict = add_pep_bb_data(pep)
    peptide_dict = {k: peptide_dict[k] for k in used_keys}
    return peptide_dict


def get_input_from_file(mol, pdb, data_id='', pdbid='', return_mol=False):
    pocmol_data, mol = get_pocmol_data(mol, pdb, data_id, pdbid, return_mol=True)
    
    # if 'torsional' in modes:
    bond_index = pocmol_data['bond_index']
    torsional_info = get_torsional_info_mol(mol, bond_index, data_id)
    pocmol_data.update(torsional_info)

    # if 'decompose' in modes:
    decom_info = {
        'brics': decompose_brics(mol),
        'mmpa': decompose_mmpa(mol),
    }
    pocmol_data.update(decom_info)
    
    if not return_mol:
        return pocmol_data
    else:
        return pocmol_data, mol


def get_pocmol_data(mol, pdb, data_id='', pdbid='', return_mol=False):
    # load mol
    mol_list = None
    if isinstance(mol, str):
        if mol.endswith('.sdf'):
            sd = Chem.SDMolSupplier(mol)
            mol_list = [m for m in sd]
            mol = mol_list[0]
            # mol = Chem.MolFromMolFile(mol)
        elif mol.endswith('.pdb'):
            mol = Chem.MolFromPDBFile(mol)
        # elif mol.endswith('peptide'): # pep design. make dummy pep with len
        elif mol.startswith('peplen_'): # pep design. make dummy pep with len
            n_res = int(mol.split('_')[1])
            mol = 'NCC(=O)' * n_res
            mol = get_make_mol_from_smiles(mol)
        # elif mol.startswith('peptide'):  # peptide sequence
        elif mol.startswith('pepseq_'):  # peptide sequence
            seq = mol.split('_')[1]
            pep_pdb = build_peptide(seq)
            mol = Chem.MolFromPDBBlock(pep_pdb)
        else: # smiles
            mol = get_make_mol_from_smiles(mol)
    else:
        mol = get_make_mol_from_smiles('C')
    mol = Chem.RemoveAllHs(mol)
    if mol_list is None:
        mol_list = [mol]
    else:
        mol_list = [Chem.RemoveAllHs(m) for m in mol_list]
        
    ligand_dict = parse_conf_list(mol_list)
    if ligand_dict['num_confs'] == 0:
        raise ValueError('No conformers found')
    smiles = Chem.MolToSmiles(mol)
    # load pdb
    if isinstance(pdb, str):
        pdb = PDBProtein(pdb)
    pocket_dict = pdb.to_dict_atom()
    # make data
    data = PocketMolData.from_pocket_mol_dicts(
        pocket_dict=torchify_dict(pocket_dict),
        mol_dict=torchify_dict(ligand_dict),
    )
    data.pdbid = pdbid
    data.data_id = data_id
    data.smiles = smiles
    if not return_mol:
        return data
    else:
        return data, mol

def extract_pocket(protein_path, mol_path, radius=10, save_path=None, criterion='center_of_mass'):
    if protein_path is None:
        return ''
    if isinstance(mol_path, Chem.Mol):
        mol = mol_path
    elif mol_path.endswith('.sdf'):
        mol = Chem.MolFromMolFile(mol_path)
    elif mol_path.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(mol_path)
    else:
        mol = None
    if mol is None:
        raise ValueError('Invalid mol file for extracting pocket:', mol_path)
    pdb = PDBProtein(protein_path)
    selected_pocket = pdb.query_residues_ligand(mol, radius=radius, criterion=criterion)
    if len(selected_pocket) == 0:
        raise ValueError('Empty pocket within the radius. Please check your pocket_args configuration.')
    pocket_block = pdb.residues_to_pdb_block(selected_pocket)
    # pdb = PDBProtein(pocket_block)
    # save pocket
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(pocket_block)
    return pocket_block


def process_raw(data_id='', mol_path=None, protein_path=None, pdbid='',
                modes=None, return_pocket=False, save_pocket=True, **kwargs):
    
    # load data
    if mol_path is None:  # used in denovo gen. only need pocket_center to define pocket
        mol = make_dummy_mol_with_coordinate(kwargs['pocket_center'])
    elif mol_path.endswith('.sdf'):
        mol = Chem.MolFromMolFile(mol_path)
    else:
        mol = Chem.MolFromSmiles(mol_path)
        # add 3D conformer
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)  # bug for large mol. see func: get_make_mol_from_smiles
        AllChem.UFFOptimizeMolecule(mol)
        # move to pocket center
        pocket_center = np.array(kwargs['pocket_center']).reshape([1, 3])
        mol = Chem.RemoveHs(mol)
        conf = mol.GetConformer(0).GetPositions()
        conf = conf - np.mean(conf, axis=0) + pocket_center
        conf_new = mol.GetConformer()
        for i in range(conf.shape[0]):
            conf_new.SetAtomPosition(i, conf[i])
        
    mol = Chem.RemoveAllHs(mol)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False)

    if modes is None:
        modes = ['extract_pocket', 'pocmol', 'torsional', 'decompose']
    
    if 'extract_pocket' in modes:
        pdb = PDBProtein(protein_path)
        if 'pocket_center' in kwargs:
            ref_mol = make_dummy_mol_with_coordinate(kwargs['pocket_center'])
        else:
            ref_mol = mol
        radius = int(kwargs.get('radius', 10))
        selected_pocket = pdb.query_residues_ligand(ref_mol,
                radius=radius,
                criterion=kwargs.get('criterion', 'center_of_mass'))
        if len(selected_pocket) == 0:
            raise ValueError('Empty pocket within the radius')
        pocket_block = pdb.residues_to_pdb_block(selected_pocket)
        pdb = PDBProtein(pocket_block)
        if save_pocket:
            # save pocket
            pocket_dir = os.path.join(os.path.dirname(os.path.dirname(protein_path)), f'pockets{radius}')
            os.makedirs(pocket_dir, exist_ok=True)
            pocket_path = os.path.join(pocket_dir, os.path.basename(protein_path).replace('_pro.pdb', '_pocket.pdb'))
            with open(pocket_path, 'w') as f:
                f.write(pocket_block)
    
    data = {}
    if 'pocmol' in modes:
        ligand_dict = parse_conf_list([mol])
        if ligand_dict['num_confs'] == 0:
            raise ValueError('No conformers found')
        pocket_dict = pdb.to_dict_atom()

        data = PocketMolData.from_pocket_mol_dicts(
            pocket_dict=torchify_dict(pocket_dict),
            mol_dict=torchify_dict(ligand_dict),
        )
        data.pdbid = pdbid
        data.data_id = data_id
        data.smiles = smiles
        
    if 'mols' in modes:
        # load mol with multiple conformers
        suppl = Chem.SDMolSupplier(mol_path)
        confs_list = []
        for i_conf in range(len(suppl)):
            conf = Chem.MolFromMolBlock(suppl.GetItemText(i_conf).replace(
                "RDKit          3D", "RDKit          2D"
            ))  # removeHs=True is default
            conf = Chem.RemoveAllHs(conf)
            confs_list.append(conf)
            
        ligand_dict = parse_conf_list(confs_list)  #NOTE: multiple conformation
        if ligand_dict['num_confs'] == 0:
            raise ValueError('No conformers found')
        ligand_dict = torchify_dict(ligand_dict)
        data = Mol3DData.from_3dmol_dicts(ligand_dict)
        data.smiles = smiles
        data.data_id = data_id
    
    if 'torsional' in modes:
        bond_index = data['bond_index']
        torsional_info = get_torsional_info_mol(mol, bond_index, data_id)
        data.update(torsional_info)

    if 'decompose' in modes:
        decom_info = {
            'brics': decompose_brics(mol),
            'mmpa': decompose_mmpa(mol),
        }
        data.update(decom_info)
    
    if return_pocket:
        return data, pocket_block
    return data


def process_raw_pep(protein_path, input_ligand_path,
                    input_pep_path=None,  # for pep info
                    ref_ligand_path=None,  # for pocket extraction
                   pocket_args={}, pocmol_args={}, return_pocket=False):
    # get pocket
    if ref_ligand_path is None:
        ref_ligand_path = input_ligand_path
    pocket_pdb = extract_pocket(protein_path, ref_ligand_path, **pocket_args)
    
    # get input ligand
    data_id = pocmol_args.get('data_id', '')
    pocmol_data, mol = get_pocmol_data(input_ligand_path, pocket_pdb, **pocmol_args,
                                  return_mol=True)
    # torsional
    bond_index = pocmol_data['bond_index']
    torsional_info = get_torsional_info_mol(mol, bond_index, data_id)
    pocmol_data.update(torsional_info)
    # decompose
    decom_info = {
        'brics': decompose_brics(mol),
        'mmpa': decompose_mmpa(mol),
    }
    pocmol_data.update(decom_info)
    
    # peptide info
    if input_pep_path is None:
        input_pep_path = pocmol_data
    pep_info = get_peptide_info(input_pep_path)
    pocmol_data.update(pep_info)
    assert torch.isclose(pocmol_data['pos_all_confs'][0], pep_info['peptide_pos'], 1e-2).all(), 'mol and pep atoms may not match'

    if return_pocket:
        return pocmol_data, pocket_pdb
    return pocmol_data
from itertools import product
from copy import deepcopy
import re

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMMPA
from rdkit.Chem.BRICS import FindBRICSBonds
from rdkit.Chem.Lipinski import RotatableBondSmarts

from utils.difflinker_decom import fragment_by_brics



def get_delinker_decom(mol):
    min_linker_length = 3
    min_frag_size = 5
    linker_smaller_than_frag = True
    min_path_frags = 2 # 2 atoms
    
    # Note: the input mol itself cannot be broken
    # get broken group smiles
    pair_list = decompose_mmpa(mol)
    
    result_list = []
    for pair in pair_list:
        result = align_mol_to_linker_frag(mol, *pair)
        if result is not None:
            pass_delinker = True
            while True:
                # # make filtering
                linker_sizes = [len(linker) for linker in result['linkers']]
                if any([size < min_linker_length for size in linker_sizes]):
                    pass_delinker = False
                    break
                frag_sizes = [len(frag) for frag in result['frags']]
                if any([size < min_frag_size for size in frag_sizes]):
                    pass_delinker = False
                    break
                if linker_smaller_than_frag and max(linker_sizes) >= max(frag_sizes):
                    pass_delinker = False
                    break
                anchors_linkers = result['anchors_linkers']
                if len(set(sum(anchors_linkers, []))) != 2:
                    pass_delinker = False  # if not 2 anchors, then min_path_frags is less than 2 atom
                    # break 
                break
            result['pass_delinker'] = pass_delinker
            result_list.append(result)
    return result_list



def get_difflinker_decom(mol):
    # parameters defined by difflinker
    min_cuts = 3
    max_cuts = 5
    min_size = 3
    num_frags_brics = [4, 5, 6, 7]

    result_list = []
    # # mmpa decomposition
    # pair_list = []
    # for n_cuts in range(min_cuts, max_cuts+1):
    #     pair_list += decompose_mmpa(mol, min_cuts=n_cuts, max_cuts=n_cuts)
    
    # for pair in pair_list:
    #     result = align_mol_to_linker_frag(mol, *pair)
    #     if result is not None:
    #         # # make filtering
    #         pass_difflinker = True
    #         while True:
    #             linker_sizes = [len(linker) for linker in result['linkers']]
    #             if any([size < min_size for size in linker_sizes]):
    #                 pass_difflinker = False
    #                 break
    #             frag_sizes = [len(frag) for frag in result['frags']]
    #             if any([size < min_size for size in frag_sizes]):
    #                 pass_difflinker = False
    #                 break
    #             if mol.GetNumAtoms() > 40:
    #                 pass_difflinker = False
    #                 break
    #             if mol.GetRingInfo().NumRings() < 3:
    #                 pass_difflinker = False
    #             break
    #         result['pass_difflinker'] = pass_difflinker
    #         result_list.append(result)
            

    # brics decomposition
    for num_frags in num_frags_brics:
        result_list += decompose_brics(mol, num_frags, min_size)

    return result_list




    


def decompose_brics(mol, num_frags, min_frag_size):

    result_list = fragment_by_brics(mol, min_frag_size=min_frag_size, num_frags=num_frags)
    
    return result_list


def decompose_mmpa(mol, min_cuts=2, max_cuts=2, pattern="[#6+0;!$(*=,#[!#6])]!@!=!#[*]"):
    """
    Use MMPA to break mol into linkers and fragments
    return: List of (linker_smiles, frag_smiles) pair
    """
    
    # break mol
    decom_smi_list = rdMMPA.FragmentMol(mol,
            minCuts=min_cuts,
            maxCuts=max_cuts,
            maxCutBonds=100,
            pattern=pattern,
            resultsAsMols=False
    )
    return decom_smi_list



def align_mol_to_linker_frag(mol, smi_linkers, smi_frags):
    try:
        smi_linkers = smi_linkers.split('.')
        smi_frags = smi_frags.split('.')
        smi_list = smi_linkers + smi_frags
        matches = seperate_submol_by_smi(deepcopy(mol), smi_list)
        if matches is None:
            return None
        
        # check
        assert set(sum(matches, [])) == set(range(mol.GetNumAtoms())), "Not all atoms matched"
        assert sum(map(len, matches)) == mol.GetNumAtoms(), "Not all atoms matched, length not equal"
        
        # seperate
        matches_linkers = matches[:len(smi_linkers):]
        matches_frags = matches[len(smi_linkers):]

        anchors_linkers = [get_anchor(mol, match) for match in matches_linkers]
        anchors_frags = [get_anchor(mol, match) for match in matches_frags]
        
        # check
        assert set(sum(matches_linkers+matches_frags, [])) == set(range(mol.GetNumAtoms())), "Not all atoms matched"
        assert sum(map(len, matches_linkers+matches_frags)) == mol.GetNumAtoms(), "Not all atoms matched, length not equal"
        
        return {
            'linkers': matches_linkers,
            'anchors_linkers': anchors_linkers,
            'frags': matches_frags,
            'anchors_frags': anchors_frags,
        }

    except Exception as e:
        print(e)
        return None


def get_anchor(mol, piece_match):
    anchor_list = []
    for idx_atom in piece_match:
        atom = mol.GetAtomWithIdx(idx_atom)
        idx_neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        if any((idx_neighbor not in piece_match) for idx_neighbor in idx_neighbors):
            anchor_list.append(idx_atom)
    return anchor_list
    
    
def seperate_submol_by_smi(mol, smi_list, submol_index=None):
    """
    Seperate the submol based on list of smiles
    Input:
        mol: rdkit mol
        smi_list: list of smiles of each piece of the submol. 
                note: the union of smi_list must be equal to the submol
                and there is no overlap between the smi_list
        submol_index: atom indices of submol in mol to be seperated. None means all atoms
    Output:
        the index list of the each piece of the submol in the mol
    """
    # if submol_index is None:
    #     submol_index = list(np.arange(mol.GetNumAtoms()))
    
    # Include dummy atoms in query
    # du = Chem.MolFromSmiles('*')
    qp = Chem.AdjustQueryParameters()
    qp.makeDummiesQueries = True
    
    # # Get idx match of each fragment
    matches_list = []
    # dummies_list = []
    for i, smi_piece in enumerate(smi_list):
        assert '.' not in smi_piece, "smi should not contain '.' for seperate_submol_by_smi"
        piece = Chem.MolFromSmiles(smi_piece)
        # remove dummy atom
        # piece = Chem.DeleteSubstructs(piece, Chem.MolFromSmiles('*'))
        # add dummy atom
        qpiece = Chem.AdjustQueryProperties(piece, qp)
        piece_matches = list(mol.GetSubstructMatches(qpiece, uniquify=False, ))
        # piece_matches = list(mol.GetSubstructMatches(piece, uniquify=True))
        
        # remove dummy atom
        piece_matches_nodummy = []
        for match in piece_matches:
            match = [match[i_atom] for i_atom in range(len(match)) if\
                piece.GetAtomWithIdx(i_atom).GetAtomicNum() != 0]
            piece_matches_nodummy.append(match)
        piece_matches = piece_matches_nodummy
        
        # get unique matches
        piece_matches = np.unique([np.sort(m) for m in piece_matches], axis=0).tolist()
        
        if submol_index is not None:
            piece_matches_real = []
            for match in piece_matches:
                # must be within the frag_match index except dummy atoms
                if all([(idx in submol_index) for idx in match]):
                    piece_matches_real.append(match)
            piece_matches = piece_matches_real
            size = len(submol_index)
        else:
            size = mol.GetNumAtoms()
        
        matches_list.append(piece_matches)

        
    # # Get match of each frag
    # num_piece = len(smi_list)
    # index_list = [np.arange(len(matches)) for matches in matches_list]
    for matches_this in product(*matches_list):
    # for index in product(*index_list):
    #     matches_this = [matches_list[i][index[i]] for i in range(len(index))]
        matches_this = [list(matches) for matches in matches_this]
        if len(set(sum(matches_this, []))) == size:
            break
    if not len(set(sum(matches_this, []))) == size:
        print("No perfect match found")
        return None
    # dummies_this = [dummies_list[i][index[i]] for i in range(len(index))]

    # matches_this = [np.sort(matches).tolist() for matches in matches_this]
    # dummies_this = [np.sort(dummies).tolist() for dummies in dummies_this]
    
    return matches_this


def find_rotatable_bond_mat(mol):
    """Find groups of contiguous rotatable bonds and return as a matrix
    from https://github.com/rdkit/rdkit/discussions/3776"""
    rot_atom_pairs = mol.GetSubstructMatches(RotatableBondSmarts)
    
    rot_mat = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=int)
    for i, j in rot_atom_pairs:
        rot_mat[i, j] = 1
        rot_mat[j, i] = 1
    return rot_mat



if __name__ == '__main__':
    mol = Chem.MolFromSmiles('CN(C(=O)c1ccc2c(c1)OCO2)C1(C(=O)NC2CCCC2)CCCCC1')
    # smi_list = ['C[*:1]', 'c1cc2c(cc1[*:2])OCO2']
    # submol_index = [0, 4,  5,  6,  7,  8,  9, 10, 11, 12]
    # matches = seperate_submol_by_smi(mol, smi_list, submol_index, include_dummy=True)
    # print(matches)
    
    decom_method = 'delinker_decom'
    if decom_method == 'delinker_decom':
        outputs = get_delinker_decom(mol)
        print(outputs)
    elif decom_method == 'difflinker_decom':
        outputs = get_difflinker_decom(mol)
        print(outputs)
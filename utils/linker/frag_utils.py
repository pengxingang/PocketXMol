"""
https://raw.githubusercontent.com/oxpig/DeLinker/master/analysis/frag_utils.py
"""
import os
import random
import re
import math
import csv

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from rdkit import Chem
from rdkit.Chem import rdMMPA
from rdkit import DataStructs
from rdkit.Chem import Descriptors, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdMolAlign
from rdkit.Chem import MolStandardize

import matplotlib.pyplot as plt

from itertools import chain, product

# from joblib import Parallel, delayed

from utils import sascorer
from utils.linker import calc_SC_RDKit

### Dataset info #####
def dataset_info(dataset): #qm9, zinc, cep
    if dataset=='qm9':
        return { 'atom_types': ["H", "C", "N", "O", "F"],
                 'maximum_valence': {0: 1, 1: 4, 2: 3, 3: 2, 4: 1},
                 'number_to_atom': {0: "H", 1: "C", 2: "N", 3: "O", 4: "F"},
                 'bucket_sizes': np.array(list(range(4, 28, 2)) + [29])
               }
    elif dataset=='zinc':
        return { 'atom_types': ['Br1(0)', 'C4(0)', 'Cl1(0)', 'F1(0)', 'H1(0)', 'I1(0)',
                'N2(-1)', 'N3(0)', 'N4(1)', 'O1(-1)', 'O2(0)', 'S2(0)','S4(0)', 'S6(0)'],
                 'maximum_valence': {0: 1, 1: 4, 2: 1, 3: 1, 4: 1, 5:1, 6:2, 7:3, 8:4, 9:1, 10:2, 11:2, 12:4, 13:6, 14:3},
                 'number_to_atom': {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5:'I', 6:'N', 7:'N', 8:'N', 9:'O', 10:'O', 11:'S', 12:'S', 13:'S'},
                 'bucket_sizes': np.array([28,31,33,35,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,53,55,58,84])
               }

    elif dataset=="cep":
        return { 'atom_types': ["C", "S", "N", "O", "Se", "Si"],
                 'maximum_valence': {0: 4, 1: 2, 2: 3, 3: 2, 4: 2, 5: 4},
                 'number_to_atom': {0: "C", 1: "S", 2: "N", 3: "O", 4: "Se", 5: "Si"},
                 'bucket_sizes': np.array([25,28,29,30, 32, 33,34,35,36,37,38,39,43,46])
               }
    else:
        print("the datasets in use are qm9|zinc|cep")
        exit(1)

##### Read files #####

def read_file(filename):
    '''Reads .smi file '''
    '''Returns array containing smiles strings of molecules'''
    smiles, names = [], []
    with open(filename, 'r') as f:
        for line in f:
            if line:
                smiles.append(line.strip().split(' ')[0])
    return smiles

def read_paired_file(filename):
    '''Reads .smi file '''
    '''Returns array containing smiles strings of molecules'''
    smiles, names = [], []
    with open(filename, 'r') as f:
        for line in f:
            if line:
                smiles.append(line.strip().split(' ')[0:2])
    return smiles

def read_triples_file(filename):
    '''Reads .smi file '''
    '''Returns array containing smiles strings of molecules'''
    smiles, names = [], []
    with open(filename, 'r') as f:
        for line in f:
            if line:
                smiles.append(line.strip().split(' ')[0:3])
    return smiles

##### Check data #####
def check_smi_atom_types(smi, dataset='zinc', verbose=False):
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        valence = atom.GetTotalValence()
        charge = atom.GetFormalCharge()
        atom_str = "%s%i(%i)" % (symbol, valence, charge)

        if atom_str not in dataset_info(dataset)['atom_types']:
            if "*" in atom_str:
                continue
            else:
                if verbose:
                    print('unrecognized atom type %s' % atom_str)
                return False
    return True

##### Fragment mols #####

def remove_dummys(smi_string):
    return Chem.MolToSmiles(Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(smi_string),Chem.MolFromSmiles('*'),Chem.MolFromSmiles('[H]'),True)[0]))

def remove_dummys_mol(smi_string):
    return Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(smi_string),Chem.MolFromSmiles('*'),Chem.MolFromSmiles('[H]'),True)[0])

def fragment_mol(smi, cid, pattern="[#6+0;!$(*=,#[!#6])]!@!=!#[*]"):
    mol = Chem.MolFromSmiles(smi)

    #different cuts can give the same fragments
    #to use outlines to remove them
    outlines = set()

    if (mol == None):
        sys.stderr.write("Can't generate mol for: %s\n" % (smi))
    else:
        frags = rdMMPA.FragmentMol(mol, minCuts=2, maxCuts=2, maxCutBonds=100, pattern=pattern, resultsAsMols=False)
        for core, chains in frags:
            output = '%s,%s,%s,%s' % (smi, cid, core, chains)
            if (not (output in outlines)):
                outlines.add(output)
        if not outlines:
            # for molecules with no cuts, output the parent molecule itself
            outlines.add('%s,%s,,' % (smi,cid))

    return outlines


def fragment_dataset(smiles, linker_min=3, fragment_min=5, min_path_length=2, linker_leq_frags=True, verbose=False):
    successes = []

    for count, smi in enumerate(smiles):
        smi = smi.rstrip()
        smiles = smi
        cmpd_id = smi

        # Fragment smi
        o = fragment_mol(smiles, cmpd_id)

        # Checks if suitable fragmentation
        for l in o:
            smiles = l.replace('.',',').split(',')
            mols = [Chem.MolFromSmiles(smi) for smi in smiles[1:]]
            add = True
            fragment_sizes = []
            for i, mol in enumerate(mols):
                # Linker
                if i == 1:
                    linker_size = mol.GetNumHeavyAtoms()
                    # Check linker at least than minimum size
                    if linker_size < linker_min:
                        add = False
                        break
                    # Check path between the fragments at least minimum
                    dummy_atom_idxs = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
                    if len(dummy_atom_idxs) != 2:
                        print("Error")
                        add = False
                        break
                    else:
                        path_length = len(Chem.rdmolops.GetShortestPath(mol, dummy_atom_idxs[0], dummy_atom_idxs[1]))-2
                        if path_length < min_path_length:
                            add = False
                            break
                # Fragments
                elif i > 1:
                    fragment_sizes.append(mol.GetNumHeavyAtoms())
                    min_fragment_size = min(fragment_sizes)
                    # Check fragment at least than minimum size
                    if mol.GetNumHeavyAtoms() < fragment_min:
                        add = False
                        break
                    # Check linker not bigger than fragments
                    if linker_leq_frags:
                        if min_fragment_size < linker_size:
                            add = False
                            break
            if add == True:
                successes.append(l)
        
        if verbose:
            # Progress
            if count % 1000 == 0:
                print("\rProcessed smiles: " + str(count), end='')
    
    # Reformat output
    fragmentations = []
    for suc in successes:
        fragmentations.append(suc.replace('.',',').split(',')[1:])
    
    return fragmentations

def get_linker(full_mol, clean_frag, starting_point):
    # INPUT FORMAT: molecule (RDKit mol object), clean fragments (RDKit mol object), starting fragments (SMILES)
        
    # Get matches of fragments
    matches = list(full_mol.GetSubstructMatches(clean_frag))
    
    # If no matches, terminate
    if len(matches) == 0:
        print("No matches")
        return ""

    # Get number of atoms in linker
    linker_len = full_mol.GetNumHeavyAtoms() - clean_frag.GetNumHeavyAtoms()
    if linker_len == 0:
        return ""
    
    # Setup
    mol_to_break = Chem.Mol(full_mol)
    Chem.Kekulize(full_mol, clearAromaticFlags=True)
    
    poss_linker = []

    if len(matches)>0:
        # Loop over matches
        for match in matches:
            mol_rw = Chem.RWMol(full_mol)
            # Get linker atoms
            linker_atoms = list(set(list(range(full_mol.GetNumHeavyAtoms()))).difference(match))
            linker_bonds = []
            atoms_joined_to_linker = []
            # Loop over starting fragments atoms
            # Get (i) bonds between starting fragments and linker, (ii) atoms joined to linker
            for idx_to_delete in sorted(match, reverse=True):
                nei = [x.GetIdx() for x in mol_rw.GetAtomWithIdx(idx_to_delete).GetNeighbors()]
                intersect = set(nei).intersection(set(linker_atoms))
                if len(intersect) == 1:
                    linker_bonds.append(mol_rw.GetBondBetweenAtoms(idx_to_delete,list(intersect)[0]).GetIdx())
                    atoms_joined_to_linker.append(idx_to_delete)
                elif len(intersect) > 1:
                    for idx_nei in list(intersect):
                        linker_bonds.append(mol_rw.GetBondBetweenAtoms(idx_to_delete,idx_nei).GetIdx())
                        atoms_joined_to_linker.append(idx_to_delete)
                        
            # Check number of atoms joined to linker
            # If not == 2, check next match
            if len(set(atoms_joined_to_linker)) != 2:
                continue
            
            # Delete starting fragments atoms
            for idx_to_delete in sorted(match, reverse=True):
                mol_rw.RemoveAtom(idx_to_delete)
            
            linker = Chem.Mol(mol_rw)
            # Check linker required num atoms
            if linker.GetNumHeavyAtoms() == linker_len:
                mol_rw = Chem.RWMol(full_mol)
                # Delete linker atoms
                for idx_to_delete in sorted(linker_atoms, reverse=True):
                    mol_rw.RemoveAtom(idx_to_delete)
                frags = Chem.Mol(mol_rw)
                # Check there are two disconnected fragments
                if len(Chem.rdmolops.GetMolFrags(frags)) == 2:
                    # Fragment molecule into starting fragments and linker
                    fragmented_mol = Chem.FragmentOnBonds(mol_to_break, linker_bonds)
                    # Remove starting fragments from fragmentation
                    linker_to_return = Chem.Mol(fragmented_mol)
                    qp = Chem.AdjustQueryParameters()
                    qp.makeDummiesQueries=True
                    for f in starting_point.split('.'):
                        qfrag = Chem.AdjustQueryProperties(Chem.MolFromSmiles(f),qp)
                        linker_to_return = AllChem.DeleteSubstructs(linker_to_return, qfrag, onlyFrags=True)
                    
                    # Check linker is connected and two bonds to outside molecule
                    if len(Chem.rdmolops.GetMolFrags(linker)) == 1 and len(linker_bonds) == 2:
                        Chem.Kekulize(linker_to_return, clearAromaticFlags=True)
                        # If for some reason a starting fragment isn't removed (and it's larger than the linker), remove (happens v. occassionally)
                        if len(Chem.rdmolops.GetMolFrags(linker_to_return)) > 1:
                            for frag in Chem.MolToSmiles(linker_to_return).split('.'):
                                if Chem.MolFromSmiles(frag).GetNumHeavyAtoms() == linker_len:
                                    return frag
                        return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(linker_to_return)))
                    
                    # If not, add to possible linkers (above doesn't capture some complex cases)
                    else:
                        fragmented_mol = Chem.MolFromSmiles(Chem.MolToSmiles(fragmented_mol), sanitize=False)
                        linker_to_return = AllChem.DeleteSubstructs(fragmented_mol, Chem.MolFromSmiles(starting_point))
                        poss_linker.append(Chem.MolToSmiles(linker_to_return))
    
    # If only one possibility, return linker
    if len(poss_linker) == 1:
        return poss_linker[0]
    # If no possibilities, process failed
    elif len(poss_linker) == 0:
        print("FAIL:", Chem.MolToSmiles(full_mol), Chem.MolToSmiles(clean_frag), starting_point)
        return ""
    # If multiple possibilities, process probably failed
    else:
        print("More than one poss linker. ", poss_linker)
        return poss_linker[0]


def get_frags(full_mol, clean_frag, starting_point):
    matches = list(full_mol.GetSubstructMatches(clean_frag))
    linker_len = full_mol.GetNumHeavyAtoms() - clean_frag.GetNumHeavyAtoms()

    if linker_len == 0:
        return full_mol

    Chem.Kekulize(full_mol, clearAromaticFlags=True)

    all_frags = []
    all_frags_lengths = []

    if len(matches)>0:
        for match in matches:
            mol_rw = Chem.RWMol(full_mol)
            linker_atoms = list(set(list(range(full_mol.GetNumHeavyAtoms()))).difference(match))
            for idx_to_delete in sorted(match, reverse=True):
                mol_rw.RemoveAtom(idx_to_delete)
            linker = Chem.Mol(mol_rw)
            if linker.GetNumHeavyAtoms() == linker_len:
                mol_rw = Chem.RWMol(full_mol)
                for idx_to_delete in sorted(linker_atoms, reverse=True):
                    mol_rw.RemoveAtom(idx_to_delete)
                frags = Chem.Mol(mol_rw)
                all_frags.append(frags)
                all_frags_lengths.append(len(Chem.rdmolops.GetMolFrags(frags)))
                if len(Chem.rdmolops.GetMolFrags(frags)) == 2:
                    return frags

    return all_frags[np.argmax(all_frags_lengths)]


##### Structural information #####
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def compute_distance_and_angle(mol, smi_linker, smi_frags):
    try:
        frags = [Chem.MolFromSmiles(frag) for frag in smi_frags.split(".")]
        frags = Chem.MolFromSmiles(smi_frags)
        linker = Chem.MolFromSmiles(smi_linker)
        # Include dummy in query
        du = Chem.MolFromSmiles('*')
        qp = Chem.AdjustQueryParameters()
        qp.makeDummiesQueries=True
        # Renumber based on frags (incl. dummy atoms)
        aligned_mols = []

        sub_idx = []
        # Align to frags and linker
        qfrag = Chem.AdjustQueryProperties(frags,qp)
        frags_matches = list(mol.GetSubstructMatches(qfrag, uniquify=False))
        qlinker = Chem.AdjustQueryProperties(linker,qp)
        linker_matches = list(mol.GetSubstructMatches(qlinker, uniquify=False))
            
        # Loop over matches
        for frag_match, linker_match in product(frags_matches, linker_matches):
            # Check if match
            f_match = [idx for num, idx in enumerate(frag_match) if frags.GetAtomWithIdx(num).GetAtomicNum() != 0]
            l_match = [idx for num, idx in enumerate(linker_match) if linker.GetAtomWithIdx(num).GetAtomicNum() != 0 and idx not in f_match]
            if len(set(list(f_match)+list(l_match))) == mol.GetNumHeavyAtoms():
            #if len(set(list(frag_match)+list(linker_match))) == mol.GetNumHeavyAtoms():
                break
        # Add frag indices
        sub_idx += frag_match
        # Add linker indices to end
        sub_idx += [idx for num, idx in enumerate(linker_match) if linker.GetAtomWithIdx(num).GetAtomicNum() != 0 and idx not in sub_idx]

        nodes_to_keep = [i for i in range(len(frag_match))]

        aligned_mols.append(Chem.rdmolops.RenumberAtoms(mol, sub_idx))
        aligned_mols.append(frags)
            
        # Renumber dummy atoms to end
        dummy_idx = []
        for atom in aligned_mols[1].GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummy_idx.append(atom.GetIdx())
        for i, mol in enumerate(aligned_mols):
            sub_idx = list(range(aligned_mols[1].GetNumHeavyAtoms()+2))
            for idx in dummy_idx:
                sub_idx.remove(idx)
                sub_idx.append(idx)
            if i == 0:
                mol_range = list(range(mol.GetNumHeavyAtoms()))
            else:
                mol_range = list(range(mol.GetNumHeavyAtoms()+2))
            idx_to_add = list(set(mol_range).difference(set(sub_idx)))
            sub_idx.extend(idx_to_add)
            aligned_mols[i] = Chem.rdmolops.RenumberAtoms(mol, sub_idx)
            
        # Get exit vectors
        exit_vectors = []
        linker_atom_idx = []
        for atom in aligned_mols[1].GetAtoms():
            if atom.GetAtomicNum() == 0:
                if atom.GetIdx() in nodes_to_keep:
                    nodes_to_keep.remove(atom.GetIdx())
                for nei in atom.GetNeighbors():
                    exit_vectors.append(nei.GetIdx())
                linker_atom_idx.append(atom.GetIdx())
                    
        # Get coords
        conf = aligned_mols[0].GetConformer()
        exit_coords = []
        for exit in exit_vectors:
            exit_coords.append(np.array(conf.GetAtomPosition(exit)))
        linker_coords = []
        for linker_atom in linker_atom_idx:
            linker_coords.append(np.array(conf.GetAtomPosition(linker_atom)))
        
        # Get angle
        v1_u = unit_vector(linker_coords[0]-exit_coords[0])
        v2_u = unit_vector(linker_coords[1]-exit_coords[1])
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
                    
        # Get linker length
        linker = Chem.MolFromSmiles(smi_linker)
        linker_length = linker.GetNumHeavyAtoms()

        # Get distance
        distance = np.linalg.norm(exit_coords[0]-exit_coords[1])
                
        # Record results
        return distance, angle
    
    except:
        print(Chem.MolToSmiles(mol), smi_linker, smi_frags)
        return None, None
    

def compute_distance_and_angle_dataset(fragmentations, path_to_conformers, dataset, verbose=False):
    if dataset not in ["ZINC", "CASF"]:
        print("Dataset must be either ZINC or CASF")
        return None, None, None, None
    # Load conformers
    conformers = Chem.SDMolSupplier(path_to_conformers)
    # Convert dataset to dictionary
    dataset_dict = {}
    for toks in fragmentations:
        if toks[0] in dataset_dict:
            dataset_dict[toks[0]].append([toks[1], toks[2]+'.'+toks[3]])
        else:
            dataset_dict[toks[0]] = [[toks[1], toks[2]+'.'+toks[3]]]
    
    # Initialise placeholders for results
    fragmentations_new = []
    distances = []
    angles = []

    du = Chem.MolFromSmiles('*')

    # Record number of failures
    fail_count = 0
    fail_count_conf = 0
    fails = []

    # Loop over conformers
    for count, mol in enumerate(conformers):
        # Check mol of conformer in fragmentations
        if mol is not None:
            mol_name = mol.GetProp("_Name") if dataset == "ZINC" else Chem.MolToSmiles(mol)
            if mol_name in dataset_dict:
                # Loop over all fragmentations of this mol
                for fragments in dataset_dict[mol_name]:
                    dist, ang = compute_distance_and_angle(mol, fragments[0], fragments[1])
                    if dist and ang:
                        fragmentations_new.append([mol_name] + fragments)
                        distances.append(dist)
                        angles.append(ang)
                    else:
                        print(fragments[0], fragments[1])
                        print(dist, ang)
                        fails.append([mol_name] + fragments)
                        fail_count += 1
        else:
            fail_count_conf += 1
    
        if verbose:
            # Progress
            if count % 1000 == 0:
                print("\rMol: %d" % count, end = '')

    if verbose:
        print("\rDone")
        print("Fail count conf %d" % fail_count_conf)
    
    return fragmentations_new, distances, angles, (fail_count, fail_count_conf, fails)


##### Drawing #####

def mol_with_atom_index( mol ):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol


##### Plotting #####

def plot_hist(results, prop_name, labels=None, alpha_val=0.5):
    if labels is None:
        labels = [str(i) for i in range(len(results))]
    results_all = []
    for res in results:
        results_all.extend(res)
        
    min_value = np.amin(results_all)
    max_value = np.amax(results_all)
    num_bins = 20.0
    binwidth = (max_value - min_value) / num_bins
    
    if prop_name in ['heavy_atoms', 'num_C', 'num_N', 'num_O', 'num_F', 'hba', 'hbd', 'ring_ct', 'avg_ring_size', 'rot_bnds', 'stereo_cnts', "Linker lengths"]:
        min_value = math.floor(min_value)
        max_value = math.ceil(max_value)
        diff = max_value - min_value
        binwidth = max(1, int(diff/num_bins))
        
    if prop_name in ["Mol_similarity", "QED"]:
        min_value = 0.0
        max_value = 1.01
        diff = max_value - min_value
        binwidth = diff/num_bins
        
    if prop_name in ["SC_RDKit"]:
        max_value = 1.01
        diff = max_value - min_value
        binwidth = diff/num_bins
 
    bins = np.arange(min_value - 2* binwidth, max_value + 2* binwidth, binwidth)
    
    dens_all = []
    for i, res in enumerate(results):
        if not labels:
            dens, _ , _ = plt.hist(np.array(res).flatten(), bins=bins, density=True, alpha=alpha_val)
        elif labels:
            dens, _ , _ = plt.hist(np.array(res).flatten(), bins=bins, density=True, alpha=alpha_val,label=labels[i])
        #dens, _ , _ = plt.hist(res, bins=bins, density=True, alpha=alpha_val, label=labels[i])
        dens_all.extend(dens)

    plt.xlabel(prop_name)
    plt.ylabel('Proportion')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1*0.8,x2/0.8,y1,1.1*max(dens_all)))
    plt.title('Distribution of ' + prop_name)
    plt.legend()
    plt.grid(True)

    plt.show()

##### Calc props #####

def calc_dataset_props(dataset, verbose=False):
    # Calculate mol props for an entire dataset of smiles strings
    results = []
    for i, smiles in enumerate(dataset):
        props = calc_mol_props(smiles)
        if props is not None:
            results.append(props)
        if verbose and i % 1000 == 0:
            print("\rProcessed smiles: " + str(i), end='')
    print("\nDone calculating properties")
    return np.array(results)

def calc_mol_props(smiles):
    # Create RDKit mol
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Error passing: %s" % smiles)
        return None
    
    # QED
    qed = Chem.QED.qed(mol)
    # Synthetic accessibility score - number of cycles (rings with > 6 atoms)
    sas = sascorer.calculateScore(mol)
    # Cyles with >6 atoms
    ri = mol.GetRingInfo()
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 6:
            nMacrocycles += 1

    prop_array = [qed, sas]

    return prop_array

def calc_sa_score_smi(smi, verbose=False):
    # Create RDKit mol object
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        if verbose:
            print("Error passing: %s" % smi)
        return None
    
    # Synthetic accessibility score
    return sascorer.calculateScore(mol)

def calc_sa_score_mol(mol, verbose=False):
    if mol is None:
        if verbose:
            print("Error passing: %s" % smi)
        return None
    # Synthetic accessibility score
    return sascorer.calculateScore(mol)


##### 2D checks #####

def ring_check_for_filter(res, clean_frag):
    check = True
    gen_mol = Chem.MolFromSmiles(res[2])
    linker = Chem.DeleteSubstructs(gen_mol, clean_frag)
    
    # Get linker rings
    ssr = Chem.GetSymmSSSR(linker)
    # Check rings
    for ring in ssr:
        for atom_idx in ring:
            for bond in linker.GetAtomWithIdx(atom_idx).GetBonds():
                if bond.GetBondType() == 2 and bond.GetBeginAtomIdx() in ring and bond.GetEndAtomIdx() in ring:
                    check = False
    return check

def ring_check_res(res, clean_frag):
    check = True
    gen_mol = Chem.MolFromSmiles(res[1])
    linker = Chem.DeleteSubstructs(gen_mol, clean_frag)
    
    # Get linker rings
    ssr = Chem.GetSymmSSSR(linker)
    # Check rings
    for ring in ssr:
        for atom_idx in ring:
            for bond in linker.GetAtomWithIdx(atom_idx).GetBonds():
                if bond.GetBondType() == 2 and bond.GetBeginAtomIdx() in ring and bond.GetEndAtomIdx() in ring:
                    check = False
    return check

def filters(results, verbose=True):
    count = 0
    total = 0
    for processed, res in enumerate(results):
        total += len(res)
        for m in res:
            # Clean frags
            du = Chem.MolFromSmiles('*')
            clean_frag = Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(m[0]),du,Chem.MolFromSmiles('[H]'),True)[0])
            # Check joined - already taken care of
            #if len(Chem.MolFromSmiles(m[1]).GetSubstructMatch(clean_frag))>0:
            # Check SA score has improved
            if calc_mol_props(m[1])[1] < calc_mol_props(m[0])[1]:
                # Check no non-aromatic double bonds in rings
                if ring_check_res(m, clean_frag):
                    count += 1
        # Progress
        if verbose:
            if processed % 10 == 0:
                print("\rProcessed %d" % processed, end="")
    print("\r",end="")
    return count/total

def sa_filter(results, verbose=True):
    count = 0
    total = 0
    for processed, res in enumerate(results):
        total += len(res)
        for m in res:
            # Check SA score has improved
            if calc_mol_props(m[1])[1] < calc_mol_props(m[0])[1]:
                count += 1
        # Progress
        if verbose:
            if processed % 10 == 0:
                print("\rProcessed %d" % processed, end="")
    print("\r",end="")
    return count/total

def ring_filter(results, verbose=True):
    count = 0
    total = 0
    du = Chem.MolFromSmiles('*')
    for processed, res in enumerate(results):
        total += len(res)
        for m in res:
            # Clean frags
            clean_frag = Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(m[0]),du,Chem.MolFromSmiles('[H]'),True)[0])
            if ring_check_res(m, clean_frag):
                count += 1
        # Progress
        if verbose:
            if processed % 10 == 0:
                print("\rProcessed %d" % processed, end="")
    print("\r",end="")
    return count/total

def check_ring_filter(linker):
    check = True 
    # Get linker rings
    ssr = Chem.GetSymmSSSR(linker)
    # Check rings
    for ring in ssr:
        for atom_idx in ring:
            for bond in linker.GetAtomWithIdx(atom_idx).GetBonds():
                if bond.GetBondType() == 2 and bond.GetBeginAtomIdx() in ring and bond.GetEndAtomIdx() in ring:
                    check = False
    return check

def check_pains(mol, pains_smarts):
    for pain in pains_smarts:
        if mol.HasSubstructMatch(pain):
            return False
    return True

def check_2d_filters(toks, pains_smarts, count=0, verbose=False):
    # Progress
    if verbose:
        if count % 1000 == 0:
            print("\rProcessed: %d" % count, end = '')
    
    # Input format: (Full Molecule (SMILES), Linker (SMILES), Unlinked Fragment 1 (SMILES), Unlinked Fragment 2 (SMILES))
    frags = Chem.MolFromSmiles(toks[2] + '.' + toks[3])
    linker = Chem.MolFromSmiles(toks[1])
    full_mol = Chem.MolFromSmiles(toks[0])
    # Remove dummy atoms from unlinked fragments
    du = Chem.MolFromSmiles('*')
    clean_frag = Chem.RemoveHs(AllChem.ReplaceSubstructs(frags,du,Chem.MolFromSmiles('[H]'),True)[0])
    
    # Check: Unlinked fragments in full molecule
    if len(full_mol.GetSubstructMatch(clean_frag))>0:
        # Check: SA score improved from unlinked fragments to full molecule
        if calc_sa_score_mol(full_mol) < calc_sa_score_mol(frags):
            # Check: No non-aromatic rings with double bonds
            if check_ring_filter(linker): 
                # Check: Pass pains filters
                if check_pains(full_mol, pains_smarts):
                    return True
            else:
                if check_ring_filter(linker):
                    print(toks)

    return False

# def check_2d_filters_dataset(fragmentations, n_cores=1, pains_smarts_loc='./wehi_pains.csv'):
#     # Load pains filters
#     with open(pains_smarts_loc, 'r') as f:
#         pains_smarts = [Chem.MolFromSmarts(line[0], mergeHs=True) for line in csv.reader(f)]
        
#     with Parallel(n_jobs=n_cores, backend='multiprocessing') as parallel:
#         results = parallel(delayed(check_2d_filters)(toks, pains_smarts, count, True) for count, toks in enumerate(fragmentations))

#     fragmentations_filtered = [toks for toks, res in zip(fragmentations, results) if res]
    
#     return fragmentations_filtered

def calc_2d_filters(toks, pains_smarts): 
    try:
        # Input format: (Full Molecule (SMILES), Linker (SMILES), Unlinked Fragments (SMILES))
        frags = Chem.MolFromSmiles(toks[2])
        linker = Chem.MolFromSmiles(toks[1])
        full_mol = Chem.MolFromSmiles(toks[0])
        # Remove dummy atoms from unlinked fragments
        du = Chem.MolFromSmiles('*')
        clean_frag = Chem.RemoveHs(AllChem.ReplaceSubstructs(frags,du,Chem.MolFromSmiles('[H]'),True)[0])
    
        res = []
        # Check: Unlinked fragments in full molecule
        if len(full_mol.GetSubstructMatch(clean_frag))>0:
            # Check: SA score improved from unlinked fragments to full molecule
            if calc_sa_score_mol(full_mol) < calc_sa_score_mol(frags):
                res.append(True)
            else:
               res.append(False)
            # Check: No non-aromatic rings with double bonds
            if check_ring_filter(linker): 
               res.append(True)
            else:
                res.append(False)
            # Check: Pass pains filters
            if check_pains(full_mol, pains_smarts):
               res.append(True)
            else:
               res.append(False)     
        return res
    except:
        return [False, False, False]

# def calc_filters_2d_dataset(results, pains_smarts_loc, n_cores=1):
#     # Load pains filters
#     with open(pains_smarts_loc, 'r') as f:
#         pains_smarts = [Chem.MolFromSmarts(line[0], mergeHs=True) for line in csv.reader(f)]
        
#     with Parallel(n_jobs=n_cores, backend='multiprocessing') as parallel:
#         filters_2d = parallel(delayed(calc_2d_filters)([toks[2], toks[4], toks[1]], pains_smarts) for toks in results)
        
#     return filters_2d

##### Metrics #####

def unique(results):
    total_dupes = 0
    total = 0
    for res in results:
        original_num = len(res)
        test_data = set(res)
        new_num = len(test_data)
        total_dupes += original_num - new_num
        total += original_num   
    return 1 - total_dupes/float(total)

def valid(results, max_entries=50):
    total = 0
    for res in results:
        total+= max_entries - len(res)
    return 1.0 - total / (len(results)*max_entries)

def recovered_by_sim(results):
    recovered = 0
    total = len(results)
    for res in results:
        success = False
        for m in res:
            #if DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(m[1]), 2, 2048), 
            #                                  AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(m[2]), 2, 2048)) == 1:
            if DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(m[0]), 2, 2048), 
                                              AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(m[2]), 2, 2048)) == 1:
            
                recovered += 1
                break
    return recovered/total

def recovered_by_smi(results):
    recovered = 0
    total = len(results)
    for res in results:
        success = False
        for m in res:
            #if m[1] == m[2]:
            if m[0] == m[2]:
                recovered += 1
                break
    return recovered/total

def recovered_by_smi_canon(results):
    recovered = 0
    total = len(results)
    for res in results:
        success = False
        for m in res:
            #if Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(m[1]))) == Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(m[2]))):
            if Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(m[0]))) == Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(m[2]))):
                recovered += 1
                break
    return recovered/total

def check_recovered_original_mol(results):
    outcomes = []
    for res in results:
        success = False
        # Load original mol and canonicalise
        orig_mol = Chem.MolFromSmiles(res[0][0])
        Chem.RemoveStereochemistry(orig_mol)
        orig_mol = Chem.MolToSmiles(Chem.RemoveHs(orig_mol))
        #orig_mol = MolStandardize.canonicalize_tautomer_smiles(orig_mol)
        # Check generated mols
        for m in res:
            gen_mol = Chem.MolFromSmiles(m[2])
            Chem.RemoveStereochemistry(gen_mol)
            gen_mol = Chem.MolToSmiles(Chem.RemoveHs(gen_mol))
            #gen_mol = MolStandardize.canonicalize_tautomer_smiles(gen_mol)
            if gen_mol == orig_mol:
                outcomes.append(True)
                success = True
                break
        if not success:
            outcomes.append(False)
    return outcomes

def average_linker_length(results):
    total_linker_length = 0
    total_num = 0
    for res in results:
        for m in res:
            linker_length = Chem.MolFromSmiles(m[1]).GetNumHeavyAtoms() - Chem.MolFromSmiles(m[0]).GetNumHeavyAtoms()
            total_linker_length+=linker_length
            total_num += 1
    return total_linker_length/total_num

def get_linker_length(results):
    linker_lengths = []
    for res in results:
        for m in res:
            linker_length = Chem.MolFromSmiles(m[1]).GetNumHeavyAtoms() - Chem.MolFromSmiles(m[0]).GetNumHeavyAtoms()
            linker_lengths.append(linker_length)
    return linker_lengths

##### Join fragments #####

def join_frag_linker(linker, st_pt, random_join=True):
    
    if linker == "":
        du = Chem.MolFromSmiles('*')
        #print(Chem.MolToSmiles(Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(st_pt),du,Chem.MolFromSmiles('[H]'),True)[0])).split('.')[0])
        return Chem.MolToSmiles(Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(st_pt),du,Chem.MolFromSmiles('[H]'),True)[0])).split('.')[0]

    combo = Chem.CombineMols(Chem.MolFromSmiles(linker), Chem.MolFromSmiles(st_pt))

    # Include dummy in query
    du = Chem.MolFromSmiles('*')
    qp = Chem.AdjustQueryParameters()
    qp.makeDummiesQueries=True

    qlink = Chem.AdjustQueryProperties(Chem.MolFromSmiles(linker),qp)
    linker_atoms = combo.GetSubstructMatches(qlink)
    if len(linker_atoms)>1:
        for l_atoms in linker_atoms:
            count_dummy = 0
            for a in l_atoms:
                if combo.GetAtomWithIdx(a).GetAtomicNum() == 0:
                    count_dummy +=1
            if count_dummy == 2:
                break
        linker_atoms = l_atoms
    else:
        linker_atoms = linker_atoms[0]
    linker_dummy_bonds = []
    linker_dummy_bonds_at = []
    linker_exit_points = []
    for atom in linker_atoms:
        if combo.GetAtomWithIdx(atom).GetAtomicNum() == 0:
            linker_dummy_bonds.append(combo.GetAtomWithIdx(atom).GetBonds()[0].GetIdx())
            linker_dummy_bonds_at.append((atom, combo.GetAtomWithIdx(atom).GetNeighbors()[0].GetIdx()))
            linker_exit_points.append(combo.GetAtomWithIdx(atom).GetNeighbors()[0].GetIdx())

    qst_pt = Chem.AdjustQueryProperties(Chem.MolFromSmiles(st_pt),qp)
    st_pt_atoms = combo.GetSubstructMatches(qst_pt)
    st_pt_atoms = list(set(range(combo.GetNumAtoms())).difference(linker_atoms))

    st_pt_dummy_bonds = []
    st_pt_dummy_bonds_at = []
    st_pt_exit_points = []
    for atom in st_pt_atoms:
        if combo.GetAtomWithIdx(atom).GetAtomicNum() == 0:
            st_pt_dummy_bonds.append(combo.GetAtomWithIdx(atom).GetBonds()[0].GetIdx())
            st_pt_dummy_bonds_at.append((atom, combo.GetAtomWithIdx(atom).GetNeighbors()[0].GetIdx()))
            st_pt_exit_points.append(combo.GetAtomWithIdx(atom).GetNeighbors()[0].GetIdx())

    combo_rw = Chem.EditableMol(combo)

    if random_join:
        np.random.shuffle(st_pt_exit_points)
        for atom_1, atom_2 in zip(linker_exit_points, st_pt_exit_points):
            if atom_1 == atom_2:
                print(linker, st_pt)
                break
            combo_rw.AddBond(atom_1, atom_2 ,order=Chem.rdchem.BondType.SINGLE)

        bonds_to_break = linker_dummy_bonds_at + st_pt_dummy_bonds_at
        for bond in sorted(bonds_to_break, reverse=True):
            combo_rw.RemoveBond(bond[0], bond[1])

        final_mol = combo_rw.GetMol()
        final_mol = sorted(Chem.MolToSmiles(final_mol).split('.'), key=lambda x: len(x), reverse=True)[0]
        return final_mol

    else:
        final_mols = []
        for st_pt_exit_pts in [st_pt_exit_points, st_pt_exit_points[::-1]]:
            combo_rw = Chem.EditableMol(combo)
            for atom_1, atom_2 in zip(linker_exit_points, st_pt_exit_pts):
                if atom_1 == atom_2:
                    print(linker, st_pt)
                    break
                combo_rw.AddBond(atom_1, atom_2 ,order=Chem.rdchem.BondType.SINGLE)

            bonds_to_break = linker_dummy_bonds_at + st_pt_dummy_bonds_at
            for bond in sorted(bonds_to_break, reverse=True):
                combo_rw.RemoveBond(bond[0], bond[1])

            final_mol = combo_rw.GetMol()
            final_mol = sorted(Chem.MolToSmiles(final_mol).split('.'), key=lambda x: len(x), reverse=True)[0]
            final_mols.append(final_mol)
        return final_mols
    
##### 3D Metrics #####
def SC_RDKit_full_mol(gen_mol, ref_mol):
    try:
        # Align
        pyO3A = rdMolAlign.GetO3A(gen_mol, ref_mol).Align()
        # Calc SC_RDKit score
        score = calc_SC_RDKit.calc_SC_RDKit_score(gen_mol, ref_mol)
        return score
    except:
        return -0.5 # Dummy score

def SC_RDKit_full_scores(gen_mols):
    return [SC_RDKit_full_mol(gen_mol, ref_mol) for (gen_mol, ref_mol) in tqdm(gen_mols)]

def SC_RDKit_frag_mol(gen_mol, ref_mol, start_pt):
    try:
        # Delete linker - Gen mol
        du = Chem.MolFromSmiles('*')
        clean_frag = Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(start_pt),du,Chem.MolFromSmiles('[H]'),True)[0])

        fragmented_mol = get_frags(gen_mol, clean_frag, start_pt)
        if fragmented_mol is not None:
            # Delete linker - Ref mol
            clean_frag_ref = Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(start_pt),du,Chem.MolFromSmiles('[H]'),True)[0])
            fragmented_mol_ref = get_frags(ref_mol, clean_frag_ref, start_pt)
            if fragmented_mol_ref is not None:
                # Sanitize
                Chem.SanitizeMol(fragmented_mol)
                Chem.SanitizeMol(fragmented_mol_ref)
                # Align
                pyO3A = rdMolAlign.GetO3A(fragmented_mol, fragmented_mol_ref).Align()
                # Calc SC_RDKit score
                score = calc_SC_RDKit.calc_SC_RDKit_score(fragmented_mol, fragmented_mol_ref)
                return score
    except:
        return -0.5 # Dummy score

def SC_RDKit_frag_scores(gen_mols):
    return [SC_RDKit_frag_mol(gen_mol, ref_mol, frag_smi) for (gen_mol, ref_mol, frag_smi) in tqdm(gen_mols)]

def rmsd_frag_mol(gen_mol, ref_mol, start_pt):
    try:
        # Delete linker - Gen mol
        du = Chem.MolFromSmiles('*')
        clean_frag = Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(start_pt),du,Chem.MolFromSmiles('[H]'),True)[0])

        fragmented_mol = get_frags(gen_mol, clean_frag, start_pt)
        if fragmented_mol is not None:
            # Delete linker - Ref mol
            clean_frag_ref = Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(start_pt),du,Chem.MolFromSmiles('[H]'),True)[0])
            fragmented_mol_ref = get_frags(ref_mol, clean_frag_ref, start_pt)
            if fragmented_mol_ref is not None:
                # Sanitize
                Chem.SanitizeMol(fragmented_mol)
                Chem.SanitizeMol(fragmented_mol_ref)
                # Align
                pyO3A = rdMolAlign.GetO3A(fragmented_mol, fragmented_mol_ref).Align()
                rms = rdMolAlign.GetBestRMS(fragmented_mol, fragmented_mol_ref)
                return rms #score
    except:
        return 100 # Dummy RMSD

def rmsd_frag_scores(gen_mols):
    return [rmsd_frag_mol(gen_mol, ref_mol, start_pt) for (gen_mol, ref_mol, start_pt) in gen_mols]

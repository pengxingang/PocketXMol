#!/usr/bin/env python
"""
sdf_to_pdb_no_residue_info.py

This script reads an SDF file without any residue information and attempts to
annotate residue types based on connectivity. It uses two strategies:
  1. If the molecule has multiple disconnected fragments, each fragment is treated as a residue.
  2. If the molecule is a single connected component (common for peptides),
     it attempts to identify peptide bonds and split the molecule accordingly using a simple SMARTS pattern.
     
Usage:
    python sdf_to_pdb_no_residue_info.py input.sdf output.pdb
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
import argparse
import sys

sys.path.append('.')
from utils.residue import get_res_name, set_atom_name

def atom_index_to_subgraph(mol, atom_list):
    edge_indices = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        begin_aid = bond.GetBeginAtomIdx()
        end_aid = bond.GetEndAtomIdx()
        if begin_aid in atom_list and end_aid in atom_list:
            edge_indices.append(i)
    return edge_indices

def get_main_chain_fragment(mol, main_chain_length):
    """
    Locate the main chain substructure in the complete molecule using a protein backbone SMARTS.
    The backbone is assumed to follow the repeating unit:
      -N-C-C(=O)-
    For one residue, use: "[N]-[C]-[C](=O)"
    For multiple residues, the pattern is constructed as:
      "[N]-[C]-[C](=O)" + ("-[N]-[C]-[C](=O)")*(main_chain_length - 1)
    Returns a tuple (main_chain_fragment, atom_mapping) where:
      - main_chain_fragment is an RDKit molecule corresponding to the main chain.
      - atom_mapping is a dictionary mapping submol atom indices to original molecule atom indices.
    """
    if main_chain_length < 1:
        raise ValueError("Main chain length must be at least 1.")
    
    # Build the protein backbone SMARTS pattern using the repeating unit -N-C-C(=O)-
    if main_chain_length == 1:
        pattern_str = "[N]-[C]-[C](=O)"
    else:
        pattern_str = "[N]-[C]-[C](=O)" + ("-[N]-[C]-[C](=O)") * (main_chain_length - 1)
    
    chain_pattern = Chem.MolFromSmarts(pattern_str)
    if chain_pattern is None:
        raise ValueError("Error creating SMARTS pattern from string: " + pattern_str)
    
    matches = mol.GetSubstructMatches(chain_pattern)
    if not matches:
        raise ValueError("No main chain substructure matching the SMARTS pattern was found.")
    
    main_chain_match = matches[0]
    # print(f"Main chain match (original atom indices): {main_chain_match}")
    
    # Extract the substructure and preserve the atom mapping.
    atom_mapping = {}
    subgraph = atom_index_to_subgraph(mol, main_chain_match)
    main_chain_fragment = Chem.PathToSubmol(mol, subgraph, atomMap=atom_mapping)
    atom_mapping = {sub_idx: orig_idx for orig_idx, sub_idx in atom_mapping.items()}
    
    return main_chain_fragment, atom_mapping



def find_peptide_bonds(main_chain):
    """
    Identify peptide bonds in the main chain using a simple SMARTS pattern for an amide linkage.
    The SMARTS pattern "[C](=[O])-[N]" matches a carbonyl carbon connected to a nitrogen.
    Returns a list of tuples, each containing the atom indices (in main_chain) that form a peptide bond.
    """
    # Define a simple peptide bond SMARTS.
    peptide_bond_smarts = "[C](=[O])-[N]"
    pattern = Chem.MolFromSmarts(peptide_bond_smarts)
    matches = main_chain.GetSubstructMatches(pattern)
    bonds = []
    for match in matches:
        # In our pattern, the order is: 0 = carbonyl carbon, 1 = oxygen, 2 = nitrogen.
        atom_idx_C = match[0]
        atom_idx_N = match[2]
        bond = main_chain.GetBondBetweenAtoms(atom_idx_C, atom_idx_N)
        if bond is not None:
            bonds.append((atom_idx_C, atom_idx_N))
    return bonds

def fragment_mol_on_peptide_bonds(mol, atom_pairs):
    """
    Break the mol at the specified peptide bond locations.
    Returns a list of molecule fragments, each representing a residue.
    """
    bonds = [mol.GetBondBetweenAtoms(*b).GetIdx() for b in atom_pairs]
    if not bonds:
        # If no peptide bonds are found, treat the entire main chain as one residue.
        print('No peptide bond was found.')
        return [mol]
    try:
        fragmented_mol = rdmolops.FragmentOnBonds(mol, bonds, addDummies=False)
        # Get the resulting fragments
        frag_atom_index = []
        fragments = Chem.GetMolFrags(fragmented_mol, asMols=True, sanitizeFrags=True,
                                     frags=frag_atom_index)
        # sort
        frag_atom_index_bb = [frag_atom_index[atoms[0]] for atoms in atom_pairs] +\
            [frag_atom_index[atom_pairs[-1][1]]]
        fragments = [fragments[i] for i in frag_atom_index_bb]
        return fragments
    except Exception as e:
        print("Error fragmenting the mol:", e)
        return [mol]

def write_pdb(residue_fragments, output_pdb, output_residue_path=''):
    """
    Write a PDB file from a list of residue fragments.
    Each fragment is written as a residue with a generic residue name.
    """
    atom_serial = 1
    pdb_lines = []
    chain_id = "A"
    resSeq = 1

    df_res_sm = []
    for i_frag, frag in enumerate(residue_fragments):
        # You may customize residue naming here (e.g., using "RES" plus a counter)
        resName, res_flag = get_res_name(frag)
        if len(resName) > 3:
            print('Not support len (res_name) > 3. Set to UNK.')
            resName = 'UNK'
        frag, atom_names = set_atom_name(frag, resName)
        df_res_sm.append({
            'i_res': i_frag,
            'resname': resName,
            'smiles': Chem.MolToSmiles(frag)
        })
        
        print(f"Residue {i_frag}: {resName}, {Chem.MolToSmiles(frag)}")

        conf = frag.GetConformer()
        for i_atom, atom in enumerate(frag.GetAtoms()):
            pos = conf.GetAtomPosition(atom.GetIdx())
            element = atom.GetSymbol()
            # A simple rule for atom naming: if single-character, add a space before it.
            if atom_names is None:
                atom_name = " " + element if len(element) == 1 else element
            else:
                atom_name = " " + atom_names[i_atom]
            pdb_line = (
                f"ATOM  {atom_serial:5d} {atom_name:4s} {resName:>3s} {chain_id:1s}{resSeq:4d}    "
                f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00          {element:>2s}"
            )
            pdb_lines.append(pdb_line)
            atom_serial += 1
        resSeq += 1

    with open(output_pdb, "w") as f:
        for line in pdb_lines:
            f.write(line + "\n")
        f.write("END\n")
    
    df_res_sm = pd.DataFrame(df_res_sm)
    df_res_sm.to_csv(output_pdb.replace('.pdb', '_sm.csv'), index=False)


def annotate_and_write_pdb(input_sdf, output_pdb, len_pep=None):
    main_chain_length = len_pep

    suppl = Chem.SDMolSupplier(input_sdf, removeHs=False, sanitize=False)
    if not suppl or len(suppl) == 0:
        print("No molecules found in the SDF file!")
        sys.exit(1)

    # For demonstration, process the first molecule in the SDF.
    mol = suppl[0]
    if mol is None:
        print("Error reading the molecule from the SDF.")
        sys.exit(1)

    # Identify the main chain fragment.
    main_chain, main_chain_atom_mapping = get_main_chain_fragment(mol, main_chain_length)
    # print(f"Identified main chain fragment with {main_chain.GetNumAtoms()} atoms.")

    # Find peptide bonds within the main chain.
    peptide_bonds_in_main_chain = find_peptide_bonds(main_chain)
    peptide_bonds = [(main_chain_atom_mapping[a0],
                      main_chain_atom_mapping[a1])
                     for a0,a1 in peptide_bonds_in_main_chain]
    # print(f"Found {len(peptide_bonds)} peptide bond(s) in the main chain with len {len_pep}.")

    # Fragment the main chain at the peptide bonds.
    residues = fragment_mol_on_peptide_bonds(mol, peptide_bonds)
    # print(f"Fragmented the main chain into {len(residues)} residue fragment(s).")

    # Write out the resulting residues to a PDB file.
    in_name = os.path.basename(input_sdf)
    print(f"---Found {len(residues)} residue(s) for {in_name}---")
    write_pdb(residues, output_pdb)
    


if __name__ == "__main__":
    import pandas as pd
    import os
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_path', type=str,
                        default='outputs_use/pepdesign_pxm_20241201_222214')
    parser.add_argument('--len_pep', type=int, default=10)
    args = parser.parse_args()
    
    # # make dir
    df = pd.read_csv(os.path.join(args.gen_path, 'gen_info.csv'))
    df = df[df['tag']=='nonstd']

    sdf_dir = os.path.join(args.gen_path, os.path.basename(args.gen_path)+'_SDF')
    save_dir = os.path.join(args.gen_path, 'PDB_robust')
    os.makedirs(save_dir, exist_ok=True)
    
    # df = df.sample(frac=1,)
    # # fix pdb
    for _, line in tqdm(df.iterrows(), total=len(df)):
        filename = line['filename']
        sdf_file = os.path.join(sdf_dir, filename.replace('.pdb', '_mol.sdf'))
        pdb_file = os.path.join(save_dir, filename)
        len_pep = args.len_pep
        # if os.path.exists(pdb_file):
        #     continue
        try:
            annotate_and_write_pdb(sdf_file, pdb_file, len_pep)
        except Exception as e:
            print(line['filename'], line['data_id'], e)
    print('Done!')
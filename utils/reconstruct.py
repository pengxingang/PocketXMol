"""PocketXMol Molecule Reconstruction Module.

This module handles reconstruction of 3D molecular structures from generated
graph representations. It converts predicted atom types, positions, and bonds
into valid RDKit molecules or PDB structures.

Key functions:
    - reconstruct_from_generated_with_edges: Main reconstruction for small molecules
    - reconstruct_pdb_from_generated_fold: Reconstruction for peptides to PDB format
    - set_rdmol_positions: Update RDKit molecule coordinates
    - add_bond_from_obmol: Bond order assignment using OpenBabel

The reconstruction process:
    1. Create base molecule from atom types and connectivity
    2. Set 3D coordinates
    3. Assign bond orders (using template or OpenBabel)
    4. Sanitize and validate final structure
"""

# Standard library imports
import itertools
import os
import re
import tempfile
from copy import deepcopy
from io import StringIO

# Third-party imports
import numpy as np
from Bio.PDB import Chain, Model, PDBParser, PDBIO, Structure
from openbabel import openbabel as ob
from rdkit import Geometry
from rdkit.Chem import AllChem as Chem
from scipy.spatial.distance import pdist, squareform

# Local imports
from process.process_torsional_info import get_mol_from_data
from utils.misc import TimeoutException, time_limit

# Module-level constants
ptable = Chem.GetPeriodicTable()

def reconstruct_pdb_from_generated_fold(mol_info, check_atom=True, gt_path=''):
    """
    Reconstruct peptide PDB structure from generated for structure prediction tasks.
    
    Takes generated 3D coordinates and updates a ground truth PDB template.
    Used for peptide design and conformation generation tasks.
    
    Args:
        mol_info: Dictionary containing:
            - atom_pos: Generated 3D coordinates [N, 3]
            - element: Atomic numbers [N]
            - task: Task type ('conf' or 'dock')
            - db, data_id: Dataset identifiers for ground truth lookup
        check_atom: Whether to verify atom types match ground truth
        gt_path: Optional path to ground truth PDB file
        
    Returns:
        PDB structure object with updated coordinates
    """
    assert mol_info['task'] in ['conf', 'dock'], 'this func only recon pos'
    parser = PDBParser(QUIET=True)

    # get gt pdb file
    db = mol_info['db']
    data_id = mol_info['data_id']
    if os.path.exists(gt_path):
        pdb_struc = parser.get_structure(data_id, gt_path)
    else:
        gt_pdb_file = f'data/{db}/files/peptides/{data_id}_pep.pdb'
        if not os.path.exists(gt_pdb_file):
            raise Exception(f'gt pdb file not found: {gt_pdb_file}')
        pdb_struc = parser.get_structure(data_id, gt_pdb_file)
    
    # get generated pos
    xyz = mol_info['atom_pos'].tolist()
    atomic_nums = mol_info['element'].tolist()

    # change coordinates
    for i_atom, atom in enumerate(pdb_struc.get_atoms()):
        if check_atom:
            # check element
            assert ptable.GetAtomicNumber(atom.element) == atomic_nums[i_atom],\
                f'element not match: {atom.element} vs {atomic_nums[i_atom]}'
        atom.set_coord(xyz[i_atom])

    # rdmol
    pdb_block = StringIO()
    io = PDBIO()
    io.set_structure(pdb_struc)
    io.save(pdb_block)
    pdb_block = pdb_block.getvalue()
    rdmol = Chem.MolFromPDBBlock(pdb_block, )
    
    return pdb_struc, rdmol


def reconstruct_pdb_from_generated(mol_info, check_atom=True, gt_path=''):
    
    if mol_info['task'] in ['conf', 'dock']:
        try:
            return reconstruct_pdb_from_generated_fold(mol_info, check_atom=check_atom, gt_path=gt_path)
        except Exception as e:
            print('recon from pdb file failed:', e)
            pass
    # get rdmol first
    try:
        with time_limit(300):
            rdmol = reconstruct_from_generated_with_edges(mol_info)
    except TimeoutException as e:
        print('Timeout for reconstructing mol')
        raise MolReconsError()
    
    # convert rdmol to pdb
    tmpfile = tempfile.NamedTemporaryFile(suffix='.sdf')
    Chem.MolToMolFile(rdmol, tmpfile.name)
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("sdf", "pdb")
    obmol = ob.OBMol()
    obConversion.ReadFile(obmol, tmpfile.name)
    
    tmpfile = tempfile.NamedTemporaryFile(suffix='.pdb')
    obConversion.WriteFile(obmol, tmpfile.name)
    parser = PDBParser(PERMISSIVE=True, QUIET=True)
    pdb_struc = parser.get_structure('tmp', tmpfile.name)
    
    pdb_struc = sort_residues(pdb_struc)

    return pdb_struc, rdmol


def sort_residues(pdb_struc):
    
    new_pdb_struc = Structure.Structure('tmp')
    for model in pdb_struc.get_models():
        new_model = Model.Model(model.id)
        for chain in model.get_chains():
            new_chain = Chain.Chain(chain.id)
            # sort residues
            all_residues = list(chain.get_residues())
            all_residues.sort(key=lambda x: x.id[1])
            for residue in all_residues:
                new_chain.add(residue)
            new_model.add(new_chain)
        new_pdb_struc.add(new_model)
    return new_pdb_struc


class MolReconsError(Exception):
    pass


def reachable_r(a,b, seenbonds):
    '''Recursive helper.'''

    for nbr in ob.OBAtomAtomIter(a):
        bond = a.GetBond(nbr).GetIdx()
        if bond not in seenbonds:
            seenbonds.add(bond)
            if nbr == b:
                return True
            elif reachable_r(nbr,b,seenbonds):
                return True
    return False


def reachable(a,b):
    '''Return true if atom b is reachable from a without using the bond between them.'''
    if a.GetExplicitDegree() == 1 or b.GetExplicitDegree() == 1:
        return False #this is the _only_ bond for one atom
    #otherwise do recursive traversal
    seenbonds = set([a.GetBond(b).GetIdx()])
    return reachable_r(a,b,seenbonds)


def forms_small_angle(a,b,cutoff=45):
    '''Return true if bond between a and b is part of a small angle
    with a neighbor of a only.'''

    for nbr in ob.OBAtomAtomIter(a):
        if nbr != b:
            degrees = b.GetAngle(a,nbr)
            if degrees < cutoff:
                return True
    return False


def make_obmol(xyz, atomic_numbers):
    mol = ob.OBMol()
    mol.BeginModify()
    atoms = []
    for xyz,t in zip(xyz, atomic_numbers):
        x,y,z = xyz
        # ch = struct.channels[t]
        atom = mol.NewAtom()
        atom.SetAtomicNum(t)
        atom.SetVector(x,y,z)
        atoms.append(atom)
    return mol, atoms


def connect_the_dots(mol, atoms, maxbond=4):
    '''Custom implementation of ConnectTheDots.  This is similar to
    OpenBabel's version, but is more willing to make long bonds 
    (up to maxbond long) to keep the molecule connected.  It also 
    attempts to respect atom type information from struct.
    atoms and struct need to correspond in their order
    Assumes no hydrogens or existing bonds.
    '''
    pt = Chem.GetPeriodicTable()

    if len(atoms) == 0:
        return

    mol.BeginModify()

    #just going to to do n^2 comparisons, can worry about efficiency later
    coords = np.array([(a.GetX(),a.GetY(),a.GetZ()) for a in atoms])
    dists = squareform(pdist(coords))
    # types = [struct.channels[t].name for t in struct.c]

    for (i,a) in enumerate(atoms):
        for (j,b) in enumerate(atoms):
            if a == b:
                break
            if dists[i,j] < 0.01:  #reduce from 0.4
                continue #don't bond too close atoms
            if dists[i,j] < maxbond:
                flag = 0
                # if indicators[i][ATOM_FAMILIES_ID['Aromatic']] and indicators[j][ATOM_FAMILIES_ID['Aromatic']]:
                    # print('Aromatic', ATOM_FAMILIES_ID['Aromatic'], indicators[i])
                    # flag = ob.OB_AROMATIC_BOND
                # if 'Aromatic' in types[i] and 'Aromatic' in types[j]:
                #     flag = ob.OB_AROMATIC_BOND
                mol.AddBond(a.GetIdx(),b.GetIdx(),1,flag)

    atom_maxb = {}
    for (i,a) in enumerate(atoms):
        #set max valance to the smallest max allowed by openbabel or rdkit
        #since we want the molecule to be valid for both (rdkit is usually lower)
        maxb = ob.GetMaxBonds(a.GetAtomicNum())
        maxb = min(maxb,pt.GetDefaultValence(a.GetAtomicNum())) 

        if a.GetAtomicNum() == 16: # sulfone check
            if count_nbrs_of_elem(a, 8) >= 2:
                maxb = 6

        # if indicators[i][ATOM_FAMILIES_ID['Donor']]:
        #     maxb -= 1 #leave room for hydrogen
        # if 'Donor' in types[i]:
        #     maxb -= 1 #leave room for hydrogen
        atom_maxb[a.GetIdx()] = maxb
    
    #remove any impossible bonds between halogens
    for bond in ob.OBMolBondIter(mol):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if atom_maxb[a1.GetIdx()] == 1 and atom_maxb[a2.GetIdx()] == 1:
            mol.DeleteBond(bond)

    def get_bond_info(biter):
        '''Return bonds sorted by their distortion'''
        bonds = [b for b in biter]
        binfo = []
        for bond in bonds:
            bdist = bond.GetLength()
            #compute how far away from optimal we are
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            ideal = ob.GetCovalentRad(a1.GetAtomicNum()) + ob.GetCovalentRad(a2.GetAtomicNum()) 
            stretch = bdist-ideal
            binfo.append((stretch,bdist,bond))
        binfo.sort(reverse=True, key=lambda t: t[:2]) #most stretched bonds first
        return binfo

    #prioritize removing hypervalency causing bonds, do more valent 
    #constrained atoms first since their bonds introduce the most problems
    #with reachability (e.g. oxygen)
    # hypers = sorted([(atom_maxb[a.GetIdx()],a.GetExplicitValence() - atom_maxb[a.GetIdx()], a) for a in atoms],key=lambda aa: (aa[0],-aa[1]))
    # for mb,diff,a in hypers:
    #     if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
    #         continue
    #     binfo = get_bond_info(ob.OBAtomBondIter(a))
    #     for stretch,bdist,bond in binfo:
    #         #can we remove this bond without disconnecting the molecule?
    #         a1 = bond.GetBeginAtom()
    #         a2 = bond.GetEndAtom()

    #         #get right valence
    #         if a1.GetExplicitValence() > atom_maxb[a1.GetIdx()] or \
    #             a2.GetExplicitValence() > atom_maxb[a2.GetIdx()]:
    #             #don't fragment the molecule
    #             if not reachable(a1,a2):
    #                 continue
    #             mol.DeleteBond(bond)
    #             if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
    #                 break #let nbr atoms choose what bonds to throw out


    binfo = get_bond_info(ob.OBMolBondIter(mol))
    #now eliminate geometrically poor bonds
    for stretch,bdist,bond in binfo:
        #can we remove this bond without disconnecting the molecule?
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        #as long as we aren't disconnecting, let's remove things
        #that are excessively far away (0.45 from ConnectTheDots)
        #get bonds to be less than max allowed
        #also remove tight angles, because that is what ConnectTheDots does
        if stretch > 0.45 or forms_small_angle(a1,a2) or forms_small_angle(a2,a1):
            #don't fragment the molecule
            if not reachable(a1,a2):
                continue
            mol.DeleteBond(bond)

    mol.EndModify()

def add_context(data):
    data.ligand_context_pos = data.ligand_pos
    data.ligand_context_element = data.ligand_element
    data.ligand_context_bond_index = data.ligand_bond_index
    data.ligand_context_bond_type = data.ligand_bond_type
    return data

periodic_table = Chem.GetPeriodicTable()
def create_sdf_string(mol_info):
    xyz = mol_info['atom_pos']
    atomic_nums = mol_info['element']
    elements = [periodic_table.GetElementSymbol(int(z)) for z in atomic_nums]
    bond_index = mol_info['bond_index']  # might be symmetric
    bond_type = mol_info['bond_type']
    n_atoms = len(atomic_nums)
    
    if 'atom_pos_masked' in mol_info:
        pos_masked = mol_info['atom_pos_masked']
        xyz = np.concatenate([xyz, pos_masked], axis=0)
        elements = elements + ['He']*len(pos_masked)
        n_atoms = n_atoms + len(pos_masked)
    
    non_symmetric = bond_index[0] < bond_index[1]
    bond_index = bond_index[:, non_symmetric]
    bond_type = bond_type[non_symmetric]
    n_bonds = len(bond_type)
    
    header = '\n Created by Python create_sdf_string function\n\n'
    counts_line = f'{n_atoms:3}{n_bonds:3}  0  0  0  0  0  0  0  0999 V2000\n'
    atoms_block = ""
    for i, (x, y, z) in enumerate(xyz):
        atoms_block += f'{x:10.4f}{y:10.4f}{z:10.4f} {elements[i]:2}  ' + '  '.join(['0']*12) + '\n'
    bonds_block = ""
    for (i, j), b in zip(bond_index.T, bond_type):
        bonds_block += f'{i+1:3d}{j+1:3d}  {b}  0\n'
    sdf_string = header + counts_line + atoms_block + bonds_block + 'M  END\n'
    return sdf_string


def reconstruct_pos(mol_info, in_mol=None):
    # data_id = mol_info['data_id']
    # db = mol_info['db']
    if in_mol is None:
        mol = get_mol_from_data(mol_info, root_dir='data')
    else:
        mol = deepcopy(in_mol)
    pos = mol_info['atom_pos'].tolist()
    assert mol.GetNumAtoms() == len(pos), 'num atoms do not match gen pos'
    conf = mol.GetConformer(0)
    for i, xyz in enumerate(pos):
        conf.SetAtomPosition(i, Geometry.Point3D(*xyz))
    
    return mol


def reconstruct_from_generated_with_edges(mol_info, check_validity=True, add_edge=None, in_mol=None):
    # for conf or dock
    if mol_info['task'] in ['conf', 'dock']:
        try:
            return reconstruct_pos(mol_info, in_mol)
        except Exception as e:
            print('Failed to use reconstruct_pos:', e)
            pass
    
    xyz = mol_info['atom_pos'].tolist()
    atomic_nums = mol_info['element'].tolist()
    if 'bond_index' not in mol_info:
        if add_edge == 'openbabel':
            try:
                return reconstruct_from_generated(mol_info)
            except:
                raise MolReconsError()
        # elif add_edge == 'edm':
        #     bond_index, bond_type = predict_bonds(atomic_nums, np.array(xyz))
        else:
            raise ValueError('add_edge must be openbabel or edm')
    else:
        bond_index = mol_info['bond_index'].tolist()
        bond_type = mol_info['bond_type'].tolist()
    n_atoms = len(atomic_nums)

    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(n_atoms)
    
    # add atoms and coordinates
    for i, atom in enumerate(atomic_nums):
        rd_atom = Chem.Atom(atom)
        rd_mol.AddAtom(rd_atom)
        rd_coords = Geometry.Point3D(*xyz[i])
        rd_conf.SetAtomPosition(i, rd_coords)
    rd_mol.AddConformer(rd_conf)
    
    # add bonds
    for i, type_this in enumerate(bond_type):
        node_i, node_j = bond_index[0][i], bond_index[1][i]
        if node_i < node_j:
            if type_this == 1:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.SINGLE)
            elif type_this == 2:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.DOUBLE)
            elif type_this == 3:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.TRIPLE)
            elif type_this == 4:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.AROMATIC)
            else:
                raise Exception('unknown bond order {}'.format(type_this))
    
    
    mol = rd_mol.GetMol()
    if check_validity:
        try:
            Chem.SanitizeMol(deepcopy(mol))
            fixed = True
            Chem.SanitizeMol(mol)
        except Exception as e:
            fixed = False
        # TODO: ok but not good solution. https://github.com/rdkit/rdkit/wiki/FrequentlyAskedQuestions
        if not fixed:
            try:
                Chem.Kekulize(deepcopy(mol))
            except Chem.rdchem.KekulizeException as e:
                err = e
                if 'Unkekulized' in err.args[0]:
                    try:
                        with time_limit(300):
                            mol, fixed = fix_aromatic(mol)
                    except TimeoutException as e:
                        print('Timeout for reconstructing mol')
                        raise MolReconsError()

        # valence error for N 
        if not fixed:
            try:
                with time_limit(300):
                    mol, fixed = fix_valence(mol)
            except TimeoutException as e:
                print('Timeout for reconstructing mol')
                raise MolReconsError()
            
        # print('s2')
        if not fixed:
            try:
                with time_limit(300):
                    mol, fixed = fix_aromatic(mol, True)
            except TimeoutException as e:
                print('Timeout for reconstructing mol')
                raise MolReconsError()
            
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            raise MolReconsError()
            # return None
        
    # check valid
    # rd_mol_check = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    # if (rd_mol_check is None) and check_validity:
    #     raise MolReconsError()
    return mol


def get_ring_sys(mol):
    all_rings = Chem.GetSymmSSSR(mol)
    if len(all_rings) == 0:
        ring_sys_list = []
    else:
        ring_sys_list = [all_rings[0]]
        for ring in all_rings[1:]:
            form_prev = False
            for prev_ring in ring_sys_list:
                if set(ring).intersection(set(prev_ring)):
                    prev_ring.extend(ring)
                    form_prev = True
                    break
            if not form_prev:
                ring_sys_list.append(ring)
    ring_sys_list = [list(set(x)) for x in ring_sys_list]
    return ring_sys_list

def fix_valence(mol):
    mol = deepcopy(mol)
    fixed = False
    cnt_loop = 0
    while True:
        try:
            Chem.SanitizeMol(deepcopy(mol))
            fixed = True
            Chem.SanitizeMol(mol)
            break
        except Chem.rdchem.AtomValenceException as e:
            err = e
        except Exception as e:
            return mol, False # from HERE: rerun sample
        cnt_loop += 1
        if cnt_loop > 100:
            break
        N4_valence = re.compile(u"Explicit valence for atom # ([0-9]{1,}) N, 4, is greater than permitted")
        index = N4_valence.findall(err.args[0])
        if len(index) > 0:
            mol.GetAtomWithIdx(int(index[0])).SetFormalCharge(1)
    return mol, fixed


def get_all_subsets(ring_list):
    all_sub_list = []
    for n_sub in range(len(ring_list)+1):
        all_sub_list.extend(itertools.combinations(ring_list, n_sub))
    return all_sub_list

def fix_aromatic(mol, strict=False):
    mol_orig = mol
    atomatic_list = [a.GetIdx() for a in mol.GetAromaticAtoms()]
    N_ring_list = []
    S_ring_list = []
    for ring_sys in get_ring_sys(mol):
        if set(ring_sys).intersection(set(atomatic_list)):
            idx_N = [atom for atom in ring_sys if mol.GetAtomWithIdx(atom).GetSymbol() == 'N']
            if len(idx_N) > 0:
                idx_N.append(-1) # -1 for not add to this loop
                N_ring_list.append(idx_N)
            idx_S = [atom for atom in ring_sys if mol.GetAtomWithIdx(atom).GetSymbol() == 'S']
            if len(idx_S) > 0:
                idx_S.append(-1) # -1 for not add to this loop
                S_ring_list.append(idx_S)
    # enumerate S
    fixed = False
    if strict:
        S_ring_list = [s for ring in S_ring_list for s in ring if s != -1]
        permutation = get_all_subsets(S_ring_list)
    else:
        permutation = list(itertools.product(*S_ring_list))
    for perm in permutation:
        mol = deepcopy(mol_orig)
        perm = [x for x in perm if x != -1]
        for idx in perm:
            mol.GetAtomWithIdx(idx).SetFormalCharge(1)
        try:
            if strict:
                mol, fixed = fix_valence(mol)
            Chem.SanitizeMol(deepcopy(mol))
            fixed = True
            Chem.SanitizeMol(mol)
            break
        except:
            continue
    # enumerate N
    if not fixed:
        if strict:
            N_ring_list = [s for ring in N_ring_list for s in ring if s != -1]
            permutation = get_all_subsets(N_ring_list)
        else:
            permutation = list(itertools.product(*N_ring_list))
        for perm in permutation:  # each ring select one atom
            perm = [x for x in perm if x != -1]
            # print(perm)
            actions = itertools.product([0, 1], repeat=len(perm))
            for action in actions: # add H or charge
                mol = deepcopy(mol_orig)
                for idx, act_atom in zip(perm, action):
                    if act_atom == 0:
                        mol.GetAtomWithIdx(idx).SetNumExplicitHs(1)
                    else:
                        mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                try:
                    if strict:
                        mol, fixed = fix_valence(mol)
                    Chem.SanitizeMol(deepcopy(mol))
                    fixed = True
                    Chem.SanitizeMol(mol)
                    break
                except:
                    continue
            if fixed:
                break
    return mol, fixed


########## the following for reconstruct_from_generated (auto add bond) ##########

def reconstruct_from_generated(mol_info):
    xyz = mol_info['atom_pos'].tolist()
    atomic_nums = mol_info['element'].tolist()
    # xyz = data.ligand_context_pos.clone().cpu().tolist()
    # atomic_nums = data.ligand_context_element.clone().cpu().tolist()
    # indicators = data.ligand_context_feature_full[:, -len(ATOM_FAMILIES_ID):].clone().cpu().bool().tolist()

    mol, atoms = make_obmol(xyz, atomic_nums)
    fixup(atoms, mol, )

    connect_the_dots(mol, atoms, 2)
    fixup(atoms, mol, )
    mol.EndModify()

    fixup(atoms, mol, )

    mol.AddPolarHydrogens()
    mol.PerceiveBondOrders()
    fixup(atoms, mol, )

    for (i,a) in enumerate(atoms):
        ob.OBAtomAssignTypicalImplicitHydrogens(a)
    fixup(atoms, mol, )

    mol.AddHydrogens()
    fixup(atoms, mol, )

    #make rings all aromatic if majority of carbons are aromatic
    for ring in ob.OBMolRingIter(mol):
        if 5 <= ring.Size() <= 6:
            carbon_cnt = 0
            aromatic_ccnt = 0
            for ai in ring._path:
                a = mol.GetAtom(ai)
                if a.GetAtomicNum() == 6:
                    carbon_cnt += 1
                    if a.IsAromatic():
                        aromatic_ccnt += 1
            if aromatic_ccnt >= carbon_cnt/2 and aromatic_ccnt != ring.Size():
                #set all ring atoms to be aromatic
                for ai in ring._path:
                    a = mol.GetAtom(ai)
                    a.SetAromatic(True)

    #bonds must be marked aromatic for smiles to match
    for bond in ob.OBMolBondIter(mol):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if a1.IsAromatic() and a2.IsAromatic():
            bond.SetAromatic(True)
            
    mol.PerceiveBondOrders()

    rd_mol = convert_ob_mol_to_rd_mol(mol)

    # Post-processing
    rd_mol = postprocess_rd_mol_1(rd_mol)
    rd_mol = postprocess_rd_mol_2(rd_mol)

    return rd_mol


def fixup(atoms, mol, ):
    '''Set atom properties to match channel.  Keep doing this
    to beat openbabel over the head with what we want to happen.'''

    mol.SetAromaticPerceived(True)  #avoid perception
    for i, atom in enumerate(atoms):
        # ch = struct.channels[t]
        # ind = indicators[i]

        # if ind[ATOM_FAMILIES_ID['Aromatic']]:
        #     atom.SetAromatic(True)
        #     atom.SetHyb(2)

        # if ind[ATOM_FAMILIES_ID['Donor']]:
        #     if atom.GetExplicitDegree() == atom.GetHvyDegree():
        #         if atom.GetHvyDegree() == 1 and atom.GetAtomicNum() == 7:
        #             atom.SetImplicitHCount(2)
        #         else:
        #             atom.SetImplicitHCount(1) 


        # elif ind[ATOM_FAMILIES_ID['Acceptor']]: # NOT AcceptorDonor because of else
        #     atom.SetImplicitHCount(0)   

        if (atom.GetAtomicNum() in (7, 8)) and atom.IsInRing():     # Nitrogen, Oxygen
            #this is a little iffy, ommitting until there is more evidence it is a net positive
            #we don't have aromatic types for nitrogen, but if it
            #is in a ring with aromatic carbon mark it aromatic as well
            acnt = 0
            for nbr in ob.OBAtomAtomIter(atom):
                if nbr.IsAromatic():
                    acnt += 1
            if acnt > 1:
                atom.SetAromatic(True)


def convert_ob_mol_to_rd_mol(ob_mol,struct=None):
    '''Convert OBMol to RDKit mol, fixing up issues'''
    ob_mol.DeleteHydrogens()
    n_atoms = ob_mol.NumAtoms()
    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(n_atoms)

    for ob_atom in ob.OBMolAtomIter(ob_mol):
        rd_atom = Chem.Atom(ob_atom.GetAtomicNum())
        #TODO copy format charge
        if ob_atom.IsAromatic() and ob_atom.IsInRing() and ob_atom.MemberOfRingSize() <= 6:
            #don't commit to being aromatic unless rdkit will be okay with the ring status
            #(this can happen if the atoms aren't fit well enough)
            rd_atom.SetIsAromatic(True)
        i = rd_mol.AddAtom(rd_atom)
        ob_coords = ob_atom.GetVector()
        x = ob_coords.GetX()
        y = ob_coords.GetY()
        z = ob_coords.GetZ()
        rd_coords = Geometry.Point3D(x, y, z)
        rd_conf.SetAtomPosition(i, rd_coords)

    rd_mol.AddConformer(rd_conf)

    for ob_bond in ob.OBMolBondIter(ob_mol):
        i = ob_bond.GetBeginAtomIdx()-1
        j = ob_bond.GetEndAtomIdx()-1
        bond_order = ob_bond.GetBondOrder()
        if bond_order == 1:
            rd_mol.AddBond(i, j, Chem.BondType.SINGLE)
        elif bond_order == 2:
            rd_mol.AddBond(i, j, Chem.BondType.DOUBLE)
        elif bond_order == 3:
            rd_mol.AddBond(i, j, Chem.BondType.TRIPLE)
        else:
            raise Exception('unknown bond order {}'.format(bond_order))

        if ob_bond.IsAromatic():
            bond = rd_mol.GetBondBetweenAtoms (i,j)
            bond.SetIsAromatic(True)

    rd_mol = Chem.RemoveHs(rd_mol, sanitize=False)

    pt = Chem.GetPeriodicTable()
    #if double/triple bonds are connected to hypervalent atoms, decrement the order

    positions = rd_mol.GetConformer().GetPositions()
    nonsingles = []
    for bond in rd_mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE or bond.GetBondType() == Chem.BondType.TRIPLE:
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            dist = np.linalg.norm(positions[i]-positions[j])
            nonsingles.append((dist,bond))
    nonsingles.sort(reverse=True, key=lambda t: t[0])

    for (d,bond) in nonsingles:
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        if calc_valence(a1) > pt.GetDefaultValence(a1.GetAtomicNum()) or \
           calc_valence(a2) > pt.GetDefaultValence(a2.GetAtomicNum()):
            btype = Chem.BondType.SINGLE
            if bond.GetBondType() == Chem.BondType.TRIPLE:
                btype = Chem.BondType.DOUBLE
            bond.SetBondType(btype)

    for atom in rd_mol.GetAtoms():
        #set nitrogens with 4 neighbors to have a charge
        if atom.GetAtomicNum() == 7 and atom.GetDegree() == 4:
            atom.SetFormalCharge(1)

    rd_mol = Chem.AddHs(rd_mol,addCoords=True)

    positions = rd_mol.GetConformer().GetPositions()
    center = np.mean(positions[np.all(np.isfinite(positions),axis=1)],axis=0)
    for atom in rd_mol.GetAtoms():
        i = atom.GetIdx()
        pos = positions[i]
        if not np.all(np.isfinite(pos)):
            #hydrogens on C fragment get set to nan (shouldn't, but they do)
            rd_mol.GetConformer().SetAtomPosition(i,center)

    try:
        Chem.SanitizeMol(rd_mol,Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE)
    except:
        raise MolReconsError()
    # try:
    #     Chem.SanitizeMol(rd_mol,Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE)
    # except: # mtr22 - don't assume mols will pass this
    #     pass
    #     # dkoes - but we want to make failures as rare as possible and should debug them
    #     m = pybel.Molecule(ob_mol)
    #     i = np.random.randint(1000000)
    #     outname = 'bad%d.sdf'%i
    #     print("WRITING",outname)
    #     m.write('sdf',outname,overwrite=True)
    #     pickle.dump(struct,open('bad%d.pkl'%i,'wb'))

    #but at some point stop trying to enforce our aromaticity -
    #openbabel and rdkit have different aromaticity models so they
    #won't always agree.  Remove any aromatic bonds to non-aromatic atoms
    for bond in rd_mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if bond.GetIsAromatic():
            if not a1.GetIsAromatic() or not a2.GetIsAromatic():
                bond.SetIsAromatic(False)
        elif a1.GetIsAromatic() and a2.GetIsAromatic():
            bond.SetIsAromatic(True)

    return rd_mol

UPGRADE_BOND_ORDER = {Chem.BondType.SINGLE:Chem.BondType.DOUBLE, Chem.BondType.DOUBLE:Chem.BondType.TRIPLE}

def postprocess_rd_mol_1(rdmol):

    rdmol = Chem.RemoveHs(rdmol)

    # Construct bond nbh list
    nbh_list = {}
    for bond in rdmol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx() 
        if begin not in nbh_list: nbh_list[begin] = [end]
        else: nbh_list[begin].append(end)
            
        if end not in nbh_list: nbh_list[end] = [begin]
        else: nbh_list[end].append(begin)

    # Fix missing bond-order
    for atom in rdmol.GetAtoms():
        idx = atom.GetIdx()
        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0:
            for j in nbh_list[idx]:
                if j <= idx: continue
                nb_atom = rdmol.GetAtomWithIdx(j)
                nb_radical = nb_atom.GetNumRadicalElectrons()
                if nb_radical > 0:
                    bond = rdmol.GetBondBetweenAtoms(idx, j)
                    bond.SetBondType(UPGRADE_BOND_ORDER[bond.GetBondType()])
                    nb_atom.SetNumRadicalElectrons(nb_radical - 1)
                    num_radical -= 1
            atom.SetNumRadicalElectrons(num_radical)

        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0:
            atom.SetNumRadicalElectrons(0)
            num_hs = atom.GetNumExplicitHs()
            atom.SetNumExplicitHs(num_hs + num_radical)
            
    return rdmol


def postprocess_rd_mol_2(rdmol):
    rdmol_edit = Chem.RWMol(rdmol)

    ring_info = rdmol.GetRingInfo()
    ring_info.AtomRings()
    rings = [set(r) for r in ring_info.AtomRings()]
    for i, ring_a in enumerate(rings):
        if len(ring_a) == 3:
            non_carbon = []
            atom_by_symb = {}
            for atom_idx in ring_a:
                symb = rdmol.GetAtomWithIdx(atom_idx).GetSymbol()
                if symb != 'C':
                    non_carbon.append(atom_idx)
                if symb not in atom_by_symb:
                    atom_by_symb[symb] = [atom_idx]
                else:
                    atom_by_symb[symb].append(atom_idx)
            if len(non_carbon) == 2:
                rdmol_edit.RemoveBond(*non_carbon)
            if 'O' in atom_by_symb and len(atom_by_symb['O']) == 2:
                rdmol_edit.RemoveBond(*atom_by_symb['O'])
                rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).SetNumExplicitHs(
                    rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).GetNumExplicitHs() + 1
                )
                rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).SetNumExplicitHs(
                    rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).GetNumExplicitHs() + 1
                )
    rdmol = rdmol_edit.GetMol()

    for atom in rdmol.GetAtoms():
        if atom.GetFormalCharge() > 0:
            atom.SetFormalCharge(0)

    return rdmol
    

def calc_valence(rdatom):
    '''Can call GetExplicitValence before sanitize, but need to
    know this to fix up the molecule to prevent sanitization failures'''
    cnt = 0.0
    for bond in rdatom.GetBonds():
        cnt += bond.GetBondTypeAsDouble()
    return cnt


def count_nbrs_of_elem(atom, atomic_num):
    '''
    Count the number of neighbors atoms
    of atom with the given atomic_num.
    '''
    count = 0
    for nbr in ob.OBAtomAtomIter(atom):
        if nbr.GetAtomicNum() == atomic_num:
            count += 1
    return count

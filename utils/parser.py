import os
import numpy as np
from rdkit import Chem
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from utils.fragment import find_rotatable_bond_mat
# from rdkit.Chem.rdchem import BondType
# from rdkit.Chem import ChemicalFeatures
# from rdkit import RDConfig

# from utils.mol2frag import mol2frag

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable', 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
# BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
# BOND_NAMES = {i: t for i, t in enumerate(BondType.names.keys())}


class PDBLigand(object):

    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    AA_NAME_NUMBER = {
        k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
    }

    BACKBONE_NAMES = ["CA", "C", "N", "O"]

    def __init__(self, data, mode='auto', removeHs=True):
        super().__init__()
        # # Read PDB file
        self.removeHs = removeHs
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if (line[0:6].strip() == 'ATOM') or (line[0:6].strip() == 'HETATM'):
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': line[0:6].strip(),  # ATOM or HETATM
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break   # Some PDBs have more than 1 model.

    def _parse(self):
        # Process atoms
        residues_tmp = {}
        for atom in self._enum_formatted_atom_lines():
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            if self.removeHs and atom['element_symb'] == 'H':
                continue
            
            nonstd = (atom['res_name'] not in self.AA_NAME_NUMBER)
            
            self.atoms.append(atom)
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            next_ptr = len(self.element)
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
            self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])

            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in residues_tmp:
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                    'chain_res_id': chain_res_id
                }
            else:
                assert residues_tmp[chain_res_id]['name'] == atom['res_name']
                assert residues_tmp[chain_res_id]['chain'] == atom['chain']
                residues_tmp[chain_res_id]['atoms'].append(next_ptr)

        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in self.BACKBONE_NAMES:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            residue['center_of_mass'] = sum_pos / sum_mass
        
        # Process backbone atoms of residues
        for residue in self.residues:
            self.amino_acid.append(self.AA_NAME_NUMBER[residue['name']])
            self.center_of_mass.append(residue['center_of_mass'])
            for name in self.BACKBONE_NAMES:
                pos_key = 'pos_%s' % name   # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])

    def to_dict_atom(self):
        return {
            'element': np.array(self.element, dtype=np.int64),
            'molecule_name': self.title,
            'pos': np.array(self.pos, dtype=np.float32),
            'is_backbone': np.array(self.is_backbone, dtype=bool),
            'atom_name': self.atom_name,
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)
        }

    def to_dict_residue(self):
        return {
            'amino_acid': np.array(self.amino_acid, dtype=np.int64),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'pos_C': np.array(self.pos_C, dtype=np.float32),
            'pos_N': np.array(self.pos_N, dtype=np.float32),
            'pos_O': np.array(self.pos_O, dtype=np.float32),
        }

    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            print(residue[criterion], distance)
            if distance < radius:
                selected.append(residue)
        return selected

    def query_residues_ligand(self, mol, radius, criterion='center_of_mass'):
        selected = []
        sel_idx = set()
        # The time-complexity is O(mn).
        mol_pos = mol.GetConformer().GetPositions()
        for center in mol_pos:
            for i, residue in enumerate(self.residues):
                distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        return selected
    
    def get_pocket_info(self, selected):
        num_res = len(selected)
        num_atoms = sum([len(residue['atoms']) for residue in selected])
        res_ids = [residue['chain_res_id'] for residue in selected]
        cover_chains = sorted(list(set([residue['chain'] for residue in selected])))
        pocket_info = {
            'num_res': num_res,
            'num_atoms': num_atoms,
            'res_ids': ';'.join(res_ids),
            'num_chains': len(cover_chains),
            'cover_chain_ids': ';'.join(cover_chains),
        }
        return pocket_info

    def get_chain_seqs(self, chains):
        if isinstance(chains, str):
            chains = [chains]
        seqs = []
        for chain in chains:
            seq = []
            for residue in self.residues:
                if residue['chain'] == chain:
                    seq.append(self.AA_NAME_SYM[residue['name']])
            seqs.append(''.join(seq))
            
        return ';'.join(seqs)


    def residues_to_pdb_block(self, residues, name='POCKET'):
        block =  "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        for residue in residues:
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        return block
    
    
def parse_mol_with_confs(mol, smiles=None, confs=None):
    if smiles is not None: # check smiles
        if smiles != Chem.MolToSmiles(mol, isomericSmiles=False):
            return None
    pos_all_confs = []
    i_conf_list = []

    data = parse_3d_mol(mol, smiles=smiles, not_pos=True)
    if data is None:
        return None
    
    # get all confs
    if confs is None:
        for i_conf in range(mol.GetNumConformers()):
            pos = mol.GetConformer(i_conf).GetPositions()
            pos_all_confs.append(pos)
            i_conf_list.append(i_conf)
            # check bond length
            # bond_lengths = np.linalg.norm(pos[data['bond_index'][0]] - pos[data['bond_index'][1]], ord=2, axis=-1)
            # if (bond_lengths > 3.5).any():
            #     print('Skipping conformer with too long bond length: %s' % bond_lengths.max())
            #     return None
    else:
        pos_all_confs = np.array(confs)[:, :len(data['element']), :]
        i_conf_list = list(range(len(confs)))

    return {
        'element': np.array(data['element']),
        'bond_index': np.array(data['bond_index']),
        'bond_type': np.array(data['bond_type']),
        'pos_all_confs': np.array(pos_all_confs, dtype=np.float32),
        'num_atoms': data['num_atoms'],
        'num_bonds': data['num_bonds'],
        'i_conf_list': i_conf_list,
        'num_confs': len(i_conf_list),
    }

class PDBProtein(object):

    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    AA_NAME_NUMBER = {
        k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
    }

    BACKBONE_NAMES = ["CA", "C", "N", "O"]

    def __init__(self, data, mode='auto', removeHs=True):
        super().__init__()
        self.removeHs = removeHs
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line[0:6].strip() == 'ATOM':
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': 'ATOM',
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break   # Some PDBs have more than 1 model.

    def _parse(self):
        # Process atoms
        residues_tmp = {}
        for atom in self._enum_formatted_atom_lines():
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            if self.removeHs and atom['element_symb'] == 'H':
                continue
            if atom['res_name'] not in self.AA_NAME_NUMBER:
                continue
            self.atoms.append(atom)
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            next_ptr = len(self.element)
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
            self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])

            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in residues_tmp:
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                    'chain_res_id': chain_res_id
                }
            else:
                assert residues_tmp[chain_res_id]['name'] == atom['res_name']
                assert residues_tmp[chain_res_id]['chain'] == atom['chain']
                residues_tmp[chain_res_id]['atoms'].append(next_ptr)

        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in self.BACKBONE_NAMES:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            residue['center_of_mass'] = sum_pos / sum_mass
        
        # Process backbone atoms of residues
        for residue in self.residues:
            self.amino_acid.append(self.AA_NAME_NUMBER[residue['name']])
            self.center_of_mass.append(residue['center_of_mass'])
            for name in self.BACKBONE_NAMES:
                pos_key = 'pos_%s' % name   # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])

    def to_dict_atom(self):
        return {
            'element': np.array(self.element, dtype=np.int64),
            'molecule_name': self.title,
            'pos': np.array(self.pos, dtype=np.float32),
            'is_backbone': np.array(self.is_backbone, dtype=bool),
            'atom_name': self.atom_name,
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)
        }

    def to_dict_residue(self):
        return {
            'amino_acid': np.array(self.amino_acid, dtype=np.int64),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'pos_C': np.array(self.pos_C, dtype=np.float32),
            'pos_N': np.array(self.pos_N, dtype=np.float32),
            'pos_O': np.array(self.pos_O, dtype=np.float32),
        }

    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            print(residue[criterion], distance)
            if distance < radius:
                selected.append(residue)
        return selected

    def query_residues_ligand(self, mol, radius, criterion='center_of_mass'):
        selected = []
        sel_idx = set()
        # The time-complexity is O(mn).
        mol_pos = mol.GetConformer().GetPositions()
        for center in mol_pos:
            for i, residue in enumerate(self.residues):
                if criterion == 'min':
                    res_pos = np.array([self.pos[atom] for atom in residue['atoms']])
                    distance_all = np.linalg.norm(res_pos - center, ord=2, axis=-1)
                    distance = distance_all.min()
                else:
                    distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        return selected
    
    def get_pocket_info(self, selected):
        num_res = len(selected)
        num_atoms = sum([len(residue['atoms']) for residue in selected])
        res_ids = [residue['chain_res_id'] for residue in selected]
        cover_chains = sorted(list(set([residue['chain'] for residue in selected])))
        pocket_info = {
            'num_res': num_res,
            'num_atoms': num_atoms,
            'res_ids': ';'.join(res_ids),
            'num_chains': len(cover_chains),
            'cover_chain_ids': ';'.join(cover_chains),
        }
        return pocket_info

    def get_chain_seqs(self, chains):
        if isinstance(chains, str):
            chains = [chains]
        seqs = []
        for chain in chains:
            seq = []
            for residue in self.residues:
                if residue['chain'] == chain:
                    seq.append(self.AA_NAME_SYM[residue['name']])
            seqs.append(''.join(seq))
            
        return ';'.join(seqs)


    def residues_to_pdb_block(self, residues, name='POCKET'):
        block =  "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        for residue in residues:
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        return block
    
    
def parse_pdb_peptide(pdb_path):
    parser = PDBParser(PERMISSIVE=0)
    structure = parser.get_structure('structure', pdb_path)
    
    pep_feature_dict = {'pos': [], 'atom_name': [], 'is_backbone': [], 'res_id': [], 'atom_to_aa_type': []}
    for atom in structure.get_atoms():
        element = atom.element
        if element == 'H':
            continue
        element = element if len(element) == 1 else element[0] + element[1].lower()
        pep_feature_dict['pos'].append(atom.get_coord())
        pep_feature_dict['atom_name'].append(atom.get_name())
        res = atom.get_parent()
        pep_feature_dict['res_id'].append(res.get_id()[1])
        resname = res.get_resname()
        aa_type = PDBProtein.AA_NAME_NUMBER[resname] if resname in PDBProtein.AA_NAME_NUMBER else len(PDBProtein.AA_NAME_NUMBER)
        pep_feature_dict['atom_to_aa_type'].append(aa_type)
        pep_feature_dict['is_backbone'].append(
            (atom.get_name() in PDBProtein.BACKBONE_NAMES) and (aa_type < len(PDBProtein.AA_NAME_NUMBER)))

    unique_res_id, res_index = np.unique(pep_feature_dict['res_id'], return_inverse=True)
    pep_feature_dict['res_index'] = res_index
    assert (np.diff(unique_res_id) == 1).all(), 'residue id is not continuous'

    pep_feature_dict = {k: np.array(v) for k, v in pep_feature_dict.items()}
    
    pep_feature_dict['seq'] = ''.join([seq1(residue.get_resname()) or 'X' for residue in structure.get_residues()])
    pep_feature_dict['pep_len'] = len(pep_feature_dict['seq'])
    pep_feature_dict['pep_path'] = pdb_path
    pep_feature_dict['atom_name'] = list(pep_feature_dict['atom_name'])
    
    if 'X' in pep_feature_dict['seq']:
        print('Warning: X in peptide sequence:', pdb_path)
    
    return pep_feature_dict


def parse_mol_with_confs(mol, smiles=None, confs=None):
    if smiles is not None: # check smiles
        if smiles != Chem.MolToSmiles(mol, isomericSmiles=False):
            return None
    pos_all_confs = []
    i_conf_list = []

    data = parse_3d_mol(mol, smiles=smiles, not_pos=True)
    if data is None:
        return None
    
    # get all confs
    if confs is None:
        for i_conf in range(mol.GetNumConformers()):
            pos = mol.GetConformer(i_conf).GetPositions()
            pos_all_confs.append(pos)
            i_conf_list.append(i_conf)
            # check bond length
            # bond_lengths = np.linalg.norm(pos[data['bond_index'][0]] - pos[data['bond_index'][1]], ord=2, axis=-1)
            # if (bond_lengths > 3.5).any():
            #     print('Skipping conformer with too long bond length: %s' % bond_lengths.max())
            #     return None
    else:
        pos_all_confs = np.array(confs)[:, :len(data['element']), :]
        i_conf_list = list(range(len(confs)))

    return {
        'element': np.array(data['element']),
        'bond_index': np.array(data['bond_index']),
        'bond_type': np.array(data['bond_type']),
        'pos_all_confs': np.array(pos_all_confs, dtype=np.float32),
        'num_atoms': data['num_atoms'],
        'num_bonds': data['num_bonds'],
        'i_conf_list': i_conf_list,
        'num_confs': len(i_conf_list),
    }


def parse_conf_list(conf_list, smiles=None):
    # data_list = [parse_drug3d_mol(conf) for conf in conf_list]
    
    element = []
    bond_index = []
    bond_type = []
    pos_all_confs = []
    i_conf_list = []
    num_atoms = 0
    num_bonds = 0  # NOTE: the num of bonds is not symtric
    for i_conf,  conf in enumerate(conf_list):
        data = parse_3d_mol(conf, smiles=smiles)
        if data is None:
            continue
        # check element
        if len(element) == 0:
            element = data['element']
            num_atoms = data['num_atoms']
        else:
            if data['num_atoms'] != num_atoms:
                print('Skipping conformer with different number of atoms')
                continue
            if not np.all(element == data['element']):
                print('Skipping conformer with different element order')
                continue
        # check bond
        if len(bond_index) == 0:
            bond_index = data['bond_index']
            bond_type = data['bond_type']
            num_bonds = data['num_bonds']
        else:
            if data['num_bonds'] != num_bonds:
                print('Skipping conformer with different number of bonds')
                continue
            if not np.all(bond_index == data['bond_index']):
                print('Skipping conformer with different bond index')
                continue
            if not np.all(bond_type == data['bond_type']):
                print('Skipping conformer with different bond type')
                continue
        pos_all_confs.append(data['pos'])
        i_conf_list.append(i_conf)

    return {
        'element': np.array(element),
        'bond_index': np.array(bond_index),
        'bond_type': np.array(bond_type),
        # 'bond_rotatable': np.array(data['bond_rotatable']),
        'pos_all_confs': np.array(pos_all_confs, dtype=np.float32),
        'num_atoms': num_atoms,
        'num_bonds': num_bonds,
        'i_conf_list': i_conf_list,
        'num_confs': len(i_conf_list),
    }


def parse_3d_mol(mol, smiles=None, not_pos=False):
    if smiles is not None: # check smiles
        if smiles != Chem.MolToSmiles(mol, isomericSmiles=False):
            return None
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    if not not_pos:
        conf = mol.GetConformer()
    ele_list = []
    pos_list = []
    for i, atom in enumerate(mol.GetAtoms()):
        ele = atom.GetAtomicNum()
        if not not_pos:
            pos = conf.GetAtomPosition(i)
            pos_list.append(list(pos))
        ele_list.append(ele)
    
    row, col = [], []
    bond_type = []
    for bond in mol.GetBonds():
        b_type = int(bond.GetBondType())
        assert b_type in [1, 2, 3, 12], 'Bond can only be 1,2,3,12 bond'
        b_type = b_type if b_type != 12 else 4
        b_index = [
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx()
        ]
        bond_type += 2*[b_type]
        row += [b_index[0], b_index[1]]
        col += [b_index[1], b_index[0]]
    
    bond_type = np.array(bond_type, dtype=np.int64)
    bond_index = np.array([row, col],dtype=np.int64)

    perm = (bond_index[0] * num_atoms + bond_index[1]).argsort()
    bond_index = bond_index[:, perm]
    bond_type = bond_type[perm]
    
    # # is rotable bond 
    # rot_mat = find_rotatable_bond_mat(mol)
    # bond_rotatable = rot_mat[bond_index[0], bond_index[1]]

    data = {
        'element': np.array(ele_list, dtype=np.int64),
        'pos': np.array(pos_list, dtype=np.float32),
        'bond_index': np.array(bond_index, dtype=np.int64),
        'bond_type': np.array(bond_type, dtype=np.int64),
        # 'bond_rotatable': np.array(bond_rotatable, dtype=np.int64),
        'num_atoms': num_atoms,
        'num_bonds': num_bonds,
    }
    return data
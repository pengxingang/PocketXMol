from rdkit import Chem
import numpy as np
from tqdm import tqdm
from .scoring_func import *
from multiprocessing import Pool
from functools import partial
from itertools import combinations
from collections import Counter
from rdkit.Chem import Fragments as frag_func


def get_drug_chem(mol):
    qed_score = qed(mol)
    sa_score = compute_sa_score(mol)
    logp_score = Crippen.MolLogP(mol)
    lipinski = obey_lipinski(mol)
    return {
        'qed': qed_score,
        'sa': sa_score,
        'logp': logp_score,
        'lipinski': lipinski
    }
    
def get_count_prop(mol):
    n_atoms, n_bonds, n_rings, weight = get_basic(mol)
    n_rotatable = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
    hacc_score = Lipinski.NumHAcceptors(mol)
    hdon_score = Lipinski.NumHDonors(mol)
    return {
        'n_atoms': n_atoms,
        'n_bonds': n_bonds,
        'n_rings': n_rings,
        'n_rotatable': n_rotatable,
        'weight': weight,
        'n_hacc': hacc_score,
        'n_hdon': hdon_score
    }
    
    
def get_global_3d(mol):
    try:
        rmsd_list = get_rdkit_rmsd(mol)
    except:
        return {}
    return {
        'rmsd_max': rmsd_list[0],
        'rmsd_min': rmsd_list[1],
        'rmsd_median': rmsd_list[2]
    }
    

def get_frags_counts(mol):
    # get atom element counts
    ele_list = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl']
    count_ele_dict = {ele: 0 for ele in ele_list}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ele_list:
            count_ele_dict[atom.GetSymbol()] += 1
    count_ele_dict = {'cnt_ele'+k: v for k, v in count_ele_dict.items()}

    # get bond types counts
    bb_list = [1, 2, 3, 4]
    count_bb_dict = {bb: 0 for bb in bb_list}
    for bond in mol.GetBonds():
        b_type = int(bond.GetBondType())
        b_type = 4 if b_type == 12 else b_type
        if b_type in bb_list:
            count_bb_dict[b_type] += 1
    count_bb_dict = {'cnt_bond'+str(k): v for k, v in count_bb_dict.items()}
    
    # get ring counts
    ring_list = [3, 4, 5, 6, 7, 8, 9]
    ring_count_dict = {ring: 0 for ring in ring_list}
    all_ssr = Chem.GetSymmSSSR(mol)
    for ssr in all_ssr:
        size = len(ssr)
        if size < ring_list[-1]:
            ring_count_dict[len(ssr)] += 1
        elif size >= ring_list[-1]:
            ring_count_dict[ring_list[-1]] += 1
    ring_count_dict = {'cnt_ring'+str(k): v for k, v in ring_count_dict.items()}
    
    return {**count_ele_dict, **count_bb_dict, **ring_count_dict}
        

def get_groups_counts(mol):
    # group functions
    func_list = ['fr_Ar_N', 'fr_C_O', 'fr_C_O_noCOO', 'fr_NH0', 'fr_NH1', 'fr_alkyl_halide', 'fr_allylic_oxid',
        'fr_amide', 'fr_aniline', 'fr_aryl_methyl', 'fr_benzene', 'fr_bicyclic', 'fr_ester', 'fr_ether',
        'fr_halogen', 'fr_methoxy', 'fr_para_hydroxylation', 'fr_piperdine', 'fr_pyridine', 'fr_sulfide','fr_sulfonamd']
    counts = []
    for f in func_list:
        counts.append(eval('frag_func.{}(mol)'.format(f)))
    return {f: c for f, c in zip(func_list, counts)}

def get_ring_topo(mol):
    n_atoms = mol.GetNumAtoms()
    # num of rings one atom is in
    rings = mol.GetRingInfo().AtomRings()
    nrings_atom_in = np.zeros(n_atoms)
    for ring in rings:
        for atom in ring:
            nrings_atom_in[atom] += 1
    hist = np.histogram(nrings_atom_in, bins=np.arange(-0.5, 9.6).tolist()+[100])[0]
    ring_topo = {f'n_atoms_in_{i}_rings': hist[i] for i in range(0, 11)}
            
    # counts of hub atoms (atoms in three or more rings)
    n_hub = np.sum(nrings_atom_in >= 3)
    ring_topo['n_hub_atoms'] = n_hub
    return ring_topo

def get_metric_one_mol(mol, metric):
    if metric == 'drug_chem':
        func = get_drug_chem
    elif metric == 'count_prop':
        func = get_count_prop
    elif metric == 'global_3d':
        func = get_global_3d
    elif metric == 'frags_counts':
        func = get_frags_counts
    elif metric == 'groups_counts':
        func = get_groups_counts
    elif metric == 'ring_topo':
        func = get_ring_topo
    else:
        raise ValueError('Invalid metric')
    
    if mol is None:
        return {}
    try:
        return func(mol)
    except Exception as e:
        print(e)
        return {}

def get_metric(mols, metric, parallel=False):
    
    func = partial(get_metric_one_mol, metric=metric)
    if not parallel:
        results = []
        for mol in tqdm(mols):
            results.append(func(mol))
    else:
        with Pool(102) as pool:
            results = list(tqdm(pool.imap(func, mols), total=len(mols)))
    
    # fix empty dict
    non_empty_example = [r for r in results if r][0]
    failed = 0
    for i, result in enumerate(results):
        if not result:
            results[i] = {k: np.nan for k in non_empty_example.keys()}
            failed += 1
    print('Total %d, failed %d' % (len(results), failed))
    return results


class Local3D:
    def __init__(self, bonds=None, bonds_pair=None, bonds_triplet=None):
        '''
        bonds: list of single bond smarts, e.g., CC, C=C, C#C, CN, C=N, C#N, CO, C=O, NO
        bonds_pair: list of double bonds, e.g.,
        bonds_triplet: list of triple bonds, e.g., 
        '''
        if bonds is not None:
            self.bonds = [Chem.MolFromSmarts(b) for b in bonds]
        if bonds_pair is not None:
            self.bonds_pair = [Chem.MolFromSmarts(a) for a in bonds_pair]
        if bonds_triplet is not None:
            self.bonds_triplet = bonds_triplet
            
    
    def calc_frequent(self, mols, type_, parallel=False):
        assert type_ in ['length', 'angle', 'dihedral']
        if type_ == 'length':
            smarts_list = self.bonds
        elif type_ == 'angle':
            smarts_list = self.bonds_pair
        elif type_ == 'dihedral':
            smarts_list = self.bonds_triplet
            
        results = {}
        for bond_obj in tqdm(smarts_list, desc=type_):
            results_this_bond = []
            if not parallel:
                for mol in mols:
                    results_this_bond.append(calc_bond_3d(mol, bond_obj, type_))
            else:
                with Pool(102) as pool:
                    func = partial(calc_bond_3d, bond_obj=bond_obj, type_=type_)
                    results_this_bond = list(pool.imap(func, mols), total=len(mols))
            results_this_bond = np.concatenate(results_this_bond)
            results[Chem.MolToSmarts(bond_obj)] = results_this_bond.tolist()
        return results
    
    
    def get_predefined(self,):
        """
        Frequent bonds/pairs/triplet of the Geom-Drug dataset
        """
        bonds_smarts = ['c:c', '[#6]-[#6]', '[#6]-[#7]', '[#6]-O', 'c:n', '[#6]=O', '[#6]-S', 'O=S','c:o', 'c:s',
                '[#6]-F', 'n:n', '[#6]-Cl', '[#6]=[#6]', '[#7]-S', '[#6]=[#7]', '[#7]-[#7]', '[#7]-O', '[#6]=S', '[#7]=O']
        pairs_smarts = ['c:c:c', '[#6]-[#6]-[#6]', '[#6]-[#7]-[#6]', '[#7]-[#6]-[#6]', 'c:c-[#6]', '[#6]-O-[#6]', 'O=[#6]-[#6]', '[#7]-c:c',
                'n:c:c', 'c:c-O', 'c:n:c', '[#6]-[#6]-O', 'O=[#6]-[#7]', ]
        triplet_smarts = ['c:c:c:c', '[#6]-[#6]-[#6]-[#6]', '[#6]-[#7]-[#6]-[#6]', '[#6]-c:c:c', '[#7]-[#6]-[#6]-[#6]', '[#7]-c:c:c', 'O-c:c:c',
                  '[#6]-[#7]-c:c', '[#7]-[#6]-c:c', 'n:c:c:c', '[#6]-[#7]-[#6]=O', '[#6]-[#6]-c:c', 'c:c-[#7]-[#6]', 'c:n:c:c', '[#6]-O-c:c']
        
        self.bonds = [Chem.MolFromSmarts(b) for b in bonds_smarts]
        self.bonds_pair = [Chem.MolFromSmarts(a) for a in pairs_smarts]
        self.bonds_triplet = [Chem.MolFromSmarts(d) for d in triplet_smarts]
        
    def get_freq_bonds(self, mols):
        bonds_sym = []
        for mol in tqdm(mols):
            for mol_bond in mol.GetBonds():
                a0 = mol_bond.GetBeginAtom()
                a1 = mol_bond.GetEndAtom()
                if a0.GetAtomicNum() > a1.GetAtomicNum():
                    a0, a1 = a1, a0
                bond_sym = ''.join([a0.GetSymbol(),
                                    str(int(mol_bond.GetBondType())),
                                    a1.GetSymbol()])
                bonds_sym.append(bond_sym)
        bonds_sym_uni = np.unique(bonds_sym, return_counts=True)
        return bonds_sym_uni
    
    def get_freq_bonds_pair(self, mols):
        bonds_sym_pairs = []
        for mol in tqdm(mols):
            for atom in mol.GetAtoms():
                idx_atom_center = atom.GetIdx()
                atom_type_center = atom.GetSymbol()
                bonds = atom.GetBonds()
                bonds_sym_list = []
                for bond in bonds:
                    atom0 = (bond.GetBeginAtom().GetSymbol(), bond.GetBeginAtom().GetIdx())
                    atom1 = (bond.GetEndAtom().GetSymbol(), bond.GetEndAtom().GetIdx())
                    bond_type = int(bond.GetBondType())
                    # if bond_type == 12:
                    #     continue
                    if atom0[1] == idx_atom_center:
                        atom_other = atom1[0]
                    else:
                        atom_other = atom0[0]
                    bond_sym = ''.join([atom_other, str(bond_type), atom_type_center])
                    bonds_sym_list.append(bond_sym)
                bonds_sym_pairs_this_atom = combinations(bonds_sym_list, r=2)
                for r in bonds_sym_pairs_this_atom:
                    bonds_sym_pairs.append(r[0]+'-'+(r[1][-1]+r[1][1:-1]+r[1][0]))
        bonds_sym_pairs = np.unique(bonds_sym_pairs, return_counts=True)
        return bonds_sym_pairs
    
    def _get_bond_symbol(self, bond, left_symbol=None, right_symbol=None):
        
        a0 = bond.GetBeginAtom().GetSymbol()
        a1 = bond.GetEndAtom().GetSymbol()
        b = str(int(bond.GetBondType()))
        if left_symbol is not None:
            assert right_symbol is None
            if a0 != left_symbol:
                a0, a1 = a1, a0
                assert a0 == left_symbol, 'left_symbol not match'
        elif right_symbol is not None:
            if a1 != right_symbol:
                a0, a1 = a1, a0
                assert a1 == right_symbol, 'right_symbol not match'
        return ''.join([a0, b, a1])
    
    def get_freq_bonds_triplet(self, mols):
        
        valid_triple_bonds = []
        for mol in tqdm(mols):
            for idx_bond, bond in enumerate(mol.GetBonds()):
                idx_begin_atom = bond.GetBeginAtomIdx()
                idx_end_atom = bond.GetEndAtomIdx()
                center_bond = self._get_bond_symbol(bond)
                begin_atom = mol.GetAtomWithIdx(idx_begin_atom)
                begin_ele = begin_atom.GetSymbol()
                end_atom = mol.GetAtomWithIdx(idx_end_atom)
                end_ele = end_atom.GetSymbol()
                begin_bonds = begin_atom.GetBonds()
                valid_left_bonds = []
                for begin_bond in begin_bonds:
                    if begin_bond.GetIdx() == idx_bond:
                        continue
                    else:
                        valid_left_bonds.append(
                            self._get_bond_symbol(begin_bond, right_symbol=begin_ele)
                        )
                if len(valid_left_bonds) == 0:
                    continue

                end_bonds = end_atom.GetBonds()
                for end_bond in end_bonds:
                    if end_bond.GetIdx() == idx_bond:
                        continue
                    else:
                        for left_bond in valid_left_bonds:
                            valid_triple_bonds.append([
                                left_bond,
                                center_bond,
                                self._get_bond_symbol(end_bond, left_symbol=end_ele)
                            ])
        triplet = []
        for triple_bonds in valid_triple_bonds:
            triplet.append('-'.join(triple_bonds))
        triplet = np.unique(triplet, return_counts=True)
        return triplet
    
    def get_counts(self, mols, smarts):
        counts = 0
        for mol in mols:
            find = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
            counts = counts + len(find)
        return counts

def calc_bond_3d(mol, bond_obj, type_):
    if mol is None:
        return []
    assert type_ in ['length', 'angle', 'dihedral']
    if type_ == 'length':
        func = Chem.rdMolTransforms.GetBondLength # (mol.GetConformer(), *match)
    elif type_ == 'angle':
        func = Chem.rdMolTransforms.GetAngleDeg
    elif type_ == 'dihedral':
        func = Chem.rdMolTransforms.GetDihedralDeg
    
    matches = mol.GetSubstructMatches(bond_obj)
    results = []
    for match in matches:
        value = func(mol.GetConformer(), *match)
        results.append(value)
    return results
    


def calculate_validity(output_dir, mol_dict=None):
    """
    Calculate the validity and connectivity of the sampled molecules
    """
    samples_path = os.path.join(output_dir, 'samples_all.pt')
    # if not os.path.exists(samples_path):
    if True:
        # print('samples_all.pt not found. cannot calculate validity and connectivity')
        connectivity = sum([1 for mol in mol_dict.values() if '.' not in Chem.MolToSmiles(mol)]) / len(mol_dict)
        return {'validity': np.nan, 'connectivity': connectivity}
    pool = torch.load(samples_path)

    # if 'edm' in output_dir:  # for edm
    if False:
        n_success = len(pool['finished'])
        n_invalid = 0
        n_disconnect = 0
        for mol in pool['failed']:
            try:
                Chem.SanitizeMol(mol)
            except:
                n_invalid += 1
                continue
            # validate molecule
            smiles = Chem.MolToSmiles(mol)
            if '.' in smiles:
                n_disconnect += 1
                continue
    elif 'moldiff' in output_dir:
        n_success = len(pool['finished'])
        n_invalid = 0
        n_disconnect = 0
        for mol_info in pool['failed']:
            if 'smiles' in mol_info.keys():
                assert '.' in mol_info['smiles']
                n_disconnect += 1
            else:
                n_invalid += 1
    else:
        n_success = len(pool['succ'])
        n_disconnect = len(pool['incomp'])
        n_invalid = len(pool['bad'])
    validity = (n_success + n_disconnect) / (n_success + n_invalid + n_disconnect)
    connectivity = n_success / (n_success + n_disconnect)
    return {'validity': validity, 'connectivity': connectivity}


class RingAnalyzer(object):
    def __init__(self) -> None:
        freq_val_rings = ['c1ccccc1', 'c1ccncc1', 'C1CCCCC1', 'C1CCNCC1', 'C1CNCCN1', 'c1ccoc1', 'c1cncnc1', 'c1ccsc1', 'C1COCCN1', 'C1CCNC1']
        self.freq_val_rings = [Chem.MolFromSmiles(ring) for ring in freq_val_rings]
        
    def get_count_ring(self, mols):
        counts = np.zeros([len(mols), len(self.freq_val_rings)], dtype=np.int64)
        for i, mol in enumerate(mols):
            for j, ring in enumerate(self.freq_val_rings):
                counts[i, j] = self.get_counts(mol, ring)
                
        counts_dict = {'cnt_ring_type_{}'.format(i): counts[:, i] for i in range(len(self.freq_val_rings))}
        return counts_dict
    
    def get_counts(self, mol, ring):
        return len(mol.GetSubstructMatches(ring))

    def get_freq_rings(self, mols, topk=10):
        all_rings = []
        for mol in mols:
            if mol is None:
                continue
            ring_list = [list(ring) for ring in Chem.GetSymmSSSR(mol)]
            ring_smiles = [Chem.MolFragmentToSmiles(mol, ring) for ring in ring_list]
            all_rings.extend(ring_smiles)
        freq_rings, counts = np.unique(all_rings, return_counts=True)
        idx = np.argsort(counts)[::-1]
        freq_rings = freq_rings[idx[:topk]]
        counts = counts[idx[:topk]]
        return {'freq_rings': freq_rings.tolist(), 'counts': counts.tolist()}
    
    
from scipy.spatial.distance import jensenshannon
def get_jsd(p, q):  # actually use js div instead of kld
    # return np.sum(p * np.log(p / q))
    return jensenshannon(p, q)

def compare_with_ref(value_list, width=None, num_bins=50, discrete=False, idx_ref=0):

    # set distribution ranges
    all_list = np.concatenate(value_list)
    all_list = all_list[~np.isnan(all_list)]
    all_list_sort = np.sort(all_list)
    if len(all_list_sort) > 10:
        max_value = all_list_sort[-5]
        min_value = all_list_sort[5]
    else:
        max_value = all_list_sort[-1]
        min_value = all_list_sort[0]
    if not discrete:
        if width is not None:
            bins = np.arange(min_value, max_value+width, width)
        else:
            bins = np.linspace(min_value, max_value, num_bins)
    else:
        bins = np.arange(min_value, max_value+1.5) - 0.5

    # calculate distributions
    hist_list = []
    for metric_method in value_list:
        hist, _ = np.histogram(metric_method, bins=bins, density=True)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        hist_list.append(hist)
        
    # calculate jsd
    jsd_list = []
    for i, hist in enumerate(hist_list):
        kld = get_jsd(hist_list[idx_ref], hist)
        jsd_list.append(kld)
        
    return (jsd_list, bins, hist_list)

if __name__ == '__main__':
    mode = 'test_calc_2d'
    if mode == 'test frequent calculation':
        mol = Chem.MolFromSmiles('CN(CC[C@H](N)CC(=O)N[C@H]1CC[C@H](N2C=C[C@@](N)(O)NC2=O)O[C@@H]1C(=O)O)C(=N)N')
        local3d = Local3D()
        freq_bonds = local3d.get_freq_bonds([mol])
        print(freq_bonds)
        freq_bonds_pair = local3d.get_freq_bonds_pair([mol])
        print(freq_bonds_pair)
        freq_bonds_triplet = local3d.get_freq_bonds_triplet([mol])
        print(freq_bonds_triplet)
        print()
    elif mode == 'test_calc_2d':
        pass
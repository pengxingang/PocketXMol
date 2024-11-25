# import copy
# import os
from itertools import product
from torch_scatter import scatter_min
import sys
sys.path.append('.')
import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
# from rdkit.Chem.AllChem import 
from torch_geometric.nn.pool import knn_graph
from torch_geometric.utils import subgraph, bipartite_subgraph, to_undirected, sort_edge_index
from torch_geometric.transforms import Compose  # imported by train.py

from models.transition import *

from utils.train_noise import get_vector, get_vector_list, combine_vectors_indexed
from utils.data import Mol3DData, PocketMolData
from utils.dataset import *
from utils.misc import *
from process.utils_process import process_raw
# from utils.train import inf_iterator
# from utils.protein_ligand import ATOM_FAMILIES
# from utils.reconstruct import reconstruct_from_generated_with_edges


CONF_SETTINGS = ['free', 'flexible', 'torsional', 'rigid']
MASKFILL_SETTINGS = {
    'decomposition': ['brics', 'mmpa', 'atom'],
    'order': ['tree', 'inv_tree', 'random'],
    'part1_pert': ['fixed', 'free', 'small', 'rigid', 'flexible'],
    'known_anchor': ['all', 'partial', 'none']
}
PEPDESIGN_SETTINGS = {
    'mode': ['full', 'sc', 'packing']
}


TRAIN_DICT = {} 
def register_transforms(name):
    def decorator(cls):
        TRAIN_DICT[name] = cls
        return cls
    return decorator

def get_transforms(config, *args, **kwargs):
    name = config.name
    return TRAIN_DICT[name](config, *args, **kwargs)


def halfedge_index_to_1d(halfedge_index, num_nodes):
    """
    halfedge_index: (2, num_edges)
    num_nodes: int
    """
    assert (halfedge_index[0] < halfedge_index[1]).all(), 'halfedge_index[0] must be smaller than halfedge_index[1]'
    id_edge = ((2 * num_nodes - halfedge_index[0] - 1) * halfedge_index[0] // 2
               + halfedge_index[1] - halfedge_index[0] - 1)
    return id_edge

def add_rdkit_conf(data=None, mol=None):
    if mol is None:
        # reconstruct rdmol from data
        # mol_info = {
        #     'atom_pos': data.node_pos,
        #     'element': data.element,
        #     'bond_index': data.bond_index,
        #     'bond_type': data.bond_type
        # }
        # mol = reconstruct_from_generated_with_edges(mol_info)  # not exactly same as before. need to be fixed
        path = os.path.join('data', data.db, 'mols', data.data_id + '.sdf')
        mol = Chem.MolFromMolFile(path)
    else:
        assert data is None, 'data and mol cannot be both not None'
    # get rdkit conformer
    # mol.Compute2DCoords()
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, clearConfs=True)
    try:
        n_tries = 5
        for i_try in range(n_tries):
            notconverge = AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            if 1-notconverge:
                break
            else:
                # AllChem.EmbedMolecule(mol)
                if i_try == n_tries-1:
                    print(f'Failed to optimize molecule after {n_tries} tries.')
    except:
        print('Error when optimizing molecule.')
        pass
    mol = Chem.RemoveHs(mol)
    pos_rdkit = mol.GetConformer().GetPositions()
    # check atom element is not changed
    assert (data.element == 
            torch.LongTensor([atom.GetAtomicNum() for atom in mol.GetAtoms()])).all().item()
    # check bond is not changed
    new_bond_type = torch.LongTensor([mol.GetBondBetweenAtoms(*[a.item() for a in bond]).GetBondType()
                                                for bond in data.bond_index.T])
    new_bond_type = torch.where(new_bond_type==12, 4, new_bond_type)  # aromatic bond
    if not (new_bond_type == data.bond_type).all().item():
        print('Warning: bond type is changed after rdkit optimization')
        raise ValueError('Warning: bond type is changed after rdkit optimization')

    # overwirte atom pos
    node_pos = torch.FloatTensor(pos_rdkit)
    data['node_pos'] = node_pos - node_pos.mean(dim=0)
    data['i_conf'] = -1
    return data

@register_transforms('cut_peptide')  # seems not used
class CutPeptide(object):
    def __init__(self, config, *args, **kwargs) -> None:
        self.config = config
        self.applicable_tasks = config.applicable_tasks
        self.individual = config.individual
        self.exclude_keys = ['is_atom_remain']

    def __call__(self, data: PocketMolData):
        task = data['task']
        if (task in self.applicable_tasks) and ('peptide_res_index' in data): # only for peptide
            # cut peptide
            config_this = self.individual[task]
            prob = config_this.prob
            # n_res = data['peptide_pep_len']
            n_res = data['peptide_res_index'].max().item() + 1
            if np.random.rand() < prob or n_res > config_this.limit_nres:
                min_nres = config_this.min_nres
                max_nres = min(config_this.max_nres, n_res-1)
                n_res_remain = np.random.randint(min_nres, max_nres+1)
                index_res_start = np.random.randint(0, n_res-n_res_remain+1)
                index_res = torch.arange(index_res_start, index_res_start+n_res_remain)
                peptide_res_index = data['peptide_res_index']
                is_atom_remain = (peptide_res_index[:, None] == index_res[None]).any(dim=1)
            
                data['is_atom_remain'] = is_atom_remain
                assert is_atom_remain.sum() > 0, 'no atom remain'
                return data
        return data


# @register_transforms('make_fake_data')
# class MakeFakeData(object):
#     def __init__(self, config, *args, **kwargs) -> None:
#         self.config = config
#         self.mol_type = config['mol_type']
#         assert self.mol_type in ['mol', 'peptide'], 'mol_type must be mol or peptide'
        
#         self.bb_names = ['N', 'CA', 'C', 'O']
    
#     def __call__(self, data: Mol3DData):
#         # make mol size
#         if self.mol_type == 'peptide':
#             # determine size
#             size = self.config.size
#             n_res = np.random.choice(size.values, weights=size.weights)
#             n_bb_atoms = 4 * n_res
#             # make fake mol
#             smiles = 
            
        
@register_transforms('overwrite_start_pos')  # for dock flex. used _mol_start.sdf provided by posebuster. but not really necessary
class OverwriteStartPos(object):
    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.start_mol_path = config['start_mol_path']
        self.appendix = config.get('appendix', '')
        self.center_policy = config.get('center_policy', 'unknown')
    
    def __call__(self, data):

        data_id = data['data_id']
        mol_path = os.path.join(self.start_mol_path, data_id + self.appendix)
        mol = Chem.MolFromMolFile(mol_path)
        mol = Chem.RemoveAllHs(mol)
        if mol is None:
            print('mol is None: ', data_id, self.i_repeat)
        atom_pos = mol.GetConformer(0).GetPositions()
        data['node_pos'] = torch.tensor(atom_pos, dtype=data['node_pos'].dtype)
        if self.center_policy == 'unknown':
            data['node_pos'] = data['node_pos'] - data['node_pos'].mean(0, keepdim=True)
        elif self.center_policy == 'known':
            data['node_pos'] = data['node_pos']- data.pocket_center
        else:
            raise f'Invalid center_policy {self.center_policy}'
            
        assert len(data['node_pos']) == len(data['node_type']), 'size not match'
        assert [atom.GetAtomicNum() for atom in mol.GetAtoms()] == data['element'].tolist(), 'element not match'
        return data

@register_transforms('overwrite_pos')
class OverwritePos(object):  # for belief to get the gen pos
    def __init__(self, config, *args, **kwargs):
        self.config = config
        if config is not None:
            self.gen_path = config.gen_path
            self.i_repeat = config.i_repeat
        else:
            self.gen_path = kwargs['gen_path']
            self.i_repeat = kwargs['i_repeat']
        
        self.df_gen = pd.read_csv(os.path.join(self.gen_path, 'gen_info.csv'))
        
    def __call__(self, data):
        # i_repeat = i_repeat if i_repeat is not None else self.i_repeat
        # assert i_repeat is not None, 'i_repeat is not specified'

        data_id = data['data_id']
        line = self.df_gen[(self.df_gen['data_id'] == data_id) & (self.df_gen['i_repeat'] == self.i_repeat)]
        assert len(line) == 1, ('find multiple lines or none with the same data_id and i_repeat', data_id, self.i_repeat, line)
        filename = line['filename'].values[0]
        mol_path = os.path.join(self.gen_path, 'SDF', filename)
        data['filename'] = filename
        data['i_repeat'] = self.i_repeat
        
        if filename.endswith('.sdf'):
            mol = Chem.MolFromMolFile(mol_path)
        elif filename.endswith('.pdb'):
            mol = Chem.MolFromPDBFile(mol_path, sanitize=False)
        if mol is None:
            print('mol is None: ', data_id, self.i_repeat)
            # print('Use random pos insteads')
            # data['node_pos'] = torch.
        else:
            atom_pos = mol.GetConformer(0).GetPositions()
            data['node_pos'] = (torch.tensor(atom_pos, dtype=data['node_pos'].dtype)
                                - data.pocket_center)
            assert len(data['node_pos']) == len(data['node_type']), 'size not match'
            assert [atom.GetAtomicNum() for atom in mol.GetAtoms()] == data['element'].tolist(), 'element not match'
        return data


@register_transforms('overwrite_pos_repeat')
class OverwritePosRepeat(object):  # for linking with unknown fragmen pos to get the initial overwrite
    def __init__(self, config, i_repeat):
        self.config = config
        self.i_repeat = i_repeat
        
    def __call__(self, data):
        
        if self.config['starategy'] == 'linking_unfixed':
            file_dir = self.config['file_dir']
            sdf_dirs = os.listdir(file_dir)
            sep_name = data['key'].split(';')[-1].replace('linking/', '').replace('/', '_sep_')
            sdf_name = [name for name in sdf_dirs if name.endswith(sep_name)]
            assert len(sdf_name) == 1, f'find {len(sdf_name)} files with name {sep_name}'
            sdf_name = sdf_name[0]
            
            # add repeat
            filename = os.path.join(sdf_name, f'repeat_{self.i_repeat}.sdf')
            mol_path = os.path.join(file_dir, filename)
            data['filename'] = filename
            data['i_repeat'] = self.i_repeat
        
        
        if filename.endswith('.sdf'):
            mol = Chem.MolFromMolFile(mol_path, sanitize=False)
        elif filename.endswith('.pdb'):
            mol = Chem.MolFromPDBFile(mol_path, sanitize=False)
        if mol is None:
            print('mol is None: ', data['data_id'], self.i_repeat)
            # print('Use random pos insteads')
            # data['node_pos'] = torch.
        else:
            atom_pos = mol.GetConformer(0).GetPositions()
            atom_pos = torch.tensor(atom_pos, dtype=data['node_pos'].dtype)
            if data['pocket_center'].shape[0] > 0:
                atom_pos = atom_pos - data.pocket_center
            else:
                atom_pos = atom_pos - atom_pos.mean(0, keepdim=True)
            data['node_pos'] = atom_pos
            assert len(data['node_pos']) == len(data['node_type']), 'size not match'
            assert [atom.GetAtomicNum() for atom in mol.GetAtoms()] == data['element'].tolist(), 'element not match'
        return data


@register_transforms('overwrite_mol_repeat')
class OverwriteMolRepeat(object):  # for mol optimize to overwrite the init mol (so that no need for re-assemble db)
    def __init__(self, config, i_repeat):
        self.config = config
        self.i_repeat = i_repeat
        
        if self.config['starategy'] == 'mol_opt':
            self.df = pd.read_csv(os.path.join(self.config['file_root'], self.config['df_path']))
        
    def __call__(self, data):
        
        if self.config['starategy'] == 'mol_opt':
            total_files = self.config['total_files']
            sdf_dir = os.path.join(self.config['file_root'], self.config['sdf_dir'])
            
            data_id = data['data_id']
            file_repeat = self.i_repeat % total_files
            
            line = self.df[(self.df['data_id'] == data_id) & (self.df['i_repeat'] == file_repeat)]
            assert len(line) == 1, 'find multiple lines or none with the same data_id and i_repeat'
            filename = line['filename'].values[0]
            
            # add repeat
            mol_path = os.path.join(sdf_dir, filename)
            data['filename'] = filename
            # data['i_repeat'] = self.i_repeat
            
            # overwite
            assert data['task'] == 'sbdd', 'only sbdd supported, otherwise process more mol info'
            new_data = process_raw(data_id=data_id, mol_path=mol_path, modes=['mols'],
                                pdbid=data.get('pdbid', ''))
            # for data_key in new_data.keys:
            #     if data_key in data.keys:
            #         data[data_key] = new_data[data_key]
            for data_key in new_data.keys():
                if data_key in data:
                    data[data_key] = new_data[data_key]
            return data


@register_transforms('variable_sc_size')
class VariableScSize(object):  # for sampling
    def __init__(self, config, *args, **kwargs) -> None:
        self.config = config
        self.applicable_tasks = config.applicable_tasks
        self.num_atoms_distri = config.num_atoms_distri
        self.exclude_keys = ['is_atom_remain', 'added_index', 'removed_index']
        
        self.not_remove = config.get('not_remove', [])

    def __call__(self, data: PocketMolData):
        task = data['task']
        if (task in self.applicable_tasks) and ('peptide_res_index' in data): # only for peptide
            n_atoms_data = data['node_type'].shape[0]
            n_res = data['peptide_pep_len']
            assert n_res == data['peptide_res_index'].max().item() + 1

            
            n_atoms_mean = self.num_atoms_distri['mean'] * n_res
            n_atoms_std = self.num_atoms_distri['std']['coef'] * n_res + self.num_atoms_distri['std']['bias']
            n_atoms_new = int(np.random.normal(n_atoms_mean, n_atoms_std))
            if n_atoms_new == n_atoms_data: # what a coincidence!!!
                pass
            elif n_atoms_new > n_atoms_data: # add atoms
                data = self.add_atoms(data, n_atoms_new, n_atoms_data)
            elif n_atoms_new < n_atoms_data: # remove atoms
                data = self.remove_atoms(data, n_atoms_new, n_atoms_data)
            # common
            if 'is_peptide' in data:
                is_peptide = data['is_peptide']
                data['is_peptide'] = is_peptide[0] * torch.ones([n_atoms_new], dtype=is_peptide.dtype)
        return data
            
    def add_atoms(self, data, n_atoms_new, n_atoms_data):
        n_add = n_atoms_new - n_atoms_data
        
        # new node
        node_type = data['node_type']
        new_node_type = torch.cat([node_type, torch.zeros([n_add], dtype=node_type.dtype)], dim=0)
        added_index = np.arange(n_atoms_data, n_atoms_new)
        
        # new node pos
        # determine positions
        is_sidechain = (~data['peptide_is_backbone'])
        n_sc = is_sidechain.sum()
        if n_sc != 0:
            node_sc = torch.nonzero(is_sidechain)[:, 0]
            node_sc_center = node_sc[torch.randint(n_sc, size=[n_add])]
        else:
            node_sc = torch.nonzero(~is_sidechain)[:, 0]  # no sc. can only use backbone
            node_sc_center = node_sc[torch.randint(len(node_sc), size=[n_add])]
        len_mu, len_sigma = 3, 0.6
        lengths = torch.randn([n_add]) * len_sigma + len_mu
        relative_pos = torch.randn([n_add, 3])
        relative_pos = relative_pos / (relative_pos.norm(dim=-1, keepdim=True)+1e-5) * lengths[:, None]
        node_pos = data['node_pos']
        node_pos_new = node_pos[node_sc_center] + relative_pos
        new_node_pos = torch.cat([node_pos, node_pos_new], dim=0)
        
        # new edge
        halfedge_index = data['halfedge_index']
        halfedge_type = data['halfedge_type']
        n_add_halfedge = n_add * n_atoms_data + n_add * (n_add - 1) // 2
        new_halfedge_type = torch.cat([halfedge_type, torch.zeros([n_add_halfedge], dtype=halfedge_type.dtype)], dim=0)
        halfedge_index_old_new = torch.stack(
            torch.meshgrid(torch.arange(n_atoms_data), torch.arange(n_atoms_data, n_atoms_new), indexing='ij'),
        dim=0).reshape(2, -1)
        halfedge_index_new_new = torch.triu_indices(n_add, n_add, offset=1) + n_atoms_data
        new_halfedge_index = torch.cat([
            halfedge_index, halfedge_index_old_new, halfedge_index_new_new], dim=1)
        new_halfedge_index, new_halfedge_type = sort_edge_index(new_halfedge_index, new_halfedge_type)

        # peptide feature
        peptide_is_backbone = data['peptide_is_backbone']
        peptide_atom_name = data['peptide_atom_name']
        new_peptide_is_backbone = torch.cat([peptide_is_backbone, torch.zeros([n_add], dtype=peptide_is_backbone.dtype)], dim=0)
        new_peptide_atom_name = peptide_atom_name + ['X'] * n_add

        data.update({
            'num_nodes': n_atoms_new,
            'node_type': new_node_type,
            'node_pos': new_node_pos,
            'halfedge_index': new_halfedge_index,
            'halfedge_type': new_halfedge_type,
            'peptide_is_backbone': new_peptide_is_backbone,
            'peptide_atom_name': new_peptide_atom_name,
            'added_index': added_index,  # for custom task to know the part name of the added nodes
        })
        return data
    
    
    def remove_atoms(self, data, n_atoms_new, n_atoms_data):
        n_remove = n_atoms_data - n_atoms_new
        peptide_is_backbone = data['peptide_is_backbone']
        peptide_is_sc = ~peptide_is_backbone
        
        
        peptide_is_removable = peptide_is_sc
        peptide_is_removable[self.not_remove] = False
        # n_bb, n_sc = peptide_is_backbone.sum(), peptide_is_sc.sum()
        n_removable = peptide_is_removable.sum()
        n_remove = min(n_remove, n_removable)

        is_atom_remain = torch.ones([n_atoms_data], dtype=torch.bool)
        index_removable = torch.nonzero(peptide_is_removable)[:, 0]
        index_remove = np.random.choice(index_removable, n_remove, replace=False)
        is_atom_remain[index_remove] = False

        # node 
        data['node_pos'] = data['node_pos'][is_atom_remain]
        data['node_type'] = data['node_type'][is_atom_remain]
        data['num_nodes'] = data['node_type'].shape[0]
        data['removed_index'] = index_remove  # for custom task to rearange the index of remaining nodes
        # edge
        halfedge_index, halfedge_type = subgraph(
            torch.nonzero(is_atom_remain)[:, 0], edge_index=data.halfedge_index,
            edge_attr=data.halfedge_type, relabel_nodes=True, num_nodes=n_atoms_data)
        data['halfedge_index'] = halfedge_index
        data['halfedge_type'] = halfedge_type
        # peptide
        data['peptide_is_backbone'] = data['peptide_is_backbone'][is_atom_remain]
        data['peptide_atom_name'] = [name for is_r, name in zip(is_atom_remain, data['peptide_atom_name'])
                                     if is_r]
        
        return data


@register_transforms('variable_mol_size')
class VariableMolSize(object):  # for sampling
    def __init__(self, config, *args, **kwargs) -> None:
        self.config = config
        # self.applicable_tasks = config.applicable_tasks
        self.num_atoms_distri = config.num_atoms_distri
        self.exclude_keys = ['is_atom_remain', 'added_index', 'removed_index']
        
        self.not_remove = config.get('not_remove', [])

    def __call__(self, data: PocketMolData):
        
        n_atoms_data = data['node_type'].shape[0]
        n_atoms_new = self.sample_n_atoms(data)

        if n_atoms_new == n_atoms_data: # what a coincidence~~~
                pass
        elif n_atoms_new > n_atoms_data: # add atoms
            data = self.add_atoms(data, n_atoms_new, n_atoms_data)
        elif n_atoms_new < n_atoms_data: # remove atoms
            data = self.remove_atoms(data, n_atoms_new, n_atoms_data)
        # common
        if 'is_peptide' in data:
            is_peptide = data['is_peptide']
            data['is_peptide'] = is_peptide[0] * torch.ones([n_atoms_new], dtype=is_peptide.dtype)
        return data

    def sample_n_atoms(self, data):
        strategy = self.num_atoms_distri['strategy']
        if strategy == 'pocket_atoms_based':
            num_atoms_pocket = len(data['pocket_pos'])
            n_atoms_mean = self.num_atoms_distri['mean']['coef'] * num_atoms_pocket + self.num_atoms_distri['mean']['bias']
            n_atoms_std = self.num_atoms_distri['std']['coef'] * num_atoms_pocket + self.num_atoms_distri['std']['bias']
            sample_func = lambda: int(np.round(np.random.normal(n_atoms_mean, n_atoms_std)))
        elif strategy == 'mol_atoms_based':
            num_atoms = len(data['node_pos'])
            n_atoms_mean = self.num_atoms_distri['mean']['coef'] * num_atoms + self.num_atoms_distri['mean']['bias']
            n_atoms_std = self.num_atoms_distri['std']['coef'] * num_atoms + self.num_atoms_distri['std']['bias']
            sample_func = lambda: int(np.round(np.random.normal(n_atoms_mean, n_atoms_std)))
        elif strategy == 'multinomial':
            sizes = self.num_atoms_distri['values']
            probs = self.num_atoms_distri['probs']
            sample_func = lambda: int(np.random.choice(sizes, p=probs))
        else:
            raise NotImplementedError(f'num_atoms_distri strategy {self.num_atoms_distri["name"]} not implemented')
        
        n_atoms_min = self.num_atoms_distri.get('min', 2)
        n_atoms_max = self.num_atoms_distri.get('max', 1000000)
        # sample n
        count_try = 0
        while True:
            n_atoms_new = sample_func()
            count_try += 1
            if n_atoms_new >= n_atoms_min and n_atoms_new <= n_atoms_max:
                break
            if count_try > 1000:
                print('Warning: too many tries to sample n_atoms for variable_mol_size')
                n_atoms_new = n_atoms_min
                break
        return n_atoms_new
            
            
    def add_atoms(self, data, n_atoms_new, n_atoms_data):
        n_add = n_atoms_new - n_atoms_data
        
        # new node
        node_type = data['node_type']
        new_node_type = torch.cat([node_type, torch.zeros([n_add], dtype=node_type.dtype)], dim=0)
        added_index = np.arange(n_atoms_data, n_atoms_new)
        
        # new node pos
        # determine positions
        len_mu, len_sigma = 2, 0.5
        node_pos = data['node_pos']
        node_pos_center = node_pos[torch.randint(n_atoms_data, size=[n_add])]

        lengths = torch.randn([n_add]) * len_sigma + len_mu
        relative_pos = torch.randn([n_add, 3])
        relative_pos = relative_pos / (relative_pos.norm(dim=-1, keepdim=True)+1e-5) * lengths[:, None]
        node_pos_new = node_pos_center + relative_pos
        new_node_pos = torch.cat([node_pos, node_pos_new], dim=0)
        
        # new edge
        halfedge_index = data['halfedge_index']
        halfedge_type = data['halfedge_type']
        n_add_halfedge = n_add * n_atoms_data + n_add * (n_add - 1) // 2
        new_halfedge_type = torch.cat([halfedge_type, torch.zeros([n_add_halfedge], dtype=halfedge_type.dtype)], dim=0)
        halfedge_index_old_new = torch.stack(
            torch.meshgrid(torch.arange(n_atoms_data), torch.arange(n_atoms_data, n_atoms_new), indexing='ij'),
        dim=0).reshape(2, -1)
        halfedge_index_new_new = torch.triu_indices(n_add, n_add, offset=1) + n_atoms_data
        new_halfedge_index = torch.cat([
            halfedge_index, halfedge_index_old_new, halfedge_index_new_new], dim=1)
        new_halfedge_index, new_halfedge_type = sort_edge_index(new_halfedge_index, new_halfedge_type)

        # peptide feature
        # peptide_is_backbone = data['peptide_is_backbone']
        # peptide_atom_name = data['peptide_atom_name']
        # new_peptide_is_backbone = torch.cat([peptide_is_backbone, torch.zeros([n_add], dtype=peptide_is_backbone.dtype)], dim=0)
        # new_peptide_atom_name = peptide_atom_name + ['X'] * n_add

        data.update({
            'num_nodes': n_atoms_new,
            'node_type': new_node_type,
            'node_pos': new_node_pos,
            'halfedge_index': new_halfedge_index,
            'halfedge_type': new_halfedge_type,
            # 'peptide_is_backbone': new_peptide_is_backbone,
            # 'peptide_atom_name': new_peptide_atom_name,
            'added_index': added_index,  # for custom task to know the part name of the added nodes
        })
        return data
    
    
    def remove_atoms(self, data, n_atoms_new, n_atoms_data):
        n_remove = n_atoms_data - n_atoms_new

        is_removable = torch.ones([n_atoms_data], dtype=torch.bool)
        is_removable[self.not_remove] = False
        n_removable = is_removable.sum()
        n_remove = min(n_remove, n_removable)

        is_atom_remain = torch.ones([n_atoms_data], dtype=torch.bool)
        index_removable = torch.nonzero(is_removable)[:, 0]
        index_remove = np.random.choice(index_removable, n_remove, replace=False)
        is_atom_remain[index_remove] = False

        # node 
        data['node_pos'] = data['node_pos'][is_atom_remain]
        data['node_type'] = data['node_type'][is_atom_remain]
        data['num_nodes'] = data['node_type'].shape[0]
        data['removed_index'] = index_remove  # for custom task to rearange the index of remaining nodes
        # edge
        halfedge_index, halfedge_type = subgraph(
            torch.nonzero(is_atom_remain)[:, 0], edge_index=data.halfedge_index,
            edge_attr=data.halfedge_type, relabel_nodes=True, num_nodes=n_atoms_data)
        data['halfedge_index'] = halfedge_index
        data['halfedge_type'] = halfedge_type
        # peptide
        # data['peptide_is_backbone'] = data['peptide_is_backbone'][is_atom_remain]
        # data['peptide_atom_name'] = [name for is_r, name in zip(is_atom_remain, data['peptide_atom_name'])
        #                              if is_r]
        
        return data



class FeaturizePocket(object):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.knn = config.knn
        
        self.atomic_numbers = torch.LongTensor([6, 7, 8, 16])    # C, N, O, S
        self.max_num_aa = 20
        
        self.follow_batch = ['pocket_pos']
        self.exclude_keys = ['pocket_element', 'pocket_molecule_name',
                             'pocket_is_backbone', 'pocket_atom_name', 'pocket_atom_to_aa_type', ]
        
        self.preset_pocket_center = config.get('center', None)

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1  # 1 for is_backbone

    def __call__(self, data:PocketMolData):
        
        # no pocket data, make dummy
        # if ('pocket_pos' not in data.keys) or (len(data['pocket_pos']) == 0):
        #     if ('pocket_pos' in data.keys) and len(data['pocket_pos'] == 0):
        if ('pocket_pos' not in data) or (len(data['pocket_pos']) == 0):
            if ('pocket_pos' in data) and len(data['pocket_pos'] == 0):
                pdbid = data['pdbid']
                print(f'Warning: empty pocket: {data["data_id"]}')
            else:
                pdbid = ''
            data.update({
                'pocket_atom_feature': torch.empty([0, self.feature_dim], dtype=torch.float),
                'pocket_knn_edge_index': torch.empty([2, 0], dtype=torch.long),
                'pocket_pos': torch.empty([0, 3], dtype=torch.float),
                'pocket_center': torch.empty([0, 3], dtype=torch.float),
                'pdbid': pdbid,
            })
            return data
        
        # if len(data['pocket_pos']) == 0:
            # raise ValueError(f"empty pocket: {data['data_id']}")
        # pocket atom features
        assert all([(e in self.atomic_numbers) for e in data.pocket_element]), 'unknown element in pocket'
        element = (data.pocket_element.view(-1, 1) == self.atomic_numbers.view(1, -1)).long()   # (N_atoms, N_elements)
        amino_acid = F.one_hot(data.pocket_atom_to_aa_type, num_classes=self.max_num_aa)
        is_backbone = data.pocket_is_backbone.view(-1, 1).long()
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        data['pocket_atom_feature'] = x.float()
        
        if 'is_atom_remain' in data:  # apply cut for pep, so need to update pocket
            raise ValueError('not supported anymore: is_atom_remain')
            is_atom_remain = data['is_atom_remain']
            pocket_pos = data['pocket_pos']
            node_pos = data['pos_all_confs'][0]
            dist = torch.norm(pocket_pos[:, None] - node_pos[None], dim=-1) # (n_pocket, n_node)
            threshold = dist.min(dim=1)[0].max()
            is_pocket_remain = dist[:, is_atom_remain].min(dim=1)[0] < threshold
            data['pocket_atom_feature'] = data['pocket_atom_feature'][is_pocket_remain]
            data['pocket_pos'] = data['pocket_pos'][is_pocket_remain]
        # pocket inner edge features
        pocket_knn_edge_index = knn_graph(
            data.pocket_pos, k=self.knn, flow='target_to_source')
        data['pocket_knn_edge_index'] = pocket_knn_edge_index
        
        # pocket pos and center
        if self.preset_pocket_center is not None:
            pocket_center = torch.tensor(self.preset_pocket_center).reshape(1, 3).float()
        else:
            pocket_center = data.pocket_pos.mean(dim=0, keepdim=True)
        data['pocket_center'] = pocket_center
        data['pocket_pos'] = data.pocket_pos - data.pocket_center
        pdbid = data['pdbid']
        data['pdbid'] = pdbid if (pdbid == pdbid) else ''
        return data


class FeaturizeMol(object):
    def __init__(self, config):
        super().__init__()
        atomic_numbers = config.chem.atomic_numbers
        mol_bond_types = config.chem.mol_bond_types
        use_mask_node = config.use_mask_node
        use_mask_edge = config.use_mask_edge
        
        self.atomic_numbers = torch.LongTensor(atomic_numbers)
        self.mol_bond_types = torch.LongTensor(mol_bond_types)
        self.num_element = self.atomic_numbers.size(0)
        self.num_bond_types = self.mol_bond_types.size(0)

        self.num_node_types = self.num_element + int(use_mask_node)
        self.num_edge_types = self.num_bond_types + 1 + int(use_mask_edge) # + 1 for the non-bonded edges
        self.use_mask_node = use_mask_node
        self.use_mask_edge = use_mask_edge
        
        self.ele_to_nodetype = {ele: i for i, ele in enumerate(atomic_numbers)}
        self.nodetype_to_ele = {i: ele for i, ele in enumerate(atomic_numbers)}
        
        
        self.follow_batch = ['node_type', 'halfedge_type']
        self.exclude_keys = ['orig_keys', 'pos_all_confs', 'num_confs', 'i_conf_list',
                             'bond_index', 'bond_type', 'num_bonds', 'num_atoms',
                             # torsional and decom
                             'bond_rotatable', 'tor_twisted_pairs', 'fixed_dist_torsion',
                             'path_mat', 'nbh_dict', 'tor_bond_mat', 'matches_graph', 'matches_iso',
                             'mmpa', 'brics',
                             # peptide
                             'peptide_pos', 'peptide_atom_name', 'peptide_is_backbone',
                             'peptide_res_id', 'peptide_atom_to_aa_type', 'peptide_res_index',
                             'peptide_seq', 'peptide_pep_len', 'peptide_pep_path',
                             ]
        
        self.mol_as_pocket_center = config.get('mol_as_pocket_center', False)
        self.is_peptide = config.get('is_peptide', False)
    
    def __call__(self, data: Mol3DData):
        
        data.num_nodes = data.num_atoms
        
        # node type
        assert np.all([ele in self.atomic_numbers for ele in data.element]), 'unknown element'
        data.node_type = torch.LongTensor([self.ele_to_nodetype[ele.item()] for ele in data.element])
        
        # atom pos: sample a conformer from data.pos_all_confs; then move to origin
        if data.get('task', '') == 'linking':
            idx = 0
        else:
            idx = np.random.randint(data.pos_all_confs.shape[0])
        atom_pos = data.pos_all_confs[idx].float()
        
        # move to center
        # if move:
        if len(getattr(data, 'pocket_center', [])) > 0:
            atom_pos = atom_pos - data.pocket_center
            if self.mol_as_pocket_center:  # change the global center as the mol center
                mol_center = atom_pos.mean(dim=0, keepdim=True)
                atom_pos = atom_pos - mol_center
                data['pocket_pos'] = data['pocket_pos'] - mol_center
                data['pocket_center'] = data['pocket_center'] + mol_center
        else:
            atom_pos = atom_pos - atom_pos.mean(dim=0)

        data.node_pos = atom_pos
        data.i_conf = data.i_conf_list[idx]
        
        # build half edge (not full because perturb for edge_ij should be the same as edge_ji)
        edge_type_mat = torch.zeros([data.num_nodes, data.num_nodes], dtype=torch.long)
        for i in range(data.num_bonds * 2):  # multiply by to is for symmtric of bond index
            edge_type_mat[data.bond_index[0, i], data.bond_index[1, i]] = data.bond_type[i]
        halfedge_index = torch.triu_indices(data.num_nodes, data.num_nodes, offset=1)
        halfedge_type = edge_type_mat[halfedge_index[0], halfedge_index[1]]
        assert len(halfedge_type) == len(halfedge_index[0])
        # max_bond = torch.norm(atom_pos[data.bond_index[0]] - atom_pos[data.bond_index[1]], dim=-1).max()
        # assert max_bond < 4, f'bond length too long: {max_bond.item()} in {data.data_id}'
        
        data.halfedge_index = halfedge_index
        data.halfedge_type = halfedge_type
        assert (data.halfedge_type > 0).sum() == data.num_bonds
        
        if 'is_atom_remain' in data:
            raise ValueError('not supported anymore: is_atom_remain')
        
        if self.is_peptide:
            data['is_peptide'] = torch.ones([data.num_nodes], dtype=torch.long)
        else:
            data['is_peptide'] = torch.zeros([data.num_nodes], dtype=torch.long)  # default is not peptide

        return data
    
    def decode_output(self, node, pos, halfedge, halfedge_index,
                      pocket_center=None):
        """
        Get the atom and bond information from the prediction (latent space)
        They should be np.array
        pred_node: [n_nodes, n_node_types]
        pred_pos: [n_nodes, 3]
        pred_halfedge: [n_halfedges, n_edge_types]
        """
        # move back to pocekt center in pdb
        if pocket_center is not None:
            pos = pos + pocket_center
        # get atom and element
        # if is_prob:
        #     pred_atom = softmax(pred_node, axis=-1)
        #     atom_type = np.argmax(pred_atom, axis=-1)
        #     atom_prob = np.max(pred_atom, axis=-1)
        # else:
        atom_type = node
        # atom_prob = np.ones(len(atom_type))
        isnot_masked_atom = (atom_type < self.num_element) & (atom_type >= 0)
        if not isnot_masked_atom.all():
            edge_index_changer = - np.ones(len(isnot_masked_atom), dtype=np.int)
            edge_index_changer[isnot_masked_atom] = np.arange(isnot_masked_atom.sum())
        atom_type = atom_type[isnot_masked_atom]
        # atom_prob = atom_prob[isnot_masked_atom]
        element = np.array([self.nodetype_to_ele[i] for i in atom_type])
        
        # get pos
        atom_pos = pos[isnot_masked_atom]
        atom_pos_masked = pos[~isnot_masked_atom]
        
        # get bond
        if self.num_edge_types == 1:
            return {
                'element': element,
                'atom_pos': atom_pos,
                # 'atom_prob': atom_prob,
            }
        # if is_prob:
        #     pred_halfedge = softmax(pred_halfedge, axis=-1)
        #     edge_type = np.argmax(pred_halfedge, axis=-1)  # omit half for simplicity
        #     edge_prob = np.max(pred_halfedge, axis=-1)
        # else:
        edge_type = halfedge
        # edge_prob = np.ones(len(edge_type))
        
        is_bond = (edge_type > 0) & (edge_type <= self.num_bond_types)  # larger is mask type
        bond_type = edge_type[is_bond]
        # bond_prob = edge_prob[is_bond]
        bond_index = halfedge_index[:, is_bond]
        if not isnot_masked_atom.all():
            bond_index = edge_index_changer[bond_index]
            bond_for_masked_atom = (bond_index < 0).any(axis=0)
            bond_index = bond_index[:, ~bond_for_masked_atom]
            bond_type = bond_type[~bond_for_masked_atom]
            # bond_prob = bond_prob[~bond_for_masked_atom]

        bond_type = np.concatenate([bond_type, bond_type])
        # bond_prob = np.concatenate([bond_prob, bond_prob])
        bond_index = np.concatenate([bond_index, bond_index[::-1]], axis=1)
        
        return {
            'element': element,
            'atom_pos': atom_pos,
            'bond_type': bond_type,
            'bond_index': bond_index,
            
            # 'atom_prob': atom_prob,
            # 'bond_prob': bond_prob,
            
            'atom_pos_masked': atom_pos_masked,
        }
    
def make_data_placeholder(n_graphs, device=None, max_size=None):
    # n_nodes_list = np.random.randint(15, 50, n_graphs)
    if max_size is None:
        n_nodes_list = np.random.normal(24.923464980477522, 5.516291901819105, size=n_graphs)
    else:
        n_nodes_list = np.array([max_size] * n_graphs)
    n_nodes_list = n_nodes_list.astype('int64')
    batch_node = np.concatenate([np.full(n_nodes, i) for i, n_nodes in enumerate(n_nodes_list)])
    halfedge_index = []
    batch_halfedge = []
    idx_start = 0
    for i_mol, n_nodes in enumerate(n_nodes_list):
        halfedge_index_this_mol = torch.triu_indices(n_nodes, n_nodes, offset=1)
        halfedge_index.append(halfedge_index_this_mol + idx_start)
        n_edges_this_mol = len(halfedge_index_this_mol[0])
        batch_halfedge.append(np.full(n_edges_this_mol, i_mol))
        idx_start += n_nodes
    
    batch_node = torch.LongTensor(batch_node)
    batch_halfedge = torch.LongTensor(np.concatenate(batch_halfedge))
    halfedge_index = torch.cat(halfedge_index, dim=1)
    
    if device is not None:
        batch_node = batch_node.to(device)
        batch_halfedge = batch_halfedge.to(device)
        halfedge_index = halfedge_index.to(device)
    return {
        # 'n_graphs': n_graphs,
        'batch_node': batch_node,
        'halfedge_index': halfedge_index,
        'batch_halfedge': batch_halfedge,
    }

#
@register_transforms('mixed')
class MixedTransform:
    def __init__(self, config, **kwargs):
        self.config = config
        self.transform_dict = {}
        for task_cfg in config.individual:
            self.transform_dict[task_cfg.name] = get_transforms(task_cfg, **kwargs)

        self.exclude_keys = sum([getattr(t, 'exclude_keys', []) 
                                 for t in self.transform_dict.values()], [])

    def __call__(self, data: Mol3DData):
        task = data['task']
        return self.transform_dict[task](data)

@register_transforms('sbdd')
@register_transforms('denovo')
class DenovoTransform:
    def __init__(self, config, **kwargs):
        self.config = config

        self.mode = mode = kwargs.get('mode', 'test')
        
        
    def __call__(self, data: Mol3DData):
        # set fixed
        data = self.set_fixed(data)

        # fake flexible features
        data = self.set_fake_features(data)

        if self.mode == 'test':
            data = self.prepare_sample(data)
        return data

    def set_fake_features(self, data):
        data.update({
            'n_domain': torch.tensor(0, dtype=torch.long),
            'domain_node_index': torch.empty([2, 0], dtype=torch.long),
            # 'domain_center_nodes': torch.empty([0, 3], dtype=torch.long),
            'tor_bonds_anno': torch.empty([0, 3], dtype=torch.long),
            'twisted_nodes_anno': torch.empty([0, 2], dtype=torch.long),
            'dihedral_pairs_anno': torch.empty([0, 3], dtype=torch.long)
        })
        return data

    def set_fixed(self, data):
        n_node = data['node_type'].shape[0]
        n_halfedge =data['halfedge_type'].shape[0]
        fixed_node, fixed_pos, fixed_halfedge = get_vector_list(
                [n_node, n_node, n_halfedge], [0, 0, 0])
        fixed_distmat = torch.zeros(n_node, n_node, dtype=torch.long)
        fixed_halfdist = fixed_distmat[data['halfedge_index'][0], data['halfedge_index'][1]]
        
        
        data.update({
            'fixed_node': fixed_node,
            'fixed_pos': fixed_pos,
            'fixed_halfedge': fixed_halfedge,
            'fixed_halfdist': fixed_halfdist,
        })
        return data
    
    def prepare_sample(self, data):
        # # make gt : NOTE: when use varible_mol_size, these are not gt
        data['gt_node_type'] = data['node_type'].clone()
        data['gt_node_pos'] = data['node_pos'].clone()
        data['gt_halfedge_type'] = data['halfedge_type'].clone()

        # # make init
        # use init_level to control this
        # data['node_type'] = torch.zeros_like(data['node_type'])
        # data['node_pos'] = torch.zeros_like(data['node_pos'])
        # data['halfedge_type'] = torch.zeros_like(data['halfedge_type'])
        return data


@register_transforms('conf')
@register_transforms('dock')
class ConfTransform:
    def __init__(self, config, **kwargs):
        self.config = config

        self.mode = mode = kwargs.get('mode', 'test')
        self.exclude_keys = ['bond_rotatable', 'tor_twisted_pairs', 'fixed_dist_torsion',
                             'path_mat', 'nbh_dict', 'tor_bond_mat', 'matches_graph', 'matches_iso']
        if mode == 'train':
            self.exclude_keys.extend([
                'task_setting', ])
        
    
        self.settings = (list(config.settings.keys()), list(config.settings.values()))
        assert np.all((s in CONF_SETTINGS) for s in self.settings[0]), f'unknown conf/dock setting {self.settings[0]}'

        self.fix_some = config.get('fix_some', None)
    
    def __call__(self, data: PocketMolData):
        # sample setting
        setting = self.sample_setting()
        if 'is_atom_remain'  in data:  # peptide is cut. only applicable for free setting
            raise ValueError('not supported anymore: is_atom_remain')
            setting = 'free' 
        data.update({'task_setting': setting})

        # set fixed
        data = self.set_fixed(data, setting)
        
        # torsional 
        data = self.set_torsional_feat(data, setting)
        
        # for the sample mode, prepare init data
        if self.mode == 'test':
            data = self.prepare_sample(data, setting)
            
        if self.mode != 'train':
            fixed_distmat_flex = torch.LongTensor(data['fixed_dist_torsion'])
            fixed_halfdist_flex = fixed_distmat_flex[data['halfedge_index'][0], data['halfedge_index'][1]]
            data.update({
                'fixed_halfdist_flex': fixed_halfdist_flex
            })
        
        return data

    def prepare_sample(self, data, setting):
        # # make gt 
        data['gt_node_type'] = data['node_type'].clone()
        data['gt_node_pos'] = data['node_pos'].clone()
        data['gt_halfedge_type'] = data['halfedge_type'].clone()

        # # make init
        setting = data['task_setting']
        # if setting == 'free':  #! no need to remove node_pos. controlled by init_step
        #     data['node_pos'] = torch.zeros_like(data['node_pos'])
        # else:  # torsional and flexible. ~~init is from rdkit~~. Not use rdkit, local is not accurate.
        #     # data = add_rdkit_conf(data=data)
        #     pass
        return data
    
    def set_torsional_feat(self, data: PocketMolData, setting):
        if (setting == 'free' and self.mode == 'train'):
        # if (setting == 'free'):
        # if 'path_mat' not in data:
            data.update({
                'n_domain': torch.tensor(0, dtype=torch.long),
                'domain_node_index': torch.empty([2, 0], dtype=torch.long),
                # 'domain_center_nodes': torch.empty([0, 3], dtype=torch.long),#TODO: remove later
                'tor_bonds_anno': torch.empty([0, 3], dtype=torch.long),
                'twisted_nodes_anno': torch.empty([0, 2], dtype=torch.long),
                'dihedral_pairs_anno': torch.empty([0, 3], dtype=torch.long)
            })
            return data #! can use it for post-processing, even in free mode
        
        # # rigid domain
        # assert setting in ['flexible', 'torsional', 'rigid']
        n_node = data['node_type'].shape[0]
        n_domain = torch.tensor(1, dtype=torch.long)
        domain_node_index = torch.stack([
            torch.zeros(n_node, dtype=torch.long),
            torch.arange(n_node, dtype=torch.long)
        ], dim=0)
        
        
        # # center nodes
        path_mat = data['path_mat']
        nbh_dict = data['nbh_dict']

        margin = path_mat.max(0)
        node_c0 = np.random.choice(np.argwhere(margin == margin.min()).reshape(-1))
        # neigh_c0 = nbh_dict[node_c0]
        # if len(neigh_c0) >= 2:
        #     neigh_margin = margin[neigh_c0]
        #     node_c1, node_c2 = np.random.choice(neigh_c0, 2, replace=False,
        #                                         p=neigh_margin/neigh_margin.sum())
        # else:
        #     raise ValueError('only one node in the domain')
        # domain_center_nodes = torch.tensor([[node_c0, node_c1, node_c2]], dtype=torch.long)

        if setting == 'rigid':
            data.update({
                'n_domain': torch.tensor(1, dtype=torch.long),
                'domain_node_index': torch.stack([
                    torch.zeros(data.num_nodes, dtype=torch.long),
                    torch.arange(data.num_nodes, dtype=torch.long)
                ], dim=0),
                # 'domain_center_nodes': domain_center_nodes,
                'tor_bonds_anno': torch.empty([0, 3], dtype=torch.long),
                'twisted_nodes_anno': torch.empty([0, 2], dtype=torch.long),
                'dihedral_pairs_anno': torch.empty([0, 3], dtype=torch.long)
            })
            return data
        
        tor_bond_mat = data['tor_bond_mat']
        n_tor_bonds = tor_bond_mat.sum() / 2
        if n_tor_bonds == 0:
            tor_bonds_anno = torch.empty([0, 3], dtype=torch.long)
            twisted_nodes_anno = torch.empty([0, 2], dtype=torch.long)
            dihedral_pairs_anno = torch.empty([0, 3], dtype=torch.long)
        else:
            # # torsional bonds
            # BFS+DFS to determine the order of the torsional bonds and the trees
            global_remain = np.ones(n_node, dtype=bool)
            for node, nbh in nbh_dict.items():
                if len(nbh) == 1:
                    global_remain[node] = False  # trick 1: frontier node no need to visit
            curr_order = 0
            query_pool = [node_c0]
            tor_bonds_anno = []
            early_stop = False  # trick 2: if all torsional bonds are found, stop
            while global_remain.any():
                #print('in loop 523')
                next_query = []
                while len(query_pool) > 0:
                    #print('in loop 526')
                    # mark the node as visited
                    curr_node = query_pool.pop(0)
                    global_remain[curr_node] = False
                    # add neighbors to query pool
                    nbh = nbh_dict[curr_node]
                    for nb_node in nbh:
                        if (not global_remain[nb_node]) or (nb_node in query_pool):
                            pass  # already visited or in query pool
                        elif tor_bond_mat[curr_node, nb_node] == 1: # find a torsional bond
                            tor_bonds_anno.append([curr_order, nb_node, curr_node])
                            next_query.append(nb_node)
                            if len(tor_bonds_anno) == n_tor_bonds:
                                early_stop = True
                                break
                        else:  # continue to search
                            query_pool.append(nb_node)
                    if early_stop:
                        break
                if early_stop:
                    break
                curr_order += 1
                query_pool = next_query

            # # twisted nodes
            tor_twisted_pairs = data['tor_twisted_pairs']
            twisted_nodes_anno = []
            for index_tor, (_, tor_left, tor_right) in enumerate(tor_bonds_anno):
                if tor_left < tor_right:
                    all_nodes_left = tor_twisted_pairs[(tor_left, tor_right)][0]
                else:
                    all_nodes_left = tor_twisted_pairs[(tor_right, tor_left)][1]
                assert (node_c0 not in all_nodes_left), 'center node c0 should not be in the twisted nodes'
                # assert (node_c1 not in all_nodes_left), 'center node c1 should not be in the twisted nodes'
                # assert (node_c2 not in all_nodes_left), 'center node c2 should not be in the twisted nodes'
                
                twisted_nodes_anno.extend([index_tor, node] for node in all_nodes_left)
            twisted_nodes_anno = torch.tensor(twisted_nodes_anno, dtype=torch.long)  # (n_twisted, 2)
            
            # # dihedral pairs
            dihedral_pairs_anno = []
            for index_tor, (_, tor_left, tor_right) in enumerate(tor_bonds_anno):
                nbh_left = [n for n in nbh_dict[tor_left] if n != tor_right]
                nbh_right = [n for n in nbh_dict[tor_right] if n != tor_left]
                dihedral_pairs_anno.extend([index_tor, node_left, node_right]
                                        for node_left, node_right in product(nbh_left, nbh_right))
            dihedral_pairs_anno = torch.tensor(dihedral_pairs_anno, dtype=torch.long)  # (n_dihedral, 3)

            tor_bonds_anno = torch.tensor(tor_bonds_anno, dtype=torch.long)
            
        # # combine
        data.update({
            'n_domain': n_domain,
            'domain_node_index': domain_node_index,
            # 'domain_center_nodes': domain_center_nodes,
            'tor_bonds_anno': tor_bonds_anno,
            'twisted_nodes_anno': twisted_nodes_anno,
            'dihedral_pairs_anno': dihedral_pairs_anno,
        })
            
        return data

    def sample_setting(self):
        setting = np.random.choice(self.settings[0], p=self.settings[1])
        return setting

    def set_fixed(self, data, setting):
        n_node = n_pos = data['node_type'].shape[0]
        n_halfedge =data['halfedge_type'].shape[0]
        fixed_node, fixed_pos, fixed_halfedge = get_vector_list(
                [n_node, n_pos, n_halfedge], [1, 0, 1])  # pos is not fixed
        data.update({
            'fixed_node': fixed_node,
            'fixed_pos': fixed_pos,
            'fixed_halfedge': fixed_halfedge,
        })
        
        # for fixed dist
        if setting == 'free':
            fixed_distmat = torch.zeros(n_node, n_node, dtype=torch.long)
        elif (setting == 'flexible') or (setting == 'torsional'):
            fixed_distmat = torch.LongTensor(data['fixed_dist_torsion'])
        elif setting == 'rigid':
            fixed_distmat = torch.ones(n_node, n_node, dtype=torch.long)
        fixed_halfdist = fixed_distmat[data['halfedge_index'][0], data['halfedge_index'][1]]
        data.update({
            'fixed_halfdist': fixed_halfdist,
        })
        
        if self.fix_some is not None:
            if self.fix_some == 'endres':  # fix the end residue, they cannot be moved
                peptide_res_index = data['peptide_res_index']
                is_endres = (peptide_res_index == 0) | (peptide_res_index == peptide_res_index.max())
                fixed_pos[is_endres] = 1
            elif self.fix_some == 'endresbb':  # fix the end residue bb., they cannot be moved
                peptide_res_index = data['peptide_res_index']
                is_endres = (peptide_res_index == 0) | (peptide_res_index == peptide_res_index.max())
                is_backbone = data['peptide_is_backbone']
                is_endresbb = is_endres & is_backbone
                fixed_pos[is_endresbb] = 1
            elif self.fix_some == 'bb':
                is_backbone = data['peptide_is_backbone']
                fixed_pos[is_backbone] = 1
            elif self.fix_some == 'firstres':
                peptide_res_index = data['peptide_res_index']
                is_first = (peptide_res_index == 0)
                fixed_pos[is_first] = 1
            else:
                raise NotImplementedError(f'unknown fix_some {self.fix_some}')
            data.update({
                'fixed_pos': fixed_pos,
            })
        
        return data


@register_transforms('fbdd')
@register_transforms('maskfill')
class MaskfillTransform:
    def __init__(self, config, **kwargs) -> None:
        self.config = config
        self.mode = mode = kwargs.get('mode', 'test')
        self.exclude_keys = ['linkers', 'anchors_linkers', 'frags', 'anchors_frags',
                             'is_known_halfedge_p1p2', 
                             'bond_rotatable', 'tor_twisted_pairs', 'fixed_dist_torsion',
                             'path_mat', 'nbh_dict', 'tor_bond_mat', 'mmpa', 'brics', 'matches_graph', 'matches_iso']
        if mode == 'train':
            self.exclude_keys.extend([
                'task_setting',
                'node_p1', 'node_p2', 'halfedge_p1', 'halfedge_p2', 'halfedge_p1p2', 'groupof_node_p1',
                'grouped_node_p1', 'grouped_anchor_p1'])
            
        self.preset_partition = config.get('preset_partition', None)
        
        self.settings_dict = {}
        if 'settings' in config:
            for key, value_dict in config.settings.items():
                self.settings_dict[key] = {'options': list(value_dict.keys()), 'weights': list(value_dict.values())}
                assert all(op in MASKFILL_SETTINGS[key] for op in self.settings_dict[key]['options']),\
                        f"unknown maskfill setting {self.settings_dict[key]} for setting {key}"
    
    def __call__(self, data: Mol3DData):
        setting = self.sample_setting()
        data.update({'task_setting': setting})
        
        # make partiotion
        data = self.pre_make_partition(data)
        data, num_part_dict = self.make_partition_features(data)
        # set features
        data = self.set_fixed(data, setting, num_part_dict)
        data = self.set_torsional_feat(data, fake=(setting['part1_pert'] != 'flexible'))

        if self.mode == 'test':
            data = self.prepare_sample(data, setting)
        return data

    def prepare_sample(self, data, setting):
        # # make gt
        data['gt_node_type'] = data['node_type'].clone()
        data['gt_node_pos'] = data['node_pos'].clone()
        data['gt_halfedge_type'] = data['halfedge_type'].clone()
        
        # # make gt for part 1
        node_type = data['node_type']
        node_pos = data['node_pos']
        halfege_type = data['halfedge_type']
        
        for i_part in [1, 2]:
            node_type_part = - torch.ones_like(node_type)
            node_type_part[data[f'node_p{i_part}']] = node_type[data[f'node_p{i_part}']].clone()
            node_pos_part = node_pos.clone()
            halfedge_type_part = - torch.ones_like(halfege_type)
            halfedge_type_part[data[f'halfedge_p{i_part}']] = halfege_type[data[f'halfedge_p{i_part}']].clone()
            data.update({
                f'gt_node_type_p{i_part}': node_type_part,
                f'gt_node_pos_p{i_part}': node_pos_part,
                f'gt_halfedge_type_p{i_part}': halfedge_type_part,
            })
        
        # # make init
        if setting['part1_pert'] in ['fixed', 'free', 'small', 'rigid', 'flexible']:
            data['node_type'][data['node_p2']] = 0
            # for p1 fixed (not has pocket), influence center before add noise
            if data['node_p1'].shape[0] > 0:
                data['node_pos'][data['node_p2']] = (data['node_pos'][data['node_p1']]).mean(dim=0)
            else:
                data['node_pos'] = torch.zeros_like(data['node_pos'])
            data['halfedge_type'][data['halfedge_p2']] = 0
            data['halfedge_type'][data['halfedge_p1p2']] = 0
        else:
            raise NotImplementedError
        return data

    def sample_setting(self):
        setting_dict = {}
        for setting, opt_dict in self.settings_dict.items():
            setting_dict[setting] = np.random.choice(opt_dict['options'], p=opt_dict['weights'])
        return setting_dict
    
    def set_fixed(self, data: Mol3DData, setting, num_part_dict):
        n_node_p1, n_node_p2 = num_part_dict['n_node_p1'], num_part_dict['n_node_p2']
        n_halfedge_p1, n_halfedge_p2 = num_part_dict['n_halfedge_p1'], num_part_dict['n_halfedge_p2']
        n_halfedge_p1p2 = num_part_dict['n_halfedge_p1p2']
        
        # # for part 1
        if setting['part1_pert'] == 'fixed':
            fixed_node_p1, fixed_pos_p1, fixed_halfedge_p1 = get_vector_list(
                [n_node_p1, n_node_p1, n_halfedge_p1], [1, 1, 1])
        elif setting['part1_pert'] == 'small':
            fixed_node_p1, fixed_pos_p1, fixed_halfedge_p1 = get_vector_list(
                [n_node_p1, n_node_p1, n_halfedge_p1], [0, 0, 0])
        elif setting['part1_pert'] in ['free', 'rigid', 'flexible']:
            fixed_node_p1, fixed_pos_p1, fixed_halfedge_p1 = get_vector_list( # pos is not fixed
                [n_node_p1, n_node_p1, n_halfedge_p1], [1, 0, 1])
        else:
            raise ValueError(f"Unknown part1_pert: {setting['part1_pert']}")
        
        # # for connections, use anchor or not
        is_known_halfedge_p1p2 = data['is_known_halfedge_p1p2']
        fixed_halfedge_p1p2 = get_vector([n_halfedge_p1p2], 0)
        fixed_halfedge_p1p2[is_known_halfedge_p1p2] = 1
        
        # # part 2
        fixed_node_p2, fixed_pos_p2, fixed_halfedge_p2 = get_vector_list(
                [n_node_p2, n_node_p2, n_halfedge_p2], [0, 0, 0])

        fixed_dict = {
            'node_p1': fixed_node_p1,
            'pos_p1': fixed_pos_p1,
            'halfedge_p1': fixed_halfedge_p1,
            'node_p2': fixed_node_p2,
            'pos_p2': fixed_pos_p2,
            'halfedge_p2': fixed_halfedge_p2,
            'halfedge_p1p2': fixed_halfedge_p1p2,
        }
        
        # # combine p1 and p2
        fixed_node = combine_vectors_indexed(
            [fixed_dict[f'node_p1'], fixed_dict[f'node_p2']],
            [data['node_p1'], data['node_p2']],
        )
        fixed_pos = combine_vectors_indexed(
            [fixed_dict[f'pos_p1'], fixed_dict[f'pos_p2']],
            [data['node_p1'], data['node_p2']],
        )
        fixed_halfedge = combine_vectors_indexed(
            [fixed_dict[f'halfedge_p1'], fixed_dict[f'halfedge_p2'], fixed_dict[f'halfedge_p1p2']],
            [data['halfedge_p1'], data['halfedge_p2'], data['halfedge_p1p2']],
        )
        data.update({
            'fixed_node': fixed_node,
            'fixed_pos': fixed_pos,
            'fixed_halfedge': fixed_halfedge,
        })
        
        # # for fixed_dist
        if setting['part1_pert'] == 'rigid':
            domain_node_index = torch.stack([
                data['groupof_node_p1'], data['node_p1']
            ], dim=0)
            n_domain = domain_node_index[0].max() + 1 if domain_node_index.shape[1] > 0 else torch.tensor(0, dtype=torch.long)
            fixed_distmat = get_rigid_distmat(domain_node_index[0], domain_node_index[1],
                                              n_domain, data['node_type'].shape[0])
            fixed_halfdist = fixed_distmat[
                data['halfedge_index'][0], data['halfedge_index'][1]]
        elif setting['part1_pert'] == 'flexible':
            domain_node_index = torch.stack([
                data['groupof_node_p1'], data['node_p1']
            ], dim=0)
            n_domain = domain_node_index[0].max() + 1 if domain_node_index.shape[1] > 0 else torch.tensor(0, dtype=torch.long)
            # n_domain = domain_node_index[0].max() + 1
            fixed_distmat = get_rigid_distmat(domain_node_index[0], domain_node_index[1],
                                              n_domain, data['node_type'].shape[0])
            fixed_dist_torsion = torch.LongTensor(data['fixed_dist_torsion'])
            fixed_distmat = torch.where(fixed_dist_torsion==0, fixed_dist_torsion, fixed_distmat)
            fixed_halfdist = fixed_distmat[
                data['halfedge_index'][0], data['halfedge_index'][1]]
        else:  # free, small, fixed
            domain_node_index = torch.empty([2, 0], dtype=torch.long)  # set rigid domain to empty
            n_domain = torch.tensor(0, dtype=torch.long)
            fixed_halfdist = torch.zeros_like(data['halfedge_type'], dtype=torch.long)  # default not fixed distances
            if setting['part1_pert'] == 'fixed':  # fixed distances among part1 nodes
                fixed_halfdist[data['halfedge_p1']] = 1
        data.update({
            'fixed_halfdist': fixed_halfdist,
            'n_domain': n_domain,
            'domain_node_index': domain_node_index,
        })

        return data
    
    def set_torsional_feat(self, data: Mol3DData, fake=False):
        domain_node_index = data['domain_node_index']
        domain_index = domain_node_index[0]
        
        if fake or (domain_index.shape[0] == 0):
            data.update({
                'tor_bonds_anno': torch.empty([0, 3], dtype=torch.long),
                'twisted_nodes_anno': torch.empty([0, 2], dtype=torch.long),
                'dihedral_pairs_anno': torch.empty([0, 3], dtype=torch.long),
            })
            return data
        
        # # rigid domain
        n_domain = domain_index.max() + 1
        n_node = data['node_type'].shape[0]
        domain_index_of_node = np.full((n_node,), -1)
        domain_index_of_node[domain_node_index[1].numpy()] = domain_node_index[0].numpy()
        
        # # center nodes
        path_mat = data['path_mat']
        nbh_dict = data['nbh_dict']
        
        fixed_distmat = get_rigid_distmat(domain_node_index[0], domain_node_index[1],
                                              n_domain, n_node)
        path_mat_domain = path_mat * fixed_distmat.numpy()
        margin = path_mat_domain.max(0)
        node_c0_list = []
        for i_domain in range(n_domain):
            margin_domain = margin[domain_index_of_node == i_domain]
            idx_domain = np.argmin(margin_domain)
            node_c0 = np.argwhere(domain_index_of_node == i_domain).flatten()[idx_domain]
            node_c0_list.append(node_c0)
        
        tor_bond_mat = data['tor_bond_mat']
        n_tor_bonds = tor_bond_mat.sum() / 2

        # # torsional bonds
        node_p1, node_p2 = data['node_p1'].numpy(), data['node_p2'].numpy()
        global_remain = np.ones(n_node, dtype=bool)
        global_remain[node_p2] = False
        for node in node_p1:
            domain_this = domain_index_of_node[node]
            nbh = nbh_dict[node]
            if len([n for n in nbh if domain_index_of_node[n] == domain_this]) == 1:
                global_remain[node] = False  # trick 1: frontier node no need to visit
            if len([n for n in nbh if domain_index_of_node[n] == domain_this]) == 0:
                global_remain[node] = False  # node in part1 has no neighbor in part1
        curr_order = 0
        query_pool = node_c0_list
        tor_bonds_anno = []
        early_stop = False
        n_tor_bonds = (tor_bond_mat[global_remain][:, global_remain]).sum() / 2
        while global_remain.any():
            #print('in loop 844')
            next_query = []
            while len(query_pool) > 0:
                #print('in loop 847')
                # mark the node as visited
                curr_node = query_pool.pop(0)
                global_remain[curr_node] = False
                domain_curr = domain_index_of_node[curr_node]
                # add neighbors IN p1 to query pool
                nbh = nbh_dict[curr_node]
                for nb_node in nbh:
                    if domain_index_of_node[nb_node] != domain_curr:
                        continue  # not in the same domain
                    if (not global_remain[nb_node]) or (nb_node in query_pool):
                        pass # already visited or in query pool
                    elif tor_bond_mat[curr_node, nb_node] == 1: # find a torsional bond
                        tor_bonds_anno.append([curr_order, nb_node, curr_node])
                        next_query.append(nb_node)
                        if len(tor_bonds_anno) == n_tor_bonds:
                            early_stop = True
                            break
                    else:  # continue to search
                        query_pool.append(nb_node)
                if early_stop:
                    break
            if early_stop:
                break
            curr_order += 1
            query_pool = next_query
            
        if len(tor_bonds_anno) == 0:
            # no tor bond
            data.update({
                'tor_bonds_anno': torch.empty([0, 3], dtype=torch.long),
                'twisted_nodes_anno': torch.empty([0, 2], dtype=torch.long),
                'dihedral_pairs_anno': torch.empty([0, 3], dtype=torch.long),
            })
            return data
        # # twisted nodes
        tor_twisted_pairs = data['tor_twisted_pairs']
        twisted_nodes_anno = []
        for index_tor, (_, tor_left, tor_right) in enumerate(tor_bonds_anno):
            if tor_left < tor_right:
                all_nodes_left = tor_twisted_pairs[(tor_left, tor_right)][0]
            else:
                all_nodes_left = tor_twisted_pairs[(tor_right, tor_left)][1]
            
            # select from the same domain
            domain_tor = domain_index_of_node[tor_left]
            assert (domain_index_of_node[tor_right] == domain_tor), 'tor_bond node should be in the same domain'
            all_nodes_left = np.array(list(all_nodes_left))
            all_nodes_left = all_nodes_left[domain_index_of_node[all_nodes_left] == domain_tor]
            
            twisted_nodes_anno.extend([index_tor, node] for node in all_nodes_left)
        twisted_nodes_anno = torch.tensor(twisted_nodes_anno, dtype=torch.long)  # (n_twisted, 2)

        # # dihedral pairs
        dihedral_pairs_anno = []
        for index_tor, (_, tor_left, tor_right) in enumerate(tor_bonds_anno):
            domain_tor = domain_index_of_node[tor_left]
            nbh_left = [n for n in nbh_dict[tor_left] if (n != tor_right) and (domain_index_of_node[n] == domain_tor)]
            nbh_right = [n for n in nbh_dict[tor_right] if (n != tor_left) and (domain_index_of_node[n] == domain_tor)]
            dihedral_pairs_anno.extend([index_tor, node_left, node_right]
                    for node_left, node_right in product(nbh_left, nbh_right))
        dihedral_pairs_anno = torch.tensor(dihedral_pairs_anno, dtype=torch.long)  # (n_dihedral, 3)

        tor_bonds_anno = torch.tensor(tor_bonds_anno, dtype=torch.long)

        data.update({
            'tor_bonds_anno': tor_bonds_anno,
            'twisted_nodes_anno': twisted_nodes_anno,
            'dihedral_pairs_anno': dihedral_pairs_anno,
        })
        return data

    def graph_to_tree_order(self, nbh_list, size_tree):
        n = len(nbh_list)
        tree_order = []
        pool = [random.randint(0, n-1)] # 0, 1, ..., n-1
        while len(tree_order) < size_tree:  # only explore size_tree nodes
            #print('in loop 924')
            curr = np.random.choice(pool)
            tree_order.append(curr)
            pool.remove(curr)
            pool.extend([nb for nb in nbh_list[curr] if
                         (nb not in tree_order) and (nb not in pool)])
        assert len(tree_order) == size_tree, 'not enough nodes in the tree order'
        assert len(set(tree_order)) == size_tree, 'tree order has duplicate nodes'
        return np.array(tree_order)
    
    def partition_from_preset(self, data: Mol3DData):
        preset_partition = self.preset_partition
        n_nodes = data['node_type'].shape[0]
        index_nodes = np.arange(n_nodes)
        
        grouped_node_p1 = preset_partition.get('grouped_node_p1', None)
        node_p2 = preset_partition.get('node_p2', None)
        assert (grouped_node_p1 is not None) or (node_p2 is not None), 'grouped_node_p1 or node_p2 should be set'
        
        if grouped_node_p1 is None:
            assert node_p2 is not None, 'node_p2 should be not None'
            node_p1 = [n for n in index_nodes if n not in node_p2]
            grouped_node_p1 = [node_p1]
        if node_p2 is None:
            assert grouped_node_p1 is not None, 'grouped_node_p1 should not be None'
            node_p1 = sum(grouped_node_p1, [])
            node_p2 = [n for n in index_nodes if n not in node_p1]
        #TODO: print partition
        grouped_anchor_p1 = preset_partition.get('grouped_anchor_p1', None)
        if grouped_anchor_p1 is None:
            grouped_anchor_p1 = [[] for _ in grouped_node_p1]
            
        data.update({
            'grouped_node_p1': grouped_node_p1,
            'grouped_anchor_p1': grouped_anchor_p1,
            'node_p2': node_p2,
        })
        return data
        

        
    def pre_make_partition(self, data: Mol3DData):
        if self.preset_partition is not None:  # set from config file. for use
            return self.partition_from_preset(data)

        n_nodes = data['node_type'].shape[0]
        setting = data['task_setting']
        setting_decom = setting['decomposition']
        setting_order = setting['order']
        
        # # subgraph neighborhood
        if setting_decom in ['brics', 'mmpa']:
            decom_dict = data[setting_decom]
            nbh_subgraphs = decom_dict['nbh_subgraphs']
        else:  # atoms
            nbh_subgraphs = data['nbh_dict']  # fake. an atom is a subgraph
            assert n_nodes == len(nbh_subgraphs), 'atom nbh info is not complete'
        n_subgraphs = len(nbh_subgraphs)
        
        # # determine subgraphs of p1 and p2
        if n_subgraphs <= 1:
            n_p1 = 0
        else:
            n_p1 = np.random.randint(1, n_subgraphs)  # [0, 1, ..., n_subgraphs-1]
        n_p2 = n_subgraphs - n_p1
        assert n_p2 > 0, 'n_p2 should be positive'
        
        # # make orders
        if setting_order != 'random':  # tree or inv_tree
            if setting_order == 'tree':
                subgraph_p1 = self.graph_to_tree_order(nbh_subgraphs, size_tree=n_p1)
            elif  setting_order == 'inv_tree':
                subgraph_p2 = self.graph_to_tree_order(nbh_subgraphs, size_tree=n_p2)
                subgraph_p1 = np.array([i for i in range(n_subgraphs) if i not in subgraph_p2])
        else:  # random
            subgraph_p1 = np.random.permutation(n_subgraphs)[:n_p1]

        # # combine subgraphs in p1 and p2 separately
        subgraph_part = subgraph_p1.tolist()
        domain_list = []
        while len(subgraph_part) > 0:
            #print('in loop 967')
            root = subgraph_part.pop(0)
            to_explore = [root]
            curr_domain = [root]
            while len(to_explore) > 0:
                #print('in loop 972')
                curr_sb = to_explore.pop(0)
                nbh_sbs = nbh_subgraphs[curr_sb]
                for nbh in nbh_sbs:
                    #print('in loop 981')
                    if (nbh in subgraph_part) and (nbh not in curr_domain) and (nbh not in to_explore):
                        curr_domain.append(nbh)
                        subgraph_part.remove(nbh)
                        if len(nbh_subgraphs[nbh]) > 1:  # not a leaf
                            to_explore.append(nbh)
            domain_list.append(curr_domain)
        
        # for curr_subgraph in subgraph_part:
        #     in_domain = [(curr_subgraph in domain) for domain in domain_list]
        #     if any(in_domain):
        #         domain = domain_list[in_domain.index(True)]
        #     else:
        #         domain = []
        #         domain_list.append(domain)
        #     domain.append(curr_subgraph)
        #     # nbh_curr = nbh_subgraphs[curr_subgraph]
        #     # domain.extend([nb for nb in nbh_curr if
        #     #         (nb not in domain) and (nb in subgraph_part)])
        # subgraph to nodes
        if setting_decom != 'atom':
            subgraphs_to_nodes = decom_dict['subgraphs']
            domain_list = [sum([subgraphs_to_nodes[sb] for sb in subgraphs], [])
                                for subgraphs in domain_list]
        grouped_node_p1 = domain_list
        node_p1 = sum(grouped_node_p1, [])
        node_p2 = [n for n in range(n_nodes) if n not in node_p1]
        assert len(node_p1) + len(node_p2) == n_nodes, 'nodes are not partitioned'
        
        # # get anchors
        in_p1 = np.zeros(n_nodes, dtype=bool)
        in_p1[node_p1] = True
        bond_index = data['bond_index']
        inter_domain = (in_p1[bond_index[0]] != in_p1[bond_index[1]])
        anchors = set(bond_index[:, inter_domain].numpy().flatten())
        grouped_anchor_p1 = [list(anchors & set(grouped_node_p1[i])) for i in range(len(grouped_node_p1))]
        
        data.update({
            'grouped_node_p1': grouped_node_p1,
            'grouped_anchor_p1': grouped_anchor_p1,
            'node_p2': node_p2,
        })
        return data


    def make_partition_features(self, data: Mol3DData):
        """
        Get the partition for fragment_linking task.
        Output is a dict containing:
        - node_p1
        - node_p2
        - halfedge_p1
        - halfedge_p2
        - halfedge_p1p2
        - grouped_halfedge_nonanchors_in_p1p2:
                List: the index of halfedge related to non-anchor nodes of p1 in halfedge_p1p2
                }
        """
        # # sample a separation from all possible separations of frag-linker
        seperation = data  # ['linking']
        
        # # get frags (p1) and linkers (p2)
        grouped_node_p1 = seperation['grouped_node_p1']
        grouped_anchor_p1 = seperation['grouped_anchor_p1']
        node_p1 = sum(grouped_node_p1, [])
        node_p2 = seperation['node_p2']
        # anchor_p2 = data['anchors_linkers'][idx_sep]
        n_frags = len(grouped_node_p1)
        
        assert len(node_p1) == len(np.unique(node_p1)), 'node_p1 has duplicate nodes'
        assert len(node_p2) == len(np.unique(node_p2)), 'node_p1 has duplicate nodes'
        assert len(set(node_p1) & set(node_p2)) == 0, 'node_p1 and node_p2 have common nodes'
        assert len(set(node_p1) | set(node_p2)) == data['node_type'].shape[0], 'node_p1 and node_p2 are not partitioned'
        
        node_p1 = torch.LongTensor(node_p1)
        groupof_node_p1 = torch.LongTensor([i for i, frag in enumerate(grouped_node_p1) for _ in frag])
        node_p2 = torch.LongTensor(node_p2)
        halfedge_index = data.halfedge_index
        if halfedge_index.shape[1] != 0:
            i_all_halfedge = torch.arange(halfedge_index.shape[1], dtype=torch.long)
            halfedge_p1 = subgraph(node_p1, halfedge_index, i_all_halfedge)[1]
            halfedge_p2 = subgraph(node_p2, halfedge_index, i_all_halfedge)[1]
        
            i_all_halfedge[halfedge_p1] = -1
            i_all_halfedge[halfedge_p2] = -1
            halfedge_p1p2 = torch.nonzero(i_all_halfedge >= 0, as_tuple=False).squeeze()
            # another way
            edge_index, i_all_edge = to_undirected(halfedge_index, i_all_halfedge)
            halfedge_p1p2_way2 = bipartite_subgraph([node_p1, node_p2], edge_index, i_all_edge)[1]
            assert (halfedge_p1p2.sort()[0] == halfedge_p1p2_way2.sort()[0]).all(), 'two ways of getting halfedge_p1p2 are different'
        else:
            halfedge_p1 = torch.empty([0], dtype=torch.long)
            halfedge_p2 = torch.empty([0], dtype=torch.long)
            halfedge_p1p2 = torch.empty([0], dtype=torch.long)
        
        # # get halfedge_nonanchors_in_p1p2_grouped
        setting = data['task_setting']
        known_anchor = setting['known_anchor']
        if known_anchor == 'none':  # all edges between p1 and p2 are unknown
            is_known_halfedge_p1p2 = torch.zeros(halfedge_p1p2.shape[0], dtype=torch.bool)
        elif known_anchor == 'partial':
            if n_frags <= 1:
                n_known_frag = 0
            else:
                n_known_frag = np.random.randint(1, n_frags)
            node_known_outer_edge = []
            for i_known_frag in np.random.choice(n_frags, n_known_frag, replace=False):
                node_this_frag = grouped_node_p1[i_known_frag]
                anchor_this_frag = grouped_anchor_p1[i_known_frag]
                node_known_outer_edge.extend([n for n in node_this_frag if n not in anchor_this_frag])
            node_known_outer_edge = torch.tensor(node_known_outer_edge, dtype=torch.long)
            halfedge_index_p1p2 = halfedge_index[:, halfedge_p1p2]
            is_known_halfedge_p1p2 = (halfedge_index_p1p2[..., None] == node_known_outer_edge).any(-1).any(0)
        elif known_anchor == 'all':
            # is_known_halfedge_p1p2 = torch.ones(halfedge_p1p2.shape[0], dtype=torch.bool)
            node_anchor = torch.tensor(sum(grouped_anchor_p1, []))
            halfedge_index_p1p2 = halfedge_index[:, halfedge_p1p2]
            is_known_halfedge_p1p2 = (halfedge_index_p1p2[..., None] != node_anchor).all(-1).all(0)
        else:
            raise ValueError(f'unknown known_anchor value: {known_anchor}')
        
        partition = {
            'node_p1': node_p1,
            'node_p2': node_p2,
            'halfedge_p1': halfedge_p1,
            'halfedge_p2': halfedge_p2,
            'halfedge_p1p2': halfedge_p1p2,
            'groupof_node_p1': groupof_node_p1,
            'is_known_halfedge_p1p2': is_known_halfedge_p1p2,
        }
        
        # # summary partition
        num_part_dict = {f'n_{key}': partition[key].shape[-1] for key in 
            ['node_p1', 'node_p2', 'halfedge_p1', 'halfedge_p2', 'halfedge_p1p2']}
        # santiy check
        assert num_part_dict['n_node_p1'] + num_part_dict['n_node_p2'] == data.node_type.shape[0]
        assert (num_part_dict['n_halfedge_p1'] + num_part_dict['n_halfedge_p2']
                + num_part_dict['n_halfedge_p1p2'] == data.halfedge_type.shape[0])

        data.update(partition)
        return data, num_part_dict


@register_transforms('pepdesign')
class PepdesignTransform:
    def __init__(self, config, **kwargs) -> None:
        self.config = config
        self.mode = mode = kwargs.get('mode', 'test')
        self.exclude_keys = ['peptide_pos', 'peptide_atom_name', 'peptide_is_backbone',
                            'peptide_res_id', 'peptide_atom_to_aa_type', 'peptide_res_index',
                            'peptide_seq', 'peptide_pep_len']
        if mode == 'train':
            self.exclude_keys.extend([
                'task_setting',
                'node_bb', 'node_sc', 'halfedge_bb', 'halfedge_sc', 'halfedge_bbsc', 'is_known_halfedge_bbsc'
            ])
            
        # variable size during training
        self.add_mask_atoms = config.get('add_mask_atoms', False)
        self.num_node_types = kwargs.get('num_node_types', None)
        
        self.settings_dict = {}
        if 'settings' in config:
            for key, value_dict in config.settings.items():
                self.settings_dict[key] = {'options': list(value_dict.keys()), 'weights': list(value_dict.values())}
                assert all(op in PEPDESIGN_SETTINGS[key] for op in self.settings_dict[key]['options']),\
                        f"unknown maskfill setting {self.settings_dict[key]} for setting {key}"
    
    def __call__(self, data: Mol3DData):
        setting = self.sample_setting()
        if data['db'] == 'pepbdb' and 'X' in data['peptide_seq'] and self.mode != 'test':
            setting['mode'] = 'packing'  # nonstd peptide not for training generation
        data.update({'task_setting': setting})
        # change the peptide indicator
        if setting['mode'] in ['full', 'sc']:
            data['is_peptide'] = torch.ones_like(data['is_peptide'])
        
            if self.add_mask_atoms and self.mode != 'test':
                data = self.add_atoms(data)
        
        # make partiotion
        data = self.pre_make_partition(data)
        data, num_part_dict = self.make_partition_features(data)
        # set features
        data = self.set_fixed(data, setting, num_part_dict)
        data = self.set_torsional_feat(data)

        if self.mode == 'test':
            data = self.prepare_sample(data, setting)
        return data

    def prepare_sample(self, data, setting):
        # # make gt: actually is not used, for pepdesign, directly copy the gt file. just for compatibility
        data['gt_node_type'] = data['node_type'].clone()  
        data['gt_node_pos'] = data['node_pos'].clone()
        data['gt_halfedge_type'] = data['halfedge_type'].clone()
        
        # # make gt for parts
        node_type = data['node_type']
        node_pos = data['node_pos']
        halfege_type = data['halfedge_type']
        
        for i_part in ['bb', 'sc']:
            node_type_part = - torch.ones_like(node_type)
            node_type_part[data[f'node_{i_part}']] = node_type[data[f'node_{i_part}']].clone()
            node_pos_part = node_pos.clone()
            halfedge_type_part = - torch.ones_like(halfege_type)
            halfedge_type_part[data[f'halfedge_{i_part}']] = halfege_type[data[f'halfedge_{i_part}']].clone()
            data.update({
                f'gt_node_type_{i_part}': node_type_part,
                f'gt_node_pos_{i_part}': node_pos_part,
                f'gt_halfedge_type_{i_part}': halfedge_type_part,
            })
        
        # # make init
        data['node_pos'][data['node_sc']] = 0
        if setting['mode'] == 'packing':
            pass
        elif setting['mode'] in ['sc', 'full']:
            data['node_type'][data['node_sc']] = 0
            data['halfedge_type'][data['halfedge_sc']] = 0
            data['halfedge_type'][data['halfedge_bbsc']] = 0
            if setting['mode'] == 'full':
                data['node_pos'][data['node_bb']] = 0
            
        return data

    def sample_setting(self):
        setting_dict = {}
        for setting, opt_dict in self.settings_dict.items():
            setting_dict[setting] = np.random.choice(opt_dict['options'], p=opt_dict['weights'])
        return setting_dict
    
    def set_fixed(self, data: Mol3DData, setting, num_part_dict):
        n_node_bb, n_node_sc = num_part_dict['n_node_bb'], num_part_dict['n_node_sc']
        n_halfedge_bb, n_halfedge_sc = num_part_dict['n_halfedge_bb'], num_part_dict['n_halfedge_sc']
        n_halfedge_bbsc = num_part_dict['n_halfedge_bbsc']
        
        # # bb graph is always fixed, default all others are not fixed
        fixed_node_bb, fixed_pos_bb, fixed_halfedge_bb = get_vector_list(
            [n_node_bb, n_node_bb, n_halfedge_bb], [1, 0, 1])
        fixed_node_sc, fixed_pos_sc, fixed_halfedge_sc, fixed_halfedge_bbsc = get_vector_list(
            [n_node_sc, n_node_sc, n_halfedge_sc, n_halfedge_bbsc], [0, 0, 0, 0])
        # for connections, CA as anchors, other bb atoms' edges with sc are known
        is_known_halfedge_bbsc = data['is_known_halfedge_bbsc']
        fixed_halfedge_bbsc[is_known_halfedge_bbsc] = 1

        # # mode-specific
        if setting['mode'] == 'full':
            pass
        else:
            fixed_pos_bb = get_vector([n_node_bb], 1)  # bb pos is fixed
            if setting['mode'] == 'sc':  
                pass
            elif setting['mode'] == 'packing':  # sc graph are fixed
                fixed_node_sc, fixed_halfedge_sc, fixed_halfedge_bbsc = get_vector_list(
                    [n_node_sc, n_halfedge_sc, n_halfedge_bbsc], [1, 1, 1])
            else:
                raise ValueError(f"Unknown mode: {setting['mode']}")
        

        fixed_dict = {
            'node_bb': fixed_node_bb,
            'pos_bb': fixed_pos_bb,
            'halfedge_bb': fixed_halfedge_bb,
            'node_sc': fixed_node_sc,
            'pos_sc': fixed_pos_sc,
            'halfedge_sc': fixed_halfedge_sc,
            'halfedge_bbsc': fixed_halfedge_bbsc,
        }
        
        # # combine p1 and p2
        fixed_node = combine_vectors_indexed(
            [fixed_dict[f'node_bb'], fixed_dict[f'node_sc']],
            [data['node_bb'], data['node_sc']],
        )
        fixed_pos = combine_vectors_indexed(
            [fixed_dict[f'pos_bb'], fixed_dict[f'pos_sc']],
            [data['node_bb'], data['node_sc']],
        )
        fixed_halfedge = combine_vectors_indexed(
            [fixed_dict[f'halfedge_bb'], fixed_dict[f'halfedge_sc'], fixed_dict[f'halfedge_bbsc']],
            [data['halfedge_bb'], data['halfedge_sc'], data['halfedge_bbsc']],
        )
        data.update({
            'fixed_node': fixed_node,
            'fixed_pos': fixed_pos,
            'fixed_halfedge': fixed_halfedge,
        })
        
        # # for fixed_dist: always free pos
        domain_node_index = torch.empty([2, 0], dtype=torch.long)  # set rigid domain to empty
        n_domain = torch.tensor(0, dtype=torch.long)
        fixed_halfdist = torch.zeros_like(data['halfedge_type'], dtype=torch.long)  # default not fixed distances
        if setting['mode'] in ['sc', 'packing']:  # fixed distances of bb
            fixed_halfdist[data['halfedge_bb']] = 1
        data.update({
            'fixed_halfdist': fixed_halfdist,
            'n_domain': n_domain,
            'domain_node_index': domain_node_index,
        })
        return data


    def set_torsional_feat(self, data: Mol3DData):
        # no torsion or flex mode for pepdesign
        data.update({
            'tor_bonds_anno': torch.empty([0, 3], dtype=torch.long),
            'twisted_nodes_anno': torch.empty([0, 2], dtype=torch.long),
            'dihedral_pairs_anno': torch.empty([0, 3], dtype=torch.long),
        })
        return data


    def pre_make_partition(self, data: Mol3DData):
        
        node_bb = torch.nonzero(data['peptide_is_backbone'])[:, 0]
        node_sc = torch.nonzero(~data['peptide_is_backbone'])[:, 0]
        

        data.update({
            'node_bb': node_bb,
            'node_sc': node_sc,
        })
        return data


    def make_partition_features(self, data: Mol3DData):

        # # get backbone and sidechain
        node_bb = data['node_bb']
        node_sc = data['node_sc']
        
        assert len(node_bb) == len(np.unique(node_bb)), 'node_bb has duplicate nodes'
        assert len(node_sc) == len(np.unique(node_sc)), 'node_sc has duplicate nodes'
        assert len(set(node_bb) & set(node_sc)) == 0, 'node_bb and node_sc have common nodes'
        assert len(set(node_bb) | set(node_sc)) == data['node_type'].shape[0], 'node_bb and node_sc are not partitioned'
        
        # # make partition for halfedge
        halfedge_index = data.halfedge_index
        i_all_halfedge = torch.arange(halfedge_index.shape[1], dtype=torch.long)
        halfedge_bb = subgraph(node_bb, halfedge_index, i_all_halfedge)[1]
        halfedge_sc = subgraph(node_sc, halfedge_index, i_all_halfedge)[1]
        i_all_halfedge[halfedge_bb] = -1
        i_all_halfedge[halfedge_sc] = -1
        halfedge_bbsc = torch.nonzero(i_all_halfedge >= 0, as_tuple=False).squeeze()
        # anchor, i.e., CA atoms
        is_known_halfedge_bbsc = torch.zeros(halfedge_bbsc.shape[0], dtype=torch.bool)
        peptide_atom_name = np.array(data['peptide_atom_name'])
        n_or_ca_atoms = np.nonzero(
            (peptide_atom_name == 'CA') | (peptide_atom_name == 'N')
        )[0]  # proline also connects to N
        node_anchor = torch.tensor([n for n in n_or_ca_atoms if n in node_bb])
        halfedge_index_bbsc = halfedge_index[:, halfedge_bbsc]
        is_known_halfedge_bbsc = (halfedge_index_bbsc[..., None] != node_anchor).all(-1).all(0)
        
        partition = {
            'node_bb': node_bb,
            'node_sc': node_sc,
            'halfedge_bb': halfedge_bb,
            'halfedge_sc': halfedge_sc,
            'halfedge_bbsc': halfedge_bbsc,
            'is_known_halfedge_bbsc': is_known_halfedge_bbsc,
        }
        
        # # summary partition
        num_part_dict = {f'n_{key}': partition[key].shape[-1] for key in 
            ['node_bb', 'node_sc', 'halfedge_bb', 'halfedge_sc', 'halfedge_bbsc']}
        # santiy check
        assert num_part_dict['n_node_bb'] + num_part_dict['n_node_sc'] == data.node_type.shape[0]
        assert (num_part_dict['n_halfedge_bb'] + num_part_dict['n_halfedge_sc']
                + num_part_dict['n_halfedge_bbsc'] == data.halfedge_type.shape[0])

        data.update(partition)
        return data, num_part_dict
    
    def add_atoms(self, data):  # ask the model to predict mask-atom type
        # num_per_res = self.add_mask_atoms.num_per_res
        # n_res = data['peptide_res_index'].max().item() + 1
        is_backbone = data['peptide_is_backbone']
        is_sidechain = ~is_backbone
        n_sc = is_sidechain.sum().item()
        
        ratio = self.add_mask_atoms.ratio
        n_add_max = np.clip(int(n_sc * ratio), a_min=0, a_max=n_sc)
        n_add = np.clip(np.random.randint(-n_add_max, n_add_max + 1), a_min=0, a_max=n_add_max)
        if n_add == 0:
            return data
        
        # determine positions
        len_mu, len_sigma = self.add_mask_atoms.len_mu, self.add_mask_atoms.len_sigma
        node_sc = torch.nonzero(is_sidechain)[:, 0]
        node_sc_center = node_sc[torch.randint(n_sc, size=[n_add])]
        lengths = torch.randn([n_add]) * len_sigma + len_mu
        relative_pos = torch.randn([n_add, 3])
        relative_pos = relative_pos / (relative_pos.norm(dim=-1, keepdim=True)+1e-5) * lengths[:, None]
        node_pos = data['node_pos']
        node_pos_new = node_pos[node_sc_center] + relative_pos
        
        # new node
        node_pos = data['node_pos']
        node_type = data['node_type']
        n_atoms_data = node_type.shape[0]
        n_atoms_new = n_atoms_data + n_add
        new_node_type = torch.cat([node_type, (self.num_node_types-1) * torch.ones([n_add], dtype=node_type.dtype)], dim=0)
        new_node_pos = torch.cat([node_pos, node_pos_new], dim=0)
        
        # new edge
        halfedge_index = data['halfedge_index']
        halfedge_type = data['halfedge_type']
        n_add_halfedge = n_add * n_atoms_data + n_add * (n_add - 1) // 2
        new_halfedge_type = torch.cat([halfedge_type, torch.zeros([n_add_halfedge], dtype=halfedge_type.dtype)], dim=0)
        halfedge_index_old_new = torch.stack(
            torch.meshgrid(torch.arange(n_atoms_data), torch.arange(n_atoms_data, n_atoms_new), indexing='ij'),
        dim=0).reshape(2, -1)
        halfedge_index_new_new = torch.triu_indices(n_add, n_add, offset=1) + n_atoms_data
        new_halfedge_index = torch.cat([
            halfedge_index, halfedge_index_old_new, halfedge_index_new_new], dim=1)
        new_halfedge_index, new_halfedge_type = sort_edge_index(new_halfedge_index, new_halfedge_type)

        # peptide feature
        peptide_is_backbone = data['peptide_is_backbone']
        peptide_atom_name = data['peptide_atom_name']
        new_peptide_is_backbone = torch.cat([peptide_is_backbone, torch.zeros([n_add], dtype=peptide_is_backbone.dtype)], dim=0)
        new_peptide_atom_name = peptide_atom_name + ['X'] * n_add
        
        data.update({
            'num_nodes': n_atoms_new,
            'node_type': new_node_type,
            'node_pos': new_node_pos,
            'halfedge_index': new_halfedge_index,
            'halfedge_type': new_halfedge_type,
            'peptide_is_backbone': new_peptide_is_backbone,
            'peptide_atom_name': new_peptide_atom_name,
        })
        if 'is_peptide' in data:
            is_peptide = data['is_peptide']
            data['is_peptide'] = torch.cat([is_peptide, torch.ones([n_add], dtype=is_peptide.dtype)], dim=0)

        return data


    
@register_transforms('linking')
class LinkingTransform(MaskfillTransform):
    def pre_make_partition(self, data: Mol3DData):
        grouped_node_p1 = [list(item) for item in data['frags']]
        grouped_anchor_p1 = [list(item) for item in data['anchors']]
        node_p2 = sum([list(item) for item in data['linkers']], [])

        data.update({
            'grouped_node_p1': grouped_node_p1,
            'grouped_anchor_p1': grouped_anchor_p1,
            'node_p2': node_p2,
        })
        return data
    
    def prepare_sample(self, data, setting):
        # # move center of p1 (anchors) to origin
        if setting['known_anchor'] == 'all':
            anchors = [i for ans in data['anchors'] for i in ans]
            center = data['node_pos'][anchors].mean(dim=0, keepdims=True)
        else:
            center = data['node_pos'][data['node_p1']].mean(dim=0, keepdims=True)
        data['node_pos'] -= center
        data['pocket_pos'] -=center
        data['pocket_center'] += center
        
        # # make gt
        data['gt_node_type'] = data['node_type'].clone()
        data['gt_node_pos'] = data['node_pos'].clone()
        data['gt_halfedge_type'] = data['halfedge_type'].clone()
        
        # # make gt for part 1
        node_type = data['node_type']
        node_pos = data['node_pos']
        halfege_type = data['halfedge_type']
        
        for i_part in [1, 2]:
            node_type_part = - torch.ones_like(node_type)
            node_type_part[data[f'node_p{i_part}']] = node_type[data[f'node_p{i_part}']].clone()
            node_pos_part = node_pos.clone()
            halfedge_type_part = - torch.ones_like(halfege_type)
            halfedge_type_part[data[f'halfedge_p{i_part}']] = halfege_type[data[f'halfedge_p{i_part}']].clone()
            data.update({
                f'gt_node_type_p{i_part}': node_type_part,
                f'gt_node_pos_p{i_part}': node_pos_part,
                f'gt_halfedge_type_p{i_part}': halfedge_type_part,
            })
        
        # # not explicitly remove p2 info. controlled by info_level
            
        return data


@register_transforms('growing')
class GrowingTransform(MaskfillTransform):
    def pre_make_partition(self, data: Mol3DData):
        setting = data['task_setting']
        assert setting['known_anchor'] == 'none', 'Only none known_anchor is supported for frag growing yet.'
        if 'init_frag' in data:
            grouped_node_p1 = [data['init_frag']]
        else:
            grouped_node_p1 = [self.preset_partition['init_frag']]
        grouped_anchor_p1 = [[]]
        if 'add_frag' in data:
            node_p2 = data['add_frag']
        else:  # exclude init_frag
            node_p2 = [n for n in range(data['num_nodes']) if n not in grouped_node_p1[0]]

        data.update({
            'grouped_node_p1': grouped_node_p1,
            'grouped_anchor_p1': grouped_anchor_p1,
            'node_p2': node_p2,
        })
        return data
    



@register_transforms('ar')
class AutoregressiveTransform(MaskfillTransform):
    def __init__(self, config, **kwargs):
        mode = kwargs.get('mode', 'test')
        # assert mode == 'test', 'ar transform only for test (sample). For training, use maskfill transform'
        assert mode != 'train', 'ar transform only for test (sample). For training, use maskfill transform'
        super().__init__(config, **kwargs)

    def sample_setting(self):
        part1_pert = self.config.get('part1_pert', 'small')
        # setting_dict = {'part1_pert': 'small', 'known_anchor': 'none'}
        # setting_dict = {'part1_pert': 'fixed', 'known_anchor': 'none'}
        setting_dict = {'part1_pert': part1_pert, 'known_anchor': 'none'}
        return setting_dict  # to be compatible with maskfill transform

    def pre_make_partition(self, data: Mol3DData):
        # # save gt in advance
        data['gt_node_type'] = data['node_type'].clone()
        data['gt_node_pos'] = data['node_pos'].clone()
        data['gt_halfedge_type'] = data['halfedge_type'].clone()
        data['gt_halfedge_index'] = data['halfedge_index'].clone()
        
        # as the initial state of the ar sampling
        data.update({
            'grouped_node_p1': [],
            'grouped_anchor_p1': [],
            'node_p2': list(range(data['num_nodes'])),
        })
        # if data['pocket_knn_edge_index'].shape[1] > 0:
        #     raise NotImplementedError
        return data
    
    # def prepare_sample(self, data, setting):
    #     return data


def get_rigid_distmat(domain_index, node_index, n_domain, n_total_node):
    fixed_distmat = torch.zeros((n_total_node, n_total_node), dtype=torch.long)
    if n_domain == 0:
        return fixed_distmat
    else:
        for i_domain in range(n_domain):
            node_this_domain = node_index[domain_index == i_domain]
            fixed_distmat[np.ix_(node_this_domain, node_this_domain)] = 1
        return fixed_distmat


@register_transforms('ar2')
class Autoregressive2Transform(MaskfillTransform):
    def __init__(self, config, **kwargs):
        mode = kwargs.get('mode', 'test')
        assert mode == 'test', 'ar2 transform only for test (sample). For training, use maskfill transform'
        super().__init__(config, **kwargs)

    def sample_setting(self):
        setting_dict = {'part1_pert': 'small', 'known_anchor': 'none'}
        return setting_dict  # to be compatible with maskfill transform

    def pre_make_partition(self, data: Mol3DData):
        # # save gt in advance
        data['gt_node_type'] = data['node_type'].clone()
        data['gt_node_pos'] = data['node_pos'].clone()
        data['gt_halfedge_type'] = data['halfedge_type'].clone()
        data['gt_halfedge_index'] = data['halfedge_index'].clone()
        
        # as the initial state of the ar sampling
        n_init = 6
        data.update({
            'grouped_node_p1': [],
            'grouped_anchor_p1': [],
            'node_p2': list(range(n_init)),
        })
        # # reset mol nodes, as in ar2, only n node in initial state
        node_type = data['node_type'][:n_init]
        node_pos = data['node_pos'][:n_init]
        is_peptide = data['is_peptide'][:n_init]
        halfedge_type = data['halfedge_type']
        halfedge_index = data['halfedge_index']
        halfedge_index, halfedge_type = subgraph(
            torch.arange(n_init), halfedge_index, halfedge_type)
        data.update({
            'node_type': node_type,
            'node_pos': node_pos,
            'halfedge_type': halfedge_type,
            'halfedge_index': halfedge_index,
            'is_peptide': is_peptide,
        })
        
        return data
    
    def prepare_sample(self, data, setting):
        return data
    


def get_rigid_distmat(domain_index, node_index, n_domain, n_total_node):
    fixed_distmat = torch.zeros((n_total_node, n_total_node), dtype=torch.long)
    if n_domain == 0:
        return fixed_distmat
    else:
        for i_domain in range(n_domain):
            node_this_domain = node_index[domain_index == i_domain]
            fixed_distmat[np.ix_(node_this_domain, node_this_domain)] = 1
        return fixed_distmat



@register_transforms('custom')
class CustomTransform:
    def __init__(self, config, **kwargs) -> None:
        self.config = config
        self.mode = mode = kwargs.get('mode', 'test')
        
        self.is_peptide = config['is_peptide']
        self.partition = config['partition']
        self.partition_names = [p['name'] for p in self.partition]
        self.fixed = config['fixed']
        
        if 'sc' in self.partition_names:  # 1) last partition; 2) must be others (to auto handle new atoms)
            assert self.partition[-1] == {'name': 'sc', 'nodes': 'others'}, 'sc part is not valid'

    def __call__(self, data: Mol3DData):
        self._check_name(data)
        
        # make partiotion
        data, _ = self.make_partition_features(data)
        
        # set features
        data = self.set_fixed(data)
        data = self.set_torsional_feat(data)
        
        # is_peptide
        is_peptide = self.is_peptide
        data['is_peptide'] = is_peptide * torch.ones_like(data['is_peptide'])

        return data

    def _check_name(self, data):
        for key in self.partition_names:
            assert f'node_part_{key}' not in data, f'find exist key node_part_{key}'
            for key2 in self.partition_names:
                assert f'halfedge_{key}_{key2}' not in data, f'find exist key halfedge_{key}_{key2}'

    def set_fixed(self, data: Mol3DData):
        n_nodes = data['node_type'].shape[0]
        n_halfedges = data['halfedge_type'].shape[0]

        # # set fixed node
        fixed_node = get_vector(n_nodes, 0)  # default not fixed
        for part in self.fixed.get('node', []):
            node_part = data[f'node_part_{part}']
            fixed_node[node_part] = 1
        
        # # set fixed pos
        fixed_pos = get_vector(n_nodes, 0)  # default not fixed
        for part in self.fixed.get('pos', []):
            node_part = data[f'node_part_{part}']
            fixed_pos[node_part] = 1
            
        # # set fixed halfedge
        fixed_halfedge = get_vector(n_halfedges, 0)  # default not fixed
        for parts in self.fixed.get('edge', []):
            halfedge_part = data[f'halfedge_part_{parts[0]}_{parts[1]}']
            fixed_halfedge[halfedge_part] = 1

        data.update({
            'fixed_node': fixed_node,
            'fixed_pos': fixed_pos,
            'fixed_halfedge': fixed_halfedge,
        })
        
        # # for fixed_dist 
        fixed_halfdist = get_vector(n_halfedges, 0)  # default not fixed distances
        # fixed if both ends with fixed_pos == 1
        for part in self.fixed.get('pos', []):
            halfedge_part = data[f'halfedge_part_{part}_{part}']
            fixed_halfdist[halfedge_part] = 1
        
        domain_node_index = torch.empty([2, 0], dtype=torch.long)  # set rigid domain to empty
        n_domain = torch.tensor(0, dtype=torch.long)
        data.update({
            'fixed_halfdist': fixed_halfdist,
            'n_domain': n_domain,
            'domain_node_index': domain_node_index,
        })

        return data
    
    def set_torsional_feat(self, data: Mol3DData):
        data.update({
            'tor_bonds_anno': torch.empty([0, 3], dtype=torch.long),
            'twisted_nodes_anno': torch.empty([0, 2], dtype=torch.long),
            'dihedral_pairs_anno': torch.empty([0, 3], dtype=torch.long),
        })
        return data

    def make_partition_features(self, data: Mol3DData):
        
        num_nodes = data['node_type'].shape[0]
        data_partition = {}
        
        # # make node partition
        # re-index due to the variable pep size
        if 'removed_index' in data:
            removed_index = data['removed_index']
            # is_removed_node = torch.zeros([num_nodes + len(removed_index)], dtype=torch.bool)
            # is_removed_node[removed_index] = True
            # index_changes = torch.cumsum(is_removed_node, dim=0)
            is_removed_node = np.zeros([num_nodes + len(removed_index)], dtype=bool)
            is_removed_node[removed_index] = True
            index_changes = np.cumsum(is_removed_node)
        else:
            # index_changes = torch.zeros([num_nodes], dtype=torch.long)
            index_changes = np.zeros([num_nodes], dtype=np.int64)
            removed_index = []
        if 'added_index' in data:
            pass # nothing to do because nodes of `sc` must be `others`, so automatically added the new atoms
        
        all_atoms = set()
        for i_order, this_part in enumerate(self.partition):
            key = this_part['name']
            value = this_part['nodes']
            if isinstance(value, list):
                value = [atom for atom in value if atom not in removed_index]
                # re-index 
                value = np.array(value) - index_changes[value]
                value_set = set(value)
                assert len(all_atoms.intersection(value_set)) == 0, 'parition has overlap'
            elif isinstance(value, str):
                assert value == 'others', f'invalid value of partition: {value}'
                assert i_order == len(self.partition) - 1, 'others should be the last partition' 
                value = np.array([atom for atom in  np.arange(num_nodes) if atom not in all_atoms])
                value_set = set(value)
            else:
                raise ValueError(f'invalid value of partition: {value}')
            data_partition.update({
                'node_part_' + key: torch.LongTensor(value)})
            all_atoms.update(value_set)
        assert len(all_atoms) == num_nodes, 'partition does not cover all atoms'
        
        # # make halfedge partition
        halfedge_index = data['halfedge_index']
        i_all_halfedge = torch.arange(halfedge_index.shape[1], dtype=torch.long)
        edge_index, i_all_edge = to_undirected(halfedge_index, i_all_halfedge)

        num_partitions = len(self.partition)
        for i_part in range(num_partitions):
            name_pi = self.partition_names[i_part]
            node_pi = data_partition[f'node_part_{name_pi}']
            for j_part in range(i_part, num_partitions):
                name_pj = self.partition_names[j_part]
                node_pj = data_partition[f'node_part_{name_pj}']
                if i_part == j_part:
                    halfedge_pij = subgraph(node_pi, halfedge_index, i_all_halfedge)[1]
                else:
                    halfedge_pij = bipartite_subgraph([node_pi, node_pj], edge_index, i_all_edge)[1]
                name = 'halfedge_part_' + name_pi + '_' + name_pj
                data_partition.update({name: halfedge_pij})

        
        # # summary partition
        num_part_dict = {f'n_{key}': data_partition[key].shape[-1] for key in data_partition.keys()}
        # santiy check
        assert sum(value for key, value in num_part_dict.items() if key.startswith('n_node_part_')) == data.node_type.shape[0]
        assert sum(value for key, value in num_part_dict.items() if key.startswith('n_halfedge_part_')) == data.halfedge_type.shape[0]

        data.update(data_partition)
        return data, num_part_dict

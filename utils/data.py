import copy
from typing import Any
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

# FOLLOW_BATCH = ['protein_element', 'ligand_context_element', 'pos_real', 'pos_fake']



def edge_index_to_index_of_edge(edge_index, batch_node):
    """
    edge_index: (2, E)
    """
    assert (edge_index[0] < edge_index[1]).all()
    num_nodes_batch = batch_node.bincounts()
    index_mol = (edge_index - num_nodes_batch).abs().min()[1]
    bias = num_nodes_batch[:index_mol]
    edge_index_in_mol = edge_index - bias
    id_edge_0, id_edge_1 = edge_index_in_mol
    id_edge = (2*num_nodes_batch[id_edge_0]-id_edge_0-1) * id_edge_0 // 2 + id_edge_1 - id_edge_0 - 1
    return id_edge


def inc_func_mol3d(self, key, value, *args, **kwargs):
    if key == 'bond_index':
        return len(self['node_type'])
    elif key == 'edge_index':
        return len(self['node_type'])
    elif key == 'halfedge_index':
        return len(self['node_type'])
    # for linking
    elif key in ['node_p1', 'node_p2', 'node_bb', 'node_sc'] or key.startswith('node_part_'):  # index of node of part in mol
        return len(self['node_type'])
    elif key in ['halfedge_p1', 'halfedge_p2', 'halfedge_p1p2',
                 'halfedge_bb', 'halfedge_sc', 'halfedge_bbsc'] or\
                     key.startswith('halfedge_part_'):  # index of halfedge of part in mol
        return len(self['halfedge_type'])
    # for rigid and torsional
    elif key == 'domain_node_index':
        return torch.tensor([[self['n_domain']], [len(self['node_type'])]]) # [2, 1]
    elif key == 'domain_center_nodes':
        return len(self['node_type'])
    elif key == 'tor_bonds_anno':
        n_node = len(self['node_type'])
        return torch.tensor([0, n_node, n_node])
    elif key == 'twisted_nodes_anno':
        n_tor = len(self['tor_bonds_anno'])
        return torch.tensor([n_tor, len(self['node_type'])])
    elif key == 'dihedral_pairs_anno':
        n_node = len(self['node_type'])
        n_tor = len(self['tor_bonds_anno'])
        return torch.tensor([n_tor, n_node, n_node])
    return None
    
    
def inf_func_pocket(self, key, value, *args, **kwargs):
    if key == 'pocket_knn_edge_index':
        return len(self['pocket_pos'])
    return None


class PocketMolData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_pocket_mol_dicts(pocket_dict=None, mol_dict=None, **kwargs):
        instance = PocketMolData(**kwargs)

        if pocket_dict is not None:
            for key, item in pocket_dict.items():
                instance['pocket_' + key] = item

        if mol_dict is not None:
            for key, item in mol_dict.items():
                instance[key] = item

        return instance
    

    def __inc__(self, key, value, *args, **kwargs):
        # for defined mol inc
        inc = inc_func_mol3d(self, key, value, *args, **kwargs)
        if inc is not None:
            return inc
        # for defined pocket inc
        inc = inf_func_pocket(self, key, value, *args, **kwargs)
        if inc is not None:
            return inc
        # undefined
        return super().__inc__(key, value, *args, **kwargs)


class Mol3DData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_3dmol_dicts(ligand_dict=None, **kwargs):
        instance = Mol3DData(**kwargs)

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance[key] = item
            instance['orig_keys'] = list(ligand_dict.keys())

        # instance['nbh_list'] = {i.item():[j.item() for k, j in enumerate(instance.ligand_bond_index[1]) if instance.ligand_bond_index[0, k].item() == i] for i in instance.ligand_bond_index[0]}
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        inc = inc_func_mol3d(self, key, value, *args, **kwargs)
        if inc is not None:
            return inc
        inc = inf_func_pocket(self, key, value, *args, **kwargs)
        if inc is not None:
            return inc
        return super().__inc__(key, value, *args, **kwargs)


def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output

    
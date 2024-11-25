import networkx as nx
import numpy as np
from copy import deepcopy
from rdkit import Chem
from rdkit.Geometry import Point3D


# def is_same_mols(mol0, mol1, node_type=True, edge_type=True):
    
#     isomorphic = nx.vf2pp_is_isomorphic(mol0, mol1, )
#     return isomorphic


# def rdmol_to_attr_graph(mol):
#     mol = deepcopy(mol)
#     Chem.SanitizeMol(mol)
#     G = rdmol_to_graph(mol)
    
#     # add attr
#     node_types = {atom.GetIdx(): atom.GetSymbol() for atom in mol.GetAtoms()}
#     nx.set_node_attributes(G, node_types)
#     edge_types = {(atom.GetBeginAtomIdx(), atom.GetEndAtomIdx()): bond.GetBondTypeAsDouble()
#                   for bond in mol.GetBonds()}
#     nx.set_edge_attributes(G, edge_types)
    
#     return G


def rdmol_to_graph(mol):
    """
    Convert rdmol to graph
    """
    # mol to graph
    bond_index = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    G = nx.from_edgelist(bond_index)
    return G


def positions_to_graph(positions, threshold=3):
    """
    Convert positions to graph
    """
    # mol to graph
    dist = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    bond_index = np.argwhere(dist <= threshold)
    bond_index = bond_index[bond_index[:, 0] < bond_index[:, 1]]
    G = nx.from_edgelist(bond_index)
    return G



def map_and_set_mol_conf(mol, positions, threshold=2.2):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    mol = deepcopy(mol)
    assert mol.GetNumAtoms() == positions.shape[0], 'input node size does not match'
    # map atom index
    G_mol = rdmol_to_graph(mol)
    
    # while True:
    #     G_pos = positions_to_graph(positions, threshold)
    #     assert G_mol.number_of_nodes() == G_pos.number_of_nodes(), 'num of nodes not match'
    #     if G_mol.number_of_edges() < G_pos.number_of_edges():
    #         threshold -= 0.01
    #     elif G_mol.number_of_edges() == G_pos.number_of_edges():
    #         break
    #     else:
    #         raise ValueError(f'threhold {threshold} too small, try larger one.')

    G_pos = positions_to_graph(positions, threshold)

    GM = nx.isomorphism.GraphMatcher(G_pos, G_mol)
    if GM.is_isomorphic():
        mapping = GM.mapping
    elif GM.subgraph_is_monomorphic():
        mapping = next(GM.subgraph_monomorphisms_iter())
    else:
        # GL_mol = nx.line_graph(G_mol)
        # GL_pos = nx.line_graph(G_pos)
        # GL_match = nx.isomorphism.GraphMatcher(GL_pos, GL_mol)
        # if GM.subgraph_is_monomorphic
        raise ValueError('Mol and positions are not isomorphic')

    # set positions
    pos_reorder = np.zeros_like(positions)
    for i, j in mapping.items():
        pos_reorder[j] = positions[i]
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        # conf.SetAtomPosition(i, Point3D(positions[i]))
        conf.SetAtomPosition(i, pos_reorder[i].tolist())
    mol.AddConformer(conf)

    # validate by bond length
    mol_pos = mol.GetConformer().GetPositions()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_len = np.linalg.norm(mol_pos[i] - mol_pos[j])
        if bond_len > threshold:
            raise ValueError(f'Bond length is too long: {bond_len}')

    return mol


if __name__ == '__main__':
    import torch
    # test_data = torch.load('data/test/linking/difflinker_data/MOAD_test.pt', map_location='cpu')
    test_data = torch.load('data/test/linking/difflinker_data/zinc_final_test.pt', map_location='cpu')

    for sample in test_data:
        # sample = test_data[160]
        smiles = sample['name']
        positions = sample['positions']
        threshold = 2.
        
        mol = map_and_set_mol_conf(smiles, positions, threshold)
        print(mol)
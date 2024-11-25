
import networkx as nx
from rdkit.Chem.BRICS import FindBRICSBonds
# from rdkit.Chem import rdMMPA
from rdkit import Chem

def decompose_given_bonds(mol, list_of_bonds):
    """
    Using networkx to decompose the molecule into fragments
    """
    # mol to graph
    bond_index = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    G = nx.from_edgelist(bond_index)
    # remove bonds
    G.remove_edges_from(list_of_bonds)
    
    # get connected components
    subgraphs = list(nx.connected_components(G))
    node_list = [list(subgraph) for subgraph in subgraphs] 
    # sort by size in descending order
    node_list = sorted(node_list, key=lambda x: len(x), reverse=True)
    assert len(sum(node_list, [])) == len(set(sum(node_list, []))), 'Duplicate atoms in subgraphs'
    
    # find anchors and neighbors
    anchors = [set() for _ in range(len(node_list))]
    nbh_subgraphs = [[] for _ in range(len(node_list))]
    connections = {}
    for bond in list_of_bonds:
        node_left, node_right = bond
        idx_left = [i for i, node in enumerate(node_list) if node_left in node][0]
        idx_right = [i for i, node in enumerate(node_list) if node_right in node][0]

        anchors[idx_left].add(node_left)
        anchors[idx_right].add(node_right)
        nbh_subgraphs[idx_left].append(idx_right)
        nbh_subgraphs[idx_right].append(idx_left)
        if (idx_left, idx_right) in connections:
            assert connections[(idx_left, idx_right)] == (node_right, node_left), "Not the same bond. Ring detected"
        connections[(idx_left, idx_right)] = (node_left, node_right)
        connections[(idx_right, idx_left)] = (node_right, node_left)
    result = {
        'subgraphs': node_list,
        'anchors_list': anchors,
        'nbh_subgraphs': nbh_subgraphs,
        'connections': connections,
    }
    return result


def decompose_mmpa(mol):
    """
    Use MMPA to break mol into linkers and fragments
    return: List of (linker_smiles, frag_smiles) pair
    """
    pattern = Chem.MolFromSmarts('[#6+0;!$(*=,#[!#6])]!@!=!#[*]')

    find_bonds =mol.GetSubstructMatches(pattern)
    
    results = decompose_given_bonds(mol, find_bonds)
    return results


def decompose_brics(mol):

    list_of_bonds = [bond[0] for bond in FindBRICSBonds(mol)]
    result = decompose_given_bonds(mol, list_of_bonds)
    
    return result


if __name__ == '__main__':
    mol = Chem.MolFromSmiles('CN(C(=O)c1ccc2c(c1)OCO2)C1(C(=O)NC2CCCC2)CCCCC1')
    print(decompose_mmpa(mol))
    print(decompose_brics(mol))
    print('Done')
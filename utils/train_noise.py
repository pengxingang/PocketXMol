"""
Define the noise controller to facilitate noise injection and sampling.
"""

from itertools import product
from typing import Any
from easydict import EasyDict
from tqdm import tqdm
import torch
from torch_geometric.utils import subgraph, bipartite_subgraph, to_undirected
# from torch.nn import Module
# from torch.nn import functional as F
# from models.transition import ContigousTransition, GeneralCategoricalTransition
from models.diffusion import *

from utils.data import Mol3DData
from utils.prior import get_prior, MolPrior
from utils.info_level import IndividualInfoLevel, get_level, WholeInfoLevel, MolInfoLevel


def get_vector(shape, value, dtype=torch.long):
    return torch.ones(shape, dtype=dtype) * value
def get_vector_list(shape_list, value_list, dtype=torch.long):
    return [get_vector(shape, value, dtype) for shape, value in zip(shape_list, value_list)]

def combine_vectors_indexed(vector_list, indices_list):
    vector_combine = torch.cat(vector_list, dim=0)
    for vector, indices in zip(vector_list, indices_list):
        vector_combine[indices] = vector
    
    # check indices are unique
    indices_combine = torch.cat(indices_list, dim=0)
    assert indices_combine.unique().size(0) == indices_combine.size(0), 'indices should be unique'
    return vector_combine


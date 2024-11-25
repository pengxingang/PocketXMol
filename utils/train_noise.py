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

TRAIN_NOISE_DICT = {}
def register_train_noise(name):
    def add_to_registery(cls):
        TRAIN_NOISE_DICT[name] = cls
        return cls
    return add_to_registery


def get_train_noiser(config, num_node_types, num_edge_types):
    name = config['name']
    return TRAIN_NOISE_DICT[name](config, num_node_types, num_edge_types)

# def get_vector(shape, value, device, dtype=torch.long):
#     return torch.ones(shape, dtype=dtype, device=device) * value
# def get_vector_list(shape_list, value_list, device, dtype=torch.long):
#     return [get_vector(shape, value, device, dtype) for shape, value in zip(shape_list, value_list)]
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
    
    

@register_train_noise('mixed')
class MixedTrainNoiser:
    def __init__(self, config, num_node_types, num_edge_types, device='cpu', **kwargs):
        super().__init__()
        self.config = config
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.device = device
        
        # define noisers and weights
        weights = np.array(config.weights)
        self.weights = weights / weights.sum()
        indi_configs = config.individual
        self.noiser_list = []
        for indi_config in indi_configs:
            noiser = get_train_noiser(indi_config, num_node_types, num_edge_types)
            self.noiser_list.append(noiser)
        assert len(self.noiser_list) == len(self.weights), 'length of noiser list and weights should be the same'
            
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # randomly choice a noiser
        noiser = np.random.choice(self.noiser_list, p=self.weights)
        return noiser(*args, **kwds)


@register_train_noise('denovo')
class DenovoTrainNoiser:
    def __init__(self,
        config, num_node_types, num_edge_types, device='cpu', **kwargs
    ):
        super().__init__()
        self.config = config
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.device = device
        
        # define prior
        self.mol_prior = MolPrior(config.prior, device, num_node_types, num_edge_types)
        # self.node_prior = get_prior(config.prior.node, device, num_node_types)
        # self.pos_prior = get_prior(config.prior.pos, device)
        # self.halfedge_prior = get_prior(config.prior.edge, device, num_edge_types)
        self.level = MolInfoLevel(config.level)


    def set_fixed(self, shape_dict):  # all zeros for denovo mol gen
        fixed_dict = {}
        for key, shape in shape_dict.items():
            fixed = torch.zeros(shape, dtype=torch.long).to(self.device)
            fixed_dict[key] = fixed
        return fixed_dict
    
    def sample_level(self, shape_dict):
        # info_dict = {}
        level_node, level_pos, level_halfedge = self.level.sample_for_mol(
            shape_dict['node'], shape_dict['halfedge']
        )
        level_dict = {
            'node': level_node,
            'pos': level_pos,
            'halfedge': level_halfedge
        }
        return level_dict

    def add_noise(self, node_type, node_pos, halfedge_type,
                  level_node, level_pos, level_halfedge,):
        device = node_pos.device

        # 2.1 perturb pos, node, edge
        node_in, pos_in, halfedge_in = self.mol_prior.add_noise(
            node_type, node_pos, halfedge_type,
            level_node, level_pos, level_halfedge
        )
        return node_in, pos_in, halfedge_in

    def __call__(self, data: Mol3DData):  # add noise in forward diffusion
        node_type = data.node_type
        node_pos = data.node_pos
        halfedge_type = data.halfedge_type
        shape_dict = {
            'node': node_type.shape[0],
            'pos': node_pos.shape[0],
            'halfedge': halfedge_type.shape[0]
        }
        
        # fixed: all zeros for denovo gen
        fixed_dict = self.set_fixed(shape_dict)
        data.update({'fixed_'+key: value for key, value in fixed_dict.items()})
        
        # info_level
        info_dict = self.sample_level(shape_dict)
        data.update({'level_'+key: value for key, value in info_dict.items()})
        
        # noised input
        node_pert, pos_pert, halfedge_pert = self.add_noise(node_type, node_pos, halfedge_type,
                                        info_dict['node'], info_dict['pos'], info_dict['halfedge'])
        data.update({'node_in': node_pert, 'pos_in': pos_pert, 'halfedge_in': halfedge_pert})
        
        # errors for model prediction
        # node_error, pos_error, halfedge_error = self.add_errors(node_type, node_pos, halfedge_type,
        #                                 node_pert, pos_pert, halfedge_pert)
        # data.update({'node_error': node_error, 'pos_error': pos_error, 'halfedge_error': halfedge_error})
        return data
    
    # def add_errors(self, node_type, node_pos, halfedge_type,
    #                node_in, pos_in, halfedge_in,):
    #     node_error = (node_type != node_in).float()
    #     pos_error = torch.norm(node_pos - pos_in, dim=-1)
    #     halfedge_error = (halfedge_type != halfedge_in).float()
    #     # normalize pos_error
    #     pos_error = torch.log10(2 * pos_error + 1)
    #     return node_error, pos_error, halfedge_error


@register_train_noise('conf')
class ConfTrainNoiser:
    def __init__(self,
        config, num_node_types, num_edge_types, device='cpu', **kwargs
    ):
        super().__init__()
        self.config = config
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.device = device
        
        # define prior
        self.pos_prior = get_prior(config.prior.pos, device)


    def set_fixed(self, shape_dict):  # in mol conf: zeros for pos, ones for node and edge
        fixed_dict = {}
        for key, shape in shape_dict.items():
            if key == 'pos':
                fixed = torch.zeros(shape, dtype=torch.long).to(self.device)
            else:
                assert key in ['node', 'halfedge'], 'key should be node, halfedge or pos'
                fixed = torch.ones(shape, dtype=torch.long).to(self.device)
            fixed_dict[key] = fixed
        return fixed_dict
    
    def sample_level(self, shape_dict):
        info_dict = {}
        for key, shape in shape_dict.items():
            if key == 'pos':
                info = torch.ones(shape) * torch.rand(1)
            else:
                info = torch.ones(shape) # for conf: node/halfedge info_level = 1
            info_dict[key] = info
        return info_dict

    def add_noise(self, node_type, node_pos, halfedge_type,
                  info_node, info_pos, info_halfedge,):
        device = node_pos.device

        # 2.1 perturb pos, node, edge
        node_pert = node_type
        pos_pert = self.pos_prior.add_noise(node_pos, info_pos)
        halfedge_pert = halfedge_type
        return node_pert, pos_pert, halfedge_pert

    def __call__(self, data: Mol3DData):  # add noise in forward diffusion
        node_type = data.node_type
        node_pos = data.node_pos
        halfedge_type = data.halfedge_type
        shape_dict = {
            'node': node_type.shape[0],
            'pos': node_pos.shape[0],
            'halfedge': halfedge_type.shape[0]
        }
        
        # budget: all one for complete mol gen
        fixed_dict = self.set_fixed(shape_dict)
        data.update({'fixed_'+key: value for key, value in fixed_dict.items()})
        
        # info_level
        info_dict = self.sample_level(shape_dict, fixed_dict)
        data.update({'level_'+key: value for key, value in info_dict.items()})
        
        node_pert, pos_pert, halfedge_pert = self.add_noise(node_type, node_pos, halfedge_type,
                                        info_dict['node'], info_dict['pos'], info_dict['halfedge'])
        data.update({'node_in': node_pert, 'pos_in': pos_pert, 'halfedge_in': halfedge_pert})
        
        # errors for model prediction
        # node_error, pos_error, halfedge_error = self.add_errors(node_type, node_pos, halfedge_type,
        #                                 node_pert, pos_pert, halfedge_pert)
        # data.update({'node_error': node_error, 'pos_error': pos_error, 'halfedge_error': halfedge_error})
        
        return data
    
    # def add_errors(self, node_type, node_pos, halfedge_type,
    #                node_in, pos_in, halfedge_in,):
    #     node_error = (node_type != node_in).float()
    #     pos_error = torch.norm(node_pos - pos_in, dim=-1)
    #     halfedge_error = (halfedge_type != halfedge_in).float()
    #     # normalize pos_error
    #     pos_error = torch.log10(2 * pos_error + 1)
    #     return node_error, pos_error, halfedge_error


@register_train_noise('linking')
class LinkingTrainNoiser:
    def __init__(self,
        config, num_node_types, num_edge_types, device='cpu', **kwargs
    ):
        super().__init__()
        self.config = config
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.device = device
        #NOTE: how to set device
        # define prior to sample from
        self.prior_p1 = MolPrior(config.prior.part1, device, num_node_types, num_edge_types).to(device)
        self.prior_p2 = MolPrior(config.prior.part2, device, num_node_types, num_edge_types).to(device)
        # define level sampler
        self.level_p1 = MolInfoLevel(config.level.part1)
        self.level_p2 = MolInfoLevel(config.level.part2)
        

    def sample_level(self, num_dict, fixed_dict, partition):
        # # level for part 1, part 2 and part1-part2
        level_dict = {}
        for i_part in [1, 2]:
            leveller = getattr(self, f'level_p{i_part}')
            # level of part
            level_node_part, level_pos_part, level_halfedge_part = leveller.sample_for_mol(
                    num_dict[f'n_node_p{i_part}'],
                    num_dict[f'n_halfedge_p{i_part}'] + (num_dict['n_halfedge_p1p2'] if i_part == 2 else 0)
                )
            level_dict.update({
                f'node_p{i_part}': level_node_part,
                f'pos_p{i_part}': level_pos_part,
                f'halfedge_p{i_part}': level_halfedge_part[:num_dict[f'n_halfedge_p{i_part}']],
            })
            if i_part == 2:
                level_dict['halfedge_p1p2'] = level_halfedge_part[num_dict['n_halfedge_p2']:]
                # edges between part1 and part2
        
        # # reset according to fixed_dict
        for key, value in level_dict.items():
            level_dict[key] = value * (1 - fixed_dict[key]) + fixed_dict[key]
        return level_dict

    def add_noise(self, node_type, node_pos, halfedge_type, level_dict, partition):
        device = node_pos.device
        
        noised_dict = {}
        for i_part in [1, 2]:
            # prepare
            node_part = node_type[partition[f'node_p{i_part}']]
            pos_part = node_pos[partition[f'node_p{i_part}']]
            halfedge_part = halfedge_type[partition[f'halfedge_p{i_part}']]
            prior_obj = getattr(self, f'prior_p{i_part}')
            # add noise
            node_in, pos_in, halfedge_in = prior_obj.add_noise(
                node_part, pos_part, halfedge_part,
                level_dict[f'node_p{i_part}'], level_dict[f'pos_p{i_part}'], level_dict[f'halfedge_p{i_part}']
            )
            noised_dict.update({
                f'node_p{i_part}': node_in,
                f'pos_p{i_part}': pos_in,
                f'halfedge_p{i_part}': halfedge_in,
            })
        noised_dict['halfedge_p1p2'] = self.prior_p2.halfedge.add_noise( # same as part2 edge noiser
            halfedge_type[partition['halfedge_p1p2']], level_dict['halfedge_p1p2']
        )

        return noised_dict

    def sample_setting(self):
        setting = np.random.choice(self.setting_configs, p=self.setting_weights)
        return setting
        
    def __call__(self, data: Mol3DData):  # add noise in forward diffusion
        node_type = data.node_type
        node_pos = data.node_pos
        halfedge_type = data.halfedge_type
        num_dict = dict(
            n_node = node_type.shape[0],
            n_halfedge = halfedge_type.shape[0]
        )
        
        # # sample a task for this data
        # setting = self.sample_setting()
        # partition, num_part_dict = self.get_partition(data)
        # num_dict.update(num_part_dict)
        
        # # fixed
        # fixed_dict = self.set_fixed(num_dict, setting, partition)
        # data.update({'fixed_'+key: value for key, value in fixed_dict.items()})
        
        # # info_level
        level_dict = self.sample_level(num_dict, fixed_dict, partition)
        data.update({'level_'+key: value for key, value in level_dict.items()})
        
        # # noised input
        in_dict = self.add_noise(node_type, node_pos, halfedge_type,
                                                            level_dict, partition)
        # data.update({'node_in': node_pert, 'pos_in': pos_pert, 'halfedge_in': halfedge_pert})
        
        # # combine parts together
        for keyword in ['fixed', 'level', 'in']:
            this_dict = locals()[keyword+'_dict']
            node = combine_vectors_indexed(
                [this_dict[f'node_p1'], this_dict[f'node_p2']],
                [partition['node_p1'], partition['node_p2']],
            )
            pos = combine_vectors_indexed(
                [this_dict[f'pos_p1'], this_dict[f'pos_p2']],
                [partition['node_p1'], partition['node_p2']],
            )
            halfedge = combine_vectors_indexed(
                [this_dict[f'halfedge_p1'], this_dict[f'halfedge_p2'], this_dict[f'halfedge_p1p2']],
                [partition['halfedge_p1'], partition['halfedge_p2'], partition['halfedge_p1p2']],
            )
            if keyword == 'in':
                data.update({'node_in': node, 'pos_in': pos, 'halfedge_in': halfedge})
            else:
                data.update({keyword+'_node': node, keyword+'_pos': pos, keyword+'_halfedge': halfedge})

        # cleaning up
        # data.pop('linking')
        # data.pop('linking_df')
        return data
    
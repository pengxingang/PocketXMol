"""
loss function for the model
"""
from copy import deepcopy
import numpy as np
import logging
from torch import nn
import torch
from torch.nn import functional as F
from torch_geometric.nn.pool import radius

from models.diffusion import index_to_log_onehot
from models.transition import GeneralCategoricalTransition, ContigousTransition
from models.corrector import get_dihedral_batch

LOSS_DICT = {}

def register_loss(name):
    def decorator(cls):
        LOSS_DICT[name] = cls
        return cls
    return decorator

def get_loss_func(config, *args, **kwargs):
    if 'name' not in config:
        logging.warning('No loss type specified, using asymloss by default')
    name = config.get('name', 'asymloss')
    if name not in LOSS_DICT:
        raise ValueError(f'Unknown loss type: {name}')
    return LOSS_DICT[name](config, *args, **kwargs)



@register_loss('individual_tasks')
class IndividualTasksLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weights = config.weights
        self.task_list = config.tasks
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        if hasattr(config, 'confidence'):
            self.cfd_loss = ConfidenceLoss(config.confidence)
        else:
            self.cfd_loss = None
            
        self.pocket_dist_config = getattr(config, 'pocket_dist', None)
        self.size_weighted = getattr(config, 'size_weighted', False)
            
        
    def _selected_mean_loss(self, loss_total, select, size_node=None):
        if select.sum() == 0:
            return torch.tensor(0.0, device=loss_total.device)
        else:
            if size_node is None:
                return torch.mean(loss_total[select])
            else:
                return torch.mean(loss_total[select] / (size_node[select]+1e-3))
        
    def pos_loss(self, node_pos, pred_pos, unfixed_pos, task_index_dict=None):
        loss_pos_total = F.mse_loss(pred_pos, node_pos, reduction='none')

        if task_index_dict is None:
            task_index_dict = {'mixed': torch.ones_like(unfixed_pos)}

        loss_pos_tasked = {}
        for task, select in task_index_dict.items():
            loss_pos = self._selected_mean_loss(loss_pos_total, unfixed_pos & select)
            loss_pos_fixed = self._selected_mean_loss(loss_pos_total, (~unfixed_pos) & select)
            loss_pos_tasked.update({
                f'{task}/pos': loss_pos,
                f'{task}/fixed_pos': loss_pos_fixed
            })

        return loss_pos_tasked
    
    def node_loss(self, node_type, pred_node, unfixed_node, task_index_dict=None):
        loss_node_total = self.ce_loss(pred_node, node_type)

        if task_index_dict is None:
            task_index_dict = {'mixed': torch.ones_like(unfixed_node)}

        loss_node_tasked = {}
        for task, select in task_index_dict.items():
            loss_node = self._selected_mean_loss(loss_node_total, unfixed_node & select)
            loss_node_fixed = self._selected_mean_loss(loss_node_total, (~unfixed_node) & select)
            loss_node_tasked.update({
                f'{task}/node': loss_node,
                f'{task}/fixed_node': loss_node_fixed
            })
    
        return loss_node_tasked
        
    def halfedge_loss(self, halfedge_type, pred_halfedge, unfixed_halfedge, task_index_dict=None, size_node=None):
        loss_edge_total = self.ce_loss(pred_halfedge, halfedge_type)

        if task_index_dict is None:
            task_index_dict = {'mixed': torch.ones_like(unfixed_halfedge)}

        loss_edge_tasked = {}
        for task, select in task_index_dict.items():
            loss_edge = self._selected_mean_loss(loss_edge_total, unfixed_halfedge & select, size_node)
            loss_edge_fixed = self._selected_mean_loss(loss_edge_total, (~unfixed_halfedge) & select, size_node)
            loss_edge_tasked.update({
                f'{task}/edge': loss_edge,
                f'{task}/fixed_edge': loss_edge_fixed
            })
        
        return loss_edge_tasked

    def dist_loss(self, gt_dist, pred_dist, unfixed_dist, batch, task_index_dict=None):
        if task_index_dict is None:
            task_index_dict = {'mixed': torch.ones_like(unfixed_dist)}
            
        # dist loss only edge in the same domain
        halfedge_index = batch['halfedge_index']
        domain_node_index = batch['domain_node_index']
        domain_index_of_node = - torch.ones_like(batch['node_type'])
        domain_index_of_node[domain_node_index[1]] = domain_node_index[0]
        inner_domain = ((domain_index_of_node[halfedge_index[0]] >= 0) & 
            (domain_index_of_node[halfedge_index[0]] == domain_index_of_node[halfedge_index[1]]) & 
            unfixed_dist)

        loss_dist_tasked = {}
        for task, select in task_index_dict.items():
            if (inner_domain & select).sum() == 0:
                loss_dist = torch.tensor(0.0, device=gt_dist.device)
            else:
                loss_dist = F.mse_loss(pred_dist[inner_domain & select], gt_dist[inner_domain & select], reduction='mean')
            # fixed_dist loss:
            if ((~unfixed_dist)&select).sum() == 0:
                loss_dist_fixed = torch.tensor(0.0, device=gt_dist.device)
            else:
                loss_dist_fixed = F.mse_loss(pred_dist[(~unfixed_dist)&select], gt_dist[(~unfixed_dist)&select], reduction='mean')
            loss_dist_tasked.update({
                f'{task}/dist': loss_dist,
                f'{task}/fixed_dist': loss_dist_fixed
            })
        return loss_dist_tasked

    def dihedral_loss(self, node_pos, pred_pos, batch, task_index_dict=None):
        
        pred_sin, pred_cos = get_dihedral_batch(pred_pos, batch['tor_bonds_anno'], batch['dihedral_pairs_anno'])
        gt_sin, gt_cos = get_dihedral_batch(node_pos, batch['tor_bonds_anno'], batch['dihedral_pairs_anno'])

        # make task index here
        if task_index_dict is None:
            task_index_dict = {'mixed': torch.ones(len(pred_sin), device=pred_sin.device, dtype=torch.bool)}

        cos_delta = pred_cos * gt_cos + pred_sin * gt_sin
        
        loss_dih_tasked = {}
        for task, select in task_index_dict.items():
            if len(cos_delta[select]) == 0:
                loss_dih = torch.tensor(0.0, device=node_pos.device)
            else:
                loss_dih = torch.mean(1 - cos_delta[select])
            loss_dih_tasked.update({
                f'{task}/dih': loss_dih
            })

        return loss_dih_tasked
    
    def forward(self, batch, outputs, tasked=True):
        
        
        # # preparation
        # gt
        node_pos = batch['node_pos']
        node_type = batch['node_type']
        halfedge_type = batch['halfedge_type']
        halfedge_index = batch['halfedge_index']
        gt_dist = torch.norm(node_pos[halfedge_index[0]] - node_pos[halfedge_index[1]], dim=-1)

        # prediction
        pred_pos = outputs['pred_pos']
        pred_node = outputs['pred_node']
        pred_halfedge = outputs['pred_halfedge']
        pred_dist = torch.norm(pred_pos[halfedge_index[0]] - pred_pos[halfedge_index[1]], dim=-1)
        
        # divide into fixed and unfixed
        unfixed_node = (batch['fixed_node'] == 0)
        unfixed_pos = (batch['fixed_pos'] == 0)
        unfixed_halfedge = (batch['fixed_halfedge'] == 0)
        unfixed_dist = (batch['fixed_halfdist'] == 0)
        
        # mol size
        if self.size_weighted:
            index_batch = batch['node_type_batch']
            size = torch.bincount(index_batch).float()
            size_node = (size[batch['halfedge_type_batch']] + 10)
            size_node = size_node / torch.mean(size_node)
        else:
            size_node = torch.ones_like(halfedge_type).float()
        
        # divide into tasks
        if tasked:
            device = node_pos.device
            task_list = self.task_list
            # task_list = np.unique(batch['task']).tolist()
            task_batch_dict = {}
            task_node_dict = {'mixed': torch.ones_like(unfixed_node)}
            task_edge_dict = {'mixed': torch.ones_like(unfixed_halfedge)}
            task_dih_dict = {'mixed': torch.ones(len(batch['dihedral_pairs_anno']), device=device, dtype=torch.bool)}
            batch_node = batch['node_type_batch']
            batch_halfedge = batch['halfedge_type_batch']
            batch_dih = batch_node.index_select(0, batch['dihedral_pairs_anno'][:,1])
            for task in task_list:
                task_batch_dict[task] = torch.tensor([t==task for t in batch['task']], device=device)
                task_node_dict[task] = task_batch_dict[task].index_select(0, batch_node)
                task_edge_dict[task] = task_batch_dict[task].index_select(0, batch_halfedge)
                task_dih_dict[task] = task_batch_dict[task].index_select(0, batch_dih)
        else:
            task_list = []
            task_batch_dict, task_node_dict, task_edge_dict, task_dih_dict = None, None, None, None
        
        # # basic loss
        loss_list = [
            self.node_loss(node_type, pred_node, unfixed_node, task_node_dict),
            self.pos_loss(node_pos, pred_pos, unfixed_pos, task_node_dict),
            self.halfedge_loss(halfedge_type, pred_halfedge, unfixed_halfedge, task_edge_dict, size_node),
            self.dist_loss(gt_dist, pred_dist, unfixed_dist, batch, task_edge_dict),
            self.dihedral_loss(node_pos, pred_pos, batch, task_dih_dict),
        ]
        loss_dict = {k:v for d in loss_list for k,v in d.items()}

        # # total los
        for task in ['mixed'] + task_list:
            loss_dict[f'{task}/total'] = sum([self.weights[loss_type.replace(task+'/', '')] * l for
                                              loss_type, l in loss_dict.items() if loss_type.startswith(task+'/')])

        # # confidence loss
        if self.cfd_loss is not None:
            loss_cfd = self.cfd_loss(batch, outputs)
            loss_dict['mixed/total'] += loss_cfd['cfd_total']
            loss_dict.update({'mixed/'+k:v for k, v in loss_cfd.items()})

        # # pocket dist
        if self.pocket_dist_config is not None:
            loss_pocket_dist = self.pocket_dist_loss(batch, outputs)
            loss_dict['mixed/total'] += loss_pocket_dist
            loss_dict.update({'mixed/p_dist': loss_pocket_dist})
        
        return loss_dict
    
    def pocket_dist_loss(self, batch, outputs):
        gt_pocket_pos = batch['pocket_pos']
        gt_node_pos = batch['node_pos']
        knn_node_pocket = radius(x=gt_pocket_pos, y=gt_node_pos, r=self.pocket_dist_config['radius'],
                                 batch_x=batch['pocket_pos_batch'],
                                 batch_y=batch['node_type_batch'])
        gt_dist = torch.norm(gt_node_pos[knn_node_pocket[0]] - gt_pocket_pos[knn_node_pocket[1]], dim=-1)
        
        pred_node_pos = outputs['pred_pos']
        pred_dist = torch.norm(pred_node_pos[knn_node_pocket[0]] - gt_pocket_pos[knn_node_pocket[1]], dim=-1)
        
        loss_pocket_dist = F.mse_loss(gt_dist, pred_dist)
        loss_pocket_dist *= self.pocket_dist_config['weight']
        return loss_pocket_dist

@register_loss('all_tasks')
class AllTasksLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
        
    def pos_loss(self, node_pos, pred_pos, unfixed_pos):
        loss_pos_total = F.mse_loss(pred_pos, node_pos, reduction='none')
        if unfixed_pos.sum() == 0:
            loss_pos = torch.tensor(0.0, device=node_pos.device)
        else:
            loss_pos = torch.mean(loss_pos_total[unfixed_pos])
        if (~unfixed_pos).sum() == 0:
            loss_pos_fixed = torch.tensor(0.0, device=node_pos.device)
        else:
            loss_pos_fixed = torch.mean(loss_pos_total[~unfixed_pos])
        return loss_pos, loss_pos_fixed
    
    def node_loss(self, node_type, pred_node, unfixed_node):
        loss_node_total = self.ce_loss(pred_node, node_type)
        if unfixed_node.sum() == 0:
            loss_node = torch.tensor(0.0, device=node_type.device)
        else:
            loss_node = torch.mean(loss_node_total[unfixed_node])
        if (~unfixed_node).sum() == 0:
            loss_node_fixed = torch.tensor(0.0, device=node_type.device)
        else:
            loss_node_fixed = torch.mean(loss_node_total[~unfixed_node])
        return loss_node, loss_node_fixed
        
    def halfedge_loss(self, halfedge_type, pred_halfedge, unfixed_halfedge):
        loss_edge_total = self.ce_loss(pred_halfedge, halfedge_type)
        if unfixed_halfedge.sum() == 0:
            loss_edge = torch.tensor(0.0, device=halfedge_type.device)
        else:
            loss_edge = torch.mean(loss_edge_total[unfixed_halfedge])
        if (~unfixed_halfedge).sum() == 0:
            loss_edge_fixed = torch.tensor(0.0, device=halfedge_type.device)
        else:
            loss_edge_fixed = torch.mean(loss_edge_total[~unfixed_halfedge])
        return loss_edge, loss_edge_fixed

    def dist_loss(self, gt_dist, pred_dist, unfixed_dist, batch):
        # dist loss only edge in the same domain
        halfedge_index = batch['halfedge_index']
        domain_node_index = batch['domain_node_index']
        domain_index_of_node = - torch.ones_like(batch['node_type'])
        domain_index_of_node[domain_node_index[1]] = domain_node_index[0]
        inner_domain = ((domain_index_of_node[halfedge_index[0]] >= 0) & 
            (domain_index_of_node[halfedge_index[0]] == domain_index_of_node[halfedge_index[1]]) & 
            unfixed_dist)
        if inner_domain.sum() == 0:
            loss_dist = torch.tensor(0.0, device=gt_dist.device)
        else:
            loss_dist = F.mse_loss(pred_dist[inner_domain], gt_dist[inner_domain], reduction='mean')
        # fixed_dist loss:
        if (~unfixed_dist).sum() == 0:
            loss_dist_fixed = torch.tensor(0.0, device=gt_dist.device)
        else:
            loss_dist_fixed = F.mse_loss(pred_dist[~unfixed_dist], gt_dist[~unfixed_dist], reduction='mean')
        return loss_dist, loss_dist_fixed

    def dihedral_loss(self, node_pos, pred_pos, batch):
        pred_sin, pred_cos = get_dihedral_batch(pred_pos, batch['tor_bonds_anno'], batch['dihedral_pairs_anno'])
        gt_sin, gt_cos = get_dihedral_batch(node_pos, batch['tor_bonds_anno'], batch['dihedral_pairs_anno'])

        cos_delta = pred_cos * gt_cos + pred_sin * gt_sin
        # sin_delta = sin_pred * cos_gt - cos_pred * sin_gt

        if len(cos_delta) == 0:
            loss_dih = torch.tensor(0.0, device=node_pos.device)
        else:
            loss_dih = torch.mean(1 - cos_delta)
        return loss_dih
    
    def forward(self, batch, outputs):
        
        
        # # preparation
        # gt
        node_pos = batch['node_pos']
        node_type = batch['node_type']
        halfedge_type = batch['halfedge_type']
        halfedge_index = batch['halfedge_index']
        gt_dist = torch.norm(node_pos[halfedge_index[0]] - node_pos[halfedge_index[1]], dim=-1)

        # prediction
        pred_pos = outputs['pred_pos']
        pred_node = outputs['pred_node']
        pred_halfedge = outputs['pred_halfedge']
        pred_dist = torch.norm(pred_pos[halfedge_index[0]] - pred_pos[halfedge_index[1]], dim=-1)
        
        # divide into fixed and unfixed
        unfixed_node = (batch['fixed_node'] == 0)
        unfixed_pos = (batch['fixed_pos'] == 0)
        unfixed_halfedge = (batch['fixed_halfedge'] == 0)
        unfixed_dist = (batch['fixed_halfdist'] == 0)
        
        # # basic loss
        loss_node, loss_node_fixed = self.node_loss(node_type, pred_node, unfixed_node)
        loss_pos, loss_pos_fixed = self.pos_loss(node_pos, pred_pos, unfixed_pos)
        loss_halfedge, loss_halfedge_fixed = self.halfedge_loss(halfedge_type, pred_halfedge, unfixed_halfedge)
        loss_dist, loss_dist_fixed = self.dist_loss(gt_dist, pred_dist, unfixed_dist, batch)
        
        loss_dih = self.dihedral_loss(node_pos, pred_pos, batch)

        # # total
        loss_unfixed = (
            loss_node * self.config['weights']['node'] + \
            loss_pos * self.config['weights']['pos'] + \
            loss_halfedge * self.config['weights']['edge'] + \
            loss_dist * self.config['weights']['dist'] + \
            loss_dih * self.config['weights']['dihedral']
        )
        loss_fixed = (
            loss_node_fixed * self.config['fixed_weights']['node'] + \
            loss_pos_fixed * self.config['fixed_weights']['pos'] + \
            loss_halfedge_fixed * self.config['fixed_weights']['edge'] + \
            loss_dist_fixed * self.config['fixed_weights']['dist']
        )
        loss = loss_unfixed + loss_fixed

        
        loss_dict = {
            'loss': loss,
            'loss_node': loss_node,
            'loss_pos': loss_pos,
            'loss_edge': loss_halfedge,
            'loss_dist': loss_dist,
            'loss_dih': loss_dih,
            'loss_fixed_total': loss_fixed,

            'loss_fixed/node': loss_node_fixed,
            'loss_fixed/pos': loss_pos_fixed,
            'loss_fixed/edge': loss_halfedge_fixed,
            'loss_fixed/dist': loss_dist_fixed,
            
        }
        return loss_dict



@register_loss('refine_torsional2')
class RefineTorsionalLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, batch, outputs):
        pred_pos = outputs['pred_pos']
        pred_node = outputs['pred_node']
        pred_halfedge = outputs['pred_halfedge']
        
        node_pos = batch['node_pos']
        node_type = batch['node_type']
        halfedge_type = batch['halfedge_type']
        
        # divide into fixed and unfixed
        unfixed_node = batch['fixed_node'] == 0
        unfixed_pos = batch['fixed_pos'] == 0
        unfixed_halfedge = batch['fixed_halfedge'] == 0
        
        
        # # for fixed dist loss
        nodes_0, nodes_1 = batch['halfedge_index']
        unfixed_dist = (batch['fixed_halfdist'] == 0)
        dist_pred = torch.norm(pred_pos[nodes_0] - pred_pos[nodes_1], dim=-1)
        dist_true = torch.norm(node_pos[nodes_0] - node_pos[nodes_1], dim=-1)
        loss_dist_total = F.mse_loss(dist_pred, dist_true, reduction='none')
        loss_dist = torch.mean(loss_dist_total[unfixed_dist]).nan_to_num()
        loss_dist_fixed = torch.mean(loss_dist_total[~unfixed_dist]).nan_to_num()
        
        # # for dihedral loss
        if 'dih_sin' in outputs:
            sin_gt, cos_gt = get_dihedral_batch(node_pos, batch['tor_bonds_anno'], batch['dihedral_pairs_anno'])
            sin_pred, cos_pred = outputs['dih_sin'], outputs['dih_cos']
            cos_delta = cos_pred * cos_gt + sin_pred * sin_gt
            # sin_delta = sin_pred * cos_gt - cos_pred * sin_gt
            loss_dih = torch.mean(1 - cos_delta).nan_to_num()
        else:
            loss_dih = torch.tensor(0.0, device=node_pos.device)
        
        
        # # pos
        loss_pos_total = F.mse_loss(pred_pos, node_pos, reduction='none')
        loss_pos = torch.mean(loss_pos_total[unfixed_pos]).nan_to_num()
        loss_pos_fixed = torch.mean(loss_pos_total[~unfixed_pos]).nan_to_num()

        # # node & edge type
        loss_node_total = self.ce_loss(pred_node, node_type)
        loss_node = torch.mean(loss_node_total[unfixed_node]).nan_to_num()
        loss_node_fixed = torch.mean(loss_node_total[~unfixed_node]).nan_to_num()
        
        loss_edge_total = self.ce_loss(pred_halfedge, halfedge_type)
        loss_edge = torch.mean(loss_edge_total[unfixed_halfedge]).nan_to_num()
        loss_edge_fixed = torch.mean(loss_edge_total[~unfixed_halfedge]).nan_to_num()
        
        # # total
        loss_fixed = (loss_pos_fixed * self.config['weights']['pos'] +\
                     loss_node_fixed * self.config['weights']['node'] +\
                     loss_edge_fixed * self.config['weights']['edge'])
        loss_total = loss_node * self.config['weights']['node'] + \
                     loss_edge * self.config['weights']['edge'] + \
                     loss_fixed * self.config['fixed_weight'] + \
                         loss_dist_fixed * self.config['weights']['fixed_dist']  # dist_fixed is default added
                     
        if 'dist' in self.config['use_pos_losses']:
            loss_total += loss_dist * self.config['weights']['dist']  # unfixed dist loss
        if 'pos' in self.config['use_pos_losses']:
            loss_total += loss_pos * self.config['weights']['pos']
        if 'dihedral' in self.config['use_pos_losses']:
            if 'dih_sin' not in outputs:
                raise ValueError('dihedral loss is not calculated!')
            loss_total += loss_dih * self.config['weights']['dihedral']
        
        loss_dict = {
            'loss': loss_total,
            # 'loss_node': loss_node,
            # 'loss_edge': loss_edge,
            'loss_fixed': loss_fixed,
            'loss_dist_fixed': loss_dist_fixed,

            'loss_dist': loss_dist,
            'loss_pos': loss_pos,
            'loss_dih': loss_dih,
        }
        return loss_dict


@register_loss('refine_torsional')
class RefineTorsionalLoss(nn.Module):
    def __init__(self, config):
        raise NotImplementedError('use refine_torsional2')
        super().__init__()
        self.config = config
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, batch, outputs):
        pred_pos = outputs['pred_pos']
        pred_node = outputs['pred_node']
        pred_halfedge = outputs['pred_halfedge']
        pred_pos_corr = outputs['pred_pos_corr']
        
        node_pos = batch['node_pos']
        node_type = batch['node_type']
        halfedge_type = batch['halfedge_type']
        
        # divide into fixed and unfixed
        unfixed_node = batch['fixed_node'] == 0
        unfixed_pos = batch['fixed_pos'] == 0
        unfixed_halfedge = batch['fixed_halfedge'] == 0
        
        
        # # for fixed dist loss
        fixed_halfdist = batch['fixed_halfdist']
        halfedge_index = batch['halfedge_index']
        halfedge_index_fixed_dist = halfedge_index[:, fixed_halfdist==1]
        nodes_0, nodes_1 = halfedge_index_fixed_dist
        dist_pred = torch.norm(pred_pos[nodes_0] - pred_pos[nodes_1], dim=-1)
        dist_true = torch.norm(node_pos[nodes_0] - node_pos[nodes_1], dim=-1)
        loss_dist = F.mse_loss(dist_pred, dist_true)
        
        # # for dihedral loss
        index_tor = batch['dihedral_pairs_anno'][:, 0]
        dihedral_ends = batch['dihedral_pairs_anno'][:, 1:]  # (n_dih, 2)
        dihedral_tor_nodes = batch['tor_bonds_anno'][:, 1:][index_tor]  # (n_dih, 2)
        sin_gt, cos_gt = get_dihedral(node_pos[dihedral_ends[:, 0]], node_pos[dihedral_tor_nodes[:, 0]],
                        node_pos[dihedral_tor_nodes[:, 1]], node_pos[dihedral_ends[:, 1]])
        sin_pred, cos_pred = outputs['dih_sin'], outputs['dih_cos']
        cos_delta = cos_pred * cos_gt + sin_pred * sin_gt
        sin_delta = sin_pred * cos_gt - cos_pred * sin_gt
        loss_dih = torch.mean(1 - cos_delta).nan_to_num() + \
                    0.5 * F.mse_loss(sin_delta, torch.zeros_like(sin_delta))
        
        # # pos
        loss_pos_total = F.mse_loss(pred_pos, node_pos, reduction='none')
        loss_pos = torch.mean(loss_pos_total[unfixed_pos]).nan_to_num()
        loss_pos_fixed = torch.mean(loss_pos_total[~unfixed_pos]).nan_to_num()
        
        # # pos corr
        loss_pos_corr_total = F.mse_loss(pred_pos_corr, node_pos, reduction='none')
        loss_pos_corr = torch.mean(loss_pos_corr_total[unfixed_pos]).nan_to_num()

        # # node & edge type
        loss_node_total = self.ce_loss(pred_node, node_type)
        loss_node = torch.mean(loss_node_total[unfixed_node]).nan_to_num()
        loss_node_fixed = torch.mean(loss_node_total[~unfixed_node]).nan_to_num()
        
        loss_edge_total = self.ce_loss(pred_halfedge, halfedge_type)
        loss_edge = torch.mean(loss_edge_total[unfixed_halfedge]).nan_to_num()
        loss_edge_fixed = torch.mean(loss_edge_total[~unfixed_halfedge]).nan_to_num()
        
        # # total
        loss_fixed = (loss_pos_fixed * self.config['weights']['pos'] +\
                     loss_node_fixed * self.config['weights']['node'] +\
                     loss_edge_fixed * self.config['weights']['edge'])
        loss_total = loss_node * self.config['weights']['node'] + \
                     loss_edge * self.config['weights']['edge'] + \
                     loss_fixed * self.config['fixed_weight']
                     
        if 'fixed_dist' in self.config['use_pos_losses']:
            loss_total += loss_dist * self.config['weights']['dist']
        if 'pos' in self.config['use_pos_losses']:
            loss_total += loss_pos * self.config['weights']['pos']
        if 'pos_corr' in self.config['use_pos_losses']:
            loss_total += loss_pos_corr * self.config['weights']['pos_corr']
        if 'dihedral' in self.config['use_pos_losses']:
            loss_total += loss_dih * self.config['weights']['dihedral']
        
        loss_dict = {
            'loss': loss_total,
            # 'loss_node': loss_node,
            # 'loss_edge': loss_edge,
            'loss_fixed': loss_fixed,

            'loss_pos': loss_pos,
            'loss_dist': loss_dist,
            'loss_pos_corr': loss_pos_corr,
            'loss_dih': loss_dih,
        }
        return loss_dict
    

@register_loss('refine')
class RefineLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.node_transition = deepcopy(node_transition)
        # self.edge_transition = deepcopy(edge_transition)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, batch, outputs):
        pred_pos = outputs['pred_pos']
        pred_node = outputs['pred_node']
        pred_halfedge = outputs['pred_halfedge']
        
        node_pos = batch['node_pos']
        node_type = batch['node_type']
        halfedge_type = batch['halfedge_type']
        # log_node_0 = batch['log_node_0']
        # log_halfedge_0 = batch['log_halfedge_0']

        # log_node_t = batch['log_node_t']
        # log_halfedge_t = batch['log_halfedge_t']
        # time_node = batch['time_node']
        # time_halfedge = batch['time_halfedge']
        
        # divide into fixed and unfixed
        unfixed_node = batch['fixed_node'] == 0
        unfixed_pos = batch['fixed_pos'] == 0
        unfixed_halfedge = batch['fixed_halfedge'] == 0
        
        # # pos
        loss_pos_total = F.mse_loss(pred_pos, node_pos, reduction='none')
        loss_pos = torch.mean(loss_pos_total[unfixed_pos]).nan_to_num()
        loss_pos_fixed = torch.mean(loss_pos_total[~unfixed_pos]).nan_to_num()

        # # node & edge type
        loss_node_total = self.ce_loss(pred_node, node_type)
        loss_node = torch.mean(loss_node_total[unfixed_node]).nan_to_num()
        loss_node_fixed = torch.mean(loss_node_total[~unfixed_node]).nan_to_num()
        
        loss_edge_total = self.ce_loss(pred_halfedge, halfedge_type)
        loss_edge = torch.mean(loss_edge_total[unfixed_halfedge]).nan_to_num()
        loss_edge_fixed = torch.mean(loss_edge_total[~unfixed_halfedge]).nan_to_num()
        
        # # total
        loss_fixed = (loss_pos_fixed * self.config['weights']['pos'] +\
                     loss_node_fixed * self.config['weights']['node'] +\
                     loss_edge_fixed * self.config['weights']['edge'])
        loss_total = loss_pos * self.config['weights']['pos'] + \
                     loss_node * self.config['weights']['node'] + \
                     loss_edge * self.config['weights']['edge'] + \
                     loss_fixed * self.config['fixed_weight']
        
        loss_dict = {
            'loss': loss_total,
            'loss_pos': loss_pos,
            'loss_node': loss_node,
            'loss_edge': loss_edge,
            'loss_fixed': loss_fixed,
        }
        return loss_dict
    

@register_loss('confidence')
class ConfidenceLoss(nn.Module):
    """
    The confidence loss
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        prob_1A = config['prob_1A']
        if prob_1A > 0 and prob_1A < 1:
            self.register_buffer('log_prob_1A', torch.tensor(np.log(prob_1A)))
        else:  # directly predict minus error
            self.register_buffer('log_prob_1A', torch.tensor(1.0))

        self.binary_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, batch, outputs):
        
        # prediction
        pred_node = outputs['pred_node']
        pred_pos = outputs['pred_pos']
        pred_halfedge = outputs['pred_halfedge']

        # gt data
        node_type = batch['node_type']
        node_pos = batch['node_pos']
        halfedge_type = batch['halfedge_type']
        
        # is right prdiction
        right_node = (node_type == pred_node.argmax(-1).detach()).to(pred_node.dtype)
        right_pos = torch.norm(node_pos - pred_pos, dim=-1).detach()
        if self.log_prob_1A < 0:  # output as logits of cfd prob
            right_pos = torch.exp(right_pos * self.log_prob_1A)
        right_halfedge = (halfedge_type == pred_halfedge.argmax(-1).detach()).to(pred_node.dtype)
        
        # confidence
        confidence_node = outputs['confidence_node']
        confidence_pos = outputs['confidence_pos']
        confidence_halfedge = outputs['confidence_halfedge']
        
        # loss
        loss_node = self.binary_loss(confidence_node[..., 0], right_node)
        # loss_pos = F.mse_loss(confidence_pos[..., 0], right_pos)  # need sigmoid ??
        if self.log_prob_1A < 0: # output as logits of cfd prob
            loss_pos = F.mse_loss(torch.sigmoid(confidence_pos[..., 0]), right_pos)  # need sigmoid ??
        else: # directly predict minus error
            loss_pos = F.mse_loss(confidence_pos[..., 0], -right_pos)
        loss_halfedge = self.binary_loss(confidence_halfedge[..., 0], right_halfedge)

        loss_total = loss_node * self.config['weights']['node'] + \
                     loss_pos * self.config['weights']['pos'] + \
                     loss_halfedge * self.config['weights']['edge']
                     
        loss_dict = {
            'cfd_total': loss_total,
            'cfd_node': loss_node,
            'cfd_pos': loss_pos,
            'cfd_edge': loss_halfedge,
        }
        return loss_dict

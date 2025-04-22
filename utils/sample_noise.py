"""
Define the noise controller to facilitate noise injection and sampling.
"""

from easydict import EasyDict
from torch_scatter import scatter_mean, scatter_min
from tqdm import tqdm
import torch
from copy import deepcopy
from scipy.optimize import linear_sum_assignment

from torch_geometric.data import Batch
from torch_geometric.nn import radius
# from torch.nn import Module
# from torch.nn import functional as F
# from models.transition import ContigousTransition, GeneralCategoricalTransition
from models.diffusion import *
from models.corrector import correct_pos_batch, kabsch_flatten, correct_pos_batch_no_tor, grad_len_to_pos, correct_pos_by_fixed_dist_batch
from utils.data import Mol3DData
from utils.prior import MolPrior
from utils.info_level import MolInfoLevel
from utils.shape import get_points_from_letter

SAMPLE_NOISE_DICT = {}
def register_sample_noise(name):
    def add_to_registery(cls):
        SAMPLE_NOISE_DICT[name] = cls
        return cls
    return add_to_registery


def get_sample_noiser(config, num_node_types, num_edge_types, *args, **kwargs):
    name = config['name']
    return SAMPLE_NOISE_DICT[name](config, num_node_types, num_edge_types, *args, **kwargs)


def dict_list2item(inputs):
    if isinstance(inputs, dict):
        return {key: dict_list2item(value) for key, value in inputs.items()}
    elif isinstance(inputs, list):
        assert len(set(inputs)), 'all task_setting should be the same for sampling'
        return inputs[0]
    else:
        return inputs

class BaseSampleNoiser:
    def __init__(self, task_name, config, num_node_types, num_edge_types,
                mode, device, ref_config=None, pos_only=False, *args, **kwargs):
        super().__init__()
        self.config = config
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.mode = mode
        self.device = device
        self.pos_only = pos_only
        

        if mode == 'sample':
            self.num_steps = config.num_steps
            self.init_step = config.get('init_step', 1)
            if ref_config is not None:
                if ref_config.name == 'mixed':
                    ref_config = [c for c in ref_config.individual if c.name==task_name]
                    if len(ref_config) > 0:
                        ref_config = ref_config[0]
                    else:
                        ref_config = None
                self.ref_prior_config = getattr(ref_config, 'prior', None)
            else:
                self.ref_prior_config = None

        
    def add_noise(self, *args, **kwargs):
        """
        Input: 
            node_type
            node_pos
            halfedge_type
            batch
            from_prior: default False
            level_dict: dictionary of info_level, default none
        Output:
            in_dict: dictionary with keys [node, pos, halfedge]
        """
        raise NotImplementedError('Please implement this for each task.')
    def sample_level(self, *args, **kwargs):
        """
        Input:
            step
            batch
        Output:
            level_dict: dictionary of info_level
        """
        raise NotImplementedError('Please implement this for each task.')

    def outputs2batch(self, *args, **kwargs):
        """
        Input:
            batch
            outputs
        Output:
            batch
        """
        raise NotImplementedError('Please implement this for each task.')
    
    def _get_task(self, batch):
        task = batch['task']
        if isinstance(task, list):
            assert len(set(task)), 'all task should be the same for sampling'
            task = task[0]
        return task
    
    def _get_setting(self, batch):
        setting = dict_list2item(batch['task_setting'])
        return setting
    
    def _fetch_data(self, batch):
        # from gt data batch. for training or sampling step=1
        node_type = batch.node_type 
        node_pos = batch.node_pos
        halfedge_type = batch.halfedge_type
        
        return [node_type.detach().clone(),
                node_pos.detach().clone(),
                halfedge_type.detach().clone()]

    def reassign_in_node(self, batch, in_dict):
        return in_dict

    
    def spring_in_pos(self, batch, in_dict):
        if batch['fixed_halfdist'].sum() > 0:  # if has fixed_dist, no need to spring bonds since lens are not perturbed
            return in_dict
        
        # get parameter
        spring_in = self.config['spring_in']
        iters = spring_in['iters']
        lr = spring_in['lr']

        # get bond index
        in_halfedge = in_dict['halfedge']
        halfedge_index = batch['halfedge_index']
        is_bond = (in_halfedge > 0) & (in_halfedge < self.num_edge_types)
        if getattr(spring_in, 'inout_bond', False):
            halfedge_type = batch['halfedge_type']
            is_bond = is_bond & (halfedge_type > 0) & (halfedge_type < self.num_edge_types)
        halfbond_index = halfedge_index[:, is_bond]
        bond_index = torch.cat([halfbond_index, halfbond_index.flip(0)], dim=-1)

        # fetch relevant pos
        in_pos = in_dict['pos']
        fixed_pos = batch['fixed_pos'].bool()
        

        # spring the pos
        with torch.enable_grad():
            # in_pos.requires_grad_(True)
            for i in range(iters):
                # check
                value_min, value_max = spring_in['min'], spring_in['max']
                std = spring_in['std']
                lens = torch.linalg.norm(in_pos[bond_index[0]] - in_pos[bond_index[1]], dim=-1)
                square = torch.where(lens < value_min, (lens - value_min) ** 2, torch.zeros_like(lens))
                square = torch.where(lens > value_max, (lens - value_max) ** 2, square)
                loss = torch.mean( square.sqrt() / std)
                
                grad = grad_len_to_pos(in_pos, bond_index, spring_in)
                # grad = torch.autograd.grad(loss, in_pos)[0]
                in_pos[~fixed_pos] = in_pos[~fixed_pos] - grad[~fixed_pos] * lr

        in_dict['pos'] = in_pos
        return in_dict
    
    def additional_process(self, batch, in_dict):
        return in_dict

    @torch.no_grad()
    def __call__(self, batch, step=None):

        # node_pos_protect = deepcopy(batch.node_pos.detach().clone())
        # # check mode and inputs consistency
        # from_batch, step = self._check_input(batch, outputs, step)
        # device = batch.node_type.device
        
        # # fetch data from batch or outputs
        node_type, node_pos, halfedge_type = self._fetch_data(batch)

        # # get info level: task-specific
        level_dict = self.sample_level(step, batch)
        
        # # adding noise after moveing all nodes to their origins
        if step == 1:
            in_dict = self.add_noise(node_type, node_pos, halfedge_type, batch,
                from_prior=True, level_dict=level_dict)  # here only the key of level_dict=level_dict is useful
        else:
            # level_dict = {k:torch.ones_like(v) for k,v in level_dict.items()}
            in_dict = self.add_noise(node_type, node_pos, halfedge_type, batch,
                                        from_prior=False, level_dict=level_dict)
        
        if self.config.get('spring_in', False):
            in_dict = self.spring_in_pos(batch, in_dict)

        if self.config.get('reassign_in', False):
            in_dict = self.reassign_in_node(batch, in_dict)
            
        in_dict = self.additional_process(batch, in_dict)
            
        
        # # use fixed_dict to reset for fixed==1
        if not (batch.node_type[batch['fixed_node']==1] ==  in_dict['node'][batch['fixed_node']==1]).all():
            print('Force fixed_node to be fixed')
            is_fixed = (batch['fixed_node']==1)
            in_dict['node'][is_fixed] = batch.node_type[is_fixed]
        # assert (batch.node_pos[(batch['fixed_pos']==1)]
        #         - in_dict['pos'][batch['fixed_pos']==1]).abs().sum() < 1e-4
        if not (batch.node_pos[(batch['fixed_pos']==1)]
                - in_dict['pos'][batch['fixed_pos']==1]).abs().sum() < 1e-4:
            in_dict['pos'][batch['fixed_pos']==1] = batch.node_pos[batch['fixed_pos']==1]
        assert (batch.halfedge_type[batch['fixed_halfedge']==1]
                ==  in_dict['halfedge'][batch['fixed_halfedge']==1]).all()

        batch.update({f'{key}_in':value for key, value in in_dict.items()})
        
        return batch

    
    def steps_loop(self, add_last=False):
        # for step in np.linspace(0, 1, self.num_steps)[::-1]:
        if add_last:
            steps = np.arange(0, self.num_steps+1) # 0, 1, ..., num_steps
        else:
            steps = np.arange(1, self.num_steps+1)  # 1, 2, ..., num_steps
        for step in steps[::-1]:
            yield (step / self.num_steps) * self.init_step # include 1 but not 0


@register_sample_noise('mixed')
class MixedSampleNoiser:
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.noiser_dict = {}
        for task_cfg in config.individual:
            self.noiser_dict[task_cfg.name] = get_sample_noiser(task_cfg, *args, **kwargs)

    def __call__(self, batch, *args, **kwargs):
        task = batch['task']
        if isinstance(task, list):
            task = task[0]
        return self.noiser_dict[task](batch, *args, **kwargs)


@register_sample_noise('dynamic')
class DynamicSettingSampleNoiser:
    def __init__(self, config, *args, **kwargs):
        self.config = config
        # process phase
        self.phases = config.phases
        self.total_steps = sum(self.phases.num_steps)
        self.cum_steps = np.cumsum(self.phases.num_steps)

        base_noise_cfg = config.base_noise
        base_noise_cfg.num_steps = self.total_steps
        self.base_noiser = get_sample_noiser(base_noise_cfg, *args, **kwargs)

    def __call__(self, batch, *args, **kwargs):
        step = kwargs['step']
        global_step = self.total_steps * (1 - step)
        index_stage = np.where(step > self.cum_steps)[0]
        settings_this_stage = self.phases.settings[index_stage]
        
        # renew setting step
        phase_step = self.phases.num_steps[index_stage]
        phase_interval = self.phases.step_intervals[index_stage]
        phase_bins = phase_interval / phase_step
        phase_local_step = global_step - max(self.cum_steps[index_stage-1], 0)
        step = phase_interval[0] - phase_bins * phase_local_step
        assert step > phase_interval[1], 'step out of interval'
        kwargs.update({'step': step})
        
        # supress settings
        batch = self._overwrite_settings(batch, settings_this_stage)
        return self.base_noiser(batch, *args, **kwargs)
        
    def _overwrite_settings(self, batch, new_settings):
        old_settings = batch.setting
        if isinstance(old_settings, list):
            assert len(old_settings) == len(set(old_settings)), 'settings are not unique for the batch'
            new_settings = [new_settings for _ in old_settings]
        else:
            raise NotImplementedError('not implement for the setting types')
        batch.update({'settings': new_settings})
        return batch
    
    def steps_loop(self, *args, **kwargs):
        return self.base_noiser.steps_loop(*args, **kwargs)

    def outputs2batch(self, batch, outputs):
        return self.base_noiser.outputs2batch(batch, outputs)


@register_sample_noise('denovo')
class DenovoSampleNoiser(BaseSampleNoiser):
    def __init__(self,
        config, num_node_types, num_edge_types,
        mode='sample', device='cpu', ref_config=None, task_name='denovo',
        **kwargs
    ):
        super().__init__(task_name, config, num_node_types, num_edge_types,
                mode, device, ref_config, **kwargs)
        
        # define prior
        prior_config = config.prior if config.prior != 'from_train' else self.ref_prior_config
        self.prior = MolPrior(prior_config, num_node_types, num_edge_types).to(device)

        # define info level
        self.level = MolInfoLevel(config.level, device=device, mode=mode)
        
        self.post_process = config.get('post_process', None)
        

    def sample_level(self, step, batch):
        level_dict = {}
        level_node, level_pos, level_halfedge = self.level.sample_for_mol(
            step,
            n_node=batch['node_type'].shape[0],
            n_pos=batch['node_type'].shape[0],
            n_edge=batch['halfedge_type'].shape[0],
        )
        level_dict.update({
            f'node': level_node,
            f'pos': level_pos,
            f'halfedge': level_halfedge,
        })
        
        
        if 'scaling_level_node' in batch:
            level_dict['node'] = level_dict['node'] ** batch['scaling_level_node']
        elif 'scaling_noise_node' in batch:
            level_dict['node'] = 1 - (1 - level_dict['node']) * batch['scaling_noise_node']
        if 'scaling_level_pos' in batch:
            level_dict['pos'] = level_dict['pos'] ** batch['scaling_level_pos']
        elif 'scaling_noise_pos' in batch:
            level_dict['pos'] = 1 - (1 - level_dict['pos']) * batch['scaling_noise_pos']
        if 'scaling_level_halfedge' in batch:
            level_dict['halfedge'] = level_dict['halfedge'] ** batch['scaling_level_halfedge']
        elif 'scaling_noise_halfedge' in batch:
            level_dict['halfedge'] = 1 - (1 - level_dict['halfedge']) * batch['scaling_noise_halfedge']
        
        return level_dict

    def add_noise(self, node_type, node_pos, halfedge_type, batch,
                  from_prior=False, level_dict=None):
        
        task = self._get_task(batch)
        # # recenter before add noise
        if (task == 'denovo'):
            batch_node = getattr(batch, 'node_type_batch',
                        torch.zeros(node_type.shape[0], dtype=torch.long, device=node_pos.device))  
            node_pos_center = scatter_mean(node_pos, batch_node, dim=0, dim_size=batch_node.max()+1)[batch_node]
            node_pos = node_pos - node_pos_center
            if self.mode == 'train':
                batch.update({'node_pos': node_pos.clone()})
        
        noised_data = self.prior.add_noise(
            node_type, node_pos, halfedge_type, 
            level_dict=level_dict, from_prior=from_prior)
        pos_in = noised_data[1]

        # # recenter after add noise
        if (task == 'denovo'):
            batch_node = getattr(batch, 'node_type_batch',
                        torch.zeros(node_type.shape[0], dtype=torch.long, device=node_pos.device))
            pos_in_center = scatter_mean(pos_in, batch_node, dim=0, dim_size=batch_node.max()+1)[batch_node]
            pos_in = pos_in - pos_in_center

        in_dict = {'node':noised_data[0], 'pos':pos_in, 'halfedge':noised_data[2]}
        return in_dict
    
    def outputs2batch(self, batch, outputs):
        
        if self.post_process is None:
            batch['node_type'] = outputs['pred_node'].argmax(-1)
            batch['node_pos'] = outputs['pred_pos']
            batch['halfedge_type'] = outputs['pred_halfedge'].argmax(-1)
        elif self.post_process == 'redock':
            fixed_node = batch['fixed_node']
            fixed_node_bool = (fixed_node == 1)
            fixed_halfedge = batch['fixed_halfedge']
            fixed_halfedge_bool = (fixed_halfedge == 1)
            batch['node_type'][~fixed_node_bool] = outputs['pred_node'].argmax(-1)[~fixed_node_bool]
            batch['halfedge_type'][~fixed_halfedge_bool] = outputs['pred_halfedge'].argmax(-1)[~fixed_halfedge_bool]
            batch['node_pos'] = outputs['pred_pos']

            redock_config = self.config.redock
            start_step = redock_config.start_step
            step = batch['step']

            if step <= start_step:  # now in dock mode
                fixed_node = torch.ones_like(fixed_node)
                fixed_halfedge = torch.ones_like(fixed_halfedge)
                batch['fixed_node'] = fixed_node
                batch['fixed_halfedge'] = fixed_halfedge
        elif self.post_process == 'corr_shape':
            step = batch['step']
            cfg_shape = self.config['corr_shape']
            corr_th_step = cfg_shape.get('corr_th_shape', 0.1)
            
            if step > corr_th_step:
                letter = cfg_shape['letter']
                length = cfg_shape.get('length', 12)
                height = cfg_shape.get('height', 2)
                corr_th_dist = cfg_shape.get('corr_th_dist', 2)
                
                delta_all = []
                pred_pos = outputs['pred_pos']
                for i_batch in range(batch['node_type_batch'].max() + 1):
                    this_batch = (batch['node_type_batch'] == i_batch)
                    
                    pred_pos_batch = pred_pos[this_batch].detach().cpu().numpy()
                    n_points = pred_pos_batch.shape[0]
                    shape_points = get_points_from_letter(letter, n_points, length=length, height=height)
                    
                    # calc dist mat and match
                    dist_mat = np.linalg.norm(pred_pos_batch[:, None] - shape_points[None], axis=-1)
                    row_ind, col_ind = linear_sum_assignment(dist_mat)
                    
                    # get delta
                    shape_points = shape_points[col_ind]
                    delta_vec = shape_points - pred_pos_batch
                    delta_dist = np.linalg.norm(delta_vec, axis=-1, keepdims=True)
                    delta_vec = np.where(delta_dist > corr_th_dist, delta_vec * (step - corr_th_step)/(1-corr_th_step), 0)
                    delta_all.append(delta_vec)
                delta_all = np.concatenate(delta_all, axis=0)
                delta_all = torch.tensor(delta_all, dtype=pred_pos.dtype, device=pred_pos.device)
            else:
                delta_all = 0
            
            batch['node_pos'] = outputs['pred_pos'] + delta_all
            batch['node_type'] = outputs['pred_node'].argmax(-1)
            batch['halfedge_type'] = outputs['pred_halfedge'].argmax(-1)
            
        else:
            raise NotImplementedError('not implement for the post_process types')
        
        if self.config.get('scaling_level', False):
            cfd_node = torch.sigmoid(outputs['confidence_node'][:, 0])
            cfd_pos = torch.sigmoid(outputs['confidence_pos'][:, 0])
            cfd_halfedge = torch.sigmoid(outputs['confidence_halfedge'][:, 0])

            batch_node = batch['node_type_batch']
            batch_halfedge = batch['halfedge_type_batch']
            n_batch = batch_node.max() + 1

            scaling_level_node = []
            scaling_level_pos = []
            scaling_level_halfedge = []
            for i_batch in range(n_batch):
                cfd_node_this = cfd_node[batch_node==i_batch]
                cfd_pos_this = cfd_pos[batch_node==i_batch]
                cfd_halfedge_this = cfd_halfedge[batch_halfedge==i_batch]
                
                s = self.config.scaling_level.s
                # scaling_node = 1 - ((cfd_node_this - cfd_node_this.min()) / (cfd_node_this.max() - cfd_node_this.min() + 1e-8)) * 2
                scaling_node = torch.median(cfd_node_this) - cfd_node_this
                scaling_node = scaling_node / scaling_node.max()
                scaling_node = scaling_node.clamp(min=0)
                scaling_node = s ** scaling_node  # 2 -> 0.5: cfd low -> high
                scaling_level_node.append(scaling_node)
                
                # scaling_pos = 1 - ((cfd_pos_this - cfd_pos_this.min()) / (cfd_pos_this.max() - cfd_pos_this.min() + 1e-8)) * 2
                scaling_pos = torch.median(cfd_pos_this) - cfd_pos_this
                scaling_pos = scaling_pos / scaling_pos.max()
                scaling_pos = scaling_pos.clamp(min=0)
                scaling_pos = s ** scaling_pos
                scaling_level_pos.append(scaling_pos)
                
                # scaling_halfedge = 1 - ((cfd_halfedge_this - cfd_halfedge_this.min()) / (cfd_halfedge_this.max() - cfd_halfedge_this.min() + 1e-8)) * 2
                scaling_halfedge = torch.median(cfd_halfedge_this) - cfd_halfedge_this
                scaling_halfedge = scaling_halfedge / scaling_halfedge.max()
                scaling_halfedge = scaling_halfedge.clamp(min=0)
                scaling_halfedge = s ** scaling_halfedge
                scaling_level_halfedge.append(scaling_halfedge)
            
            scaling_level_node = torch.cat(scaling_level_node)
            scaling_level_pos = torch.cat(scaling_level_pos)
            scaling_level_halfedge = torch.cat(scaling_level_halfedge)
            batch.update({
                'scaling_level_node': scaling_level_node,
                'scaling_level_pos': scaling_level_pos,
                'scaling_level_halfedge': scaling_level_halfedge,
            })
        elif self.config.get('shift_level', False):
            end_step = self.config.shift_level.end_step
            step = batch['step']

            scaling = (step - end_step) / (self.init_step - end_step) + 0.01
            scaling = np.clip(scaling, 0.01, 1)
            batch.update({
                'scaling_noise_node': scaling,
                'scaling_noise_pos': scaling,
                'scaling_noise_halfedge': scaling,
            })
        elif self.config.get('shift_typelevel', False):
            end_step = self.config.shift_typelevel.end_step
            step = batch['step']

            scaling = (step - end_step) / (self.init_step - end_step) + 0.01
            scaling = np.clip(scaling, 0.01, 1)
            batch.update({
                'scaling_noise_node': scaling,
                'scaling_noise_halfedge': scaling,
            })
        
        
        return batch
    
    def additional_process(self, batch, in_dict):
        if self.post_process == 'redock':
            fixed_node = batch['fixed_node'].bool()
            fixed_halfedge = batch['fixed_halfedge'].bool()
            in_dict['node'][fixed_node] = batch['node_type'][fixed_node].clone()
            in_dict['halfedge'][fixed_halfedge] = batch['halfedge_type'][fixed_halfedge].clone()
        return in_dict


@register_sample_noise('sbdd')
class SBDDSamplNoiser(DenovoSampleNoiser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, task_name='sbdd')


@register_sample_noise('conf')
class ConfSampleNoiser(BaseSampleNoiser):
    def __init__(self,
        config, num_node_types, num_edge_types,
        mode='sample', device='cpu', ref_config=None, task_name='conf',
        **kwargs
    ):
        super().__init__(task_name, config, num_node_types, num_edge_types,
            mode, device, ref_config, pos_only=True, **kwargs)
        
        # define prior
        prior_config = config.prior if config.prior != 'from_train' else self.ref_prior_config
        self.prior = MolPrior(prior_config, num_node_types, num_edge_types, 
                              pos_only=True).to(device)

        # define info level
        self.level = MolInfoLevel(config.level, device=device, mode=mode)
        
        self.pre_process = config.get('pre_process', None)
        self.post_process = config.get('post_process', None)

    def sample_level(self, step, batch):
        setting = self._get_setting(batch)
        # setting = 'free'
        if setting == 'free':
            level_pos = self.level.sample_for_mol(step,
                n_pos=batch['node_type'].shape[0],
            )
            level_dict = {'pos': level_pos}
        elif setting == 'flexible':
            n_trans = getattr(batch, 'num_graphs', 1)
            n_rot = getattr(batch, 'num_graphs', 1)
            n_tor = batch['tor_bonds_anno'].shape[0]
            L_trans, L_rot, L_tor = self.level.sample_for_mol(step,
                n_trans=n_trans,
                n_rot=n_rot,
                n_tor=n_tor,
            )
            level_dict = {'trans':L_trans, 'rot':L_rot, 'tor': L_tor}
        elif setting == 'torsional':
            n_tor = batch['tor_bonds_anno'].shape[0]
            level_tor = self.level.sample_for_mol(step,
                n_tor=n_tor,
            )
            level_dict = {'tor': level_tor}
        elif setting == 'rigid':
            n_trans = getattr(batch, 'num_graphs', 1)
            n_rot = getattr(batch, 'num_graphs', 1)
            L_trans, L_rot = self.level.sample_for_mol(step,
                n_trans=n_trans,
                n_rot=n_rot,
            )
            level_dict = {'trans':L_trans, 'rot':L_rot}
        return level_dict

    def add_noise(self, node_type, node_pos, halfedge_type, batch,
                   from_prior=False, level_dict=None):
        # batch_node = getattr(batch, 'node_type_batch', 
        #                      torch.zeros(node_type.shape[0], dtype=torch.long, device=device))
        # node_pos = node_pos - scatter_mean(node_pos, batch_node, dim=0, dim_size=batch_node.max()+1)[batch_node]
        task = self._get_task(batch)
        setting = self._get_setting(batch)
        # setting = 'free'
        additional_kwargs = {
            'node_type': None, 'halfedge_type': None,
        }
        if setting == 'free':
            # pass
            if 'node_type_batch' in batch:
                mol_size = torch.bincount(batch['node_type_batch'])
                mol_size = mol_size[batch['node_type_batch']]
            else:
                mol_size = batch['num_nodes'] * torch.ones(node_type.shape[0], dtype=torch.long, device=node_type.device)
            additional_kwargs.update({
                'mol_size': mol_size,
            })
        elif setting in ['flexible', 'torsional']:
            additional_kwargs.update({
                'tor_bonds_anno': batch['tor_bonds_anno'],
                'twisted_nodes_anno': batch['twisted_nodes_anno'],
                'domain_node_index': batch['domain_node_index'],
            })
            # if setting == 'flexible':
            #     additional_kwargs.update({
            #         'domain_center_nodes': batch['domain_center_nodes'],
            #     })
        elif setting == 'rigid':
            additional_kwargs.update({
                'domain_node_index': batch['domain_node_index'],
            })
                
        # # recenter before add_noise
        if (task == 'conf'):
            batch_node = getattr(batch, 'node_type_batch',
                        torch.zeros(node_type.shape[0], dtype=torch.long, device=node_pos.device))
            node_pos_center = scatter_mean(node_pos, batch_node, dim=0, dim_size=batch_node.max()+1)[batch_node]
            node_pos = node_pos - node_pos_center
            if self.mode == 'train':
                batch.update({'node_pos': node_pos.clone()})  # can be ommited since featurizer has done this
            
        pos_in = self.prior.add_noise(node_pos=node_pos.clone(), level_dict=level_dict,
                                      from_prior=from_prior, **additional_kwargs,)
        
        # # recenter after add_noise
        if (task == 'conf'):
            batch_node = getattr(batch, 'node_type_batch',
                        torch.zeros(node_type.shape[0], dtype=torch.long, device=node_pos.device))
            pos_in_center = scatter_mean(pos_in, batch_node, dim=0, dim_size=batch_node.max()+1)[batch_node]
            pos_in = pos_in - pos_in_center
            if (setting == 'free') and (self.config.get('recenter', 'default') == 'norotate'):
                # decouple rotation and translation
                domain_index = batch_node
                global_rot, global_trans = kabsch_flatten(pos_in, node_pos, domain_index)
        
                # apply global rotation and translation
                pos_corrected_expand = torch.matmul(
                    pos_in[:, None, :],
                    global_rot.transpose(1, 2)[domain_index]
                ) + global_trans[domain_index]

                pos_in = pos_corrected_expand.squeeze(1)
                
        
        in_dict = {'node':node_type, 'pos':pos_in, 'halfedge':halfedge_type}
        
        return in_dict

    def reassign_in_node(self, batch, in_dict):
        if (self.mode == 'train') and ('is_atom_remain' not in batch):  # not applicable for cutt peptide
            matches_iso = batch['matches_iso']
            base_order = np.sort(matches_iso[0])
            # assert (np.diff(base_order)>0).sum(), 'base_order should be in ascending order'
            node_pos = batch['node_pos'][base_order]  # (n, 3)
            in_pos_orig = in_dict['pos']  # (N, 3)
            in_pos_syms = in_pos_orig[matches_iso, ...]  # (M, n, 3)
            rmsd = torch.norm(node_pos[None] - in_pos_syms, p=2, dim=-1)  # (M, n)
            rmsd_mean = torch.mean(rmsd, dim=1)  # (M)
            min_index = torch.argmin(rmsd_mean, dim=0) # ()
            in_dict['pos'][base_order] = in_pos_syms[min_index]
            assert (in_dict['node'][base_order] == in_dict['node'][matches_iso[min_index]]).all(), 'node type not symmetry'

        return in_dict
    
    def additional_process(self, batch, in_dict):
        if self.pre_process is not None:
            if self.pre_process == 'fix_closest':   # fixed, cannot be changed
                if 'node_closest' in batch:
                    node_closest = batch['node_closest']
                else:
                    node_closest = []
                    for i_mol in range(batch['node_type_batch'].max().item()+1):
                        dist_mat = torch.cdist(batch['gt_node_pos'][batch['node_type_batch']==i_mol],
                                                  batch['pocket_pos'][batch['pocket_pos_batch']==i_mol])
                        node_closest_this = torch.argmin(dist_mat.min(dim=1)[0], dim=0)
                        node_closest_this = node_closest_this + (batch['node_type_batch']<i_mol).sum()
                        node_closest.append(node_closest_this)
                    node_closest = torch.stack(node_closest)
                    batch['node_closest'] = node_closest
                batch['fixed_pos'][node_closest] = 1
                batch['node_pos'][node_closest] = batch['gt_node_pos'][node_closest].clone()  # for following check of node_pos == in_pos
                in_dict['pos'][node_closest] = batch['gt_node_pos'][node_closest].clone()
            elif self.pre_process == 'fix_some':  # fix some atoms, cannot be changed
                fixed_pos = batch['fixed_pos'].bool()
                if 'gt_node_pos' in  batch:
                    in_dict['pos'][fixed_pos] = batch['gt_node_pos'][fixed_pos].clone()
                else:
                    in_dict['pos'][fixed_pos] = batch['node_pos'][fixed_pos].clone()
            else:
                raise NotImplementedError('not implemented for pre_process:', self.pre_process)
        return in_dict

    def outputs2batch(self, batch, outputs):
        
        if self.post_process is None:
            setting = self._get_setting(batch)
            if setting in ['flexible', 'torsional']:
                # correct pos
                pred_pos = correct_pos_batch(batch, outputs)
                # pred_pos = correct_pos_by_fixed_dist_batch(batch, outputs)
            elif setting == 'rigid':
                pred_pos = correct_pos_batch_no_tor(batch, outputs)
            else:
                pred_pos = outputs['pred_pos']
        elif isinstance(self.post_process, dict):  # for use
            pred_pos = outputs['pred_pos'].clone()
            name = self.post_process.name
            assert name == 'know_some', 'Only know_some post_process is supported.'
            num_mols = batch['node_type_batch'].max() + 1
            num_nodes = batch['num_nodes'] // num_mols

            step = batch['step']
            if step > 0.2:  # directly fixed
                if 'orig_fixed_pos' not in batch:
                    self.pre_process = 'fix_some'  # fix the some atoms
                    batch['orig_fixed_pos'] = batch['fixed_pos'].clone()
                    for i_batch in range(num_mols):
                        for one_setting in self.post_process.atom_space:
                            atom_index = one_setting['atom'] + num_nodes * i_batch
                            batch['fixed_pos'][atom_index] = 1
                            # modify the coord
                            if 'coord' in one_setting:
                                coord = torch.tensor(one_setting['coord'], dtype=batch['node_pos'].dtype, device=batch['node_pos'].device)
                                coord -= batch['pocket_center'][i_batch]
                                batch['node_pos'][atom_index] = coord
            else:
                # reset fixed_pos
                if 'orig_fixed_pos' in batch:
                    self.pre_process = None
                    batch['fixed_pos'] = batch['orig_fixed_pos']
                for i_batch in range(num_mols):
                    for one_setting in self.post_process.atom_space:
                        atom_index = one_setting['atom'] + num_nodes * i_batch
                        radius = one_setting['radius']
                        if 'coord' in one_setting: # check equal to gt_node_pos
                            coord = torch.tensor(one_setting['coord'], dtype=batch['node_pos'].dtype, device=batch['node_pos'].device)
                            coord -= batch['pocket_center'][i_batch]
                        else:
                            coord = batch['gt_node_pos'][atom_index]
                        # correct pos
                        dist = torch.norm(pred_pos[atom_index] - coord, dim=-1)
                        if dist > radius:  # move
                            pred_pos[atom_index] = coord
        else:  # str
            if self.post_process == 'correct_pos':  # correct pos just like flex mode
                corr_config = self.config.get('correct_pos', None)
                interval_steps = corr_config['interval_steps']
                if batch['step'] >= interval_steps[0] and batch['step'] <= interval_steps[1]:
                    pred_pos = correct_pos_batch(batch, outputs, use_pos='gt')
                else:
                    pred_pos = outputs['pred_pos']
            elif self.post_process == 'correct_dist':  # correct the interval pairwise distances using gradient descent
                corr_config = self.config.get('correct_dist', None)
                if corr_config is not None and 'threshold_step' in corr_config:
                    threshold_step = corr_config['threshold_step']
                else:
                    threshold_step = 1000000000000  # boring
                if batch['step'] <= threshold_step:
                    pred_pos = correct_pos_by_fixed_dist_batch(batch, outputs, use_pos='gt', config=corr_config)
                else:
                    pred_pos = outputs['pred_pos']
            elif self.post_process == 'correct_center':
                corr_config = self.config.get('correct_center', None)
                pred_pos = outputs['pred_pos']
                batch_index = batch['node_type_batch']
                gt_center = scatter_mean(batch['gt_node_pos'], index=batch_index, dim=0)
                pred_center = scatter_mean(pred_pos, index=batch_index, dim=0)
                delta_pos = gt_center - pred_center
                delta_dist = torch.norm(delta_pos, p=2, dim=-1)[..., None]
                translation = torch.where(
                    delta_dist > corr_config.radius,
                    delta_pos,
                    torch.zeros_like(delta_pos)
                )
                pred_pos = pred_pos + translation[batch_index]
            elif self.post_process == 'correct_closest':  # know the closest atom pos within a sphere
                corr_config = self.config.get('correct_closest', None)
                radius = corr_config['radius']
                # find the closest pos
                if 'node_closest' in batch:
                    node_closest = batch['node_closest']
                else:
                    node_closest = []
                    for i_mol in range(batch['node_type_batch'].max().item()+1):
                        dist_mat = torch.cdist(batch['gt_node_pos'][batch['node_type_batch']==i_mol],
                                                  batch['pocket_pos'][batch['pocket_pos_batch']==i_mol])
                        node_closest_this = torch.argmin(dist_mat.min(dim=1)[0], dim=0)
                        node_closest_this = node_closest_this + (batch['node_type_batch']<i_mol).sum()
                        node_closest.append(node_closest_this)
                    node_closest = torch.stack(node_closest)
                    batch['node_closest'] = node_closest
                # correct pos
                gt_closest_pos = batch['gt_node_pos'][node_closest]  # known closest pos. so use gt
                pred_closest_pos = outputs['pred_pos'][node_closest]
                delta_pos = gt_closest_pos - pred_closest_pos
                delta_dist = torch.norm(delta_pos, p=2, dim=-1)[..., None]
                translation = torch.where(
                    delta_dist > radius,
                    delta_pos,
                    torch.zeros_like(pred_closest_pos)
                )
                pred_pos = outputs['pred_pos'].clone()
                pred_pos += translation[batch['node_type_batch']]
            elif self.post_process == 'flex_to_free':  # change the mode from flex to free
                corr_config = self.config.get('flex_to_free', None)
                change_step = corr_config['change_step']
                if batch['step'] > change_step:
                    pred_pos = correct_pos_batch(batch, outputs)  # use_pos can be 'gt' or not since the rigidity is not changed for step > change_step
                else:
                    # mode change
                    batch['task_setting'] = ['free' for _ in batch['task_setting']]
                    # fixed indicators
                    batch['fixed_halfdist'] = torch.zeros_like(batch['fixed_halfdist'])
                    # not correct pos
                    pred_pos = outputs['pred_pos']
            elif self.post_process == 'transparency':  # out = in
                pred_pos = batch['pos_in'].clone()
            else:
                raise NotImplementedError('not implemented for post_process:', self.post_process)
            
        fixed_pos = (batch['fixed_pos'] == 1)
        pred_pos[fixed_pos] = batch['node_pos'][fixed_pos].clone()

        batch['node_pos'] = pred_pos.clone()
        # node_type and halfedge_type are is unchanged
        
        # if 'confidence_pos' in outputs:
        #     batch['confidence_pos'] = outputs['confidence_pos']
        #     batch['confidence_node'] = outputs['confidence_node']
        #     batch['confidence_halfedge'] = outputs['confidence_halfedge']
        return batch


@register_sample_noise('dock')
class DockSamplNoiser(ConfSampleNoiser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, task_name='dock')


@register_sample_noise('fbdd')
@register_sample_noise('maskfill')
class MaskfillSampleNoiser(BaseSampleNoiser):
    def __init__(self,
        config, num_node_types, num_edge_types,
        mode='sample', device='cpu', ref_config=None, task_name='maskfill',
        **kwargs
    ):
        super().__init__(task_name, config, num_node_types, num_edge_types,
            mode, device, ref_config, **kwargs)
        
        # define prior
        prior_part1 = config.prior.part1 if config.prior.part1 != 'from_train' else self.ref_prior_config.part1
        self.prior_p1 = MolPrior(prior_part1, num_node_types, num_edge_types).to(device)
        prior_part2 = config.prior.part2 if config.prior.part2 != 'from_train' else self.ref_prior_config.part2
        self.prior_p2 = MolPrior(prior_part2, num_node_types, num_edge_types).to(device)

        # define info level
        self.level_p1 = MolInfoLevel(config.level.part1, device=device, mode=mode)
        self.level_p2 = MolInfoLevel(config.level.part2, device=device, mode=mode)
        

    def sample_level(self, step, batch):
        # # level for part 1, part 2 and part1-part2
        level_dict = {}
        
        # for part1
        setting = self._get_setting(batch)
        part1_pert = setting['part1_pert']
        leveller = self.level_p1
        if part1_pert == 'fixed':
            pass
        elif part1_pert == 'free':
            level_pos_p1 = leveller.sample_for_mol(step, n_pos=batch['node_p1'].shape[0])
            level_dict.update({
                'pos_p1': level_pos_p1,
            })
        elif part1_pert == 'small':
            n_node = n_pos = batch['node_p1'].shape[0]
            n_halfedge = batch['halfedge_p1'].shape[0]
            level_node_p1, level_pos_p1, level_halfedge_p1 = leveller.sample_for_mol(step,
                n_node=n_node, n_pos=n_pos, n_edge=n_halfedge,)
            level_dict.update({
                'node_p1': level_node_p1, 'pos_p1': level_pos_p1, 'halfedge_p1': level_halfedge_p1
            })
            task = self._get_task(batch)
            if task == 'ar':  # noise_level p1 is scaled by ratio of p1. NOTE: seems not used
                batch_node = batch['node_type_batch']
                n_nodes_batch = batch_node.bincount()
                node_p1 = batch['node_p1']
                batch_p1 = batch_node[node_p1]
                if len(batch_p1) != 0:
                    n_p1_batch = batch_p1.bincount()
                else:
                    n_p1_batch = torch.zeros_like(n_nodes_batch)
                ratio_p1_batch = n_p1_batch.float() / n_nodes_batch.float()
                ratio_p1_node = ratio_p1_batch[batch_p1]
                ratio_p1_halfedge = ratio_p1_batch[batch['halfedge_type_batch'][batch['halfedge_p1']]]
                level_dict.update({
                    'node_p1': (1-level_node_p1) * ratio_p1_node + level_node_p1,
                    'pos_p1': (1-level_pos_p1) * ratio_p1_node + level_pos_p1,
                    'halfedge_p1': (1-level_halfedge_p1) * ratio_p1_halfedge + level_halfedge_p1,
                })
                raise NotImplementedError('not implemented for ar noise_level. seems not used')

        elif part1_pert == 'rigid':
            n_trans = n_rot = torch.sum(batch['n_domain']).item()
            L_trans_p1, L_rot_p1 = leveller.sample_for_mol(step, n_trans=n_trans, n_rot=n_rot)
            level_dict.update({
                'trans_p1': L_trans_p1, 'rot_p1': L_rot_p1
            })
        elif part1_pert == 'flexible':
            n_trans = n_rot = torch.sum(batch['n_domain']).item()
            n_tor = batch['tor_bonds_anno'].shape[0]
            L_trans_p1, L_rot_p1, L_tor_p1 = leveller.sample_for_mol(step,
                n_trans=n_trans, n_rot=n_rot, n_tor=n_tor)
            level_dict.update({
                'trans_p1': L_trans_p1, 'rot_p1': L_rot_p1, 'tor_p1': L_tor_p1
            })
        
        # for part2 and edge of p1p2
        leveller = self.level_p2
        n_node = n_pos = batch['node_p2'].shape[0]
        n_halfedge = batch['halfedge_p2'].shape[0]
        n_halfedge_p1p2 = batch['halfedge_p1p2'].shape[0]
        L_node_p2, L_pos_p2, L_halfedge_p2_and_p1p2 = leveller.sample_for_mol(step,
            n_node=n_node, n_pos=n_pos, n_edge=(n_halfedge + n_halfedge_p1p2))
        level_dict.update({
            'node_p2': L_node_p2,
            'pos_p2': L_pos_p2,
            'halfedge_p2': L_halfedge_p2_and_p1p2[:n_halfedge],
        })
        
        # p1p2 fixed
        level_p1p2 = L_halfedge_p2_and_p1p2[n_halfedge:]
        fixed_halfedge = batch['fixed_halfedge']
        halfedge_p1p2 = batch['halfedge_p1p2']
        fixed_p1p2 = (fixed_halfedge[halfedge_p1p2] == 1)
        level_p1p2[fixed_p1p2] = 1
        level_dict.update({
            'halfedge_p1p2': level_p1p2,
        })
        
        return level_dict
    
    def has_pocket(self, batch):
        return batch['pocket_pos'].shape[0] > 0

    def add_noise(self, node_type, node_pos, halfedge_type, batch,
                   from_prior=False, level_dict=None):
        
        task = self._get_task(batch)
        setting = self._get_setting(batch)
        part1_pert = setting['part1_pert']

        # centering before add_noise
        # if task == 'linking':
        if (not self.has_pocket(batch)) and (task != 'linking'):
            batch_node = getattr(batch, 'node_type_batch',
                        torch.zeros(node_type.shape[0], dtype=torch.long, device=node_pos.device))
            node_pos_center = scatter_mean(node_pos, batch_node, dim=0, dim_size=batch_node.max()+1)[batch_node]
            node_pos = node_pos - node_pos_center
            batch.update({'node_pos': node_pos.clone()})  #NOTE: a little leakage during sampling since p2 gaussian center is gt p1 center

        # part 1 noise
        if task in ['maskfill', 'linking', 'sbdd', 'denovo']:  # sbdd, denovo in ar mode
            from_prior_part1 = False
        elif task in ['fbdd', 'growing']:
            from_prior_part1 = from_prior
        else:
            raise ValueError(f'Unknown task: {task} to set from_prior_part1')
        prior = self.prior_p1
        level_dict_p1 = {k[:-3]:v for k, v in level_dict.items() if k.endswith('_p1')}
        
        if part1_pert == 'fixed':
            pass
        elif part1_pert == 'small':
            node_p1 = batch['node_p1']
            halfedge_p1 = batch['halfedge_p1']
            node_type[node_p1], node_pos[node_p1], halfedge_type[halfedge_p1] = prior.add_noise(
                node_type[node_p1], node_pos[node_p1], halfedge_type[halfedge_p1],
                level_dict_p1, from_prior=from_prior_part1,  # not from prior for part 1 because part 1 is not totally masked
            )
        elif part1_pert == 'free':
            node_p1 = batch['node_p1']
            node_pos[node_p1] = prior.add_noise(
                None, node_pos[node_p1], None,
                level_dict=level_dict_p1, from_prior=from_prior_part1, pos_only=True
            )
        else:  # 'rigid', 'flexible'
            node_p1 = batch['node_p1']
            additional_dict = {
                'pos_only': True,
                'node_type': None, 'halfedge_type': None, # placeholder
                'domain_node_index': batch['domain_node_index']
            }
            if part1_pert == 'flexible':
                additional_dict.update({
                    'tor_bonds_anno': batch['tor_bonds_anno'],
                    'twisted_nodes_anno': batch['twisted_nodes_anno'],
                })
            assert all([n in node_p1 for n in batch['domain_node_index'][1]]), 'node in domain not in node_p1'
            node_pos = prior.add_noise(
                node_pos=node_pos, level_dict=level_dict_p1,
                from_prior=from_prior_part1, **additional_dict,
            )
        
        # part 2 noise
        prior = self.prior_p2
        level_dict_p2 = {k[:-3]:v for k, v in level_dict.items() if k.endswith('_p2')}
        node_p2 = batch['node_p2']
        halfedge_p2 = batch['halfedge_p2']
        node_type[node_p2], node_pos[node_p2], halfedge_type[halfedge_p2] = prior.add_noise(
            node_type[node_p2], node_pos[node_p2], halfedge_type[halfedge_p2],
            level_dict_p2, from_prior
        )
        halfedge_p1p2 = batch['halfedge_p1p2']
        halfedge_type[halfedge_p1p2] = prior.halfedge.add_noise(
            halfedge_type[halfedge_p1p2], level_dict['halfedge_p1p2'], from_prior
        )
        
        
        # centering after add_noise
        # if task == 'linking':
        # if not self.has_pocket(batch):
        if (not self.has_pocket(batch)) and (task != 'linking'):
            if part1_pert != 'fixed':
            # only centering in non-fixed case where global coord system doesn't exist
                pos_in_center = scatter_mean(node_pos, batch_node, dim=0, dim_size=batch_node.max()+1)[batch_node]
                node_pos = node_pos - pos_in_center
        
        in_dict = {'node':node_type, 'pos':node_pos, 'halfedge': halfedge_type}
        return in_dict

    def outputs2batch(self, batch, outputs):
        
        # task = self._get_task(batch)
        # if task == 'ar':
        #     return self.outputs2batch_ar(batch, outputs)
        
        setting = self._get_setting(batch)
        part1_pert = setting['part1_pert']

        # node_p1, node_p2 = batch['node_p1'], batch['node_p2']
        fixed_node = (batch['fixed_node'] == 1)
        fixed_pos = (batch['fixed_pos'] == 1)
        fixed_halfedge = (batch['fixed_halfedge'] == 1)

        batch['node_type'][~fixed_node] = outputs['pred_node'][~fixed_node].argmax(-1).clone()
        batch['node_pos'][~fixed_pos] = outputs['pred_pos'][~fixed_pos].clone()
        batch['halfedge_type'][~fixed_halfedge] = outputs['pred_halfedge'][~fixed_halfedge].argmax(-1).clone()
        
        if part1_pert =='rigid':
            pred_pos = correct_pos_batch_no_tor(batch, outputs)
            batch['node_pos'] = pred_pos
        elif part1_pert == 'flexible':
            pred_pos = correct_pos_batch(batch, outputs)
            batch['node_pos'] = pred_pos
        
        return batch

    def outputs2batch_ar2(self, batch, outputs):
        # if len(batch['pocket_pos']) > 0:
        #     raise NotImplementedError('pocket_pos not supported for ar yet. note the initial atom')
        device = batch['node_type'].device

        fixed_node = (batch['fixed_node'] == 1)
        fixed_pos = (batch['fixed_pos'] == 1)
        fixed_halfedge = (batch['fixed_halfedge'] == 1)
        batch['node_type'][~fixed_node] = outputs['pred_node'][~fixed_node].argmax(-1).clone()
        batch['node_pos'][~fixed_pos] = outputs['pred_pos'][~fixed_pos].clone()
        batch['halfedge_type'][~fixed_halfedge] = outputs['pred_halfedge'][~fixed_halfedge].argmax(-1).clone()

        
        # follow_batch = [k[:-6] for k in batch.keys if k.endswith('_batch')]
        follow_batch = [k[:-6] for k in batch.keys() if k.endswith('_batch')]
        mol_data_list = batch.to_data_list()
        for data in mol_data_list:
            if data['node_type'].shape[0] < data['gt_node_type'].shape[0]:
                n_frag = 6
                # new nodes
                data.update({
                    'node_type': torch.cat([data['node_type'], torch.zeros(n_frag, dtype=torch.long, device=device)]),
                    'node_pos': torch.cat([data['node_pos'], torch.zeros(n_frag, 3, dtype=torch.float, device=device)]),
                    'is_peptide': torch.cat([data['is_peptide'],
                                data['is_peptide'][0] * torch.zeros(n_frag, dtype=torch.long, device=device)]),
                })
                # new edges
                halfedge_index = data['halfedge_index']
                halfedge_type = data['halfedge_type']
                n_nodes = data['node_type'].shape[0]
                edge_type_mat = torch.zeros(n_nodes, n_nodes, dtype=torch.long, device=device)
                for i_edge in range(halfedge_type.shape[0]):
                    edge_type_mat[halfedge_index[0, i_edge], halfedge_index[1, i_edge]] = halfedge_type[i_edge]
                # edge_type_mat[:, -1] = -1
                # edge_type_mat[-1, :] = -1
                halfedge_index = torch.triu_indices(n_nodes, n_nodes, offset=1, device=device)
                halfedge_type = edge_type_mat[halfedge_index[0], halfedge_index[1]]
                data.update({
                    'halfedge_index': halfedge_index,
                    'halfedge_type': halfedge_type
                })
                # node_p1, node_p2
                node_p1 = torch.arange(n_nodes-n_frag, dtype=torch.long, device=device)
                # node_p2 = torch.tensor([n_nodes-n_frag], dtype=torch.long, device=device)
                node_p2 = torch.arange(n_nodes-n_frag, n_nodes, dtype=torch.long, device=device)
                # halfedge_p1, halfedge_p2, halfedge_p1p2
                is_left_p1 = (halfedge_index[0] < n_nodes-n_frag)
                is_right_p1 = (halfedge_index[1] < n_nodes-n_frag)
                is_halfedge_p1 = is_left_p1 & is_right_p1
                is_halfedge_p2 = (~is_left_p1) & (~is_right_p1)
                is_halfedge_p1p2 = (~is_halfedge_p1) & (~is_halfedge_p2)
                halfedge_p1 = torch.nonzero(is_halfedge_p1).squeeze(-1)
                halfedge_p2 = torch.nonzero(is_halfedge_p2).squeeze(-1)
                halfedge_p1p2 = torch.nonzero(is_halfedge_p1p2).squeeze(-1)
                # fixed
                fixed_node = torch.zeros(n_nodes, dtype=torch.long, device=device)
                fixed_pos = torch.zeros(n_nodes, dtype=torch.long, device=device)
                fixed_halfedge = torch.zeros(halfedge_type.shape[0], dtype=torch.long, device=device)
                fixed_halfdist = torch.zeros(halfedge_type.shape[0], dtype=torch.long, device=device)
            else: # finished
                n_nodes = data['node_type'].shape[0]
                halfedge_index = data['halfedge_index']
                node_p1 = torch.arange(n_nodes, dtype=torch.long, device=device)
                node_p2 = torch.tensor([], dtype=torch.long, device=device)
                halfedge_p1 = torch.arange(halfedge_index.shape[1], dtype=torch.long, device=device)
                halfedge_p2 = torch.tensor([], dtype=torch.long, device=device)
                halfedge_p1p2 = torch.tensor([], dtype=torch.long, device=device)
                # fixed
                # fixed_node = torch.ones(n_nodes, dtype=torch.long, device=device)
                # fixed_pos = torch.ones(n_nodes, dtype=torch.long, device=device)
                # fixed_halfedge = torch.ones(halfedge_index.shape[1], dtype=torch.long, device=device)
                # fixed_halfdist = torch.ones(halfedge_index.shape[1], dtype=torch.long, device=device)
                fixed_node = torch.zeros(n_nodes, dtype=torch.long, device=device)
                fixed_pos = torch.zeros(n_nodes, dtype=torch.long, device=device)
                fixed_halfedge = torch.zeros(halfedge_index.shape[1], dtype=torch.long, device=device)
                fixed_halfdist = torch.zeros(halfedge_index.shape[1], dtype=torch.long, device=device)
            data.update({
                'node_p1': node_p1,
                'node_p2': node_p2,
                'halfedge_p1': halfedge_p1,
                'halfedge_p2': halfedge_p2,
                'halfedge_p1p2': halfedge_p1p2,
                'fixed_node': fixed_node,
                'fixed_pos': fixed_pos,
                'fixed_halfedge': fixed_halfedge,
                'fixed_halfdist': fixed_halfdist,
            })
        batch = Batch.from_data_list(mol_data_list, follow_batch=follow_batch)

        return batch

    def additional_process(self, batch, in_dict):
        
        # reset those in_dict with fixed==1
        fixed_node = (batch['fixed_node'] == 1)
        in_dict['node'][fixed_node] = batch['node_type'][fixed_node].clone()
        fixed_pos = (batch['fixed_pos'] == 1)
        in_dict['pos'][fixed_pos] = batch['node_pos'][fixed_pos].clone()
        fixed_halfedge = (batch['fixed_halfedge'] == 1)
        in_dict['halfedge'][fixed_halfedge] = batch['halfedge_type'][fixed_halfedge].clone()
        return in_dict

    def outputs2batch_ar(self, batch, outputs, step_ar=None, cfd_traj=None):
        
        ar_config = self.config.ar_config
        ar_strategy = getattr(ar_config, 'strategy', 'default')
        
        if ar_strategy == 'default':
        
            batch_node = batch['node_type_batch']
            node_p1 = batch['node_p1']
            node_p2 = batch['node_p2']
            is_node_p1 = torch.zeros_like(batch_node, dtype=torch.bool)
            is_node_p1[node_p1] = True

            # get the node to be added to p1 from p2 for each data
            pred_node = outputs['pred_node']
            cfd_node = outputs['confidence_node']
            cfd_pos = outputs['confidence_pos']
            n_batch = batch_node.max() + 1
            n_sizes = batch_node.bincount()
            added_node = []
            for i_batch in range(n_batch):
                is_curr_p2 = (batch_node == i_batch) & (~is_node_p1)
                if is_curr_p2.sum() == 0:
                    continue
                pred_node_curr_p2 = pred_node[is_curr_p2]
                cfd_node_curr_p2 = cfd_node[is_curr_p2]
                cfd_pos_curr_p2 = cfd_pos[is_curr_p2]
                
                # # selecte generated nodes of this ar step
                ref_prob_type = ar_config.ref_prob_type
                if ref_prob_type == 'pred_node':
                    ref_prob = F.softmax(pred_node_curr_p2, dim=-1)
                    ref_prob = ref_prob.max(-1)[0]  # prob of the selected node_type
                elif ref_prob_type == 'cfd_node':
                    ref_prob = torch.sigmoid(cfd_node_curr_p2)[:, 0]
                elif ref_prob_type == 'cfd_pos':
                    ref_prob = torch.sigmoid(cfd_pos_curr_p2)[:, 0]
                else:
                    raise NotImplementedError('ref_prob_type not implemented:', ref_prob_type)
                ref_prob = (ref_prob-ref_prob.min()) / (ref_prob.max() - ref_prob.min()+1e-4)+1e-3
                
                size_select = ar_config.size_select
                select_strategy = ar_config.select_strategy
                if size_select >= 1:  # select fixed number
                    size_select = int(size_select)
                elif size_select < 1 and size_select > 0:  # select ratio
                    size_select = int(np.round(size_select * is_curr_p2.sum().item()))
                    size_select = max(size_select, 1)
                if select_strategy == 'random':
                    sel_node_curr = torch.unique(torch.multinomial(ref_prob, num_samples=size_select, replacement=True))
                elif select_strategy == 'top':
                    sel_node_curr = torch.argsort(ref_prob, descending=True)[:size_select]
                
                index_sel_node_curr = torch.nonzero(is_curr_p2)[sel_node_curr]
                added_node.extend(index_sel_node_curr)
            
            # change node_p1 and node_p2
            is_node_p1[torch.cat(added_node)] = True
            node_p1 = torch.nonzero(is_node_p1).squeeze(-1)
            node_p2 = torch.nonzero(~is_node_p1).squeeze(-1)

            # change halfedge_p1, halfedge_p2, halfedge_p1p2
            halfedge_index = batch['halfedge_index']
            left_in_p1 = is_node_p1[halfedge_index[0]]
            right_in_p1 = is_node_p1[halfedge_index[1]]
            is_halfedge_p1 = left_in_p1 & right_in_p1
            is_halfedge_p2 = (~left_in_p1) & (~right_in_p1)
            is_halfedge_p1p2 = (~is_halfedge_p1) & (~is_halfedge_p2)
            halfedge_p1 = torch.nonzero(is_halfedge_p1).squeeze(-1)
            halfedge_p2 = torch.nonzero(is_halfedge_p2).squeeze(-1)
            halfedge_p1p2 = torch.nonzero(is_halfedge_p1p2).squeeze(-1)
            
            part1_pert = self._get_setting(batch)['part1_pert']
            fixed_node = batch['fixed_node']
            fixed_pos = batch['fixed_pos']
            fixed_halfedge = batch['fixed_halfedge']
            fixed_halfdist = batch['fixed_halfdist']
            if part1_pert == 'small':
                pass
            elif part1_pert == 'fixed':
                fixed_node[node_p1] = 1
                fixed_pos[node_p1] = 1
                fixed_halfedge[halfedge_p1] = 1
                fixed_halfdist[halfedge_p1] = 1
                
            
            # # fixed the mol if no atoms in p2 (finished)
            batch_halfedge = batch['halfedge_type_batch']
            is_node_p2 = ~is_node_p1
            batch_mol_p2 = torch.unique(batch_node[is_node_p2])
            batch_mol_noinp2 = torch.tensor([i for i in range(batch_node.max()+1) if i not in batch_mol_p2],
                                                dtype=torch.long, device=batch_node.device)
            is_node_noinp2 = (batch_node[:, None]==batch_mol_noinp2[None]).any(-1)
            is_halfedge_noinp2 = (batch_halfedge[:, None]==batch_mol_noinp2[None]).any(-1)
            fixed_node[is_node_noinp2] = 1
            fixed_pos[is_node_noinp2] = 1
            fixed_halfedge[is_halfedge_noinp2] = 1
            # is_finished = torch.zeros_like(batch_node, dtype=torch.bool)
            # is_finished[is_node_noinp2] = True
            
            # reset 
            temperature = len(batch_node) / len(node_p1)
            batch.update({
                'node_p1': node_p1,
                'node_p2': node_p2,
                'halfedge_p1': halfedge_p1,
                'halfedge_p2': halfedge_p2,
                'halfedge_p1p2': halfedge_p1p2,

                # 'node_type': torch.multinomial(F.softmax(outputs['pred_node']/temperature, -1), 1)[:,0],
                # 'node_type': outputs['pred_node'].argmax(-1),
                # 'node_pos': outputs['pred_pos'],
                # 'halfedge_type': outputs['pred_halfedge'].argmax(-1),
                
                'fixed_node': fixed_node,
                'fixed_pos': fixed_pos,
                'fixed_halfedge': fixed_halfedge,
                'fixed_halfdist': fixed_halfdist,
            })
        elif ar_strategy == 'refine_partial':
            batch_node = batch['node_type_batch']
            # mol_size = batch_node.bincount()[batch_node]

            threshold_node = ar_config.threshold_node
            threshold_pos = ar_config.threshold_pos
            threshold_bond = ar_config.threshold_bond
            max_ar_step = ar_config.max_ar_step
            max_p2_ratio = ar_config.get('max_p2_ratio', 1)
            change_init_step = ar_config.get('change_init_step', None)
            if change_init_step is not None:
                self.init_step = change_init_step
            
            cfd_node = torch.sigmoid(outputs['confidence_node'][:, 0])
            cfd_pos = torch.sigmoid(outputs['confidence_pos'][:, 0])
            cfd_halfedge = torch.sigmoid(outputs['confidence_halfedge'][:, 0])
    
            # edge to bond, get cfd_node_with_bond
            halfedge_index = batch['halfedge_index']
            pred_halfedge = outputs['pred_halfedge'].argmax(-1)
            is_halfbond = (pred_halfedge > 0)
            halfbond_index = halfedge_index[:, is_halfbond]
            cfd_halfbond = cfd_halfedge[is_halfbond]
            bond_index = torch.cat([halfbond_index, halfbond_index.flip(0)], dim=-1)
            cfd_bond = torch.cat([cfd_halfbond, cfd_halfbond], dim=0)
            cfd_node_with_bond = scatter_mean(cfd_bond, bond_index[0], dim=0, dim_size=batch['node_type'].shape[0])
            
            # select nodes
            sel_node_curr_p2 = []
            possibility_p2_list = [
                (threshold_node - cfd_node).clamp(min=0),
                (threshold_pos - cfd_pos).clamp(min=0),
                (threshold_bond - cfd_node_with_bond).clamp(min=0),
            ]
            for possibility_p2 in possibility_p2_list:
                size_select = int(np.round((possibility_p2>0).sum().item() *
                                (1 - step_ar / max_ar_step) *
                                max_p2_ratio))
                if size_select > 0:
                    sel_this = torch.unique(torch.multinomial(
                        possibility_p2+1e-7, num_samples=size_select, replacement=True))
                else:
                    sel_this = []
                sel_node_curr_p2.extend(sel_this)
            sel_node_curr_p2 = torch.unique(torch.tensor(sel_node_curr_p2))
            sel_node_curr_p2 = sel_node_curr_p2 if len(sel_node_curr_p2) > 0 else []
            
            is_node_p2 = torch.zeros_like(batch_node, dtype=torch.bool)
            is_node_p2[sel_node_curr_p2] = True
            is_node_p1 = ~is_node_p2
            # print('is_node_p2', is_node_p2.sum().item())
            
            # change node_p1 and node_p2
            node_p1 = torch.nonzero(is_node_p1).squeeze(-1)
            node_p2 = torch.nonzero(is_node_p2).squeeze(-1)
            
            # change halfedge_p1, halfedge_p2, halfedge_p1p2
            left_in_p1 = is_node_p1[halfedge_index[0]]
            right_in_p1 = is_node_p1[halfedge_index[1]]
            is_halfedge_p1 = left_in_p1 & right_in_p1
            is_halfedge_p2 = (~left_in_p1) & (~right_in_p1)
            is_halfedge_p1p2 = (~is_halfedge_p1) & (~is_halfedge_p2)
            halfedge_p1 = torch.nonzero(is_halfedge_p1).squeeze(-1)
            halfedge_p2 = torch.nonzero(is_halfedge_p2).squeeze(-1)
            halfedge_p1p2 = torch.nonzero(is_halfedge_p1p2).squeeze(-1)

            batch.update({
                'node_p1': node_p1,
                'node_p2': node_p2,
                'halfedge_p1': halfedge_p1,
                'halfedge_p2': halfedge_p2,
                'halfedge_p1p2': halfedge_p1p2,
                
                # 'node_type': outputs['pred_node'].argmax(-1),
                # 'node_pos': outputs['pred_pos'],
                # 'halfedge_type': outputs['pred_halfedge'].argmax(-1),
            })
        elif ar_strategy == 'refine':  # juan (involution) with neighbor 
            batch_node = batch['node_type_batch']
            node_pos = outputs['pred_pos']
            # get config
            threshold_node = ar_config.threshold_node
            threshold_pos = ar_config.threshold_pos
            threshold_bond = ar_config.threshold_bond
            threshold_cfd_traj = ar_config.get('threshold_cfd_traj', 0)  # cfd pos traj. last half mean
            max_ar_step = ar_config.max_ar_step
            max_p2_ratio = ar_config.get('max_p2_ratio', 1)
            change_init_step = ar_config.get('change_init_step', None)
            if change_init_step is not None:
                self.init_step = change_init_step
            r = ar_config.r

            # select center
            if cfd_traj is not None:
                cfd_pos_lasthalf = torch.sigmoid(torch.stack(cfd_traj[-int(self.num_steps*self.init_step*0.5):])).mean(0)
            cfd_node = torch.sigmoid(outputs['confidence_node'][:, 0])
            cfd_pos = torch.sigmoid(outputs['confidence_pos'][:, 0])
            cfd_halfedge = torch.sigmoid(outputs['confidence_halfedge'][:, 0])
    
            # edge to bond, get cfd_node_with_bond
            halfedge_index = batch['halfedge_index']
            pred_halfedge = outputs['pred_halfedge'].argmax(-1)
            is_halfbond = (pred_halfedge > 0)
            halfbond_index = halfedge_index[:, is_halfbond]
            cfd_halfbond = cfd_halfedge[is_halfbond]
            bond_index = torch.cat([halfbond_index, halfbond_index.flip(0)], dim=-1)
            cfd_bond = torch.cat([cfd_halfbond, cfd_halfbond], dim=0)
            cfd_node_with_bond = scatter_mean(cfd_bond, bond_index[0], dim=0, dim_size=batch['node_type'].shape[0])
            
            is_center_p2 = (
                (cfd_pos_lasthalf <= threshold_cfd_traj) |
                (cfd_node <= threshold_node) |
                (cfd_pos <= threshold_pos) |
                (cfd_node_with_bond <= threshold_bond)
            )
            # index_min_cfd_pos = scatter_min(cfd_pos, batch_node, dim=0, dim_size=batch_node.max()+1)[1]
            index_min_cfd_pos = torch.zeros_like(batch_node, dtype=torch.bool)
            is_center_p2[index_min_cfd_pos] = True
            # is_center_p2[batch['node_p1']] = False  # tenure
            if 'is_finished' in batch:
                is_center_p2[batch['is_finished']] = False
            center_pos = node_pos[is_center_p2]
            batch_center = batch_node[is_center_p2]
            # select neighbor
            assign_index = radius(x=node_pos, y=center_pos, r=r,
                                  batch_x=batch_node, batch_y=batch_center)
            sel_node_curr_p2 = torch.unique(assign_index[1])

            # print(step_ar, sel_node_curr_p2.detach().cpu().numpy())
            # print(cfd_pos.detach().cpu().numpy())
            if step_ar >= max_ar_step:
                sel_node_curr_p2 = []
            is_node_p2 = torch.zeros_like(batch_node, dtype=torch.bool)
            is_node_p2[sel_node_curr_p2] = True
            is_node_p1 = ~is_node_p2
            # print('is_node_p2', is_node_p2.sum().item())

            # change node_p1 and node_p2
            node_p1 = torch.nonzero(is_node_p1).squeeze(-1)
            node_p2 = torch.nonzero(is_node_p2).squeeze(-1)
            
            # change halfedge_p1, halfedge_p2, halfedge_p1p2
            halfedge_index = batch['halfedge_index']
            left_in_p1 = is_node_p1[halfedge_index[0]]
            right_in_p1 = is_node_p1[halfedge_index[1]]
            is_halfedge_p1 = left_in_p1 & right_in_p1
            is_halfedge_p2 = (~left_in_p1) & (~right_in_p1)
            is_halfedge_p1p2 = (~is_halfedge_p1) & (~is_halfedge_p2)
            halfedge_p1 = torch.nonzero(is_halfedge_p1).squeeze(-1)
            halfedge_p2 = torch.nonzero(is_halfedge_p2).squeeze(-1)
            halfedge_p1p2 = torch.nonzero(is_halfedge_p1p2).squeeze(-1)
            
            
            # fixed (free mode of p1)
            # reset fixed indicators
            setting = self._get_setting(batch)
            if setting['part1_pert'] == 'small':
                fixed_node = batch['fixed_node']
                fixed_pos = batch['fixed_pos']
                fixed_halfedge = batch['fixed_halfedge']
            elif setting['part1_pert'] == 'free':
                fixed_node = torch.zeros_like(batch['fixed_node'])
                fixed_node[is_node_p1] = 1
                fixed_pos = torch.zeros_like(batch['fixed_pos'])
                fixed_halfedge = torch.zeros_like(batch['fixed_halfedge'])
                fixed_halfedge[is_halfedge_p1] = 1
            
            # fixed the mol if no atoms in p2 (finished)
            batch_halfedge = batch['halfedge_type_batch']
            batch_mol_p2 = torch.unique(batch_node[is_node_p2])
            batch_mol_noinp2 = torch.tensor([i for i in range(batch_node.max()+1) if i not in batch_mol_p2],
                                                dtype=torch.long, device=batch_node.device)
            is_node_noinp2 = (batch_node[:, None]==batch_mol_noinp2[None]).any(-1)
            is_halfedge_noinp2 = (batch_halfedge[:, None]==batch_mol_noinp2[None]).any(-1)
            fixed_node[is_node_noinp2] = 1
            fixed_pos[is_node_noinp2] = 1
            fixed_halfedge[is_halfedge_noinp2] = 1
            is_finished = torch.zeros_like(batch_node, dtype=torch.bool)
            is_finished[is_node_noinp2] = True

            batch.update({
                'node_p1': node_p1,
                'node_p2': node_p2,
                'halfedge_p1': halfedge_p1,
                'halfedge_p2': halfedge_p2,
                'halfedge_p1p2': halfedge_p1p2,
                'is_finished': is_finished,
                
                # 'node_type': outputs['pred_node'].argmax(-1),
                # 'node_pos': outputs['pred_pos'],
                # 'halfedge_type': outputs['pred_halfedge'].argmax(-1),
                
                'fixed_node': fixed_node,
                'fixed_pos': fixed_pos,
                'fixed_halfedge': fixed_halfedge,
            })
        else:
            raise NotImplementedError('ar_strategy not implemented:', ar_strategy)
            
            
        return batch

@register_sample_noise('pepdesign')
class PepdesignSampleNoiser(BaseSampleNoiser):
    def __init__(self,
        config, num_node_types, num_edge_types,
        mode='sample', device='cpu', ref_config=None, task_name='pepdesign',
        **kwargs
    ):
        super().__init__(task_name, config, num_node_types, num_edge_types,
            mode, device, ref_config, **kwargs)
        
        # define prior
        prior_bb = config.prior.bb if config.prior.bb != 'from_train' else self.ref_prior_config.bb
        self.prior_bb = MolPrior(prior_bb, num_node_types, num_edge_types).to(device)
        prior_sc = config.prior.sc if config.prior.sc != 'from_train' else self.ref_prior_config.sc
        self.prior_sc = MolPrior(prior_sc, num_node_types, num_edge_types).to(device)

        # define info level
        self.level_bb = MolInfoLevel(config.level.bb, device=device, mode=mode)
        self.level_sc = MolInfoLevel(config.level.sc, device=device, mode=mode)

    def sample_level(self, step, batch):
        
        level_dict = {}
        setting = self._get_setting(batch)
        mode = setting['mode']
        n_node_sc = batch['node_sc'].shape[0]
        n_halfedge_sc = batch['halfedge_sc'].shape[0]
        n_halfedge_bbsc = batch['halfedge_bbsc'].shape[0]
        
        if mode == 'full':
            level_pos_bb = self.level_bb.sample_for_mol(step, n_pos=batch['node_bb'].shape[0])
            level_node_sc, level_pos_sc, level_halfedge_sc_and_bbsc = self.level_sc.sample_for_mol(
                step, n_node=n_node_sc, n_pos=n_node_sc, n_edge=(n_halfedge_sc + n_halfedge_bbsc))
            level_dict.update({
                'pos_bb': level_pos_bb,
                'node_sc': level_node_sc,
                'pos_sc': level_pos_sc,
                'halfedge_sc': level_halfedge_sc_and_bbsc[:n_halfedge_sc],
                'halfedge_bbsc': level_halfedge_sc_and_bbsc[n_halfedge_sc:],
            })
        elif mode == 'sc':
            level_node_sc, level_pos_sc, level_halfedge_sc_and_bbsc = self.level_sc.sample_for_mol(
                step, n_node=n_node_sc, n_pos=n_node_sc, n_edge=(n_halfedge_sc + n_halfedge_bbsc))
            level_dict.update({
                'node_sc': level_node_sc,
                'pos_sc': level_pos_sc,
                'halfedge_sc': level_halfedge_sc_and_bbsc[:n_halfedge_sc],
                'halfedge_bbsc': level_halfedge_sc_and_bbsc[n_halfedge_sc:],
            })
        elif mode == 'packing':
            level_pos_sc = self.level_sc.sample_for_mol(step, n_pos=n_node_sc)
            level_dict.update({
                'pos_sc': level_pos_sc,
            })

        # halfedge_bbsc fixed
        if 'halfedge_bbsc' in level_dict:
            level_bbsc = level_dict['halfedge_bbsc']
            fixed_halfedge = batch['fixed_halfedge']
            halfedge_bbsc = batch['halfedge_bbsc']
            fixed_bbsc = (fixed_halfedge[halfedge_bbsc] == 1)
            level_bbsc[fixed_bbsc] = 1
            level_dict.update({
                'halfedge_bbsc': level_bbsc,
            })
        
        return level_dict
    
    def add_noise(self, node_type, node_pos, halfedge_type, batch,
                   from_prior=False, level_dict=None):
        
        setting = self._get_setting(batch)
        mode = setting['mode']

        # bb noise
        if mode == 'full':
            level_dict_bb = {k[:-3]:v for k, v in level_dict.items() if k.endswith('_bb')}
            node_bb = batch['node_bb']
            node_pos[node_bb] = self.prior_bb.add_noise(
                None, node_pos[node_bb], None,
                level_dict=level_dict_bb, from_prior=from_prior, pos_only=True
            )
        
        # sc noise
        node_sc = batch['node_sc']
        halfedge_sc = batch['halfedge_sc']
        halfedge_bbsc = batch['halfedge_bbsc']
        level_dict_sc = {k[:-3]:v for k, v in level_dict.items() if k.endswith('_sc')}
        if mode in ['full', 'sc']:
            node_type[node_sc], node_pos[node_sc], halfedge_type[halfedge_sc] = self.prior_sc.add_noise(
                node_type[node_sc], node_pos[node_sc], halfedge_type[halfedge_sc],
                level_dict_sc, from_prior
            )
            halfedge_type[halfedge_bbsc] = self.prior_sc.halfedge.add_noise(
                halfedge_type[halfedge_bbsc], level_dict['halfedge_bbsc'], from_prior
            )
        elif mode == 'packing':
            node_pos[node_sc] = self.prior_sc.add_noise(
                None, node_pos[node_sc], None,
                level_dict=level_dict_sc, from_prior=from_prior, pos_only=True
            )
        else:
            raise ValueError(f'Unknown mode: {mode}')

        in_dict = {'node':node_type, 'pos':node_pos, 'halfedge': halfedge_type}
        return in_dict

    def outputs2batch(self, batch, outputs):
        
        setting = self._get_setting(batch)
        mode = setting['mode']

        # node_p1, node_p2 = batch['node_p1'], batch['node_p2']
        fixed_node = (batch['fixed_node'] == 1)
        fixed_pos = (batch['fixed_pos'] == 1)
        fixed_halfedge = (batch['fixed_halfedge'] == 1)

        batch['node_type'][~fixed_node] = outputs['pred_node'][~fixed_node].argmax(-1).clone()
        batch['node_pos'][~fixed_pos] = outputs['pred_pos'][~fixed_pos].clone()
        batch['halfedge_type'][~fixed_halfedge] = outputs['pred_halfedge'][~fixed_halfedge].argmax(-1).clone()
        
        return batch


@register_sample_noise('custom')
class CustomSampleNoiser(BaseSampleNoiser):
    def __init__(self,
        config, num_node_types, num_edge_types,
        mode='sample', device='cpu', ref_config=None, task_name='custom',
        **kwargs
    ):
        super().__init__(task_name, config, num_node_types, num_edge_types,
            mode, device, ref_config, **kwargs)
        
        self.noiser_names = list(config.prior.keys())
        # # define prior
        self.prior_dict = {}
        for this_prior in config.prior.items():
            name = this_prior[0]
            prior = MolPrior(this_prior[1], num_node_types, num_edge_types).to(device)
            self.prior_dict[name] = prior

        # # define info_level
        self.level_dict = {}
        for this_level in config.level.items():
            name = this_level[0]
            leveller = MolInfoLevel(this_level[1], device=device, mode=mode)
            self.level_dict[name] = leveller
            assert name in self.noiser_names, f'Undefined name {name} of level_dict'
        assert len(self.prior_dict) == len(self.level_dict), 'config size mismatch: leveller'

        # # mapper of noisers
        self.mapper_dict = config.mapper
        assert all(name in self.noiser_names for name in self.mapper_dict.keys()), f'Undefined name {name} in mapper'
        assert len(self.prior_dict) == len(self.mapper_dict), 'config size mismatch: mapper'
        
        # # correction. post correcting preset variables
        self.correction = config.get('correction', [])

    def sample_level(self, step, batch):
        level_dict = {}
        for name in self.noiser_names:
            leveller = self.level_dict[name]
            mapper = self.mapper_dict[name]
            level_kwargs = {}
            what_list = []
            if 'node' in mapper:
                part_list = mapper['node']
                level_kwargs['n_node'] = sum(batch[f'node_part_{part}'].shape[0] for part in part_list)
                what_list.append('node')
            if 'pos' in mapper:
                part_list = mapper['pos']
                level_kwargs['n_pos'] = sum(batch[f'node_part_{part}'].shape[0] for part in part_list)
                what_list.append('pos')
            if 'edge' in mapper:
                part_list = mapper['edge']
                level_kwargs['n_edge'] = sum(batch[f'halfedge_part_{parts[0]}_{parts[1]}'].shape[0] for parts in part_list)
                what_list.append('halfedge')

            this_info_level = leveller.sample_for_mol(step, **level_kwargs)
            if len(level_kwargs) == 1:
                this_info_level = [this_info_level]  # stupid
            level_dict.update({ f'{what}_noiser_{name}': this_info_level[i]
                               for i, what in enumerate(what_list)})
        return level_dict

    def has_pocket(self, batch):
        return batch['pocket_pos'].shape[0] > 0

    def add_noise(self, node_type, node_pos, halfedge_type, batch,
                   from_prior=False, level_dict=None):
        
        for name in self.noiser_names:
            prior = self.prior_dict[name]
            mapper = self.mapper_dict[name]
            level_dict_part = {k.replace(f'_noiser_{name}', ''): v for k, v in level_dict.items()
                               if k.endswith(f'_noiser_{name}')}
            
            node_part = torch.concat([batch[f'node_part_{part}'] for part in mapper['node']]) if 'node' in mapper else []
            pos_part = torch.concat([batch[f'node_part_{part}'] for part in mapper['pos']]) if 'pos' in mapper else []
            halfedge_part = torch.concat([batch[f'halfedge_part_{part[0]}_{part[1]}'] for 
                                part in mapper['edge']]) if 'edge' in mapper else []
            noised_mol = prior.add_noise(
                node_type[node_part], node_pos[pos_part], halfedge_type[halfedge_part],
                level_dict_part, (from_prior and prior.config.get('from_prior', True))
            )
            if prior.pos_only:
                node_pos[pos_part] = noised_mol
            else:
                node_type[node_part], node_pos[pos_part], halfedge_type[halfedge_part] = noised_mol
        
        in_dict = {'node':node_type, 'pos':node_pos, 'halfedge': halfedge_type}
        return in_dict

    def outputs2batch(self, batch, outputs):
        
        # task = self._get_task(batch)
        # if task == 'ar':
        #     return self.outputs2batch_ar(batch, outputs)
        
        fixed_node = (batch['fixed_node'] == 1)
        fixed_pos = (batch['fixed_pos'] == 1)
        fixed_halfedge = (batch['fixed_halfedge'] == 1)

        # # protect correction
        if 'node' in self.correction:
            node_part_corr = torch.concat([batch[f'node_part_{part}'] for part in self.correction['node']])
            node_corrected = batch['node_type'][node_part_corr].clone()
        if 'pos' in self.correction:
            pos_part_corr = torch.concat([batch[f'node_part_{part}'] for part in self.correction['pos']])
            pos_corrected = batch['node_pos'][pos_part_corr].clone()
        if 'edge' in self.correction:
            halfedge_part_corr = torch.concat([batch[f'halfedge_part_{part[0]}_{part[1]}'] for 
                                part in self.correction['edge']])
            halfedge_corrected = batch['halfedge_type'][halfedge_part_corr].clone()
        
        batch['node_type'][~fixed_node] = outputs['pred_node'][~fixed_node].argmax(-1).clone()
        batch['node_pos'][~fixed_pos] = outputs['pred_pos'][~fixed_pos].clone()
        batch['halfedge_type'][~fixed_halfedge] = outputs['pred_halfedge'][~fixed_halfedge].argmax(-1).clone()

        # # apply correction
        if 'node' in self.correction:
            batch['node_type'][node_part_corr] = node_corrected
        if 'pos' in self.correction:
            batch['node_pos'][pos_part_corr] = pos_corrected
        if 'edge' in self.correction:
            batch['halfedge_type'][halfedge_part_corr] = halfedge_corrected
        
        return batch

    def _get_correction(self, batch):
        if 'node' in self.correction:
            node_part_corr = torch.concat([batch[f'node_part_{part}'] for part in self.correction['node']])
            node_corrected = batch[node_part_corr]
        if 'pos' in self.correction:
            pos_part_corr = torch.concat([batch[f'node_part_{part}'] for part in self.correction['pos']])
            pos_corrected = batch[pos_part_corr]
        if 'edge' in self.correction:
            halfedge_part_corr = torch.concat([batch[f'halfedge_part_{part[0]}_{part[1]}'] for 
                                part in self.correction['edge']])
            halfedge_corrected = batch[halfedge_part_corr]
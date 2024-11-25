import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter_mean
from utils.motion import RobustAngleSO3Distribution, apply_axis_angle_rotation,\
        apply_torsional_rotation_multiple_domains, robust_sample_angle, sample_uniform_angle
from models.corrector import kabsch_flatten

PRIOR_DICT = {}
def register_prior(name):
    def decorator(cls):
        PRIOR_DICT[name] = cls
        return cls
    return decorator

def get_prior(config, *args, **kwargs):
    if config is None:
        return None
    name = config.name
    if name != 'from_train':
        return PRIOR_DICT[name](config, *args, **kwargs) #.to(device)
    else:
        train_config = kwargs.pop('train_config')
        return get_prior(train_config, *args, **kwargs)


class MolPrior(nn.Module):
    def __init__(self, config, num_node_types, num_edge_types, pos_only=False):
        super().__init__()
        self.config = config
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.pos_only = pos_only or getattr(config, 'pos_only', False)
        
        self.pos = get_prior(config.pos)
        if not self.pos_only:
            self.node = get_prior(config.node, num_classes=num_node_types)
            self.halfedge = get_prior(config.edge, num_classes=num_edge_types)
    
    @torch.no_grad()
    def add_noise(self, node_type, node_pos, halfedge_type,
                  level_dict, from_prior, pos_only=False, **kwargs):
        # avoid data leakage for info_level == 0
        
        pos_pert = self.pos.add_noise(node_pos, level_dict, from_prior, **kwargs)
        if not (self.pos_only or pos_only):
            node_pert = self.node.add_noise(node_type, level_dict['node'], from_prior)
            halfedge_pert = self.halfedge.add_noise(halfedge_type, level_dict['halfedge'], from_prior)
            return node_pert, pos_pert, halfedge_pert
        else:
            return pos_pert
    
@register_prior('allpos')
# @register_prior('flexible')
class AllPosPrior(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_prior = get_prior(getattr(config, 'pos', None))
        self.translation_prior = get_prior(getattr(config, 'translation', None))
        self.rotation_prior = get_prior(getattr(config, 'rotation', None))
        self.torsional_prior = get_prior(getattr(config, 'torsional', None))
        
    @torch.no_grad()
    def add_noise(self, pos, info_level, from_prior, **kwargs):
        if 'pos' in info_level:
            pos = self.pos_prior.add_noise(pos, info_level['pos'], from_prior,
                                           kwargs.get('mol_size', None))
            assert ((key not in info_level) for key in ['trans', 'rot', 'tor']), 'pos noise is not compatiable with flexible noise'
            return pos
        # domain_index = kwargs['domain_index']
        # n_domain = domain_index.max() + 1
        if 'trans' in info_level:
            pos = self.translation_prior.add_noise(pos, info_level['trans'],
                    from_prior, kwargs['domain_node_index'])
        if 'rot' in info_level:
            pos = self.rotation_prior.add_noise(pos, info_level['rot'],
                    from_prior, kwargs['domain_node_index'])
        if 'tor' in info_level:
            pos = self.torsional_prior.add_noise(pos, info_level['tor'], from_prior,
                            kwargs['tor_bonds_anno'], kwargs['twisted_nodes_anno'],
                            kwargs['domain_node_index'])
        return pos


@register_prior('torsional')
class TorsionalPrior(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sigma_max = getattr(config, 'sigma_max', 1) * torch.pi
        self.decouple = getattr(config, 'decouple', True)  # decouple from rototranslation

    @torch.no_grad()
    def add_noise(self, pos, info_level, from_prior,
                  tor_bonds_anno, twisted_nodes_anno,
                  domain_node_index=None):
        """
        pos: positions, [n_node, 3]. NOTE: not part of pos but all positions
        info_level: [n_tor_bond]
        tor_bonds_anno: [n_tor_bond, 3]
        twisted_nodes_anno: [n_twisted, 2]
        """
        tor_order = tor_bonds_anno[:, 0]
        tor_bonds = tor_bonds_anno[:, 1:]
        
        index_tor = twisted_nodes_anno[:, 0]
        twisted_nodes = twisted_nodes_anno[:, 1]

        if len(tor_order) == 0:
            return pos

        # prepare sigma and angles
        sigmas = (1 - info_level) * self.sigma_max  # (n_tor_edge)
        if not from_prior:
            angles = robust_sample_angle(sigmas)
        else:  # in [-pi, pi], x + uniform is uniform, thus no data leakage
            angles_not_prior = robust_sample_angle(sigmas)
            angles_prior = sample_uniform_angle(sigmas)
            angles = torch.where(info_level == 0, angles_prior, angles_not_prior)

        # apply torsional rotation
        pos_tor = apply_torsional_rotation_multiple_domains(
            pos, tor_order, tor_bonds, angles,
            twisted_nodes, index_tor,
        )

        # decouple from rototranslation by minimizing rmsd
        if self.decouple:
            domain_index, node_index = domain_node_index
            global_rot, global_trans = kabsch_flatten(pos_tor[node_index], pos[node_index], domain_index)
    
            # apply global rotation and translation
            pos_corrected_expand = torch.matmul(
                pos_tor[node_index, None, :],
                global_rot.transpose(1, 2)[domain_index]
            ) + global_trans[domain_index]

            pos_tor[node_index] = pos_corrected_expand.squeeze(1)
        return pos_tor


@register_prior('rigid')
class RigidPrior(nn.Module):
    def __init__(self, config):
        raise NotImplementedError('Deprecated! Use flexible or allpos prior instead.')
        super().__init__()
        self.config = config
        self.translation_prior = get_prior(config.translation) if config.translation else None
        self.rotation_prior = get_prior(config.rotation) if config.rotation else None
    
    @torch.no_grad()
    def add_noise(self, pos, info_level, **kwargs):
        """
        pos: positions, [n_node, 3]
        info_level: [n_domain]
        domain_index: [n_node]
        Return:
            positions with rigid perturbation (translation & rotation) of each domain, [n_node, 3]
        """
        domain_index = kwargs['domain_index']
        n_domain = domain_index.max() + 1
        # translation
        if self.translation_prior:
            pos = self.translation_prior.add_noise(pos, info_level[:n_domain], **kwargs)
        # rotation
        if self.rotation_prior:
            pos = self.rotation_prior.add_noise(pos, info_level[n_domain:n_domain*2], **kwargs)
        return pos


@register_prior('rotation')
class RotationPrior(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sigma_max = getattr(config, 'sigma_max', 1) * torch.pi
        self.angle_distr = RobustAngleSO3Distribution()
        
    @torch.no_grad()
    def add_noise(self, pos, info_level, from_prior,
                  domain_node_index):
        """
        pos: positions, [n_node, 3]
        info_level: [n_domain]
        domain_node_index: [2, n_node_domain]
        Return:
            positions with rotation perturbation of each domain, [n_node, 3]
        """
        device = pos.device
        n_domain = info_level.shape[0]
        domain_index, node_index = domain_node_index
        sigmas = self.sigma_max * (1 - info_level)
        
        # sample
        axes = self._sample_axis(n_domain, device)
        angles = self._sample_anlge(sigmas, is_unfiform=False)
        if from_prior:
            angles_uniform = self._sample_anlge(sigmas, is_unfiform=True)
            angles = torch.where(info_level == 0, angles_uniform, angles)

        # apply rotate around com instead of center
        # pos_domain = scatter_mean(pos, domain_index, dim=0)  # center of each domain
        # pos_center = pos[domain_center_nodes].mean(dim=1)
        pos_center = scatter_mean(pos[node_index], domain_index, dim=0)
        pos_rel = pos[node_index] - pos_center[domain_index]  # relative position to the center of each domain
        pos_update = (apply_axis_angle_rotation(pos_rel, axes[domain_index], angles[domain_index])
                        + pos_center[domain_index])

        pos = pos.clone()
        pos[node_index] = pos_update
        return pos
    
    def _sample_axis(self, n, device):
        axes = F.normalize(torch.randn(n, 3, device=device), dim=-1)
        return axes

    def _sample_anlge(self, sigmas, is_unfiform):
        angles = self.angle_distr.sample(sigmas, is_uniform=is_unfiform)
        return angles


@register_prior('translation')
class TranslationPrior(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ve = getattr(config, 've', True)
        if self.ve:  # variance explode
            sigma_max = getattr(config, 'sigma_max', 1)
            self.register_buffer('sigma_max', torch.tensor(sigma_max))
        else:  # variance perserve
            mean = getattr(config, 'mean', 0)
            std = getattr(config, 'std', 1)
            self.register_buffer('mean', torch.tensor(mean))
            self.register_buffer('std', torch.tensor(std))

    
    def add_noise(self, x, info_level, from_prior, domain_node_index):
        n_domain = info_level.shape[0]
        domain_index, node_index = domain_node_index
        
        if info_level.dim() < x.dim():
            info_level_exp = info_level[:, None].expand(n_domain, x.shape[1])
        else:
            info_level_exp = info_level
        
        pos_domain = x[node_index]
        if self.ve:
            noise_domain = torch.randn_like(pos_domain[:n_domain]) * self.sigma_max
            pert = pos_domain + (1 - info_level_exp[domain_index]) * noise_domain[domain_index]
            if from_prior:
                center_domain = scatter_mean(pos_domain, domain_index, dim=0)
                pert = torch.where(info_level_exp[domain_index] == 0, 
                            pert - center_domain[domain_index], pert)
        else:
            noise_domain = torch.randn_like(pos_domain[:n_domain]) * self.std + self.mean
            pos_center_before = scatter_mean(pos_domain, domain_index, dim=0)
            pos_center_after = (info_level_exp).sqrt() * pos_center_before +\
                    (1 - info_level_exp).sqrt() * noise_domain
            pert = pos_domain - pos_center_before[domain_index] + pos_center_after[domain_index]
            if from_prior:
                pass  # when from_prior, info_level_exp==0. auto cancelling out x
        pert = torch.where(info_level_exp[domain_index] == 1, pos_domain, pert)
        
        x = x.clone()
        x[node_index] = pert
        return x


@register_prior('gaussian_simple')
class GaussianExplodePrior(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        sigma_max = getattr(config, 'sigma_max', 1) 
        # self.sigma_max = nn.Parameter(torch.tensor(sigma_max), requires_grad=False)
        self.register_buffer('sigma_max', torch.tensor(sigma_max))
        self.sigma_func = getattr(config, 'sigma_func', None)
        
    @torch.no_grad()
    def add_noise(self, x, info_level, from_prior, mol_size=None):

        if info_level.dim() < x.dim():
            info_level_exp = info_level[:, None].expand_as(x)
        else:
            info_level_exp = info_level

        # NOTE: when info_level == 0, the prior mean is 0. DIFFERENT from GaussianPrior
        # x = torch.where(info_level_exp == 0, torch.zeros_like(x), x)

        if mol_size is None or self.sigma_func is None:
            noise = torch.zeros_like(x)
            noise.normal_(mean=0, std=self.sigma_max)
        else:
            assert len(mol_size) == len(x), 'Error: mol_size and x have different dim'
            if self.sigma_func == 'sqrt': # 
                sigma = self.sigma_max * mol_size.sqrt()
                noise = torch.randn_like(x) * sigma[:, None].clamp(min=1)
            elif self.sigma_func == 'linbias':
                sigma = (0.08 * mol_size + 1).clamp(min=5)
                noise = torch.randn_like(x) * sigma[:, None]
            elif self.sigma_func == 'sqrtbias':
                sigma = ((mol_size - 40).clamp(min=0).sqrt() + 2).clamp(min=5)
                noise = torch.randn_like(x) * sigma[:, None]
            elif self.sigma_func == 'seg59':
                sigma = torch.clamp(0.08 * mol_size + 1, min=5, max=9)
                noise = torch.randn_like(x) * sigma[:, None]
            else:
                raise NotImplementedError(f'Error: sigma_func {self.sigma_func} not implemented')
            
        pert = x + (1 - info_level_exp) * noise
        if  from_prior:
            pert = torch.where(info_level_exp == 0, noise, pert)
            
        pert = torch.where(info_level_exp == 1, x, pert)
        return pert
    

@register_prior('gaussian')
class GaussianPrior(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        mean = getattr(config, 'mean', 0)
        std = getattr(config, 'std', 1)
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))
        
    @torch.no_grad()
    def add_noise(self, x, info_level, from_prior):
        if info_level.dim() < x.dim():
            info_level_exp = info_level[:, None].expand_as(x)
        else:
            info_level_exp = info_level
        noise = torch.zeros_like(x)
        noise.normal_(mean=self.mean, std=self.std)
        pert = info_level_exp.sqrt() * x + (1 - info_level_exp).sqrt() * noise
        if from_prior:
            pass  # when from_prior, info_level_exp==0. auto cancelling out x
            # pert = torch.where(info_level_exp == 0, noise, pert)
        pert = torch.where(info_level_exp == 1, x, pert)
        return pert

    # def sample_prior(self, x):
    #     return self.add_noise(x, info_level=torch.zeros_like(x))

@register_prior('categorical')
class CategoricalPrior(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        prior_type = config.prior_type
        prior_probs = getattr(config, 'prior_probs', None)
        probs = self.get_prior(prior_type, prior_probs)
        self.prior_probs = nn.Parameter(probs, requires_grad=False)
        
    def get_prior(self, prior_type, prior_probs):
        if prior_type == 'uniform':
            probs = 1. / self.num_classes * torch.ones(self.num_classes)
        elif prior_type == 'tomask':
            probs = torch.zeros(self.num_classes)
            probs[-1] = 1.
        elif prior_type == 'tomask_half':
            probs = torch.zeros(self.num_classes)
            probs[-1] = 0.5
            probs[:-1] = 0.5 / (self.num_classes - 1)
        elif prior_type == 'predefined':
            assert prior_probs is not None, 'Error: prior_probs is None and prior_type is predefined'
            assert len(prior_probs) == self.num_classes, f'Error: len(prior_probs) != {self.num_classes}'
            prior_probs = np.array([float(p) for p in prior_probs])
            prior_probs = prior_probs / sum(prior_probs)
            probs = torch.tensor(prior_probs)
        else:
            raise NotImplementedError(f'Error: prior_type {prior_type} not implemented')
        return probs
        # self.sampler = torch.distributions.Categorical(probs=probs)

    @torch.no_grad()
    def add_noise(self, x, info_level, from_prior, return_posterior=False):
        """
        x: (N,)
        info_level: (N,)
        """

        
        x_onehot = F.one_hot(x, self.num_classes).float()
        info_level_exp = info_level[:, None].expand_as(x_onehot)

        prior = self.prior_probs[None, :].expand_as(x_onehot)
        prob = info_level_exp * x_onehot + (1 - info_level_exp) * prior  # (N, K)
        if from_prior:
            pass  # when from_prior, info_level_exp==0. auto cancelling out x_onehot
        # sample from prob
        pert = torch.multinomial(prob, num_samples=1).squeeze(-1) # (N,)
        pert = torch.where(info_level == 1, x, pert)
        prob = torch.where(info_level_exp == 1, x_onehot, prob)
        if return_posterior:
            return pert, prob
        return pert
        

"""
Define and sample info_level
"""
import torch
from torch import nn
import numpy as np

INFO_LEVEL_DICT = {}
def register_info_level(name):
    def decorator(cls):
        INFO_LEVEL_DICT[name] = cls
        return cls
    return decorator

# def get_level(config, *args, **kwargs):
#     name = config.name
#     return INFO_LEVEL_DICT[name](config, *args, **kwargs)

def get_level(name, *args, **kwargs):
    return INFO_LEVEL_DICT[name](*args, **kwargs)

class MolInfoLevel:
    def __init__(self, config, device=None, mode='train'):
        self.config = config
        self.name = config.name
        self.min = config.min
        self.max = config.max
        self.asym = getattr(config, 'asym', None)

        if self.name == 'advance':
            self.step2level = AdvanceScaler(config.step2level).to(device)
        elif self.name == 'power':
            self.step2level = PowerScaler(config.step2level).to(device)
        elif self.name == 'exp':
            self.step2level = ExpScaler(config.step2level).to(device)

        self.device = device
        self.mode = mode
        self.allowed_keys = ['n_node', 'n_pos', 'n_edge', 'n_trans', 'n_rot', 'n_tor']
        
    def _sample(self, step):
        if step is None:
            step = torch.rand((), device=self.device)
        
        if self.asym is None:
            level = self._step2level(step)
            return level
        else:
            if step < 0.5:
                step_pos = 9 / 5 *step
                step_type = 1 / 5 * step
            else:
                step_pos = 1 / 5 * step + 0.8
                step_type = 9 / 5 * step - 0.8
            if self.asym == 'pos_first':
                pass
            elif self.asym == 'pos_last':
                step_pos, step_type = step_type, step_pos
            level_pos = self._step2level(step_pos.clamp(min=0, max=1))
            level_type = self._step2level(step_type.clamp(min=0, max=1))
            return (level_pos, level_type)
        
    def _step2level(self, step):
        if step == 1:  # step = 1 -> level = 0
            level = torch.zeros((), device=self.device)
        else:
            # transform. NOTE: negative correlation
            if self.name == 'uniform':  # uniformly sample from [min, max)
                level = 1 - step 
            elif self.name in ['advance', 'power', 'exp']:
                level = self.step2level(step)
            else:
                raise ValueError(f'Unknown info_level name: {self.name}')
        
        # scale 
        level = level * (self.max - self.min) + self.min
        return level
        
    def set_value(self, shape, value):
        return torch.ones(shape, device=self.device) * value
        
    def sample_for_mol(self, step, **kwargs):
        if self.mode == 'train':
            assert step is None, 'step should not be given in train mode'
        elif self.mode == 'sample':
            assert step is not None, 'step should be given in sample mode'
        else:
            raise ValueError(f'Unknown mode: {self.mode}')
        value = self._sample(step)

        value_list = []
        for key in kwargs:
            assert key in self.allowed_keys, f'Unknown key: {key}'
            if self.asym is None:
                value_list.append(self.set_value(kwargs[key], value))
            else:
                if key in ['n_pos', 'n_trans', 'n_rot', 'n_tor']:
                    value_list.append(self.set_value(kwargs[key], value[0]))
                elif key in ['n_node', 'n_edge']:
                    value_list.append(self.set_value(kwargs[key], value[1]))
                else:
                    raise ValueError(f'Unknown key {key}')

        if len(value_list) == 1:
            value_list = value_list[0]
        return value_list


@register_info_level('individual')
class IndividualInfoLevel:
    # def __init__(self, config):
    #     self.config = config
    #     pass

    def sample_from_shape(self, shape):
        """
        Directly sample from [0, 1)
        """
        info = torch.rand(shape)
        return info
    

@register_info_level('preset')
class PresetInfoLevel:
    # def __init__(self, config):
    #     self.config = config
    #     pass

    def sample_from_shape(self, shape, value):
        info = torch.ones(shape) * value
        return info
    
@register_info_level('whole')
class WholeInfoLevel:
    # def __init__(self, config):
    #     self.config = config

    def sample_from_shape(self, shape, *args, **kwargs):
        value = torch.rand(1)
        info = torch.ones(shape) * value
        return info
    

@register_info_level('diff')
class DiffInfoTraj(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.num_steps = config['num_steps']
        self.traj_func = TrajFunction(config)
        # steps_list = np.linspace(0, 1, self.num_steps)
        # self.level_list = np.array([traj_func(step) for step in steps_list])
        
    def sample_from_shape(self, shape, step):
        level = self.traj_func(step)
        info = torch.ones(shape, device=level.device) * level
        return info
        


class AdvanceScaler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        scale_start = nn.Parameter(torch.tensor(config['scale_start']), requires_grad=False)
        scale_end = nn.Parameter(torch.tensor(config['scale_end']), requires_grad=False)
        width = nn.Parameter(torch.tensor(config['width']), requires_grad=False)
        self.setup(scale_start, scale_end, width)
        
    def setup(self, scale_start, scale_end, width):
        self.k = width
        A0 = scale_end
        A1 = scale_start

        self.a = (A0-A1)/(torch.sigmoid(-self.k) - torch.sigmoid(self.k))
        self.b = 0.5 * (A0 + A1 - self.a)
        
    def __call__(self, x):
        x = 2 * x - 1
        return self.a * torch.sigmoid(- self.k * x) + self.b


class PowerScaler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer('k', torch.tensor(config['k']))
        # assert self.k >= 1, 'k should be >= 1'
        
    def __call__(self, x):
        return 1 - torch.pow(x, self.k)


class ExpScaler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer('k', torch.tensor(config['k']))
        # assert self.k > 0, 'k should be > 0'
        
    def __call__(self, x):
        value = (torch.exp(self.k * x) - torch.exp(self.k)) / (1 - torch.exp(self.k))
        return value



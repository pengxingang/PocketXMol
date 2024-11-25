import os
import time
import random
import logging
import torch
import numpy as np
import yaml
from easydict import EasyDict
from logging import Logger
from tqdm.auto import tqdm
import signal
from contextlib import contextmanager


class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class BlackHole(object):
    def __setattr__(self, name, value):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, name):
        return self

def make_config(path, path_add=None):
    if os.path.exists(path):
        config = load_config(path)
    else: # it is get from config_preset and config_modify
        config_dir = os.path.dirname(path)
        config_name = os.path.basename(path)[:os.path.basename(path).rfind('.')]
        # get preset and all modify
        preset_path = os.path.join(config_dir, 'config_preset.yml')
        modify_path = os.path.join(config_dir, 'config_modify.yml')
        assert os.path.exists(preset_path), 'not found {}'.format(path)
        assert os.path.exists(modify_path), 'config_modify.yml not found in {}'.format(config_dir)
        # load preset and modify for this 
        preset = load_config(preset_path)
        modify = load_config(modify_path)
        modify = modify[config_name]
        # apply modify to preset
        def apply_modify(preset, modify):
            for k, v in modify.items():
                if isinstance(v, dict):
                    apply_modify(preset[k], v)
                else:
                    preset[k] = v
        apply_modify(preset, modify)
        config = preset

    # add
    if path_add is None:
        return config
    else:
        if not os.path.exists(path_add):
            path_add = os.path.join(os.path.dirname(path), path_add)
        config_add = make_config(path_add)
        config.update(config_add)
        return config


def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))

def save_config(config, path):
    def to_dict(config):
        if isinstance(config, EasyDict):
            return {k: to_dict(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [to_dict(v) for v in config]
        else:
            return config
        # new_config = dict()
        # for key, value in config.items():
        #     if isinstance(value, EasyDict):
        #         new_config[key] = to_dict(value)
        #     elif isinstance(value, list):
        #         new_config[key] = [to_dict(v) if isinstance(v, EasyDict) else v for v in value]
        #     else:
        #         new_config[key] = value
        # return new_config
    config = to_dict(config)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
def load_train_config_from_ckpt(ckpt_path):
    dirname = os.path.dirname(ckpt_path)
    files = [f for f in os.listdir(dirname) if f.endswith('.yaml') or f.endswith('.yml')]
    if len(files) == 0:
        raise ValueError('No config file found in {}'.format(dirname))
    elif len(files) > 1:
        raise ValueError('Multiple config files found in {}'.format(dirname))
    file = files[0]
    config_path = os.path.join(dirname, file)
    return load_config(config_path)


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k:v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))

def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(0, inverse, perm)
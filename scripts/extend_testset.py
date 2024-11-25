import os
import argparse
import torch
from easydict import EasyDict
from tqdm.auto import tqdm

from utils.datasets import get_dataset
from utils.misc import *
from utils.chem import *


def get_chain_name(fn):
    return os.path.basename(fn)[:6]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='./data/crossdocked_pocket10')
    parser.add_argument('-s', '--split', type=str, default='./data/crossdocked_pocket10_test120_split.pt')
    parser.add_argument('-u', '--subset', type=str, default='test')
    parser.add_argument('-o', '--out', type=str, default=None)
    parser.add_argument('--protein_root', type=str, default='./data/crossdocked')
    parser.add_argument('--ligand_root', type=str, default='./data/crossdocked_pocket10')
    args = parser.parse_args()

    # Logger
    logger = get_logger('dock',)
    logger.info(args)

    # Data
    logger.info('Loading data...')
    dataset, subsets = get_dataset(EasyDict({
        'name': 'pl',
        'path': args.dataset,
        'split': args.split,
    }))

    chains = set([get_chain_name(d.ligand_filename) for d in subsets[args.subset]])

    logger.info('Number of chains: %d' % len(chains))

    split = {'train': [], 'val': [], 'test': []}
    for i, data in enumerate(tqdm(dataset)):
        if get_chain_name(data.ligand_filename) in chains:
            split[args.subset].append(i)

    print('Number of datapoints: %d' % len(split[args.subset]))

    # Save
    if args.out is None:
        split_name = os.path.basename(args.split)
        split_name = split_name[:split_name.rfind('.')]
        docked_name = '%s_extended.pt' % (split_name, )
        out_path = os.path.join(os.path.dirname(args.dataset), docked_name)
    else:
        out_path = args.out

    logger.info('Saving the extended split to: %s' % out_path)
    torch.save(split, out_path)

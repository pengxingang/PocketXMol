import os
import argparse
import random
import torch
from tqdm.auto import tqdm

from utils.datasets import PocketLigandPairDataset, Subset


def get_chain_name(fn):
    return os.path.basename(fn)[:6]


def get_pdb_name(fn):
    return os.path.basename(fn)[:4]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./data/crossdocked_pocket10')
    parser.add_argument('--original_split', type=str, default='./data/crossdocked_pocket10_split.pt')
    parser.add_argument('--output_split', type=str, default='./data/crossdocked_pocket10_test_split.pt')
    parser.add_argument('--num_pockets', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()
    
    dataset = PocketLigandPairDataset(args.dataset)
    split = torch.load(args.original_split)
    subsets = {k:Subset(dataset, indices=v) for k, v in split.items()}

    train_pdb = {get_pdb_name(d.ligand_filename) for d in tqdm(subsets['train'])} 

    test_id = []
    pdb_visited = set()
    for idx in tqdm(split['test'], 'Filter'):
        pdb_name = get_pdb_name(dataset[idx].ligand_filename)
        if pdb_name not in train_pdb and pdb_name not in pdb_visited:
            test_id.append(idx)
            pdb_visited.add(pdb_name)

    print('Number of PDBs: %d' % len(test_id))
    print('Number of PDBs: %d' % len(pdb_visited))

    random.Random(args.seed).shuffle(test_id)
    test_id = test_id[:args.num_pockets]
    print('Number of selected: %d' % len(test_id))

    torch.save({
        'train': [],
        'val': [],
        'test': test_id,
    }, args.output_split)

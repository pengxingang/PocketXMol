import os
import argparse
import random
import torch
from tqdm.auto import tqdm

from utils.datasets import PocketLigandPairDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./data/crossdocked_pocket10')
    parser.add_argument('--dest', type=str, default='./data/crossdocked_pocket10_split.pt')
    parser.add_argument('--train', type=int, default=150000)
    parser.add_argument('--val', type=int, default=1000)
    parser.add_argument('--test', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()

    allowed_elements = {1,6,7,8,9,15,16,17}

    dataset = PocketLigandPairDataset(args.path)
    elements = {i:set() for i in range(90)}
    for i, data in enumerate(tqdm(dataset, desc='Filter')):
        for e in data.ligand_element:
            elements[e.item()].add(i)

    all_id = set(range(len(dataset)))
    blocked_id = set().union(*[
        elements[i] for i in elements.keys() if i not in allowed_elements
    ])
    allowed_id = list(all_id - blocked_id)
    random.Random(args.seed).shuffle(allowed_id)

    print('Allowed: %d' % len(allowed_id))

    train_id = allowed_id[:args.train]
    val_id = allowed_id[args.train : args.train + args.val]
    test_id = allowed_id[args.train + args.val : args.train + args.val + args.test]

    torch.save({
        'train': train_id,
        'val': val_id,
        'test': test_id,
    }, args.dest)

    print('Train %d, Validation %d, Test %d.' % (len(train_id), len(val_id), len(test_id)))
    print('Done.')

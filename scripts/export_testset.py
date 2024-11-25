import os
import shutil
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
    parser.add_argument('--split', type=str, default='./data/crossdocked_pocket10_test_split.pt')
    parser.add_argument('--original', type=str, default='./data/crossdocked')
    parser.add_argument('--dest', type=str, default='./data/export_test')
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    
    dataset = PocketLigandPairDataset(args.dataset)
    split = torch.load(args.split)
    subsets = {k:Subset(dataset, indices=v) for k, v in split.items()}
    test_set = subsets['test']
    print('Number of test datapoints: %d' % len(test_set))

    for i, data in enumerate(tqdm(test_set)):
        ligand_fn = data.ligand_filename
        protein_fn = os.path.join(os.path.dirname(data.ligand_filename), os.path.basename(data.ligand_filename)[:10]+'.pdb')
        ligand_dest = os.path.join(args.dest, '%d_%s' % (i, os.path.basename(ligand_fn)))
        protein_dest = os.path.join(args.dest, '%d_%s' % (i, os.path.basename(protein_fn)))

        shutil.copyfile(os.path.join(args.dataset, ligand_fn), ligand_dest)
        shutil.copyfile(os.path.join(args.original, protein_fn), protein_dest)

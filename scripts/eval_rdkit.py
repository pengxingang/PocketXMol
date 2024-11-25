import sys
sys.path.append('.')
import pandas as pd
import os
from tqdm.auto import tqdm
import torch
import numpy as np
import pandas as pd
# from copy import deepcopy
# from itertools import combinations
from multiprocessing import Pool

# from matplotlib import pyplot as plt

from rdkit import Chem
# from rdkit.Chem.QED import qed
from easydict import EasyDict

from utils.datasets import *
from utils.baseline import *
from utils.scoring_func import get_rdkit_rmsd
# from utils.transforms import LigandRandomMask


def parse_expr_name(expr_name):
    if expr_name.startswith('default'):
        s1 = expr_name.find('-')
        s2 = expr_name.find('_')
        return expr_name[:s1], int(expr_name[s1+1: s2])
    elif expr_name.startswith('vn_sample'):
        s1 = 9
        s2 = expr_name[s1+1:].find('_') + s1 + 1
        return expr_name[:s1], int(expr_name[s1+1: s2])


def load_test_set():
    dataset, subsets = get_dataset(EasyDict({
        'name': 'pl',
        'path': './data/crossdocked_pocket10',
        'split': './data/crossdocked_pocket10_test120_split.pt',
    }))
    _data_list = [d for d in subsets['test']]
    ref_mol_list = []
    for data in tqdm(_data_list):
        mol = next(iter(Chem.SDMolSupplier(os.path.join('../parsed_data/crossdocked_pocket10', data.ligand_filename))))
        ref_mol_list.append(mol)
    return ref_mol_list


def load_cvae():
    mols_lig, _, docked_lig = load_liGAN_mols('../baseline_data/liGAN_CVAE', 100)
    cvae_mol_list = []
    for group in tqdm(mols_lig):
        if group is None: continue
        for mol in group:
            cvae_mol_list.append(mol)
    return cvae_mol_list


def load_shitong():
    # Load results
    N = 100
    output_root = '../codes/outputs'  # shitong model
    cfg_name = 'default'
    results_dict = {}
    for expr_name in tqdm(os.listdir(output_root)):
        if not expr_name.startswith(cfg_name): continue
        expr_dir = os.path.join(output_root, expr_name)
        result_path = os.path.join(expr_dir, 'results.pt')
        if os.path.exists(result_path):
            result = torch.load(result_path)
            _, idx = parse_expr_name(expr_name)
            results_dict[idx] = result
    results = []
    for i in range(max(results_dict.keys()) + 1):
        if i in results_dict:
            results.append(results_dict[i])
        else:
            results.append(None)
    shitong_results = results[:N]
    shitong_mol_list = [r['mol'] for result in shitong_results for r in result]
    return shitong_mol_list


def load_new():
    # Load results
    N = 100
    output_root = './outputs/gen_0114'  # new
    # output_root = '../outputs/gen_0112_2'  # new model #!CHANGE HERE
    cfg_name = 'vn_sample'
    results_dict = {}
    for expr_name in tqdm(os.listdir(output_root)):
        if not expr_name.startswith(cfg_name): continue
        expr_dir = os.path.join(output_root, expr_name)
        result_path = os.path.join(expr_dir, 'results.pt')
        # result_path = os.path.join(expr_dir, 'results_1.pt')
        if os.path.exists(result_path):
            try:
                result = torch.load(result_path)
            except:
                print('Error', expr_name)
            _, idx = parse_expr_name(expr_name)
            results_dict[idx] = result
    results = []
    for i in range(max(results_dict.keys()) + 1):
        if i in results_dict:
            results.append(results_dict[i])
        else:
            results.append(None)
    new_results = results[:N]
    new_mol_list = [r['mol'] for result in new_results for r in result]
    return new_mol_list


def main():

    cvae_mol_list = load_cvae()
    cvae_rmsd = []
    with Pool(32) as pool:
        for r in tqdm(pool.imap_unordered(get_rdkit_rmsd, cvae_mol_list, chunksize=50),  total=len(cvae_mol_list)):
            cvae_rmsd.append(r)
    print(np.mean(cvae_rmsd), np.median(cvae_rmsd))
    np.save('outputs/rmsd/cvae.npy', cvae_rmsd)

    shitong_mol_list = load_shitong()
    shitong_rmsd = []
    with Pool(32) as pool:
        for r in tqdm(pool.imap_unordered(get_rdkit_rmsd, shitong_mol_list, chunksize=50),  total=len(shitong_mol_list)):
            shitong_rmsd.append(r)
    np.save('outputs/rmsd/shitong.npy', shitong_rmsd)
    print(np.mean(shitong_rmsd), np.median(shitong_rmsd))

    new_mol_list = load_new()
    new_rmsd = []
    with Pool(32) as pool:
        for r in tqdm(pool.imap_unordered(get_rdkit_rmsd, new_mol_list, chunksize=50),  total=len(new_mol_list)):
            new_rmsd.append(r)
    np.save('outputs/rmsd/new.npy', new_rmsd)
    print(np.mean(new_rmsd), np.median(new_rmsd))
    print()


def eval_rdkit():
    data_dir = './outputs/rmsd'
    cvae_rmsd = np.load(os.path.join(data_dir, 'cvae.npy'))
    shitong_rmsd = np.load(os.path.join(data_dir, 'shitong.npy'))
    new_rmsd = np.load(os.path.join(data_dir, 'new.npy'))
    print('cvae', np.nanmean(cvae_rmsd), np.nanmedian(cvae_rmsd))
    print('shitong', np.nanmean(shitong_rmsd), np.nanmedian(shitong_rmsd))
    print('new', np.nanmean(new_rmsd), np.nanmedian(new_rmsd))
    print()


def load_all_mols(model_name, n_pock):

if __name__ == '__main__':
    # main()
    eval_rdkit()
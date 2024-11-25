import os
import argparse
from tqdm.auto import tqdm
import torch
from rdkit import Chem
from joblib import Parallel, delayed

from utils.datasets import *
from utils.visualize import *
from utils.transforms import *
from utils.docking import *
from utils.sascorer import *
from utils.baseline import *
from utils.similarity import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data/crossdocked_pocket10')
    parser.add_argument('--ref_docked', type=str, default='./data/crossdocked_pocket10_test120_split_test_docked.pt')
    
    parser.add_argument('--split_path', type=str, default='./data/crossdocked_pocket10_clustered_train100k_test155.pt')
    parser.add_argument('-n', '--num_workers', type=int, default=32)
    parser.add_argument('-g', '--limit_gen', type=int, default=-1)
    parser.add_argument('-r', '--limit_ref', type=int, default=-1)
    parser.add_argument('-s', '--shuffle', action='store_true', default=False)
    parser.add_argument('--high_affinity', action='store_true', default=False)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    split_path = args.split_path

    # Load docking results
    ref_docked = torch.load(args.ref_docked)
    vina_ref = [d[0].affinity for d in ref_docked]
    # Load dataset
    dataset = PocketLigandPairDataset(dataset_path, transform=None)
    split = torch.load(split_path)
    subsets = {k:Subset(dataset, indices=v) for k, v in split.items()}
    train_set = subsets['train']

    ligands_train = []
    for data in tqdm(train_set):
        suppl = Chem.SDMolSupplier(os.path.join(dataset_path, data.ligand_filename))
        for mol in suppl:
            ligands_train.append(mol)

    def parse_expr_name(expr_name):
        s1 = expr_name.find('-')
        s2 = expr_name.find('_')
        return expr_name[:s1], int(expr_name[s1+1: s2])

    # Load results
    N = 100
    output_root = './outputs'
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
    results = results[:N]

    ligands_gen = []
    if args.high_affinity:
        for i in range(len(vina_ref[:N])):
            score_ref = vina_ref[i]
            for docked in results[i]:
                aff = docked['vina'][0].affinity
                if aff <= score_ref:
                    ligands_gen.append(docked['mol'])
    else:
        for gen in results:
            for g in gen:
                ligands_gen.append(g['mol'])

    smiles_train = set()
    for m in tqdm(ligands_train):
        smiles_train.add(Chem.MolToSmiles(Chem.RemoveHs(m)))
    ligands_train_unique = [Chem.MolFromSmiles(s) for s in tqdm(smiles_train)]

    if args.shuffle:
        random.shuffle(ligands_train_unique)
        random.shuffle(ligands_gen)

    ligands_train_unique = ligands_train_unique[:args.limit_ref]
    ligands_gen = ligands_gen[:args.limit_gen]

    print('[INFO]')
    print('  Number of unique refs: %d' % len(ligands_train_unique))
    print('  Number of generated:   %d' % len(ligands_gen))

    sim_matrix = Parallel(n_jobs=args.num_workers)(delayed(tanimoto_sim_N_to_1)(ligands_train_unique, gen) for gen in tqdm(ligands_gen))
    if args.high_affinity:
        torch.save(sim_matrix, 'sim_matrix_g%d_r%d_highaffinity.pt' % (args.limit_gen, args.limit_ref,))
    else:   
        torch.save(sim_matrix, 'sim_matrix_g%d_r%d.pt' % (args.limit_gen, args.limit_ref,))


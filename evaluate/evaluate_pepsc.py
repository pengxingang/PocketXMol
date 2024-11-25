"""
mainly calculate the side chain rmsd (for pepsc task)
"""
import sys
import os
import argparse
import pandas as pd
import pickle
from multiprocessing import Pool
from tqdm import tqdm
import shutil
sys.path.append('.')
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from Bio.PDB.DSSP import DSSP
from rdkit.Chem.rdMolAlign import CalcRMS
from rdkit import Chem

from utils.reconstruct import *
from utils.misc import *
from utils.scoring_func import *
from utils.evaluation import *
from utils.dataset import TestTaskDataset
from utils.docking_vina import VinaDockingTask
from process.process_torsional_info import get_mol_from_data
from evaluate.evaluate_mols import evaluate_mol_dict, get_mols_dict_from_gen_path,\
                        get_dir_from_prefix, combine_gt_gen_metrics
from evaluate.utils_eval import combine_receptor_ligand, combine_chains
from evaluate.rosetta import pep_score
from process.utils_process import get_pdb_angles



# def load_baseline_mols(gen_path, gt_dir, protein_dir, file_dir_name='SDF'):
#     df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))
#     df_gen_raw = df_gen.copy()
#     df_gen['file_path'] = df_gen['filename'].apply(lambda x: os.path.join(gen_path, file_dir_name, x))
#     df_gen['gt_path'] = df_gen['data_id'].apply(lambda x: os.path.join(gt_dir, x+'_pep.pdb'))
#     df_gen['protein_path'] = df_gen['data_id'].apply(lambda x: os.path.join(protein_dir, x+'_pro.pdb'))

#     df_gen = df_gen.groupby('data_id').agg(dict(
#         aaseq=lambda x: x.tolist(),
#         filename=lambda x: x.tolist(),
#         file_path=lambda x: x.tolist(),
#         gt_path=lambda x: x.iloc[0],
#         protein_path=lambda x: x.iloc[0],
#     ))
#     df_gen['data_id'] = df_gen.index
#     gen_dict = df_gen.to_dict('index')
#     return gen_dict, df_gen_raw



def load_gen_mols(gen_path, gt_dir, protein_dir):
    df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))
    df_gen_raw = df_gen.copy()
    df_gen['file_path'] = df_gen['filename'].apply(lambda x: os.path.join(gen_path, 'SDF', x))
    df_gen['gt_path'] = df_gen['data_id'].apply(lambda x: os.path.join(gt_dir, x+'_pep.pdb'))
    df_gen['protein_path'] = df_gen['data_id'].apply(lambda x: os.path.join(protein_dir, x+'_pro.pdb'))

    df_gen = df_gen.groupby('data_id').agg(dict(
        aaseq=lambda x: x.tolist(),
        filename=lambda x: x.tolist(),
        tag=lambda x: x.tolist(),
        file_path=lambda x: x.tolist(),
        gt_path=lambda x: x.iloc[0],
        protein_path=lambda x: x.iloc[0],
    ))
    df_gen['data_id'] = df_gen.index
    gen_dict = df_gen.to_dict('index')
    return gen_dict, df_gen_raw


def get_sc_rmsd(mol_0, mol_1):
    parser = PDBParser()
    
    mol_0 = parser.get_structure('mol_0', mol_0)[0]
    mol_1 = parser.get_structure('mol_1', mol_1)[0]
    
    # get chain
    mol_0 = mol_0.child_list[0]
    mol_1 = mol_1.child_list[0]
    if len(mol_0) != len(mol_1):
        return []
    
    sc_rmsd_list = []
    for res_0, res_1 in zip(mol_0, mol_1):
        if res_0.resname == res_1.resname and res_1.resname != 'GLY':
            rmsd = get_res_sc_rmsd(res_0, res_1)
            sc_rmsd_list.append(rmsd)
    
    return sc_rmsd_list


def get_res_sc_rmsd(res_0, res_1):
    bb_names = ['N', 'CA', 'C', 'O']
    atom_names_0 = [a.name for a in res_0]
    atom_names_1 = [a.name for a in res_1]
    # remove bb and H atoms
    atom_names_0 = [a for a in atom_names_0 if a not in bb_names and a[0] != 'H']
    atom_names_1 = [a for a in atom_names_1 if a not in bb_names and a[0] != 'H']
    
    atom_rmsd = []
    for name in atom_names_0:
        if name not in atom_names_1:
            continue
            # return np.nan
        else:
            atom_0 = res_0[name]
            atom_1 = res_1[name]
            rmsd = np.linalg.norm(atom_0.coord - atom_1.coord)
            atom_rmsd.append(rmsd)
    return np.mean(atom_rmsd) if len(atom_rmsd) > 0 else np.nan

    

def evaluate_one_input(gen_info):
    # parse gt
    gt_path = gen_info['gt_path']
    pdb_parser = PDBParser()
    gt_pdb = pdb_parser.get_structure('gt', gt_path)
    gt_seq = seq1(''.join([r.resname for r in gt_pdb.get_residues()]))
    non_std = 'X' in gt_seq
    len_pep = len(gt_seq)
    
    result_list = []
    # gt_path_mol = gt_path.replace('peptides', 'mols').replace('_pep.pdb', '_mol.sdf')
    for i in range(len(gen_info['file_path'])):
        tag = gen_info['tag'][i]
        aaseq = gen_info['aaseq'][i]
        if aaseq != aaseq:  # aaseq is nan
            continue
        sc_rmsd = get_sc_rmsd(gt_path, gen_info['file_path'][i])
        
        result_list.append({
            'data_id': gen_info['data_id'],
            'non_std_data': non_std,
            'filename': gen_info['filename'][i],
            'succ': (tag != tag),  # tag is nan: means succ
            'gt_seq': gt_seq,
            'aaseq': aaseq,
            'sc_rmsd_list': sc_rmsd,
        })
    
    return result_list



def evaluate_metrics(gen_path, gt_dir, protein_dir, num_workers):
    if 'baseline' in gen_path:
        gen_dict, _ = load_baseline_mols(gen_path, gt_dir, protein_dir)
    else:
        gen_dict, df_gen = load_gen_mols(gen_path, gt_dir, protein_dir)
    
    result_list = []
    # basic results
    for data_id, gen_info in tqdm(gen_dict.items(), total=len(gen_dict)):
        result_list.extend(
            evaluate_one_input(gen_info)
        )

    # return results
    result = pd.DataFrame(result_list)
    return result
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_root', type=str, default='outputs_paper/pepdesign_pepbdb')
    parser.add_argument('--exp_name', type=str, default='msel_pepsc')
    parser.add_argument('--gt_dir', type=str, default='data/pepbdb/files/peptides')
    parser.add_argument('--protein_dir', type=str, default='data/pepbdb/files/proteins')
    parser.add_argument('--num_workers', type=int, default=64)
    parser.add_argument('--baseline', type=str, default='')
    args = parser.parse_args()
    
    if args.baseline == '':  # our method generated
        gen_path = get_dir_from_prefix(args.result_root, args.exp_name)
    else:  # baseline method generated 
        gen_path = f'./baselines/pepdesign/{args.baseline}'

    df_metric = evaluate_metrics(gen_path, args.gt_dir, args.protein_dir, args.num_workers)
    df_metric.to_csv(os.path.join(gen_path, 'sc_metrics.csv'), index=False)

    print('Done.')

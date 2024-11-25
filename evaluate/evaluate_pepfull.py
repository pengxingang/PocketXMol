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
# from utils.dataset import TestTaskDataset
# from utils.docking_vina import VinaDockingTask
# from process.process_torsional_info import get_mol_from_data
from evaluate.evaluate_mols import evaluate_mol_dict, get_mols_dict_from_gen_path,\
                        get_dir_from_prefix, combine_gt_gen_metrics
from evaluate.utils_eval import combine_receptor_ligand, combine_chains
# from evaluate.rosetta import pep_score
from process.utils_process import get_pdb_angles


def get_ss(pdb_path):
    parser = PDBParser()
    structure = parser.get_structure('pep', pdb_path)[0]
    try:
        dssp = DSSP(structure, pdb_path, dssp='mkdssp')
    except FileNotFoundError:
        dssp_path = os.path.expanduser('~/anaconda3/envs/mol/bin/mkdssp')
        dssp = DSSP(structure, pdb_path, dssp=dssp_path)
    ss = [dssp[key][2] for key in dssp.keys()]
    return ''.join(ss)


def load_baseline_mols(gen_path, gt_dir, protein_dir, file_dir_name='SDF'):
    df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))
    df_gen_raw = df_gen.copy()
    df_gen['file_path'] = df_gen['filename'].apply(lambda x: os.path.join(gen_path, file_dir_name, x))
    df_gen['gt_path'] = df_gen['data_id'].apply(lambda x: os.path.join(gt_dir, x+'_pep.pdb'))
    df_gen['protein_path'] = df_gen['data_id'].apply(lambda x: os.path.join(protein_dir, x+'_pro.pdb'))

    df_gen = df_gen.groupby('data_id').agg(dict(
        aaseq=lambda x: x.tolist(),
        filename=lambda x: x.tolist(),
        file_path=lambda x: x.tolist(),
        gt_path=lambda x: x.iloc[0],
        protein_path=lambda x: x.iloc[0],
    ))
    df_gen['data_id'] = df_gen.index
    gen_dict = df_gen.to_dict('index')
    return gen_dict, df_gen_raw



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


def prepare_complex(cpx_dir, df_gen):
    # make the complex dir
    # cpx_dir = os.path.join(gen_dir, 'complex')
    os.makedirs(cpx_dir, exist_ok=True)
    
    # combine rec and lig as one complex pdb
    for _, line in tqdm(df_gen.iterrows(), total=len(df_gen), desc='combine rec and lig'):
        protein_path = line['protein_path']

        # combine chains in the protein
        combchain_dir = os.path.dirname(protein_path).replace('/proteins', '/proteins_combchain')
        os.makedirs(combchain_dir, exist_ok=True)
        combchain_path = os.path.join(combchain_dir,
                        os.path.basename(protein_path).replace('.pdb', '_combchain.pdb'))
        if not os.path.exists(combchain_path):
            combine_chains(protein_path, save_path=combchain_path)

        # combine rec and lig
        filename = line['filename']
        file_path = line['file_path']
        cpx_path = os.path.join(cpx_dir, filename.replace('.pdb', '_cpx.pdb'))
        if not os.path.exists(cpx_path):
            combine_receptor_ligand(combchain_path, file_path, cpx_path)


def copy_complex_dir(cpx_dir, tgt_dir):
    os.makedirs(tgt_dir)
    for filename in os.listdir(cpx_dir):
        if not filename.endswith('.pdb'):
            print(f'Not pdb file in complex_dir: {filename}')
        src_path = os.path.join(cpx_dir, filename)
        tgt_path = os.path.join(tgt_dir, filename.replace('_cpx.pdb', '.pdb'))
        shutil.copy(src_path, tgt_path)


def seq_identity(seq1, seq2):
    if len(seq1) != len(seq2):
        return np.nan
        # raise ValueError('Length of two sequences are not equal.')
    return sum((s1 == s2) and s2 != 'X' for s1, s2 in zip(seq1, seq2)) / len(seq1)


def get_bb_rmsd(mol_0, mol_1, len_bb):
    if mol_0.endswith('.sdf'):
        mol_0 = Chem.MolFromMolFile(mol_0)
    elif mol_0.endswith('.pdb'):
        mol_0 = Chem.MolFromPDBFile(mol_0)
    else:
        raise ValueError('Unknown file format for mol_0.')
    if mol_1.endswith('.sdf'):
        mol_1 = Chem.MolFromMolFile(mol_1)
    elif mol_1.endswith('.pdb'):
        mol_1 = Chem.MolFromPDBFile(mol_1)
    else:
        raise ValueError('Unknown file format for mol_1.')
    if mol_0 is None or mol_1 is None:
        return np.nan

    # get bb
    bb_smarts = '[#7][#6][#6](=[#8])' * len_bb
    bb_smarts = Chem.MolFromSmarts(bb_smarts)
    bb_0_matches = mol_0.GetSubstructMatches(bb_smarts)
    bb_1_matches = mol_1.GetSubstructMatches(bb_smarts)
    if len(bb_0_matches) == 0 or len(bb_1_matches) == 0:
        bb_smarts = '[#7][#6][#6](~[#8])' * len_bb  # nonstd C=O was parsed as C-O
        bb_smarts = Chem.MolFromSmarts(bb_smarts)
        bb_0_matches = mol_0.GetSubstructMatches(bb_smarts)
        bb_1_matches = mol_1.GetSubstructMatches(bb_smarts)
        if len(bb_0_matches) == 0 or len(bb_1_matches) == 0:
            return np.nan
    
    atom_mapping = []
    for bb_0_match in bb_0_matches:
        for bb_1_match in bb_1_matches:
            atom_mapping.append(list(zip(bb_0_match, bb_1_match)))
            
    rmsd = CalcRMS(mol_0, mol_1, map=atom_mapping)
    return rmsd
    
    

def evaluate_one_input(gen_info):
    # parse gt
    gt_path = gen_info['gt_path']
    pdb_parser = PDBParser()
    gt_pdb = pdb_parser.get_structure('gt', gt_path)
    gt_seq = seq1(''.join([r.resname for r in gt_pdb.get_residues()]))
    non_std = 'X' in gt_seq
    len_pep = len(gt_seq)
    
    result_list = []
    gt_path_mol = gt_path.replace('peptides', 'mols').replace('_pep.pdb', '_mol.sdf')
    for i in range(len(gen_info['file_path'])):
        # succ
        if 'tag' in gen_info:
            tag = gen_info['tag'][i]
            tag = tag if tag == tag else 'succ'
        else:
            tag = 'succ'
        # aar
        aaseq = gen_info['aaseq'][i]
        if aaseq == aaseq:
            seq_iden = seq_identity(gt_seq, aaseq)
        else:
            seq_iden = np.nan
        # RMSD
        file_path = gen_info['file_path'][i]
        gen_path_mol = file_path.replace('.pdb', '_mol.sdf')
        if os.path.exists(gen_path_mol):
            rmsd = get_bb_rmsd(gt_path_mol, gen_path_mol, len_pep)
        else:
            rmsd = np.nan
        if rmsd != rmsd:
            rmsd = get_bb_rmsd(gt_path_mol, file_path, len_pep)
            
        # ss
        if tag == 'succ':
            ss = get_ss(file_path)
            # angles
            angle_results = get_pdb_angles(file_path, angle_list=["psi", "phi"])
            if angle_results is not None:
                psi = [angles['psi'] for angles in angle_results.values()]
                phi = [angles['phi'] for angles in angle_results.values()]
            else:
                psi, phi = [], []
        else:
            ss = ''
            psi, phi = [], []
        
        result_list.append({
            'data_id': gen_info['data_id'],
            'non_std_data': non_std,
            'filename': gen_info['filename'][i],
            'tag': tag,
            'succ': (tag == 'succ'),
            'aaseq': aaseq,
            'seq_iden': seq_iden,
            'rmsd': rmsd,
            'ss': ss,
            'phi': phi,
            'psi': psi,
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

    # rosetta socres
    # df_score = calc_rosetta_pep_score(gen_path, gen_dict, num_workers)
    # df_data_id = df_gen[['data_id', 'filename']].copy()
    # df_data_id['filename'] = df_data_id['filename'].apply(lambda x: x.replace('.pdb', ''))
    # df_score = df_data_id.merge(df_score, on='filename', how='right')
    # df_score = pd.DataFrame()

    # return results
    result = pd.DataFrame(result_list)
    return result
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_root', type=str, default='outputs_test/pepdesign_pepbdb')
    parser.add_argument('--exp_name', type=str, default='base_pxm')
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
    df_metric.to_csv(os.path.join(gen_path, 'metrics.csv'), index=False)

    print('Done.')

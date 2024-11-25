"""
This file contains the code for evaluating the structure consistency of the PDL1 peptide structures
predicted by the model and the alphafold.
NOTE: use pymol env
"""

import numpy as np
import pickle
import os
from functools import partial
from multiprocessing import Pool
import subprocess

import pandas as pd
from tqdm import tqdm

from pymol import cmd


BB_ATOMS = ['N', 'CA', 'C', 'O']

def bb_rmsd_pymol(pose1, pose2, align_chainid=['R', 'B'], rmsd_chainid=['L', 'C']):
    
    cmd.delete('all')
    cmd.load(pose1, 'pose1')
    cmd.load(pose2, 'pose2')
    
    cmd.align(f'pose1 and chain {align_chainid[0]}',
              f'pose2 and chain {align_chainid[1]}')
    
    rmsd = cmd.rms_cur(f'pose1 and chain {rmsd_chainid[0]} and bb.',
                       f'pose2 and chain {rmsd_chainid[1]} and bb.',
                       matchmaker=-1, cycles=0)
    return rmsd

def get_paths_one_gen(cpx_dir, af_dir, filename, filename2seq):
    gen_path = os.path.join(cpx_dir, filename)
    # # alphafold
    if filename not in filename2seq:
        return
    aaseq = filename2seq[filename]
    filename_af2 = 'PDL1_' + aaseq
    af_path = os.path.join(af_dir, filename_af2, 'ranked_0.pdb')
    if not os.path.exists(af_path):
        return
    
    # succ
    this_path_dict = {
        'gen': gen_path,
        'alphafold': af_path
    }
    return this_path_dict


def rmsd_one_gen(path_dict):
    gen_path = path_dict['gen']
    rmsd_dict = {'filename': os.path.basename(gen_path).replace('.pdb', '')}
    for key in path_dict:
        if key != 'gen':
            rmsd_this = bb_rmsd_pymol(gen_path, path_dict[key])
            rmsd_dict[key] = rmsd_this
    return rmsd_dict


def calc_consistency_one_gen(filename, cpx_dir, af_dir, save_dir, filename2seq):
    save_path = os.path.join(save_dir, filename.replace('.pdb', '.pkl'))
    if os.path.exists(save_path):
        return
    
    path_dict = get_paths_one_gen(cpx_dir, af_dir, filename, filename2seq)
    if path_dict is None:
        return
    rmsd_dict = rmsd_one_gen(path_dict)

    # save
    with open(save_path, 'wb') as f:
        pickle.dump(rmsd_dict, f)
    
    
        
# def calc_consistency(outdir, savedir, df_meta):
def calc_consistency(df_meta, save_dir, complex_dir, af_dir):
    df_meta['filename'] = df_meta['filename'].astype('str') + '.pdb'
    filename2seq = df_meta.set_index('filename')['aaseq'].to_dict()
    
    all_files = os.listdir(complex_dir)
    all_files = set(all_files)
    with Pool(16) as p:
        list(tqdm(p.imap_unordered(
            partial(calc_consistency_one_gen, cpx_dir=complex_dir, af_dir=af_dir, save_dir=save_dir,
                    filename2seq=filename2seq),
            all_files), total=len(all_files), desc='Calculating consistency...'
        ))
    return

if __name__ == '__main__':
    
    # df_meta = pd.read_csv('outputs_use/PDL1/combine_1211/seq_sel_and_top.csv')
    # outdir = 'outputs_use/PDL1/combine_1211'  # note: need change function parameter format
    
    # denovo
    # df_meta = pd.read_csv('outputs_use/PDL1/combine_1211/seq_sel_568.csv')
    # complex_dir = 'outputs_use/PDL1/combine_1211/complex'
    # af_dir = 'outputs_use/PDL1/combine_1211/post_1216/alphafold/outputs'
    # save_dir = 'outputs_use/PDL1/combine_1211/post_1216/struc_consis_af'

    # opt
    df_meta = pd.read_csv('outputs_use/PDL1/combine_pep_opt_1220/merge.csv')
    complex_dir = 'outputs_use/PDL1/combine_pep_opt_1220/complex'
    af_dir = 'outputs_use/PDL1/combine_pep_opt_1220/alphafold/outputs'
    save_dir = 'outputs_use/PDL1/combine_pep_opt_1220/struc_consis_af'

    # get pdb path dict
    os.makedirs(save_dir, exist_ok=True)
    calc_consistency(df_meta, save_dir, complex_dir, af_dir)
    
    print('Done.')
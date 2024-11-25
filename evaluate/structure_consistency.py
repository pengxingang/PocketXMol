"""
This file contains the code for evaluating the structure consistency of the PDL1 peptide structures
predicted by the model, the rosetta and the alphafold.
"""

import numpy as np
import pickle
import os
from functools import partial
from multiprocessing import Pool
import subprocess

import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB import Superimposer
PATH_DOCKQ = 'XXX'  # dockq path

BB_ATOMS = ['N', 'CA', 'C', 'O']

def bb_rmsd_biopython(pose1, pose2, chainid='L'):
    """
    Use biopython to read pdb files and calculate the rmsd of backbone atoms of given chain.
    Note: not moving the inputs so the inputs should be aligned in advance
    """
    parser = PDBParser(QUIET=True)
    pdb1 = parser.get_structure('pose1', pose1)[0][chainid]
    pdb2 = parser.get_structure('pose2', pose2)[0]
    try:
        pdb2 = pdb2[chainid]
    except KeyError:
        assert 'alphafold' in pose2
        pdb2 = pdb2['C']

    bb_pdb1 = np.array([atom.coord for atom in pdb1.get_atoms() if atom.name in BB_ATOMS])
    bb_pdb2 = np.array([atom.coord for atom in pdb2.get_atoms() if atom.name in BB_ATOMS])
    assert len(bb_pdb1) == len(bb_pdb2), 'The number of backbone atoms are not equal.'

    # calc rmsd
    rmsd = np.sqrt((bb_pdb1 - bb_pdb2) ** 2).sum(axis=-1).mean(axis=0)
    return rmsd

def bb_rmsd_dockq_PDL1(pose1, pose2):
    # run dockq
    cmd = ['python', f'{PATH_DOCKQ}/DockQ.py', pose1, pose2]
    cmd += ['-model_chain1', 'R', '-native_chain1', 'B', '-no_needle']
    output = subprocess.run(cmd, capture_output=True, text=True)
    if output.returncode != 0:
        raise ValueError('DockQ errored:' + output.stderr)
    
    results = output.stdout.split('\n')[-4:-1]
    if 'DockQ' not in results[-1]:
        raise ValueError('DockQ failed: ' + output.stdout)
    
    # irmsd = results[0].split()[1]
    lrmsd = results[1].split()[1]
    # dockq = results[2].split()[1]
    print(output.stdout)
    
    return float(lrmsd)


def bb_rmsd_dockq_PDL1(pose1, pose2):
    # run dockq
    cmd = ['python', f'{PATH_DOCKQ}/DockQ.py', pose1, pose2]
    cmd += ['-model_chain1', 'R', '-native_chain1', 'B', '-no_needle']
    output = subprocess.run(cmd, capture_output=True, text=True)
    if output.returncode != 0:
        raise ValueError('DockQ errored:' + output.stderr)
    
    results = output.stdout.split('\n')[-4:-1]
    if 'DockQ' not in results[-1]:
        raise ValueError('DockQ failed: ' + output.stdout)
    
    # irmsd = results[0].split()[1]
    lrmsd = results[1].split()[1]
    # dockq = results[2].split()[1]
    print(output.stdout)
    
    return float(lrmsd)


def bb_rmsd_biopython_align_PDL1(pose_fixed, pose_move):
    """
    Use biopython to read pdb files and calculate the rmsd of backbone atoms of given chain.
    Note: Align sub-structures align_chainid
    align the first chain and rmsd the second
    """
    parser = PDBParser(QUIET=True)
    pdb_fixed = parser.get_structure('pose1', pose_fixed)[0]
    pdb_move = parser.get_structure('pose2', pose_move)[0]

    sup = Superimposer()
    pose_fixed_alg, pose_fixed_calc = list(pdb_fixed.get_chains())
    pose_move_alg, pose_move_calc = list(pdb_move.get_chains())
    # sup.set_atoms(pose_fixed_alg, pose_move_alg)
    
    bb_pdb1_atoms = [atom for atom in pose_fixed_calc.get_atoms() if atom.name in BB_ATOMS]
    bb_pdb2_atoms = [atom for atom in pose_move_calc.get_atoms() if atom.name in BB_ATOMS]

    sup.set_atoms(bb_pdb1_atoms, bb_pdb2_atoms)
    #! this aligned lig chain. not completely reasonable. but if it is small, gloabl align must be small.
    sup.apply(bb_pdb2_atoms)

    bb_pdb1 = np.array([atom.coord for atom in bb_pdb1_atoms])
    bb_pdb2 = np.array([atom.coord for atom in bb_pdb2_atoms])
    assert len(bb_pdb1) == len(bb_pdb2), 'The number of backbone atoms are not equal.'


    # calc rmsd
    rmsd = np.sqrt((bb_pdb1 - bb_pdb2) ** 2).sum(axis=-1).mean(axis=0)
    return rmsd



def prepare_pdb_paths(outdir, filename2seq):
    complex_dir = os.path.join(outdir, 'complex')
    other_dirs = {
        'foldx': os.path.join(outdir, 'foldx', 'repaired'),
        'rosetta': os.path.join(outdir, 'rosetta', 'refine_score'),
        'alphafold': os.path.join(outdir,'alphafold', 'outputs')
    }
    
    # path in rosetta dir should remove the appendix added by rosetta
    rose_files = [f[:-14]+'.pdb' for f in os.listdir(other_dirs['rosetta']) if f.endswith('.pdb')]
    path_list = []
    for filename in tqdm(os.listdir(complex_dir), desc='finding path...'):
        gen_path = os.path.join(complex_dir, filename)
        # foldx
        foldx_path = os.path.join(other_dirs['foldx'], filename)
        if not os.path.exists(foldx_path):
            continue
        # rosetta
        rose_path = os.path.join(other_dirs['rosetta'], filename)
        if filename not in rose_files:
            continue
        # alphafold
        filename_noext = filename.replace('.pdb', '')
        if filename_noext not in filename2seq:
            continue
        aaseq = filename2seq[filename_noext]
        filename_af2 = 'PDL1_' + aaseq
        af_path = os.path.join(other_dirs['alphafold'], filename_af2, 'ranked_0.pdb')
        if not os.path.exists(af_path):
            continue
        # combine
        this_path_dict = {
            'gen': gen_path,
            'foldx': foldx_path,
            'rosetta': rose_path,
            'alphafold': af_path
        }
        path_list.append(this_path_dict)
    return path_list


def get_paths_one_gen(outdir, filename, filename2rosename, filename2seq):
    complex_dir = os.path.join(outdir, 'complex')
    other_dirs = {
        'foldx': os.path.join(outdir, 'foldx', 'repaired'),
        'rosetta': os.path.join(outdir, 'rosetta', 'refine_score'),
        'alphafold': os.path.join(outdir,'alphafold', 'outputs')
    }
    
    gen_path = os.path.join(complex_dir, filename)
    # # foldx
    foldx_path = os.path.join(other_dirs['foldx'], filename)
    if not os.path.exists(foldx_path):
        return
    # # rosetta
    if filename not in filename2rosename:
        return
    rose_filename = filename2rosename[filename]
    rose_path = os.path.join(other_dirs['rosetta'], rose_filename)
    # # alphafold
    if filename not in filename2seq:
        return
    aaseq = filename2seq[filename]
    filename_af2 = 'PDL1_' + aaseq
    af_path = os.path.join(other_dirs['alphafold'], filename_af2, 'ranked_0.pdb')
    if not os.path.exists(af_path):
        return
    
    # succ
    this_path_dict = {
        'gen': gen_path,
        # 'foldx': foldx_path,
        # 'rosetta': rose_path,
        'alphafold': af_path
    }
    return this_path_dict


def rmsd_one_gen(path_dict):
    gen_path = path_dict['gen']
    rmsd_dict = {'filename': os.path.basename(gen_path).replace('.pdb', '')}
    for key in path_dict:
        if key != 'gen':
            if key != 'alphafold':
                rmsd_this = bb_rmsd_biopython(gen_path, path_dict[key])
            else:
                rmsd_this = bb_rmsd_biopython_align_PDL1(gen_path, path_dict[key])
            rmsd_dict[key] = rmsd_this
    return rmsd_dict


def calc_consistency_one_gen(filename, outdir, savedir, filename2rosename, filename2seq):
    save_path = os.path.join(savedir, filename.replace('.pdb', '.pkl'))
    if os.path.exists(save_path):
        return
    
    path_dict = get_paths_one_gen(outdir, filename, filename2rosename, filename2seq)
    if path_dict is None:
        return
    rmsd_dict = rmsd_one_gen(path_dict)

    # save
    with open(save_path, 'wb') as f:
        pickle.dump(rmsd_dict, f)
    
    
        
def calc_consistency(outdir, savedir, df_meta):
    df_meta['filename'] = df_meta['filename'] + '.pdb'
    df_meta['rosename'] = df_meta['description'].str[:-5] + '.pdb'
    filename2rosename = df_meta.set_index('filename')['rosename'].to_dict()
    filename2seq = df_meta.set_index('filename')['aaseq'].to_dict()
    
    all_files = os.listdir(os.path.join(outdir, 'complex'))
    all_files = set(all_files)
    with Pool(168) as p:
        list(tqdm(p.imap_unordered(
            partial(calc_consistency_one_gen, outdir=outdir, savedir=savedir,
                    filename2rosename=filename2rosename, filename2seq=filename2seq),
            all_files), total=len(all_files), desc='Calculating consistency...'
        ))
    return

if __name__ == '__main__':
    outdir = 'outputs_use/PDL1/combine_1211'
    df_meta = pd.read_csv('outputs_use/PDL1/combine_1211/seq_sel_and_top.csv')

    # get pdb path dict
    savedir = os.path.join(outdir, 'struc_consis_af')
    os.makedirs(savedir, exist_ok=True)
    calc_consistency(outdir, savedir, df_meta)
    
    print('Done.')
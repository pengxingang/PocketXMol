import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
from tqdm import tqdm
from copy import deepcopy
from multiprocessing import Pool
from functools import partial

def assign_mol_conformers_mmff(mol, num_confs=1, num_threads=0):
    """Assign conformers to a molecule using RDKit's ETKDG method"""
    # Generate 3D conformers
    mol = Chem.AddHs(deepcopy(mol))
    cids = AllChem.EmbedMultipleConfs(mol, num_confs)
    # Optimize the conformers
    AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=num_threads)
    mol = Chem.RemoveAllHs(mol)
    return mol

def add_rdkit_conf_single(filename, in_dir, out_dir):
    mol = Chem.MolFromMolFile(os.path.join(in_dir, filename))
    try:
        mol = assign_mol_conformers_mmff(mol)
    except Exception as e:
        print(f'Error with {filename}: {e}')
        return
    out_fn = os.path.join(out_dir, filename)
    writer = Chem.SDWriter(out_fn)
    writer.write(mol)
    writer.close()
    return mol

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--in_dir', type=str, default='baselines/linking/protacdb/delinker_known_anchors/2DSDF')
    # parser.add_argument('--out_dir', type=str, default='baselines/linking/protacdb/delinker_known_anchors/SDF')
    parser.add_argument('--in_dir', type=str, default='baselines/linking/moad/delinker_known_anchors/2DSDF')
    parser.add_argument('--out_dir', type=str, default='baselines/linking/moad/delinker_known_anchors/SDF')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    filenames = [f for f in os.listdir(args.in_dir) if f.endswith('.sdf')]
    with Pool(64) as p:
        list(tqdm(p.imap_unordered(partial(add_rdkit_conf_single,
                    in_dir=args.in_dir, out_dir=args.out_dir),
                    filenames), total=len(filenames)))
    print('Done')
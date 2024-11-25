import sys

sys.path.append(".")

import argparse
import os
import shutil
import subprocess
import tempfile
import pickle

from tqdm import tqdm
import pandas as pd
from Bio.PDB import Selection
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import PDBIO
from distrun.api.joblib import Parallel as d_Parallel
from distrun.api.joblib import delayed as d_delayed
from joblib import Parallel, delayed

from evaluate.rosetta import fixbb_repack

import gzip
import warnings
from Bio import BiopythonExperimentalWarning



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # rfdiffusion+mpnn pepfull
    parser.add_argument('--complex_dir', type=str, default='baselines/pepdesign/full_rfdiff_mpnn/complex_for_rosetta')
    parser.add_argument('--pack_dir', type=str, default='baselines/pepdesign/full_rfdiff_mpnn/rosetta/pack')
    parser.add_argument('--num_workers', type=int, default=126)
    args = parser.parse_args()
    os.makedirs(args.pack_dir, exist_ok=True)
    
    fixbb_repack(args.complex_dir, args.pack_dir, replace=False, n_proc=args.num_workers)

    print('Done.')


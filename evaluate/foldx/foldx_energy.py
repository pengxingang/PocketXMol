import sys
sys.path.append("../..")

import os
import pandas as pd
from tqdm import tqdm
import shutil
import argparse

from PDButils.FoldX import FoldXSession
# from distrun.api.joblib import Parallel, delayed
import numpy as np
# from joblib import Parallel as joblib_Parallel
# from joblib import delayed as joblib_delayed
from multiprocessing import Pool
from functools import partial


def fetch_binding_affinity(path):
    with open(path, 'r') as f:
        u = f.readlines()
    line = u[-1].split("\t")
    result = {
        'clash_pmhc': float(line[-5]),
        'clash_tcr': float(line[-4]),
        'energy': float(line[-3]),
        'stable_pmhc': float(line[-2]),
        'stable_tcr': float(line[-1]),
    }
    return result

def process_one_line(input_path, chain_tuple):
    filename = os.path.basename(input_path)
    input_dir = os.path.dirname(input_path)

    with FoldXSession() as session:
        session.preprocess_data(input_dir, filename)
        session.execute_foldx(pdb_name=filename,
            command_name='AnalyseComplex', options=[f'--analyseComplexChains={chain_tuple}'])
        fxout_path = session.path(f'Summary_{filename.split(".")[0]}_AC.fxout')

        assert(os.path.exists(fxout_path))
        result = fetch_binding_affinity(fxout_path)
        result = {'filename': filename.replace('.pdb', ''), **result}

    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--chain_tuple', type=str, default = 'AB,CD')
    args = parser.parse_args()

    root_dir = 'outputs/foldx/repaired'
    
    result_list = []
    input_path_list = [os.path.join(root_dir, file) for file in os.listdir(root_dir)]
    with Pool(60) as p:
        result_list = list(
            tqdm(p.imap_unordered(
                partial(process_one_line, chain_tuple=args.chain_tuple), input_path_list),
                total=len(os.listdir(root_dir))))

        
    df = pd.DataFrame(result_list)
    df = df.sort_values(by=['filename']).reset_index(drop=True)
    df.to_csv(f'outputs/foldx/energy.csv')
    

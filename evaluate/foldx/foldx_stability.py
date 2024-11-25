"""
Calculate stability using foldx
"""

# import sys
# sys.path.append("..")
from functools import partial
import os

from PDButils.commonIO import ParsePDB, PrintPDB
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB import Selection
from PDButils.FoldX import FoldXSession
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

def fetch_stability_score(path):
    u = pd.read_csv(path, sep='\t', header=None)
    return u.values[0][1]

def foldx_stability(stru):

    with FoldXSession() as session:
        name = '1.pdb'
        pdb_path = session.path(name)
        PrintPDB(stru, pdb_path)

        session.execute_foldx(pdb_name=name, command_name='Stability')
        fxout_name = f"{name[:-4]}_0_ST.fxout"
        fxout_path = session.path(fxout_name)
        score = fetch_stability_score(fxout_path)

    return score


def stability_one_file(filename, input_dir):
    input_path = os.path.join(input_dir, filename)
    # output_path = os.path.join(output_dir, filename)

    stru = ParsePDB(input_path)
    score = foldx_stability(stru)
    # return
    return {
        'filename': filename.replace('.pdb', ''),
        'stability': score,
    }

if __name__ == '__main__':
    
    use_repaired = True
    if use_repaired:
        input_dir = 'outputs/foldx/repaired'
    else:
        input_dir = 'outputs/pred_pdb/raw'
    output_dir = 'outputs/foldx/'
    os.makedirs(output_dir, exist_ok=True)

    
    files = os.listdir(input_dir)
    with Pool(64) as p:
        results = list(tqdm(p.imap_unordered(
            partial(stability_one_file, input_dir), files), total=len(files)))

    df = pd.DataFrame(results)
    df = df.sort_values(by='filename')
    df = df.reset_index(drop=True)
    if use_repaired:
        df = df.rename(columns={'stability': 'stability_repaired'}, )
    else:
        df = df.rename(columns={'stability': 'stability_raw'}, )
    
    # save or merge
    save_path = os.path.join(output_dir, 'stability.csv')
    if os.path.exists(save_path):
        df_before = pd.read_csv(save_path, index_col=0)
        df = df_before.merge(df, on='filename')
    df.to_csv(save_path)
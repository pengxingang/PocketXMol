import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import argparse

import sys
sys.path.append('.')
from evaluate.utils_eval import combine_receptor_ligand, combine_chains
from evaluate.evaluate_mols import get_dir_from_prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='base_pxm')
    parser.add_argument('--result_root', type=str, default='outputs_test/dock_pepbdb')
    parser.add_argument('--baseline', type=str, default='')
    parser.add_argument('--db', type=str, default='pepbdb')
    args = parser.parse_args()

    result_root = args.result_root
    exp_name = args.exp_name

    # # get generate dir
    gen_path = get_dir_from_prefix(result_root, exp_name)
    print('gen_path:', gen_path)
    df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))

    pdb_path = os.path.join(gen_path, 'SDF')
    protein_dir = f'data/{args.db}/files/proteins'


    # # process
    complex_dir = os.path.join(gen_path, 'complex')
    os.makedirs(complex_dir, exist_ok=True)
    # shuffle df_gen
    df_gen = df_gen.sample(frac=1).reset_index(drop=True)
    print('df_gen.shape=', df_gen.shape)


    for _, line in tqdm(df_gen.iterrows(), total=len(df_gen), desc='combine rec and lig'):

        data_id = line['data_id']
        protein_path = os.path.join(protein_dir, data_id+'_pro.pdb')

        # # combine chains in the protein
        combchain_dir = protein_dir.replace('/proteins', '/proteins_combchain')
        os.makedirs(combchain_dir, exist_ok=True)
        combchain_path = os.path.join(combchain_dir,
                        os.path.basename(protein_path).replace('.pdb', '_combchain.pdb'))
        if not os.path.exists(combchain_path):
            combine_chains(protein_path, save_path=combchain_path)

        # # combine rec and lig
        filename = line['filename']
        # filename = str(line['filename']) + '.pdb'
        # path = line['path']
        file_path = os.path.join(gen_path, 'SDF', filename)
        cpx_name = filename.replace('.pdb', '_cpx.pdb')
        cpx_path = os.path.join(complex_dir, cpx_name)
        if not os.path.exists(cpx_path):
            try:
                combine_receptor_ligand(combchain_path, file_path, cpx_path)
            except Exception as e:
                print(f'Error for {data_id} {filename}:', e)
                continue

    print('Done')
import numpy as np
import os
import argparse

import pandas as pd
from tqdm import tqdm
import torch

import sys
sys.path.append('.')
from evaluate.evaluate_mols import get_dir_from_prefix


def add_cfd_to_csv(gen_dir, sort):
    df_gen = pd.read_csv(os.path.join(gen_dir, 'gen_info.csv'))
    
    df_gen['confidence'] = np.nan
    for index, row in tqdm(df_gen.iterrows(), total=len(df_gen)):
        filename = row['filename']
        pt_path = os.path.join(gen_dir, 'SDF', filename.replace('.pdb', '.pt').replace('.sdf', '.pt'))
        if os.path.exists(pt_path):
            output = torch.load(pt_path)
            df_gen.loc[index, 'confidence'] = torch.mean(output['confidence_pos']).item()
    if sort:
        df_gen = df_gen.sort_values('confidence', ascending=False)
    return df_gen


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_root', type=str, default='outputs_use/PDL1/dockpep')
    parser.add_argument('--exp_name', type=str, default='')
    args = parser.parse_args()

    gen_path = get_dir_from_prefix(args.result_root, args.exp_name)
    print('gen_path:', gen_path)
    
    df_gen = add_cfd_to_csv(gen_path, sort=True)
    df_gen.to_csv(os.path.join(gen_path, 'confidence.csv'), index=False)
    print('Done.')
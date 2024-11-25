import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import re
from multiprocessing import Pool
from functools import partial

import torch
# from rdkit import Chem
# from rdkit.Chem import AllChem

import sys
sys.path.append('.')
from evaluate.evaluate_mols import get_dir_from_prefix
from utils.misc import make_config
from utils.docking_aux_scores import calc_clash, prepare_inputs


def make_ranking_score(gen_path):
    df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))
    df_aux = pd.read_csv(os.path.join(gen_path, 'aux_scores.csv'))
    # merge aux
    df_ranking = df_gen.merge(df_aux, on=['filename', 'data_id'], how='left')
    # tuned scores
    path_tuned = os.path.join(gen_path, 'tuned_cfd.csv')
    if os.path.exists(path_tuned):
        df_cfd = pd.read_csv(path_tuned)
        df_ranking = df_ranking.merge(df_cfd, on=['filename', 'data_id', 'i_repeat'], how='left')
    else:
        print('Tuned confidence scores (tuned_cfd.csv) not found.')
        
    # ranking scores
    df_ranking['self_ranking'] = df_ranking['cfd_traj'] + df_ranking['no_clashes'].astype('int') + df_ranking['stereo'].astype('int')
    if 'tuned_cfd' in df_ranking.columns:
        df_ranking['tuned_ranking'] = df_ranking['tuned_cfd'] + df_ranking['no_clashes'].astype('int') + df_ranking['stereo'].astype('int')
    return df_ranking


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='base_pxm')
    parser.add_argument('--result_root', type=str, default='./outputs_test/dock_posebusters')
    parser.add_argument('--db', type=str, default='poseboff')
    parser.add_argument('--sub_dir', type=str, default='')
    parser.add_argument('--gt_mol', type=str, default='')  # set as 'config' for use
    args = parser.parse_args()

    result_root = args.result_root
    exp_name = args.exp_name
    # # generate dir
    gen_path = get_dir_from_prefix(result_root, exp_name)
    save_path = os.path.join(gen_path, 'aux_scores.csv')
    # if os.path.exists(save_path):
    #     print(f'Already exists {save_path}, skip')
    #     exit()
    
    if args.gt_mol == '':  # for test set
        if args.db != '':
            db = args.db
        else:
            db = re.findall(r'dock_([a-z]+)_', exp_name) 
            if len(db) != 1:
                print('Not found db, use poseboff as default')
                db = 'poseboff'
            else:
                db = db[0]
        file_dir = f'data/{db}/files'
        assert os.path.exists(file_dir), f'file_dir {file_dir} does not exist'
    else:
        yml_path = [f for f in os.listdir(gen_path) if f.endswith('.yml')][0]
        sa_config = make_config(os.path.join(gen_path, yml_path))
        file_dir = dict(sa_config.data)

    
    # # load gen df
    df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))

    sub_dir = 'SDF' if not args.sub_dir else args.sub_dir
    inputs_list = prepare_inputs(df_gen, gen_path, file_dir, sub_dir=sub_dir)
    
    with Pool(64) as p:
        clash_results = list(tqdm(p.imap_unordered(
            partial(calc_clash),
            inputs_list), total=len(inputs_list)))
    
    df_clash = pd.DataFrame(clash_results)
    df_clash.to_csv(save_path, index=False)
    
    
    print('Making ranking score: ranking.csv')
    df_ranking = make_ranking_score(gen_path)
    df_ranking.to_csv(os.path.join(gen_path, 'ranking.csv'), index=False)
    
    print('Done')
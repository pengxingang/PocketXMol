import sys
import os
import json
import argparse
import pandas as pd
import pickle
from tqdm.auto import tqdm
sys.path.append('.')

from utils.reconstruct import *
from utils.misc import *
from utils.scoring_func import *
from utils.evaluation import *
from easydict import EasyDict
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 


def get_mols_dict_from_gen_path(gen_path):
    mols_dir = os.path.join(gen_path, 'SDF')
    # mols_dict = {mol_name[:-4]:Chem.MolFromMolFile(os.path.join(mols_dir, mol_name))
    mols_dict = {mol_name:Chem.MolFromMolFile(os.path.join(mols_dir, mol_name))
                 for mol_name in os.listdir(mols_dir) if (mol_name[-4:] == '.sdf') and
                 ('-all.sdf' not in mol_name) and ('-in.sdf' not in mol_name) and
                 ('-out.sdf' not in mol_name) and ('-raw.sdf' not in mol_name)}
    print(f'Get {len(mols_dict)} from path {gen_path}')
    return mols_dict


def get_dir_from_prefix(result_root, exp_name):
    if '_202' in exp_name:
        gen_path = os.path.join(result_root, exp_name)
        assert os.path.exists(gen_path), f'path {gen_path} not exist'
        return gen_path

    gen_path_prefix = '^' + exp_name + '_202[0-9|_]*$'  # 2030 will cause error. no need to worry, I think. haha
    gen_path = [x for x in os.listdir(result_root) if re.findall(gen_path_prefix, x)]
    if len(gen_path) == 0:
        print(f'experiment {exp_name} is not found in the directory {result_root}')
        exit()
    elif len(gen_path) > 1:
        print(f'Found multiple experiments with name {exp_name} in {result_root}: {gen_path}')
        exit()
    assert len(gen_path) == 1, f'exp {exp_name} is not unique/found in {result_root}: {gen_path}'
    gen_path = os.path.join(result_root, gen_path[0])
    print(f'Get path {gen_path}.')
    return gen_path


def load_mols_from_generated(exp_name, result_root):
    # prepare data path
    all_exp_paths = os.listdir(result_root)
    sdf_dir = [path for path in all_exp_paths
                      if (path.startswith(exp_name) and path.endswith('_SDF'))]
    assert len(sdf_dir) == 1, f'Found more than one or none sdf directory of sampling with prefix `{exp_name}` and suffix `_SDF` in {result_root}: {sdf_dir}'
    sdf_dir = sdf_dir[0]
    
    raise NotImplementedError('The next has undefined value args')
    sdf_dir = os.path.join(args.result_root, sdf_dir)
    metrics_dir = sdf_dir.replace('_SDF', '')
    df_path = os.path.join(metrics_dir, 'mols.csv')
    mol_names = [mol_name for mol_name in os.listdir(sdf_dir) if (mol_name[-4:] == '.sdf') and ('traj' not in mol_name) ]
    mol_ids = np.sort([int(mol_name[:-4]) for mol_name in mol_names])
        
    # load sdfs
    mol_dict_raw = {mol_id:Chem.MolFromMolFile(os.path.join(sdf_dir, '%d.sdf' % mol_id))
                for mol_id in mol_ids}
    mol_dict = {mol_id:mol for mol_id, mol in mol_dict_raw.items() if mol is not None}
    print('Load success:', len(mol_dict), 'failed:', len(mol_dict_raw)-len(mol_dict))

    # load df
    if os.path.exists(df_path):
        df = pd.read_csv(df_path, index_col=0)
    else:
        df = pd.DataFrame(index=list(mol_dict.keys()))
        df.index.name = 'mol_id'
    return mol_dict, df, metrics_dir, df_path


def evaluate_mol_dict(mol_dict, metrics_list, metric_path):
    """
    dict of mols: {mol_id: mol}
    """
    if not os.path.exists(metric_path):
        os.makedirs(metric_path)
    logger = get_logger('eval', metric_path)

    df = pd.DataFrame(index=mol_dict.keys())
    df_path = os.path.join(metric_path, 'mol_metric.csv')
    logger.info(f'Calculating {len(mol_dict)} mol metrics...')
    for metric_name in metrics_list:
        logger.info(f'Computing {metric_name} metrics...')
        if metric_name in ['drug_chem', 'count_prop', 'global_3d', 'frags_counts', 'groups_counts']:
            parallel =True
            results_list = get_metric(mol_dict.values(), metric_name, parallel=parallel)
            if list(results_list[0].keys())[0] not in df.columns:
                df = pd.concat([df, pd.DataFrame(results_list, index=mol_dict.keys())], axis=1)
            else:
                df.loc[mol_dict.keys(), results_list[0].keys()] = pd.DataFrame(
                    results_list, index=mol_dict.keys())
            df.to_csv(df_path)
        elif metric_name == 'local_3d':
            local3d = Local3D()
            local3d.get_predefined()
            logger.info(f'Computing local 3d - bond lengths metric...')
            lengths = local3d.calc_frequent(mol_dict.values(), type_='length', parallel=False)
            logger.info(f'Computing local 3d - bond angles metric...')
            angles = local3d.calc_frequent(mol_dict.values(), type_='angle', parallel=False)
            logger.info(f'Computing local 3d - dihedral angles metric...')
            dihedral = local3d.calc_frequent(mol_dict.values(), type_='dihedral', parallel=False)
            local3d = {'lengths': lengths, 'angles': angles, 'dihedral': dihedral}
            with open(os.path.join(metric_path, 'local3d.json'), 'w') as f:
                json.dump(local3d, f, indent=2)
                # f.write(pickle.dumps(local3d))
        elif metric_name == 'validity':
            validity = calculate_validity(
                output_dir=metric_path,
                # is_edm=('e3_diffusion_for_molecules' in metric_path),
            )
            with open(os.path.join(metric_path, 'validity.json'), 'w') as f:
                json.dump(validity, f, indent=2)
                # f.write(pickle.dumps(validity))
            logger.info(f'Validity : {validity}')
        elif metric_name == 'similarity':
            sim = SimilarityAnalysis()
            # uniqueness = sim.get_novelty_and_uniqueness(mol_dict.values())
            similarity = sim.get_diversity(mol_dict.values())
            # uniqueness['diversity'] = diversity
            # sim_with_val = sim.get_sim_with_val(mol_dict.values())
            # uniqueness['sim_with_val'] = sim_with_val
            save_path = os.path.join(metric_path, 'similarity.json')
            with open(save_path, 'w') as f:
                json.dump(similarity, f, indent=2)
                # f.write(pickle.dumps(uniqueness))
            logger.info(f'Similarity : {similarity}')
        elif metric_name == 'ring_type':
            ring_analyzer = RingAnalyzer()
            # cnt of ring type (common in val set)
            # cnt_ring_type = ring_analyzer.get_count_ring(mol_dict.values())
            # if list(cnt_ring_type.keys())[0] not in df.columns:
            #     df = pd.concat([df, pd.DataFrame(cnt_ring_type, index=mol_dict.keys())], axis=1)
            # else:
            #     df.loc[mol_dict.keys(), cnt_ring_type.keys()] = pd.DataFrame(
            #         cnt_ring_type, index=mol_dict.keys())
            # df.to_csv(df_path)
            # top n freq ring type
            freq_dict = ring_analyzer.get_freq_rings(mol_dict.values())
            with open(os.path.join(metric_path, 'freq_ring_type.json'), 'w') as f:
                json.dump(freq_dict, f, indent=2)
                # f.write(pickle.dumps(freq_dict))
            
            
    logger.info(f'Saving metrics to {metric_path}')
    logger.info(f'Done calculating mol metrics.')
    
    
def combine_gt_gen_metrics(gen_path, gt_metric_path, metrics_list):
    
    # for metric_name in metrics_list:
    #     if metric_name in ['drug_chem', 'count_prop', 'global_3d', 'frags_counts', 'groups_counts']:
    if os.path.exists(os.path.join(gen_path, 'vina.csv')):
        df_vina = pd.read_csv(os.path.join(gen_path, 'vina.csv'), index_col=0)
    else:
        df_vina = None
    df_gen = pd.read_csv(os.path.join(gen_path, 'mol_metric.csv'), index_col=0)
    df_gt = pd.read_csv(os.path.join(gt_metric_path, 'mol_metric.csv'), index_col=0)
    df_mean = df_gen.mean(axis=0)
    df_median = df_gen.median(axis=0)
    # df_jsd = df_gen.apply(lambda x: get_jsd(x, df_gt[x.name]), axis=0)
    df_ref_mean = df_gt.mean(axis=0)
    df_ref_median = df_gt.median(axis=0)
    df_combine = pd.concat([df_mean, df_median, df_ref_mean, df_ref_median], axis=1)
    df_combine.columns = ['mean', 'median', 'ref_mean', 'ref_median']
    if df_vina is not None:
        for col in df_vina.columns:
            df_combine.loc[col, 'mean'] = df_vina[col].mean()
            df_combine.loc[col, 'median'] = df_vina[col].median()

    if 'validity':
        validity = json.load(open(os.path.join(gen_path, 'validity.json')))
        for key, value in validity.items():
            df_combine.loc[key, 'mean'] = value
        
    if 'local_3d' in metrics_list:
        # metric_base = 'lengths'
        local3d_gen = json.load(open(os.path.join(gen_path, 'local3d.json')))
        local3d_gt = json.load(open(os.path.join(gt_metric_path, 'local3d.json')))
        for metric_base in ['lengths', 'angles', 'dihedral']:
            metrics_list = list(local3d_gt[metric_base].keys())
            print(metric_base, ':', metrics_list, '\n')

            width = 0.02 if metric_base == 'lengths' else 5
            metric_dict = {}
            for metric in metrics_list:
                # set width and discrete
                # get jsd
                values_list = [local3d_gt[metric_base][metric], local3d_gen[metric_base][metric]]
                kld_list, bins, hist_list = compare_with_ref(values_list, width=width,
                                                            discrete=False)
                metric_dict[metric] = kld_list[1]
            mean_jsd = np.nanmean(list(metric_dict.values()))
            df_combine.loc[f'jsd_{metric_base}', 'mean'] = mean_jsd

    # save 
    df_combine.to_csv(os.path.join(gen_path, 'comb_metric.csv'))
    return df_combine


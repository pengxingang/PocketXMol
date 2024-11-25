import sys
import os
import argparse
import pandas as pd
import pickle
import re
import json
import numpy as np
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
sys.path.append('.')
from multiprocessing import Pool
from copy import deepcopy
# from utils.reconstruct import *
# from utils.graph import rdmol_to_attr_graph, is_same_mols
from utils.scoring_func import get_diversity
from utils.evaluation import get_drug_chem
# from utils.dataset import TestTaskDataset
from process.process_torsional_info import get_mol_from_data
from evaluate.evaluate_mols import get_dir_from_prefix
#                         get_dir_from_prefix, combine_gt_gen_metrics
from evaluate.from_delinker import calc_SC_RDKit_score




def load_linking_gt(db):
    assert db in ['geom', 'moad', 'protacdb'], 'db must be geom or moad, not {}'.format(db)
    test_path = f'data/test/dfs/linking_{db}.csv'
    df_test = pd.read_csv(test_path)
    if 'has_mapping' in df_test:
        df_test = df_test[df_test['has_mapping']]
    print(f'Db: {db} with {len(df_test)} samples.')
    
    if db == 'moad':
        db_path = f'data/{db}/files/mols'
    else:
        db_path = f'data/{db}/mols'
    
    data_list = []
    for _, line in df_test.iterrows():
        data_id = line['data_id']
        sdf_name = data_id + ('.sdf' if 'files' not in db_path else '_mol.sdf')
        data_dict = {
            'sep_id': line['sep_id'],
            'linkers': sum([list(item) for item in eval(line['linkers'])], []),
            'fragments': sum([list(item) for item in eval(line['frags'])], []),
            'anchors': sum([list(item) for item in eval(line['anchors'])], []),
            'gt_mol': Chem.MolFromMolFile(os.path.join(db_path, sdf_name))
        }
        data_list.append(data_dict)

    return data_list


def load_linking_gen(data_list, gen_dir, use_repeats):
    
    sep_id_path = os.path.join(gen_dir, 'sep_id_list.csv')
    if os.path.exists(sep_id_path):
        df_sep_ids = pd.read_csv(sep_id_path)
        sep_id_list = df_sep_ids['sep_id'].values
    else:
        sep_id_list = [sample['sep_id'] for sample in data_list]
    
    df_gen = pd.read_csv(os.path.join(gen_dir, 'gen_info.csv'))
    if 'sep_id' not in df_gen:
        df_gen['sep_id'] = df_gen['key'].apply(lambda x: x.split(';')[-1].replace('linking/', ''))
    if use_repeats > 0:
        df_gen = df_gen[df_gen['i_repeat'] < use_repeats]
        
    new_data_list = []
    for sample in tqdm(data_list):
        sep_id = sample['sep_id']
        if sep_id not in sep_id_list:
            continue  # for delinker/3dlinker in moad. large mol (>48atoms) and minor element not generated
        df_this = df_gen[df_gen['sep_id'] == sep_id]
        gen_mols = []
        for filename in df_this['filename'].values:
            mol_path = os.path.join(gen_dir, 'SDF', filename)
            gen_mols.append(mol_path)
        sample['gen_mols'] = gen_mols
        new_data_list.append(deepcopy(sample))
    print('Updated data list length', len(new_data_list))
    print('Total gen mols', sum(len(sample['gen_mols']) for sample in new_data_list))
    
    return new_data_list
    

# def get_gt_mol(data_id, db):
#     return get_mol_from_data({'data_id': data_id, 'db': db}, root_dir='data')

# def load_gt_mols(sep_id_list, sep_path, db):
#     df_linking = pd.read_csv(sep_path)
#     df_linking.set_index('sep_id', inplace=True)
#     assert df_linking.index.is_unique, 'sep_id is not unique'

#     # load mols
#     gt_dict = {}
#     linking_atoms_dict = {}
#     for sep_id in sep_id_list:
#         data_id = sep_id.split('/')[0]
#         mol = get_gt_mol(data_id, db)
#         gt_dict[sep_id] = mol
#         linking_atoms = sum(eval(df_linking.loc[sep_id, 'linkers']), [])
#         linking_atoms_dict[sep_id] = linking_atoms
    
#     return gt_dict, linking_atoms_dict

def neutralizeRadicals(mol):
    mol = deepcopy(mol)
    for a in mol.GetAtoms():
        if a.GetNumRadicalElectrons()>0 or a.GetFormalCharge()>0:
            a.SetNumRadicalElectrons(0)
            a.SetFormalCharge(0)
    return mol

def get_metric_linking_one_input(inputs):
    mol_list = inputs['gen_mols']
    gt_mol = inputs['gt_mol']
    size_gt = gt_mol.GetNumAtoms()
    gt_finger = Chem.RDKFingerprint(gt_mol)

    frag_atoms = inputs['fragments']
    frag_smi = Chem.MolFragmentToSmiles(gt_mol, frag_atoms, isomericSmiles=False)
    frag = Chem.MolFromSmiles(frag_smi)
    if frag is None:
        frag = Chem.MolFromSmarts(frag_smi)
    
    if len(mol_list) == 0:
        return {
            'sep_id': inputs['sep_id'],
            'n_mols': 0,
        }, {}
    
    if isinstance(mol_list[0], str):
        mol_list = [Chem.MolFromMolFile(mol_fn) for mol_fn in mol_list]
        
    # validity: rdkit-readable, conncted, frag is in the mol
    n_mols = len(mol_list)
    valid_dict = {}
    valid_mol_list = []
    for mol_path, mol in zip(inputs['gen_mols'], mol_list):
        filename = os.path.basename(mol_path)
        if mol is None:
            valid_dict[filename] = False
        elif '.' in Chem.MolToSmiles(mol):
            valid_dict[filename] = False
        # elif len(mol.GetSubstructMatches(frag)) == 0:
        #     valid_dict[filename] = False
        elif mol.GetNumAtoms() <= len(frag_atoms):
            valid_dict[filename] = False
        else:
            valid_dict[filename] = True
            valid_mol_list.append(mol)
    n_valid = len(valid_mol_list)
    
    # uniqueness
    diversity_dict = get_diversity(valid_mol_list)  # dict of ['diversity', 'uniqueness']
    
    # qed, sa
    chem_list = [get_drug_chem(mol) for mol in valid_mol_list]
    qed_list = [c['qed'] for c in chem_list]
    sa_list = [c['sa'] for c in chem_list]

    # recover
    is_recover = False
    rmsd_list = []
    # gt_graph = rdmol_to_attr_graph(gt_mol)
    # gt_smiles = Chem.MolToSmiles(gt_mol, isomericSmiles=False)
    for mol in valid_mol_list:
        # gen_graph = rdmol_to_attr_graph(gen_graph)
        # recover_this = is_same_mols(gt_graph, gen_graph)
        # is_recover = is_recover | recover_this
        fg = Chem.RDKFingerprint(mol)
        similarity = Chem.DataStructs.FingerprintSimilarity(gt_finger, fg)
        # smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        if similarity == 1:
            is_recover = True
            # calculate rdkit rmsd
            try:
                rmsd = Chem.rdMolAlign.GetBestRMS(mol, gt_mol)
                rmsd_list.append(rmsd)
            except RuntimeError as e:
                if 'No sub-structure match found between the reference and probe mol' in e.args[0]:
                    try:
                        rmsd = Chem.rdMolAlign.GetBestRMS(neutralizeRadicals(mol), gt_mol)
                        rmsd_list.append(rmsd)
                    except RuntimeError as e:
                        if 'No sub-structure match found between the reference and probe mol' in e.args[0]:
                            pass
                        else:
                            raise e
                else:
                    raise e
        # else: #++
        #     mol = neutralizeRadicals(mol)
        #     smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        #     if smiles == gt_smiles:
        #         is_recover = True
        #         # calculate rdkit rmsd
        #         rmsd = Chem.rdMolAlign.GetBestRMS(mol, gt_mol)
        #         rmsd_list.append(rmsd)
    
    # SCRDkit
    sc_list = []
    for mol in valid_mol_list:
        try:
            _ = Chem.rdMolAlign.GetO3A(mol, gt_mol).Align()
            sc_score = calc_SC_RDKit_score(mol, gt_mol)
            sc_list.append(sc_score)
        except:
            pass
    sc_list = np.array(sc_list)
    
    return {
        'sep_id': inputs['sep_id'],
        'n_mols': n_mols,
        'n_valid': n_valid,
        'uniqueness': diversity_dict['uniqueness'],
        'diversity': diversity_dict['diversity'],
        'qed_mean': np.mean(qed_list),
        'sa_mean': np.mean(sa_list),
        'is_recover': is_recover,
        'rmsd_mean': np.mean(rmsd_list) if len(rmsd_list) else np.nan,
        'rmsd_min': np.min(rmsd_list) if len(rmsd_list) else np.nan,
        'sc_70': (sc_list > 0.7).mean(),
        'sc_80': (sc_list > 0.8).mean(),
        'sc_90': (sc_list > 0.9).mean(),
        'sc_mean': np.mean(sc_list),
        'sc_max': np.max(sc_list) if len(sc_list)>0 else np.nan,
    }, valid_dict


def get_metrics_linking(data_list, n_workers):
    # n_workers = 16
    if n_workers == 1:
        pass
    else:
        results_list = []
        valid_dict = {}
        with Pool(n_workers) as p:
            for result in tqdm(p.imap_unordered(get_metric_linking_one_input, data_list), total=len(data_list)):
                results_list.append(result[0])
                valid_dict.update(**result[1])
    df_metrics = pd.DataFrame(results_list)
    df_valid = pd.DataFrame.from_dict(valid_dict, orient='index', columns=['valid']).reset_index(names='filename')

    return df_metrics, df_valid

def evaluate_linking(gen_path, db, use_repeats, n_workers):
    
    data_list = load_linking_gt(db)
    data_list = load_linking_gen(data_list, gen_path, use_repeats)
    
    
    df_metrics = get_metrics_linking(data_list, n_workers)
    return df_metrics



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_root', type=str, default='outputs_paper/linking_moad/fixed')
    parser.add_argument('--exp_name', type=str, default='msel_fixed_unknown')
    parser.add_argument('--db', type=str, default='moad')
    parser.add_argument('--baseline', type=str, default='delinker_known_anchors')
    parser.add_argument('--use_repeats', type=int, default=-1)
    parser.add_argument('--n_workers', type=int, default=16)
    args = parser.parse_args()
    
    db = args.db
    if args.baseline == '':
        exp_name = args.exp_name  #.replace('/', '_')
        gen_path = get_dir_from_prefix(args.result_root, exp_name)
        assert db in gen_path, f'db name {db} not in gen_path: {gen_path}'
        if db == 'protacdb' and 'lv' in args.exp_name:
            use_repeats = 30
        else:
            use_repeats = args.use_repeats
    else:
        baseline = args.baseline
        gen_path = os.path.join('baselines/linking', db, baseline)
        use_repeats = args.use_repeats
    print(f'Evaluate linking: {gen_path}')
    df_metrics, df_valid = evaluate_linking(gen_path, db, use_repeats, args.n_workers)

    df_metrics.to_csv(os.path.join(gen_path, 'metrics.csv'), index=False)
    df_valid.to_csv(os.path.join(gen_path, 'valid.csv'), index=False)
import sys
sys.path.append(".")
import os
from functools import partial
from multiprocessing import Pool
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd

# from Bio.PDB.StructureBuilder import StructureBuilder
# from Bio.PDB import Selection
from tqdm import tqdm

from evaluate.foldx.FoldX import FoldXSession
from evaluate.rosetta import ParsePDB, PrintPDB
from evaluate.evaluate_mols import get_dir_from_prefix

# from evaluate.energy.foldx_repair import repair_one_file
# from evaluate.energy.foldx_stability import stability_one_file
# from evaluate.energy.foldx_energy import process_one_line


def calc_foldx_score(complex_dir, output_dir, num_workers, chain_tuple, repair=True):
    all_files = os.listdir(complex_dir)
    # shuffle
    all_files = np.random.permutation(all_files)
    
    if repair:
        # repair
        repair_dir = os.path.join(output_dir, 'repaired')
        os.makedirs(repair_dir, exist_ok=True)
        with Pool(num_workers) as p:
            results = list(tqdm(
                p.imap_unordered(
                    partial(repair_one_file, input_dir=complex_dir, output_dir=repair_dir),
                    all_files
                ),
                total=len(all_files), desc='Foldx repairing...')
            )
    else:
        repair_dir = complex_dir
        
    # stability
    stability_dir = os.path.join(output_dir, 'stability')
    os.makedirs(stability_dir, exist_ok=True)
    with Pool(num_workers) as p:
        list(tqdm(
            p.imap_unordered(
                partial(stability_one_file, input_dir=repair_dir, output_dir=stability_dir),
                all_files
            ),
            total=len(all_files), desc='Calculating foldx stability...')
        )
    # df = pd.DataFrame(results).sort_values('filename').reset_index(drop=True)
    
    # energy
    energy_dir = os.path.join(output_dir, 'energy')
    os.makedirs(energy_dir, exist_ok=True)
    # chain_tuple = 'R,L'
    with Pool(num_workers) as p:
        list(
            tqdm(p.imap_unordered(
                partial(energy_one_file, input_dir=repair_dir, output_dir=energy_dir, chain_tuple=chain_tuple),
                all_files
            ),
            total=len(all_files), desc='Calculating energy...')
        )
    # df_energy = pd.DataFrame(result_list).sort_values('filename').reset_index(drop=True)
    # df = df.merge(df_energy, on='filename')
    
    # df.to_csv(os.path.join(output_dir, 'foldx.csv'))
    # return df
    
    
# ----- repair -----


def foldx_repair(stru):

    with FoldXSession() as session:
        name = '1.pdb'
        pdb_path = session.path(name)
        PrintPDB(stru, pdb_path)

        # session.execute_foldx(pdb_name=name, command_name='RepairPDB', options=['--pdbHydrogens=True'])
        session.execute_foldx(pdb_name=name, command_name='RepairPDB')

        output_name = '1_Repair.pdb'
        repaired_stru = ParsePDB(session.path(output_name))

    return repaired_stru


def repair_one_file(name, input_dir, output_dir):
    try:
        input_path = os.path.join(input_dir, name)
        output_path = os.path.join(output_dir, name)
        if os.path.exists(output_path):
            return
        stru = ParsePDB(input_path)
        stru = foldx_repair(stru)
        PrintPDB(stru, output_path)
    except:
        print(f'Error in repairing for {name}')
        return


# ----- stability -----

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


def stability_one_file(filename, input_dir, output_dir=None):
    try:
        if output_dir is not None:
            save_path = os.path.join(output_dir, filename.replace('.pdb', '.pkl'))
            if os.path.exists(save_path):
                return 
            
        input_path = os.path.join(input_dir, filename)
        if not os.path.exists(input_path):
            print('File not found for stability:', input_path)
            return

        stru = ParsePDB(input_path)
        score = foldx_stability(stru)
        # return
        result = {
            'filename': filename.replace('.pdb', ''),
            'stability': score,
        }
        if output_dir is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(result, f)
    except Exception as e:
        print('ERROR: ', filename, e)
    
    
# ----- energy -----
def fetch_binding_affinity(path):
    with open(path, 'r') as f:
        u = f.readlines()
    line = u[-1].split("\t")
    result = {
        'clash_rec': float(line[-5]),
        'clash_lig': float(line[-4]),
        'energy': float(line[-3]),
        'stable_rec': float(line[-2]),
        'stable_lig': float(line[-1]),
    }
    return result

def energy_one_file(filename, input_dir, output_dir, chain_tuple):
    try:
        save_path = os.path.join(output_dir, filename.replace('.pdb', '.pkl'))
        if os.path.exists(save_path):
            return

        input_path = os.path.join(input_dir, filename)
        if not os.path.exists(input_path):
            print('File not found for energy:', input_path)
            return
        
        with FoldXSession() as session:
            session.preprocess_data(input_dir, filename)
            session.execute_foldx(pdb_name=filename,
                command_name='AnalyseComplex', options=[f'--analyseComplexChains={chain_tuple}'])
            name = filename.replace('.pdb', '')
            fxout_path = session.path(f'Summary_{name}_AC.fxout')

            assert(os.path.exists(fxout_path)), 'fxout file not found'
            result = fetch_binding_affinity(fxout_path)
            result = {'filename': filename.replace('.pdb', ''), **result}

        # return result
        # save
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)
    except Exception as e:
        print('ERROR: ', filename, e)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='base_pxm')
    parser.add_argument('--result_root', type=str, default='outputs_test/dock_pepbdb')
    
    parser.add_argument('--complex_dir', type=str, default='')
    parser.add_argument('--foldx_dir', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=126)
    parser.add_argument('--chain_tuple', type=str, default='R,L')
    # parser.add_argument('--save_pdb', type=bool, default=False)
    args = parser.parse_args()
    
    if args.exp_name and args.result_root:
        gen_path = get_dir_from_prefix(args.result_root, args.exp_name)
        print('gen_path:', gen_path)
        complex_dir = os.path.join(gen_path, 'complex')
        foldx_dir = os.path.join(gen_path, 'foldx')
    else:
        complex_dir = args.complex_dir
        foldx_dir = args.foldx_dir
    
    calc_foldx_score(complex_dir, foldx_dir, num_workers=args.num_workers, chain_tuple=args.chain_tuple)
    
    # make df
    df_foldx = []
    for file in os.listdir(os.path.join(foldx_dir, 'energy')):
        if file.endswith('.pkl'):
            path = os.path.join(foldx_dir, 'energy', file)
            with open(path, 'rb') as f:
                score = pickle.load(f)
            df_foldx.append(score)
    df_foldx = pd.DataFrame(df_foldx)
    df_foldx.to_csv(os.path.join(foldx_dir, 'foldx.csv'), index=False)
    
    print('Done.')
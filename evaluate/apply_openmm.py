# Use moldiff env to run this script (openmm, openff, pymol)

import argparse
import os
import io
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import re
from multiprocessing import Pool
from functools import partial 

from pymol import cmd
import openmm
from openmm import unit
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator, GAFFTemplateGenerator
from openff.toolkit.utils.rdkit_wrapper import RDKitToolkitWrapper
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D



def get_dir_from_prefix(result_root, exp_name):
    if '_202' in exp_name:
        gen_path = os.path.join(result_root, exp_name)
        assert os.path.exists(gen_path), f'path {gen_path} not exist'
        return gen_path

    gen_path_prefix = '^' + exp_name + '_202[0-9|_]*$'
    gen_path = [x for x in os.listdir(result_root) if re.findall(gen_path_prefix, x)]
    assert len(gen_path) == 1, f'exp {exp_name} is not unique/found in {result_root}: {gen_path}'
    gen_path = os.path.join(result_root, gen_path[0])
    print(f'Get path {gen_path}.')
    return gen_path


def get_openmm_robust(inputs, save_dir):
    try:
        get_openmm_one_file(inputs, save_dir)
    except Exception as e:
        print('Error:', e)
        return

def get_openmm_one_file(inputs, save_dir, not_moving_protein=True):
    protein_path = inputs['protein_path']
    ligand_path = inputs['ligand_path']
    filename = inputs['filename']
    
    # # get data path
    pdb_path = protein_path
    mol_path = ligand_path
    assert os.path.exists(pdb_path), f'pdb_path not exist: {pdb_path}'
    assert os.path.exists(mol_path), f'mol_path not exist: {mol_path}'
    save_cpx_path = os.path.join(save_dir, filename + '_omm_cpx.pdb')
    save_pdb_path = os.path.join(save_dir, filename + '_omm_pro.pdb')
    save_mol_path = os.path.join(save_dir, filename + '.sdf')
    if os.path.exists(save_mol_path):
        return

    # # molecules 
    tool = RDKitToolkitWrapper()
    rdmol = Chem.MolFromMolFile(mol_path, removeHs=False)
    # smiles_old = Chem.MolToSmiles(rdmol, False)
    rdmol = Chem.AddHs(rdmol, addCoords=True)
    ligand = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)
    tool.assign_partial_charges(ligand, 'mmff94')
    ligand_positions = ligand.conformers[0]
    ligand_topology = ligand.to_topology()

    # # force field
    omm_forcefield = openmm.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml') #! amber14-all.xml contains ff14sb for protein
    gaff = SMIRNOFFTemplateGenerator(molecules=[ligand])  #! default openff-2.1.0 (sage)
    omm_forcefield.registerTemplateGenerator(gaff.generator)

    # # load the structures
    pdb = openmm.app.PDBFile(pdb_path)
    modeller = openmm.app.Modeller(pdb.topology, pdb.positions)
    modeller.add(ligand_topology.to_openmm(), ligand_positions)

    # # create system
    system = omm_forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=openmm.app.CutoffNonPeriodic ,
        nonbondedCutoff=10 * unit.angstrom,
        constraints=openmm.app.HBonds,
        #     rigidWater=True,
    )
    if not_moving_protein:
        for i in range(len(pdb.positions)):
            system.setParticleMass(i, 0)
    topology = modeller.getTopology()
    positions = modeller.getPositions()

    # # build simulatior
    platform = openmm.Platform.getPlatformByName("CUDA")
    integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1 / unit.picosecond, 0.002*unit.picoseconds)
    simulation = openmm.app.Simulation(topology, system, integrator,platform)
    simulation.context.setPositions(positions)

    # # minimization 
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    einit = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    # simulation.minimizeEnergy()  # maxIterations=1000, )
    simulation.minimizeEnergy(tolerance=0.01 * unit.kilojoules_per_mole, maxIterations=100000)
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    efinal = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    print(f'Energies before and after are {einit} and {efinal}.')
    
    # # save energies
    energy = {
        'filename': filename,
        'init_energy': einit,
        'final_energy': efinal
    }
    with open(os.path.join(save_dir, f'{filename}_energy.pkl'), 'wb') as file:
        pickle.dump(energy, file)

    # # save pdb
    with io.StringIO() as f:
        openmm.app.PDBFile.writeFile(
            simulation.topology,
            state.getPositions(),
            f,
            # open(save_path, "w"),
            keepIds=True)
        pdb_block = f.getvalue()
    with open(save_cpx_path, 'w+') as file:
        file.write(pdb_block)


    # # complex pdb split with pymol
    cmd.delete('all')
    cmd.read_pdbstr(pdb_block, 'complex')
    # cmd.load(save_path, 'complex')
    cmd.extract('mol', 'complex and chain 1')
    cmd.save(save_pdb_path, 'complex')
    # cmd.save(save_mol_path, 'mol')

    n_atoms = rdmol.GetNumAtoms()
    mol_pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstroms)[-n_atoms:]
    conf = rdmol.GetConformer()
    for i in range(n_atoms):
        x, y, z = mol_pos[i]
        conf.SetAtomPosition(i, Point3D(x,y,z))
    Chem.MolToMolFile(rdmol, save_mol_path)



def prepare_inputs(df_gen, gen_dir, protein_dir):
    data_list = []
    for _, line in df_gen.iterrows():
        data_id = line['data_id']
        filename = line['filename'].replace('.sdf', '')
        
        protein_path = os.path.join(protein_dir, data_id+'_pro_fixed.pdb')
        ligand_path = os.path.join(gen_dir, 'SDF', filename+'.sdf')
        data_list.append({
            'filename': filename,
            'protein_path': os.path.abspath(protein_path),
            'ligand_path': os.path.abspath(ligand_path),
        })
    return data_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--result_root', type=str, default='outputs_test')
    parser.add_argument('--db', type=str, default='')
    args = parser.parse_args()

    result_root = args.result_root
    exp_name = args.exp_name
    
    # db = exp_name.split('_')[-2]
    if args.db != '':
        db = args.db
    else:
        db = re.findall(r'dock_([a-z]+)_', exp_name)
        if len(db) != 1:
            print('Not given db, use posboff as default')
            db = 'poseboff'
        else:
            db = db[0]
        assert db in ['pbdock', 'poseb', 'poseboff'], f'Unknown db {db} for docking eval'
    
    
    # # get inputs
    gen_path = get_dir_from_prefix(result_root, exp_name)
    df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))
    protein_dir = f'data/{db}/files/proteins_fixed'
    
    save_dir = os.path.abspath(os.path.join(gen_path, 'openmm'))
    os.makedirs(save_dir, exist_ok=True)
    
    df_gen = df_gen.sample(frac=1,)
    inputs_list = prepare_inputs(df_gen, gen_path, protein_dir)
    
    with Pool(8) as f:
        list(tqdm(f.imap_unordered(partial(get_openmm_robust, save_dir=save_dir),
                    inputs_list), total=len(inputs_list)))

    print('Done')




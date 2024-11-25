import os
import subprocess
from Bio.PDB import PDBParser, PDBIO, Structure, Chain
import tempfile
from copy import deepcopy

PATH_DOCKQ =  'XXX'  # dockq path


def combine_chains(pdb_file_path, combined_chain_id='R', bias=200, save_path=None):
    # Create a structure object from the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file_path)[0]
    
    # combine all residue in one chain
    combined_chain = Chain.Chain(combined_chain_id)
    for i_chain, chain in enumerate(structure):
        start_index_chain = next(chain.get_residues()).id[1]
        bias_insertion = 0
        for residue in chain:
            residue.detach_parent()
            if residue.id[2] != ' ':
                bias_insertion += 1
            if i_chain == 0:
                index = residue.id[1] - start_index_chain + 1 + bias_insertion
            else:
                index = residue.id[1] - start_index_chain + 1 + bias_insertion + (bias + last_index)
            residue.id = (residue.id[0], index, ' ')
            combined_chain.add(residue)
        last_index = residue.id[1]
            
    # create a new structure
    combined_structure = combined_chain
    # Structure.Structure('combined')
    # combined_structure.add(combined_chain)

    # save
    if save_path is not None:
        io = PDBIO()
        io.set_structure(combined_structure)
        io.save(save_path)

    return combined_structure

def combine_receptor_ligand(protein_path, ligand_path, output_path,
                           rec_chain_id='R'):
    parser = PDBParser()
    protein = parser.get_structure('protein', protein_path)[0]
    receptor_chain = list(protein.get_chains())
    assert len(receptor_chain) == 1, 'protein should have only one chain'
    assert receptor_chain[0].id == rec_chain_id, 'protein chain id does not match'

    ligand = parser.get_structure('ligand', ligand_path)[0]
    ligand_chain = list(ligand.get_chains())
    if len(ligand_chain) != 1:
        print('ligand have more than one chain')
    ligand_chain = ligand_chain[0]
    
    # rename chain and combine
    ligand_chain.id = 'L'
    protein.add(ligand_chain)
    
    # save 
    io = PDBIO()
    io.set_structure(protein)
    io.save(output_path)
    return protein


def get_dockq(lig_pred_path, lig_gt_path, rec_path):

    # make tempraory complex paths
    tmp = tempfile.NamedTemporaryFile()
    tmp_pred = tmp.name + '_pred.pdb'
    tmp_gt = tmp.name + '_gt.pdb'
    
    
    # make pred complex
    parser = PDBParser()
    for lig_path, tmp_path in zip([lig_pred_path, lig_gt_path], [tmp_pred, tmp_gt]):
        rec = parser.get_structure('rec', rec_path)[0]
        lig = parser.get_structure('lig', lig_path)[0]
        lig_chain = list(lig.get_chains())
        assert len(lig_chain) == 1
        lig_chain = lig_chain[0]
        
        lig_id = lig_chain.id
        rec.add(lig_chain)
        io = PDBIO()
        io.set_structure(rec)
        io.save(tmp_path)

    
    # run dockq
    cmd = ['python', f'{PATH_DOCKQ}/DockQ.py', tmp_pred, tmp_gt]
    cmd += ['-model_chain1', lig_id, '-no_needle']
    output = subprocess.run(cmd, capture_output=True, text=True)
    if output.returncode != 0:
        raise ValueError('DockQ errored:' + output.stderr)
    
    results = output.stdout.split('\n')[-4:-1]
    if 'DockQ' not in results[-1]:
        raise ValueError('DockQ failed: ' + output.stdout)
    
    irmsd = results[0].split()[1]
    lrmsd = results[1].split()[1]
    dockq = results[2].split()[1]
    print(output.stdout)

    # remove temporary files
    os.remove(tmp_pred)
    os.remove(tmp_gt)
    
    return {
        'irmsd': float(irmsd),
        'lrmsd': float(lrmsd),
        'dockq': float(dockq),
    }

if __name__ == '__main__':
    lig_pred_path = 'outputs/dockpep1008/M5_apep_no_domain_noise_dock_pepbdb_free_20231008_140427/PDB/0.pdb'
    lig_gt_path = 'data/pepbdb/files/mols/pepbdb_6peu_M_mol.pdb'
    rec_path = 'data/pepbdb/files/proteins/pepbdb_6peu_M_pro.pdb'

    dockq = get_dockq(lig_pred_path, lig_gt_path, rec_path)
    print(dockq)
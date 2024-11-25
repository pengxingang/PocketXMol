import numpy as np
try:
    import Bio.PDB
    from pdbfixer import PDBFixer
    from rdkit import Chem
    from openmm.app.pdbfile import PDBFile
except:
    from pymol import cmd

def align_two_pdbs_biopython(sample_path, refer_path):
    raise NotImplementedError
    # Select what residues numbers you wish to align
    # and put them in a list
    # start_id = 1
    # end_id   = 70
    # atoms_to_be_aligned = range(start_id, end_id + 1)

    # Start the parser
    pdb_parser = Bio.PDB.PDBParser(QUIET = True)

    # Get the structures
    ref_structure = pdb_parser.get_structure("reference", refer_path)
    sample_structure = pdb_parser.get_structure("sample", sample_path)

    # Use the first model in the pdb-files for alignment
    # Change the number 0 if you want to align to another structure
    ref_model = ref_structure[0]
    sample_model = sample_structure[0]

    # Make a list of the atoms (in the structures) you wish to align.
    # In this case we use CA atoms whose index is in the specified range
    ref_atoms = []
    ref_index = []
    sample_atoms = []
    sample_index = []

    # Iterate of all chains in the model in order to find all residues
    for ref_chain in ref_model:
    # Iterate of all residues in each model in order to find proper atoms
        for ref_res in ref_chain:
            # Check if residue number ( .get_id() ) is in the list
            # if ref_res.get_id()[1] in atoms_to_be_aligned:
            # Append CA atom to list
            if ref_res.resname != 'HOH':
                ref_atoms.append(ref_res['CA'])
                ref_index.append(ref_res.get_id()[1])

    # Do the same for the sample structure
    for sample_chain in sample_model:
        for sample_res in sample_chain:
            # if sample_res.get_id()[1] in atoms_to_be_aligned:
            if sample_res.resname != 'HOH':
                sample_atoms.append(sample_res['CA'])
                sample_index.append(sample_res.get_id()[1])

    # Find common residue atoms
    common_index = np.intersect1d(sample_index, ref_index)
    sample_atoms = [sample_atoms[i] for i, idx in enumerate(sample_index) if idx in common_index]
    ref_atoms = [ref_atoms[i] for i, idx in enumerate(ref_index) if idx in common_index]


    # Now we initiate the superimposer:
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(ref_atoms, sample_atoms)
    super_imposer.apply(sample_model.get_atoms())

    # Print RMSD:
    print(super_imposer.rms)

    # Save the aligned version of 1UBQ.pdb
    sample_path = sample_path[:-4] + '_aligned.pdb'
    io = Bio.PDB.PDBIO()
    io.set_structure(sample_structure) 
    io.save(sample_path)
    return sample_path


def align_two_pdbs(refer_path, sample_path, save_path, cycles=5):
    cmd.delete('all')
    cmd.load(refer_path, 'refer')
    cmd.load(sample_path, 'sample')
    results = cmd.align('sample', 'refer', cycles=cycles)
    rmsd_after, rmsd_before = results[0], results[3]
    # print(results)
    # cmd.h_add('sample')
    cmd.remove('(hydro)')
    cmd.remove('resn hoh')
    cmd.save(save_path, 'sample')
    return rmsd_after, rmsd_before


# def remove_H_from_mol(sample_path, save_path):
#     mol = Chem.MolFromMolFile(sample_path)
#     mol = Chem.RemoveHs(mol)
#     Chem.MolToMolFile(mol, save_path)


def remove_H_from_pdb(sample_path, save_path):
    cmd.delete('all')
    cmd.load(sample_path, 'sample')
    cmd.remove('hydrogens')
    cmd.save(save_path, 'sample')

def add_H_to_pdb(sample_path, save_path):
    cmd.delete('all')
    cmd.load(sample_path, 'sample')
    cmd.h_add('sample')
    cmd.save(save_path, 'sample')

def align_protein_ligand_pairs(
    sample_sdf_path,
    sample_pdb_path,
    refer_sdf_path,
    refer_pdb_path,
    align_sdf_path,
    align_pdb_path
):
    cmd.delete('all')
    # load data
    cmd.load(sample_sdf_path, 'sample_lig')
    cmd.load(sample_pdb_path, 'sample_rec')
    cmd.load(refer_sdf_path, 'refer_lig')
    cmd.load(refer_pdb_path, 'refer_rec')
    cmd.create('sample_complex', 'sample_lig or sample_rec')
    cmd.create('refer_complex', 'refer_lig or refer_rec')
    # align
    results = cmd.align('sample_complex', 'refer_complex', cutoff=1.5, cycles=5)
    rmsd_after, rmsd_before = results[0], results[3]

    # extract 
    cmd.extract('sample_lig_align', 'sample_complex and resi 0')
    cmd.extract('sample_rec_align', 'sample_complex')

    # save
    # sample_sdf_path = sample_sdf_path[:-4] + '_aligned.sdf'
    # sample_pdb_path = sample_pdb_path[:-4] + '_aligned.pdb'
    cmd.save(align_sdf_path, 'sample_lig_align')
    cmd.save(align_pdb_path, 'sample_rec_align')
    print('RMSD after and before:', rmsd_after, rmsd_before)
    return rmsd_after, rmsd_before


def fix_pdb(pdb_path, save_path, is_print=True, add_hydrogen=True):
    if is_print: print("Creating PDBFixer...")
    fixer = PDBFixer(pdb_path)
    if is_print: print("Finding missing residues...")
    fixer.findMissingResidues()

    chains = list(fixer.topology.chains())
    keys = fixer.missingResidues.keys()
    for key in list(keys):
        chain = chains[key[0]]
        if key[1] == 0 or key[1] == len(list(chain.residues())):
            # if is_print: print("ok")
            del fixer.missingResidues[key]

    if is_print: print("Finding nonstandard residues...")
    fixer.findNonstandardResidues()
    if is_print: print("Replacing nonstandard residues...")
    fixer.replaceNonstandardResidues()
    if is_print: print("Removing heterogens...")
    fixer.removeHeterogens(keepWater=False)

    if is_print: print("Finding missing atoms...")
    fixer.findMissingAtoms()
    if is_print: print("Adding missing atoms...")
    fixer.addMissingAtoms()
    if add_hydrogen:
        if is_print: print("Adding missing hydrogens...")
        fixer.addMissingHydrogens(7)

    if is_print: print("Writing PDB file...")
    PDBFile.writeFile(
        fixer.topology,
        fixer.positions,
        open(save_path, "w"),
        keepIds=True)



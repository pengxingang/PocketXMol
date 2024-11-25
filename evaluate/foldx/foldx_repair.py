from functools import partial
import sys
sys.path.append("..")
import os

from PDButils.commonIO import ParsePDB, PrintPDB
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB import Selection
from PDButils.FoldX import FoldXSession
from tqdm import tqdm
from multiprocessing import Pool

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
        return

if __name__ == '__main__':
    input_dir = 'outputs/pred_pdb/raw'
    output_dir = 'outputs/foldx/repaired'
    os.makedirs(output_dir, exist_ok=True)
    
    files = os.listdir(input_dir)
    with Pool(64) as p:
        results = list(tqdm(p.imap_unordered(
            partial(repair_one_file, input_dir, output_dir), files), total=len(files)))

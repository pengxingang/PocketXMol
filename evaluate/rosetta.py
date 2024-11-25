import sys

sys.path.append("..")

import argparse
import os
import shutil
import subprocess
import tempfile
import pickle

from tqdm import tqdm
import pandas as pd
from Bio.PDB import Selection
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import PDBIO
from distrun.api.joblib import Parallel as d_Parallel
from distrun.api.joblib import delayed as d_delayed
from joblib import Parallel, delayed

import gzip
import warnings
from Bio import BiopythonExperimentalWarning


def calc_interface_score(complex_dir, inter_dir, num_workers, ):
    # calc rosetta score
    interface_score(complex_dir, inter_dir, n_proc=num_workers, )
    save_as_df(os.path.dirname(inter_dir), 'interface')


def calc_rosetta_pep_score(complex_dir, rose_dir, num_workers, pass_list=None):
    # make rosetta dir and input dir

    # calc rosetta score
    # df_score = None
    for mode in ['refine_score', 'score_only', 'min_only']:
    # for mode in ['score_only', 'min_only']:
        rose_mode_dir = os.path.join(rose_dir, mode)
        os.makedirs(rose_mode_dir, exist_ok=True)
        scores_list = pep_score(mode, complex_dir, rose_mode_dir, replace=False, n_proc=num_workers, pass_list=pass_list)
        save_as_df(rose_dir, mode)
        
    #     df_score_this = pd.DataFrame(sum(scores_list, []))
    #     df_score_this.rename(columns={'total_score': mode, 'reweighted_sc': f'{mode}_weighted',
    #                                   'description': f'{mode}_out'}, inplace=True)
    #     if df_score is None:
    #         df_score = df_score_this
    #     else:
    #         df_score = df_score.merge(df_score_this, on='in_pdb')
            
    # df_score['filename'] = df_score['in_pdb'].apply(lambda x: x[:-4] if x else x)
    # return df_score




def ParsePDB(pdb_path, firstModel=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiopythonExperimentalWarning)
        parser_pdb = PDBParser(QUIET=True)
        if pdb_path.endswith(".gz"):
            with gzip.open(pdb_path, "rt") as f:
                structure_pdb = parser_pdb.get_structure(None, f)
        else:
            structure_pdb = parser_pdb.get_structure(None, pdb_path)

        if firstModel:
            return structure_pdb[0]
        else:
            return structure_pdb

def PrintPDB(structure, path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiopythonExperimentalWarning)
        io = PDBIO()
        io.set_structure(structure)
        io.save(path)



class RosettaProcessorException(BaseException):
    pass



class RosettaSession:
    def __init__(self, platform_postfix=".default.linuxgccrelease"):
        super().__init__()
        self.platform_postfix = platform_postfix
        self.tmpdir = tempfile.TemporaryDirectory()
        self.pdb_names = []

    def cleanup(self):
        self.tmpdir.cleanup()
        self.tmpdir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    @property
    def workdir(self):
        return self.tmpdir.name

    def path(self, filename):
        return os.path.join(self.workdir, filename)

    def pdb_path(self, name):
        return self.path(name + ".pdb")

    def pdb_string(self, name):
        with open(self.pdb_path(name)) as f:
            return f.read()

    def store_pdb(self, name, pdb):
        if pdb[-4:].lower() == ".pdb":
            shutil.copy(pdb, self.path(name + ".pdb"))
        else:
            with open(self.path(name + ".pdb"), "w") as f:
                f.write(pdb)
        self.pdb_names.append(name)

    @property
    def _relax_exec(self):
        exec_path = os.path.join("relax" + self.platform_postfix)
        return exec_path
    
    @property
    def _pep_dock(self):
        exec_path = os.path.join("FlexPepDocking" + self.platform_postfix)
        return exec_path
    
    @property
    def _interface_analyze(self):
        exec_path = os.path.join("InterfaceAnalyzer" + self.platform_postfix)
        return exec_path

    @property
    def _fixbb_exec(self):
        exec_path = os.path.join("fixbb" + self.platform_postfix)
        return exec_path

    @property
    def _rosetta_scripts_exec(self):
        exec_path = os.path.join("rosetta_scripts" + self.platform_postfix)
        return exec_path

    @property
    def _residue_energy_breakdown(self):
        exec_path = os.path.join("residue_energy_breakdown" + self.platform_postfix)
        return exec_path

    @property
    def _score(self):
        exec_path = os.path.join("score_jd2" + self.platform_postfix)
        return exec_path

    def copy_bfact(self, input_path, output_path):
        assert os.path.exists(input_path) and os.path.exists(output_path)
        stru1 = ParsePDB(input_path)
        stru2 = ParsePDB(output_path)
        for res1, res2 in zip(
            Selection.unfold_entities(stru1, "R"), Selection.unfold_entities(stru2, "R")
        ):
            score1 = list(Selection.unfold_entities(res1, "A"))[0].bfactor
            for atom2 in Selection.unfold_entities(res2, "A"):
                atom2.bfactor = score1

        PrintPDB(stru2, output_path)
        
        
    @property
    def _database(self, ):
        return 'ROSETTA_DATABASE_PATH'

    def pep_dock(self, mode, name, raw_pdb_path, nstruct, rec_chain='R'):
        assert mode in [
            "flexpep_prepack",
            "flexpep_score_only",
            "flexPepDockingMinimizeOnly",
            # "lowres_preoptimize",
            "pep_refine", # refine
            # "lowres_abinitio", # ab initio
        ], f"mode {mode} not supported"
        cmd = [
            self._pep_dock,
            f"-{mode}",
            "-in:file:s",
            self.pdb_path(name),
            "-receptor_chain",
            rec_chain,
            # "-peptide_chain",
            # "L",
            "-nstruct",
            f"{nstruct}",
            "-ex1",
            "-ex2aro",
            "-database",
            self._database,
        ]
        try:
            out = subprocess.run(cmd, cwd=self.workdir, capture_output=True, text=True)
            if out.returncode != 0:
                raise RosettaProcessorException(
                    "Rosetta failed to pep_dock PDB file. Return code: %d\n%s"
                    % (out.returncode, out.stdout)
                )
        except Exception as e:
            print('Failed for', name)
            return []

        name_list = []

        for i in range(1, nstruct + 1):
            outfile_name = name + "_" + str(i).zfill(4)
            name_list.append(outfile_name)

            if not os.path.exists(self.pdb_path(outfile_name)):
                raise RosettaProcessorException(
                    f"Rosetta failed to generate PDB file. File {outfile_name} not found."
                )
            self.pdb_names.append(outfile_name)
            
        # parse score
        # results_list = self._parse_pep_score('score.sc')
        return name_list

    def _parse_interface_score(self, fname):
        if not os.path.exists(self.path(fname)):
            return {}
        with open(self.path(fname)) as f:
            lines = f.readlines()

        headers = lines[1].split()[1:]
        values = lines[2].split()[1:]
        result_dict = dict(zip(headers, values))
        return result_dict

    def _parse_pep_score(self, fname):
        if not os.path.exists(self.path(fname)):
            return []
        with open(self.path(fname)) as f:
            lines = f.readlines()

        headers = lines[1].split()
        results_list = []
        for line in lines[2:]:
            line = line.split()
            this_data = {}
            this_data['in_pdb'] = self.pdb_names[0]
            for col_index in [-1, 1, 27]:
                key = headers[col_index]
                value = line[col_index]
                this_data[key] = value
            results_list.append(this_data)
        return results_list
    
    def copy_out_pdbs(self, output_dir, name_list):
        for pdb_name in name_list:
            shutil.copy(self.pdb_path(pdb_name), os.path.join(output_dir, pdb_name + ".pdb"))


    def relax(self, name, raw_pdb_path, nstruct, bfact=True):
        cmd = [
            self._relax_exec,
            "-relax:fast",
            "-relax:jump_move",
            "false",
            # '-relax:bb_move', 'false',
            "-relax:constrain_relax_to_start_coords",
            "-in:file:s",
            self.pdb_path(name),
            "-nstruct",
            f"{nstruct}",
            "-output_pose_energies_table",
            "true",
        ]
        try:
            out = subprocess.run(cmd, cwd=self.workdir, capture_output=True, text=True)
            if out.returncode != 0:
                raise RosettaProcessorException(
                    "Rosetta failed to relax PDB file. Return code: %d\n%s"
                    % (out.returncode, out.stdout)
                )
        except:
            return []

        name_list = []

        for i in range(1, nstruct + 1):
            relaxed_name = name + "_" + str(i).zfill(4)
            name_list.append(relaxed_name)

            if not os.path.exists(self.pdb_path(relaxed_name)):
                raise RosettaProcessorException(
                    "Rosetta failed to relax PDB file. File not found."
                )
            self.pdb_names.append(relaxed_name)
            try:
                if bfact:
                    self.copy_bfact(raw_pdb_path, self.pdb_path(relaxed_name))
            except:
                pass
        return name_list

    def create_resfile(self):
        resfile_path = self.path("resfile")
        with open(resfile_path, "w") as f:
            f.write("NATAA")

    def create_xml(self):
        raise NotImplementedError('removed')

    def nov16_repack(self, name, raw_pdb_path):
        self.create_xml()

        cmd = [
            self._rosetta_scripts_exec,
            "-s",
            self.pdb_path(name),
            "-parser:protocol",
            "tmp.xml",
            "-overwrite",
            "--corrections::beta_nov16",
        ]
        out = subprocess.run(cmd, cwd=self.workdir, capture_output=True, text=True)

        name_list = []
        relaxed_name = name + "_" + str(1).zfill(4)
        name_list.append(relaxed_name)
        self.pdb_names.append(relaxed_name)
        self.copy_bfact(raw_pdb_path, self.pdb_path(relaxed_name))

        return name_list

    def fixbb_repack(self, name, raw_pdb_path):
        self.create_resfile()
        cmd = [
            self._fixbb_exec,
            "-s",
            self.pdb_path(name),
            "-resfile",
            "resfile",
        ]
        out = subprocess.run(cmd, cwd=self.workdir, capture_output=True, text=True)

        name_list = []
        relaxed_name = name + "_" + str(1).zfill(4)
        name_list.append(relaxed_name)
        self.pdb_names.append(relaxed_name)
        self.copy_bfact(raw_pdb_path, self.pdb_path(relaxed_name))

        return name_list

    def _parse_position(self, position):
        if ":" in position:
            chain_resseq, icode = position.split(":")
            chain = chain_resseq[-1]
            resseq = int(chain_resseq[:-1])
        else:
            if "-" in position:
                chain = None
                resseq = None
                icode = None
            else:
                chain = position[-1]
                resseq = int(position[:-1])
                icode = " "

        return chain, resseq, icode

    def _parse_energy_table(self, fname):
        with open(self.path(fname)) as f:
            lines = f.readlines()

        table = []
        header = lines[0].split()
        for line in lines[1:]:
            line = line.split()
            names = header[8:-1]
            values = list(map(lambda x: float(x), line[8:-1]))

            position1 = line[3]
            chain1, resseq1, icode1 = self._parse_position(position1)
            position2 = line[6]
            chain2, resseq2, icode2 = self._parse_position(position2)

            energies = dict(zip(names, values))
            table.append(
                {
                    "chain1": chain1,
                    "resseq1": resseq1,
                    "icode1": icode1,
                    "chain2": chain2,
                    "resseq2": resseq2,
                    "icode2": icode2,
                    **energies,
                }
            )

        table = pd.DataFrame(table)
        return table

    def residue_energy_breakdown(self, name):
        out_path = self.path(name + ".energies")
        cmd = [
            self._residue_energy_breakdown,
            "-in:file:s",
            self.pdb_path(name),
            "-out:file:silent",
            out_path,
        ]
        out = subprocess.run(cmd, cwd=self.workdir, capture_output=True, text=True)
        if out.returncode != 0:
            raise RosettaProcessorException(
                "Rosetta failed to calculate per-residue energies. Return code: %d\n%s"
                % (out.returncode, out.stdout)
            )

        if not os.path.exists(out_path):
            raise RosettaProcessorException(
                "Rosetta failed to calculate per-residue energies. Energy table output not found."
            )

        table = self._parse_energy_table(fname=name + ".energies")
        return table

    def tot_energy(self, name):
        out_path = self.path(name + ".energies")
        cmd = [
            self._score,
            "-in:file:s",
            self.pdb_path(name),
            "-out:file:silent",
            out_path,
        ]
        out = subprocess.run(cmd, cwd=self.workdir, capture_output=True, text=True)
        if out.returncode != 0:
            raise RosettaProcessorException(
                "Rosetta failed to calculate per-residue energies. Return code: %d\n%s"
                % (out.returncode, out.stdout)
            )

        if not os.path.exists(out_path):
            raise RosettaProcessorException(
                "Rosetta failed to calculate per-residue energies. Energy table output not found."
            )

        os.system(f"cat {out_path}")
        assert 0
        # table = self._parse_energy_table(fname=name + '.energies')
        return None

    def interface_analyze(self, name, rec_chain='R', lig_chain='L'):
        cmd = [
            self._interface_analyze,
            "-in:file:s",
            self.pdb_path(name),
            "-interface",
            rec_chain+'_'+lig_chain,
            "-out:file:score_only",
            "score.sc",
            "-add_regular_scores_to_scorefile",
            # "-pack_input", # not used since the input is packed by flexpepdock
            "-pack_separated"
        ]
        try:
            out = subprocess.run(cmd, cwd=self.workdir, capture_output=True, text=True)
            if out.returncode != 0:
                raise RosettaProcessorException(
                    "Rosetta failed to pep_dock PDB file. Return code: %d\n%s"
                    % (out.returncode, out.stdout)
                )
        except:
            print('Failed for', name)


def pep_score(mode, input_dir, output_dir, replace=False, n_proc=1, pass_list=None):
    assert mode in ['score_only', 'refine_score', 'min_only']
    def process_one_file(filename):
        if not "pdb" in filename:
            return []
        pdbname = filename.replace('.pdb', '')
        # print("Rosetta Processing:", pdbname)
        pdb_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, pdbname+'.pkl')

        if not replace and os.path.exists(output_path):
            print(pdbname, "is already processed")
            return 

        with RosettaSession() as session:
            session.store_pdb(pdbname, pdb_path)
            nstruct = 1  # not support nstruct > 1
            if mode == 'score_only':
                name_list = session.pep_dock('flexpep_score_only', pdbname, pdb_path, nstruct=nstruct)
            elif mode == 'min_only':
                name_list = session.pep_dock('flexPepDockingMinimizeOnly', pdbname, pdb_path, nstruct=nstruct)
            elif mode == 'refine_score':
                prename_list = session.pep_dock('flexpep_prepack', pdbname, pdb_path, nstruct=nstruct)
                nstruct = 4  # 
                name_list = []
                for filename in prename_list:
                    name_list += session.pep_dock('pep_refine', filename, pdb_path, nstruct=nstruct)
            # choose the one with best score
            score_list = session._parse_pep_score('score.sc')
            if len(score_list) == 0:
                return []
            score_list = [s for s in score_list if s['description'] in name_list]
            min_score = min([float(s['total_score']) for s in score_list])
            score_list = [s for s in score_list if float(s['total_score']) <= min_score]
            name_list = [s['description'] for s in score_list]
            if mode == 'refine_score':
                # copy pdb file
                session.copy_out_pdbs(output_dir, name_list)
            
        # save score
        with open(output_path, 'wb') as f:
            pickle.dump(score_list[0], f)
        return score_list

    all_files = os.listdir(input_dir)
    if pass_list is not None:
        # all_files = [f for f in tqdm(all_files, desc='filtering...') if f in list(pass_list)]
        all_files = set(all_files).intersection(set(pass_list))
    else:
        all_files = set(all_files)
    # all_files = [f for f in all_files if 'pepbdb_6v7o_C_73_cpx' in f]
    results_list = Parallel(n_jobs=n_proc)(
        delayed(process_one_file)(filename) for filename in tqdm(all_files, desc='Start Rosetta...')
    )
    return results_list


def interface_score(input_dir, output_dir, n_proc=1,):
    def process_one_file(filename):
        output_path = os.path.join(output_dir, filename.replace('.pdb', '.pkl'))
        if os.path.exists(output_path):
            return 
        if not "pdb" in filename:
            return
        pdbname = filename.replace('.pdb', '')
        # print("Rosetta Processing:", pdbname)
        pdb_path = os.path.join(input_dir, filename)

        with RosettaSession() as session:
            session.store_pdb(pdbname, pdb_path)
            # nstruct = 1  # not support nstruct > 1
            session.interface_analyze(pdbname)
            
            result_dict = session._parse_interface_score('score.sc')
            
        # save 
        with open(output_path, 'wb') as f:
            pickle.dump(result_dict, f)
        # return result_dict

    all_files = os.listdir(input_dir)
    all_files = set(all_files)
    results_list = Parallel(n_jobs=n_proc)(
        delayed(process_one_file)(filename) for filename in tqdm(all_files, desc='Start Rosetta...')
    )

    return results_list


def fetch_per_pair_energy(input_dir, output_dir, replace=False, n_proc=1):
    def process_one_file(filename):
        if not "pdb" in filename:
            return
        pdbname = filename.replace('.pdb', '')
        print("Rosetta Processing:", pdbname)
        pdb_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{pdbname}.csv")

        if not replace and os.path.exists(output_path):
            print(pdbname, "is already processed")
            return

        with RosettaSession() as session:
            session.store_pdb(pdbname, pdb_path)
            nstruct = 1
            assert nstruct == 1
            relaxed_name_list = session.relax(pdbname, pdb_path, nstruct=nstruct)
            for relaxed_name in relaxed_name_list:
                table = session.residue_energy_breakdown(relaxed_name)
                table.to_csv(output_path)

    Parallel(n_jobs=n_proc)(
        delayed(process_one_file)(filename) for filename in os.listdir(input_dir)
    )


def score_tot_energy(input_dir, output_dir, replace=False, n_proc=1):
    def process_one_file(filename):
        if not "pdb" in filename:
            return
        pdbname = filename.replace('.pdb', '')
        print("Rosetta Processing:", pdbname)
        pdb_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{pdbname}.csv")

        if not replace and os.path.exists(output_path):
            print(pdbname, "is already processed")
            return

        with RosettaSession() as session:
            session.store_pdb(pdbname, pdb_path)
            nstruct = 1
            assert nstruct == 1
            relaxed_name_list = session.relax(pdbname, pdb_path, nstruct=nstruct)
            for relaxed_name in relaxed_name_list:
                session.tot_energy(relaxed_name)
                # table.to_csv(output_path)

    Parallel(n_jobs=n_proc)(
        delayed(process_one_file)(filename) for filename in os.listdir(input_dir)
    )


def relax_one_file(input_path, output_dir, replace, nstruct):
    filename = os.path.basename(input_path)
    input_dir = os.path.dirname(input_path)
    if not "pdb" in filename:
        return
    pdbname = filename.replace('.pdb', '')
    print("Rosetta Processing:", pdbname)
    pdb_path = os.path.join(input_dir, filename)

    output_name = []

    with RosettaSession() as session:
        session.store_pdb(pdbname, pdb_path)

        for i in range(1, nstruct + 1):
            relaxed_name = pdbname + "_" + str(i).zfill(4)
            output_name.append(relaxed_name)

        # check if all the output is exist in the output dir
        exist_sum = 0
        for i, relaxed_name in enumerate(output_name):
            output_path = os.path.join(output_dir, relaxed_name + ".pdb")
            if os.path.exists(output_path):
                exist_sum += 1
        if exist_sum == nstruct:
            return

        relaxed_name_list = session.relax(pdbname, pdb_path, nstruct=nstruct)

        for i, relaxed_name in enumerate(relaxed_name_list):
            output_path = os.path.join(output_dir, relaxed_name + ".pdb")

            if not replace and os.path.exists(output_path):
                # print(pdbname, "is already processed")
                continue

            shutil.copy(session.pdb_path(relaxed_name), output_path)


def relax(input_dir, output_dir, replace=False, n_proc=1, nstruct=1):
    raise NotImplementedError('removed')


def fixbb_repack(input_dir, output_dir, replace=False, n_proc=1):
    def process_one_file(filename):
        if not "pdb" in filename:
            return
        pdbname = filename.replace('.pdb', '')
        print("Rosetta Processing:", pdbname)
        pdb_path = os.path.join(input_dir, filename)

        with RosettaSession() as session:
            session.store_pdb(pdbname, pdb_path)
            relaxed_name_list = session.fixbb_repack(pdbname, pdb_path)

            for i, relaxed_name in enumerate(relaxed_name_list):
                output_path = os.path.join(output_dir, relaxed_name + ".pdb")

                if not replace and os.path.exists(output_path):
                    print(pdbname, "is already processed")
                    return

                shutil.copy(session.pdb_path(relaxed_name), output_path)

    Parallel(n_jobs=n_proc)(
        delayed(process_one_file)(filename) for filename in os.listdir(input_dir)
    )


def nov16_repack(input_dir, output_dir, replace=False, n_proc=1):
    def process_one_file(filename):
        if not "pdb" in filename:
            return
        pdbname = filename.replace('.pdb', '')
        print("Rosetta Processing:", pdbname)
        pdb_path = os.path.join(input_dir, filename)

        with RosettaSession() as session:
            session.store_pdb(pdbname, pdb_path)
            relaxed_name_list = session.nov16_repack(pdbname, pdb_path)

            for i, relaxed_name in enumerate(relaxed_name_list):
                output_path = os.path.join(output_dir, relaxed_name + ".pdb")

                if not replace and os.path.exists(output_path):
                    print(pdbname, "is already processed")
                    return

                shutil.copy(session.pdb_path(relaxed_name), output_path)

    Parallel(n_jobs=n_proc)(
        delayed(process_one_file)(filename) for filename in os.listdir(input_dir)
    )


def save_as_df(rosetta_dir, mode):
    df = []
    for file in tqdm(os.listdir(os.path.join(rosetta_dir, mode))):
        path = os.path.join(rosetta_dir, mode, file)
        if not path.endswith('.pkl'):
            continue
        try:
            with open(path, 'rb') as f:
                score = pickle.load(f)
        except:
            continue
        df.append(score)
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(os.path.join(rosetta_dir, mode+'.csv')), index=False)
    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--complex_dir', type=str, default='outputs_use/PDL1/combine_1215_opt/complex')
    # parser.add_argument('--rosetta_dir', type=str, default='outputs_use/PDL1/combine_1215_opt/rosetta')
    # parser.add_argument('--filter_path', type=str, default='outputs_use/PDL1/combine_1211/seq_filter_foldx_253k.csv')
    # parser.add_argument('--filter_path', type=str, default='')
    # parser.add_argument('--complex_dir', type=str, default='outputs_use/PDL1/combine_pep_opt_1220/complex')
    # parser.add_argument('--rosetta_dir', type=str, default='outputs_use/PDL1/combine_pep_opt_1220/rosetta')
    # parser.add_argument('--complex_dir', type=str, default='outputs_paper/dock_pepbdb/msel_base_20240105_160647/complex')
    # parser.add_argument('--rosetta_dir', type=str, default='outputs_paper/dock_pepbdb/msel_base_20240105_160647/rosetta')
    # our pepfull
    # parser.add_argument('--complex_dir', type=str, default='outputs_paper/pepdesign_pepbdb/msel_pepfull_20240312_210440/complex')
    # parser.add_argument('--rosetta_dir', type=str, default='outputs_paper/pepdesign_pepbdb/msel_pepfull_20240312_210440/rosetta')
    # parser.add_argument('--filter_path', type=str, default='outputs_paper/pepdesign_pepbdb/msel_pepfull_20240312_210440/gen_info.csv')
    # rfdiffusion+mpnn pepfull
    # parser.add_argument('--complex_dir', type=str, default='baselines/pepdesign/full_rfdiff_mpnn/complex_for_rosetta')
    # parser.add_argument('--rosetta_dir', type=str, default='baselines/pepdesign/full_rfdiff_mpnn/rosetta')
    # rfdiffusion+mpnn pepfull from packed interface analyzer
    parser.add_argument('--complex_dir', type=str, default='baselines/pepdesign/full_rfdiff_mpnn/rosetta/pack')
    parser.add_argument('--rosetta_dir', type=str, default='baselines/pepdesign/full_rfdiff_mpnn/rosetta')
    parser.add_argument('--tasks', type=str, default='interface_cpx ')
    # pepbdb
    # parser.add_argument('--complex_dir', type=str, default='baselines/pepdesign/pepbdb/files/complex')
    # parser.add_argument('--rosetta_dir', type=str, default='baselines/pepdesign/pepbdb/rosetta')
    parser.add_argument('--filter_path', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=126)
    # parser.add_argument('--tasks', type=str, default='flexpepdock interface_cpx')
    args = parser.parse_args()
    
    # # rosetta flexpepdock
    if args.filter_path:
        # pass_list = (pd.read_csv(args.filter_path)['filename']+'.pdb').values
        df = pd.read_csv(args.filter_path)  # pepdesign eval
        pass_list = df.loc[df['tag'].isna(), 'filename'].str.replace('.pdb', '_cpx.pdb', regex=True).values
    else:
        pass_list = None
    
    if 'flexpepdock' in args.tasks:
        calc_rosetta_pep_score(args.complex_dir, args.rosetta_dir, num_workers=args.num_workers,
                            pass_list=pass_list)

    # # interface analyzer
    inter_dir = os.path.join(args.rosetta_dir, 'interface')
    os.makedirs(inter_dir, exist_ok=True)
    if 'interface_refined' in args.tasks:
        refine_dir = os.path.join(args.rosetta_dir, 'refine_score')
        calc_interface_score(refine_dir, inter_dir, num_workers=args.num_workers)
    elif 'interface_cpx' in args.tasks:
        cpx_dir = args.complex_dir
        calc_interface_score(cpx_dir, inter_dir, num_workers=args.num_workers)
    
    print('Done.')
    print(args.tasks)


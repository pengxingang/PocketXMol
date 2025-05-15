# PocketXMol

This is a *preliminary* code release for the pocket-interacting foundation model "**PocketXMol**".

> Please note that this code is a preview version and has yet to be cleaned up and refactored. It may be hard to read but should be functional for running. We will continue to improve the code and provide more detailed instructions in the future.  

> This repository is modified from the [MolDiff](https://github.com/pengxingang/MolDiff) repository (a good starting point for diffusion-based molecular generation). We thank the authors for their work.

<!-- toc -->
**Contents**
- [Setup](#setup)
  * [Environment](#environment)
  * [Data and model weights](#data-and-model-weights)
    * [Example data and model weights for sampling](#example-data-and-model-weights-for-sampling)
    * [Processed test data and model weights for sampling](#processed-test-data-and-model-weights-for-sampling)
    * [Processed data for training](#processed-data-for-training)
    * [Raw data and processing steps for training](#raw-data-and-processing-steps-for-training)
- [Sample for provided data](#sample-for-provided-data)
    * [Usage examples](#usage-examples)
    * [Confidence scores for sampled molecules](#confidence-scores-for-sampled-molecules)
    * [Basic configuration explanation](#basic-configuration-explanation)
    * [Customized setting explanation](#customized-setting-explanation)
- [Sample for data in test sets](#sample-for-data-in-test-sets)
  * [Small molecule docking](#small-molecule-docking)
  * [Peptide docking](#peptide-docking)
  * [Molecular conformation generation](#molecular-conformation-generation)
  * [Structure-based drug design (SBDD)](#structure-based-drug-design-sbdd)
  * [3D molecule generation](#3d-molecule-generation)
  * [Fragment linking](#fragment-linking)
  * [PROTAC design](#protac-design)
  * [Fragment growing](#fragment-growing)
  * [*De novo* peptide design](#de-novo-peptide-design)
  * [Peptide inverse folding](#peptide-inverse-folding)
- [Train](#train)
<!-- tocstop -->



# Setup
## Dependency

To setup the environment on a Linux server, you can use [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) to create a new environment `pxm` from the `environment.yml` file (for CUDA 11.7) using the following commands (takes several minutes):
```bash
conda env create -f environment.yml
conda activate pxm
```
If you have a different CUDA version, you may need to modify the versions of the pytorch-related packages in the `environment.yml` file.


Or you can install the dependencies **manually**.
For example, for CUDA 12.6, using the following command:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install pytorch-lightning
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
pip install biopython==1.83 rdkit==2023.9.3 peptidebuilder==1.1.0
pip install openbabel=3.1.1.1  # or, conda install -c conda-forge openbabel -y
pip install lmdb easydict==1.9 numpy==1.24 pandas==1.5.2
pip install tensorboard
```



## Data and model weights

### Example data and model weights for sampling

The example data are included in the `data/examples` directory, which are used to demonstrate the usage in [Sample for provided data](#sample-for-provided-data).

The model weights for sampling are included in the file `model_weights.tar.gz` available from the [Google Drive](https://drive.google.com/file/d/1Hu6qTkCyNUPPsQLLHL1kBFiwRbKUOFLs/view?usp=drive_link).
Download and extract it using the command:
```bash
tar -zxvf model_weights.tar.gz
```
After extraction, there will be a directory named `data/trained_model` which contains the trained model weights for sampling.

### Processed test data and model weights for sampling

For **sampling for test sets**, the processed test data and trained model weights are included in the file `data_test.tar.gz` available from the [Google Drive](https://drive.google.com/drive/folders/1-nmm2O_bHdYastqbtSkIXJRL247MkAWf?usp=sharing).
Download and extract it using the command:
```bash
tar -zxvf data_test.tar.gz
```
After extraction, there will be a directory named `data` which contains:
- test sets: `test` for benchmark related information; `csd` for CrossDocked2020 set, `geom` for GEOM-Drug set, `moad` for Binding MOAD set, `pepbdb` for PepBDB set, `poseboff` for PoseBusters set, and `protacdb` for PROTAC-DB set.
- trained model weights in the `trained_model` directory for sampling.
- example data files (in `examples` directory) for demonstrating the sampling for user-provided files.


### Processed data for training

For **training**, the demonstrative processed training data are in the file `data_train_processed_reduced.tar.gz` from the [Google Drive](https://drive.google.com/drive/folders/1-nmm2O_bHdYastqbtSkIXJRL247MkAWf?usp=sharing).
The complete processed training data are too large (>500G) so we provide a reduced subset just to demonstrate the training process. Similarly, download and extract it using the command:
```bash
tar -zxvf data_train_processed_reduced.tar.gz
```
Then there is a directory named `data_training` containing reduced training sets for demonstrative training.

### Raw data and processing steps for training

If you want to train the model with the full training data, please follow the instructions in the [process/process_steps.md](process/process_raw_steps.md) file to process the raw data for complete training.



# Sample for provided data

> We provide interactive Colab notebooks for sampling. You can find the notebooks in the [notebooks](notebooks) directory:
> - Dock [![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pengxingang/PocketXMol/blob/master/notebooks/PXM_Dock.ipynb)
> - Peptide Design [![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pengxingang/PocketXMol/blob/master/notebooks/PXM_PeptideDesign.ipynb)
> - Small molecule design [![dock](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pengxingang/PocketXMol/blob/master/notebooks/PXM_SmallMoleculeDesign.ipynb)

## Usage examples

Here, we demonstrate some examples of sampling using the provided data in the `data/examples` directory.


Run the following command:
```python
python scripts/sample_use.py \
    --config_task configs/sample/examples/dock_smallmol.yml \
    --outdir outputs_examples \
    --device cuda:0
```
The configuration files are in `configs/sample/examples`, including:
- Docking
    - `dock_smallmol.yml`: dock a small molecule to a protein pocket
    - `dock_smallmol_flex.yml`: dock a small molecule to a protein pocket using flexible noise
    - `dock_smallmol_84663.yml`: dock the molecule 84663 to caspase-9
    - `dock_pep.yml`: dock a peptide to a protein pocket
    - `dock_pep_fix_some.yml`: dock, with fixed coordinates of some atoms
    - `dock_pep_know_some.yml`: dock, with constrained coordinates of some atoms
- Small molecule design
    - `sbdd.yml`: design drug-like molecules for protein pocket
    - `sbdd_simple.yml`: design drug-like molecules for protein pocket (no refinement rounds)
    - `growing_unfixed_frag.yml`: fragment growing with unfixed fragment pose, i.e., design small molecules containing a specified fragment graph for protein pocket
    - `growing_fixed_frag.yml`: fragment growing with fixed fragment pose, i.e., design small molecules containing a specified fragment with fixed pose for protein pocket
- Peptide design
    - `pepdesign.yml`: design peptides for protein pocket
- Design with customized settings:
    - `pepdesign_hot136E`: this directory considers a specific peptide design case. Based on the PD1-PDL1 complex (PDB ID: 3BIK), we found a hot spot residue 136E on PD1 interacting with PDL1. 
    We aim to design a PDL1-binding peptide considering this interaction. 
    We extract the protein fragment around 136E as the input peptide and the PDL1 chain as the target (in `data/examples/hot136E`).
    There are several strategies for designing peptdes (see [Customized setting explanation](#customized-setting-explanation) for configuration explanation):
        - `fixed_Glu_CCOOH`: design peptide whose 6th residue (count from 1) is Glu and its -CCOOH group pose is fixed as input.
        - `fixed_CCOOH`: the designed peptide contains a -CCOOH group and its pose is fixed as input. But the -CCOOH may not be at the 6th residue and may not be in Glu (Asp can also contain -CCOOH).
        - `fixed_CCOOH_init0.9`: the setting is the same as `fixed_CCOOH`, but the initial noisy peptide is generated by adding noise to the peptide of the input file, instead of being sampled from the noise prior. The only difference is the parameter `noise/init_step`. Hint: by setting `noise/init_step`$< 1$, the initial noisy coordinates will be sampled from the Gaussian noise with mean equal to the input coordinates instead of the noise space center. 
        - `unfixed_Glu`: design peptides with Glu at the 6th residue. No atom coordinates are fixed.
        - `unfixed_CCOOH`: design peptides containing -CCOOH group (i.e., containing Glu or Asp), but the -CCOOH group can be at any residue index. No atom coordinates are fixed.
        - `unfixed_CCOOH_from_inputs`: the setting is the same as `unfixed_CCOOH`, but the initial pose of the -CCOOH group are sampled based on the input peptide. 
        This align with our intuition that the -CCOOH group in the designed peptide should interact with the protein in a similar way as the input one.

More examples are on the way.

## Confidence scores for sampled molecules

The self-confidence scores are in the `gen_info.csv` (column `cfd_traj`) file produced during the sampling process. To calculate other confidence scores for the generated molecules, use the command like this:
```python
python scripts/believe_use_pdb.py \
    --exp_name pepdesign_pxm_20210101_150132 \
    --result_root outputs_use \
    --config configs/sample/confidence/tuned_cfd.yml \
    --device cuda:0
```
The parameters:
- `result_root` is the directory containing the sampling experiments (equal to the parameter `outdir` of the sampling command).
- `exp_name` is the name of the sampling experiment directory (looks like `pepdesign_pxm_20210101_150132`). If there is only one experiment with the name starting with the `exp_name`, the appended timestamp can be omitted (i.e., set as `pepdesign_pxm`).
- `config` is the confidence model configuration file. They are in `configs/sample/confidence` including:
    - `tuned_cfd.yml`: the tuned confidence predictor
    - `flex_cfd.yml`: using original model with flexible noise for confidence prediction

After running, the `.csv` files of confidence scores will be saved at the `ranking` sub-directory.

## Basic configuration explanation

You can refer to these configuration files and adapt to your own data and tasks. Here are some simple explanations of the configuration.

Typically there are five main blocks: `sample`, `data`, `transforms`, `task`, and `noise`. The first three keys define the data and sampling parameters, and the last two define the task.
In most cases, you only need to find a task template configuration file and modify the first three blocks.

- `sample`: the sampling parameters, including base random seed, batch size, and the number of generated molecules. The parameter `save_traj_prob` means the frequency of saving the generation trajectories.
- `data`: the input data, including
    - `protein_path`: the path to the protein PDB file.
    - `input_ligand`: the information of input ligand.
        - For small molecule, it can be a SDF file path, the SMILES string or `None` (for *de novo* small mol design).
        - For peptide, it can be the PDB file path, the sequence string (with prefix `pepseq_`, e.g., pepseq_DTVFALFW, for docking) or the sequence length (with prefix `peplen_`, e.g., peplen_10, for *de novo* design).
    - `is_pep`: bool, whether the ligand is peptide. It is used to create the PDB files for the generated molecules. If not set, it will be automatically determined according to `input_ligand`.
    - `pocket_args`: dict of pocket parameters, including
        - `ref_ligand_path`: path to the reference molecule file (SDF or PDB). This molecule is used to determine the pocket from the complete protein, i.e., the residues within a certain distance to the reference molecule are defined as pocket residues. Exclusive to `pocket_coord`.
        - `pocket_coord`: the coordinate of the pocket. The pocket will be defined as the residues near the coordinate. Exclusive to `ref_ligand_path`. If neither `ref_ligand_path` nor `pocket_coord` is set, it will use `input_ligand` as reference.
        - `radius`: the residues within the radius to the reference ligand or the pocket coordinate are defined as pocket residues. Default is 10.
        - `criterion`: the criterion to define the residue distance, be one of ['center_of_mass', 'min']. Default is 'center_of_mass'.
    - `pocmol_args`: user-defined identifiers. Not important.
        - `data_id`
        - `pdbid`
- `transforms` (optional): the extra data processing parameters, including
    - `featurizer_pocket`:
        - `center`: coordinate space center for denoising. It influences sampling atom coordinates from the Gaussian noise at the first step. If not set, it will be automatically defined as the average of pocket atom coordinates. (You can also use `featurizer/mol_as_pocket_center` to specify the pocket center)
    - `featurizer`
        - `mol_as_pocket_center`: bool, use the center coordinates of the ligand as the space center. If set to `True`, the parameter `data/pocket_args/input_ligand` should be SDF/PDB file. (You can also use `featurizer_pocket/center` to specify the pocket center)
    - `variable_mol_size`: distributions of the number of atoms for small-molecule designing tasks.
    - `variable_sc_size`: distributions of number of side-chain atoms for peptide designing. The default value should work well.
- `task`: the task and its specific mode.
- `noise`: the task nosie parameters.


## Customized setting explanation
Here we explain the customized settings in the `examples/pepdesign_hot136E` directory. 

In these settings, we defined a task called `custom`. (Basically, all the previous common tasks can be expressed through this `custom` task.)
The basic idea is to (1) define several groups of noise, (2) partition the molecules into several parts, and (3) map the noise groups to the molecule parts.

In their config files, the `sample` and `data` blocks are the same as the common tasks. For other blocks:

- `transforms`: similar as the common tasks, but with some additional settings:
    - The `variable_sc_size/applicable_tasks` should contain the task name `custom`.
    - Some side-chain atoms of input peptides will be randomly removed for variable sizes. Set `variable_sc_size/not_remove` to exclude side-chain atoms from being removed. This is a list of atom indices in the input peptide (starting from 0).
- `task`: In `task/transform`, please define:
    - `is_peptide`: wheter the task is related to peptide or small molecule. This is the prompt $\mathbf{P}^{\text{pep}}$ in the paper.
    - `partition`: this is where you define how you partition the molecule. It is a list of dictionaries, and each dictionary contains:
        - `name`: the name of the part.
        - `nodes`: the atom indices of the part. The atom indices are 0-based.
    - `fixed`: define which variables are fixed as input, including:
        - `node`: list of molecular parts whose atom types are fixed.
        - `pos`: list of molecular parts whose atom coordinates are fixed.
        - `edge`: list of molecular part pairs whose inner bond types are fixed.
- `noise`: 
    - `num_steps`: the number of sampling steps. It is an integer and $100$ should work well.
    - `init_step`: the initial step of noise. It is a scalar in $(0, 1]$ and default is $1$. During the sampling, the step will decay from `init_step` to $0$ linearly. Larger value means more noise. If it is set as $1$, the initial noisy molecule will be sampled from the noise prior without considering the input molecules. Specifically, the coordinates will be sampled from the Gaussian noise with mean equal to the noise space center.
    If it is less than $1$ (and the parameter `from_prior` in the noise group is not set as `False` (default)), the initial noisy coordinates will be sampled from the Gaussian noise with mean equal to the input coordinates instead of the noise space center.
    - `prior`: define the noise prior distributions for different noise groups. This is a dictionary, and each key is the noise group name and the value is the noise prior distribution. Tips:
        - For each noise group, define the noise prior distributions for `node` (atom type), `pos` (atom coordinate), and `edge` (bond type). You can refer to the noise prior settings in the training configuration file (`configs/train/train_pxm_reduced.yml`) for reference.
        - If there is only `pos` noise, you can set `pos_only` as `True`. 
        - Set `from_prior` as `True` (default) to sample the initial noisy coordinates completely from the noise prior. If you want to consider the input coordinates, you can set `from_prior` as `False` to disable the initial noisy coordinates sampling from the noise prior but based on the input coordinates (see `unfixed_CCOOH_from_inputs.yml`) even if `noise/init_step` is set as $1$. This is useful when some atom coordinates of the input molecule can provide a good starting point for the generation or their approximate coordinates are known.
    - `level`: define information level strategies for different noise group. Information level strategy controls the noise scale at each step, i.e., it is a mapping from the step to the information level (within the interval $[0,1]$, information level is $1-$ *noise level*). Tips:
        - You can refere to the level settings in the training configuration file (`configs/train/train_pxm_reduced.yml`) for reference.
        - Usually the *uniform level* should work well for *de novo* generation. If you want to preserve more information of the input file, you can set the `min` level as a larger value.
    - `mapper`: define the mapping from the noise groups to the molecule parts. This is a dictionary, and each key is the noise group name and the value is the molecule part name of the variables `node` (atom type), `pos` (atom coordinate), and `edge` (bond type).
        




# Sample for data in test sets

We provide the configuration files for sampling in the test sets of individual tasks.

> NOTE:
> - The **batch size** for sampling is defined in the configuration files. They were verified on an 80G A100 GPU. If the batch size is too large for your GPU memory, please reduce `batch_size` in the configuration files or directly set the batch size in the command line (e.g., `--batch_size 100`).
> - Typical running time for individual test sets is around 1 ~ 6 hours on a single A100 GPU.
> - After sampling, there will be a new directory in the specified `outdir` containing the generated results. The new directory is named as `{exp_name}_{timestamp}` where `exp_name` is created from the names of the configuration file and `timestamp` is the time when the experiment starts. Within it, the `SDF` subdirectory contains the generated molecules, and files `gen_info.csv` and `log.txt` contain the generation information.

## Small molecule docking
Sample docking poses for 428 pairs of protein and small-molecule in the PoseBusters set.
```python
python scripts/sample_drug3d.py \
    --config_task configs/sample/test/dock_poseboff/base.yml \
    --outdir outputs_test/dock_posebusters \
    --device cuda:0
```

The task configuration files are in `configs/sample/test/dock_poseboff`.
Configuration files include:
- `base.yml`: dock using Gaussian noise (default)
- `base_flex.yml`: dock using flexible noise
<!-- - `prior_chirality.yml`: dock with prior knowledge of chirality -->
- `prior_center.yml`: dock with prior knowledge of the molecular center
- `prior_bond_length.yml`: dock with prior knowledge of bond length
- `prior_anchor.yml`: dock with prior knowledge of approximate anchor atom coordinate
- `prior_fix_anchor.yml`: dock with fixed anchor atom coordinate



#### Confidence scores
The self-confidence scores are in the `gen_info.csv` (column `cfd_traj`) file produced during the sampling process. To calculate other confidence scores for the generated molecular poses, use the following command: 
```python
python scripts/believe.py \
    --exp_name base_pxm \
    --result_root outputs_test/dock_posebusters \
    --config configs/sample/confidence/tuned_cfd.yml \
    --device cuda:0
```
The parameters:
- `result_root` is the directory containing the sampling experiments (equal to the parameter `outdir` of the sampling command).
- `exp_name` is the name of the sampling experiment directory (looks like `base_pxm_20241030_225401`). If there is only one experiment with the name starting with the `exp_name`, the appended timestamp can be omitted (i.e., set as `base_pxm`).
- `config` is the confidence model configuration file. They are in `configs/sample/confidence` including:
    - `tuned_cfd.yml`: the tuned confidence predictor
    - `flex_cfd.yml`: using original model with flexible noise for confidence prediction

#### Ranking scores
To get the ranking scores for pose selection, after obtaining the confidence scores, use the following command:
```python
python scripts/rank_pose.py \
    --exp_name base_pxm \
    --result_root outputs_test/dock_posebusters \
    --db poseboff
```
to produce the `ranking.csv` file which contains the `self_ranking` and `tuned_ranking` columns as ranking scores.


## Peptide docking
Sample docking poses for 79 pairs of protein and peptide in the peptide docking test set.
```python
python scripts/sample_pdb.py \
    --config_task configs/sample/test/dock_pepbdb/base.yml \
    --outdir outputs_test/dock_pepbdb \
    --device cuda:0
```
The task configuration files are in `configs/sample/test/dockpep_pepbdb`.
Configuration files include:
- `base.yml`: dock using Gaussian noise (default)
- `base_flex.yml`: dock using flexible noise
- `prior_fix_anchor.yml`: dock with fixed anchor atom coordinate
- `prior_fix_first_residue.yml`: dock with fixed first residue atom coordinates
- `prior_fix_terminal_residue.yml`: dock with fixed both terminal residue atom coordinates
- `prior_fix_backbone.yml`: dock with fixed backbone atom coordinates

<!-- TODO: remove foldx. set same as small molecule docking -->

<!-- To apply FoldX to rank the docked peptide poses, use the following commands to prepare the FoldX input files and run FoldX:
```python
python evaluate/prepare_cpx.py --result_root outputs_test/dock_pepbdb \
            --exp_name base_pxm
python evaluate/foldx/foldx_pipeline.py --result_root outputs_test/dock_pepbdb \
            --exp_name base_pxm  --num_workers  126
```
The parameter `result_root` is the directory containing the sampling experiments (equal to the parameter `outdir` of the sampling command).
The parameter `exp_name` is the name of the sampling experiment directory (looks like `base_pxm_20240531_225401`). If there is only one experiment with the name starting with the `exp_name`, the appended timestamp can be omitted (i.e., set as `base_pxm`). -->




## Molecular conformation generation
Sample molecular conformations for the 199 molecules in the conformation test set.
```python
python scripts/sample_drug3d.py \
    --config_task configs/sample/test/conf_geom/base.yml \
    --outdir outputs_test/conf_geom \
    --device cuda:0
```

## Structure-based drug design (SBDD)
Sample drug-like molecules for the 100 protein pockets in the SBDD test set.
```python
python scripts/sample_drug3d.py \
    --config_task configs/sample/test/sbdd_csd/base.yml \
    --outdir outputs_test/sbdd_csd \
    --device cuda:0
```
The task configuration files are in `configs/sample/test/sbdd_csd`.
Configuration files include:
- `base.yml`: sbdd using refine-based sampling strategy (default)
- `ar.yml`: sbdd using an auto-regressive-like sampling strategy
- `simple.yml`: sbdd with only one generation round, not using confidence scores for sampling
- `base_mol_size.yml`: sbdd using refine-based sampling strategy with molecular sizes determined from reference molecules


## 3D molecule generation
Generate drug-like molecules with the sizes as the GEOM-Drug validation set.
```python
python scripts/sample_drug3d.py \
    --config_task configs/sample/test/denovo_geom/base.yml \
    --outdir outputs_test/denovo_geom \
    --device cuda:0
```
The task configuration files are in `configs/sample/test/denovo_geom`.
Configuration files include:
- `base.yml`: molecule generation using refine-based sampling strategy (default)
- `ar.yml`: molecule generation using an auto-regressive-like sampling strategy
- `simple.yml`: molecule generation with only one generation round, not using confidence scores for sampling


## Fragment linking
Design molecules by linking fragments for the 416 pairs of proteins and fragments in the fragment linking test set.
```python
python scripts/sample_drug3d.py \
    --config_task configs/sample/test/linking_moad/known_connect.yml \
    --outdir outputs_test/linking_moad \
    --device cuda:0
```
<!-- python scripts/sample_drug3d.py --outdir outputs_test/linking_moad \
            --device cuda:0 --config_task configs/sample/test/linking_moad/known_connect.yml -->
The task configuration files are in `configs/sample/test/linking_moad`.
Configuration files include:
- `known_connect.yml`: fragment linking with known connecting atoms of fragments
- `unknown_connect.yml`: fragment linking with unknown connecting atoms of fragments


## PROTAC design
Design PROTAC molecules by linking fragments for the 43 fragment pairs in the PROTAC-DB test set.
```python
python scripts/sample_drug3d.py \
    --config_task configs/sample/test/linking_protacdb/fixed_fragpos.yml \
    --outdir outputs_test/linking_protacdb \
    --device cuda:0
```
The task configuration files are in `configs/sample/test/linking_protacdb`.
Configuration files include (all assume known connecting atoms of fragments):
- `fixed_fragpos.yml`: fragment linking with fixed fragment poses
- `unfixed_lv0.yml`- `unfixed_lv4.yml`: fragment linking with unfixed fragment poses. The input fragment poses were derived by randomly perturb the true fragment poses with different levels of noise (lv0=smallest).


## Fragment growing
Design molecules through growing fragments for the 53 pairs of fragment and protein in the fragment growing test set.
```python
python scripts/sample_drug3d.py \
    --config_task configs/sample/test/growing_csd/base.yml \
    --outdir outputs_test/growing_csd \
    --device cuda:0
```
The task configuration file is `configs/sample/test/growing_csd/base.yml`.



## *De novo* peptide design
Design peptides for the 35 protein pockets in the peptide design test set.
```python
python scripts/sample_pdb.py \
    --config_task configs/sample/test/pepdesign_pepbdb/base.yml \
    --outdir outputs_test/pepdesign_pepbdb \
    --device cuda:0
```
The task configuration file is `configs/sample/test/pepdesign_pepbdb/base.yml`.


## Peptide inverse folding
Design peptides for the 35 pairs of backbone structures and protein pockets in the peptide design test set.
```python
python scripts/sample_pdb.py \
    --config_task configs/sample/test/pepinv_pepbdb/base.yml \
    --outdir outputs_test/pepinv_pepbdb \
    --device cuda:0
```
The task configuration file is `configs/sample/test/pepinv_pepbdb/base.yml`.



# Train



Make sure to download and extract the training data `data_training_processed_reduced.tar.gz` as described in the [Data and model weights](#data-and-model-weights) section.
Then run the following command to train the model with reduced data:
```python
python scripts/train_pl.py --config configs/train/train_pxm_reduced.yml --num_gpus 1
```
You can specify the number of GPUs to use by setting the `num_gpus` parameter.
The training configuration file is defined in `configs/train/train_pxm_reduced.yml`. 
You can change the `batch_size` parameter in the configuration file to adjust to your GPU memory.

If you want to train the model with the full training data, please follow the instructions in the [Raw data and processing steps](#raw-data-and-processing-steps) section to process the raw data for training. Then, modify `data.dataset.root` and `data.dataset.assembly_path` in the training configuration file to point to the full training data directory and run the training command as above.
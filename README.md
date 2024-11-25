# PocketXMol

This is a *preliminary* code release for the pocket-interacting foundation model "**PocketXMol**".

> Please note that this code is a preview version and has yet to be cleaned up and refactored. It may be hard to read but should be functional for running.

<!-- toc -->
**Contents**
- [Setup](#setup)
  * [Environment](#environment)
  * [Data and model weights](#data-and-model-weights)
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
- [Sample for provided data](#sample-for-provided-data)
- [Train](#train)
<!-- tocstop -->



# Setup
## Environment

To setup the environment on a Linux server, you can use [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) to create a new environment `pxm` from the `environment.yml` file (for CUDA 11.7) using the following commands (takes several minutes):
```bash
conda env create -f environment.yml
conda activate pxm
```
If you have a different CUDA version, you may need to modify the versions of the pytorch-related packages in the `environment.yml` file.


## Data and model weights

### Processed test data and model weights for sampling

For **sampling**, the processed data and trained model weights are included in the file `data_test.tar.gz` available from the [Google Drive](https://drive.google.com/drive/folders/1-nmm2O_bHdYastqbtSkIXJRL247MkAWf?usp=sharing).
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

### Raw data and processing steps

If you want to train the model with the full training data, please follow the instructions in the [process/process_steps.md](process/process_raw_steps.md) file to process the raw data for complete training.




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
The self-confidence scores are in the `gen_info.csv` file produced during the sampling process. To calculate other confidence scores for the generated molecular poses, use the following command: 
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
- `config` is the confidence model confifuration file. They are in `configs/sample/confidence` including:
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


## PROTAC linker design
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



# Sample for provided data
Here, we demonstrate some examples of sampling using the provided data in the `data/examples` directory.
For small-molecule tasks, run the following command:
```python
python scripts/sample_use.py \
    --config_task configs/sample/examples/dockmol.yml \
    --outdir outputs_examples \
    --device cuda:0
```
The configuration files are in `configs/sample/examples`, including:
- `dockmol.yml`: dock a small molecule to a protein pocket
- `dockmol_flex.yml`: dock a small molecule to a protein pocket using flexible noise
- `sbdd.yml`: design drug-like molecules for protein pocket
- `growing.yml`: grow a fragment to design molecules for protein pocket
- `growing_fix_frag.yml`: grow a fragment to design molecules for protein pocket with fixed fragment pose

For peptide tasks, run the following command:
```python
python scripts/sample_use_pdb.py \
    --config_task configs/sample/examples/dockpep.yml \
    --outdir outputs_examples \
    --device cuda:0
```
The configuration files are in `configs/examples`, including:
- `dockpep.yml`: dock a peptide to a protein pocket
- `pepdesign.yml`: design peptides for protein pocket



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
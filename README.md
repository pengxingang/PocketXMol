# PocketXMol

This repository provides the original code release for **PocketXMol**, a pocket-interacting foundation model introduced in the paper **Learning fundamental atom interactions for unified modeling of 3D molecule generation** published in *Cell*.

> This repository is modified from the [MolDiff](https://github.com/pengxingang/MolDiff) repository. We thank the authors for their work.


## Capabilities

PocketXMol is an AI generative model that learns fundamental **atom interactions**, enabling applications governed by atom interactions within a pocket, including:

- **Structure prediction**: Small-molecule docking, peptide docking, and molecular conformation generation.
- **Molecular design**: Structure-based drug design (SBDD), fragment linking/growing, PROTAC design, de novo peptide design, and peptide inverse folding.
- **Complex manipulation**: Combining prediction and design, such as partial structure prediction/generation.


## Quickstart

### 0) Try it in Colab

Interactive notebooks are available under [notebooks/](notebooks) and on Colab:

> - **Dock**: [![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pengxingang/PocketXMol/blob/master/notebooks/PXM_Dock.ipynb)
> - **Peptide Design**: [![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pengxingang/PocketXMol/blob/master/notebooks/PXM_PeptideDesign.ipynb)
> - **Small Molecule Design**: [![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pengxingang/PocketXMol/blob/master/notebooks/PXM_SmallMoleculeDesign.ipynb)

### 1) Create the environment

Using conda to create the environment. For CUDA 11.7, use the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate pxm
```

For other options (manual pip), see [Setup Guide](docs/setup.md).

### 2) Download model weights

Training/Test data and model weights are hosted on Zenodo: https://zenodo.org/records/17801271

For sampling, download `model_weights.tar.gz` from Zenodo and extract it:

```bash
tar -zxvf model_weights.tar.gz
```

### 3) Run an example
Run the following command to sample small molecule docking poses for an example protein-ligand pair:

```bash
python scripts/sample_use.py \
        --config_task configs/sample/examples/dock_smallmol.yml \
        --outdir outputs_examples \
        --device cuda:0
```
*(If you encounter GPU OOM errors, add `--batch_size 50` to adjust batch_size or modify the config).*

For more examples and detailed instructions, see the [Sampling Guide](docs/sample_provided_data.md).

**Outputs**:

A new directory `{exp_name}_{timestamp}` will be created in `outputs_examples/`:

- `{exp_name}_{timestamp}_SDF/`: Final generated molecules (SDF or PDB format).
- `SDF/`: Sampling trajectories (if enabled).
- `gen_info.csv`: Metadata and confidence scores (`cfd_traj`).
- `log.txt`: Execution logs.


## Documentation 

### [Setup Guide](docs/setup.md)
- Installation via Conda or manual Pip.
- Instructions for downloading weights and test data from Zenodo.
- Directory structure explainer.

### [Sampling with Provided Data](docs/sample_provided_data.md)
- **User Guide** for running inference/generation on provided examples.
- Catalog of configuration files in `configs/sample/examples/` (Docking, SBDD, Linking/Growing, Optimization, Peptide design, etc.).
- Guide to [Confidence Scoring](docs/sample_provided_data.md#confidence-scoring).
- Explanation of [Custom Tasks](docs/sample_provided_data.md#4-defining-custom-tasks) (noise groups, partitions, mapping).

### [Benchmarking & Test Sets](docs/sample_test_sets.md)
- Instructions for reproducing benchmark results (PoseBusters, PepBDB, GEOM, CSD, MOAD, PROTAC-DB).
- Specific commands and configs for each benchmark task.
- Evaluation tools for ranking poses.

### [Training Guide](docs/train.md)
- Instructions for training on the reduced demo set.
- Pointers for processing full datasets.

### [Raw Data Processing](docs/data_processing.md)
- Step-by-step instructions for processing raw training data from scratch.
- Required for full-dataset training (>500 GB).

---

### Additional References

- [Source Code Guide](docs/source_code_guide.md): Architecture overview and code walkthrough.
- [notebooks/](notebooks): Interactive Colab-ready notebooks.


##  License / Contributing / Citation

MIT License. See [LICENSE](LICENSE). 

Pull requests, bug reports, and feedback are very welcome — please feel free to submit a PR! This is a one-person maintained project, so any help is greatly appreciated!!!

Paper cite this article in press as Peng et al., Learning fundamental atom interactions for unified modeling of 3D molecule generation, *Cell* (2026), https://doi.org/10.1016/j.cell.2026.01.003

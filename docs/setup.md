# Installation & Data Setup

> [← Back to Documentation Index](README.md)

## 1. Environment Setup

### Option A: Conda (Recommended)

To set up the environment on Linux (with CUDA 11.7), use [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) to create a new environment `pxm` from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate pxm
```

> **Note:** If you have a different CUDA version, modify the pytorch-related package versions in `environment.yml` before creating the environment.

### Option B: Manual Installation (Pip)

If you need a custom setup (e.g., for **CUDA 12.6**), install dependencies manually:

```bash
# PyTorch for CUDA 12.6
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install pytorch-lightning
pip install torch_geometric

# PyG dependencies
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.6.0+cu126.html

# Bio/Chem informatics
pip install biopython==1.83 rdkit==2023.9.3 peptidebuilder==1.1.0
pip install openbabel==3.1.1.1  # or: conda install -c conda-forge openbabel -y

# Utilities
pip install lmdb easydict==1.9 numpy==1.24 pandas==1.5.2 scipy==1.10.1
pip install tensorboard  # for training only
```

## 2. Data & Model Weights

All training/test data and model weights are available on [Zenodo](https://zenodo.org/records/17801271).

### For Inference/Sampling (Required)
The `model_weights.tar.gz` archive contains trained checkpoints. Download and extract it:

```bash
wget https://zenodo.org/records/17801271/files/model_weights.tar.gz
tar -zxvf model_weights.tar.gz
```
*Creates: `data/trained_models/` containing the weights.*

*(Note: Simple example data is already included in `data/examples` inside this repo.)*

### For Benchmarking (Optional)
To run benchmarks on standard test sets (PoseBusters, CrossDocked, etc.), download `data_test.tar.gz`:

```bash
wget https://zenodo.org/records/17801271/files/data_test.tar.gz
tar -zxvf data_test.tar.gz
```
*Creates:*
*   `data/test`: Benchmark metadata
*   `data/csd`, `data/geom`, `data/moad`, etc.: Processed test sets.

### For Training (Optional)
To train on the reduced demonstration dataset, download `data_train_processed.tar.gz`:

```bash
wget https://zenodo.org/records/17801271/files/data_train_processed.tar.gz
tar -zxvf data_train_processed.tar.gz
```
*Creates: `data_train` with reduced training sets.*

**Full Dataset Training:**
The complete processed training data (>500 GB) is not provided as a single archive. To train with the full dataset, follow the instructions in [Raw Data Processing](data_processing.md) to process the raw data yourself.

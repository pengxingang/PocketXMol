# Raw Data Processing for Full Training

> [← Back to Documentation Index](README.md)

> This guide walks through preparing and processing the raw training data from scratch.
> Only needed if you want to train PocketXMol on the **full** dataset (>500 GB processed).
> For the reduced demo training set, see [Training Guide](train.md).

## 1. Download Raw Data

All raw data files are hosted on [Zenodo](https://zenodo.org/records/17801271).

1.  **Training metadata & splits**: Download [`data_train_raw.tar.gz`](https://zenodo.org/records/17801271/files/data_train_raw.tar.gz?download=1) and extract it to the project root.
2.  **Raw molecule/protein/peptide files**: Download [`raw_files.zip`](https://zenodo.org/records/17801271/files/raw_files.zip?download=1), unzip it, and extract each dataset archive into the corresponding `data_train/{db}/` directory.
3.  **Uni-Mol data** (114.76 GB, separate download): Download the [molecular pretrain data](https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/pretrain/ligands.tar.gz) from the [Uni-Mol repository](https://github.com/deepmodeling/Uni-Mol/tree/main/unimol). Extract `ligands.tar.gz` into `data_train/unmi/files/`.

### Expected Directory Structure

After extraction, you should have the `data_train/` directory:

```
data_train/
├── geom/
│   ├── dfs/
│   │   └── meta_uni.csv
│   └── mols/                    # from raw_files/geom.tar.gz
├── qm9/
│   ├── dfs/
│   │   └── meta_uni.csv
│   └── mols/                    # from raw_files/qm9.tar.gz
├── unmi/
│   ├── dfs/
│   │   └── meta_uni.csv
│   └── files/                   # from Uni-Mol ligands.tar.gz
├── csd/
│   ├── dfs/
│   │   └── meta_filter_w_pocket.csv
│   └── files/                   # from raw_files/csd.tar.gz
│       ├── proteins/
│       └── mols/
├── pbdock/
│   ├── dfs/
│   │   └── meta_filter_w_pocket.csv
│   └── files/                   # from raw_files/pbdock.tar.gz
│       ├── proteins/
│       └── mols/
├── moad/
│   ├── dfs/
│   │   └── meta_uni.csv
│   └── files/                   # from raw_files/moad.tar.gz
│       ├── proteins/
│       └── mols/
├── cremp/
│   ├── dfs/
│   │   └── meta_uni.csv
│   └── mols/                    # from raw_files/cremp.tar.gz
├── apep/
│   ├── dfs/
│   │   └── meta_uni.csv
│   └── files/                   # from raw_files/apep.tar.gz
│       ├── proteins/
│       ├── mols/
│       └── peptides/
├── pepbdb/
│   ├── dfs/
│   │   └── meta_filter.csv
│   └── files/                   # from raw_files/pepbdb.tar.gz
│       ├── proteins/
│       ├── mols/
│       └── peptide/
└── assemblies/
    └── split_train_val.csv      # train/val split
```

## 2. Process Each Database

Run the following commands **in order** for each database. All commands should be executed from the project root directory.

### GEOM-Drug
```bash
python process/geom/process_mols.py
python process/process_torsional_info.py --db_name geom
python process/process_decompose_info.py --db_name geom
```
*Output: `data_train/geom/lmdb/`*

### QM9
```bash
python process/qm9/process_mols.py
python process/process_torsional_info.py --db_name qm9
python process/process_decompose_info.py --db_name qm9
```
*Output: `data_train/qm9/lmdb/`*

### Uni-Mol (unmi)
```bash
python process/unmi/process_mols.py
python process/process_torsional_info.py --db_name unmi
python process/process_decompose_info.py --db_name unmi
```
*Output: `data_train/unmi/lmdb/`*

### CrossDocked (csd)
```bash
python process/csd/extract_pockets.py
python process/csd/process_pocmol.py
python process/process_torsional_info.py --db_name csd
python process/process_decompose_info.py --db_name csd
```
*Output: `data_train/csd/lmdb/` and `data_train/csd/files/pockets10/`*

### PDBbind (pbdock)
```bash
python process/pbdock/extract_pockets.py
python process/pbdock/process_pocmol.py
python process/process_torsional_info.py --db_name pbdock
python process/process_decompose_info.py --db_name pbdock
```
*Output: `data_train/pbdock/lmdb/` and `data_train/pbdock/files/pockets10/`*

### Binding MOAD
```bash
python process/moad/extract_pockets.py
python process/moad/process_pocmol.py
python process/process_torsional_info.py --db_name moad
python process/process_decompose_info.py --db_name moad
```
*Output: `data_train/moad/lmdb/` and `data_train/moad/files/pockets10/`*

### CREMP
```bash
python process/process_mols.py --db_name cremp
```
*Output: `data_train/cremp/lmdb/`*

### AlphaFoldDB-Peptide (apep)
```bash
python process/extract_pockets.py --db_name apep
python process/process_pocmol.py --db_name apep
python process/process_peptide_allinone.py --db_name apep
python process/process_torsional_info.py --db_name apep
python process/process_decompose_info.py --db_name apep
```
*Output: `data_train/apep/lmdb/` and `data_train/apep/files/pockets10/`*

### PepBDB
```bash
python process/process_pocmol_allinone.py --db_name pepbdb
python process/process_peptide_allinone.py --db_name pepbdb
```
*Output: `data_train/pepbdb/lmdb/` and `data_train/pepbdb/files/pockets10/`*

## 3. Generate Training Split

After all databases are processed, run:

```bash
python process/make_assembly_lmdb.py
```

This generates the training/validation split LMDB in `data_train/assemblies/`, which is used directly by the training script.

## 4. Start Training

Update the training config to point to your processed data and run training as described in [Training Guide](train.md).

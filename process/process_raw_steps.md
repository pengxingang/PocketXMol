
> This note contains the scripts to process the raw data for training PocketXMol.

# Step 1: Prepare raw training data
1. The raw data and files are on [Zenodo](https://zenodo.org/records/17801271).
    - Download the file [`data_train_raw.tar.gz`](https://zenodo.org/records/17801271/files/data_train_raw.tar.gz?download=1) and extract it to the root directory of the project.
    - Download the raw protein/molecule/peptide files for each database (except for the Uni-Mol data) zipped in the file [`raw_files.zip`](https://zenodo.org/records/17801271/files/raw_files.zip?download=1), unzip it, and extract datasets to the corresponding directory.
2. The raw Uni-Mol data is very large (114.76GB). You can download the [molecular pretrain data](https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/pretrain/ligands.tar.gz) directly from the original [Uni-Mol repository](https://github.com/deepmodeling/Uni-Mol/tree/main/unimol). Then extract the `ligands.tar.gz` file to the `data_train/unmi/files` directory.

Then you will get the `data_train` directory with the following structure:
```bash
data_train
├── geom
│   ├── dfs
│   │   └── meta_uni.csv
│   └── mols  # extracted from raw_files/geom.tar.gz
├── qm9
│   ├── dfs
│   │   └── meta_uni.csv
│   └── mols  # extracted from raw_files/qm9.tar.gz
├── unmi
│   ├── dfs
│   │   └── meta_uni.csv
│   └── files  # extracted downloaded ligands.tar.gz here
├── csd
│   ├── dfs
│   │   └── meta_filter_w_pocket.csv
│   └── files  # extracted from raw_files/csd.tar.gz
│       ├── proteins
│       └── mols
├── pbdock
│   ├── dfs
│   │   └── meta_filter_w_pocket.csv
│   └── files  # extracted from raw_files/pbdock.tar.gz
│       ├── proteins
│       └── mols
├── moad
│   ├── dfs
│   │   └── meta_uni.csv
│   └── files  # extracted from raw_files/moad.tar.gz
│       ├── proteins
│       └── mols
├── cremp
│   ├── dfs
│   │   └── meta_uni.csv
│   └── mols # extracted from raw_files/cremp.tar.gz
├── apep
│   ├── dfs
│   │   └── meta_uni.csv
│   └── files  # extracted from raw_files/apep.tar.gz
│       ├── proteins
│       ├── mols
│       └── peptides
├── pepbdb
│   ├── dfs
│   │   └── meta_filter.csv
│   └── files  # extracted from raw_files/pepbdb.tar.gz
│       ├── proteins
│       ├── mols
│       └── peptide
└── assemblies  # train/val split for training
    └── split_train_val.csv
```


# Step 2: Process raw data
The following steps are required to process the raw data files. The Python commands for each database should be run in order.
## GEOM-Drug (geom)
Run
```bash
python process/geom/process_mols.py
python process/process_torsional_info.py --db_name geom
python process/process_decompose_info.py --db_name geom
```
and you will get the processed data (lmdb) in the `data_train/geom/lmdb` directory.

## QM9 (qm9)
Run
```bash
python process/qm9/process_mols.py
python process/process_torsional_info.py --db_name qm9
python process/process_decompose_info.py --db_name qm9
```
and you will get the processed data (lmdb) in the `data_train/qm9/lmdb` directory.


## Uni-Mol data (unmi)
Run
```bash
python process/unmi/process_mols.py
python process/process_torsional_info.py --db_name unmi
python process/process_decompose_info.py --db_name unmi
```
and you will get the processed data (lmdb) in the `data_train/unmi/lmdb` directory.

## CrossDocked (csd)
Run
```bash
python process/csd/extract_pockets.py
python process/csd/process_pocmol.py
python process/process_torsional_info.py --db_name csd
python process/process_decompose_info.py --db_name csd
```
and you will get the processed data (lmdb) in the `data_train/csd/lmdb` directory and pocket data in the `data_train/csd/files/pockets10` directory.

## PDBbind (pbdock)
Run
```bash
python process/pbdock/extract_pockets.py
python process/pbdock/process_pocmol.py
python process/process_torsional_info.py --db_name pbdock
python process/process_decompose_info.py --db_name pbdock
```
and you will get the processed data (lmdb) in the `data_train/pbdock/lmdb` directory and pocket data in the `data_train/pbdock/files/pockets10` directory.

## Binding MOAD (moad)
Run
```bash
python process/moad/extract_pockets.py
python process/moad/process_pocmol.py
python process/process_torsional_info.py --db_name moad
python process/process_decompose_info.py --db_name moad
```
and you will get the processed data (lmdb) in the `data_train/moad/lmdb` directory and pocket data in the `data_train/moad/files/pockets10` directory.


## CREMP (cremp)
Run
```bash
python process/process_mols.py --db_name cremp
```
and you will get the processed data (lmdb) in the `data_train/cremp/lmdb` directory.

## AlphaFoldDB-Peptide (apep)
Run
```bash
python process/extract_pockets.py --db_name apep
python process/process_pocmol.py --db_name apep
python process/process_peptide_allinone.py --db_name apep
python process/process_torsional_info.py --db_name apep
python process/process_decompose_info.py --db_name apep
```
and you will get the processed data (lmdb) in the `data_train/apep/lmdb` directory and pocket data in the `data_train/apep/files/pockets10` directory.

## PepBDB (pepbdb)
Run
```bash
python process/process_pocmol_allinone.py --db_name pepbdb
python process/process_peptide_allinone.py --db_name pepbdb
```
and you will get the processed data (lmdb) in the `data_train/pepbdb/lmdb` directory and pocket data in the `data_train/pepbdb/files/pockets10` directory.


## Training data split
Finall run
```bash
python process/make_assembly_lmdb.py
```
to generate the training/validation split data (lmdb) in the `data_train/assemblies` directory for model training.

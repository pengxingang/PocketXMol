# PocketXMol Source Code Guide

> [← Back to Documentation Index](README.md)

This guide is for users who want to dive into the PocketXMol source code. It explains the core structure, the train/sample workflows, and where to start when you need to trace a behavior or add a feature.

## Table of Contents

- [PocketXMol Source Code Guide](#pocketxmol-source-code-guide)
  - [Table of Contents](#table-of-contents)
  - [1. Overview](#1-overview)
  - [2. Core Files](#2-core-files)
  - [3. Workflows](#3-workflows)
    - [3.1 Training](#31-training)
    - [3.2 Sampling](#32-sampling)
  - [4. Key Concepts](#4-key-concepts)
    - [4.1 Data Structures](#41-data-structures)
    - [4.2 Transforms and Noisers](#42-transforms-and-noisers)
    - [4.3 Configuration System](#43-configuration-system)
  - [5. File Reference](#5-file-reference)
    - [5.1 `models/` — Neural Network Components](#51-models--neural-network-components)
    - [5.2 `utils/` — Transforms, Data, and Utilities](#52-utils--transforms-data-and-utilities)
    - [5.3 `scripts/` — Entry Points](#53-scripts--entry-points)

---

## 1. Overview

PocketXMol is a denoising diffusion system for generating molecules conditioned on protein pockets. The code is organized around four concerns:

- **Data and transforms**: Convert raw protein/molecule inputs into graph-based tensors, including featurization and task-specific transformations (set task prompt).
- **Noise adding**: Add noise to the inputs during training and sampling.
- **Model**: Predict denoised atom types, coordinates, and bonds.
- **Sampling/training**: Orchestrate noise schedules, task-specific transforms, and reconstruction.

**High-level data flow for sampling:**

```
Raw inputs (PDB/SDF/SMILES)
  → Featurize pocket + molecule (transforms)
  → Task transform (partition, constraints, sizes, set task prompt)
  → Noise schedule (train: add noise, sample: reverse noise)
  → Model forward pass (denoise predictions)
  → Reconstruction (construct RDKit/PDB)
  → Outputs (SDF/PDB + gen_info.csv)
```

**Repository map:**

| Directory | Purpose |
|---|---|
| [scripts/](../scripts/) | Entry points: training, sampling, confidence scoring |
| [models/](../models/) | Denoiser, graph encoders, loss definitions |
| [utils/](../utils/) | Transforms, noise schedules, reconstruction utilities |
| [configs/](../configs/) | Task and model YAML configurations |
| [evaluate/](../evaluate/) | Benchmark and evaluation scripts |
| [process/](../process/) | Raw data processing |

---

## 2. Core Files

If you only read a few files, start with these:

**Entry points:**
- [scripts/sample_use.py](../scripts/sample_use.py) — main sampling entry point (loads configs, extracts pocket, runs sampling loop, writes outputs).
- [scripts/train_pl.py](../scripts/train_pl.py) — PyTorch Lightning training entry point (`DataModule`, `ModelLightning`).
- [scripts/sample_drug3d.py](../scripts/sample_drug3d.py) — sampling script for benchmark small molecule tasks.
- [scripts/sample_pdb.py](../scripts/sample_pdb.py) — sampling script for benchmark peptide tasks.

**Model and sampling:**
- [models/maskfill.py](../models/maskfill.py) — main model `PMAsymDenoiser` (encodes pocket context + molecule graph, predicts denoised `node`, `pos`, `edge`).
- [models/sample.py](../models/sample.py) — sampling loop (`sample_loop3`), output splitting (`seperate_outputs2`), confidence scoring (`get_cfd_traj`).
- [models/loss.py](../models/loss.py) — loss registry (`individual_tasks`, `asymloss`).

**Transforms, noise, and reconstruction:**
- [utils/transforms.py](../utils/transforms.py) — transform registry (`@register_transforms`), featurizers (`FeaturizeMol`, `FeaturizePocket`), task transforms (`{TASK}Transform`).
- [utils/sample_noise.py](../utils/sample_noise.py) — sampling-side noisers and noise groups.
- [utils/reconstruct.py](../utils/reconstruct.py) — graph → RDKit/PDB reconstruction.
- [utils/dataset.py](../utils/dataset.py) — dataset and dataloader definitions (LMDB, multi-task).
- [utils/info_level.py](../utils/info_level.py) — noise schedule implementations.

---

## 3. Workflows

### 3.1 Training

**Flow chart:**

```
 Load config (configs/train/*.yml)
         │
         ▼
 Build DataModule (scripts/train_pl.py)
    ┌────┴────┐
    ▼         ▼
 Build     Load data
 transforms  from LMDB
 (utils/)   (utils/dataset.py)
    └────┬────┘
         ▼
 Forward diffusion / add noise (utils/sample_noise.py)
         │
         ▼
 Model forward (models/maskfill.py)
         │
         ▼
 Compute loss (models/loss.py)
         │
         ▼
 Backprop + checkpoint (PyTorch Lightning)
```

**Step by step:**

1. **Load config** — `scripts/train_pl.py` reads a training YAML (`configs/train/`).
2. **Build DataModule** — `DataModule` in `scripts/train_pl.py` selects datasets and task weights; data loaded from LMDB via `ForeverTaskDataset` in `utils/dataset.py`.
3. **Build transforms** — composed into a single pipeline: featurizers (`FeaturizePocket`, `FeaturizeMol`) → task transform (`{TASK}Transform`) → noiser (`{TASK}SampleNoiser`). In training, the noiser is the last element in the transform list, so each sample is noise-injected before batching.
4. **Forward diffusion** — the noiser (`{TASK}SampleNoiser` from `utils/sample_noise.py`) randomly samples a noise level, then adds noise to produce `node_in`, `pos_in`, `halfedge_in`.
5. **Model forward** — `PMAsymDenoiser` in `models/maskfill.py` predicts denoised atom types, coordinates, and bonds.
6. **Loss** — task-specific losses computed via registry in `models/loss.py`.
7. **Checkpoint/logging** — saved to `data/trained_models/{exp_name}/` by PyTorch Lightning.

**Detailed call-graph:**

```
scripts/train_pl.py
│
├── load_config(args.config)                          # utils/misc.py
│
├── DataModule(config)                                # scripts/train_pl.py
│   ├── Build transform list (Compose):
│   │   ├── FeaturizePocket(config)                   # utils/transforms.py — featurize protein residues
│   │   ├── FeaturizeMol(config)                      # utils/transforms.py — featurize molecule atoms/bonds
│   │   ├── TaskTransform(config)                     # utils/transforms.py — e.g. ConfTransform, MaskfillTransform
│   │   └── SampleNoiser(config)                      # utils/sample_noise.py — add diffusion noise
│   │       (Note: in training, the noiser is part of the transform list.
│   │        Each sample is noise-injected before batching, so training_step
│   │        receives batch with node_in/pos_in/halfedge_in already set.)
│   ├── ForeverTaskDataset(db_list, transforms)       # utils/dataset.py
│   │   └── LMDBDatabase(path)                        #   load pre-processed data from LMDB
│   └── DataLoader(dataset, batch_size, ...)          # PyTorch
│
├── ModelLightning(config, in_dims)                   # scripts/train_pl.py
│   ├── PMAsymDenoiser(config.model)                  # models/maskfill.py
│   │   ├── ContextNodeEdgeNet (pocket encoder)       # models/graph_context.py
│   │   ├── AtomEmbeddingVN + BondEmbedding           # models/embedding.py
│   │   └── NodeEdgeNet (denoiser backbone)           # models/graph.py
│   └── loss_fn = registered_loss(config)             # models/loss.py
│
├── Trainer.fit(model, datamodule)                    # PyTorch Lightning
│   └── training_step(batch)
│       ├── model.forward(batch) → predictions        # models/maskfill.py
│       └── loss_fn(predictions, targets)             # models/loss.py
│
└── Checkpoints → data/trained_models/{exp}/checkpoints/
```

### 3.2 Sampling

**Flow chart:**

```
 Load configs (--config_task + --config_model)
         │
         ▼
 Merge with training config (scripts/sample_use.py)
         │
         ▼
 Prepare inputs: pocket extraction + ligand parsing (utils/parser.py)
         │
         ▼
 Build transforms + noiser (utils/transforms.py, utils/sample_noise.py)
         │
         ▼
 ┌─► Reverse diffusion loop (models/sample.py)
 │       │
 │       ▼
 │   Add noise at current level (utils/sample_noise.py)
 │       │
 │       ▼
 │   Model forward (models/maskfill.py)
 │       │
 │       ▼
 │   Update molecule from predictions
 │       │
 └───────┘  (repeat for each step)
         │
         ▼
 Reconstruct molecules (utils/reconstruct.py)
         │
         ▼
 Write SDF/PDB + gen_info.csv
```

**Step by step:**

1. **Load configs** — `scripts/sample_use.py` reads `--config_task` and `--config_model` YAMLs; training config loaded from the checkpoint directory via `utils/misc.py:load_train_config_from_ckpt`.
2. **Merge configs** — sampling config overrides training transforms/noise fields when keys already exist (`scripts/sample_use.py`).
3. **Prepare inputs** — pocket extraction (`utils/parser.py:PDBProtein`) and input ligand processing (`scripts/sample_use.py:get_input_data`).
4. **Build transforms** — featurize pocket/molecule, optionally add `VariableMolSize`/`VariableScSize` (for SBDD/peptide design), then apply task transform (`utils/transforms.py`). Build sampling noiser separately via `utils/sample_noise.py:get_sample_noiser` (in sampling, the noiser is **not** in the transform list — it is called iteratively inside the loop).
5. **Reverse diffusion (iterations)** — `models/sample.py:sample_loop3` iteratively denoises. At each step the noiser injects noise at the current level, the model predicts denoised outputs, and the molecule is updated. Noise schedule driven by `utils/info_level.py:MolInfoLevel`.
6. **Reconstruct** — `utils/reconstruct.py:reconstruct_from_generated_with_edges` (small molecules) or `reconstruct_pdb_from_generated_fold` (peptides).
7. **Write outputs** — SDF/PDB files + `gen_info.csv` written to `{outdir}/{exp_name}_{timestamp}/`.

**Detailed call-graph:**

```
scripts/sample_use.py
│
├── load_config(args.config_task, args.config_model)  # utils/misc.py
├── load_train_config_from_ckpt(ckpt_path)            # utils/misc.py
├── Merge: sample config overrides train config       #   (transform/noise keys)
│
├── get_input_data(protein_path, input_ligand, ...)   # scripts/sample_use.py
│   ├── PDBProtein(pdb_path)                          # utils/parser.py
│   │   └── .query_residues_ligand(radius=...)        #   extract pocket residues
│   └── parse input ligand (SDF/PDB/SMILES/pepseq)    # utils/parser.py / RDKit
│
├── Build transforms (Compose)                        # utils/transforms.py
│   ├── FeaturizePocket → pocket graph tensors
│   ├── FeaturizeMol → molecule graph tensors
│   ├── (VariableMolSize / VariableScSize)             #   only for SBDD / peptide design
│   └── TaskTransform → set partition, fixed masks, task prompt
│
├── get_sample_noiser(noise_config)                   # utils/sample_noise.py
│   └── e.g. DockSamplNoiser / MaskfillSampleNoiser
│       └── MolInfoLevel(level_config)                # utils/info_level.py
│
├── Load PMAsymDenoiser from checkpoint               # models/maskfill.py
│
├── sample_loop3(model, batch, noiser, ...)           # models/sample.py
│   └── for step in noiser.steps_loop():
│       ├── noiser.add_noise(batch, step)             #   inject noise at current level
│       ├── model.forward(batch) → predictions        # models/maskfill.py
│       ├── noiser.update(batch, predictions)          #   update molecule from predictions
│       └── correct_pos_batch(batch) [optional]       # models/corrector.py
│
├── seperate_outputs2(batch) → per-molecule dicts      # models/sample.py
├── get_cfd_traj(outputs) → confidence scores          # models/sample.py
│
├── Reconstruct
│   ├── reconstruct_from_generated_with_edges(...)    # utils/reconstruct.py (small mol)
│   └── reconstruct_pdb_from_generated_fold(...)      # utils/reconstruct.py (peptide)
│
└── Write outputs
    ├── {outdir}/{exp_name}_{timestamp}_SDF/*.sdf
    └── gen_info.csv
```

---

## 4. Key Concepts

### 4.1 Data Structures

- **`PocketMolData`** (`utils/data.py`): PyG `Data` subclass holding molecule and pocket tensors. This is the main container used throughout sampling; it carries both pocket and molecule fields.
- **Batch fields** (PyG `Batch`): molecules are batched via PyTorch Geometric. Key fields:
  - `node_type`, `node_pos`: atom types and 3D coordinates, with `node_type_batch` for graph-level indexing.
  - `halfedge_type`, `halfedge_index`: bond types stored as directed half-edges, with `halfedge_type_batch`.
- **Task prompt fields**: binary masks that tell the model what is fixed:
  - `fixed_node`: atom type is fixed (do not change).
  - `fixed_pos`: coordinate is fixed (do not move).
  - `fixed_halfedge` / `fixed_halfdist`: bond type / distance is fixed.
  - `is_peptide`: peptide prompt indicator.
- **Pocket fields**: `pocket_atom_feature`, `pocket_pos`, `pocket_pos_batch`, etc. Pocket is always treated as fixed context.

**Debugging tip:** dump `data.keys()` and tensor shapes after each transform to see how fields evolve through the pipeline.

### 4.2 Transforms and Noisers

The two most important pipeline stages are **task transforms** (set the task prompt) and **noisers** (add diffusion noise). Both live in `utils/` and are composed into a single data pipeline: `FeaturizePocket → FeaturizeMol → TaskTransform → SampleNoiser`.

**Task transforms** (`utils/transforms.py`) — each `{TASK}Transform` class calls `sample_setting()` to randomly pick a `task_setting` dict, then sets the `fixed_*` masks accordingly. For example, `ConfTransform` (registered as both `dock` and `conf`) picks a perturbation mode (`free`/`rigid`/`torsional`/`flexible`); `MaskfillTransform` picks a decomposition strategy and perturbation level for the known fragment; `PepdesignTransform` picks a mode (`full`/`sc`/`packing`).

The result of a task transform is a set of binary masks (`fixed_node`, `fixed_pos`, `fixed_halfedge`, `fixed_halfdist`, `is_peptide`) stored in the `data` object.

**Noisers** (`utils/sample_noise.py`) — each `{TASK}SampleNoiser` reads `task_setting` and the `fixed_*` masks, then:
1. Samples a noise level via `MolInfoLevel` (`utils/info_level.py`): during training, level is random; during sampling, level follows a schedule.
2. Calls `MolPrior.add_noise()` (`utils/prior.py`) to inject Gaussian noise (positions) and categorical noise (atom/bond types).
3. Writes the noisy inputs to `node_in`, `pos_in`, `halfedge_in` — these are what the model consumes.

Fixed atoms (where `fixed_*=1`) receive no noise or reduced noise, depending on the task setting.

### 4.3 Configuration System

Sampling uses a **task config** plus a **model config**:

- Task configs: [configs/sample/examples/](../configs/sample/examples/)
- Testset task configs: [configs/sample/test/](../configs/sample/test)
- Model config: [configs/sample/pxm.yml](../configs/sample/pxm.yml)

In sampling, transform/noise values in the task config override the training config when the key already exists. New keys are not injected automatically.


## 5. File Reference

This section lists important files in `models/`, `utils/`, and `scripts/` for readers who need to locate specific functionality.

### 5.1 `models/` — Neural Network Components

| File | Purpose | Key Classes / Functions |
|---|---|---|
| `maskfill.py` | **Main model** (please ignore the confusing file name) — denoiser for pocket–molecule systems | `PMAsymDenoiser` |
| `sample.py` | Reverse diffusion sampling loop and output splitting | `sample_loop3`, `seperate_outputs2`, `get_cfd_traj` |
| `loss.py` | Loss registry and multi-task loss computation | `@register_loss`, `individual_tasks`, `asymloss` |

### 5.2 `utils/` — Transforms, Data, and Utilities

| File | Purpose | Key Classes / Functions |
|---|---|---|
| `transforms.py` | **Transform registry** — featurizers, task transforms, variable-size transforms | `@register_transforms`, `FeaturizeMol`, `FeaturizePocket`, `{TASK}Transform`, `CustomTransform` |
| `sample_noise.py` | Sampling-side noise controllers for all task types | `get_sample_noiser`, `{TASK}SampleNoiser`, `CustomSampleNoiser` |
| `info_level.py` | Noise schedule: step → information level mapping | `MolInfoLevel`, `AdvanceScaler` |
| `reconstruct.py` | Graph → molecule reconstruction (RDKit / PDB) | `reconstruct_from_generated_with_edges`, `reconstruct_pdb_from_generated_fold` |
| `data.py` | Core data containers (PyG `Data` subclasses) | `PocketMolData` |
| `dataset.py` | Dataset and dataloader definitions (LMDB, multi-task) | `ForeverTaskDataset`, `UseDataset`, `LMDBDatabase` |
| `parser.py` | PDB protein/ligand parsing and pocket extraction | `PDBProtein`, `PDBLigand`, `parse_pdb_peptide` |
| `misc.py` | Config loading, logging, seeding, timeout utilities | `load_config`, `seed_all`, `get_logger` |
| `prior.py` | Noise prior distribution definitions | `MolPrior` |
| `motion.py` | Rigid-body and torsional rotation utilities | `apply_axis_angle_rotation`, `apply_torsional_rotation_multiple_domains` |

### 5.3 `scripts/` — Entry Points

| File | Purpose |
|---|---|
| `sample_use.py` | Main sampling script for user-provided data |
| `sample_drug3d.py` | Batch sampling for benchmark small molecule tasks |
| `sample_pdb.py` | Batch sampling for benchmark peptide tasks |
| `train_pl.py` | PyTorch Lightning training entry point (`DataModule`, `ModelLightning`) |
| `believe.py` / `believe_use_pdb.py` | Post-hoc confidence scoring for generated molecules |


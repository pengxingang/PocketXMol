# Benchmarking on Test Sets

> [← Back to Documentation Index](README.md)

This guide details how to reproduce benchmark results on the provided test sets.

> **General Notes:**
> - **Hardware:** Verified on an 80G A100 GPU. Batch sizes in configs are optimized for this.
> - **Memory:** If you hit OOM, reduce batch size via command line: `--batch_size 100`.
> - **Runtime:** 1 ~ 6 hours per test set on a single A100.
> - **Output:** Results are saved under the directory passed by `--outdir`, with run folders named `{exp_name}_{timestamp}`.

## 1. Small Molecule Docking (PoseBusters)
**Dataset:** 428 protein-ligand pairs.

> **Naming note:** `poseboff` (PoseBusters official) is the internal dataset key used by configs for the PoseBusters benchmark.

```bash
python scripts/sample_drug3d.py \
    --config_task configs/sample/test/dock_poseboff/base.yml \
    --outdir outputs_test/dock_posebusters \
    --device cuda:0
```

**Variants (Configs in `configs/sample/test/dock_poseboff/`):**
- `base.yml`: Gaussian noise (Standard).
- `base_flex.yml`: Flexible noise.
- `prior_center.yml`: Prior knowledge of molecular center.
- `prior_bond_length.yml`: Prior knowledge of bond length.
- `prior_anchor.yml`: Prior knowledge of approximate anchor atom coordinate.
- `prior_fix_anchor.yml`: Fixed anchor atom coordinate.

### Ranking
**1. Confidence Scoring**
Calculate `self_ranking` and `tuned_ranking` scores:

```bash
python scripts/believe.py \
    --exp_name base_pxm \
    --result_root outputs_test/dock_posebusters \
    --config configs/sample/confidence/tuned_cfd.yml \
    --device cuda:0
```

**2. Ranking**
Generate final ranking CSVs:

```bash
python scripts/rank_pose.py \
    --exp_name base_pxm \
    --result_root outputs_test/dock_posebusters \
    --db poseboff
```

## 2. Peptide Docking (PepBDB)
**Dataset:** 79 protein-peptide pairs.

```bash
python scripts/sample_pdb.py \
    --config_task configs/sample/test/dock_pepbdb/base.yml \
    --outdir outputs_test/dock_pepbdb \
    --device cuda:0
```

**Variants (Configs in `configs/sample/test/dock_pepbdb/`):**
- `base.yml`: Gaussian noise (default).
- `base_flex.yml`: Flexible noise.
- `prior_fix_anchor.yml`: Fixed anchor atom coordinate.
- `prior_fix_first_residue.yml`: Fixed first residue atom coordinates.
- `prior_fix_terminal_residue.yml`: Fixed both terminal residue atom coordinates.
- `prior_fix_backbone.yml`: Fixed backbone atom coordinates.

## 3. Molecular Conformation (GEOM)
**Dataset:** 199 molecules.

```bash
python scripts/sample_drug3d.py \
    --config_task configs/sample/test/conf_geom/base.yml \
    --outdir outputs_test/conf_geom \
    --device cuda:0
```

## 4. SBDD (Structure-Based Drug Design)
**Dataset:** 100 protein pockets (CrossDocked/CSD).

```bash
python scripts/sample_drug3d.py \
    --config_task configs/sample/test/sbdd_csd/base.yml \
    --outdir outputs_test/sbdd_csd \
    --device cuda:0
```
**Variants (Configs in `configs/sample/test/sbdd_csd/`):**
- `base.yml`: Refine-based sampling strategy (default).
- `ar.yml`: Autoregressive-like sampling strategy.
- `simple.yml`: One generation round, no confidence scores.
- `base_mol_size.yml`: Refine-based with molecular sizes from reference molecules.

## 5. De Novo 3D Molecule Generation (GEOM-Drug)
**Dataset:** Generate molecules with size distribution matching GEOM-Drug validation set.

```bash
python scripts/sample_drug3d.py \
    --config_task configs/sample/test/denovo_geom/base.yml \
    --outdir outputs_test/denovo_geom \
    --device cuda:0
```

**Variants (Configs in `configs/sample/test/denovo_geom/`):**
- `base.yml`: Refine-based sampling strategy (default).
- `ar.yml`: Autoregressive-like sampling strategy.
- `simple.yml`: One generation round, no confidence scores.

## 6. Fragment Linking (MOAD & PROTAC-DB)

**MOAD (416 pairs):**
```bash
python scripts/sample_drug3d.py \
    --config_task configs/sample/test/linking_moad/known_connect.yml \
    --outdir outputs_test/linking_moad \
    --device cuda:0
```
**Variants (Configs in `configs/sample/test/linking_moad/`):**
- `known_connect.yml`: Known connecting atoms of fragments.
- `unknown_connect.yml`: Unknown connecting atoms.

**PROTAC-DB (43 pairs):**
```bash
python scripts/sample_drug3d.py \
    --config_task configs/sample/test/linking_protacdb/fixed_fragpos.yml \
    --outdir outputs_test/linking_protacdb \
    --device cuda:0
```
**Variants (Configs in `configs/sample/test/linking_protacdb/` — all assume known connecting atoms):**
- `fixed_fragpos.yml`: Fixed fragment poses.
- `unfixed_lv0.yml` – `unfixed_lv4.yml`: Unfixed fragment poses with increasing levels of perturbation noise (lv0 = smallest).

## 7. Fragment Growing (CSD)
**Dataset:** 53 fragment-protein pairs.

```bash
python scripts/sample_drug3d.py \
    --config_task configs/sample/test/growing_csd/base.yml \
    --outdir outputs_test/growing_csd \
    --device cuda:0
```

## 8. Peptide Design (PepBDB)

**De Novo Design:**
```bash
python scripts/sample_pdb.py \
    --config_task configs/sample/test/pepdesign_pepbdb/base.yml \
    --outdir outputs_test/pepdesign_pepbdb \
    --device cuda:0
```

**Inverse Folding (Backbone fixed, design sequence):**
```bash
python scripts/sample_pdb.py \
    --config_task configs/sample/test/pepinv_pepbdb/base.yml \
    --outdir outputs_test/pepinv_pepbdb \
    --device cuda:0
```

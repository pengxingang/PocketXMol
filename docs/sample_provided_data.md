# User Guide: Sampling with Provided Data

> [← Back to Documentation Index](README.md)

This guide covers how to run various generation tasks using the example data provided in `data/examples`.

## Table of Contents

- [1. Quick Start](#1-quick-start)
- [2. Example Configurations](#2-example-configurations)
  - [2.1 Docking](#21-docking)
  - [2.2 Small Molecule Design](#22-small-molecule-design)
  - [2.3 Peptide Design](#23-peptide-design)
- [3. Configuration Reference](#3-configuration-reference)
  - [3.1 Sampling Parameters (`sample`)](#31-sampling-parameters-sample)
  - [3.2 Input Data (`data`)](#32-input-data-data)
  - [3.3 Feature Overrides (`transforms`, optional)](#33-feature-overrides-transforms-optional)
    - [3.3.1 Denoising space center](#331-denoising-space-center)
    - [3.3.2 Generated molecule size — small molecule](#332-generated-molecule-size--small-molecule)
    - [3.3.3 Generated side-chain size — peptide](#333-generated-side-chain-size--peptide)
  - [3.4 Task Definition (`task`)](#34-task-definition-task)
    - [3.4.1 Task: `dock`](#341-task-dock)
    - [3.4.2 Task: `sbdd`](#342-task-sbdd)
    - [3.4.3 Task: `maskfill`](#343-task-maskfill-fragment-linking--growing--partial-optimization)
    - [3.4.4 Task: `pepdesign`](#344-task-pepdesign)
  - [3.5 Noise Configuration (`noise`)](#35-noise-configuration-noise)
    - [3.5.1 Common fields](#351-common-fields)
    - [3.5.2 Noise: `dock`](#352-noise-dock)
    - [3.5.3 Noise: `sbdd` (simple mode)](#353-noise-sbdd-simple-mode)
    - [3.5.4 Noise: `maskfill`](#354-noise-maskfill-two-groups-part1part2)
    - [3.5.5 Noise: `pepdesign`](#355-noise-pepdesign-two-groups-bbsc)
    - [3.5.6 Noise for optimization tasks](#356-noise-for-optimization-tasks)
    - [3.5.7 Noise schedule reference](#357-noise-schedule-reference)
- [4. Defining Custom Tasks](#4-defining-custom-tasks)
  - [4.1 How to define a custom task](#41-how-to-define-a-custom-task)

---

## 1. Quick Start

We provide ready-to-use example configs for supported tasks — see [Section 2](#2-example-configurations) for the full catalog.

Run the sampling script with a task configuration file:

```bash
python scripts/sample_use.py \
    --config_task configs/sample/examples/dock_smallmol.yml \
    --config_model configs/sample/pxm.yml \
    --outdir outputs_examples \
    --device cuda:0
```

**CLI arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config_task` | Yes | — | Path to the task configuration YAML (see [Section 3](#3-configuration-reference)) |
| `--config_model` | No | `configs/sample/pxm.yml` | Model config (checkpoint path, architecture). Usually no need to change. |
| `--outdir` | No | `./outputs_use` | Root output directory |
| `--device` | No | `cuda:0` | Device (e.g., `cuda:0`, `cpu`) |
| `--batch_size` | No | `0` (use config) | Override `sample.batch_size` from config |

(If you encounter OOM errors, reduce `batch_size` in the config YAML or override it: `--batch_size 50`)

**Outputs:** Results are saved in `outputs_examples/{exp_name}_{timestamp}/`, where
- `exp_name` is the combination of file names of `config_task` and `config_model` (default: pxm).
- timestamp is in `YYYYMMDD_HHMMSS` format.

The output directory contains:
>   - `*_SDF/`: Generated molecules.
>   - `SDF/`: Sampling trajectories (if enabled).
>   - `gen_info.csv`: Metadata and confidence scores. The self-confidence score is in the `cfd_traj` column.



**Confidence Scoring**

The self-confidence score exists in the `cfd_traj` column of `gen_info.csv`. For tuned ranking scores, you can compute a tuned ranking score for generated molecules:

```bash
python scripts/believe_use_pdb.py \
    --exp_name pepdesign_pxm \
    --result_root outputs_use \
    --config configs/sample/confidence/tuned_cfd.yml \
    --device cuda:0
```

- **`--exp_name`**: Partial match for your experiment directory name (e.g., `pepdesign_pxm` matches `pepdesign_pxm_20241201_211051`). The script searches `result_root` for directories containing this string.
- **`--config`**: Choose `tuned_cfd.yml` (recommended) or `flex_cfd.yml`.
- **Output**: Scores are saved to `ranking/*.csv` inside the experiment folder.

---

## 2. Example Configurations

Example configs are located in `configs/sample/examples/`.

### 2.1 Docking
| Config File | Description |
|---|---|
| `dock_smallmol.yml` | Dock a small molecule to a protein pocket |
| `dock_smallmol_flex.yml` | Dock a small molecule using flexible noise |
| `dock_smallmol_84663.yml` | Dock the molecule 84663 to caspase-9 |
| `dock_pep.yml` | Dock a peptide to a protein pocket |
| `dock_pep_fix_some.yml` | Dock with fixed coordinates for specific atoms |
| `dock_pep_know_some.yml` | Dock with constrained (known) coordinates for some atoms |

### 2.2 Small Molecule Design

**Structure-Based Drug Design (SBDD)**
| Config File | Description |
|---|---|
| `sbdd.yml` | Design drug-like molecules for a protein pocket |
| `sbdd_simple.yml` | Simplified SBDD (no refinement rounds) |

**Fragment Linking / Growing**
| Config File | Description |
|---|---|
| `linking_fixed_frags.yml` | Link fragments with fixed poses |
| `linking_fixed_frags_connecting.yml` | Link fragments with fixed poses and specified connecting atoms |
| `linking_unfixed_frags_unknown.yml` | Link fragments (poses unknown) |
| `linking_unfixed_frags_frominput.yml` | Link fragments, using input poses as initial poses |
| `linking_unfixed_frags_approx.yml` | Link fragments, using input poses as approximate (unfixed, allowing movement) |
| `linking_redesign_frags.yml` | Link fragments, allowing slight redesign (some atom types can change) of fragments |
| `growing_fixed_frag.yml` | Grow from a fixed fragment anchor |
| `growing_unfixed_frag_unknown.yml` | Grow from a fragment graph (pose unknown) |

**Molecular Optimization**
| Config File | Description |
|---|---|
| `opt_mol.yml` | Optimize full input molecules for a pocket |
| `opt_partial.yml` | Optimize specific fragments of input molecules |

### 2.3 Peptide Design
| Config File | Description |
|---|---|
| `pepdesign_denovo.yml` | De novo peptide generation |
| `pepdesign_fix_pos.yml` | Constrained: fixed positions and types for some residues |
| `pepdesign_fix_type.yml` | Constrained: fixed residue types (positions flexible) |
| `pepdesign_fix_pos_and_type.yml` | Constrained: some residues fixed, some types fixed (positions flexible) |
| `pepdesign_invfold.yml` | Inverse folding: design sequence for a given backbone |
| `pepdesign_invfold_bbflex.yml` | Inverse folding with backbone flexibility |
| `pepdesign_sc_pack.yml` | Side-chain packing (given backbone + types) |
| `pepdesign_opt.yml` | Peptide optimization: design peptides similar to input |

---

## 3. Configuration Reference


The configs are YAML files with **5 main blocks**:

| Block | Purpose | Key Question |
|-------|---------|--------------|
| `sample` | Runtime controls | *How many molecules to generate?* |
| `data` | Input protein & ligand | *What are the input structures?* |
| `transforms` | Feature engineering overrides | *How to preprocess?* (optional) |
| `task` | Task definition & mode | *What task to perform?* |
| `noise` | Diffusion noise configuration | *How to denoise?* |

In most cases, you only need to find a suitable task template config and modify `sample`, `data`, and `transforms`. The following subsections describe each block in detail.

---

### 3.1 Sampling Parameters (`sample`)

Controls how the sampling process runs.

```yaml
sample:
  seed: 2024
  batch_size: 100
  num_mols: 100
  save_traj_prob: 0.05
```

| Field | Type | Meaning |
|-------|------|---------|
| `seed` | `int` | Random seed |
| `batch_size` | `int` | Number of molecules processed in parallel. Can be overridden with `--batch_size` CLI flag. Larger → faster but more GPU memory |
| `num_mols` | `int` | Total number of molecules to generate |
| `save_traj_prob` | `float` ∈ [0, 1] | Fraction of molecules whose sampling trajectories are saved. `0` = none (fastest); `0.05` = 5% (for debugging); `1` = all |

**Tips:**
- Start with `batch_size: 100` and increase if memory permits.
- For quick testing, set `num_mols: 10`.

---

### 3.2 Input Data (`data`)

Specifies the input protein and ligand structures.

```yaml
data:
  protein_path: data/examples/dock/8C7Y_TXV_protein.pdb
  input_ligand: data/examples/dock/8C7Y_TXV_ligand_start_conf.sdf
  is_pep: false
  pocket_args:
    pocket_coord: [-8.2570, 85.1810, 19.0500]
    radius: 15
  pocmol_args:
    data_id: dock_8C7Y_TXV
    pdbid: 8C7Y
```

#### `data.protein_path`

- **Type:** `string` (file path)
- **Meaning:** Path to the protein PDB file. All chains in the PDB are considered for pocket extraction.
- **Tips:** Remove unnecessary entities (water, ions, co-crystallized ligands) from the PDB file.
- **Note:**
  - PocketXMol does not differentiate between single-chain and multi-chain proteins. All residues within the radius threshold are included regardless of chain ID.
  - If your PDB contains multiple chains, make sure only the relevant chains are present, or adjust the `radius` to focus on the desired binding interface.

#### `data.input_ligand`

- **Type:** `string` or `None`
- **Meaning:** Input ligand specification. The format depends on the task:

| Format | Description | Example | Use Case |
|--------|-------------|---------|----------|
| `.sdf` file path | Small molecule structure file | `data/ligand.sdf` | Docking, growing, linking, optimization |
| `SMILES` string | Chemical structure string | `c1ccccc1` | Docking |
| `.pdb` file path | Peptide structure file | `data/peptide.pdb` | Peptide docking, peptide design |
| `pepseq_<seq>` | Peptide sequence | `pepseq_DTVFALFW` | Peptide docking |
| `peplen_<n>` | Linear peptide length | `peplen_10` | De novo peptide design |
| `None` / omitted | No input ligand | — | De novo SBDD |

#### `data.is_pep`

- **Type:** `bool`
- **Meaning:** Whether the ligand is a peptide. Controls output format (`.pdb` vs `.sdf`).
- **Note:** Auto-determined from `input_ligand` if not explicitly set.

#### `data.pocket_args`

Defines how the protein pocket is extracted:

| Field | Type | Meaning |
|-------|------|---------|
| `ref_ligand_path` | `string` (file path) | Reference ligand (SDF/PDB) — pocket residues within `radius` of any atom. **Mutually exclusive** with `pocket_coord`. |
| `pocket_coord` | `list` [x, y, z] | Explicit pocket center coordinate. **Mutually exclusive** with `ref_ligand_path`. |
| `radius` | `float` (Å) | Distance threshold for pocket residue selection. Default: `10`. |
| `criterion` | `string` | Distance metric: `'center_of_mass'` (default) or `'min'` (closest atom). |

**Pocket extraction logic:** From the protein, all residues whose centers/closest-atoms are within `radius` Å of the reference (ligand atoms or coordinate point) are selected to form the pocket.

**Tips:**
- Use `radius: 10`–`15` with `ref_ligand_path`.
- Use `radius: 15`–`20` with `pocket_coord` (point-based, needs larger radius).
- If neither is set, falls back to `input_ligand` position.
- Visualize extracted pocket in output dir `{output_dir}/{exp_name}_{timestamp}_SDF/0_inputs/pocket_block.pdb` using PyMOL to verify.

#### `data.pocmol_args` (optional)

User-defined identifiers for experiment tracking:

| Field | Type | Meaning |
|-------|------|---------|
| `data_id` | `string` | Identifier for this input (e.g., `dock_8C7Y_TXV`) |
| `pdbid` | `string` | PDB code (e.g., `8C7Y`) |

---

### 3.3 Feature Overrides (`transforms`, optional)

Overrides data processing parameters. In most cases you can omit this block entirely and use defaults. Add specific fields only when necessary for three main purposes:

1. **Set the denoising space center** (`featurizer_pocket.center` or `featurizer.mol_as_pocket_center`) — controls where in 3D space molecules are generated.
2. **Control generated molecule size** (`variable_mol_size`) — for small-molecule design tasks (SBDD, fragment growing/linking) where the output atom count is variable.
3. **Control generated peptide side-chain size** (`variable_sc_size`) — for peptide design tasks where side-chain atom counts are sampled from a distribution.

Override only the specific sub-fields you need. Below we organize available overrides by **function** rather than by config field name.

#### 3.3.1 Denoising space center

By default, the denoising process is centered at the **pocket centroid**. 

You can override this to control where in 3D space molecules are generated. Two mutually exclusive options:


**Option A — Explicit coordinates** (`featurizer_pocket.center`):

Provide `[x, y, z]` coordinates directly. Default: centroid of pocket atoms.

```yaml
transforms:
  featurizer_pocket:
    center: [-8.2570, 85.1810, 19.0500]
```

Use when: 
- You know the exact binding site center, or for linker design (you can set to midpoint of two fragments).
- The pocket centroid is not ideal. For example, after generation, the molecules are consistently generated shifted to one side of the pocket. Setting an explicit center can correct this bias.

**Option B — Use input ligand center** (`featurizer.mol_as_pocket_center`):

Use the center of mass of `input_ligand` as the space center. Requires `input_ligand` with 3D coordinates (SDF/PDB).

```yaml
transforms:
  featurizer:
    mol_as_pocket_center: true
```

Use when:
- You have a known ligand pose and want to generate around that specific position.
- Fragment-based tasks (linking, growing, optimization) where the input molecule's position defines the generation region.

#### 3.3.2 Generated molecule size — small molecule

Field: `variable_mol_size`. Controls the atom count distribution for small-molecule design tasks (SBDD, growing, linking) where the output atom count is variable.

```yaml
transforms:
  variable_mol_size:
    name: variable_mol_size
    num_atoms_distri:
      strategy: mol_atoms_based   # Gaussian distribution
      mean:
        coef: 0     # coefficient for input mol size
        bias: 28    # constant term
      std:
        coef: 0
        bias: 2
      min: 5
    not_remove: [0, 1, 2, 3, 4, 5, 6]  # atom indices to keep (for fragment tasks)
```

**Size distribution:** Let $N_{\text{in}}$ = number of atoms in `input_ligand` (if provided):
- Mean: $\mu = \texttt{bias} + \texttt{coef} \times N_{\text{in}}$
- Std: $\sigma = \texttt{std.bias} + \texttt{std.coef} \times N_{\text{in}}$
- Sampled size $\sim \mathcal{N}(\mu, \sigma^2)$, clamped to $[\texttt{min}, \infty)$

**`not_remove` field:** List of atom indices (0-based) that must not be removed during variable-size sampling. **Critical for fragment growing/linking** — set this to the same atom indices as the fragment atoms in `task.transform.preset_partition.grouped_node_p1`.



#### 3.3.3 Generated side-chain size — peptide

Field: `variable_sc_size`. Controls side-chain atom count distribution for peptide design. Defaults generally work well — override only to bias toward more/fewer atoms per residue.

```yaml
transforms:
  variable_sc_size:
    name: variable_sc_size
    applicable_tasks: ['pepdesign']
    num_atoms_distri:
      mean: 8
      std:
        coef: 0.3817
        bias: 1.8727
```

**Why sample side-chain sizes?** In peptide design, only the peptide length (number of residues) is specified upfront — the total atom count depends on which residue types are eventually designed. This transform samples a target atom count during generation to guide the model toward peptides with a reasonable number of atoms.

**Size distribution:** Let $L$ = peptide length (number of residues):
- Mean: $\mu = \texttt{mean} \times L$
- Std: $\sigma = \texttt{std.bias} + \texttt{std.coef} \times L$
- Sampled size $\sim \mathcal{N}(\mu, \sigma^2)$

With the defaults (`mean: 8`, `std.coef: 0.3817`, `std.bias: 1.8727`), a 10-residue peptide targets ~80 ± 6 atoms. This is a hyper-parameter and we suggest not to change it unless you know what you are doing.

> **Note:** For constrained peptide design (fixing some residues), also set `not_remove` in `variable_sc_size` to preserve the fixed residue atoms, similar to fragment tasks above.

---

### 3.4 Task Definition (`task`)

Defines the generation task and its mode. The `task.name` field selects one of four task types:

| `task.name` | Task Family | Use Cases |
|-------------|-------------|-----------|
| `dock` | Docking | Small molecule & peptide docking |
| `sbdd` | Structure-based drug design | De novo small molecule design, optimization |
| `maskfill` | Mask-and-fill | Fragment linking, growing, partial optimization; **more general tasks** |
| `pepdesign` | Peptide design | De novo peptide, inverse folding, side-chain packing |
| `custom` | Custom task | Advanced user-defined tasks (see [Section 4](#4-defining-custom-tasks)) |

#### 3.4.1 Task: `dock`

```yaml
task:
  name: dock
  transform:
    name: dock
    settings:
      free: 1        # weight for Gaussian noise mode
      flexible: 0    # weight for flexible noise mode
```

**`settings`** controls the noise mode for docking (sampled stochastically):
- `free: 1, flexible: 0` — **Gaussian noise**: positional noise is isotropic Gaussian. Simpler, works for most cases, and much better performance.
- `free: 0, flexible: 1` — **Flexible noise**: noise decomposes into translation + rotation + torsional components. Better for preserving internal geometry.

> See `dock_smallmol.yml` (Gaussian) and `dock_smallmol_flex.yml` (flexible) for complete examples.

For **docking with fixed atoms** (e.g., fixing some atom positions), use the docking task variant in `dock_pep_fix_some.yml` which adds:
```yaml
task:
  name: dock
  transform:
    name: dock
    settings:
      free: 1
      flexible: 0
    fix_some:                  # union of all below
      res_bb: [0, 1]           # fix backbone atoms of residues 0 and 1 (peptide only)
      res_sc: [0]              # fix side-chain atoms of residue 0 (peptide only)
      atom: [73, 74, 75, 76]   # fix atoms by index (works for both small molecules and peptides)
```

> **Note:** The `fix_some` constraint works for both small molecules and peptides. For **small molecules**, use `atom: [...]` to specify fixed atom indices from the input SDF file. For **peptides**, you can additionally use `res_bb` and `res_sc` to fix entire residue backbones/side-chains by residue index. The final set of fixed atoms is the union of all specified selectors. The noise block also needs `pre_process: fix_some` (see `dock_pep_fix_some.yml`).

For **docking with constrained (known) atom positions**, use `dock_pep_know_some.yml` which restricts atoms to approximate spherical regions instead of fixing them exactly:
```yaml
noise:
  ...
  post_process:
    name: know_some
    atom_space:
      - atom: 0                              # atom index (0-based)
        coord: [-3.522, -13.459, -21.684]    # sphere center; omit to use input coords
        radius: 2                            # allowed radius (Å)
      - atom: 4
        radius: 2
```

Each entry in `atom_space` defines a spherical constraint for one atom:
- `atom` (`int`): Atom index (0-based) in the input ligand file.
- `coord` (`list [x,y,z]`, optional): Center of the allowed sphere. If omitted, uses the atom's coordinate from the input ligand file.
- `radius` (`float`, Å): Radius of the allowed sphere. The atom is guided toward this region during denoising.

#### 3.4.2 Task: `sbdd`

Two main SBDD modes:
- **`name: ar`** (default, see `sbdd.yml`): Auto-regressive generation with iterative refinement. Generates atoms in rounds, refining after each round:

```yaml
task:
  name: sbdd
  transform:
    name: ar           # auto-regressive refinement mode
    part1_pert: small   # small perturbation for existing atoms
```

- **`name: sbdd`** (simple, see `sbdd_simple.yml`): One-shot generation without refinement.
```yaml
task:
  name: sbdd
  transform:
    name: sbdd         # simple one-shot generation mode
```


#### 3.4.3 Task: `maskfill` (fragment linking / growing / partial optimization)

The `maskfill` task is the most general:
- It partitions the molecule into **part1** (known/fixed) and **part2** (to be generated), making it suitable for fragment linking, growing, and partial optimization.
- **part1** is more likely to be docked (can also fixed or slightly re-designed), while **part2** is to be designed.
- For fragment linking/growing, the given fragments are in part1, and the generated linker/grown part is part2.
- For partial optimization, the optimized part is part2, and the rest of the molecule is part1.



```yaml
task:
  name: maskfill
  transform:
    name: maskfill
    preset_partition:
      grouped_node_p1: [[0, 1, 2, 3, 4, 5, 6], [23, 24, 25, ...]]
    settings:
      part1_pert:
        fixed: 1          # fixing fragment positions
      known_anchor:
        none: 1           # no constraint on connecting atoms
```

**`preset_partition`** — Defines which atoms belong to part1 (fragments):

| Field | Type | Meaning |
|-------|------|---------|
| `grouped_node_p1` | `list[list[int]]` | Atom indices (0-based) for each fragment group in part1. E.g., `[[0,1,2], [10,11,12]]` defines two fragments. |
| `node_p2` | `list[int]` | (Alternative) Flat list of atom indices in part2. Auto-computed if `grouped_node_p1` is set. |
| `grouped_anchor_p1` | `list[list[int]]` | (Optional) Allowed connecting atoms for each fragment group. |

**`settings.part1_pert`** — How to treat part1  positions (sampled stochastically):

| Value | Meaning |
|-------|---------|
| `fixed: 1` | Fragment positions are **fixed** exactly as in the input (linking/growing with known poses) |
| `free: 1` | Fragment positions are **free** to move (linking/growing with unknown poses) |
| `small: 1` | Fragment positions and atom types are perturbed |

**`settings.known_anchor`** — Connecting atom constraints:

| Value | Meaning |
|-------|---------|
| `none: 1` | No constraint on which atoms connect parts (default) |
| `all: 1` | Connecting atoms are specified in `grouped_anchor_p1` |

> See `linking_fixed_frags.yml` (linking), `growing_fixed_frag.yml` (growing), `opt_partial.yml` (partial optimization) for full examples.

#### 3.4.4 Task: `pepdesign`

```yaml
task:
  name: pepdesign
  transform:
    name: pepdesign
    settings:
      mode:
        full: 1       # full peptide design (backbone dock + side-chain design)
        sc: 0          # side-chain design (backbone fixed)
        packing: 0     # side-chain packing (given backbone + side-chain atom types)
```

**`settings.mode`** (sampled stochastically by weight):

| Mode | Description | Use Case |
|------|-------------|----------|
| `full` | Design backbone + side-chains + sequence | De novo peptide design |
| `sc` | Design side-chains + sequence (backbone fixed) | Inverse folding / sequence design |
| `packing` | Design side-chain conformations only (backbone + sequence fixed) | Side-chain packing |

For **constrained design**, add constraint fields in `task.transform` to fix specific residue positions and/or types:

**Fix positions** (`fix_pos`, see `pepdesign_fix_pos.yml`): Fix the 3D coordinates of specified atoms during generation.
```yaml
task:
  transform:
    name: pepdesign
    fix_pos:
      res_bb: [0, 1, 8, 9]    # fix backbone of residues 0, 1, 8, 9
      res_sc: [0, 1]           # fix side-chains of residues 0, 1
      atom: [73, 74, 75, 76, 77, 78]  # fix specific atoms by index
    settings:
      mode:
        full: 1
```

**Fix types only** (`fix_type_only`, see `pepdesign_fix_type.yml`): Fix the atom types (residue types) but allow positions to be flexible.
```yaml
task:
  transform:
    name: pepdesign
    fix_type_only:
      res_bb: [0, 1, 2, 3, 4]   # fix backbone types of residues 0–4
      res_sc: [0, 1, 2, 3, 4]   # fix side-chain types of residues 0–4
      atom: []
    settings:
      mode:
        full: 1
```

**Fix both** (`fix_pos` + `fix_type_only`, see `pepdesign_fix_pos_and_type.yml`): Combine both constraints — some residues have fixed positions, others have fixed types only.
```yaml
task:
  transform:
    name: pepdesign
    fix_pos:
      res_bb: [8]       # fix position of residue 8
      res_sc: [8]
    fix_type_only:
      res_bb: [6, 7, 9]  # fix types of residues 6, 7, 9 (positions flexible)
      res_sc: [6, 7, 9]
    settings:
      mode:
        full: 1
```

> **Important:** When using `fix_pos` or `fix_type_only`, you must also set `transforms.variable_sc_size.not_remove` to include the side-chain atom indices of all constrained residues. This prevents those atoms from being removed during variable-size sampling. You can get atom indices in PyMOL by selecting residues and running: `iterate (sele and sc.), print(rank, end=',')`.

> See `pepdesign_denovo.yml`, `pepdesign_invfold.yml`, `pepdesign_sc_pack.yml` for full examples.

---

### 3.5 Noise Configuration (`noise`)

Configures the diffusion noise process. Different task types use different noise group structures.

**Noise group concept:** PocketXMol applies diffusion noise independently to different **groups** of atoms. Each noise group has its own prior distribution and noise schedule, allowing fine-grained control over which parts are generated from scratch (full noise) and which parts are preserved or lightly perturbed (low noise). For example:
- **Docking** uses a single group for atom coordinates
- **SBDD** uses a single group for the entire molecule for atom coordinates and atom types (including bond types)
- **maskfill** uses two groups: `part1` (like docking or SBDD) and `part2` (like SBDD).
- **Peptide design** uses two groups: `bb` backbone (like docking) and `sc` sidechains (like SBDD).

Each group defines: (1) a **prior** — the noise distribution to sample from at maximum noise, and (2) a **level** schedule — how the noise scales across denoising steps.

#### 3.5.1 Common fields

| Field | Type | Meaning |
|-------|------|---------|
| `name` | `string` | Noise type, typically matches `task.name` (`dock`, `maskfill`, `sbdd`, `pepdesign`) |
| `num_steps` | `int` | Number of denoising steps. Default: `100`. |
| `init_step` | `float` ∈ (0, 1] | Initial noise scale. `1.0` = start from pure noise (default); `< 1.0` = start from input pose with partial noise (useful for optimization/refinement). |
| `prior` | `dict` or `string` | Noise prior distributions per group. `from_train` = use training defaults. |
| `level` | `dict` | Noise schedule: maps denoising step → information level per group. |

> **Note:** The `noise.name` does not have to match `task.name`. For example, the `sbdd` task with `ar` mode actually uses `noise.name: maskfill` because it internally works with the part1/part2 noise group structure. The `noise.name` selects the noise group structure, not the task.

**Field details:**

- **`num_steps`**: Number of denoising steps. The denoising step linearly decreases from `init_step` to 0 over `num_steps` iterations. Default `100` works well for most tasks.

- **`init_step`**: Initial noise intensity, a float in $(0, 1]$. Controls how the generation starts:
  - `1.0` (default): Start from pure noise — atom coordinates are sampled from the noise prior (centered at the denoising space center). Any positional information in `input_ligand` is ignored.
  - `< 1.0`: Start from the input pose — initial noisy coordinates are sampled from noise centered at the input atom positions instead of the space center. Smaller values → output more similar to input. Useful for **optimization/refinement** tasks (see [Section 3.5.6](#356-noise-for-optimization-tasks)).
  - This is a global setting for all noise groups. For per-group control, use the `from_prior` field in the noise prior definition (see below).

- **`prior`**: Noise prior distributions. Defines the distribution from which maximum-noise samples are drawn. Can be:
  - `from_train` — use the training default distributions (recommended for most cases).
  - A dict with per-group prior definitions. Each group can specify:
    - `pos_only: true` — only apply positional noise (atom types and bonds are preserved).
    - `from_prior: false` — when `init_step < 1.0`, start noise from input coordinates rather than from the prior distribution. Useful when input coordinates provide a good starting point.
    - Sub-distributions for `node` (atom types), `pos` (coordinates), and `edge` (bond types).

- **`level`**: Noise schedule — maps the denoising step to an information level (1 = fully denoised, 0 = maximum noise). Can be per-group. See [Section 3.5.7](#357-noise-schedule-reference) for available schedule types.

#### 3.5.2 Noise: `dock`

Single noise group — the simplest configuration:

```yaml
noise:
  name: dock
  num_steps: 100
  prior: from_train
  level:
    name: advance
    min: 0.
    max: 1.
    step2level:
      scale_start: 0.99999
      scale_end: 0.00001
      width: 3
```

- `prior: from_train` loads the training default noise distributions.
- `level` uses the `advance` schedule (sigmoid-shaped curve from MolDiff), which provides smooth denoising.

#### 3.5.3 Noise: `sbdd` (simple mode)

The simple SBDD mode (`task.transform.name: sbdd`) uses a single noise group, identical to docking:

```yaml
noise:
  name: sbdd
  num_steps: 100
  prior: from_train
  level:
    name: advance
    min: 0.
    max: 1.
    step2level:
      scale_start: 0.99999
      scale_end: 0.00001
      width: 3
```

This generates molecules in one shot without refinement. See `sbdd_simple.yml`.

#### 3.5.4 Noise: `maskfill` (two groups: part1/part2)

The `maskfill` noise structure uses **two noise groups**: `part1` (existing/known atoms) and `part2` (newly generated atoms). This structure is shared by:
- **SBDD with auto-regressive refinement** (`task.transform.name: ar`) — uses `noise.name: maskfill`
- **Fragment linking/growing** (`task.name: maskfill`) — uses `noise.name: maskfill`

**SBDD (auto-regressive) example** (from `sbdd.yml`):

```yaml
noise:
  name: maskfill
  num_steps: 100
  ar_config:              # auto-regressive refinement settings
    strategy: refine
    r: 3                  # refinement ratio
    threshold_node: 0.98  # convergence threshold for atom types
    threshold_pos: 0.91   # convergence threshold for positions
    threshold_bond: 0.98  # convergence threshold for bonds
    max_ar_step: 10       # max refinement rounds
    change_init_step: 1   # initial noise step for refinement rounds
  prior:
    part1: from_train
    part2: from_train
  level:
    part1:
      name: uniform
      min: 0.6            # high info level → existing atoms stay close to current state
      max: 1.0
    part2:
      name: advance
      min: 0.0
      max: 1.0
      step2level:
        scale_start: 0.99999
        scale_end: 1.0e-05
        width: 3
```

**Key points:**
- `part1` uses `uniform` level with `min: 0.6` — existing atoms retain substantial structure (noise level stays low).
- `part2` uses `advance` level from `0.0` to `1.0` — new atoms are generated from full noise.
- `ar_config` enables iterative refinement: generate → check convergence → refine unconverged atoms.

**Fragment linking example** (explicit prior distributions):

```yaml
noise:
  name: maskfill
  num_steps: 100
  prior:
    part1:
      pos_only: true    # only positional noise for fragments
      pos:
        name: allpos
        pos:
          name: gaussian_simple
          sigma_max: 1
        translation:
          name: translation
          ve: false
          mean: 0
          std: 1
        rotation:
          name: rotation
          sigma_max: 0.0002
        torsional:
          name: torsional
          sigma_max: 0.2
    part2:
      node:
        name: categorical
        prior_type: predefined
        prior_probs: [3, 2, 2, 2, 1, 1, 1, 0.3, 0.3, 0.3, 0.3, 13.2]
      pos:
        name: allpos
        pos:
          name: gaussian_simple
          sigma_max: 1
      edge:
        name: categorical
        prior_type: tomask_half
  level:
    part1:
      name: advance
      min: 0.
      max: 1.
      ...
    part2:
      name: advance
      min: 0.0
      max: 1.0
      ...
```

**Key points:**
- `part1.pos_only: true` — only positional noise is applied (atom types / bonds are preserved).
- `part2` has full noise: `node` (atom types), `pos` (coordinates), `edge` (bond types) are all noised and regenerated.
- `prior_probs` defines the atom type prior distribution (unnormalized weights for each atom type category).

#### 3.5.5 Noise: `pepdesign` (two groups: bb/sc)

Peptide design uses backbone (`bb`) and side-chain (`sc`) noise groups:

```yaml
noise:
  name: pepdesign
  num_steps: 100
  prior:
    bb: from_train
    sc: from_train
  level:
    bb:
      name: advance
      min: 0.
      max: 1.
      step2level:
        scale_start: 0.99999
        scale_end: 0.00001
        width: 3
    sc:
      name: advance
      min: 0.
      max: 1.
      step2level:
        scale_start: 0.99999
        scale_end: 0.00001
        width: 3
```

For **inverse folding** (`pepdesign_invfold.yml`), the `mode` is set to `sc: 1` so only side-chains and sequence are designed, while the backbone structure comes from the input. The noise settings remain the same as de novo design — the backbone is preserved through the task mode rather than the noise schedule.

For **inverse folding with backbone flexibility** (`pepdesign_invfold_bbflex.yml`), the mode is `full: 1` but the `bb` noise group uses a high `min` level and `from_prior: False` to keep the backbone close to the input:

```yaml
noise:
  prior:
    bb:
      from_prior: false   # start backbone noise from input coords, not from random
      pos_only: true       # only positional noise for backbone (types preserved)
      pos:
        name: allpos
        pos:
          name: gaussian_simple
          sigma_max: 3
    sc: ...                # full noise for side-chains
  level:
    bb:
      name: advance
      min: 0.95            # very high info level → backbone barely moves
      max: 1.
      ...
    sc:
      name: advance
      min: 0.
      max: 1.
      ...
```

For **side-chain packing** (`pepdesign_sc_pack.yml`), mode is `packing: 1`. Both backbone structure and residue types are fixed; only side-chain conformations are generated. Note that `variable_sc_size` is **not needed** for packing since the atom counts are determined by the known residue types.

#### 3.5.6 Noise for optimization tasks

Set `init_step < 1.0` to start the denoising process from the input structure instead of pure noise:

```yaml
noise:
  name: sbdd
  num_steps: 50          # fewer steps for refinement
  init_step: 0.5         # smaller → more similar to input
  prior: from_train
  level:
    name: advance
    min: 0.
    max: 1.
    step2level:
      scale_start: 0.99999
      scale_end: 0.00001
      width: 3
```

- `init_step: 0.5` means the denoising starts halfway — the initial noisy molecule is a perturbed version of the input rather than pure noise.
- Smaller `init_step` → output more similar to input.
- See `opt_mol.yml` for a full optimization example.

#### 3.5.7 Noise schedule reference

Two schedule types are available for `level`:

**`advance`** (sigmoid-shaped, recommended for most tasks):
```yaml
name: advance
min: 0.0
max: 1.0
step2level:
  scale_start: 0.99999   # info level when noise_step ≈ 0
  scale_end: 0.00001     # info level when noise_step ≈ 1
  width: 3               # sigmoid steepness
```
Maps denoising step to information level via a sigmoid curve (from MolDiff). Higher `width` → sharper transition.

**`uniform`** (linear, useful for constrained parts):
```yaml
name: uniform
min: 0.6
max: 1.0
```
Maps denoising step linearly: $\text{info\_level} = \text{max} - (\text{max} - \text{min}) \times \text{noise\_step}$. Setting `min > 0` ensures the atoms never receive too much noise — useful for preserving known fragment positions.

---

## 4. Defining Custom Tasks

For advanced use cases beyond the built-in task types, the `custom` task allows defining arbitrary molecule partitions, per-part constraints, and independent noise groups. This section demonstrates with the `pepdesign_hot136E` example in `configs/sample/examples/pepdesign_hot136E/`.

> All common tasks (docking, SBDD, maskfill, pepdesign) can also be expressed through the `custom` task. The basic idea is to: (1) partition the molecule into named parts, (2) define noise groups with independent priors/schedules, and (3) map noise groups to molecule parts.

**Background:** Based on the PD-1/PD-L1 complex (PDB: 3BIK), a hot-spot residue 136E on PD-1 interacting with PDL1 was identified. The goal is to design a PDL1-binding peptide considering this interaction. The protein fragment around 136E serves as the input peptide and the PDL1 chain as the target (data in `data/examples/hot136E`).

**Example scenarios:**

| Config | Description |
|---|---|
| `fixed_Glu_CCOOH` | 6th residue is Glu with its -CCOOH group pose fixed as input |
| `fixed_CCOOH` | -CCOOH group with fixed pose (may be at any residue, may be Glu or Asp) |
| `fixed_CCOOH_init0.9` | Same as `fixed_CCOOH`, but initial noisy peptide starts from input coordinates (`noise.init_step` < 1) |
| `unfixed_Glu` | Glu at 6th residue, no coordinates fixed |
| `unfixed_CCOOH` | Contains -CCOOH group (Glu or Asp) at any position, no coordinates fixed |
| `unfixed_CCOOH_from_inputs` | Same as `unfixed_CCOOH`, but initial -CCOOH pose is biased toward input coordinates |

### 4.1 How to define a custom task

A custom task requires configuring three things: **partition**, **constraints**, and **multi-group noise**.

**1. Task Transform (`task.transform`)**

Defines the molecule partition and constraints:

```yaml
task:
  name: custom
  transform:
    is_peptide: true        # or false for small molecules
    partition:
      - name: hotspot       # named part
        nodes: [40, 41, 42, 43, 44]   # 0-based atom indices
      - name: rest
        nodes: [0, 1, ..., 39, 45, ...]
    fixed:
      node: [hotspot]       # fix atom types for these parts
      pos: [hotspot]        # fix coordinates for these parts
      edge: [[hotspot]]     # fix bonds within/between these parts
```

| Field | Type | Meaning |
|-------|------|---------|
| `is_peptide` | `bool` | Whether the molecule is a peptide |
| `partition` | `list[dict]` | Named parts, each with `name` and `nodes` (0-based atom indices) |
| `fixed.node` | `list[string]` | Part names whose atom types are fixed from input |
| `fixed.pos` | `list[string]` | Part names whose coordinates are fixed from input |
| `fixed.edge` | `list[list[string]]` | Part name pairs whose bond types are fixed |

**2. Noise Settings (`noise`)**

Define independent noise groups and map them to molecule parts:

```yaml
noise:
  num_steps: 100
  init_step: 1.0         # or < 1 for refinement from input
  prior:
    group_fixed:          # noise group for the fixed part
      pos_only: true
      from_prior: false   # start from input coordinates
      pos:
        name: gaussian_simple
        sigma_max: 0.5
    group_flex:           # noise group for the flexible part
      node: ...           # full noise distributions
      pos: ...
      edge: ...
  level:
    group_fixed:
      name: uniform
      min: 0.8            # high info level → low noise
      max: 1.0
    group_flex:
      name: advance
      min: 0.0
      max: 1.0
      step2level: ...
  mapper:                 # map noise groups → molecule parts
    group_fixed:
      node: hotspot
      pos: hotspot
      edge: hotspot
    group_flex:
      node: rest
      pos: rest
      edge: rest
```

| Field | Type | Meaning |
|-------|------|---------|
| `init_step` | `float` ∈ (0, 1] | `1.0` = pure noise start; `< 1.0` = start from input (perturbed). Step decays from `init_step` to 0 linearly. |
| `prior.<group>.pos_only` | `bool` | If `true`, only positional noise (no atom type / bond noise) |
| `prior.<group>.from_prior` | `bool` | `true` = sample initial coords from noise prior even when `init_step` < 1. `false` (default) = center noise on input coords. |
| `level.<group>.min` | `float` | Minimum information level. Higher → less noise → more input preservation. |
| `mapper` | `dict` | Maps each noise group to molecule parts for `node`, `pos`, `edge` channels. |

**(See the YAML files in `configs/sample/examples/pepdesign_hot136E/` for exact syntax. Refer to `configs/train/train_pxm_reduced.yml` for training default noise settings.)**

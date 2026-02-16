# Training

> [← Back to Documentation Index](README.md)

## 1. Demo Training (Reduced Dataset)

This repo includes a reduced dataset for demonstration purposes.

**Prerequisites:**
1. Ensure you have downloaded and extracted `data_train_processed.tar.gz` (see [Setup Guide](setup.md)).
2. Activate your environment (`conda activate pxm`).

**Run Training:**
```bash
python scripts/train_pl.py \
    --config configs/train/train_pxm_reduced.yml \
    --num_gpus 1
```

> **Note:**
> - Config file: `configs/train/train_pxm_reduced.yml`.
> - To adjust for GPU memory, modify `batch_size` in the config file.
> - `num_gpus 1` specifies single-GPU training.

## 2. Full Training (Custom Data)

The complete training dataset exceeds 500GB. To train from scratch:

1.  **Process Raw Data:** Follow instructions in [Raw Data Processing](data_processing.md).
2.  **Update Config:**
    -   Open `configs/train/train_pxm_reduced.yml` (or create a copy).
    -   Update `data.dataset.root` and `data.dataset.assembly_path` to point to your full processed data.
3.  **Run Training:** Use the same command as above with your updated config.

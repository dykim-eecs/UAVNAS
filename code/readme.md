# UAV-NAS Search & Training (PyTorch)

DARTS-based Neural Architecture Search and fixed-architecture training pipeline for
the UAV-NAS project. The discovered architecture from this stage is exported to ONNX
and consumed by the FPGA inference accelerator under [`../hls`](../hls).

## Files

| File | Purpose |
|---|---|
| `architect.py` | Optimizer wrapper for the architecture parameters (alphas) |
| `fileio.py` | `WaveformDataset`: loads `.fc32` IQ files into `(2, 32, 3072)` tensors |
| `genotypes.py` | `Genotype` namedtuple and the NAS-discovered architecture (`NASGenotype`) |
| `model_search.py` | DARTS super-network used during the architecture search |
| `operations.py` | Primitive ops (sep/dil conv, pool, identity, ...) used by both networks |
| `train_search.py` | Architecture search training loop (writes the best genotype) |
| `infer.py` | Trains the fixed (post-search) network and exports it to ONNX |
| `utils.py` | Shared `AverageMeter`, `accuracy`, and `EarlyStopping` |
| `run_experiments.py` | Hyperparameter sweep over `(init_channels, layers)` |
| `graph.py` | Plots `train_val_log.csv` produced by `infer.py` |

## Prerequisites

- Python 3.9+
- PyTorch (with CUDA)
- NumPy, pandas, matplotlib

## Dataset Layout

`WaveformDataset` expects one subdirectory per class, named `00`, `01`, ..., `12`,
each containing `.fc32` files (raw float32 interleaved I/Q samples):

```
<dataset_root>/
  00/*.fc32
  01/*.fc32
  ...
  12/*.fc32
```

## 1. Architecture Search

```bash
python3 train_search.py \
    --train_dir /path/to/dataset/train \
    --epochs 100 \
    --init_channels 4 \
    --layers 2 \
    --genotype_out best_genotype.pkl
```

Outputs:

- `alpha_loss_log.csv` — per-epoch alpha mean and train/val loss
- `alpha_mean_plot.png`, `loss_plot.png` — training curves
- `best_genotype.pkl` — best discovered genotype (when `--genotype_out` is set)

After the search, copy the printed `Genotype(...)` into the `NASGenotype` variable
in `genotypes.py` (or pass `--genotype_file best_genotype.pkl` to `infer.py`).

## 2. Fixed-Architecture Training and ONNX Export

```bash
python3 infer.py \
    --train_dir /path/to/dataset/train \
    --test_dir  /path/to/dataset/test \
    --epochs 100 \
    --init_channels 4 \
    --layers 2 \
    --save_checkpoint best_model.pth \
    --onnx_out model.onnx
```

Outputs:

- `best_model.pth` — best PyTorch checkpoint (selected by validation loss)
- `model.onnx` — ONNX export consumed by the HLS accelerator
- `train_val_log.csv` — per-epoch train/val loss and accuracy

## 3. Hyperparameter Sweep (optional)

```bash
python3 run_experiments.py
```

Sweeps over `init_channels = {1,2,4,8,16}` × `layers = {1,2,4,8,16}` and streams
results to `full_experiment_results_<timestamp>.csv`.

## Plotting

```bash
python3 graph.py
```

Reads `train_val_log.csv` and writes `accuracy_curve.png` / `loss_curve.png`.

# ReaKt-v2 — Penicillin Soft-Sensor Benchmark

Benchmark comparing multiple model architectures as soft-sensors for online penicillin prediction from Raman spectra on **IndPenSim V3** (100 batches, 2,001 Raman channels).

The core pipeline is: **frozen beta-VAE encoder** (Raman -> 64-dim latent z) -> **PI-LSTM** (z sequence -> Penicillin g/L).

---

## Setup

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place `100_Batches_IndPenSim_V3.csv` in the project root (not tracked by git -- obtain separately).

---

## Quick Reference

| What you want to do | Command |
|---|---|
| Train one model (default seed) | `python train.py --model beta_vae_pi_lstm` |
| Train with a specific seed | `python train.py --model beta_vae_pi_lstm --seed 123` |
| Evaluate a checkpoint | `python evaluate.py --model beta_vae_pi_lstm` |
| Evaluate a specific seed | `python evaluate.py --model beta_vae_pi_lstm --seed 123` |
| Multi-seed validation (5 seeds) | `python run_seeds.py --model beta_vae_pi_lstm` |
| Multi-seed validation (10 seeds) | `python run_seeds.py --model beta_vae_pi_lstm --n-seeds 10` |
| Quick smoke test (3 seeds, 20 ep) | `python run_seeds.py --model pi_lstm --n-seeds 3 --epochs 20` |
| Beta sweep (encoder + LSTM) | `PYTHONPATH=. python scripts/beta_sweep_e2e.py` |
| Static baselines (PLS, PCA-PCR) | `PYTHONPATH=. python scripts/pls_pca_baseline_k64.py` |

---

## 1. Single Training Run

```bash
# Basic usage — loads configs/beta_vae_pi_lstm.yaml, seed=42
python train.py --model beta_vae_pi_lstm

# Override seed (checkpoint auto-named to *_s123.pt to avoid collision)
python train.py --model beta_vae_pi_lstm --seed 123

# Override epochs, device, checkpoint path
python train.py --model cdae_pi_lstm --epochs 50 --device cpu --ckpt /tmp/test.pt

# Regime ablation (train R1, test R2)
python train.py --model beta_vae_pi_lstm --regime-ablation

# Disable early stopping (train full epochs)
python train.py --model beta_vae_pi_lstm --no-early-stopping
```

### train.py options

| Flag | Description |
|---|---|
| `--model NAME` | Algorithm name (required, see Models below) |
| `--seed N` | Override random seed; auto-renames checkpoint to `*_s{N}.pt` |
| `--epochs N` | Override training epochs from config |
| `--ckpt PATH` | Override checkpoint save path |
| `--config PATH` | Override YAML config file |
| `--device cpu/cuda` | Force device (default: auto-detect) |
| `--regime-ablation` | Train on R1 (batches 1-30), test on R2 (31-60) |
| `--no-early-stopping` | Disable early stopping, run all epochs |
| `--no-plots` | Skip saving prediction plots |

### What happens under the hood

1. Load YAML config from `configs/<model>.yaml`
2. Set seed (`random`, `numpy`, `torch`)
3. Load CSV, detect batches, split into train/val/test
4. Build Raman encoder (if needed), encode spectra -> latents
5. Build sliding-window dataset (per-batch, no cross-batch windows)
6. Train with early stopping, LR scheduling, physics loss annealing
7. Evaluate on val + test, save checkpoint with metrics embedded
8. Save prediction plots

---

## 2. Single Evaluation

```bash
# Evaluate default checkpoint
python evaluate.py --model beta_vae_pi_lstm

# Evaluate a specific seed's checkpoint
python evaluate.py --model beta_vae_pi_lstm --seed 123

# Evaluate a custom checkpoint
python evaluate.py --model beta_vae_pi_lstm --ckpt path/to/model.pt

# Evaluate ALL models with checkpoints on disk
python evaluate.py --model all

# With prediction plots
python evaluate.py --model all --plots
```

### evaluate.py options

| Flag | Description |
|---|---|
| `--model NAME/all` | Model to evaluate (default: `all`) |
| `--seed N` | Override seed; auto-resolves `*_s{N}.pt` checkpoint |
| `--ckpt PATH` | Custom checkpoint path |
| `--plots` | Save per-batch prediction grids |
| `--scatter PATH` | Save scatter comparison plot |
| `--regime-ablation` | Use regime-ablation splits |
| `--test-regime 2/3` | Which regime to test on (with `--regime-ablation`) |

---

## 3. Multi-Seed Validation (Recommended for Papers)

Use `run_seeds.py` to train and evaluate across multiple seeds in one command. Results are aggregated into a CSV with mean +/- std.

```bash
# Default: 5 seeds (42, 123, 456, 789, 1024)
python run_seeds.py --model beta_vae_pi_lstm

# 10 seeds
python run_seeds.py --model beta_vae_pi_lstm --n-seeds 10

# Custom seeds
python run_seeds.py --model beta_vae_pi_lstm --seeds 0 1 2 3 4 5 6 7 8 9

# Quick smoke test
python run_seeds.py --model pi_lstm --n-seeds 3 --epochs 20 --device cpu

# Resume interrupted sweep (skip seeds with existing checkpoints)
python run_seeds.py --model beta_vae_pi_lstm --skip-existing

# Evaluate only (no training, just aggregate existing results)
python run_seeds.py --model beta_vae_pi_lstm --eval-only

# With regime ablation
python run_seeds.py --model beta_vae_pi_lstm --n-seeds 5 --regime-ablation
```

### Output structure

```
outputs/seed_sweep/<model>/
    results.csv                    # Per-seed metrics + mean/std summary
    checkpoints/
        <model>_s42.pt             # Checkpoint per seed
        <model>_s123.pt
        ...
    seed_42.log                    # Training log per seed
    seed_123.log
    ...
```

### results.csv format

```csv
seed,val_pen_rmse,val_pen_mae,val_pen_r2,test_pen_rmse,test_pen_mae,test_pen_r2
42,0.1523,0.1102,0.9987,0.3340,0.2397,0.9985
123,0.1489,0.1078,0.9988,0.3412,0.2455,0.9984
...
mean,0.1506,0.1090,0.9988,0.3376,0.2426,0.9985
std,0.0024,0.0017,0.0001,0.0051,0.0041,0.0001
```

### run_seeds.py options

| Flag | Description |
|---|---|
| `--model NAME` | Algorithm name (required) |
| `--seeds N [N ...]` | Explicit seed list |
| `--n-seeds N` | Number of seeds if `--seeds` not given (default: 5) |
| `--epochs N` | Override training epochs |
| `--device cpu/cuda` | Force device |
| `--regime-ablation` | Use regime-ablation splits |
| `--skip-existing` | Skip seeds with existing checkpoints |
| `--eval-only` | Only aggregate, no training |
| `--out-dir PATH` | Custom output directory |

---

## 4. Full Benchmark Pipeline

```bash
# Step 1: Train beta-VAE encoder
PYTHONPATH=. python scripts/train_autoencoder.py --model beta_vae

# Step 2: Full beta sweep (encoder + PI-LSTM per beta)
PYTHONPATH=. python scripts/beta_sweep_e2e.py

# Step 3: Static baselines
PYTHONPATH=. python scripts/pls_pca_baseline_k64.py

# Step 4: Alternative temporal models
PYTHONPATH=. python scripts/train_latent_ode.py
PYTHONPATH=. python scripts/train_neural_cde.py
PYTHONPATH=. python scripts/train_sindyc.py

# Step 5: Multi-seed validation of best model
python run_seeds.py --model beta_vae_pi_lstm --n-seeds 10

# Step 6: Aggregate comparison table
PYTHONPATH=. python scripts/benchmark_summary.py
```

For long-running scripts, use background execution:
```bash
nohup .venv/bin/python -u scripts/beta_sweep_e2e.py > outputs/sweep.log 2>&1 &
```

---

## Layout

```
ReaKt-v2/
    train.py                        # Train any model (YAML config + CLI overrides)
    evaluate.py                     # Evaluate checkpoints
    run_seeds.py                    # Multi-seed sweep with CSV aggregation

    configs/                        # YAML config per model (21 files)
        beta_vae_pi_lstm.yaml       # beta-VAE + PI-LSTM (main model)
        cdae_pi_lstm.yaml           # CDAE + PI-LSTM
        cvae_pi_lstm.yaml           # CVAE + PI-LSTM
        pi_lstm.yaml                # Process-only PI-LSTM
        ...

    scripts/
        train_autoencoder.py        # Train CDAE / CVAE / beta-VAE encoder
        beta_sweep_e2e.py           # End-to-end beta sweep (encoder + PI-LSTM)
        pls_pca_baseline_k64.py     # PLS and PCA-PCR static baselines
        train_bvae_mlp.py           # beta-VAE + MLP head
        train_latent_ode.py         # Latent ODE baseline
        train_neural_cde.py         # Neural CDE baseline
        train_sindyc.py             # SINDYc baseline
        benchmark_summary.py        # Aggregate all results
        baseline_temporal_poly.py   # Time-only polynomial baseline
        ...                         # Analysis scripts (SHAP, UMAP, etc.)

    src/
        config.py                   # YAML -> Config object
        autoencoder/                # CDAE_Raman, CVAE_Raman, BetaVAE_Raman
        data/                       # Dataset, encoders (CDAE/CVAE/PCA/PLS/beta-VAE)
        models/                     # PyTorch nn.Module definitions
        algorithms/                 # fit / evaluate / save / load per model
        physics/                    # Mass-balance loss terms
        evaluation/                 # compute_metrics (RMSE, MAE, R2)
        visualization/              # Training history + prediction plots

    checkpoints/                    # Saved weights (not tracked by git)
    outputs/                        # Results and plots
        seed_sweep/                 # Multi-seed results
        beta_sweep_e2e/             # Beta sweep results
        benchmark/                  # Final comparison table
```

---

## Models

| Model | Config | Input | Type |
|---|---|---|---|
| `beta_vae_pi_lstm` | `beta_vae_pi_lstm.yaml` | 64-dim mu from frozen beta-VAE | Temporal (PI-LSTM) |
| `cdae_pi_lstm` | `cdae_pi_lstm.yaml` | 64-dim z from frozen CDAE | Temporal (PI-LSTM) |
| `cvae_pi_lstm` | `cvae_pi_lstm.yaml` | 64-dim mu from frozen CVAE | Temporal (PI-LSTM) |
| `pi_lstm` | `pi_lstm.yaml` | 23 process features | Temporal (PI-LSTM) |
| `pca_pi_lstm` | `pca_pi_lstm.yaml` | 4-dim PCA Raman latents | Temporal (PI-LSTM) |
| `pls_pi_lstm` | `pls_pi_lstm.yaml` | 4-dim PLS Raman latents | Temporal (PI-LSTM) |
| `neural_ode` | `neural_ode.yaml` | 23 process features | Continuous-time ODE |
| `beta_vae_lstm` | `beta_vae_lstm.yaml` | 64-dim mu from frozen beta-VAE | Temporal (vanilla LSTM) |
| `process_pi_lstm` | `process_pi_lstm.yaml` | 23 process features | Temporal (pen-only head) |

**Architecture**: 2-layer LSTM(64) -> LayerNorm -> Linear(64->32)+LeakyReLU -> Linear(32->1)+Sigmoid.
Physics loss: dP = k_prod * r_net * dt, annealed lambda: 0 -> 0.1 over epochs 30-110.

---

## Evaluation Protocol

- **Train**: Fault-free batches (1-90), 80/20 random split for train/val
- **Test**: Faulty batches (91-100), fixed OOD test set
- **Metrics**: RMSE, MAE, R2 computed at **sparse offline measurement points** only (~20 per batch, ~200 total), in **g/L** units
- **Regime ablation**: Train on R1 (batches 1-30), test on R2 (31-60) or R3 (61-90)

---

## Key Results (Single Seed, seed=42)

| Model | P1 Test RMSE | P2 Test RMSE |
|---|---|---|
| β-VAE-LSTM (β=0.005) | 0.132 | 0.289 |
| CVAE-LSTM | 0.135 | 0.672 |
| CDAE-LSTM | 0.336 | 0.741 |
| PLS-LSTM | 0.520 | 0.634 |
| PCA-PCR (static) | 0.456 | 0.474 |
| PLS (static) | 0.335 | 0.318 |
| LSTM (raw features) | 1.629 | 1.985 |

Full results: `outputs/protocol2_results.json`, `outputs/benchmark/comparison_table.csv`

---

## 5. Multi-Seed Reproducibility (Mean ± Std)

The paper reports **mean ± std over 5 seeds** (42, 1, 2, 3, 4). Two evaluation protocols:
- **Protocol 1**: Train on R1–R3 (batches 0–89, 72 train / 18 val), test on R4 (fault, 90–99)
- **Protocol 2**: Train on R1 only (batches 0–29, 24 train / 6 val), test on R4

### 5.1 LSTM-based models (via `run_seeds.py`)

These models use a **frozen pre-trained encoder** and only retrain the PI-LSTM per seed:

```bash
# Protocol 1 (default: train R1-R3, test R4 = fault)
python run_seeds.py --model beta_vae_pi_lstm --seeds 42 1 2 3 4 --device cuda:0
python run_seeds.py --model cvae_pi_lstm     --seeds 42 1 2 3 4 --device cuda:0
python run_seeds.py --model cdae_pi_lstm     --seeds 42 1 2 3 4 --device cuda:0
python run_seeds.py --model pls_pi_lstm      --seeds 42 1 2 3 4 --device cuda:0
python run_seeds.py --model pca_pi_lstm      --seeds 42 1 2 3 4 --device cuda:0
python run_seeds.py --model pi_lstm          --seeds 42 1 2 3 4 --device cuda:0
python run_seeds.py --model neural_ode       --seeds 42 1 2 3 4 --device cuda:0

# Protocol 2 (regime ablation: train R1 only, test R4)
python run_seeds.py --model beta_vae_pi_lstm --seeds 42 1 2 3 4 --regime-ablation --device cuda:0
python run_seeds.py --model cvae_pi_lstm     --seeds 42 1 2 3 4 --regime-ablation --device cuda:0
python run_seeds.py --model cdae_pi_lstm     --seeds 42 1 2 3 4 --regime-ablation --device cuda:0
python run_seeds.py --model pls_pi_lstm      --seeds 42 1 2 3 4 --regime-ablation --device cuda:0
python run_seeds.py --model pi_lstm          --seeds 42 1 2 3 4 --regime-ablation --device cuda:0
python run_seeds.py --model neural_ode       --seeds 42 1 2 3 4 --regime-ablation --device cuda:0
```

Results per model in `outputs/seed_sweep/<model>/results.csv`.

> **Note**: `run_seeds.py` reuses the frozen encoder checkpoint (e.g., `checkpoints/beta_vae_best.pt`).
> Only LSTM weights are re-initialized per seed. The data split also changes per seed.

### 5.2 End-to-end retraining (encoder + LSTM per seed)

For the strongest claim, retrain both the β-VAE encoder AND the LSTM from scratch per seed.
Use `scripts/multi_seed_train.py`:

```bash
# Protocol 1: retrain β-VAE (300 epochs) + PI-LSTM (200 epochs) × 4 seeds
PYTHONPATH=. python scripts/multi_seed_train.py --protocol 1 --device cuda:0

# Protocol 2: same but train only on R1
PYTHONPATH=. python scripts/multi_seed_train.py --protocol 2 --device cuda:0

# Both protocols in one run:
PYTHONPATH=. python scripts/multi_seed_train.py --protocol both --device cuda:0
```

Seeds: 1, 2, 3, 4 (seed=42 already has checkpoints from prior runs).
Outputs: `outputs/multi_seed/protocol{1,2}_results.json`.

**Estimated time**: ~8 min per seed (VAE ~4 min + LSTM ~4 min on V100), ~32 min per protocol.

### 5.3 Static baselines (PCA-PCR, PLS, SVR, time-only)

These train from scratch each time (fast, pure sklearn), so multi-seed is trivial:

```bash
# All static baselines + both protocols + 5 seeds:
PYTHONPATH=. python scripts/run_all_seeds.py --models cubic_poly mean_traj pca_pcr pls svr \
    --seeds 42 1 2 3 4 --protocol both --device cpu
```

Or use the dedicated script which evaluates them inline:
```bash
PYTHONPATH=. python scripts/protocol2_eval.py
```

### 5.4 JIT-FT and VAE-JITL variants

These require a pre-built window database and a global LSTM checkpoint. Multi-seed
variability comes from the global model seed (the JIT-FT adaptation is deterministic
given K neighbors):

```bash
# β-VAE JIT-FT with regime ablation (uses beta_vae_pi_lstm_regime_ablation.pt)
PYTHONPATH=. python scripts/cvae_jit_ft.py --model beta_vae --regime-ablation \
    --device cuda:0 --K 150

# CVAE JIT-FT
PYTHONPATH=. python scripts/cvae_jit_ft.py --model cvae --regime-ablation \
    --device cuda:0 --K 200

# VAE-JITL
PYTHONPATH=. python scripts/vae_jitl.py --regime-ablation --K 500
```

To run multi-seed JIT-FT, retrain the global model per seed first (step 5.1 or 5.2),
then rebuild the DB and evaluate:
```bash
for SEED in 42 1 2 3 4; do
  # Point to per-seed global checkpoint
  PYTHONPATH=. python scripts/cvae_jit_ft.py --model beta_vae --regime-ablation \
      --global-ckpt outputs/seed_sweep/beta_vae_pi_lstm/checkpoints/beta_vae_pi_lstm_s${SEED}.pt \
      --db-cache outputs/multi_seed/jit_ft_db_s${SEED}.npz \
      --rebuild-db --device cuda:0 --K 150
done
```

### 5.5 Full automated benchmark (all models, both protocols)

```bash
# Complete run: all models, seeds 42/1/2/3/4, both protocols
PYTHONPATH=. python scripts/run_all_seeds.py --protocol both --seeds 42 1 2 3 4 --device cuda:0
```

This produces `outputs/multi_seed/summary.json` with the final table.

---

## Evaluation Protocol

- **Protocol 1 (standard)**: Train on R1–R3 (batches 0–89), 80/20 batch-wise random split → 72 train / 18 val. Test on R4 (fault batches 90–99, 10 batches).
- **Protocol 2 (regime ablation)**: Train on R1 only (batches 0–29), 80/20 → 24 train / 6 val. Test on R4.
- **Metrics**: RMSE, MAE, R² computed at **sparse offline measurement points** only (~20 per batch, ~198 test points), in **g/L** units.
- **NRMSE**: RMSE / max(Penicillin) × 100%, where max = 36.18 g/L.
- **Δ(%)**: (Test RMSE − Val RMSE) / Val RMSE × 100% — measures generalization gap.

---

## Reproducibility

- Default seed: 42 (controls train/val split, weight init, data shuffling)
- Use `run_seeds.py` with 5 seeds to report mean ± std for paper
- cuDNN non-determinism may cause ±1% run-to-run variation on GPU
- Checkpoints embed the full config + metrics for traceability
- Hardware used: DGX Station, 4× Tesla V100-DGXS-16GB (single GPU per experiment)
- PyTorch 2.5.1+cu124

---

*Internal project -- ReaKt / Genopole. Not for redistribution.*

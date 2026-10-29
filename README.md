# Penicillin Soft-Sensor Benchmark

Benchmark comparing model architectures as soft-sensors for online penicillin prediction
from Raman spectra on **IndPenSim V3** (100 batches, 2,001 Raman channels).

Core pipeline: **frozen Raman encoder** (CDAE / CVAE / β-VAE / PCA / PLS → latent z)
→ **PI-LSTM** (z sequence → Penicillin g/L, physics-informed loss).

---

## Setup

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place `100_Batches_IndPenSim_V3.csv` in the project root (not tracked — download from
[industrialpenicillinsimulation.com](http://www.industrialpenicillinsimulation.com/)).

Scripts under `scripts/` need the project root on `PYTHONPATH`:
```bash
PYTHONPATH=. python scripts/<script>.py
```

---

## Quick Reference

| Goal | Command |
|---|---|
| Train one model | `python train.py --model beta_vae_pi_lstm` |
| Evaluate a checkpoint | `python evaluate.py --model beta_vae_pi_lstm` |
| Multi-seed benchmark (5 seeds, both protocols) | `python run_seeds.py --model beta_vae_pi_lstm` |
| Train a Raman encoder from scratch | `PYTHONPATH=. python scripts/train_autoencoder.py --model beta_vae` |
| β sweep (encoder + LSTM per β) | `PYTHONPATH=. python scripts/beta_sweep_e2e.py` |
| Static baselines (PLS, PCA-PCR) | `PYTHONPATH=. python scripts/pls_pca_baseline_k64.py` |
| JIT fine-tuning adaptation | `PYTHONPATH=. python scripts/cvae_jit_ft.py --model beta_vae` |
| Local-PLS adaptation (VAE-JITL) | `PYTHONPATH=. python scripts/vae_jitl.py` |

---

## 1. Single Training Run — `train.py`

```bash
python train.py --model beta_vae_pi_lstm                      # seed=42, configs/<model>.yaml
python train.py --model beta_vae_pi_lstm --seed 123            # checkpoint auto-renamed *_s123.pt
python train.py --model cdae_pi_lstm --epochs 50 --device cpu
python train.py --model beta_vae_pi_lstm --regime-ablation     # train R1, test R2
```

| Flag | Description |
|---|---|
| `--model NAME` | Required. See [Models](#models) for the full list. |
| `--seed N` | Overrides seed; renames checkpoint to `*_s{N}.pt`. |
| `--epochs N` | Overrides `n_epochs` from the YAML config. |
| `--ckpt PATH` / `--config PATH` | Override output checkpoint / input config. |
| `--device cpu/cuda` | Force device (default: auto-detect). |
| `--regime-ablation` | Train on R1 (batches 1–30), test on R2 (31–60). |
| `--no-early-stopping` | Run all epochs, ignore early-stopping patience. |
| `--no-plots` | Skip saving prediction/history plots. |
| `--raman-ckpt` / `--raman-scaler` | Point to a specific (e.g. per-seed) frozen encoder instead of the config default. |

What it does: loads `configs/<model>.yaml` → sets seeds → loads CSV & splits batches →
encodes Raman spectra (if applicable) → builds per-batch sliding windows → trains with
early stopping + annealed physics loss → evaluates on val/test → saves checkpoint
(weights + embedded config + metrics) and plots.

---

## 2. Single Evaluation — `evaluate.py`

```bash
python evaluate.py --model beta_vae_pi_lstm                    # default checkpoint
python evaluate.py --model beta_vae_pi_lstm --seed 123          # resolves *_s123.pt
python evaluate.py --model all --plots                          # every checkpoint on disk
```

| Flag | Description |
|---|---|
| `--model NAME/all` | Model to evaluate (default: `all`). |
| `--seed N` | Resolves `*_s{N}.pt`. |
| `--ckpt PATH` | Custom checkpoint path. |
| `--plots` / `--scatter PATH` | Save prediction grids / scatter comparison. |
| `--regime-ablation --test-regime 2/3` | Evaluate on R2 or R3 instead of the fault set. |

---

## 3. Multi-Seed Benchmark — `run_seeds.py`

Trains and evaluates across multiple seeds, both protocols, and aggregates mean ± std.

```bash
python run_seeds.py --model beta_vae_pi_lstm                              # 5 default seeds, both protocols
python run_seeds.py --model beta_vae_pi_lstm --seeds 42 123 456           # custom seeds
python run_seeds.py --model beta_vae_pi_lstm --protocol 1                 # one protocol only
python run_seeds.py --model all --device cuda                             # every model, sequential
python run_seeds.py --model beta_vae_pi_lstm --eval-only                  # re-aggregate only
python run_seeds.py --model beta_vae_pi_lstm --skip-existing --parallel 1 # resume, avoid GPU/RAM OOM
```

| Flag | Description |
|---|---|
| `--model NAME/all` | Single model or every registered model + adaptation variant. |
| `--protocol 1/2/both` | 1 = train R1–R3, test R4 (fault). 2 = train R1 only, test R4 (also reports R2/R3). |
| `--seeds N [N ...]` / `--n-seeds N` | Explicit list, or count of default seeds (42, 123, 456, 789, 1024). |
| `--epochs N` / `--device` | Override epochs / force device. |
| `--skip-existing` | Resume: skip seeds with existing checkpoints. |
| `--eval-only` | No training, just aggregate existing checkpoints/caches. |
| `--parallel N` | Concurrent GPU jobs (default 4). **Lower this (1–2) for adaptation models** (`*_jit_ft`, `vae_jitl`) — each JIT-FT job loads a full encoder + LSTM + K-NN window DB and 4+ in parallel can OOM a small GPU/host. |
| `--out-dir PATH` | Custom output root (default `outputs/seed_sweep/<model>`). |

Adaptation models (`beta_vae_jit_ft`, `cvae_jit_ft`, `vae_jitl`) are trained in Phase 1/2
(their base LSTM + encoder), then Phase 3 runs the on-the-fly adaptation script per seed —
no extra flags needed, just pass the model name like any other.

**Output**: `outputs/seed_sweep/<model>/`
```
summary.csv                      # mean ± std / NRMSE% / Δ% per protocol
p{1,2}/checkpoints/<model>_s{seed}.pt
p{1,2}/encoders/{type}_s{seed}.pt + {type}_scaler_s{seed}.joblib   # per-seed Raman encoder, if applicable
p{1,2}/seed_{seed}.log
```

---

## 4. Raman Encoders — `scripts/train_autoencoder.py`

Trains a CDAE / CVAE / β-VAE encoder from scratch (independent of any LSTM).

```bash
PYTHONPATH=. python scripts/train_autoencoder.py --model beta_vae \
    --max-beta 0.005 --anneal-schedule linear --anneal-epochs 100 --anneal-delay 20
PYTHONPATH=. python scripts/train_autoencoder.py --model cvae --beta 1.0
PYTHONPATH=. python scripts/train_autoencoder.py --model cdae --noise-std 0.1
```

Key flags: `--model {cdae,cvae,beta_vae,beta_vae_light}`, `--latent-dim` (default 64),
`--max-beta` / `--anneal-schedule {linear,cyclical}` / `--anneal-epochs` / `--anneal-delay`
(β-VAE only), `--regime-ablation` (train on R1 only), `--seed`, `--ckpt` / `--scaler`.

Static baselines (PCA / PLS) use dedicated scripts with a similar CLI:
`scripts/pca_baseline.py`, `scripts/pls_baseline.py`, `scripts/pls_pca_baseline_k64.py`
(the 64-component joint sweep used for the benchmark table), `scripts/svr_baseline.py`.

---

## 5. β Sweep — `scripts/beta_sweep_e2e.py`

End-to-end sweep: trains a β-VAE encoder **and** its PI-LSTM for each β value, evaluates
on R1val/R2/R3/Fault, and reports the U-curve (reconstruction vs. disentanglement trade-off).

```bash
PYTHONPATH=. python scripts/beta_sweep_e2e.py                                   # default 8-point log grid
PYTHONPATH=. python scripts/beta_sweep_e2e.py --betas 0.001 0.005 0.05 1.0
PYTHONPATH=. python scripts/beta_sweep_e2e.py --skip-trained --out-dir outputs/my_sweep
PYTHONPATH=. python scripts/beta_sweep_e2e.py --plots-only                      # reuse existing sweep_results.json
```

`--skip-trained` resumes a sweep by reusing checkpoints already on disk per β
(`checkpoints/<tag>/{vae.pt,vae_scaler.joblib,lstm.pt}`) — only evaluation is re-run, so
metrics always reflect the current code. For long sweeps, launch detached:
```bash
nohup .venv/bin/python -u scripts/beta_sweep_e2e.py > outputs/sweep.log 2>&1 &
```

---

## 6. On-the-Fly Adaptation (JIT-FT / VAE-JITL)

Two local-adaptation variants on top of a **frozen** global PI-LSTM:

- **JIT-FT** (`scripts/cvae_jit_ft.py`): retrieves the K nearest training windows by
  symmetric-KL distance in latent space, briefly fine-tunes a cloned copy of the global
  LSTM on them, predicts, then discards the clone.
- **VAE-JITL** (`scripts/vae_jitl.py`): same K-NN retrieval but fits a fresh **local PLS**
  regression per query instead of fine-tuning a neural net — more robust, slightly higher
  bias.

```bash
PYTHONPATH=. python scripts/cvae_jit_ft.py --model beta_vae --K 150 \
    --global-ckpt outputs/seed_sweep/beta_vae_pi_lstm/p1/checkpoints/beta_vae_pi_lstm_s42.pt \
    --cvae-ckpt   outputs/seed_sweep/beta_vae_pi_lstm/p1/encoders/beta_vae_s42.pt \
    --cvae-scaler outputs/seed_sweep/beta_vae_pi_lstm/p1/encoders/beta_vae_scaler_s42.joblib

PYTHONPATH=. python scripts/vae_jitl.py --K 500 --local-model pls --n-components 10
```

> **Important**: `--cvae-ckpt`/`--cvae-scaler` (or `--ckpt`/`--scaler` for `vae_jitl.py`)
> **must** match the exact encoder the `--global-ckpt` LSTM was trained with — its `feat`
> scaler is calibrated to that encoder's latent range. `run_seeds.py` resolves this
> automatically per seed; when calling the scripts directly, always pass matching paths.

Other useful flags (`cvae_jit_ft.py`): `--local-epochs` / `--local-lr` (fine-tune budget,
default 8 steps @ 2e-5), `--confidence-threshold` (skip fine-tuning when the K-NN distance
is low, 0 = always fine-tune), `--regime-ablation` (R1-only DB, evaluates R2/R3/Fault),
`--rebuild-db` (invalidate a stale `.npz` window cache).

---

## 7. Protocol 2 Standalone Eval — `scripts/protocol2_eval.py`

No CLI flags — runs a fixed battery of static baselines + JIT-FT variants on the R1-only
(Protocol 2) split and prints a comparison table. Edit the script's constants to change
checkpoints/encoders.

```bash
PYTHONPATH=. python scripts/protocol2_eval.py
```

---

## Models

| Model | Config | Input | Notes |
|---|---|---|---|
| `beta_vae_pi_lstm` | `beta_vae_pi_lstm.yaml` | 64-dim μ, frozen β-VAE | Main model |
| `cvae_pi_lstm` | `cvae_pi_lstm.yaml` | 64-dim μ, frozen CVAE | |
| `cdae_pi_lstm` | `cdae_pi_lstm.yaml` | 64-dim z, frozen CDAE | |
| `pca_pi_lstm` / `pls_pi_lstm` | `pca_pi_lstm.yaml` / `pls_pi_lstm.yaml` | 4-dim linear Raman latents | Static baselines |
| `pi_lstm` / `process_pi_lstm` | `pi_lstm.yaml` | 23 raw process features | No Raman encoder |
| `neural_ode` | `neural_ode.yaml` | 23 process features | Continuous-time |
| `beta_vae_lstm` | `beta_vae_lstm.yaml` | 64-dim μ | Vanilla LSTM (no physics loss) |
| `beta_vae_liquid`, `beta_vae_mamba`, `beta_vae_lp_pi_lstm`, `beta_vae_latent_lp_pi_lstm`, `delta_cdae_pi_lstm`, `cdae_process_pi_lstm` | matching `.yaml` | various | Architecture ablations |
| `beta_vae_jit_ft`, `cvae_jit_ft`, `vae_jitl` | — (run via `scripts/`) | same latents | On-the-fly adaptation, see §6 |

Architecture: 2-layer LSTM(64) → LayerNorm → Linear(64→32)+LeakyReLU → Linear(32→1)+Sigmoid.
Physics loss: `dP = k_prod * r_net * dt`, λ annealed 0 → 0.1 over epochs 30–110.

---

## Evaluation Protocol

- **Protocol 1**: Train on R1–R3 (batches 0–89, 80/20 random split → 72 train / 18 val). Test on R4 (fault, 90–99).
- **Protocol 2 (regime ablation)**: Train on R1 only (batches 0–29 → 24 train / 6 val). Test on R4 (also reports R2/R3 generalization).
- **Metrics**: RMSE, MAE, R² computed at sparse offline measurement points only (~20/batch, ~200 total), in **g/L**.
- **NRMSE%**: RMSE / 8.481 g/L (mean sparse-point penicillin concentration) × 100.
- **Δ%**: (Test RMSE − Val RMSE) / Val RMSE × 100 — generalization gap.

---

## Layout

```
train.py / evaluate.py / run_seeds.py     Core pipeline entry points (only Python files at repo root)

configs/<model>.yaml                      One YAML per model (data/model/training sections)

scripts/                                  Everything else: training variants, sweeps, adaptation,
                                           analysis. All run from the repo root:
                                           `PYTHONPATH=. python scripts/<script>.py`
    train_autoencoder.py                  Train CDAE / CVAE / β-VAE encoder
    pca_baseline.py, pls_baseline.py, svr_baseline.py, pls_pca_baseline_k64.py
    beta_sweep_e2e.py                     End-to-end β sweep (encoder + LSTM)
    cvae_jit_ft.py, vae_jitl.py            On-the-fly adaptation
    protocol2_eval.py                      Fixed P2 comparison table
    multi_seed_train.py, run_all_seeds.py  End-to-end multi-seed retraining (encoder+LSTM)
    benchmark_summary.py                   Aggregate all results into one table
    train_latent_ode.py, train_neural_cde.py, train_sindyc.py, train_bvae_mlp.py, train_capacity_vae_lstm.py
                                            Alternative temporal-model baselines
    shap_*.py, latent_*.py, evaluate_dci.py, evaluate_mig.py, compute_sap.py, eval_bvae_z52.py,
    analyze_rotation.py, analyze_latent_norm.py, latent_geometry.py, compute_pr_k95_sweep.py,
    beta_sweep_viz.py, plot_pr_scaled.py   Interpretability / latent-space analysis (ad hoc, standalone)
    run_control.sh, run_rotate.sh, run_lstm_seeds.sh, run_fixed_epochs.sh, run_beta_sweep.sh,
    run_ncde_ablations.sh                  Shell orchestration for the scripts above (multi-seed loops)

src/
    config.py                              YAML -> Config object
    latent_norm.py                         Optional latent-scaling intervention (used by train.py)
    autoencoder/                           CDAE_Raman, CVAE_Raman, BetaVAE_Raman
    data/                                  Dataset + per-encoder-type loaders
    models/                                nn.Module definitions
    algorithms/                            fit / evaluate / save / load per model (REGISTRY)
    physics/                               Mass-balance physics-loss terms
    evaluation/                            compute_metrics (RMSE, MAE, R2)
    visualization/                         Training-history + prediction plots

checkpoints/                               Saved weights (not tracked by git)
outputs/                                   All generated results, logs, and plots — nothing is
                                            ever written outside this directory (or checkpoints/)
    seed_sweep/                            run_seeds.py results
    seed_sweep_beta/                       run_beta_sweep.sh results (β sweep via run_seeds.py)
    beta_sweep_e2e/                        beta_sweep_e2e.py results
    full_rotate_experiments_v2/, _v3/      run_rotate.sh / run_lstm_seeds.sh results
    latent_norm_experiments/               run_latent_norm.py results
    benchmark/                             Final comparison table
```

> **Convention**: every script, regardless of where it lives (`scripts/` or root), is meant to be
> invoked **from the repository root** so that relative paths (dataset CSV, `outputs/`,
> `checkpoints/`) resolve correctly — e.g. `PYTHONPATH=. python scripts/latent_geometry.py`,
> not `cd scripts && python latent_geometry.py`.

---

## Reproducibility

- Default seed: 42 (controls train/val split, weight init, data shuffling).
- Use `run_seeds.py` with ≥5 seeds to report mean ± std.
- Checkpoints embed the full config + metrics for traceability.
- cuDNN non-determinism may cause ±1% run-to-run variation on GPU.
- Each adaptation-model evaluation re-encodes Raman spectra with the **exact** encoder
  the evaluated LSTM was trained with (see §6) — mixing encoders across seeds silently
  inflates test RMSE by 10–100x without raising an error, so always double-check
  `--cvae-ckpt`/`--cvae-scaler` (or let `run_seeds.py` resolve them automatically).

---

*Internal project — ReaKt / Genopole. Not for redistribution.*

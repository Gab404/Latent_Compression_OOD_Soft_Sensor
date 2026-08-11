# ReaKt-v2 — Penicillin Soft-Sensor Benchmark

Benchmark comparing multiple model architectures as soft-sensors for online penicillin prediction from Raman spectra on **IndPenSim V3** (100 batches, 2,001 Raman channels).

The core pipeline is: **frozen β-VAE encoder** (Raman → 64-dim latent z) → **PI-LSTM** (z sequence → Penicillin g/L).

---

## Setup

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place `100_Batches_IndPenSim_V3.csv` in the project root (not tracked by git — obtain separately).

---

## Quickstart

```bash
# 1. Train β-VAE encoder (saves to checkpoints/)
PYTHONPATH=. python scripts/train_autoencoder.py --model beta_vae

# 2. Run full β sweep + PI-LSTM training
PYTHONPATH=. python scripts/beta_sweep_e2e.py

# 3. Run static baselines (PLS, PCA-PCR)
PYTHONPATH=. python scripts/pls_pca_baseline_k64.py

# 4. Run alternative temporal models
PYTHONPATH=. python scripts/train_latent_ode.py
PYTHONPATH=. python scripts/train_neural_cde.py
PYTHONPATH=. python scripts/train_sindyc.py

# 5. Aggregate benchmark results
PYTHONPATH=. python scripts/benchmark_summary.py
```

All long-running scripts should be launched in the background:
```bash
nohup .venv/bin/python -u scripts/beta_sweep_e2e.py > outputs/sweep.log 2>&1 &
```

---

## Layout

```
ReaKt-v2/
├── train.py                        # Generic train entry-point (YAML configs)
├── evaluate.py                     # Evaluate one or all checkpoints
│
├── scripts/
│   ├── train_autoencoder.py        # Train CDAE / CVAE / β-VAE encoder
│   ├── beta_sweep_e2e.py           # End-to-end β sweep (encoder + PI-LSTM)
│   ├── pls_pca_baseline_k64.py     # PLS and PCA-PCR static baselines
│   ├── train_bvae_mlp.py           # β-VAE + MLP head
│   ├── train_latent_ode.py         # Latent ODE baseline
│   ├── train_neural_cde.py         # Neural CDE baseline
│   ├── train_sindyc.py             # SINDYc baseline
│   ├── train_capacity_vae_lstm.py  # Capacity-annealing VAE + LSTM
│   ├── benchmark_summary.py        # Aggregate all results into comparison table
│   ├── shap_temporal_profile.py    # SHAP temporal importance profiles
│   └── ...                         # Latent space analysis scripts
│
├── configs/                        # YAML config per model
├── checkpoints/                    # Saved weights (not tracked by git)
├── outputs/                        # Results and plots (JSON results tracked)
│   ├── benchmark/                  # Final comparison table + bar charts
│   ├── beta_sweep_e2e/
│   │   └── sweep_results.json      # β sweep scalar results
│   ├── sindyc/sweep_results.json
│   └── neural_cde/ncde_summary.json
└── src/
    ├── config.py
    ├── autoencoder/                # CDAE_Raman, CVAE_Raman, BetaVAE_Raman
    ├── data/                       # Dataset, encoders (CDAE/CVAE/PCA/PLS/β-VAE)
    ├── models/                     # PyTorch model definitions
    ├── algorithms/                 # fit / evaluate / save / load per model
    ├── physics/                    # Mass-balance loss terms
    ├── evaluation/                 # compute_metrics
    └── visualization/              # Training history and prediction plots
```

---

## Models

| Model | Input | Task |
|---|---|---|
| `β-VAE` | Raman spectra (2001-d) | Encoder-decoder reconstruction; latent z used downstream |
| `β-VAE + PI-LSTM` | 64-dim μ from frozen β-VAE | Penicillin g/L prediction (sparse MSE + physics loss) |
| `β-VAE + MLP` | 64-dim μ from frozen β-VAE | Penicillin g/L prediction (static, no temporal context) |
| `PLS K=64` | Raman spectra (2001-d) | Penicillin g/L (static linear regression) |
| `PCA-PCR K=64` | Raman spectra (2001-d) | Penicillin g/L (static linear regression) |
| `Latent ODE` | 64-dim μ from frozen β-VAE | Penicillin g/L (continuous-time ODE dynamics) |
| `Neural CDE` | 64-dim μ from frozen β-VAE | Penicillin g/L (controlled differential equation) |
| `SINDYc` | 64-dim μ from frozen β-VAE | Penicillin g/L (sparse symbolic dynamics) |

**β-VAE architecture**: 4× Conv1d encoder → flatten → μ/log σ² heads (latent dim=64) → symmetric decoder.  
**PI-LSTM architecture**: 2-layer LSTM(64) → LayerNorm → Linear(64→32)+LeakyReLU → Linear(32→1)+Sigmoid.  
Physics loss (ΔP ≈ k_prod · r_net · dt) annealed λ: 0 → 0.1 over epochs 30–110.

---

## Key Results

Train on R1 (batches 0–29), OOD test on Fault batches (90–99). Metrics at sparse offline measurement points (~20/batch).

| Model | Fault RMSE (g/L) |
|---|---|
| β-VAE + PI-LSTM (β=0.5) | **0.229** |
| PLS K=64 | 0.256 |
| PCA-PCR K=64 | 0.318 |
| β-VAE + PI-LSTM (β=0.05) | 0.334 |
| β-VAE + MLP | 0.549 |
| Latent ODE 64-dim | 1.006 |
| Neural CDE h=32 rk4 | 1.716 |
| SINDYc | 8.64+ |

Full results: `outputs/benchmark/comparison_table.csv`

---

## Reproducibility

Seed 42 throughout. Train/val split applied to fault-free batches only (R1); Fault batches (90–99) are the fixed OOD test set. cuDNN non-determinism may cause ±1% run-to-run variation on GPU.

---

*Internal project — ReaKt / Genopole. Not for redistribution.*

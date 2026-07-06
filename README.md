# ReaKt-v2 — Penicillin Soft-Sensor Benchmark

Physics-informed LSTM soft sensors for online penicillin prediction from Raman spectra, benchmarked on **IndPenSim V3** (100 batches, 2,001 Raman channels).

---

## Setup

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place `100_Batches_IndPenSim_V3.csv` in the project root.  
CDAE/CVAE encoder weights (`cdae_best.pt`, `cvae_best.pt` + scalers) must be present in `checkpoints/`.

---

## Quickstart

```bash
# 1. Fit static baselines and PCA/PLS Raman encoders (run once)
python scripts/pca_baseline.py
python scripts/pls_baseline.py
python scripts/svr_baseline.py

# 2. Train a temporal model
python train.py --model cvae_pi_lstm

# 3. Evaluate
python evaluate.py --model cvae_pi_lstm

# 4. Full benchmark comparison
python evaluate.py --model all --scatter outputs/scatter_all.png
```

**Regime-ablation split** (train on batches 1–30, test on 31–60):

```bash
python train.py    --model cvae_pi_lstm --regime-ablation --ckpt checkpoints/cvae_pi_lstm_regime_ablation.pt
python evaluate.py --model cvae_pi_lstm --regime-ablation --ckpt checkpoints/cvae_pi_lstm_regime_ablation.pt
```

---

## Layout

```
ReaKt-v2/
├── train.py                  # Train any temporal model
├── evaluate.py               # Evaluate 1 or all checkpoints
│
├── scripts/
│   ├── train_autoencoder.py  # Pretrain CDAE / CVAE encoder
│   ├── pca_baseline.py       # Fit PCR baseline + PCA encoder
│   ├── pls_baseline.py       # Fit PLS baseline + PLS encoder
│   ├── svr_baseline.py       # Fit SVR baseline
│   ├── shap_analysis.py      # SHAP XAI for CVAE-PI-LSTM
│   ├── shap_ablation.py      # PI vs vanilla SHAP comparison
│   ├── cvae_jit_ft.py        # CVAE-PI-LSTM with JIT fine-tuning
│   └── vae_jitl.py           # VAE just-in-time learning
│
├── configs/                  # YAML config per temporal model
├── checkpoints/              # Saved weights and fitted encoders
├── outputs/                  # Training plots and SHAP artefacts
└── src/
    ├── config.py
    ├── autoencoder/          # CDAE_Raman, CVAE_Raman
    ├── data/                 # Dataset, encoders (CDAE/CVAE/PCA/PLS)
    ├── models/               # PyTorch model definitions
    ├── algorithms/           # fit / evaluate / save / load per model
    ├── physics/              # Mass-balance loss terms
    ├── evaluation/           # compute_metrics, print_metrics_table
    └── visualization/        # Training history and prediction plots
```

---

## Models

| Key | Input | Encoder | Notes |
|---|---|---|---|
| `pi_lstm` | 23 process vars | — | Logistic growth + Luedeking-Piret physics |
| `neural_ode` | 23 process vars | — | MLP right-hand-side, torchdiffeq |
| `process_pi_lstm` | 23 process vars | — | Same arch as Raman variants, no spectra |
| `cdae_pi_lstm` | Raman | CDAE 64-d | Non-linear unsupervised encoder |
| `cvae_pi_lstm` | Raman | CVAE 64-d | Posterior mean only at inference |
| `pca_pi_lstm` | Raman | PCA 4-d | Linear unsupervised |
| `pls_pi_lstm` | Raman | PLS 4-d | Linear supervised (target: P) |
| `cdae_process_pi_lstm` | Raman + process | CDAE 64-d | Best fault-batch RMSE |
| `delta_cdae_pi_lstm` | Raman | CDAE 64-d | ΔP head, cumsum at inference |

All Raman variants: SG 1st derivative → StandardScaler → encoder → MinMaxScaler → 2-layer LSTM(64) → FC(32) → P head.  
Physics loss (`ΔP ≈ k_prod · r_net · dt`) annealed 0 → 0.1 over epochs 30–110.

---

## Results

Metrics at sparse offline measurement points only (~20/batch). Split: seed=42, 72 train / 18 FF-val / 10 fault-test.

### Penicillin (g/L)

| Model | Val RMSE | Val R² | Test RMSE | Test R² |
|---|---|---|---|---|
| pi_lstm | 1.558 | 0.9765 | 1.629 | 0.9641 |
| process_pi_lstm | 1.707 | 0.9718 | 1.703 | 0.9608 |
| pca_pi_lstm | 0.128 | 0.9998 | 0.524 | 0.9963 |
| pls_pi_lstm | 0.111 | 0.9999 | 0.520 | 0.9963 |
| **cdae_pi_lstm** | **0.088** | **0.9999** | 0.336 | 0.9985 |
| cvae_pi_lstm | 0.122 | 0.9999 | 0.135 | 0.9998 |
| cdae_process_pi_lstm | 0.126 | 0.9998 | **0.227** | **0.9993** |

### Regime ablation (train R1 1–30, test R2 31–60)

| Model | Val RMSE | Test RMSE | RMSE Δ |
|---|---|---|---|
| cvae_pi_lstm | 0.151 | 0.335 | +122% |
| pls_pi_lstm | 0.428 | 0.473 | +11% |
| pi_lstm (Pen) | 1.533 | 2.217 | +45% |

---

## Reproducibility

Seed 42 throughout. Split applied to fault-free batches only; faulty batches (91–100) form the fixed test set.  
cuDNN non-determinism may cause ±1% run-to-run variation on GPU.

---

*Internal project — ReaKt / Genopole. Not for redistribution.*

"""
src/algorithms/process_pi_lstm.py
==================================
ProcessPILSTMAlgorithm — PI-LSTM driven exclusively by the 23 process state
variables (temperature, pH, dissolved O₂, substrate, aeration, feeds, …).
No Raman spectra, no encoder.

Architecture
------------
  Input  : (B, T, 23)  — MinMaxScaled process variables
  Model  : CDAEPILSTMModel(input_size=23)
             2-layer LSTM(23→64) + LayerNorm + FC(32) + LeakyReLU
             → head_pen  (Sigmoid → pen_norm)
             → head_rate (linear  → r_net)
  Physics: ΔP ≈ k_prod · r_net · dt  (same simplified mass balance as all
           other Raman-encoder variants)

Role in the benchmark
----------------------
  The existing ``pi_lstm`` predicts both Biomass and Penicillin using the full
  Luedeking-Piret / logistic-growth physics (4 learnable ODE parameters).
  This variant isolates the process-variable signal on the Penicillin task
  alone, with the same architecture and physics loss as the Raman variants,
  making it a direct apples-to-apples comparator:

    cdae_pi_lstm          — Raman (CDAE 64-d) only
    process_pi_lstm       — process variables (23-d) only      ← this model
    cdae_process_pi_lstm  — Raman + process variables (87-d)

  Comparing these three reveals how much of the predictive power comes from
  the Raman signal versus the process state, and how much they complement each
  other.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.base import REGISTRY, BaseAlgorithm, ScalerBundle
from src.config import Config
from src.data.dataset import PROCESS_FEATURE_COLS, TARGET_COLS
from src.evaluation.metrics import compute_metrics
from src.models.cdae_pi_lstm import CDAEPILSTMModel
from src.physics.losses import compute_cdae_pinn_loss


class ProcessPILSTMAlgorithm(BaseAlgorithm):
    """
    PI-LSTM for Penicillin prediction using process state variables only.

    Config sections expected
    ------------------------
    model.hidden_size       : 64
    model.num_lstm_layers   : 2
    model.fc_hidden         : 32
    model.lstm_dropout      : 0.2
    model.k_prod_init       : 1.0
    training.seq_len        : 24
    training.dt             : 0.2
    training.batch_size     : 64
    training.n_epochs       : 200
    training.lr             : 1e-3
    training.lr_physics     : 1e-2
    training.weight_decay   : 1e-4
    training.lr_factor      : 0.5
    training.lr_patience    : 15
    training.es_patience    : 30
    training.grad_clip      : 5.0
    training.vanilla_mode           : false
    training.lambda_physics_start   : 0.0
    training.lambda_physics_end     : 0.1
    training.lambda_anneal_delay    : 30
    training.lambda_anneal_epochs   : 80
    """

    name = "process_pi_lstm"

    def __init__(self, cfg: Config, device: torch.device) -> None:
        super().__init__(cfg, device)
        self._dt_n: float = 0.0

    # ── Feature columns ───────────────────────────────────────────────────────

    def _feature_cols(self) -> List[str]:
        return list(PROCESS_FEATURE_COLS)   # 23 columns, fixed order

    # ── Abstract implementations ──────────────────────────────────────────────

    def _build_model(self) -> CDAEPILSTMModel:
        m      = self.cfg.model
        n_feat = len(self._feature_cols())   # 23
        return CDAEPILSTMModel(
            input_size=n_feat,
            hidden_size=int(m.hidden_size),
            num_lstm_layers=int(m.num_lstm_layers),
            fc_hidden=int(m.fc_hidden),
            lstm_dropout=float(m.lstm_dropout),
            k_prod_init=float(m.get("k_prod_init", 1.0)),
        )

    def _fit_scalers(self, train_batches) -> ScalerBundle:
        feat_cols = self._feature_cols()

        X_all   = np.vstack([b[feat_cols].values for b in train_batches])
        bio_all = np.concatenate(
            [b[TARGET_COLS[0]].values for b in train_batches]
        ).reshape(-1, 1)
        pen_all = np.concatenate(
            [b[TARGET_COLS[1]].values for b in train_batches]
        ).reshape(-1, 1)

        feat_scaler = MinMaxScaler().fit(X_all)
        bio_scaler  = MinMaxScaler().fit(bio_all)
        pen_scaler  = MinMaxScaler().fit(pen_all)

        # dt_n: Time (h) is the first column of PROCESS_FEATURE_COLS
        t_all    = np.concatenate([b["Time (h)"].values for b in train_batches])
        t_range  = float(t_all.max() - t_all.min())
        self._dt_n = float(self.cfg.training.dt) / max(t_range, 1e-8)

        print(
            f"  Scalers fitted  |  "
            f"input_dim={X_all.shape[1]} (process only)  "
            f"n_train_rows={X_all.shape[0]:,}  "
            f"dt_n={self._dt_n:.6f}"
        )
        return ScalerBundle(feat=feat_scaler, bio=bio_scaler, pen=pen_scaler)

    def _build_dataset(self, batches, scalers: ScalerBundle) -> TensorDataset:
        feat_cols = self._feature_cols()
        seq_len   = int(self.cfg.training.seq_len)
        X_list, y_list = [], []

        for b in batches:
            feats = scalers.feat.transform(
                b[feat_cols].values
            ).astype(np.float32)
            pen_n = scalers.pen.transform(
                b[TARGET_COLS[1]].values.reshape(-1, 1)
            ).flatten().astype(np.float32)

            T = len(feats)
            for i in range(T - seq_len + 1):
                X_list.append(feats[i : i + seq_len])
                y_list.append(pen_n[i : i + seq_len])

        X = torch.tensor(np.array(X_list, dtype=np.float32))
        y = torch.tensor(np.array(y_list, dtype=np.float32))
        print(
            f"  Sliding-window dataset: {len(X_list):,} windows  "
            f"(seq_len={seq_len}, n_feat={X.shape[-1]})"
        )
        return TensorDataset(X, y)

    def _train_epoch(
        self,
        loader: DataLoader,
        lambda_physics: float = 0.0,
    ) -> Dict[str, float]:
        grad_clip = float(self.cfg.training.grad_clip)
        ep_total, ep_data, ep_phys = [], [], []

        for x_b, y_b in loader:
            x_b = x_b.to(self.device)
            y_b = y_b.to(self.device)

            self._optimizer.zero_grad(set_to_none=True)
            pred = self.model(x_b)
            loss, ld = compute_cdae_pinn_loss(
                model=self.model,
                pred=pred,
                y_pen_norm=y_b,
                dt_n=self._dt_n,
                lambda_physics=lambda_physics,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self._optimizer.step()

            ep_total.append(ld["loss"])
            ep_data.append(ld["loss_data"])
            ep_phys.append(ld["loss_phys"])

        return {
            "train_loss":  float(np.mean(ep_total)),
            "train_data":  float(np.mean(ep_data)),
            "train_phys":  float(np.mean(ep_phys)),
            "lambda_phys": float(lambda_physics),
            "k_prod":      float(self.model.k_prod.item()),
        }

    def _val_epoch(self, loader: DataLoader) -> float:
        val_mses: List[float] = []
        with torch.no_grad():
            for x_b, y_b in loader:
                x_b = x_b.to(self.device)
                y_b = y_b.to(self.device)
                pred = self.model(x_b)
                val_mses.append(F.mse_loss(pred[:, :, 0], y_b).item())
        return float(np.mean(val_mses))

    def evaluate(self, batches) -> Dict[str, Any]:
        """Sliding-window inference; metrics at sparse offline measurement points."""
        self.model.eval()
        feat_cols = self._feature_cols()
        seq_len   = int(self.cfg.training.seq_len)

        pen_true:      List[np.ndarray] = []
        pen_pred_list: List[np.ndarray] = []
        y_preds_pen:   List[np.ndarray] = []
        y_sparses_pen: List[np.ndarray] = []
        y_preds_bio:   List[np.ndarray] = []
        y_sparses_bio: List[np.ndarray] = []

        with torch.no_grad():
            for b in batches:
                feats  = self.scalers.feat.transform(
                    b[feat_cols].values
                ).astype(np.float32)
                sp_pen = b["_penicillin_sparse"].values
                sp_bio = b["_biomass_sparse"].values
                T      = len(feats)

                if T < seq_len:
                    continue

                N_win   = T - seq_len + 1
                windows = np.stack([feats[i : i + seq_len] for i in range(N_win)])
                x_all   = torch.tensor(
                    windows, dtype=torch.float32, device=self.device
                )

                preds_pen_n = np.full(T, np.nan)
                for start in range(0, N_win, 512):
                    chunk   = x_all[start : start + 512]
                    out_pen = self.model(chunk)[:, -1, 0].cpu().numpy()
                    s = seq_len - 1 + start
                    preds_pen_n[s : s + len(out_pen)] = out_pen

                valid_mask    = ~np.isnan(preds_pen_n)
                pred_pen_full = np.full(T, np.nan)
                if valid_mask.any():
                    pred_pen_full[valid_mask] = self.scalers.pen.inverse_transform(
                        preds_pen_n[valid_mask].reshape(-1, 1)
                    ).flatten()

                y_preds_pen.append(pred_pen_full)
                y_sparses_pen.append(sp_pen.copy())

                mask_pen = ~np.isnan(sp_pen) & ~np.isnan(pred_pen_full)
                if mask_pen.sum() > 0:
                    pen_true.append(sp_pen[mask_pen])
                    pen_pred_list.append(pred_pen_full[mask_pen])

                y_preds_bio.append(np.full(T, np.nan))
                y_sparses_bio.append(sp_bio.copy())

        if pen_true:
            pen_metrics = compute_metrics(
                np.concatenate(pen_true), np.concatenate(pen_pred_list)
            )
        else:
            pen_metrics = {"RMSE": float("nan"), "MAE": float("nan"),
                           "R2": float("nan"), "n": 0}

        bio_metrics = {"RMSE": float("nan"), "MAE": float("nan"),
                       "R2": float("nan"), "n": 0}

        print(
            f"  [Penicillin] "
            f"RMSE={pen_metrics.get('RMSE', float('nan')):.3f} g/L  "
            f"MAE={pen_metrics.get('MAE', float('nan')):.3f} g/L  "
            f"R2={pen_metrics.get('R2', float('nan')):.4f}  "
            f"n={pen_metrics.get('n', 0)}"
        )
        print("  [Biomass   ] N/A (Process-PI-LSTM predicts Penicillin only)")

        return {
            "pen":           pen_metrics,
            "bio":           bio_metrics,
            "y_preds_pen":   y_preds_pen,
            "y_sparses_pen": y_sparses_pen,
            "y_preds_bio":   y_preds_bio,
            "y_sparses_bio": y_sparses_bio,
        }

    # ── Overrides ─────────────────────────────────────────────────────────────

    def _make_optimizer(self) -> torch.optim.Optimizer:
        tr = self.cfg.training
        return torch.optim.AdamW([
            {
                "params":       self.model.network_parameters(),
                "lr":           float(tr.lr),
                "weight_decay": float(tr.weight_decay),
            },
            {
                "params":       self.model.physics_parameters(),
                "lr":           float(tr.lr_physics),
                "weight_decay": 0.0,
            },
        ])

    def _is_es_active(self, epoch: int) -> bool:
        tr      = self.cfg.training
        vanilla = bool(getattr(tr, "vanilla_mode", False))
        if vanilla:
            return True
        delay = int(getattr(tr, "lambda_anneal_delay", 0))
        return epoch > delay

    def _epoch_extra_kwargs(self, epoch: int) -> Dict[str, Any]:
        tr      = self.cfg.training
        vanilla = bool(getattr(tr, "vanilla_mode", False))
        if vanilla:
            return {"lambda_physics": 0.0}

        delay         = int(getattr(tr, "lambda_anneal_delay", 0))
        anneal_epochs = max(int(getattr(tr, "lambda_anneal_epochs", 1)), 1)
        lam_start     = float(getattr(tr, "lambda_physics_start", 0.0))
        lam_end       = float(getattr(tr, "lambda_physics_end",   0.1))

        epoch_after_delay = epoch - delay
        if epoch_after_delay <= 0:
            frac = 0.0
        elif anneal_epochs <= 1:
            frac = 1.0
        else:
            frac = min((epoch_after_delay - 1) / (anneal_epochs - 1), 1.0)

        return {"lambda_physics": lam_start + (lam_end - lam_start) * frac}


# ── Register ──────────────────────────────────────────────────────────────────
REGISTRY[ProcessPILSTMAlgorithm.name] = ProcessPILSTMAlgorithm

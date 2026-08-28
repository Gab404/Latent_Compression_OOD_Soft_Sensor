"""
src/algorithms/pi_lstm.py
=========================
PILSTMAlgorithm — wraps BioreactorLogiPINN with a full fit/evaluate pipeline.

Key design choices
------------------
- Sliding-window dataset (seq_len windows, no cross-batch overlap)
- Two AdamW param groups: lr=1e-3 for network, lr=1e-2 for physics params
- Physics annealing: lambda ramps 0 → lambda_end over ``lambda_anneal_epochs``
  epochs, after a ``lambda_anneal_delay``-epoch data-only warmup phase
- Early stopping is frozen during the delay phase (prevents ES from firing
  before the ODE constraint has had any effect)
- Evaluation: vectorised sliding-window inference at sparse measurement times
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.base import REGISTRY, BaseAlgorithm, ScalerBundle
from src.config import Config
from src.data.dataset import FEATURE_PRESETS, RAMAN_LATENT_DIM, TARGET_COLS
from src.models.pi_lstm import BioreactorLogiPINN
from src.physics.losses import compute_pinn_loss


class PILSTMAlgorithm(BaseAlgorithm):
    """
    Physics-Informed LSTM algorithm.

    Config sections expected
    ------------------------
    data.feature_preset   : "process_23"   (maps to 23-col list)
    data.use_raman        : bool
    model.hidden_size     : 64
    model.num_lstm_layers : 1
    model.fc_hidden       : 32
    model.lstm_dropout    : 0.0
    model.r_n_init        : 45.0
    model.y_max_n_init    : 1.05
    model.alpha_n_init    : 1.0
    model.beta_n_init     : 1.0
    training.seq_len      : 24
    training.dt           : 0.2
    training.batch_size   : 64
    training.n_epochs     : 200
    training.lr           : 1e-3
    training.lr_physics   : 1e-2
    training.weight_decay : 1e-4
    training.lr_factor    : 0.5
    training.lr_patience  : 15
    training.es_patience  : 30
    training.grad_clip    : 5.0
    training.vanilla_mode           : false
    training.lambda_physics_start   : 0.0
    training.lambda_physics_end     : 0.1
    training.lambda_anneal_delay    : 30
    training.lambda_anneal_epochs   : 80
    training.lambda_r_neg           : 1.0
    """

    name = "pi_lstm"

    def __init__(self, cfg: Config, device: torch.device):
        super().__init__(cfg, device)
        self._dt_n: float = 0.0   # normalised dt; filled in _fit_scalers

    # ── Feature column helper ─────────────────────────────────────────────────

    def _feature_cols(self) -> List[str]:
        preset = self.cfg.data.get("feature_preset", "process_23")
        cols   = list(FEATURE_PRESETS[preset])
        if self.cfg.data.get("use_raman", False):
            # raman_latent_dim in config allows V4 (512-d), V5 (32-d), or CDAE (64-d)
            # to all work without changing this code.
            n_raman = int(self.cfg.data.get("raman_latent_dim", RAMAN_LATENT_DIM))
            cols = cols + [f"_raman_{i}" for i in range(n_raman)]
        return cols

    # ── Abstract implementations ──────────────────────────────────────────────

    def _build_model(self) -> BioreactorLogiPINN:
        m   = self.cfg.model
        n_feat = len(self._feature_cols())
        return BioreactorLogiPINN(
            input_size=n_feat,
            hidden_size=int(m.hidden_size),
            num_lstm_layers=int(m.num_lstm_layers),
            fc_hidden=int(m.fc_hidden),
            lstm_dropout=float(m.lstm_dropout),
            r_n_init=float(m.r_n_init),
            y_max_n_init=float(m.y_max_n_init),
            alpha_n_init=float(m.alpha_n_init),
            beta_n_init=float(m.beta_n_init),
        )

    def _fit_scalers(self, train_batches) -> ScalerBundle:
        feat_cols = self._feature_cols()

        X_all  = np.vstack([b[feat_cols].values for b in train_batches])
        bio_all = np.concatenate(
            [b[TARGET_COLS[0]].values for b in train_batches]
        ).reshape(-1, 1)
        pen_all = np.concatenate(
            [b[TARGET_COLS[1]].values for b in train_batches]
        ).reshape(-1, 1)

        feat_scaler = MinMaxScaler()
        bio_scaler  = MinMaxScaler()
        pen_scaler  = MinMaxScaler()
        feat_scaler.fit(X_all)
        bio_scaler.fit(bio_all)
        pen_scaler.fit(pen_all)

        # Compute normalised dt (used in the physics loss)
        t_range    = float(feat_scaler.data_range_[0])   # time column is index 0
        self._dt_n = float(self.cfg.training.dt) / max(t_range, 1e-8)

        return ScalerBundle(feat=feat_scaler, bio=bio_scaler, pen=pen_scaler)

    def _build_dataset(self, batches, scalers: ScalerBundle) -> TensorDataset:
        feat_cols = self._feature_cols()
        seq_len   = int(self.cfg.training.seq_len)
        X_parts, y_parts = [], []

        for b in batches:
            feats = scalers.feat.transform(
                b[feat_cols].values
            ).astype(np.float32)
            bio_n = scalers.bio.transform(
                b[TARGET_COLS[0]].values.reshape(-1, 1)
            ).flatten().astype(np.float32)
            pen_n = scalers.pen.transform(
                b[TARGET_COLS[1]].values.reshape(-1, 1)
            ).flatten().astype(np.float32)
            tgts = np.stack([bio_n, pen_n], axis=-1)   # (T, 2)

            T = len(feats)
            if T < seq_len:
                continue

            # Vectorised sliding windows — no Python loop, O(1) allocations
            n_feat = feats.shape[1]
            X_win = np.lib.stride_tricks.sliding_window_view(
                feats, (seq_len, n_feat)
            )[:, 0].copy()                              # (N_win, seq_len, n_feat)
            y_win = np.lib.stride_tricks.sliding_window_view(
                tgts, (seq_len, 2)
            )[:, 0].copy()                              # (N_win, seq_len, 2)
            X_parts.append(X_win)
            y_parts.append(y_win)

        X = torch.from_numpy(np.concatenate(X_parts, axis=0))
        y = torch.from_numpy(np.concatenate(y_parts, axis=0))
        print(f"  Sliding-window dataset: {len(X):,} windows  "
              f"(seq_len={seq_len}, n_feat={X.shape[-1]})")
        return TensorDataset(X, y)

    def _train_epoch(self, loader: DataLoader, lambda_physics: float = 0.0) -> Dict[str, float]:
        grad_clip  = float(self.cfg.training.grad_clip)
        lambda_r   = float(self.cfg.training.lambda_r_neg)
        use_amp    = self._use_amp
        ep_total, ep_data, ep_phys = [], [], []

        for x_b, y_b in loader:
            if x_b.device != self.device:
                x_b = x_b.to(self.device, non_blocking=True)
                y_b = y_b.to(self.device, non_blocking=True)

            self._optimizer.zero_grad(set_to_none=True)

            # Wrap the full forward+loss in bfloat16 autocast.
            # The LSTM GEMMs (dominant cost) run on Tensor Cores in BF16.
            # BF16 has the same exponent range as FP32 so no overflow risk.
            with torch.autocast(device_type=self.device.type,
                                dtype=torch.bfloat16, enabled=use_amp):
                loss, ld = compute_pinn_loss(
                    self.model,
                    x_b,
                    y_b,
                    dt_n=self._dt_n,
                    lambda_physics=lambda_physics,
                    lambda_r_neg=lambda_r,
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
            "r_n":         float(self.model.r.item()),
            "y_max_n":     float(self.model.y_max.item()),
        }

    def _val_epoch(self, loader: DataLoader) -> float:
        use_amp    = self._use_amp
        total_loss = torch.zeros(1, device=self.device)
        n_batches  = 0
        with torch.no_grad():
            for x_b, y_b in loader:
                if x_b.device != self.device:
                    x_b = x_b.to(self.device, non_blocking=True)
                    y_b = y_b.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type,
                                    dtype=torch.bfloat16, enabled=use_amp):
                    pred = self.model(x_b)
                total_loss += F.mse_loss(pred.float(), y_b.float())
                n_batches  += 1
        return (total_loss / max(n_batches, 1)).item()

    def evaluate(self, batches) -> Dict[str, Any]:
        """
        Sliding-window inference; metrics computed at sparse measurement times.

        Returns
        -------
        {"bio": {RMSE, MAE, R2, n}, "pen": {...},
         "y_preds_bio": [...], "y_preds_pen": [...],
         "y_sparses_bio": [...], "y_sparses_pen": [...]}
        """
        self.model.eval()
        feat_cols = self._feature_cols()
        seq_len   = int(self.cfg.training.seq_len)

        bio_true, bio_pred_list = [], []
        pen_true, pen_pred_list = [], []
        y_preds_bio:   list = []
        y_preds_pen:   list = []
        y_sparses_bio: list = []
        y_sparses_pen: list = []

        with torch.no_grad():
            for b in batches:
                feats   = self.scalers.feat.transform(
                    b[feat_cols].values
                ).astype(np.float32)
                sp_bio  = b["_biomass_sparse"].values
                sp_pen  = b["_penicillin_sparse"].values
                T       = len(feats)

                if T < seq_len:
                    continue

                # Build all windows at once — vectorised
                N_win   = T - seq_len + 1
                windows = np.stack([feats[i : i + seq_len] for i in range(N_win)])
                x_all   = torch.tensor(windows, dtype=torch.float32, device=self.device)

                preds_n = np.full((T, 2), np.nan)
                for start in range(0, N_win, 512):
                    chunk = x_all[start : start + 512]
                    out   = self.model(chunk)[:, -1, :].cpu().numpy()   # last step
                    preds_n[seq_len - 1 + start : seq_len - 1 + start + len(out)] = out

                # Biomass — inverse transform + collect sparse points
                pred_bio_full = self.scalers.bio.inverse_transform(
                    np.where(np.isnan(preds_n[:, 0:1]), 0.0, preds_n[:, 0:1])
                ).flatten()
                pred_bio_full[np.isnan(preds_n[:, 0])] = np.nan
                y_preds_bio.append(pred_bio_full)
                y_sparses_bio.append(sp_bio.copy())

                mask_bio = ~np.isnan(sp_bio) & ~np.isnan(preds_n[:, 0])
                if mask_bio.sum() > 0:
                    pred_bio_sp = self.scalers.bio.inverse_transform(
                        preds_n[mask_bio, 0:1]
                    ).flatten()
                    bio_true.append(sp_bio[mask_bio])
                    bio_pred_list.append(pred_bio_sp)

                # Penicillin — inverse transform + collect sparse points
                pred_pen_full = self.scalers.pen.inverse_transform(
                    np.where(np.isnan(preds_n[:, 1:2]), 0.0, preds_n[:, 1:2])
                ).flatten()
                pred_pen_full[np.isnan(preds_n[:, 1])] = np.nan
                y_preds_pen.append(pred_pen_full)
                y_sparses_pen.append(sp_pen.copy())

                mask_pen = ~np.isnan(sp_pen) & ~np.isnan(preds_n[:, 1])
                if mask_pen.sum() > 0:
                    pred_pen_sp = self.scalers.pen.inverse_transform(
                        preds_n[mask_pen, 1:2]
                    ).flatten()
                    pen_true.append(sp_pen[mask_pen])
                    pen_pred_list.append(pred_pen_sp)

        from src.evaluation.metrics import aggregate_metrics
        metrics = aggregate_metrics(bio_true, bio_pred_list, pen_true, pen_pred_list)
        metrics["y_preds_bio"]    = y_preds_bio
        metrics["y_preds_pen"]    = y_preds_pen
        metrics["y_sparses_bio"]  = y_sparses_bio
        metrics["y_sparses_pen"]  = y_sparses_pen
        return metrics

    # ── Overrides ─────────────────────────────────────────────────────────────

    def _make_optimizer(self) -> torch.optim.Optimizer:
        """Two param groups: network (lr=1e-3, wd=1e-4) / physics (lr=1e-2, wd=0)."""
        tr = self.cfg.training
        physics_names = {"_r_raw", "_ymax_raw", "_alpha_raw", "_beta_raw"}
        physics_params = [
            p for n, p in self.model.named_parameters() if n in physics_names
        ]
        network_params = [
            p for n, p in self.model.named_parameters() if n not in physics_names
        ]
        return torch.optim.AdamW([
            {"params": network_params,
             "lr": float(tr.lr), "weight_decay": float(tr.weight_decay)},
            {"params": physics_params,
             "lr": float(tr.lr_physics), "weight_decay": 0.0},
        ])

    def _is_es_active(self, epoch: int) -> bool:
        """
        Freeze ES during the physics delay phase in PINN mode.

        In vanilla_mode the patience runs from epoch 1.
        In PINN mode the patience starts only after lambda_physics > 0.
        """
        tr = self.cfg.training
        vanilla = bool(getattr(tr, "vanilla_mode", False))
        if vanilla:
            return True
        delay = int(getattr(tr, "lambda_anneal_delay", 0))
        return epoch > delay  # active once ramp has started

    def _epoch_extra_kwargs(self, epoch: int) -> Dict[str, Any]:
        """Compute and return the annealed lambda_physics for this epoch."""
        tr      = self.cfg.training
        vanilla = bool(getattr(tr, "vanilla_mode", False))
        if vanilla:
            return {"lambda_physics": 0.0}

        delay         = int(getattr(tr, "lambda_anneal_delay", 0))
        anneal_epochs = max(int(getattr(tr, "lambda_anneal_epochs", 1)), 1)
        lam_start     = float(getattr(tr, "lambda_physics_start", 0.0))
        lam_end       = float(getattr(tr, "lambda_physics_end", 0.1))

        epoch_after_delay = epoch - delay
        if epoch_after_delay <= 0:
            frac = 0.0
        elif anneal_epochs <= 1:
            frac = 1.0
        else:
            frac = min((epoch_after_delay - 1) / (anneal_epochs - 1), 1.0)

        lambda_physics = lam_start + (lam_end - lam_start) * frac
        return {"lambda_physics": lambda_physics}


# Register
REGISTRY[PILSTMAlgorithm.name] = PILSTMAlgorithm

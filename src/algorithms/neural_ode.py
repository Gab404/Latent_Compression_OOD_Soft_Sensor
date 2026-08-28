"""
src/algorithms/neural_ode.py
============================
NeuralODEAlgorithm — wraps NeuralODEModel with a full fit/evaluate pipeline.

Key design choices
------------------
- Segment dataset: seg_len=50 steps, stride=10 (teacher forcing at boundaries)
- Training solver: rk4 (fast, fixed-step)
- Evaluation solver: dopri5 (adaptive, accurate)
- Full-trajectory ODE evaluation from y0 — no teacher forcing at test time
- Single AdamW param group (no physics parameters to separate)
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
from src.models.neural_ode import NeuralODEModel


class NeuralODEAlgorithm(BaseAlgorithm):
    """
    Neural ODE algorithm  (d[X,P]/dt = f_θ([X,P], U)).

    Config sections expected
    ------------------------
    data.feature_preset  : "process_23"
    data.use_raman       : bool
    model.n_state        : 2
    model.hidden         : 64
    training.seg_len     : 50
    training.seg_stride  : 10
    training.dt          : 0.2
    training.batch_size  : 64
    training.n_epochs    : 150
    training.lr          : 1e-3
    training.weight_decay: 1e-4
    training.lr_factor   : 0.5
    training.lr_patience : 15
    training.es_patience : 30
    training.grad_clip   : 5.0
    training.train_solver: "rk4"
    training.eval_solver : "dopri5"
    """

    name = "neural_ode"

    def __init__(self, cfg: Config, device: torch.device):
        super().__init__(cfg, device)
        # Precompute the relative time grid for segment-level training
        seg_len      = int(cfg.training.seg_len)
        dt           = float(cfg.training.dt)
        self._t_span = torch.arange(seg_len, dtype=torch.float32).to(device) * dt

    # ── Feature column helper ─────────────────────────────────────────────────

    def _feature_cols(self) -> List[str]:
        preset = self.cfg.data.get("feature_preset", "process_23")
        cols   = list(FEATURE_PRESETS[preset])
        if self.cfg.data.get("use_raman", False):
            n_raman = int(self.cfg.data.get("raman_latent_dim", RAMAN_LATENT_DIM))
            cols = cols + [f"_raman_{i}" for i in range(n_raman)]
        return cols

    # ── Abstract implementations ──────────────────────────────────────────────

    def _build_model(self) -> NeuralODEModel:
        m      = self.cfg.model
        n_ctrl = len(self._feature_cols())
        return NeuralODEModel(
            n_state=int(m.n_state),
            n_ctrl=n_ctrl,
            hidden=int(m.hidden),
        )

    def _fit_scalers(self, train_batches) -> ScalerBundle:
        feat_cols = self._feature_cols()

        feat_all = np.vstack([b[feat_cols].values for b in train_batches])
        bio_all  = np.concatenate(
            [b[TARGET_COLS[0]].values for b in train_batches]
        ).reshape(-1, 1)
        pen_all  = np.concatenate(
            [b[TARGET_COLS[1]].values for b in train_batches]
        ).reshape(-1, 1)

        feat_scaler = MinMaxScaler()
        bio_scaler  = MinMaxScaler()
        pen_scaler  = MinMaxScaler()
        feat_scaler.fit(feat_all)
        bio_scaler.fit(bio_all)
        pen_scaler.fit(pen_all)

        return ScalerBundle(feat=feat_scaler, bio=bio_scaler, pen=pen_scaler)

    def _build_dataset(self, batches, scalers: ScalerBundle) -> TensorDataset:
        """
        Slide a window of length ``seg_len`` over each batch.

        Each sample: (U_seg, y0_seg, y_seg)
            U_seg  : (seg_len, n_ctrl) — normalised control inputs
            y0_seg : (2,)              — normalised [X0, P0] (teacher forcing)
            y_seg  : (seg_len, 2)      — normalised [X, P] targets
        """
        feat_cols  = self._feature_cols()
        seg_len    = int(self.cfg.training.seg_len)
        stride     = int(self.cfg.training.seg_stride)

        U_list:  list = []
        y0_list: list = []
        y_list:  list = []

        for b in batches:
            U_all = scalers.feat.transform(
                b[feat_cols].values
            ).astype(np.float32)                             # (T, n_ctrl)

            bio_n = scalers.bio.transform(
                b[TARGET_COLS[0]].values.reshape(-1, 1)
            ).astype(np.float32)                             # (T, 1)
            pen_n = scalers.pen.transform(
                b[TARGET_COLS[1]].values.reshape(-1, 1)
            ).astype(np.float32)                             # (T, 1)

            y_all = np.concatenate([bio_n, pen_n], axis=-1)  # (T, 2)
            T = len(b)
            for start in range(0, T - seg_len + 1, stride):
                end = start + seg_len
                U_list.append(U_all[start:end])   # (seg_len, n_ctrl)
                y0_list.append(y_all[start])       # (2,)
                y_list.append(y_all[start:end])    # (seg_len, 2)

        U_t  = torch.tensor(np.array(U_list),  dtype=torch.float32)
        y0_t = torch.tensor(np.array(y0_list), dtype=torch.float32)
        y_t  = torch.tensor(np.array(y_list),  dtype=torch.float32)
        print(f"  Segment dataset: {len(U_list):,} segments  "
              f"(seg_len={seg_len}, stride={stride}, n_ctrl={U_t.shape[-1]})")
        return TensorDataset(U_t, y0_t, y_t)

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        solver     = str(self.cfg.training.train_solver)
        grad_clip  = float(self.cfg.training.grad_clip)
        t_span     = self._t_span
        use_amp    = self._use_amp

        total_loss = 0.0
        n_samples  = 0

        for U_b, y0_b, y_b in loader:
            if U_b.device != self.device:
                U_b  = U_b.to(self.device,  non_blocking=True)
                y0_b = y0_b.to(self.device, non_blocking=True)
                y_b  = y_b.to(self.device,  non_blocking=True)

            self._optimizer.zero_grad()

            with torch.autocast(device_type=self.device.type,
                                dtype=torch.bfloat16, enabled=use_amp):
                traj = self.model(y0_b, t_span, U_b, method=solver)
            loss = F.mse_loss(traj.float(), y_b.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self._optimizer.step()

            total_loss += loss.item() * len(y0_b)
            n_samples  += len(y0_b)

        return {"train_loss": total_loss / max(n_samples, 1)}

    def _val_epoch(self, loader: DataLoader) -> float:
        solver    = str(self.cfg.training.train_solver)
        t_span    = self._t_span
        use_amp   = self._use_amp
        total     = torch.zeros(1, device=self.device)
        n_batches = 0

        with torch.no_grad():
            for U_b, y0_b, y_b in loader:
                if U_b.device != self.device:
                    U_b  = U_b.to(self.device,  non_blocking=True)
                    y0_b = y0_b.to(self.device, non_blocking=True)
                    y_b  = y_b.to(self.device,  non_blocking=True)
                with torch.autocast(device_type=self.device.type,
                                    dtype=torch.bfloat16, enabled=use_amp):
                    traj = self.model(y0_b, t_span, U_b, method=solver)
                total     += F.mse_loss(traj.float(), y_b.float())
                n_batches += 1

        return (total / max(n_batches, 1)).item()

    def evaluate(self, batches) -> Dict[str, Any]:
        """
        Full-trajectory ODE integration from y0 (no teacher forcing).

        Uses ``dopri5`` adaptive solver for accurate long-horizon rollouts.

        Returns
        -------
        {"bio": {RMSE, MAE, R2, n}, "pen": {...},
         "y_preds_bio": [...], "y_preds_pen": [...],
         "y_sparses_bio": [...], "y_sparses_pen": [...]}
        """
        self.model.eval()
        feat_cols   = self._feature_cols()
        eval_solver = str(self.cfg.training.eval_solver)
        dt          = float(self.cfg.training.dt)

        bio_true, bio_pred_list = [], []
        pen_true, pen_pred_list = [], []
        y_preds_bio:   list = []
        y_preds_pen:   list = []
        y_sparses_bio: list = []
        y_sparses_pen: list = []

        with torch.no_grad():
            for b in batches:
                T = len(b)

                # Full time grid
                t_full = torch.arange(T, dtype=torch.float32, device=self.device) * dt

                # Normalised control inputs: (1, T, n_ctrl)
                U_full = torch.tensor(
                    self.scalers.feat.transform(
                        b[feat_cols].values
                    ).astype(np.float32),
                    dtype=torch.float32,
                ).unsqueeze(0).to(self.device)

                # Initial conditions (from dense interpolated target at step 0)
                x0 = float(self.scalers.bio.transform([[b[TARGET_COLS[0]].iloc[0]]])[0, 0])
                p0 = float(self.scalers.pen.transform([[b[TARGET_COLS[1]].iloc[0]]])[0, 0])
                y0 = torch.tensor([[x0, p0]], dtype=torch.float32, device=self.device)

                # Full-trajectory integration
                traj    = self.model(y0, t_full, U_full, method=eval_solver)
                traj_np = traj.squeeze(0).cpu().numpy()   # (T, 2)

                # Inverse transform to physical g/L
                bio_phys = self.scalers.bio.inverse_transform(traj_np[:, 0:1]).squeeze()
                pen_phys = self.scalers.pen.inverse_transform(traj_np[:, 1:2]).squeeze()

                y_preds_bio.append(bio_phys)
                y_preds_pen.append(pen_phys)
                y_sparses_bio.append(b["_biomass_sparse"].values.copy())
                y_sparses_pen.append(b["_penicillin_sparse"].values.copy())

                # Collect sparse measurement points for metrics
                for sparse_col, traj_phys, t_list, p_list in [
                    ("_biomass_sparse",    bio_phys, bio_true, bio_pred_list),
                    ("_penicillin_sparse", pen_phys, pen_true, pen_pred_list),
                ]:
                    mask = b[sparse_col].notna().values
                    if mask.sum() == 0:
                        continue
                    t_list.append(b[sparse_col].values[mask])
                    p_list.append(traj_phys[mask])

        from src.evaluation.metrics import aggregate_metrics
        metrics = aggregate_metrics(bio_true, bio_pred_list, pen_true, pen_pred_list)
        metrics["y_preds_bio"]   = y_preds_bio
        metrics["y_preds_pen"]   = y_preds_pen
        metrics["y_sparses_bio"] = y_sparses_bio
        metrics["y_sparses_pen"] = y_sparses_pen
        return metrics


# Register
REGISTRY[NeuralODEAlgorithm.name] = NeuralODEAlgorithm

"""
src/algorithms/base.py
======================
ScalerBundle dataclass and BaseAlgorithm ABC.

All algorithm implementations inherit from BaseAlgorithm and register
themselves in the module-level REGISTRY dict by assigning a unique ``name``
class variable.  Registration is triggered when each subclass module is
imported (see ``src/algorithms/__init__.py``).
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config

# ── Algorithm registry ────────────────────────────────────────────────────────
# Populated by each subclass module at import time:
#   REGISTRY["pi_lstm"] = PILSTMAlgorithm
REGISTRY: Dict[str, type] = {}


# ── ScalerBundle ──────────────────────────────────────────────────────────────

@dataclass
class ScalerBundle:
    """
    Container for the MinMaxScalers shared across all algorithms.

    feat  : fitted on the process (± Raman) feature columns
    bio   : fitted on the biomass target
    pen   : fitted on the penicillin target
    raman : (optional) reserved for future use
    """

    feat:  MinMaxScaler
    bio:   MinMaxScaler
    pen:   MinMaxScaler
    raman: Optional[MinMaxScaler] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise scalers as pickle bytes for torch.save."""
        return {
            "feat":  pickle.dumps(self.feat),
            "bio":   pickle.dumps(self.bio),
            "pen":   pickle.dumps(self.pen),
            "raman": pickle.dumps(self.raman) if self.raman is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScalerBundle":
        """Deserialise scalers from pickle bytes."""
        return cls(
            feat=pickle.loads(d["feat"]),
            bio=pickle.loads(d["bio"]),
            pen=pickle.loads(d["pen"]),
            raman=pickle.loads(d["raman"]) if d.get("raman") is not None else None,
        )


# ── BaseAlgorithm ─────────────────────────────────────────────────────────────

class BaseAlgorithm(ABC):
    """
    Abstract base class for all algorithm implementations.

    Subclasses MUST define
    ----------------------
    name             : ClassVar[str]  — unique key in REGISTRY, e.g. ``"pi_lstm"``
    _build_model()   : return an **uninitialized** ``nn.Module`` (no ``.to(device)``)
    _fit_scalers()   : fit scalers on train_batches, return ``ScalerBundle``
    _build_dataset() : convert a batch list + scalers → ``TensorDataset``
    _train_epoch()   : one training epoch; return dict of float metrics
    _val_epoch()     : one validation epoch; return scalar val loss
    evaluate()       : full evaluation on a batch list; return metrics dict

    Overridable hooks (defaults provided)
    --------------------------------------
    _make_optimizer()       : build the optimizer (default: AdamW, single group)
    _is_es_active(epoch)    : whether ES patience ticks (default: always True)
    _epoch_extra_kwargs(ep) : extra kwargs forwarded to ``_train_epoch``
    """

    name: ClassVar[str] = ""

    def __init__(self, cfg: Config, device: torch.device):
        self.cfg     = cfg
        self.device  = device
        self.model:     Optional[nn.Module]            = None
        self.scalers:   Optional[ScalerBundle]         = None
        self._history:  Dict[str, list]                = {}
        self._optimizer: Optional[torch.optim.Optimizer] = None
        # AMP flag — set to True in fit() when device is CUDA.
        # Subclasses read this in _train_epoch / _val_epoch.
        self._use_amp: bool = False

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Construct and return the nn.Module (not yet on device)."""
        ...

    @abstractmethod
    def _fit_scalers(self, train_batches) -> ScalerBundle:
        """Fit MinMaxScalers on training batches; return a ScalerBundle."""
        ...

    @abstractmethod
    def _build_dataset(self, batches, scalers: ScalerBundle) -> TensorDataset:
        """Convert a list of batch DataFrames + fitted scalers → TensorDataset."""
        ...

    @abstractmethod
    def _train_epoch(self, loader: DataLoader, **kwargs) -> Dict[str, float]:
        """
        Run one training epoch.

        The model is already in ``train()`` mode when this is called.
        The optimizer is available as ``self._optimizer``.

        Returns
        -------
        metrics : dict of float — e.g. {"train_loss": 0.012, "train_data": 0.011}
        """
        ...

    @abstractmethod
    def _val_epoch(self, loader: DataLoader) -> float:
        """
        Run one validation epoch under ``torch.no_grad()``.

        Returns
        -------
        val_loss : scalar float
        """
        ...

    @abstractmethod
    def evaluate(self, batches) -> Dict[str, Any]:
        """
        Full evaluation on a list of batch DataFrames.

        Returns
        -------
        metrics : dict  — at minimum {"bio": {RMSE, MAE, R2, n}, "pen": {...}}
        """
        ...

    # ── Overridable hooks ─────────────────────────────────────────────────────

    def _make_optimizer(self) -> torch.optim.Optimizer:
        """
        Default: single AdamW param group.

        Override in PILSTMAlgorithm to split network / physics param groups.
        """
        tr = self.cfg.training
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=float(tr.lr),
            weight_decay=float(tr.weight_decay),
        )

    def _is_es_active(self, epoch: int) -> bool:
        """
        Return True when early-stopping patience should increment.

        Override in PILSTMAlgorithm to freeze ES during the physics delay
        phase (prevents ES from firing before the ODE constraint has acted).
        """
        return True

    def _epoch_extra_kwargs(self, epoch: int) -> Dict[str, Any]:
        """
        Return extra keyword arguments forwarded to ``_train_epoch``.

        Override in PILSTMAlgorithm to pass the annealed ``lambda_physics``.
        """
        return {}

    # ── Main training loop ────────────────────────────────────────────────────

    def fit(
        self,
        train_batches,
        val_batches,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """
        Common training loop:

          1. ``_fit_scalers`` on train_batches.
          2. Build TensorDatasets + DataLoaders via ``_build_dataset``.
          3. ``_build_model``, move to device.
          4. ``_make_optimizer`` + ReduceLROnPlateau scheduler.
          5. Epoch loop: _train_epoch → _val_epoch → ES → LR step.
          6. Restore best weights at end.

        Returns
        -------
        history : dict of per-epoch metric lists
        """
        tr = self.cfg.training

        # 1 — Scalers
        if verbose:
            print("[fit] Fitting scalers ...")
        self.scalers = self._fit_scalers(train_batches)

        # 2 — Datasets
        if verbose:
            print("[fit] Building datasets ...")
        train_ds = self._build_dataset(train_batches, self.scalers)
        val_ds   = self._build_dataset(val_batches,   self.scalers)

        # AMP: use bfloat16 on CUDA (no GradScaler needed — same exponent range as fp32)
        self._use_amp = (self.device.type == "cuda")

        # Pre-move the entire dataset to GPU VRAM once, eliminating per-batch
        # host→device transfers. The whole training set is ≈90 MB for the largest
        # models, well within the 6 GB VRAM budget.
        if self.device.type == "cuda":
            train_ds = TensorDataset(
                *(t.to(self.device, non_blocking=True) for t in train_ds.tensors)
            )
            val_ds = TensorDataset(
                *(t.to(self.device, non_blocking=True) for t in val_ds.tensors)
            )
            # GPU-resident tensors → no workers or pinning needed
            loader_kw: dict = {"num_workers": 0, "pin_memory": False}
        else:
            # CPU path: background workers overlap preprocessing with GPU compute
            loader_kw = {
                "num_workers":       4,
                "pin_memory":        True,
                "persistent_workers": True,
                "prefetch_factor":   2,
            }

        bs = int(tr.batch_size)
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True,  **loader_kw
        )
        val_loader = DataLoader(
            val_ds,   batch_size=bs, shuffle=False, **loader_kw
        )
        if verbose:
            amp_tag = "bfloat16 AMP" if self._use_amp else "float32"
            gpu_tag = "GPU-resident" if self.device.type == "cuda" else "CPU"
            print(f"[fit] DataLoader: batch={bs}, {gpu_tag} dataset, {amp_tag}")

        # 3 — Model
        if verbose:
            print("[fit] Building model ...")
        self.model = self._build_model().to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if verbose:
            print(f"      Trainable parameters: {n_params:,}")

        # 4 — Optimizer + scheduler
        self._optimizer = self._make_optimizer()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            factor=float(tr.lr_factor),
            patience=int(tr.lr_patience),
            min_lr=1e-6,
        )

        n_epochs    = int(tr.n_epochs)
        es_patience = int(tr.es_patience)

        best_val_loss = float("inf")
        best_state:   Optional[dict] = None
        patience_ctr  = 0
        self._history = {}

        if verbose:
            print(f"[fit] Training for up to {n_epochs} epochs ...")

        # 5 — Epoch loop
        for epoch in range(1, n_epochs + 1):
            extra = self._epoch_extra_kwargs(epoch)

            # Train
            self.model.train()
            train_metrics = self._train_epoch(train_loader, **extra)

            # Validate
            self.model.eval()
            val_loss = self._val_epoch(val_loader)

            # LR scheduler
            scheduler.step(val_loss)
            lr_now = float(self._optimizer.param_groups[0]["lr"])

            # Best state + ES
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                patience_ctr = 0
            elif self._is_es_active(epoch):
                patience_ctr += 1
            # else: patience frozen (e.g. physics delay phase)

            # Accumulate history
            self._history.setdefault("val_loss",  []).append(val_loss)
            self._history.setdefault("lr",        []).append(lr_now)
            self._history.setdefault("patience",  []).append(patience_ctr)
            for k, v in train_metrics.items():
                self._history.setdefault(k, []).append(v)

            if verbose:
                star = "*" if improved else " "

                def _fmt(v: float) -> str:
                    """Auto-scale: scientific for very small/large, fixed otherwise."""
                    if v == 0.0:
                        return "0.000000"
                    a = abs(v)
                    if a < 1e-4 or a >= 1e4:
                        return f"{v:.3e}"
                    return f"{v:.6f}"

                extras_str = "  ".join(
                    f"{k}={_fmt(v)}" for k, v in train_metrics.items()
                    if k not in ("lambda_phys",)   # skip noisy scalar, already implied
                )
                print(
                    f"  {star} Ep {epoch:4d}/{n_epochs}"
                    f"  val={val_loss:.5f}  {extras_str}"
                    f"  lr={lr_now:.2e}  pat={patience_ctr}"
                )

            if patience_ctr >= es_patience:
                if verbose:
                    print(
                        f"\n  [EarlyStopping] No improvement for "
                        f"{es_patience} epochs — stopped at epoch {epoch}."
                    )
                break

        # 6 — Restore best
        if best_state is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )
            if verbose:
                print(f"\n  Best weights restored  (val_loss={best_val_loss:.6f})")

        self.model.eval()
        return self._history

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(
        self,
        path:  str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save to a unified checkpoint dict.

        Checkpoint keys
        ---------------
        algorithm_name, model_state, scalers, cfg, history
        + any keys in ``extra`` (e.g. val_metrics, test_metrics)
        """
        assert self.model   is not None, "Call fit() before save()"
        assert self.scalers is not None, "Call fit() before save()"

        ckpt: Dict[str, Any] = {
            "algorithm_name": self.name,
            "model_state":    self.model.state_dict(),
            "scalers":        self.scalers.to_dict(),
            "cfg":            self.cfg.to_dict(),
            "history":        self._history,
        }
        if extra:
            ckpt.update(extra)

        torch.save(ckpt, path)
        print(f"  Checkpoint saved -> {path}")

    @classmethod
    def load(cls, path: str, device: torch.device) -> "BaseAlgorithm":
        """
        Load a checkpoint and reconstruct the correct algorithm subclass.

        Uses ``REGISTRY`` to look up the subclass by ``algorithm_name``.
        Ensure all algorithm modules are imported before calling this
        (``from src.algorithms import *`` will do it).
        """
        ckpt    = torch.load(path, map_location=device, weights_only=False)
        name    = ckpt["algorithm_name"]

        if name not in REGISTRY:
            raise KeyError(
                f"Algorithm '{name}' not in REGISTRY. "
                f"Available: {list(REGISTRY.keys())}"
            )

        AlgoCls          = REGISTRY[name]
        cfg              = Config.from_dict(ckpt["cfg"])
        instance         = AlgoCls(cfg, device)
        instance.scalers = ScalerBundle.from_dict(ckpt["scalers"])
        instance.model   = instance._build_model().to(device)
        instance.model.load_state_dict(ckpt["model_state"])
        instance.model.eval()
        instance._history = ckpt.get("history", {})
        return instance

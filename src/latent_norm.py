"""
latent_norm.py
==============
Intervention sur la normalisation des latents en entrée du LSTM.

Par défaut, chaque algorithme ajuste un MinMaxScaler PAR DIMENSION sur les
features. Pour une dimension quasi effondrée, ce MinMax étire un bruit
minuscule sur toute la plage [0, 1].

Modes :
  minmax : comportement d'origine (aucune modification)
  global : un seul facteur d'échelle commun à toutes les dimensions latentes.
           x' = (x - min_d) / R,  R = plus grande plage parmi les dimensions.
           Les dimensions plates restent plates : le LSTM voit la géométrie
           réelle du code (PR après scaling = PR brut).
  active : MinMax d'origine, mais les dimensions dont l'écart-type est
           inférieur à tol * (écart-type max) sont mises à 0 (entrée
           constante, donc aucun signal transmis au LSTM).

Seules les colonnes latentes (_raman_*) sont concernées ; les éventuelles
autres features gardent le MinMax d'origine.

Usage dans train.py :
    from latent_norm import apply_latent_norm, LatentNormScaler
    apply_latent_norm(alg, args.latent_norm, args.active_tol)   # avant fit()
"""
from __future__ import annotations

import numpy as np


class LatentNormScaler:
    """Enveloppe le MinMaxScaler d'origine et remplace la transformation
    des colonnes latentes."""

    def __init__(self, base, X_train: np.ndarray, raman_idx, mode: str, tol: float = 0.05):
        self.base = base
        self.idx = np.asarray(raman_idx, dtype=int)
        self.mode = mode
        Xr = X_train[:, self.idx]

        if mode == "global":
            self.lo = Xr.min(axis=0)
            self.R = float((Xr.max(axis=0) - Xr.min(axis=0)).max())
            if self.R <= 0:
                self.R = 1.0
            print(f"  [LATENT-NORM] global : échelle commune R={self.R:.4g} "
                  f"sur {len(self.idx)} dimensions latentes")
        elif mode == "active":
            std = Xr.std(axis=0)
            inactive_mask = std < tol * std.max()
            self.inactive = self.idx[inactive_mask]
            print(f"  [LATENT-NORM] active : {len(self.idx) - len(self.inactive)} "
                  f"dimensions gardées / {len(self.idx)} (tol={tol})")
        else:
            raise ValueError(f"mode inconnu : {mode}")

    def transform(self, X):
        X = np.asarray(X)
        Y = np.array(self.base.transform(X), dtype=np.float64, copy=True)
        if self.mode == "global":
            Y[:, self.idx] = (X[:, self.idx] - self.lo) / self.R
        elif self.mode == "active":
            Y[:, self.inactive] = 0.0
        return Y

    def __getattr__(self, name):
        # data_range_, inverse_transform, etc. -> MinMaxScaler d'origine
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)


def apply_latent_norm(alg, mode: str, tol: float = 0.05) -> None:
    """Remplace, à l'intérieur de fit(), le scaler des features par un
    LatentNormScaler. À appeler AVANT alg.fit()."""
    if mode in (None, "minmax"):
        return

    orig_fit_scalers = alg._fit_scalers

    def _patched_fit_scalers(train_batches):
        bundle = orig_fit_scalers(train_batches)
        cols = alg._feature_cols()
        idx = [i for i, c in enumerate(cols) if c.startswith("_raman_")]
        if not idx:
            raise RuntimeError("Aucune colonne _raman_* dans les features : "
                               "l'intervention n'a pas de sens pour ce modèle.")
        X = np.vstack([b[cols].values for b in train_batches]).astype(np.float64)
        bundle.feat = LatentNormScaler(bundle.feat, X, idx, mode, tol)
        return bundle

    alg._fit_scalers = _patched_fit_scalers

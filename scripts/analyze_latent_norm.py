#!/usr/bin/env python
"""
analyze_latent_norm.py
======================
Compare, pour chaque modèle, la RMSE sous panne (R4) avec normalisation
global / active contre la baseline MinMax d'origine, seed par seed.

Baseline MinMax : log outputs/latent_norm_experiments/{model}_minmax_s{seed}.log
s'il existe, sinon valeur du summary.csv du sweep (même pipeline).

Usage : PYTHONPATH=. python scripts/analyze_latent_norm.py
"""
import re
from pathlib import Path

import numpy as np
from scipy import stats

import latent_geometry as lg

OUT_DIR = Path("./outputs/latent_norm_experiments")
FAULT_RE = re.compile(r"Fault\s+\[Penicillin\].*?RMSE=([0-9.]+)")
MODELS = ["beta_vae", "vae", "cdae"]
enc = {(e["model"], e["seed"]): e for e in lg.ENCODERS}


def r4_from_log(model, norm, seed):
    log = OUT_DIR / f"{model}_{norm}_s{seed}.log"
    if not log.exists():
        return np.nan
    found = FAULT_RE.findall(log.read_text(errors="ignore"))
    return float(found[-1]) if found else np.nan


for m in MODELS:
    rows = []
    for s in lg.SEEDS:
        base = r4_from_log(m, "minmax", s)
        if np.isnan(base):
            base = lg.rmse_from_summary(enc[(m, s)]["rmse_csv"], s)
        rows.append((s, base, r4_from_log(m, "global", s), r4_from_log(m, "active", s)))
    a = np.array(rows, dtype=float)

    print(f"\n=== {m} : RMSE R4 (g/L) ===")
    print(f"{'seed':>6} {'minmax':>8} {'global':>8} {'active':>8}")
    for r in a:
        print(f"{int(r[0]):>6} " + " ".join(f"{v:>8.3f}" if not np.isnan(v) else f"{'-':>8}" for v in r[1:]))
    print(f"{'moy.':>6} " + " ".join(f"{np.nanmean(a[:, j]):>8.3f}" for j in (1, 2, 3)))

    for j, name in [(2, "global"), (3, "active")]:
        ok = ~np.isnan(a[:, 1]) & ~np.isnan(a[:, j])
        if ok.sum() < 3:
            print(f"  {name:>6} vs minmax : pas assez de runs ({ok.sum()})")
            continue
        d = a[ok, j] - a[ok, 1]
        _, p = stats.ttest_rel(a[ok, j], a[ok, 1])
        print(f"  {name:>6} vs minmax : Δ moyen {d.mean():+.3f} g/L "
              f"({100 * d.mean() / a[ok, 1].mean():+.1f} %), "
              f"{(d < 0).sum()}/{ok.sum()} seeds améliorés, t apparié p = {p:.3f}")

#!/usr/bin/env python
"""
run_latent_norm.py
==================
Lance l'intervention sur la normalisation des latents :
3 modèles x 5 seeds x 2 normalisations (global, active) = 30 runs.

Les chemins des encodeurs sont repris de latent_geometry.ENCODERS
(ceux que tu as déjà corrigés). Un run dont le log contient déjà une
ligne Fault est sauté : le script peut être relancé après interruption.

Usage :
    PYTHONPATH=. python scripts/run_latent_norm.py                         # les 30 runs
    PYTHONPATH=. python scripts/run_latent_norm.py --norms minmax --models vae cdae --seeds 42
        -> vérification : doit reproduire la valeur du summary.csv
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import latent_geometry as lg

MODEL_ALIAS = {
    "beta_vae": "beta_vae_pi_lstm",
    "vae":      "cvae_pi_lstm",
    "cdae":     "cdae_pi_lstm",
}
OUT_DIR = Path("./outputs/latent_norm_experiments")
FAULT_RE = re.compile(r"Fault\s+\[Penicillin\].*?RMSE=([0-9.]+)")

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=["beta_vae", "vae", "cdae"])
parser.add_argument("--seeds", nargs="+", type=int, default=lg.SEEDS)
parser.add_argument("--norms", nargs="+", default=["global", "active"],
                    choices=["minmax", "global", "active"])
parser.add_argument("--active-tol", type=float, default=0.05)
args = parser.parse_args()

OUT_DIR.mkdir(exist_ok=True)
enc = {(e["model"], e["seed"]): e for e in lg.ENCODERS}

jobs = [(m, s, n) for m in args.models for s in args.seeds for n in args.norms]
print(f"{len(jobs)} runs prévus -> {OUT_DIR}/\n")

for i, (m, s, n) in enumerate(jobs, 1):
    e = enc[(m, s)]
    tag = f"{m}_{n}_s{s}"
    log = OUT_DIR / f"{tag}.log"
    if log.exists() and FAULT_RE.search(log.read_text(errors="ignore")):
        print(f"[{i}/{len(jobs)}] {tag} : déjà fait, sauté")
        continue

    cmd = [
        sys.executable, "train.py",
        "--model", MODEL_ALIAS[m], "--regime-ablation", "--seed", str(s),
        "--raman-ckpt", e["ckpt"], "--raman-scaler", e["scaler"],
        "--latent-norm", n, "--active-tol", str(args.active_tol),
        "--no-plots", "--ckpt", str(OUT_DIR / f"{tag}.pt"),
    ]
    print(f"[{i}/{len(jobs)}] {tag} ...", end=" ", flush=True)
    with open(log, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    txt = log.read_text(errors="ignore")
    found = FAULT_RE.findall(txt)
    if found:
        extra = [l.strip() for l in txt.splitlines() if "[LATENT-NORM]" in l]
        print(f"R4 = {float(found[-1]):.3f}   {extra[0] if extra else ''}")
    else:
        print(f"ÉCHEC, voir {log}")

print(f"\nTerminé. Analyse : PYTHONPATH=. python scripts/analyze_latent_norm.py")

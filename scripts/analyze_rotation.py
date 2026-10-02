#!/usr/bin/env python
"""
Analyse de l'expérience de rotation : chaque mode comparé à "identity",
apparié par (seed encodeur, seed LSTM).

Gère les deux formats de noms de logs :
  {mode}_s{seed}.log            (v2, fixed_epochs)
  {mode}_s{seed}_l{lseed}.log   (v3, plusieurs seeds LSTM)

Usage :
  python scripts/analyze_rotation.py ./outputs/full_rotate_experiments_v2
  python scripts/analyze_rotation.py ./outputs/full_rotate_experiments_v3
  python scripts/analyze_rotation.py ./outputs/full_rotate_experiments_fixed_epochs
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

OUT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "./outputs/full_rotate_experiments_v2")
TAIL = 0.4   # seuil de défaillance sous panne (g/L)

NAME = re.compile(r"^(identity|control|targeted|full)_s(\d+)(?:_l(\d+))?\.log$")
METRIC = re.compile(r"(R2|Fault)\s+\[Penicillin\].*?RMSE=([0-9.]+)")

res = defaultdict(dict)   # res[mode][(seed, lseed)] = {"R2": x, "Fault": y}
for log in sorted(OUT_DIR.glob("*.log")):
    m = NAME.match(log.name)
    if not m:
        continue
    mode, seed, lseed = m.group(1), int(m.group(2)), m.group(3)
    key = (seed, int(lseed) if lseed else None)
    txt = log.read_text(errors="ignore")
    if "ÉCHEC" in txt or "[ROTATION] OK" not in txt:
        print(f"  !! {log.name} : wrapper non confirmé ou run planté, ignoré")
        continue
    vals = {s: float(v) for s, v in METRIC.findall(txt)}   # dernière occurrence
    if "R2" in vals and "Fault" in vals:
        res[mode][key] = vals

if "identity" not in res:
    sys.exit("Aucun run identity valide : impossible de comparer.")

modes = [m for m in ["identity", "control", "targeted", "full"] if m in res]
keys = sorted(res["identity"])
print(f"\nDossier : {OUT_DIR}   |   paires de référence (identity) : {len(keys)}")

for split, label in [("R2", "R2 (nominal)"), ("Fault", "R4 (pannes)")]:
    print(f"\n=== {label} ===")
    head = f"{'seed':>6} {'lstm':>5} " + " ".join(f"{m:>10}" for m in modes)
    print(head)
    for k in keys:
        row = [res[m].get(k, {}).get(split) for m in modes]
        print(f"{k[0]:>6} {str(k[1] or '-'):>5} "
              + " ".join(f"{v:>10.3f}" if v is not None else f"{'-':>10}" for v in row))

    print()
    for m in modes:
        x = np.array([res[m][k][split] for k in keys if k in res[m]])
        print(f"  {m:>9} : moyenne {x.mean():.3f}  médiane {np.median(x):.3f}  "
              f"(n={len(x)})", end="")
        if split == "Fault":
            print(f"   runs > {TAIL} g/L : {(x > TAIL).sum()}/{len(x)}", end="")
        print()

    print()
    for m in modes[1:]:
        common = [k for k in keys if k in res[m]]
        if len(common) < 3:
            print(f"  {m:>9} vs identity : pas assez de paires ({len(common)})")
            continue
        b = np.array([res["identity"][k][split] for k in common])
        x = np.array([res[m][k][split] for k in common])
        d = x - b
        _, p_t = stats.ttest_rel(x, b)
        msg = (f"  {m:>9} vs identity : Δ moyen {d.mean():+.3f} g/L "
               f"({100 * d.mean() / b.mean():+.1f} %), Δ médian {np.median(d):+.3f}, "
               f"{(d > 0).sum()}/{len(d)} dégradés, t apparié p={p_t:.3f}")
        if len(d) >= 6:
            _, p_w = stats.wilcoxon(x, b)
            msg += f", Wilcoxon p={p_w:.3f}"
        print(msg)

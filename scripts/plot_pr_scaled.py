#!/usr/bin/env python
"""
plot_pr_scaled.py
=================
Nuage PR après MinMax (dimensionnalité effective vue par le LSTM) contre
RMSE sous panne, pour les 15 encodeurs. Lit outputs/latent_geometry.csv
(généré par scripts/latent_geometry.py).

Usage : PYTHONPATH=. python scripts/plot_pr_scaled.py   ->   outputs/pr_scaled_vs_rmse.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

g = pd.read_csv("outputs/latent_geometry.csv").dropna(subset=["rmse_r4"])
labels = {"beta_vae": r"$\beta$-VAE ($\beta$=0.005)", "vae": r"VAE ($\beta$=1)", "cdae": "CDAE"}
colors = {"beta_vae": "tab:green", "vae": "tab:blue", "cdae": "tab:red"}

rho, p = stats.spearmanr(g["pr_scaled"], g["rmse_r4"])

fig, ax = plt.subplots(figsize=(5.5, 4))
for m, sub in g.groupby("model"):
    ax.scatter(sub["pr_scaled"], sub["rmse_r4"], s=45,
               color=colors.get(m), label=labels.get(m, m), edgecolor="k", linewidth=0.4)
    for _, r in sub.iterrows():
        ax.annotate(str(int(r["seed"])), (r["pr_scaled"], r["rmse_r4"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("Effective dimensionality at LSTM input\n(participation ratio after per-dimension MinMax)")
ax.set_ylabel("Fault-regime RMSE (g/L)")
ax.set_title(f"Spearman ρ = {rho:+.2f} (n = {len(g)})", fontsize=10)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("outputs/pr_scaled_vs_rmse.png", dpi=300)
fig.savefig("outputs/pr_scaled_vs_rmse.pdf")
print(f"outputs/pr_scaled_vs_rmse.png / .pdf   (rho={rho:+.2f}, p={p:.4f}, n={len(g)})")

#!/usr/bin/env python3
"""
beta_sweep_multiseed.py
=======================
Multi-seed β-VAE + PI-LSTM sweep with real-time reporting.

For each β value (outer loop), trains all seeds in parallel (max --parallel),
then immediately prints mean RMSE ± variance after every completed seed.

Seed-42 results are reused from outputs/beta_sweep_e2e/ (no retraining).

Usage
-----
    PYTHONPATH=. python scripts/beta_sweep_multiseed.py --device cuda --parallel 2
    PYTHONPATH=. python scripts/beta_sweep_multiseed.py --eval-only
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Constants ──────────────────────────────────────────────────────────────────

BETAS_DEFAULT = [0.05, 0.5, 1.0]
SEEDS_DEFAULT = [42, 123, 456, 789, 1024]

SEED42_DIR  = _ROOT / "outputs" / "beta_sweep_e2e"
MULTI_DIR   = _ROOT / "outputs" / "beta_sweep_multiseed"
ANNEAL_DIR  = _ROOT / "outputs" / "beta_sweep_multiseed_anneal"  # annealed runs
CSV_PATH    = str(_ROOT / "100_Batches_IndPenSim_V3.csv")

REGIME_IDS = {
    "R1":    set(range(0,  30)),
    "R2":    set(range(30, 60)),
    "R3":    set(range(60, 90)),
    "Fault": set(range(90, 100)),
}

N_MIG_SAMPLES = 5000
ACTIVE_KL_THR = 0.01

_FACTOR_PATTERNS = {
    "Biomass":     r"Biomass|X_offline|X\(g",
    "Penicillin":  r"Penicillin|P_offline|P\(g",
    "Glucose":     r"Glucose|Sugar feed|Fs:",
    "DO2":         r"[Dd]issolved.oxygen|DO2|DO\(",
    "pH":          r"^pH\b|:pH\b",
    "Temperature": r"Temperature|Temp|:T\(K\)",
}

_PRINT_LOCK = threading.Lock()


def tprint(*args, **kwargs):
    with _PRINT_LOCK:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


# ── Path helpers ───────────────────────────────────────────────────────────────

def _tag(beta: float) -> str:
    return f"b{beta:.5f}".rstrip("0").rstrip(".")


def _ckpt_dir(seed: int, beta: float, anneal: bool = False) -> Path:
    if seed == 42:
        return SEED42_DIR / "checkpoints" / _tag(beta)
    base = ANNEAL_DIR if anneal else MULTI_DIR
    return base / f"s{seed}" / "checkpoints" / _tag(beta)


def _sweep_json(seed: int, anneal: bool = False) -> Path:
    if seed == 42:
        return SEED42_DIR / "sweep_results.json"
    base = ANNEAL_DIR if anneal else MULTI_DIR
    return base / f"s{seed}" / "sweep_results.json"


def _eval_json(seed: int, anneal: bool = False) -> Path:
    if seed == 42:
        return SEED42_DIR / "eval_r2_r3_separate.json"
    base = ANNEAL_DIR if anneal else MULTI_DIR
    return base / f"s{seed}" / "eval_r2r3.json"


def _log_path(seed: int, beta: float, anneal: bool = False) -> Path:
    tag  = _tag(beta)
    base = ANNEAL_DIR if anneal else MULTI_DIR
    return base / f"s{seed}" / f"train_{tag}.log"


# ── Training: one subprocess per (beta, seed) ──────────────────────────────────

def _train_one(beta: float, seed: int, device: str, skip_trained: bool,
               anneal_delay: int = 0, anneal_epochs: int = 1) -> int:
    """Train ONE (beta, seed) pair. Returns exit code."""
    anneal = (anneal_delay > 0 or anneal_epochs > 1)
    cd = _ckpt_dir(seed, beta, anneal)
    vae_done  = (cd / "vae.pt").exists() and (cd / "vae_scaler.joblib").exists()
    lstm_done = (cd / "lstm.pt").exists()

    if skip_trained and vae_done and lstm_done:
        tprint(f"  [β={beta}, s={seed}] checkpoints exist — skipping training")
        return 0

    base    = ANNEAL_DIR if anneal else MULTI_DIR
    out_dir = str(base / f"s{seed}")
    log_file = _log_path(seed, beta, anneal)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u",
        str(_ROOT / "scripts" / "beta_sweep_e2e.py"),
        "--betas",            str(beta),
        "--seed",             str(seed),
        "--out-dir",          out_dir,
        "--device",           device,
        "--vae-anneal-delay", str(anneal_delay),
        "--vae-anneal-epochs", str(anneal_epochs),
    ]
    if skip_trained:
        cmd.append("--skip-trained")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT) + ":" + env.get("PYTHONPATH", "")

    tprint(f"\n  ┌─ [β={beta}, s={seed}] Training started → {log_file.name}")

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(_ROOT), env=env, text=True, bufsize=1,
        )
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            # Forward key lines to the main terminal
            stripped = line.rstrip()
            if any(kw in stripped for kw in (
                "Epoch", "β =", "Training", "LSTM", "saved", "stopping",
                "best val", "PI-LSTM", "recon", "kl=",
            )):
                tprint(f"    [β={beta}, s={seed}] {stripped}")
        proc.wait()

    rc = proc.returncode
    status = "✓ done" if rc == 0 else f"✗ FAILED (rc={rc})"
    tprint(f"  └─ [β={beta}, s={seed}] {status}")
    return rc


# ── Evaluation: R1/R2/R3/R4 + S_sep + MIG ─────────────────────────────────────

def _encode_batches(batches, encoder) -> List:
    encoded = []
    for b in batches:
        b2 = b.copy()
        latents = encoder.encode_dataframe(b2)
        for d in range(latents.shape[1]):
            b2[f"_raman_{d}"] = latents[:, d]
        encoded.append(b2)
    return encoded


def _eval_lstm(alg, enc_batches) -> Dict[str, float]:
    m = alg.evaluate(enc_batches)
    return {"RMSE": float(m["pen"]["RMSE"]), "MAE": float(m["pen"]["MAE"])}


def _ssep_mig(encoder, r2r3_enc, n_samples=N_MIG_SAMPLES) -> Dict[str, float]:
    from scipy.stats import pearsonr
    from sklearn.feature_selection import mutual_info_regression

    mu_parts, fac_parts = [], {k: [] for k in _FACTOR_PATTERNS}
    for b in r2r3_enc:
        lc = [c for c in b.columns if c.startswith("_raman_")]
        if not lc:
            continue
        mu_parts.append(b[lc].values.astype(np.float32))
        for fname, pat in _FACTOR_PATTERNS.items():
            for col in b.columns:
                if re.search(pat, col):
                    fac_parts[fname].append(b[col].values.astype(np.float64))
                    break

    if not mu_parts:
        return {"s_sep": float("nan"), "mig": float("nan")}

    Z = np.concatenate(mu_parts, axis=0)
    fac = {k: np.concatenate(v) for k, v in fac_parts.items() if v}

    rng = np.random.default_rng(42)
    idx = rng.choice(len(Z), min(n_samples, len(Z)), replace=False)
    Zs  = Z[idx]

    # S_sep
    s_sep = float("nan")
    if "Penicillin" in fac and "Glucose" in fac:
        pen = fac["Penicillin"][idx];  glc = fac["Glucose"][idx]
        ok  = np.isfinite(pen) & np.isfinite(glc)
        if ok.sum() > 50:
            Zo = Zs[ok]
            rp = np.array([abs(pearsonr(Zo[:, d], pen[ok])[0]) for d in range(64)])
            rg = np.array([abs(pearsonr(Zo[:, d], glc[ok])[0]) for d in range(64)])
            if np.argmax(rp) != np.argmax(rg):
                s_sep = float(min(rp.max(), rg.max()))
            else:
                s_sep = 0.0

    # MIG
    mig = float("nan")
    gaps = []
    for fname in _FACTOR_PATTERNS:
        if fname not in fac:
            continue
        v = fac[fname][idx]
        ok2 = np.isfinite(v)
        if ok2.sum() < 100:
            continue
        try:
            mi = mutual_info_regression(Zs[ok2], v[ok2], n_neighbors=5, random_state=0)
        except Exception:
            continue
        if len(mi) < 2:
            continue
        mi_s = np.sort(mi)[::-1]
        n_b  = max(int(2 * ok2.sum() ** (1/3)), 5)
        cnt, _ = np.histogram(v[ok2], bins=n_b)
        p = cnt / cnt.sum(); p = p[p > 0]
        H = float(-np.sum(p * np.log(p + 1e-12)))
        if H > 1e-6:
            gaps.append((mi_s[0] - mi_s[1]) / H)
    if gaps:
        mig = float(np.mean(gaps))

    return {"s_sep": s_sep, "mig": mig}


def evaluate_one(beta: float, seed: int, buckets: Dict, device_str: str,
                 anneal: bool = False) -> Dict:
    """Load (beta, seed) checkpoint and evaluate on R1/R2/R3/R4 + S_sep + MIG."""
    import torch
    from src.algorithms.base import BaseAlgorithm
    from src.data.beta_vae_encoder import BetaVAERamanEncoderV2

    cd = _ckpt_dir(seed, beta, anneal)
    vae_ckpt   = str(cd / "vae.pt")
    vae_scaler = str(cd / "vae_scaler.joblib")
    lstm_ckpt  = str(cd / "lstm.pt")

    if not all(Path(p).exists() for p in [vae_ckpt, vae_scaler, lstm_ckpt]):
        tprint(f"  [β={beta}, s={seed}] Missing checkpoint — skipping eval")
        return {}

    device  = torch.device(device_str)
    encoder = BetaVAERamanEncoderV2(vae_ckpt, vae_scaler, device)
    alg     = BaseAlgorithm.load(lstm_ckpt, device)

    r1_enc   = _encode_batches(buckets["R1"],    encoder)
    r2_enc   = _encode_batches(buckets["R2"],    encoder)
    r3_enc   = _encode_batches(buckets["R3"],    encoder)
    r4_enc   = _encode_batches(buckets["Fault"], encoder)

    m_r1 = _eval_lstm(alg, r1_enc)
    m_r2 = _eval_lstm(alg, r2_enc)
    m_r3 = _eval_lstm(alg, r3_enc)
    m_r4 = _eval_lstm(alg, r4_enc)

    disentangle = _ssep_mig(encoder, r2_enc + r3_enc)

    # active_dims from VAE
    active = float("nan")
    sweep_path = _sweep_json(seed, anneal)
    if sweep_path.exists():
        rows = json.load(open(sweep_path))
        for r in rows:
            if abs(r.get("beta", -1) - beta) < 1e-9:
                active = r.get("active_dims", float("nan"))
                break

    delta = (m_r4["RMSE"] - m_r1["RMSE"]) / m_r1["RMSE"] * 100

    return {
        "beta":       beta,
        "seed":       seed,
        "active":     active,
        "r1":         m_r1["RMSE"],
        "r2":         m_r2["RMSE"],
        "r3":         m_r3["RMSE"],
        "r4":         m_r4["RMSE"],
        "delta_pct":  delta,
        "s_sep":      disentangle["s_sep"],
        "mig":        disentangle["mig"],
    }


# ── Real-time stats printer ────────────────────────────────────────────────────

def _print_running_stats(beta: float, completed: List[Dict]) -> None:
    """Print mean ± variance across all seeds done so far for this beta."""
    if not completed:
        return

    def _stat(key):
        vals = [r[key] for r in completed
                if isinstance(r.get(key), float) and not np.isnan(r[key])]
        if not vals:
            return "?"
        if len(vals) == 1:
            return f"{np.mean(vals):.3f}"
        return f"{np.mean(vals):.3f} ± {np.std(vals):.3f} (var={np.std(vals)**2:.4f})"

    seeds_done = [r["seed"] for r in completed]
    tprint(f"\n  {'─'*70}")
    tprint(f"  β={beta}  running stats  [{len(completed)}/{len(SEEDS_DEFAULT)} seeds: {seeds_done}]")
    tprint(f"  {'─'*70}")
    tprint(f"  R1 RMSE  : {_stat('r1')}")
    tprint(f"  R2 RMSE  : {_stat('r2')}")
    tprint(f"  R3 RMSE  : {_stat('r3')}")
    tprint(f"  R4 RMSE  : {_stat('r4')}")
    tprint(f"  Δ%       : {_stat('delta_pct')}")
    tprint(f"  S_sep    : {_stat('s_sep')}")
    tprint(f"  MIG      : {_stat('mig')}")
    tprint(f"  {'─'*70}\n")


# ── Load data ──────────────────────────────────────────────────────────────────

def _load_buckets(csv_path: str) -> Dict:
    import pandas as pd
    from src.data.dataset import TARGET_COLS, FAULT_COL

    tprint("  Loading CSV ...", flush=True)
    df = pd.read_csv(csv_path)
    t = df["Time (h)"].values
    bids = np.zeros(len(df), dtype=np.int32)
    bid = 0
    for i in range(1, len(df)):
        if t[i] < t[i-1]:
            bid += 1
        bids[i] = bid
    df["_batch_id"]          = bids
    df["_penicillin_sparse"] = df[TARGET_COLS[1]].copy()
    df["_biomass_sparse"]    = df[TARGET_COLS[0]].copy()
    for tc in TARGET_COLS:
        df[tc] = df[tc].interpolate(method="linear").ffill().bfill()

    buckets: Dict[str, List] = {k: [] for k in REGIME_IDS}
    for b_id in range(int(df["_batch_id"].max()) + 1):
        b = df[df["_batch_id"] == b_id].copy().reset_index(drop=True)
        if len(b) < 50:
            continue
        for regime, ids in REGIME_IDS.items():
            if b_id in ids:
                buckets[regime].append(b)
                break
    for k, v in buckets.items():
        tprint(f"    {k}: {len(v)} batches")
    return buckets


# ── Aggregation ────────────────────────────────────────────────────────────────

def _print_final_table(all_results: Dict[float, List[Dict]]) -> None:
    """Print the completed table with mean ± std and variance."""
    tprint(f"\n{'═'*100}")
    tprint("  FINAL RESULTS — mean ± std across seeds")
    tprint(f"{'═'*100}")
    tprint(f"  {'β':>6}  {'Active':>6}  {'R1':>16}  {'R2':>16}  {'R3':>16}  "
           f"{'R4 RMSE':>16}  {'Δ%':>8}  {'S_sep':>10}  {'MIG':>10}")
    tprint(f"  {'─'*98}")

    summary = []
    for beta in sorted(all_results):
        rows = all_results[beta]
        if not rows:
            continue

        def _ms(key, digits=3):
            vals = [r[key] for r in rows
                    if isinstance(r.get(key), (int, float)) and not np.isnan(r[key])]
            if not vals:
                return "?", float("nan"), float("nan"), float("nan")
            m, s = float(np.mean(vals)), float(np.std(vals))
            label = f"{m:.{digits}f}±{s:.{digits}f}" if len(vals) > 1 else f"{m:.{digits}f}"
            return label, m, s, s**2

        act_vals = [r["active"] for r in rows
                    if isinstance(r.get("active"), (int, float)) and not np.isnan(r["active"])]
        act_str  = str(int(round(np.mean(act_vals)))) if act_vals else "?"

        r1_str,  r1_m,  _, _  = _ms("r1")
        r2_str,  r2_m,  _, _  = _ms("r2")
        r3_str,  r3_m,  _, _  = _ms("r3")
        r4_str,  r4_m, r4_s, r4_v = _ms("r4")
        dp_str,  _, _, _       = _ms("delta_pct", 1)
        ss_str,  _, _, _       = _ms("s_sep", 2)
        mg_str,  _, _, _       = _ms("mig", 3)

        n = len([r for r in rows if r.get("r4") and not np.isnan(r["r4"])])
        tprint(f"  {beta:>6.3f}  {act_str:>6}  {r1_str:>16}  {r2_str:>16}  "
               f"{r3_str:>16}  {r4_str:>16}  {dp_str:>8}  {ss_str:>10}  "
               f"{mg_str:>10}  (n={n})")
        tprint(f"  {'':>6}  {'':>6}  {'':>16}  {'':>16}  {'':>16}  "
               f"var={r4_v:.5f}  std={r4_s:.3f}")

        summary.append({
            "beta": beta, "n_seeds": n,
            "r1_mean": r1_m, "r2_mean": r2_m, "r3_mean": r3_m, "r4_mean": r4_m,
            "r4_std": r4_s, "r4_var": r4_v,
            **{f"{k}_mean": float(np.mean([r[k] for r in rows if r.get(k) and not np.isnan(r[k])]))
               if any(r.get(k) for r in rows) else float("nan")
               for k in ["s_sep", "mig", "delta_pct"]},
        })

    tprint(f"  {'─'*98}\n")

    MULTI_DIR.mkdir(parents=True, exist_ok=True)
    with open(MULTI_DIR / "focused_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    tprint(f"  Saved → {MULTI_DIR / 'focused_summary.json'}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed β-VAE sweep with real-time RMSE reporting.")
    parser.add_argument("--betas",    nargs="+", type=float, default=BETAS_DEFAULT)
    parser.add_argument("--seeds",    nargs="+", type=int,   default=SEEDS_DEFAULT)
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--parallel", type=int, default=2,
                        help="Concurrent (beta,seed) training jobs (default 2).")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-trained", action="store_true", default=True)
    parser.add_argument("--no-skip-trained", dest="skip_trained", action="store_false")
    parser.add_argument("--anneal", action="store_true",
                        help="Use β annealing (delay=20, ramp=100 epochs) instead of "
                             "fixed β. Strongly recommended for β≤0.05 to reduce "
                             "S_sep variance caused by dimensional collapse.")
    args = parser.parse_args()

    tprint(f"\n{'═'*70}")
    tprint(f"  β-VAE MULTI-SEED SWEEP  (focused)")
    tprint(f"  betas   : {args.betas}")
    tprint(f"  seeds   : {args.seeds}")
    tprint(f"  parallel: {args.parallel}")
    anneal_delay  = 20  if args.anneal else 0
    anneal_epochs = 100 if args.anneal else 1
    if args.anneal:
        tprint(f"  schedule: anneal (delay={anneal_delay}, ramp={anneal_epochs})")
    else:
        tprint(f"  schedule: fixed β (no annealing)")
    tprint(f"{'═'*70}\n")

    # ── Load data once (needed for evaluation) ────────────────────────────────
    tprint("  Loading CSV for evaluation ...")
    buckets = _load_buckets(CSV_PATH)

    # ── Per-beta result store ─────────────────────────────────────────────────
    all_results: Dict[float, List[Dict]] = {b: [] for b in args.betas}
    result_lock = threading.Lock()

    def _handle(beta: float, seed: int) -> Dict:
        """Train (if needed) + evaluate one (beta, seed)."""
        anneal = args.anneal
        if seed != 42 and not args.eval_only:
            rc = _train_one(beta, seed, args.device, args.skip_trained,
                            anneal_delay=anneal_delay,
                            anneal_epochs=anneal_epochs)
            if rc != 0:
                return {}

        tprint(f"  [β={beta}, s={seed}] Evaluating ...", flush=True)
        result = evaluate_one(beta, seed, buckets, args.device, anneal=anneal)
        if not result:
            return {}

        tprint(f"  [β={beta}, s={seed}] "
               f"R1={result['r1']:.3f}  R2={result['r2']:.3f}  "
               f"R3={result['r3']:.3f}  R4={result['r4']:.3f}  "
               f"Δ={result['delta_pct']:+.1f}%  "
               f"S_sep={result['s_sep']:.2f}  MIG={result['mig']:.3f}")

        with result_lock:
            all_results[beta].append(result)
            _print_running_stats(beta, list(all_results[beta]))

        eval_path = _eval_json(seed, anneal)
        if seed != 42 and not eval_path.exists():
            eval_path.parent.mkdir(parents=True, exist_ok=True)
            existing = json.load(open(eval_path)) if eval_path.exists() else []
            existing.append(result)
            with open(eval_path, "w") as f:
                json.dump(existing, f, indent=2, default=float)

        return result

    # ── Process each beta with parallel seeds ─────────────────────────────────
    for beta in args.betas:
        tprint(f"\n{'━'*70}")
        tprint(f"  β = {beta}")
        tprint(f"{'━'*70}")

        # Load seed-42 result from cache (no training)
        sweep42 = {r["beta"]: r
                   for r in json.load(open(_sweep_json(42)))
                   if _sweep_json(42).exists()} if _sweep_json(42).exists() else {}
        eval42  = {r["beta"]: r
                   for r in json.load(open(_eval_json(42)))
                   if _eval_json(42).exists()} if _eval_json(42).exists() else {}

        # Seed 42 is always processed inline (no training, just load from files)
        if 42 in args.seeds:
            s42 = sweep42.get(beta, {})
            e42 = eval42.get(beta, {})
            if s42 and e42:
                r42 = {
                    "beta": beta, "seed": 42,
                    "active":    s42.get("active_dims", float("nan")),
                    "r1":        s42.get("lstm_rmse_r1val", float("nan")),
                    "r2":        e42.get("rmse_r2", float("nan")),
                    "r3":        e42.get("rmse_r3", float("nan")),
                    "r4":        s42.get("lstm_rmse_fault", float("nan")),
                    "delta_pct": (s42["lstm_rmse_fault"] - s42["lstm_rmse_r1val"])
                                 / s42["lstm_rmse_r1val"] * 100
                                 if s42.get("lstm_rmse_r1val") else float("nan"),
                    "s_sep":     e42.get("s_sep", float("nan")),
                    "mig":       e42.get("mig",   float("nan")),
                }
                all_results[beta].append(r42)
                tprint(f"  [β={beta}, s=42] (reused)  "
                       f"R1={r42['r1']:.3f}  R4={r42['r4']:.3f}  "
                       f"S_sep={r42['s_sep']:.2f}  MIG={r42['mig']:.3f}")
                _print_running_stats(beta, list(all_results[beta]))
            else:
                tprint(f"  [β={beta}, s=42] cached results not found, evaluating ...")
                _handle(beta, 42)

        # Other seeds: train + eval in parallel
        other_seeds = [s for s in args.seeds if s != 42]
        if other_seeds:
            with ThreadPoolExecutor(max_workers=args.parallel) as ex:
                futs = {ex.submit(_handle, beta, s): s for s in other_seeds}
                for fut in as_completed(futs):
                    exc = fut.exception()
                    if exc:
                        seed = futs[fut]
                        tprint(f"  [β={beta}, s={seed}] EXCEPTION: {exc}")

    # ── Final aggregated table ────────────────────────────────────────────────
    _print_final_table(all_results)


if __name__ == "__main__":
    main()

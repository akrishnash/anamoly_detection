#!/usr/bin/env python3
"""
Diagnose the Ceiling — Score Overlap Between Benign and Malicious Flows
======================================================================
This is the core diagnostic of the "anomalous != malicious" study.

It answers ONE question: how much do the Isolation Forest anomaly scores for
benign flows overlap with the scores for malicious flows?

  - If the two score distributions are well separated  -> a good threshold exists.
  - If they overlap                                     -> NO threshold can win.
    The 60% performance is then a structural ceiling, not a tuning problem.

We quantify the overlap two ways:
  1. AUC  = P(a random attack scores more anomalous than a random benign flow).
            0.5 = no signal, 1.0 = perfect separation.
  2. KS   = the largest vertical gap between the two score CDFs.
            This EQUALS the best achievable (TPR - FPR) over all thresholds
            (max Youden's J), i.e. the ceiling of any threshold-based detector.

Run on real data (CTU-13 CSVs in sample_logs/):
    python diagnose_overlap.py --contamination 0.05

Run the synthetic demo (no data needed, just to see the figure):
    python diagnose_overlap.py --demo
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

ATTACK_CSV = "sample_logs/CTU13_Attack_Traffic.csv"
NORMAL_CSV = "sample_logs/CTU13_Normal_Traffic.csv"
N_SAMPLE = 6000
DROP_COLS = {"Unnamed: 0", "Label"}


# ---------------------------------------------------------------------------
# Data: real CTU-13 pipeline, or a synthetic demo
# ---------------------------------------------------------------------------

def scores_from_ctu13(contamination: float) -> tuple[np.ndarray, np.ndarray]:
    """Run the real IF pipeline and return (scores, y_true). Lower score = more anomalous."""
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    atk = pd.read_csv(ATTACK_CSV).sample(n=N_SAMPLE, random_state=42)
    nor = pd.read_csv(NORMAL_CSV).sample(n=N_SAMPLE, random_state=42)
    atk["true_label"] = 1
    nor["true_label"] = 0
    df = pd.concat([atk, nor], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    feat_cols = [c for c in df.columns if c not in DROP_COLS and c != "true_label"]
    X = df[feat_cols].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    X = X.loc[:, X.var() > 0]                       # drop zero-variance cols
    skewed = X.skew()[lambda s: s > 2].index
    X[skewed] = np.log1p(X[skewed].clip(lower=0))   # tame heavy tails

    Xs = StandardScaler().fit_transform(X.values)
    model = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=42, n_jobs=-1)
    model.fit(Xs)
    scores = model.score_samples(Xs)               # lower = more anomalous
    return scores, df["true_label"].values


def scores_demo() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic scores that mimic the real problem: heavily overlapping piles.

    Benign flows cluster at 'normal' scores; attacks are SUPPOSED to be lower,
    but stealthy ones blend right into the benign pile -> overlap.
    """
    rng = np.random.default_rng(42)
    benign = rng.normal(0.0, 1.0, 6000)
    # Two kinds of attacks: a few obvious (well separated) + many stealthy (overlapping)
    obvious = rng.normal(-3.0, 0.7, 1500)
    stealthy = rng.normal(-0.3, 1.0, 4500)          # sits on top of benign
    attack = np.concatenate([obvious, stealthy])
    scores = np.concatenate([benign, attack])
    y = np.concatenate([np.zeros(len(benign)), np.ones(len(attack))]).astype(int)
    return scores, y


# ---------------------------------------------------------------------------
# Measure the overlap
# ---------------------------------------------------------------------------

def measure(scores: np.ndarray, y: np.ndarray) -> dict:
    benign = scores[y == 0]
    attack = scores[y == 1]

    # AUC: higher anomaly = lower score, so feed -scores so "more anomalous" ranks high.
    auc = roc_auc_score(y, -scores)

    # KS distance between the two score distributions (= max Youden's J = the ceiling).
    ks_stat, _ = ks_2samp(attack, benign)

    return {
        "auc": auc,
        "ks": ks_stat,
        "benign": benign,
        "attack": attack,
        "n_benign": len(benign),
        "n_attack": len(attack),
    }


# ---------------------------------------------------------------------------
# Figure: left = the two piles, right = the CDF gap (what KS literally is)
# ---------------------------------------------------------------------------

def plot(m: dict, title_tag: str, out_path: str) -> None:
    benign, attack = m["benign"], m["attack"]
    lo, hi = np.percentile(np.concatenate([benign, attack]), [0.5, 99.5])
    grid = np.linspace(lo, hi, 400)
    bins = np.linspace(lo, hi, 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.patch.set_facecolor("#F4F6FA")
    for ax in (ax1, ax2):
        ax.set_facecolor("#EAEFF7")
        ax.grid(True, alpha=0.25, linestyle="--")

    C_BEN, C_ATK = "#2196F3", "#E53935"

    # ---- Left: the two piles, with their overlap shaded ----
    ax1.hist(benign, bins=bins, density=True, alpha=0.55, color=C_BEN,
             label=f"Benign  (n={m['n_benign']:,})")
    ax1.hist(attack, bins=bins, density=True, alpha=0.55, color=C_ATK,
             label=f"Malicious  (n={m['n_attack']:,})")

    # Shade the overlap = min of the two smoothed densities
    hb, _ = np.histogram(benign, bins=bins, density=True)
    ha, _ = np.histogram(attack, bins=bins, density=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    ax1.fill_between(centers, np.minimum(hb, ha), color="#6A1B9A", alpha=0.45,
                     label="OVERLAP (confusable)", zorder=3)
    ax1.set_xlabel("Isolation Forest anomaly score  (lower = more anomalous)", fontsize=9)
    ax1.set_ylabel("density", fontsize=9)
    ax1.set_title("The two score piles overlap\n"
                  f"AUC = {m['auc']:.3f}   (0.5 = no signal, 1.0 = perfect)",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8.5, loc="upper left")

    # ---- Right: the CDFs, with the KS gap drawn as a vertical arrow ----
    cdf_b = np.searchsorted(np.sort(benign), grid, side="right") / len(benign)
    cdf_a = np.searchsorted(np.sort(attack), grid, side="right") / len(attack)
    ax2.plot(grid, cdf_b, color=C_BEN, linewidth=2, label="Benign CDF")
    ax2.plot(grid, cdf_a, color=C_ATK, linewidth=2, label="Malicious CDF")

    gap = np.abs(cdf_a - cdf_b)
    k = int(np.argmax(gap))
    ax2.annotate("", xy=(grid[k], cdf_a[k]), xytext=(grid[k], cdf_b[k]),
                 arrowprops=dict(arrowstyle="<->", color="#FB8C00", lw=2.5))
    ax2.text(grid[k], (cdf_a[k] + cdf_b[k]) / 2,
             f"  KS = {m['ks']:.3f}", color="#E65100", fontsize=11, fontweight="bold",
             va="center")
    ax2.set_xlabel("Isolation Forest anomaly score", fontsize=9)
    ax2.set_ylabel("cumulative fraction", fontsize=9)
    ax2.set_title("KS = biggest vertical gap between the CDFs\n"
                  f"= the CEILING: best possible (TPR - FPR) = {m['ks']:.3f}",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8.5, loc="lower right")

    fig.suptitle(f"Anomalous ≠ Malicious  —  Score Overlap Diagnosis  ({title_tag})",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs("graphs", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[+] Figure saved -> {out_path}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose benign/malicious score overlap")
    ap.add_argument("--contamination", type=float, default=0.05,
                    help="IF contamination for the real CTU-13 run (default 0.05 = realistic)")
    ap.add_argument("--demo", action="store_true",
                    help="Use synthetic scores (no dataset required)")
    args = ap.parse_args()

    have_data = os.path.exists(ATTACK_CSV) and os.path.exists(NORMAL_CSV)
    if args.demo or not have_data:
        if not args.demo:
            print("[!] CTU-13 CSVs not found in sample_logs/ — running synthetic demo instead.")
        scores, y = scores_demo()
        tag, out = "SYNTHETIC DEMO", "graphs/overlap_demo.png"
    else:
        print(f"[*] Running real CTU-13 pipeline at contamination={args.contamination} ...")
        scores, y = scores_from_ctu13(args.contamination)
        tag, out = f"CTU-13, contamination={args.contamination:.0%}", "graphs/overlap_ctu13.png"

    m = measure(scores, y)

    print("\n" + "=" * 60)
    print("  SCORE OVERLAP DIAGNOSIS")
    print("=" * 60)
    print(f"  Benign flows   : {m['n_benign']:,}")
    print(f"  Malicious flows: {m['n_attack']:,}")
    print(f"  AUC            : {m['auc']:.3f}   (0.5 = no signal)")
    print(f"  KS distance    : {m['ks']:.3f}   (= ceiling: max achievable TPR-FPR)")
    print("=" * 60)
    if m["ks"] < 0.5:
        print("  -> Heavy overlap. No threshold can do well. The ceiling is STRUCTURAL.")
        print("     Fix = change the representation (Act 2), not the threshold.")
    else:
        print("  -> Decent separation. A good threshold exists; tune it.")
    print()

    plot(m, tag, out)


if __name__ == "__main__":
    main()

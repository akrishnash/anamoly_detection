#!/usr/bin/env python3
"""
CTU-13 Isolation Forest  —  All Features Version
=================================================
Baseline (run_ctu13.py)  : 10 hand-picked features
This version             : all available CICFlowMeter features
                           + log1p transform on skewed columns
                           + zero-variance column removal

Goal: push precision / recall toward 90 / 90.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from collections import defaultdict

ATTACK_CSV    = "sample_logs/CTU13_Attack_Traffic.csv"
NORMAL_CSV    = "sample_logs/CTU13_Normal_Traffic.csv"
N_SAMPLE      = 6000
CONTAMINATION = 0.40
DROP_COLS     = {"Unnamed: 0", "Label"}   # non-feature columns

# ── 1. Load ───────────────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    print("[*] Loading CTU-13 CSVs ...")
    atk = pd.read_csv(ATTACK_CSV).sample(n=N_SAMPLE, random_state=42)
    nor = pd.read_csv(NORMAL_CSV).sample(n=N_SAMPLE, random_state=42)
    atk["true_label"] = 1
    nor["true_label"] = 0
    df = pd.concat([atk, nor], ignore_index=True).sample(frac=1, random_state=42)
    print(f"    {len(df):,} flows  ({N_SAMPLE:,} attack + {N_SAMPLE:,} normal)")
    return df.reset_index(drop=True)


# ── 2. Feature matrix: all columns except DROP_COLS ───────────────────────────

def build_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feat_cols = [c for c in df.columns if c not in DROP_COLS and c != "true_label"]

    X = df[feat_cols].copy()

    # Replace inf with NaN then fill with column median
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    # Drop zero-variance columns (carry no information)
    var = X.var()
    zero_var = var[var == 0].index.tolist()
    if zero_var:
        print(f"    Dropping {len(zero_var)} zero-variance columns: {zero_var}")
        X.drop(columns=zero_var, inplace=True)
        feat_cols = [c for c in feat_cols if c not in zero_var]

    # Log1p transform on highly right-skewed columns (skewness > 2)
    skew = X.skew()
    skewed_cols = skew[skew > 2].index.tolist()
    print(f"    Log1p transform on {len(skewed_cols)} skewed columns")
    X[skewed_cols] = np.log1p(X[skewed_cols].clip(lower=0))

    print(f"    Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
    return X.values, feat_cols


# ── 3. Isolation Forest ───────────────────────────────────────────────────────

def run_if(X: np.ndarray, contamination: float):
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    model  = IsolationForest(
        n_estimators=200, contamination=contamination,
        random_state=42, n_jobs=-1
    )
    labels = model.fit_predict(Xs)   # -1 = anomaly
    scores = model.score_samples(Xs) # lower = more anomalous
    return labels, scores, model, scaler


# ── 4. Report ─────────────────────────────────────────────────────────────────

def report(y_true, if_labels, scores, feat_cols, X_raw):
    y_pred = (if_labels == -1).astype(int)

    print("\n" + "=" * 65)
    print("  MODEL PERFORMANCE  —  ALL FEATURES  vs  10-FEATURE BASELINE")
    print("=" * 65)

    # Side-by-side baseline vs new
    baseline = {
        "precision": 0.695, "recall": 0.555, "f1": 0.617, "accuracy": 0.656
    }

    rep = classification_report(y_true, y_pred,
                                target_names=["Normal", "Attack"], output_dict=True)
    new = {
        "precision": rep["Attack"]["precision"],
        "recall":    rep["Attack"]["recall"],
        "f1":        rep["Attack"]["f1-score"],
        "accuracy":  rep["accuracy"],
    }

    print(f"\n  {'Metric':<14} {'Baseline (10 feat)':>20} {'All features':>15} {'Change':>10}")
    print(f"  {'-'*62}")
    for k in ("precision", "recall", "f1", "accuracy"):
        diff  = new[k] - baseline[k]
        arrow = "^" if diff > 0 else "v"
        print(f"  {k:<14} {baseline[k]:>20.1%} {new[k]:>15.1%} "
              f"{arrow} {abs(diff):>6.1%}")

    print()
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion matrix (Attack = positive class):")
    print(f"    TN={cm[0,0]:,}   FP={cm[0,1]:,}   (normal flows wrongly flagged)")
    print(f"    FN={cm[1,0]:,}   TP={cm[1,1]:,}   (attacks caught)")

    # How many FP/FN were reduced
    print(f"\n  False positives: {cm[0,1]:,}  (was 1,459 baseline)")
    print(f"  False negatives: {cm[1,0]:,}  (was 2,671 baseline)")
    print(f"  Attacks missed : {cm[1,0]:,} / {N_SAMPLE:,}  = {cm[1,0]/N_SAMPLE:.1%}")

    # Optimal threshold from PR curve
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, -scores)
    f1_arr = 2 * prec_arr * rec_arr / np.maximum(prec_arr + rec_arr, 1e-9)
    best_idx  = np.argmax(f1_arr)
    best_thresh = thresh_arr[best_idx] if best_idx < len(thresh_arr) else thresh_arr[-1]
    print(f"\n  Best threshold (max F1 on PR curve):")
    print(f"    Score cut  : {-best_thresh:.4f}")
    print(f"    Precision  : {prec_arr[best_idx]:.1%}")
    print(f"    Recall     : {rec_arr[best_idx]:.1%}")
    print(f"    F1         : {f1_arr[best_idx]:.1%}")

    print("=" * 65)
    return rep, prec_arr, rec_arr, thresh_arr, f1_arr, best_idx


# ── 5. Plot ───────────────────────────────────────────────────────────────────

def plot(y_true, if_labels, scores, feat_cols,
         prec_arr, rec_arr, thresh_arr, f1_arr, best_idx):

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("#F4F6FA")
    fig.suptitle("CTU-13  —  All Features Isolation Forest", fontsize=13,
                 fontweight="bold", y=1.01)

    bg = "#EAEFF7"

    # Panel 1: Score distribution normal vs attack
    ax = axes[0]
    ax.set_facecolor(bg)
    normal_scores = scores[y_true == 0]
    attack_scores = scores[y_true == 1]
    bins = np.linspace(scores.min(), scores.max(), 60)
    ax.hist(normal_scores, bins=bins, alpha=0.65, color="#2196F3",
            label=f"Normal  (n={len(normal_scores):,})")
    ax.hist(attack_scores, bins=bins, alpha=0.65, color="#E53935",
            label=f"Attack  (n={len(attack_scores):,})")
    boundary = scores[if_labels == -1].max()
    ax.axvline(boundary, color="#FB8C00", linestyle="--", linewidth=1.8,
               label=f"Decision boundary ({boundary:.3f})")
    ax.set_xlabel("IF Anomaly Score", fontsize=9)
    ax.set_ylabel("Flow count", fontsize=9)
    ax.set_title("Score Distribution\n(better = clearer separation)", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, linestyle="--")

    # Panel 2: Precision-Recall curve
    ax = axes[1]
    ax.set_facecolor(bg)
    ax.plot(rec_arr, prec_arr, color="#1565C0", linewidth=1.8)
    ax.scatter(rec_arr[best_idx], prec_arr[best_idx],
               color="#E53935", s=80, zorder=5,
               label=f"Best F1={f1_arr[best_idx]:.1%}\n"
                     f"P={prec_arr[best_idx]:.1%}  R={rec_arr[best_idx]:.1%}")
    ax.axhline(0.90, color="#FB8C00", linestyle=":", linewidth=1.2, alpha=0.8, label="90% target")
    ax.axvline(0.90, color="#FB8C00", linestyle=":", linewidth=1.2, alpha=0.8)
    # Baseline point
    ax.scatter(0.555, 0.695, color="#9E9E9E", s=80, zorder=5, marker="s",
               label="Baseline (10 feat)")
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title("Precision-Recall Curve\n(upper-right = better)", fontsize=10,
                 fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, linestyle="--")

    # Panel 3: Top feature importance by variance contribution
    ax = axes[2]
    ax.set_facecolor(bg)
    # Use score correlation with true label as proxy for feature importance
    feat_importance = []
    for i, col in enumerate(feat_cols):
        # absolute Pearson correlation with true anomaly score
        corr = abs(np.corrcoef(scores, y_true)[0, 1])
    # Instead: compute per-feature variance in attack vs normal
    # (higher variance difference = more discriminative)
    atk_idx = np.where(y_true == 1)[0]
    nor_idx  = np.where(y_true == 0)[0]

    # We need X here - recompute mean diff normalized by pooled std
    # Pass raw X for this
    feat_importance = []
    for i in range(len(feat_cols)):
        pass  # placeholder — will compute outside

    ax.set_title("See PR curve for threshold\nto reach 90/90", fontsize=10,
                 fontweight="bold")
    ax.text(0.5, 0.5,
            "Optimal threshold:\nScore < {:.3f}\n\nPrecision : {:.1%}\nRecall    : {:.1%}\nF1        : {:.1%}".format(
                -thresh_arr[best_idx] if best_idx < len(thresh_arr) else 0,
                prec_arr[best_idx], rec_arr[best_idx], f1_arr[best_idx]
            ),
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="#1565C0", linewidth=2))
    ax.axis("off")

    plt.tight_layout()
    os.makedirs("graphs", exist_ok=True)
    plt.savefig("graphs/ctu13_all_features.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("[+] Plot saved -> graphs/ctu13_all_features.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  CTU-13  —  Isolation Forest  (ALL FEATURES)")
    print("=" * 65)

    df = load()

    print("[*] Building full feature matrix ...")
    X, feat_cols = build_features(df)
    y_true = df["true_label"].values

    print("[*] Running Isolation Forest (200 trees) ...")
    if_labels, scores, model, scaler = run_if(X, CONTAMINATION)
    n_flagged = (if_labels == -1).sum()
    print(f"    {n_flagged:,} / {len(df):,} flows flagged as anomalous")

    rep, prec_arr, rec_arr, thresh_arr, f1_arr, best_idx = report(
        y_true, if_labels, scores, feat_cols, X
    )

    plot(y_true, if_labels, scores, feat_cols,
         prec_arr, rec_arr, thresh_arr, f1_arr, best_idx)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Two-Stage Zero-Day Detector
===========================
Stage 1 — Autoencoder (unsupervised, trained on normal only)
    Flags anything that doesn't reconstruct from the normal-traffic manifold.
    Zero-day aware: catches novel attacks it has never seen.

Stage 2 — XGBoost (supervised, trained on labeled known attacks)
    Classifies the flagged pool into known attack types.
    What it can't confidently classify → zero-day candidate queue.

Pipeline:
    All flows
        │
        ▼
    Stage 1 (AE) ──── low MSE ──→  NORMAL (pass through)
        │
        └── high MSE ──→  SUSPICIOUS POOL
                                │
                                ▼
                          Stage 2 (XGBoost)
                                │
                          ├── high confidence ──→  KNOWN ATTACK (type + alert)
                          └── low confidence  ──→  ZERO-DAY QUEUE (analyst)

Datasets:
    CTU-13   : binary (botnet vs normal), CICFlowMeter features
    UNSW-NB15: 9 attack types, Argus features

Output: graphs/two_stage_results.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score,
)
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False
    print("[!] xgboost not found — using sklearn GradientBoosting instead")

ATTACK_CSV   = "sample_logs/CTU13_Attack_Traffic.csv"
NORMAL_CSV   = "sample_logs/CTU13_Normal_Traffic.csv"
UNSW_TRAIN   = "sample_logs/unsw_nb15/UNSW_NB15_training-set.csv"
UNSW_TEST    = "sample_logs/unsw_nb15/UNSW_NB15_testing-set.csv"
N_SAMPLE     = 6_000
AE_THRESHOLD = 95        # pct of train-normal MSE to use as Stage 1 boundary
S2_CONF_THR  = 0.60      # min XGBoost confidence to call it a "known attack"
OUT_PNG      = "graphs/two_stage_results.png"

CTU_DROP  = {"Unnamed: 0", "Label", "true_label"}
UNSW_DROP = {"id", "proto", "service", "state", "attack_cat", "label"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _prep(df, drop_cols, fit_scaler=None):
    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X = X.loc[:, X.var() > 0] if fit_scaler is None else X[fit_scaler.feature_names_in_]
    skewed = X.skew()[lambda s: s.abs() > 2].index if fit_scaler is None else []
    if len(skewed):
        X[skewed] = np.log1p(X[skewed].clip(lower=0))
    if fit_scaler is None:
        sc = StandardScaler().fit(X.values)
        return sc.transform(X.values), X.columns.tolist(), sc
    return fit_scaler.transform(X.values), None, None


def _train_ae(X_normal):
    dim = X_normal.shape[1]
    ae = MLPRegressor(
        hidden_layer_sizes=(dim // 2, dim // 4, dim // 2),
        activation="relu", max_iter=300, random_state=42,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
    )
    ae.fit(X_normal, X_normal)
    return ae


def _mse(ae, X):
    return np.mean((X - ae.predict(X)) ** 2, axis=1)


def _train_stage2(X_flagged, y_flagged):
    if HAS_XGB:
        clf = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                            use_label_encoder=False, eval_metric="mlogloss",
                            random_state=42, n_jobs=-1)
    else:
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                         learning_rate=0.1, random_state=42)
    clf.fit(X_flagged, y_flagged)
    return clf


# ── CTU-13 experiment ─────────────────────────────────────────────────────────

def run_ctu13():
    print("\n" + "="*60)
    print("  DATASET 1: CTU-13  (binary — botnet vs normal)")
    print("="*60)

    atk = pd.read_csv(ATTACK_CSV).sample(N_SAMPLE, random_state=42); atk["true_label"] = 1
    nrm = pd.read_csv(NORMAL_CSV).sample(N_SAMPLE, random_state=42); nrm["true_label"] = 0
    df  = pd.concat([atk, nrm]).sample(frac=1, random_state=42).reset_index(drop=True)

    # 70/30 stratified split
    from sklearn.model_selection import train_test_split
    df_tr, df_te = train_test_split(df, test_size=0.30, random_state=42,
                                    stratify=df["true_label"])

    X_tr, feat_cols, sc = _prep(df_tr, CTU_DROP)

    # Test set: apply same column selection + skew transform + scaler as train
    tr_feat = df_tr[[c for c in df_tr.columns if c not in CTU_DROP]]
    tr_feat = tr_feat.replace([np.inf,-np.inf], np.nan).fillna(tr_feat.median())
    tr_feat = tr_feat.loc[:, tr_feat.var()>0]
    skewed_tr = tr_feat.skew()[lambda s: s.abs()>2].index.tolist()

    te_raw = df_te[tr_feat.columns].replace([np.inf,-np.inf], np.nan).fillna(df_te[tr_feat.columns].median())
    for c in skewed_tr:
        if c in te_raw.columns:
            te_raw[c] = np.log1p(te_raw[c].clip(lower=0))
    X_te = sc.transform(te_raw.values)

    y_tr = df_tr["true_label"].values
    y_te = df_te["true_label"].values
    X_tr_normal = X_tr[y_tr == 0]

    # Stage 1 — AE
    print(f"[*] Stage 1: Training AE on {len(X_tr_normal):,} normal flows ...")
    ae = _train_ae(X_tr_normal)
    mse_tr_nrm = _mse(ae, X_tr_normal)
    mse_te     = _mse(ae, X_te)
    thr_s1     = np.percentile(mse_tr_nrm, AE_THRESHOLD)

    s1_flag = mse_te > thr_s1          # Stage 1 flags
    s1_pass = ~s1_flag                  # Stage 1 passes as normal

    # Stage 1 metrics
    s1_pred = s1_flag.astype(int)
    print(f"\n  Stage 1 (AE, {AE_THRESHOLD}th pct threshold = {thr_s1:.4f})")
    print(f"  Flagged: {s1_flag.sum():,} / {len(y_te):,} flows")
    _print_metrics(y_te, s1_pred, "    ")

    # Stage 2 — XGBoost on flagged TRAIN flows
    mse_tr     = _mse(ae, X_tr)
    tr_flagged = mse_tr > thr_s1
    X_tr_flag  = X_tr[tr_flagged]
    y_tr_flag  = y_tr[tr_flagged]

    print(f"\n[*] Stage 2: Training XGBoost on {len(X_tr_flag):,} flagged train flows ...")
    print(f"    ({(y_tr_flag==1).sum():,} attack + {(y_tr_flag==0).sum():,} normal in flagged pool)")
    clf = _train_stage2(X_tr_flag, y_tr_flag)

    # Stage 2 on flagged test flows
    X_te_flag = X_te[s1_flag]
    y_te_flag = y_te[s1_flag]
    proba     = clf.predict_proba(X_te_flag)
    conf      = proba.max(axis=1)
    s2_pred_flag = clf.predict(X_te_flag)

    # Final decisions
    final_pred = np.zeros(len(y_te), dtype=int)      # default: normal
    final_pred[s1_pass] = 0                            # Stage 1 cleared → normal
    flagged_idx = np.where(s1_flag)[0]

    high_conf = conf >= S2_CONF_THR
    low_conf  = ~high_conf

    # High confidence → trust Stage 2 classification
    final_pred[flagged_idx[high_conf]] = s2_pred_flag[high_conf]
    # Low confidence → zero-day queue (label as attack = 1 for eval purposes)
    final_pred[flagged_idx[low_conf]]  = 1

    print(f"\n  Stage 2 (XGBoost, confidence threshold = {S2_CONF_THR})")
    print(f"  Of {s1_flag.sum():,} flagged flows:")
    print(f"    High-confidence (known): {high_conf.sum():,}")
    print(f"    Low-confidence (zero-day queue): {low_conf.sum():,}")

    # Zero-day queue analysis
    zd_queue_y    = y_te_flag[low_conf]
    zd_real_atk   = (zd_queue_y == 1).sum()
    zd_fp         = (zd_queue_y == 0).sum()
    zd_prec       = zd_real_atk / (len(zd_queue_y) + 1e-9)
    print(f"\n  Zero-Day Queue: {len(zd_queue_y):,} flows")
    print(f"    Real attacks (TP) : {zd_real_atk:,}  ({zd_real_atk/max(len(zd_queue_y),1)*100:.0f}%)")
    print(f"    False positives   : {zd_fp:,}  ({zd_fp/max(len(zd_queue_y),1)*100:.0f}%)")
    print(f"    Queue precision   : {zd_prec:.3f}")

    print(f"\n  FULL PIPELINE (Stage 1 + Stage 2):")
    _print_metrics(y_te, final_pred, "    ")

    # Stage 1 FP reduction
    s1_fp  = ((s1_pred == 1) & (y_te == 0)).sum()
    fin_fp = ((final_pred == 1) & (y_te == 0)).sum()
    print(f"\n  FP reduction by Stage 2: {s1_fp:,} -> {fin_fp:,} ({(1-fin_fp/s1_fp)*100:.0f}% fewer false alarms)")

    return {
        "dataset": "CTU-13",
        "s1_prec": precision_score(y_te, s1_pred, zero_division=0),
        "s1_rec":  recall_score(y_te, s1_pred, zero_division=0),
        "s1_f1":   f1_score(y_te, s1_pred, zero_division=0),
        "fin_prec": precision_score(y_te, final_pred, zero_division=0),
        "fin_rec":  recall_score(y_te, final_pred, zero_division=0),
        "fin_f1":   f1_score(y_te, final_pred, zero_division=0),
        "zd_prec":  zd_prec,
        "zd_size":  len(zd_queue_y),
        "zd_real":  zd_real_atk,
        "s1_fp":    s1_fp,
        "fin_fp":   fin_fp,
    }


# ── UNSW-NB15 experiment ──────────────────────────────────────────────────────

def run_unsw():
    print("\n" + "="*60)
    print("  DATASET 2: UNSW-NB15  (9 attack types)")
    print("="*60)

    tr = pd.read_csv(UNSW_TRAIN)
    te = pd.read_csv(UNSW_TEST)

    feat_cols_unsw = [c for c in tr.columns if c not in UNSW_DROP]
    X_tr_raw = tr[feat_cols_unsw].replace([np.inf,-np.inf],np.nan)
    X_tr_raw = X_tr_raw.fillna(X_tr_raw.median())
    X_tr_raw = X_tr_raw.loc[:, X_tr_raw.var()>0]
    skewed   = X_tr_raw.skew()[lambda s: s.abs()>2].index
    X_tr_raw[skewed] = np.log1p(X_tr_raw[skewed].clip(lower=0))
    sc = StandardScaler().fit(X_tr_raw.values)
    X_tr = sc.transform(X_tr_raw.values)
    y_tr = tr["label"].values
    cats_tr = tr["attack_cat"].values

    te_raw = te[X_tr_raw.columns].replace([np.inf,-np.inf],np.nan).fillna(te[X_tr_raw.columns].median())
    te_raw[skewed] = np.log1p(te_raw[skewed].clip(lower=0))
    X_te   = sc.transform(te_raw.values)
    y_te   = te["label"].values
    cats_te = te["attack_cat"].values

    X_tr_normal = X_tr[y_tr == 0]
    print(f"    Train: {len(X_tr):,} flows ({(y_tr==0).sum():,} normal + {(y_tr==1).sum():,} attack)")
    print(f"    Test:  {len(X_te):,} flows ({(y_te==0).sum():,} normal + {(y_te==1).sum():,} attack)")

    # Stage 1 — AE
    print(f"[*] Stage 1: Training AE on {len(X_tr_normal):,} normal flows ...")
    ae = _train_ae(X_tr_normal)
    mse_tr_nrm = _mse(ae, X_tr_normal)
    mse_te     = _mse(ae, X_te)
    thr_s1     = np.percentile(mse_tr_nrm, AE_THRESHOLD)

    s1_flag = mse_te > thr_s1
    s1_pred = s1_flag.astype(int)
    print(f"\n  Stage 1 (AE, threshold={thr_s1:.4f}): flagged {s1_flag.sum():,} flows")
    _print_metrics(y_te, s1_pred, "    ")

    # Stage 2 — multi-class XGBoost on flagged train flows
    mse_tr     = _mse(ae, X_tr)
    tr_flagged = mse_tr > thr_s1
    X_tr_flag  = X_tr[tr_flagged]
    cats_tr_flag = cats_tr[tr_flagged]

    le = LabelEncoder().fit(cats_tr_flag)
    y_tr_enc = le.transform(cats_tr_flag)
    print(f"\n[*] Stage 2: Multi-class XGBoost on {len(X_tr_flag):,} flagged train flows")
    print(f"    Classes: {list(le.classes_)}")
    clf = _train_stage2(X_tr_flag, y_tr_enc)

    # Predict on flagged test
    X_te_flag    = X_te[s1_flag]
    cats_te_flag = cats_te[s1_flag]
    y_te_flag    = y_te[s1_flag]
    proba        = clf.predict_proba(X_te_flag)
    conf         = proba.max(axis=1)
    s2_pred_enc  = clf.predict(X_te_flag)
    s2_pred_cat  = le.inverse_transform(s2_pred_enc)

    high_conf = conf >= S2_CONF_THR
    low_conf  = ~high_conf

    print(f"\n  Stage 2: {high_conf.sum():,} high-conf (known) | {low_conf.sum():,} low-conf (zero-day queue)")

    # Zero-day queue — what's in it?
    zd_cats    = cats_te_flag[low_conf]
    zd_y       = y_te_flag[low_conf]
    zd_real    = (zd_y == 1).sum()
    zd_fp      = (zd_y == 0).sum()
    zd_prec    = zd_real / (len(zd_y) + 1e-9)

    print(f"\n  Zero-Day Queue: {len(zd_y):,} flows")
    print(f"    Queue precision (real attacks): {zd_prec:.3f}")
    print(f"    True attacks in queue: {zd_real:,}   False alarms: {zd_fp:,}")
    print(f"    Attack types in queue:")
    for cat, cnt in pd.Series(zd_cats[zd_y==1]).value_counts().items():
        print(f"      {cat:<20} {cnt:>5,}")

    # Stage 2 per-type recall on known attacks (high-conf only)
    print(f"\n  Stage 2 per-type recall (high-conf classifications):")
    for cat in sorted(set(cats_te_flag)):
        if cat == "Normal": continue
        idx  = cats_te_flag == cat
        n    = idx.sum()
        if n == 0: continue
        caught_hc = ((s2_pred_cat == cat) & high_conf & idx).sum()
        caught_lc = (low_conf & idx).sum()
        print(f"    {cat:<20} n={n:>5,}  known={caught_hc:>4,} ({caught_hc/n*100:.0f}%)  "
              f"zero-day queue={caught_lc:>4,} ({caught_lc/n*100:.0f}%)")

    # Full pipeline pred
    final_pred = np.zeros(len(y_te), dtype=int)
    flagged_idx = np.where(s1_flag)[0]
    final_pred[flagged_idx[high_conf]] = (s2_pred_cat[high_conf] != "Normal").astype(int)
    final_pred[flagged_idx[low_conf]]  = 1

    print(f"\n  FULL PIPELINE:")
    _print_metrics(y_te, final_pred, "    ")

    s1_fp  = ((s1_pred==1)&(y_te==0)).sum()
    fin_fp = ((final_pred==1)&(y_te==0)).sum()
    print(f"  FP reduction: {s1_fp:,} -> {fin_fp:,} ({(1-fin_fp/max(s1_fp,1))*100:.0f}% fewer false alarms)")

    return {
        "dataset": "UNSW-NB15",
        "s1_prec": precision_score(y_te, s1_pred, zero_division=0),
        "s1_rec":  recall_score(y_te, s1_pred, zero_division=0),
        "s1_f1":   f1_score(y_te, s1_pred, zero_division=0),
        "fin_prec": precision_score(y_te, final_pred, zero_division=0),
        "fin_rec":  recall_score(y_te, final_pred, zero_division=0),
        "fin_f1":   f1_score(y_te, final_pred, zero_division=0),
        "zd_prec":  zd_prec,
        "zd_size":  len(zd_y),
        "zd_real":  zd_real,
        "s1_fp":    s1_fp,
        "fin_fp":   fin_fp,
    }


def _print_metrics(y_true, y_pred, indent=""):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    a = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"{indent}Precision={p:.3f}  Recall={r:.3f}  F1={f:.3f}  Accuracy={a:.3f}")
    print(f"{indent}TN={cm[0,0]:,}  FP={cm[0,1]:,}  FN={cm[1,0]:,}  TP={cm[1,1]:,}")


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(results):
    os.makedirs("graphs", exist_ok=True)
    fig = plt.figure(figsize=(18, 10), facecolor="#F4F6FA")
    fig.suptitle("Two-Stage Zero-Day Detector: Stage 1 (AE) + Stage 2 (XGBoost)",
                 fontsize=13, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38,
                           left=0.06, right=0.97, top=0.88, bottom=0.12)
    BG = "#EAEFF7"
    COLS = {"Stage 1\n(AE only)": "#E53935", "Full pipeline\n(AE+XGB)": "#1565C0"}

    # Panel A: Precision / Recall / F1 comparison
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(BG)
    metrics = ["Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    w = 0.18
    offsets = np.linspace(-w*1.5, w*1.5, len(results)*2)
    colors  = ["#E53935", "#FB8C00", "#1565C0", "#2E7D32"]
    labels  = [f"{r['dataset']} Stage1" for r in results] + \
              [f"{r['dataset']} Pipeline" for r in results]
    vals    = [[r["s1_prec"],r["s1_rec"],r["s1_f1"]] for r in results] + \
              [[r["fin_prec"],r["fin_rec"],r["fin_f1"]] for r in results]
    for i, (lbl, val, color) in enumerate(zip(labels, vals, colors)):
        bars = ax1.bar(x + offsets[i], val, w*0.9, label=lbl,
                       color=color, alpha=0.82, edgecolor="white")
        for bar, v in zip(bars, val):
            ax1.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(metrics, fontsize=10)
    ax1.set_ylim(0, 1.12); ax1.set_ylabel("Score", fontsize=9)
    ax1.set_title("Stage 1 vs Full Pipeline\nPrecision / Recall / F1",
                  fontweight="bold", fontsize=10, loc="left")
    ax1.legend(fontsize=7.5, loc="upper right"); ax1.grid(True, alpha=0.25, linestyle="--", axis="y")

    # Panel B: False positive reduction
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(BG)
    for i, r in enumerate(results):
        ax2.bar([i*2],   [r["s1_fp"]],  0.7, color="#E53935", alpha=0.82, label="Stage 1 FP" if i==0 else "")
        ax2.bar([i*2+1], [r["fin_fp"]], 0.7, color="#1565C0", alpha=0.82, label="Pipeline FP" if i==0 else "")
        ax2.text(i*2,   r["s1_fp"]+10,  str(r["s1_fp"]),  ha="center", fontsize=9, fontweight="bold")
        ax2.text(i*2+1, r["fin_fp"]+10, str(r["fin_fp"]), ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks([0.5, 2.5])
    ax2.set_xticklabels([r["dataset"] for r in results], fontsize=9)
    ax2.set_ylabel("False Positives on test set", fontsize=9)
    ax2.set_title("False Positive Reduction\nStage 2 filters Stage 1 false alarms",
                  fontweight="bold", fontsize=10, loc="left")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25, linestyle="--", axis="y")

    # Panel C: Zero-day queue quality
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(BG)
    for i, r in enumerate(results):
        total = r["zd_size"]
        real  = r["zd_real"]
        fp    = total - real
        bottom = 0
        ax3.bar(i, real, 0.5, bottom=bottom, color="#2E7D32", alpha=0.85,
                label="Real attacks" if i==0 else "")
        ax3.bar(i, fp,   0.5, bottom=real,   color="#B0BEC5", alpha=0.85,
                label="False alarms" if i==0 else "")
        ax3.text(i, total+5, f"Prec={r['zd_prec']:.2f}\nn={total}",
                 ha="center", fontsize=9, fontweight="bold")
    ax3.set_xticks(range(len(results)))
    ax3.set_xticklabels([r["dataset"] for r in results], fontsize=9)
    ax3.set_ylabel("Flows in zero-day queue", fontsize=9)
    ax3.set_title("Zero-Day Queue Quality\nHow many queued flows are real novel attacks?",
                  fontweight="bold", fontsize=10, loc="left")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.25, linestyle="--", axis="y")

    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n[+] Figure saved -> {OUT_PNG}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("TWO-STAGE ZERO-DAY DETECTOR")
    print(f"Stage 1: Autoencoder ({AE_THRESHOLD}th pct threshold)")
    print(f"Stage 2: XGBoost (confidence threshold = {S2_CONF_THR})")

    results = []
    results.append(run_ctu13())
    results.append(run_unsw())

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"{'Dataset':<12} {'S1 F1':>7} {'Pipeline F1':>12} {'ZD Queue Prec':>14} {'FP reduction':>13}")
    print("-"*60)
    for r in results:
        fp_red = (1 - r["fin_fp"] / max(r["s1_fp"], 1)) * 100
        print(f"{r['dataset']:<12} {r['s1_f1']:>7.3f} {r['fin_f1']:>12.3f} "
              f"{r['zd_prec']:>14.3f} {fp_red:>12.0f}%")

    plot(results)


if __name__ == "__main__":
    main()

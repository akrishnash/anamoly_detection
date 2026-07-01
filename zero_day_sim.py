#!/usr/bin/env python3
"""
Zero-Day Simulation — The Critical Experiment
==============================================
Question: If the two-stage system has NEVER seen an attack type during training,
will it still catch it in the zero-day queue?

Setup:
  - Dataset: UNSW-NB15 (9 attack types)
  - Stage 1: AE trained on normal flows only (no labels needed)
  - Stage 2: GBM/XGBoost trained on 5 KNOWN attack types only
  - HIDDEN (zero-day): 4 attack types withheld from Stage 2 entirely
      Backdoor, Shellcode, Analysis, Worms
      (rare, structurally different from common attacks)

If the architecture works:
  - Stage 1 (AE) flags novel traffic as anomalous
  - Stage 2 can't classify it (low confidence) -> zero-day queue
  - Zero-day queue should contain those hidden attack types at HIGH precision

This is the evidence that makes the paper publishable.

Output:
  graphs/zero_day_sim.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    def make_clf(n_classes):
        return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             use_label_encoder=False, eval_metric="mlogloss",
                             random_state=42, n_jobs=-1)
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    def make_clf(n_classes):
        return GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                          learning_rate=0.1, random_state=42)

UNSW_TRAIN  = "sample_logs/unsw_nb15/UNSW_NB15_training-set.csv"
UNSW_TEST   = "sample_logs/unsw_nb15/UNSW_NB15_testing-set.csv"
DROP_COLS   = {"id", "proto", "service", "state", "attack_cat", "label"}
AE_PCT      = 95
S2_CONF     = 0.55
OUT_PNG     = "graphs/zero_day_sim.png"

# Attack types Stage 2 is trained on (KNOWN)
KNOWN_TYPES = {"Generic", "Exploits", "DoS", "Fuzzers", "Reconnaissance"}

# Attack types completely hidden from Stage 2 during training (ZERO-DAY)
HIDDEN_TYPES = {"Backdoor", "Shellcode", "Analysis", "Worms"}


def load():
    tr = pd.read_csv(UNSW_TRAIN)
    te = pd.read_csv(UNSW_TEST)

    feat_cols = [c for c in tr.columns if c not in DROP_COLS]
    X_tr_raw = tr[feat_cols].replace([np.inf,-np.inf], np.nan)
    X_tr_raw = X_tr_raw.fillna(X_tr_raw.median())
    X_tr_raw = X_tr_raw.loc[:, X_tr_raw.var()>0]
    skewed   = X_tr_raw.skew()[lambda s: s.abs()>2].index
    X_tr_raw[skewed] = np.log1p(X_tr_raw[skewed].clip(lower=0))
    sc       = StandardScaler().fit(X_tr_raw.values)
    X_tr     = sc.transform(X_tr_raw.values)
    y_tr     = tr["label"].values
    cats_tr  = tr["attack_cat"].str.strip().values

    te_raw   = te[X_tr_raw.columns].replace([np.inf,-np.inf], np.nan).fillna(te[X_tr_raw.columns].median())
    te_raw[skewed] = np.log1p(te_raw[skewed].clip(lower=0))
    X_te     = sc.transform(te_raw.values)
    y_te     = te["label"].values
    cats_te  = te["attack_cat"].str.strip().values

    return X_tr, y_tr, cats_tr, X_te, y_te, cats_te


def train_ae(X_normal):
    dim = X_normal.shape[1]
    ae = MLPRegressor(hidden_layer_sizes=(dim//2, dim//4, dim//2),
                      activation="relu", max_iter=300, random_state=42,
                      early_stopping=True, validation_fraction=0.1,
                      n_iter_no_change=15)
    ae.fit(X_normal, X_normal)
    return ae


def mse(ae, X):
    return np.mean((X - ae.predict(X))**2, axis=1)


def main():
    print("ZERO-DAY SIMULATION")
    print(f"Known attack types  : {sorted(KNOWN_TYPES)}")
    print(f"Hidden (zero-day)   : {sorted(HIDDEN_TYPES)}")
    print()

    X_tr, y_tr, cats_tr, X_te, y_te, cats_te = load()

    # Stage 1 — AE on normal only
    X_tr_normal = X_tr[y_tr == 0]
    print(f"[*] Stage 1: training AE on {len(X_tr_normal):,} normal flows ...")
    ae = train_ae(X_tr_normal)

    mse_tr_nrm = mse(ae, X_tr_normal)
    mse_te     = mse(ae, X_te)
    thr_s1     = np.percentile(mse_tr_nrm, AE_PCT)
    s1_flag    = mse_te > thr_s1
    print(f"    Stage 1 flags {s1_flag.sum():,} / {len(y_te):,} flows  (threshold={thr_s1:.4f})")

    # Stage 2 — train ONLY on known attack types + normal (hidden types excluded)
    mse_tr       = mse(ae, X_tr)
    thr_tr       = np.percentile(mse_tr[y_tr==0], AE_PCT)
    tr_flagged   = mse_tr > thr_tr

    # Keep only Normal and KNOWN attack types for Stage 2 training
    known_mask = np.array([(c == "Normal" or c in KNOWN_TYPES) for c in cats_tr])
    train_s2_idx = np.where(tr_flagged & known_mask)[0]
    X_s2_tr  = X_tr[train_s2_idx]
    y_s2_tr  = cats_tr[train_s2_idx]   # multi-class category labels

    le  = LabelEncoder().fit(y_s2_tr)
    clf = make_clf(len(le.classes_))
    clf.fit(X_s2_tr, le.transform(y_s2_tr))
    print(f"[*] Stage 2: trained on {len(X_s2_tr):,} flagged known-type flows")
    print(f"    Classes known to Stage 2: {list(le.classes_)}")
    print()

    # Test pipeline on test set
    X_te_flag    = X_te[s1_flag]
    cats_te_flag = cats_te[s1_flag]
    y_te_flag    = y_te[s1_flag]
    proba        = clf.predict_proba(X_te_flag)
    conf         = proba.max(axis=1)
    pred_cat     = le.inverse_transform(clf.predict(X_te_flag))

    high_conf = conf >= S2_CONF
    low_conf  = ~high_conf   # zero-day queue

    # Results per attack type in the zero-day queue
    print("=" * 62)
    print("  PER-TYPE RESULT (of Stage 1 flagged flows)")
    print(f"  {'Attack type':<22} {'In S1 flag':>10} {'Known':>8} {'ZD queue':>10} {'Missed':>8}")
    print("-" * 62)

    type_results = {}
    all_cats = sorted(set(cats_te_flag) - {"Normal"})
    for cat in all_cats:
        mask     = cats_te_flag == cat
        n        = mask.sum()
        known_n  = (high_conf & (pred_cat == cat) & mask).sum() if cat in KNOWN_TYPES else 0
        zd_n     = (low_conf & mask).sum()
        is_hidden = cat in HIDDEN_TYPES
        marker   = "  [ZERO-DAY]" if is_hidden else ""
        print(f"  {cat:<22} {n:>10,} {known_n:>8,} {zd_n:>10,}   {(n-known_n-zd_n):>4,}{marker}")
        type_results[cat] = {"n": n, "known": known_n, "zd": zd_n, "hidden": is_hidden}
    print("=" * 62)

    # Zero-day queue breakdown
    zd_cats  = cats_te_flag[low_conf]
    zd_y     = y_te_flag[low_conf]
    zd_total = len(zd_cats)
    zd_real  = (zd_y == 1).sum()
    zd_fp    = (zd_y == 0).sum()
    zd_hidden_n = sum((zd_cats == c).sum() for c in HIDDEN_TYPES)

    print(f"\n  ZERO-DAY QUEUE: {zd_total:,} flows total")
    print(f"    Real attacks       : {zd_real:,}  ({zd_real/max(zd_total,1)*100:.0f}%)")
    print(f"    False alarms (nrm) : {zd_fp:,}  ({zd_fp/max(zd_total,1)*100:.0f}%)")
    print(f"    Hidden-type flows  : {zd_hidden_n:,}  ({zd_hidden_n/max(zd_total,1)*100:.0f}% of queue)")
    print(f"\n    Queue composition (attack types):")
    for cat, cnt in pd.Series(zd_cats).value_counts().items():
        tag = "  <- HIDDEN (zero-day)" if cat in HIDDEN_TYPES else ""
        print(f"      {cat:<22} {cnt:>5,}{tag}")

    # Detection rate for hidden types
    print(f"\n  HIDDEN TYPE DETECTION RATE (how many reached the ZD queue):")
    for cat in sorted(HIDDEN_TYPES):
        total_te = (cats_te == cat).sum()
        in_s1    = (cats_te_flag == cat).sum()
        in_zd    = (zd_cats == cat).sum()
        s1_pct   = in_s1/max(total_te,1)*100
        zd_pct   = in_zd/max(total_te,1)*100
        print(f"    {cat:<12} total={total_te:>4,}  Stage1 flagged={in_s1:>4,} ({s1_pct:.0f}%)  "
              f"ZD queue={in_zd:>4,} ({zd_pct:.0f}%)")

    plot(type_results, zd_cats, zd_y, HIDDEN_TYPES, KNOWN_TYPES)

    return type_results, zd_hidden_n, zd_total


def plot(type_results, zd_cats, zd_y, hidden_types, known_types):
    os.makedirs("graphs", exist_ok=True)
    fig = plt.figure(figsize=(18, 9), facecolor="#F4F6FA")
    fig.suptitle(
        "Zero-Day Simulation: Stage 2 trained on 5 known attack types\n"
        "Do the 4 HIDDEN types land in the zero-day queue?",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38,
                           left=0.06, right=0.97, top=0.87, bottom=0.14)
    BG = "#EAEFF7"

    # Panel A: Per-type flagging bar chart
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(BG)
    cats     = list(type_results.keys())
    n_vals   = [type_results[c]["n"]    for c in cats]
    zd_vals  = [type_results[c]["zd"]   for c in cats]
    kn_vals  = [type_results[c]["known"] for c in cats]
    missed   = [type_results[c]["n"] - type_results[c]["known"] - type_results[c]["zd"]
                for c in cats]
    colors   = ["#E53935" if type_results[c]["hidden"] else "#1565C0" for c in cats]
    y_pos    = np.arange(len(cats))
    ax1.barh(y_pos, kn_vals, 0.6, color="#2E7D32", alpha=0.8, label="Classified (known)")
    ax1.barh(y_pos, zd_vals, 0.6, left=kn_vals, color="#FB8C00", alpha=0.85,
             label="Zero-day queue")
    ax1.barh(y_pos, missed, 0.6, left=[k+z for k,z in zip(kn_vals,zd_vals)],
             color="#B0BEC5", alpha=0.6, label="Missed (Stage 1)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([
        ("* " + c if type_results[c]["hidden"] else "  " + c) for c in cats
    ], fontsize=8.5)
    ax1.set_xlabel("Flows flagged by Stage 1", fontsize=9)
    ax1.set_title("* = HIDDEN from Stage 2\nWhere do each type's flagged flows go?",
                  fontweight="bold", fontsize=10, loc="left")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.25, linestyle="--", axis="x")

    # Panel B: Zero-day queue composition donut
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(BG)
    zd_series   = pd.Series(zd_cats).value_counts()
    pie_labels  = list(zd_series.index)
    pie_vals    = list(zd_series.values)
    pie_colors  = ["#E53935" if c in hidden_types else
                   ("#2E7D32" if c in known_types else "#90A4AE")
                   for c in pie_labels]
    wedges, texts, autotexts = ax2.pie(
        pie_vals, labels=pie_labels, colors=pie_colors,
        autopct=lambda p: f"{p:.0f}%" if p > 3 else "",
        pctdistance=0.78, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white"),
    )
    for t in texts:
        t.set_fontsize(8)
    for at in autotexts:
        at.set_fontsize(7.5)
    ax2.set_title("Zero-Day Queue Composition\nRed = hidden attack types",
                  fontweight="bold", fontsize=10, loc="left")
    zd_real = (zd_y==1).sum(); zd_total = len(zd_y)
    ax2.text(0, -0.08,
             f"Queue precision: {zd_real/max(zd_total,1)*100:.0f}%\n({zd_real:,} real attacks / {zd_total:,})",
             ha="center", fontsize=9, fontweight="bold", transform=ax2.transAxes)

    # Panel C: Hidden-type detection rate per stage
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(BG)
    hidden_cats = sorted(hidden_types)
    zd_cats_arr = np.array(zd_cats)
    # Total in test set
    all_cats_full_te = np.concatenate([
        np.full(v, k) for k, v in pd.Series(zd_cats).value_counts().items()
    ])
    # Get detection rates
    x      = np.arange(len(hidden_cats))
    w      = 0.3
    s1_rates, zd_rates = [], []
    for c in hidden_cats:
        tot = type_results[c]["n"] if c in type_results else 0
        zd  = type_results[c]["zd"] if c in type_results else 0
        s1_rates.append(tot)
        zd_rates.append(zd)

    b1 = ax3.bar(x - w/2, s1_rates, w, color="#E53935", alpha=0.8, label="Stage 1 flagged")
    b2 = ax3.bar(x + w/2, zd_rates, w, color="#FB8C00", alpha=0.85, label="Reached ZD queue")
    for bar, v in zip(b1, s1_rates):
        ax3.text(bar.get_x()+bar.get_width()/2, v+0.5, str(v),
                 ha="center", fontsize=8.5, fontweight="bold")
    for bar, v in zip(b2, zd_rates):
        ax3.text(bar.get_x()+bar.get_width()/2, v+0.5, str(v),
                 ha="center", fontsize=8.5, fontweight="bold")
    ax3.set_xticks(x); ax3.set_xticklabels(hidden_cats, fontsize=9)
    ax3.set_ylabel("Flows in test set", fontsize=9)
    ax3.set_title("Hidden Types: How Many Reach the ZD Queue?\n(Of flows flagged by Stage 1)",
                  fontweight="bold", fontsize=10, loc="left")
    ax3.legend(fontsize=8.5)
    ax3.grid(True, alpha=0.25, linestyle="--", axis="y")

    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n[+] Figure saved -> {OUT_PNG}")


if __name__ == "__main__":
    main()

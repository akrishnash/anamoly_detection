#!/usr/bin/env python3
"""
CTU-13: Attack vs Normal Dataset Comparison  (20 K rows each)
=============================================================
Trains Isolation Forest on the combined 40 K-flow dataset, then
produces a two-page comparison graph:

  Page 1 — graphs/ctu13_comparison.png
      Panel A : What is Normal Traffic?   (feature bar chart)
      Panel B : What is Attack Traffic?   (feature bar chart)
      Panel C : IF Score Distribution KDE (attack vs normal)
      Panel D : Top-5 Feature Box Plots
      Panel E : ROC Curve
      Panel F : Precision-Recall Curve
      Panel G : Contamination Threshold Sweep
      Panel H : Confusion Matrix

Output: graphs/ctu13_comparison.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)

warnings.filterwarnings("ignore")

ATTACK_CSV    = "sample_logs/CTU13_Attack_Traffic.csv"
NORMAL_CSV    = "sample_logs/CTU13_Normal_Traffic.csv"
N_SAMPLE      = 20_000
CONTAMINATION = 0.40
OUT_PNG       = "graphs/ctu13_comparison.png"

FEATURE_COLS = [
    "Flow Byts/s", "Flow Pkts/s",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Flow Duration", "Pkt Len Mean",
    "SYN Flag Cnt", "RST Flag Cnt", "FIN Flag Cnt", "ACK Flag Cnt",
]
FEATURE_LABELS = [
    "Byte Rate\n(B/s)", "Pkt Rate\n(pkts/s)",
    "Fwd Bytes", "Bwd Bytes",
    "Flow Duration\n(µs)", "Pkt Len Mean",
    "SYN Flags", "RST Flags", "FIN Flags", "ACK Flags",
]

CATK = "#E53935"
CNRM = "#1565C0"
BG   = "#EAEFF7"


# ── 1. Load ───────────────────────────────────────────────────────────────────

def load():
    print("[*] Loading datasets ...")
    attack = pd.read_csv(ATTACK_CSV)
    normal = pd.read_csv(NORMAL_CSV)
    print(f"    Full Attack dataset : {len(attack):>7,} rows  |  {attack.shape[1]} features")
    print(f"    Full Normal dataset : {len(normal):>7,} rows  |  {normal.shape[1]} features")

    atk_s = attack.sample(n=min(N_SAMPLE, len(attack)), random_state=42).copy()
    nrm_s = normal.sample(n=min(N_SAMPLE, len(normal)), random_state=42).copy()
    atk_s["true_label"] = 1
    nrm_s["true_label"] = 0

    combined = pd.concat([atk_s, nrm_s], ignore_index=True).sample(
        frac=1, random_state=42).reset_index(drop=True)
    print(f"    Sampled combined    : {len(combined):,} rows  "
          f"({len(atk_s):,} attack + {len(nrm_s):,} normal)")
    return combined, atk_s, nrm_s


def _clean(df, cols):
    return df[cols].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)


# ── 2. Train Isolation Forest ─────────────────────────────────────────────────

def train_if(df):
    print("[*] Training Isolation Forest on 40 K flows ...")
    X  = _clean(df, FEATURE_COLS).values
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    model = IsolationForest(n_estimators=200, contamination=CONTAMINATION,
                            random_state=42, n_jobs=-1)
    df = df.copy()
    df["if_raw"]  = model.fit_predict(Xs)
    df["score"]   = model.score_samples(Xs)
    df["flagged"] = (df["if_raw"] == -1).astype(int)
    return df, model, sc


# ── 3. Print metrics ──────────────────────────────────────────────────────────

def print_metrics(df):
    y_true  = df["true_label"].values
    y_pred  = df["flagged"].values
    scores  = df["score"].values

    print("\n" + "=" * 60)
    print("  RESULTS  (20 K attack + 20 K normal = 40 K flows)")
    print("=" * 60)
    print(classification_report(y_true, y_pred,
                                target_names=["Normal", "Attack"], digits=3))
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    fpr, tpr, _ = roc_curve(y_true, -scores)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(y_true, -scores)
    print(f"\n  ROC-AUC        : {roc_auc:.4f}")
    print(f"  Avg Precision  : {ap:.4f}")

    atk_sc  = df[df["true_label"] == 1]["score"]
    nrm_sc  = df[df["true_label"] == 0]["score"]
    thresh  = df[df["flagged"] == 1]["score"].max()
    print(f"\n  IF score (lower = more anomalous)   threshold ~= {thresh:.4f}")
    print(f"    Attack : mean={atk_sc.mean():.4f}  median={atk_sc.median():.4f}  std={atk_sc.std():.4f}")
    print(f"    Normal : mean={nrm_sc.mean():.4f}  median={nrm_sc.median():.4f}  std={nrm_sc.std():.4f}")
    print("=" * 60 + "\n")
    return fpr, tpr, roc_auc, ap, cm


# ── 4. Threshold sweep ────────────────────────────────────────────────────────

def threshold_sweep(df):
    scores = df["score"].values
    y_true = df["true_label"].values
    contam = np.arange(0.05, 0.76, 0.05)
    prec, rec, f1s = [], [], []
    for c in contam:
        n    = int(len(scores) * c)
        thr  = np.sort(scores)[n]
        pred = (scores <= thr).astype(int)
        cm   = confusion_matrix(y_true, pred)
        tp, fp, fn = cm[1,1], cm[0,1], cm[1,0]
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f = 2*p*r / (p+r+1e-9)
        prec.append(p); rec.append(r); f1s.append(f)
    return contam, prec, rec, f1s


# ── 5. Plot ───────────────────────────────────────────────────────────────────

def plot(df, atk_raw, nrm_raw, fpr, tpr, roc_auc, ap, cm,
         contam, prec, rec, f1s):

    os.makedirs("graphs", exist_ok=True)

    atk_clean = _clean(atk_raw, FEATURE_COLS)
    nrm_clean = _clean(nrm_raw, FEATURE_COLS)

    # Cohen's d per feature
    disc_cols = [
        ("SYN Flag Cnt",    "SYN Flags"),
        ("Flow Pkts/s",     "Pkt Rate (pkts/s)"),
        ("Fwd Pkts/s",      "Fwd Pkts/s"),
        ("Bwd Pkts/s",      "Bwd Pkts/s"),
        ("FIN Flag Cnt",    "FIN Flags"),
        ("Pkt Len Std",     "Pkt Len Std"),
        ("Pkt Len Mean",    "Pkt Len Mean"),
        ("ACK Flag Cnt",    "ACK Flags"),
        ("RST Flag Cnt",    "RST Flags"),
        ("Flow Byts/s",     "Byte Rate"),
        ("TotLen Fwd Pkts", "Fwd Bytes"),
        ("TotLen Bwd Pkts", "Bwd Bytes"),
    ]
    cohen_ds, cohen_labels, cohen_colors = [], [], []
    for col, label in disc_cols:
        av = atk_clean[col].values if col in atk_clean.columns else np.zeros(len(atk_clean))
        nv = nrm_clean[col].values if col in nrm_clean.columns else np.zeros(len(nrm_clean))
        pooled = np.sqrt((av.std()**2 + nv.std()**2) / 2) + 1e-9
        d = (av.mean() - nv.mean()) / pooled
        cohen_ds.append(d)
        cohen_labels.append(label)
        cohen_colors.append(CATK if d > 0 else CNRM)

    fig = plt.figure(figsize=(22, 28), facecolor="#F4F6FA")
    fig.suptitle(
        "CTU-13 Botnet Dataset  —  Attack vs Normal Traffic  (20,000 flows each)\n"
        "Isolation Forest Anomaly Detection: Full Comparison",
        fontsize=14, fontweight="bold", y=0.997,
    )

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.40,
                           top=0.968, bottom=0.04, left=0.06, right=0.97)

    # ── Row 0, Col 0-1: Feature Discrimination Power (Cohen's d) ─────────────
    ax_disc = fig.add_subplot(gs[0, :2])
    ax_disc.set_facecolor(BG)
    y_pos = np.arange(len(cohen_labels))
    bars_d = ax_disc.barh(y_pos, cohen_ds, color=cohen_colors, alpha=0.82,
                          edgecolor="white", linewidth=0.8)
    ax_disc.axvline(0, color="black", linewidth=1.2)
    ax_disc.set_yticks(y_pos)
    ax_disc.set_yticklabels(cohen_labels, fontsize=8)
    ax_disc.set_xlabel("Cohen's d  (attack mean - normal mean) / pooled std", fontsize=9)
    ax_disc.set_title(
        "What Actually Separates Attack from Normal?\n"
        "Feature Discrimination Power  |  Red = attack higher, Blue = normal higher",
        fontweight="bold", fontsize=10, color="#37474F", loc="left",
    )
    ax_disc.grid(True, alpha=0.25, linestyle="--", axis="x")
    for bar, d in zip(bars_d, cohen_ds):
        xpos = bar.get_width() + 0.01 if d >= 0 else bar.get_width() - 0.01
        ha   = "left" if d >= 0 else "right"
        ax_disc.text(xpos, bar.get_y() + bar.get_height()/2,
                     f"{d:+.2f}", va="center", ha=ha, fontsize=7.5, fontweight="bold")
    ax_disc.text(0.98, 0.04,
        "SYN flags (d=+0.86): attack opens many\n"
        "  connections = port scanning / C&C setup\n"
        "Pkt/s (d=+0.63): attack is bimodal —\n"
        "  near-zero C&C beacons + DDoS bursts\n"
        "Pkt Len (d=-0.19): normal has larger\n"
        "  packets (HTTP content, file data)\n"
        "Bwd Bytes (d=-0.02): attack C&C gets\n"
        "  tiny/empty replies (stealthy by design)",
        transform=ax_disc.transAxes, fontsize=7.5, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#90A4AE", alpha=0.92))

    # ── Row 0, Col 2-3: Overlapping histograms for top 2 discriminators ───────
    inner_top = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 2:],
                                                 hspace=0.55, wspace=0.45)
    hist_feats = [
        ("SYN Flag Cnt",  "SYN Flag Count",
         "Normal: mostly 0 SYN (established flows)\nAttack: frequent SYN = port scans / C&C setup"),
        ("Flow Pkts/s",   "Packet Rate  (pkts/s)",
         "Normal: broad, moderate rate (real traffic)\nAttack: BIMODAL — near-zero C&C + spike floods"),
        ("Flow Byts/s",   "Byte Rate  (bytes/s)",
         "Normal: active data transfer (higher bytes)\nAttack: near-zero (C&C beacon) or extreme (DDoS)"),
        ("Pkt Len Mean",  "Mean Packet Length  (bytes)",
         "Normal: larger packets = HTTP/file content\nAttack: tiny packets = empty C&C commands"),
    ]
    for idx, (col, label, note) in enumerate(hist_feats):
        ax_h = fig.add_subplot(inner_top[idx // 2, idx % 2])
        ax_h.set_facecolor(BG)
        av = atk_clean[col].clip(upper=atk_clean[col].quantile(0.995)).values
        nv = nrm_clean[col].clip(upper=nrm_clean[col].quantile(0.995)).values
        all_vals = np.concatenate([av, nv])
        bins = np.linspace(0, np.percentile(all_vals, 99), 50)
        ax_h.hist(nv, bins=bins, color=CNRM, alpha=0.55, label="Normal", density=True)
        ax_h.hist(av, bins=bins, color=CATK, alpha=0.55, label="Attack", density=True)
        ax_h.set_title(label, fontsize=8, fontweight="bold")
        ax_h.set_xlabel("Value", fontsize=7)
        ax_h.set_ylabel("Density", fontsize=7)
        ax_h.legend(fontsize=7, loc="upper right")
        ax_h.grid(True, alpha=0.2, linestyle="--")
        ax_h.text(0.5, 0.97, note, transform=ax_h.transAxes, fontsize=6.5,
                  va="top", ha="center",
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#B0BEC5", alpha=0.88))

    # ── Row 1, Col 0-1: IF Score KDE ──────────────────────────────────────────
    ax_kde = fig.add_subplot(gs[1, :2])
    ax_kde.set_facecolor(BG)
    atk_scores = df[df["true_label"] == 1]["score"].values
    nrm_scores = df[df["true_label"] == 0]["score"].values
    threshold  = df[df["flagged"] == 1]["score"].max()
    for sc, color, lbl in [(nrm_scores, CNRM, "Normal"), (atk_scores, CATK, "Attack")]:
        xs  = np.linspace(sc.min() - 0.01, sc.max() + 0.01, 500)
        kde = gaussian_kde(sc, bw_method=0.12)
        ax_kde.fill_between(xs, kde(xs), alpha=0.30, color=color)
        ax_kde.plot(xs, kde(xs), color=color, linewidth=2.2, label=f"{lbl}  (n=20,000)")
    ax_kde.axvline(threshold, color="black", linestyle="--", linewidth=1.8,
                   label=f"Decision boundary ({threshold:.3f})")
    ax_kde.fill_betweenx([0, ax_kde.get_ylim()[1] if ax_kde.get_ylim()[1] > 0 else 30],
                         ax_kde.get_xlim()[0] if ax_kde.get_xlim()[0] < threshold else threshold,
                         threshold, alpha=0.06, color=CATK)
    ax_kde.set_title(
        "Isolation Forest Score Distribution: Attack vs Normal\n"
        "Flows left of boundary are flagged as anomalies",
        fontweight="bold", fontsize=10, loc="left",
    )
    ax_kde.set_xlabel("IF Anomaly Score  (lower = more anomalous)", fontsize=9)
    ax_kde.set_ylabel("Density", fontsize=9)
    ax_kde.legend(fontsize=9)
    ax_kde.grid(True, alpha=0.25, linestyle="--")

    # ── Row 1, Col 2-3: Side-by-side feature box plots (top 4 features) ───────
    top4 = [
        ("Flow Byts/s",  "Byte Rate (B/s)"),
        ("Flow Pkts/s",  "Pkt Rate (pkts/s)"),
        ("RST Flag Cnt", "RST Flags"),
        ("SYN Flag Cnt", "SYN Flags"),
    ]
    inner = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1, 2:], wspace=0.45)
    for idx, (col, label) in enumerate(top4):
        ax_b = fig.add_subplot(inner[idx])
        ax_b.set_facecolor(BG)
        a_v = np.log1p(atk_clean[col].values)
        n_v = np.log1p(nrm_clean[col].values)
        bp  = ax_b.boxplot(
            [n_v, a_v], patch_artist=True, widths=0.55,
            medianprops=dict(color="black", linewidth=2.2),
            flierprops=dict(marker=".", markersize=1.5, alpha=0.25),
            whiskerprops=dict(linewidth=1.3),
            capprops=dict(linewidth=1.3),
        )
        bp["boxes"][0].set_facecolor(CNRM + "99")
        bp["boxes"][1].set_facecolor(CATK + "99")
        ax_b.set_xticks([1, 2])
        ax_b.set_xticklabels(["Normal", "Attack"], fontsize=8)
        ax_b.set_title(f"log(1+{label})", fontsize=8, fontweight="bold")
        ax_b.set_ylabel("log1p", fontsize=7)
        ax_b.grid(True, alpha=0.25, linestyle="--", axis="y")

    # ── Row 2, Col 0-1: ROC curve ─────────────────────────────────────────────
    ax_roc = fig.add_subplot(gs[2, :2])
    ax_roc.set_facecolor(BG)
    ax_roc.plot(fpr, tpr, color=CATK, linewidth=2.2,
                label=f"IF Detector  (AUC = {roc_auc:.3f})")
    ax_roc.plot([0,1],[0,1],"k--", linewidth=1, alpha=0.4, label="Random (AUC = 0.500)")
    ax_roc.fill_between(fpr, tpr, alpha=0.13, color=CATK)
    # Mark operating point
    y_pred  = df["flagged"].values
    y_true  = df["true_label"].values
    op_fpr  = (y_pred[(y_true==0)] == 1).sum() / (y_true==0).sum()
    op_tpr  = (y_pred[(y_true==1)] == 1).sum() / (y_true==1).sum()
    ax_roc.scatter([op_fpr],[op_tpr], color="black", zorder=5, s=60,
                   label=f"Operating point  (FPR={op_fpr:.2f}, TPR={op_tpr:.2f})")
    ax_roc.set_title("ROC Curve  —  Attack Detection Performance", fontweight="bold", fontsize=10, loc="left")
    ax_roc.set_xlabel("False Positive Rate  (normal flagged as attack)", fontsize=9)
    ax_roc.set_ylabel("True Positive Rate  (attack correctly caught)", fontsize=9)
    ax_roc.legend(fontsize=8)
    ax_roc.grid(True, alpha=0.25, linestyle="--")

    # ── Row 2, Col 2-3: Precision-Recall ──────────────────────────────────────
    ax_pr = fig.add_subplot(gs[2, 2:])
    ax_pr.set_facecolor(BG)
    pr_p, pr_r, _ = precision_recall_curve(y_true, -df["score"].values)
    ax_pr.plot(pr_r, pr_p, color="#8E24AA", linewidth=2.2,
               label=f"PR curve  (AP = {ap:.3f})")
    ax_pr.axhline(y_true.mean(), color="gray", linestyle="--", linewidth=1.2,
                  label=f"No-skill baseline ({y_true.mean():.2f})")
    ax_pr.fill_between(pr_r, pr_p, alpha=0.13, color="#8E24AA")
    ax_pr.scatter(
        [(y_pred[y_true==1]==1).sum()/(y_true==1).sum()],
        [(y_pred[y_true==1]==1).sum()/(y_pred==1).sum()],
        color="black", zorder=5, s=60,
        label=f"Operating point",
    )
    ax_pr.set_title("Precision-Recall Curve", fontweight="bold", fontsize=10, loc="left")
    ax_pr.set_xlabel("Recall  (fraction of attacks caught)", fontsize=9)
    ax_pr.set_ylabel("Precision  (of flagged flows, how many are real attacks)", fontsize=9)
    ax_pr.legend(fontsize=8)
    ax_pr.grid(True, alpha=0.25, linestyle="--")

    # ── Row 3, Col 0-2: Threshold sweep ───────────────────────────────────────
    ax_sw = fig.add_subplot(gs[3, :3])
    ax_sw.set_facecolor(BG)
    pct = contam * 100
    ax_sw.plot(pct, prec, color=CNRM,     linewidth=2, marker="o", ms=4, label="Precision")
    ax_sw.plot(pct, rec,  color=CATK,     linewidth=2, marker="s", ms=4, label="Recall")
    ax_sw.plot(pct, f1s,  color="#FB8C00", linewidth=2, marker="^", ms=4, label="F1 Score")
    ax_sw.axvline(CONTAMINATION*100, color="black", linestyle=":", linewidth=1.8,
                  label=f"Current setting ({CONTAMINATION:.0%})")
    # Shade optimal F1 region
    best_f1_idx = int(np.argmax(f1s))
    ax_sw.axvspan(pct[max(0,best_f1_idx-1)], pct[min(len(pct)-1,best_f1_idx+1)],
                  alpha=0.12, color="#FB8C00", label=f"Best F1 ~= {max(f1s):.3f}")
    ax_sw.set_title(
        "Precision / Recall / F1  vs  Contamination Threshold\n"
        "Shows how changing the fraction of flagged flows shifts the trade-off",
        fontweight="bold", fontsize=10, loc="left",
    )
    ax_sw.set_xlabel("Contamination %  (% of total flows flagged as attacks)", fontsize=9)
    ax_sw.set_ylabel("Score", fontsize=9)
    ax_sw.set_ylim(0, 1.08)
    ax_sw.legend(fontsize=8, ncol=3)
    ax_sw.grid(True, alpha=0.25, linestyle="--")

    # ── Row 3, Col 3: Confusion matrix ────────────────────────────────────────
    ax_cm = fig.add_subplot(gs[3, 3])
    ax_cm.set_facecolor(BG)
    im = ax_cm.imshow(cm, cmap="Blues", aspect="auto")
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i,j] > cm.max()/2 else "black"
            ax_cm.text(j, i, f"{labels[i][j]}\n{cm[i,j]:,}",
                       ha="center", va="center", fontsize=11,
                       fontweight="bold", color=color)
    ax_cm.set_xticks([0,1]); ax_cm.set_yticks([0,1])
    ax_cm.set_xticklabels(["Pred Normal", "Pred Attack"], fontsize=9)
    ax_cm.set_yticklabels(["True Normal", "True Attack"], fontsize=9)
    ax_cm.set_title("Confusion Matrix", fontweight="bold", fontsize=10)
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    # Legend patches for dataset colors
    fig.legend(
        handles=[
            mpatches.Patch(color=CNRM, label="Normal Traffic (benign campus flows)"),
            mpatches.Patch(color=CATK, label="Attack Traffic (CTU-13 botnet flows)"),
        ],
        loc="lower center", ncol=2, fontsize=10, framealpha=0.95,
        bbox_to_anchor=(0.5, 0.003),
        title="Dataset Legend", title_fontsize=10,
    )

    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[+] Graph saved -> {OUT_PNG}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df, atk_raw, nrm_raw = load()
    df, model, scaler    = train_if(df)
    fpr, tpr, roc_auc, ap, cm = print_metrics(df)
    contam, prec, rec, f1s    = threshold_sweep(df)
    plot(df, atk_raw, nrm_raw,
         fpr, tpr, roc_auc, ap, cm,
         contam, prec, rec, f1s)


if __name__ == "__main__":
    main()

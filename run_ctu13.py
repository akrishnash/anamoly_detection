#!/usr/bin/env python3
"""
CTU-13 Botnet Dataset — Anomaly Detection with Isolation Forest
================================================================
Downloads used:
  sample_logs/CTU13_Attack_Traffic.csv   (botnet / attack flows)
  sample_logs/CTU13_Normal_Traffic.csv   (benign flows)

Both are CICFlowMeter-format NetFlow CSVs from:
  https://github.com/imfaisalmalik/CTU13-CSV-Dataset

The script:
  1. Loads & samples both CSVs
  2. Synthesises timestamps so flows span a single day
  3. Runs Isolation Forest on 10 flow-level features
  4. Labels flagged flows by heuristic rules
  5. Prints a report and saves a 4-panel graph
  6. Shows precision / recall vs the ground-truth Label column
"""

import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# ── Config ───────────────────────────────────────────────────────────────────
ATTACK_CSV  = "sample_logs/CTU13_Attack_Traffic.csv"
NORMAL_CSV  = "sample_logs/CTU13_Normal_Traffic.csv"
N_SAMPLE    = 6000          # rows to sample from each class
CONTAMINATION = 0.40        # ~40 % of combined set is real attacks
BASE_DATE   = datetime(2024, 1, 15)
OUT_PNG     = "graphs/anomaly_report_ctu13.png"

FEATURE_COLS = [
    "flow_byts_s", "flow_pkts_s",
    "fwd_bytes", "bwd_bytes", "total_pkts",
    "syn_flag", "rst_flag", "fin_flag",
    "flow_duration_s", "pkt_len_mean",
]

ANOMALY_COLORS = {
    "DDoS / Flood":       "#E53935",
    "Port Scan":          "#FB8C00",
    "Brute Force":        "#8E24AA",
    "Botnet C&C":         "#1565C0",
    "Data Exfiltration":  "#4E342E",
    "Unknown":            "#757575",
}
SEV_ORDER  = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
SEV_PREFIX = {"CRITICAL": "[!!!]", "HIGH": "[!! ]", "MEDIUM": "[!  ]", "LOW": "[.  ]"}


# ── 1. Load & prepare ─────────────────────────────────────────────────────────

def load_and_merge() -> pd.DataFrame:
    for path in (ATTACK_CSV, NORMAL_CSV):
        if not os.path.exists(path):
            sys.exit(f"[-] File not found: {path}\n"
                     "    Run:  curl -L <URL> -o " + path)

    print("[*] Loading CTU-13 CSVs ...")
    attack = pd.read_csv(ATTACK_CSV)
    normal = pd.read_csv(NORMAL_CSV)
    print(f"    Attack flows : {len(attack):>7,}")
    print(f"    Normal flows : {len(normal):>7,}")

    attack = attack.sample(n=min(N_SAMPLE, len(attack)), random_state=42)
    normal = normal.sample(n=min(N_SAMPLE, len(normal)), random_state=42)
    attack["true_label"] = 1
    normal["true_label"] = 0

    df = pd.concat([attack, normal], ignore_index=True).sample(
        frac=1, random_state=42).reset_index(drop=True)
    print(f"    Combined (sampled, shuffled): {len(df):,} flows")
    return df


def featurise(df: pd.DataFrame) -> pd.DataFrame:
    """Map CICFlowMeter columns to our feature names and synthesise timestamps."""
    out = pd.DataFrame()

    # Timestamps: spread flows evenly across 24 hours
    n = len(df)
    step = timedelta(days=1) / n
    out["timestamp"] = [BASE_DATE + i * step for i in range(n)]

    # Core features
    out["flow_byts_s"]     = pd.to_numeric(df["Flow Byts/s"],  errors="coerce").fillna(0).clip(0)
    out["flow_pkts_s"]     = pd.to_numeric(df["Flow Pkts/s"],  errors="coerce").fillna(0).clip(0)
    out["fwd_bytes"]       = pd.to_numeric(df["TotLen Fwd Pkts"], errors="coerce").fillna(0)
    out["bwd_bytes"]       = pd.to_numeric(df["TotLen Bwd Pkts"], errors="coerce").fillna(0)
    out["total_pkts"]      = (pd.to_numeric(df["Tot Fwd Pkts"], errors="coerce").fillna(0) +
                              pd.to_numeric(df["Tot Bwd Pkts"], errors="coerce").fillna(0))
    out["syn_flag"]        = pd.to_numeric(df["SYN Flag Cnt"],  errors="coerce").fillna(0)
    out["rst_flag"]        = pd.to_numeric(df["RST Flag Cnt"],  errors="coerce").fillna(0)
    out["fin_flag"]        = pd.to_numeric(df["FIN Flag Cnt"],  errors="coerce").fillna(0)
    out["flow_duration_s"] = pd.to_numeric(df["Flow Duration"], errors="coerce").fillna(0) / 1e6
    out["pkt_len_mean"]    = pd.to_numeric(df["Pkt Len Mean"],  errors="coerce").fillna(0)

    # Ground truth  (1 = attack, 0 = normal)
    out["true_label"] = df["true_label"].values
    return out


# ── 2. Isolation Forest ───────────────────────────────────────────────────────

def run_if(feat: pd.DataFrame, contamination: float):
    X = feat[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=42, n_jobs=-1)
    feat = feat.copy()
    feat["if_label"] = model.fit_predict(Xs)    # -1=anomaly, 1=normal
    feat["if_score"] = model.score_samples(Xs)
    return feat, model


# ── 3. Anomaly type classification ───────────────────────────────────────────

def _z(val, mean, std) -> float:
    return (val - mean) / max(abs(std), 1e-9)


def classify(row: pd.Series, stats: dict) -> tuple:
    z_byts = _z(row["flow_byts_s"],     stats["byts_mean"], stats["byts_std"])
    z_pkts = _z(row["flow_pkts_s"],     stats["pkts_mean"], stats["pkts_std"])
    z_fwd  = _z(row["fwd_bytes"],       stats["fwd_mean"],  stats["fwd_std"])
    z_rst  = _z(row["rst_flag"],        stats["rst_mean"],  stats["rst_std"])
    z_syn  = _z(row["syn_flag"],        stats["syn_mean"],  stats["syn_std"])
    z_dur  = _z(row["flow_duration_s"], stats["dur_mean"],  stats["dur_std"])

    if z_byts > 4 or z_pkts > 4:
        return "DDoS / Flood", "CRITICAL"
    if z_rst > 3 and z_syn > 1:
        return "Port Scan", "HIGH"
    if z_rst > 3 and row["total_pkts"] < stats["pkts_mean"]:
        return "Brute Force", "HIGH"
    if z_fwd > 3 and z_dur > 1:
        return "Data Exfiltration", "CRITICAL"
    if z_dur > 3 and z_byts < 0:
        return "Botnet C&C", "MEDIUM"
    return "Unknown", "LOW"


def label_anomalies(feat: pd.DataFrame) -> list:
    flagged = feat[feat["if_label"] == -1]
    normal  = feat[feat["if_label"] ==  1]
    if flagged.empty:
        return []

    stats = {
        "byts_mean": normal["flow_byts_s"].mean(),     "byts_std": normal["flow_byts_s"].std(),
        "pkts_mean": normal["flow_pkts_s"].mean(),     "pkts_std": normal["flow_pkts_s"].std(),
        "fwd_mean":  normal["fwd_bytes"].mean(),       "fwd_std":  normal["fwd_bytes"].std(),
        "rst_mean":  normal["rst_flag"].mean(),        "rst_std":  normal["rst_flag"].std(),
        "syn_mean":  normal["syn_flag"].mean(),        "syn_std":  normal["syn_flag"].std(),
        "dur_mean":  normal["flow_duration_s"].mean(), "dur_std":  normal["flow_duration_s"].std(),
    }

    results = []
    for _, row in flagged.iterrows():
        atype, sev = classify(row, stats)
        results.append({
            "timestamp":    row["timestamp"],
            "type":         atype,
            "severity":     sev,
            "if_score":     row["if_score"],
            "flow_byts_s":  row["flow_byts_s"],
            "total_pkts":   row["total_pkts"],
            "fwd_bytes":    row["fwd_bytes"],
            "syn_flag":     row["syn_flag"],
            "rst_flag":     row["rst_flag"],
            "true_label":   row["true_label"],
        })
    return sorted(results, key=lambda x: x["timestamp"])


# ── 4. Console report ─────────────────────────────────────────────────────────

def print_report(anomalies: list, feat: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("  CTU-13 BOTNET — ISOLATION FOREST ANOMALY REPORT")
    print("=" * 72)

    if not anomalies:
        print("  No anomalies detected.")
        print("=" * 72)
        return

    # Deduplicate: keep 1 per type per 30-min bucket
    seen: dict = {}
    deduped = []
    for a in anomalies:
        bucket = a["timestamp"].replace(minute=(a["timestamp"].minute // 30) * 30,
                                        second=0, microsecond=0)
        key = (a["type"], bucket)
        if key not in seen or a["if_score"] < seen[key]["if_score"]:
            seen[key] = a
    deduped = sorted(seen.values(), key=lambda x: x["timestamp"])

    for i, a in enumerate(
        sorted(deduped, key=lambda x: (SEV_ORDER.get(x["severity"], 9), x["if_score"])), 1
    ):
        prefix = SEV_PREFIX.get(a["severity"], "[   ]")
        gt = "Attack" if a["true_label"] == 1 else "Normal"
        print(f"\n  #{i:02d}  {prefix}  {a['severity']}")
        print(f"        Type         : {a['type']}")
        print(f"        Time         : {a['timestamp'].strftime('%Y-%m-%d  %H:%M:%S')}")
        print(f"        IF Score     : {a['if_score']:.4f}  (lower = more anomalous)")
        print(f"        Byte rate    : {a['flow_byts_s']:>12,.1f}  bytes/s")
        print(f"        Total pkts   : {int(a['total_pkts']):>12,}")
        print(f"        Fwd bytes    : {a['fwd_bytes']:>12,.0f}")
        print(f"        SYN / RST    : {int(a['syn_flag'])} / {int(a['rst_flag'])}")
        print(f"        Ground Truth : {gt}")

    # Summary counts by type
    counts: dict = defaultdict(int)
    for a in deduped:
        counts[a["type"]] += 1
    print("\n" + "-" * 72)
    print(f"  Anomaly episodes shown (deduplicated): {len(deduped)}")
    for t, c in sorted(counts.items()):
        print(f"    * {t:<24} {c}")

    # Precision / Recall vs ground truth labels
    print("\n" + "-" * 72)
    print("  MODEL PERFORMANCE  (IF prediction vs. CTU-13 ground-truth labels)")
    print()
    y_true = feat["true_label"].values        # 1=attack, 0=normal
    y_pred = (feat["if_label"] == -1).astype(int)  # -1→1 (anomaly=attack)
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(y_true, y_pred,
                                target_names=["Normal (0)", "Attack (1)"],
                                digits=3))
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion matrix:")
    print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    print("=" * 72 + "\n")


# ── 5. Visualisation ──────────────────────────────────────────────────────────

def plot(feat: pd.DataFrame, anomalies: list) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(17, 13), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1, 1.3]})
    fig.patch.set_facecolor("#F4F6FA")
    fig.suptitle("CTU-13 Botnet Traffic — Isolation Forest Anomaly Detection",
                 fontsize=14, fontweight="bold", y=0.995)

    # 5-min resampled series
    ts = feat.set_index("timestamp")
    byts   = ts["flow_byts_s"].resample("5min").mean().fillna(0)
    pkts   = ts["total_pkts"].resample("5min").sum().fillna(0)
    rst    = ts["rst_flag"].resample("5min").sum().fillna(0)
    score  = ts["if_score"].resample("5min").min()

    # Boundary = highest score among flagged flows
    flagged_scores = feat[feat["if_label"] == -1]["if_score"]
    boundary = flagged_scores.max() if not flagged_scores.empty else -0.05

    bg = "#EAEFF7"
    ax1, ax2, ax3, ax4 = axes

    ax1.set_facecolor(bg)
    ax1.fill_between(byts.index, byts.values, alpha=0.35, color="#1565C0")
    ax1.plot(byts.index, byts.values, color="#1565C0", linewidth=1.1)
    ax1.set_ylabel("Avg Byte Rate (B/s)", fontsize=9)
    ax1.set_title("Flow Byte Rate", fontsize=10, loc="left", pad=3)
    ax1.grid(True, alpha=0.25, linestyle="--")

    ax2.set_facecolor(bg)
    ax2.fill_between(pkts.index, pkts.values, alpha=0.35, color="#2E7D32")
    ax2.plot(pkts.index, pkts.values, color="#2E7D32", linewidth=1.1)
    ax2.set_ylabel("Total Packets / 5min", fontsize=9)
    ax2.set_title("Packet Volume", fontsize=10, loc="left", pad=3)
    ax2.grid(True, alpha=0.25, linestyle="--")

    ax3.set_facecolor(bg)
    ax3.bar(rst.index, rst.values, width=pd.Timedelta("4min"),
            color="#C62828", alpha=0.75)
    ax3.set_ylabel("RST Flags / 5min", fontsize=9)
    ax3.set_title("RST Flag Count  (failed / rejected connections)", fontsize=10, loc="left", pad=3)
    ax3.grid(True, alpha=0.25, linestyle="--")

    ax4.set_facecolor(bg)
    ax4.plot(score.index, score.values, color="#37474F", linewidth=1.2)
    ax4.fill_between(score.index, score.values, score.min(),
                     alpha=0.18, color="#37474F")
    ax4.axhline(boundary, color="#E53935", linestyle=":", linewidth=1.4,
                alpha=0.85, label=f"Decision boundary ({boundary:.3f})")
    ax4.set_ylabel("IF Anomaly Score", fontsize=9)
    ax4.set_title("Isolation Forest Score  (lower = more anomalous)",
                  fontsize=10, loc="left", pad=3)
    ax4.grid(True, alpha=0.25, linestyle="--")
    ax4.legend(fontsize=8, loc="upper right")

    # Anomaly overlays (deduplicated by type & 30-min bucket)
    seen: dict = {}
    for a in anomalies:
        b = a["timestamp"].replace(minute=(a["timestamp"].minute // 30) * 30,
                                   second=0, microsecond=0)
        key = (a["type"], b)
        if key not in seen or a["if_score"] < seen[key]["if_score"]:
            seen[key] = a
    deduped = sorted(seen.values(), key=lambda x: x["timestamp"])

    legend_handles: dict = {}
    label_y: dict = {}

    for a in deduped:
        color = ANOMALY_COLORS.get(a["type"], "#757575")
        t     = a["timestamp"]
        t_end = t + pd.Timedelta("5min")
        for ax in axes:
            ax.axvspan(t, t_end, alpha=0.12, color=color, zorder=0)
            ax.axvline(t, color=color, linestyle="--", linewidth=1.5, alpha=0.8, zorder=1)

        ymax  = float(byts.max()) or 1.0
        frac  = label_y.get(a["type"], 0.92)
        label_y[a["type"]] = max(frac - 0.16, 0.45)
        short = a["type"].replace(" ", "\n").replace("/", "/\n")
        ax1.annotate(short, xy=(t, ymax * frac),
                     fontsize=7, color=color, ha="center", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.25", fc="white",
                               ec=color, alpha=0.92, linewidth=1), zorder=5)
        if a["type"] not in legend_handles:
            legend_handles[a["type"]] = mpatches.Patch(color=color, label=a["type"])

    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax4.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=40, ha="right", fontsize=8)
    ax4.set_xlabel("Synthetic Time (flows spread across 24h)", fontsize=9)

    if legend_handles:
        fig.legend(handles=list(legend_handles.values()),
                   loc="lower center", ncol=min(len(legend_handles), 5),
                   bbox_to_anchor=(0.5, 0.002), fontsize=9, framealpha=0.92,
                   title="Detected Anomaly Types", title_fontsize=9)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.tight_layout(rect=[0, 0.055, 1, 0.975])
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[+] Graph saved -> {OUT_PNG}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  CTU-13 Botnet — Isolation Forest  (real-world data)")
    print("=" * 64)
    print(f"  Sample size   : {N_SAMPLE:,} flows per class")
    print(f"  Contamination : {CONTAMINATION:.0%}  (matching true attack ratio)")
    print(f"  Features      : {len(FEATURE_COLS)}")
    print("=" * 64)

    raw  = load_and_merge()
    feat = featurise(raw)

    print("[*] Running Isolation Forest ...")
    feat, model = run_if(feat, CONTAMINATION)
    n_flagged = (feat["if_label"] == -1).sum()
    print(f"    {n_flagged} / {len(feat)} flows flagged as anomalous")

    print("[*] Classifying anomaly types ...")
    anomalies = label_anomalies(feat)

    print_report(anomalies, feat)
    plot(feat, anomalies)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
How Isolation Forest Works — Step-by-Step Walkthrough
======================================================
Uses a tiny 13-point dataset (10 normal + 3 anomalies) so every
tree, every split, and every decision path is visible.

Run:  python explain_isolation_forest.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from sklearn.ensemble import IsolationForest
from sklearn.tree._tree import TREE_LEAF

# ── 1. Tiny, hand-crafted dataset ─────────────────────────────────────────────
#
# Three features (same as anomaly_detector_v2.py):
#   n_dst_ports  — how many unique destination ports in this window
#   n_bytes_kb   — total bytes transferred (KB)
#   failed_ratio — fraction of failed/rejected connections
#
# Normal traffic stays in a tight cluster.
# Each anomaly breaks ONE feature far outside that cluster.

FEATURE_NAMES = ["n_dst_ports", "n_bytes_kb", "failed_ratio"]

normal_data = np.array([
    # ports  bytes   fail
    [  2,    150,   0.01],
    [  3,    200,   0.02],
    [  1,    180,   0.00],
    [  4,    120,   0.03],
    [  2,    170,   0.01],
    [  3,    160,   0.02],
    [  2,    190,   0.00],
    [  1,    140,   0.01],
    [  3,    210,   0.02],
    [  2,    155,   0.01],
], dtype=float)

anomaly_data = np.array([
    # ports  bytes   fail
    [847,    190,   0.02],   # Port Scan   — n_dst_ports explodes
    [  3,   5000,   0.01],   # Exfiltration— n_bytes_kb  explodes
    [  2,    165,   0.97],   # Brute Force — failed_ratio explodes
], dtype=float)

X      = np.vstack([normal_data, anomaly_data])
labels = (["Normal"] * 10) + ["Port Scan", "Exfiltration", "Brute Force"]
colors = (["#2196F3"] * 10) + ["#FB8C00", "#4E342E", "#8E24AA"]

# ── 2. Fit a very small Isolation Forest so we can inspect it ─────────────────
#
# n_estimators=5  → only 5 trees  (normally 100-200)
# max_samples=13  → each tree sees all 13 points (dataset is tiny)
# contamination   → we expect ~3/13 ≈ 23% anomalies

model = IsolationForest(
    n_estimators=5,
    max_samples=13,
    contamination=3/13,
    random_state=7,
)
model.fit(X)

scores     = model.score_samples(X)   # lower = more anomalous
predictions = model.predict(X)        # -1 = anomaly, +1 = normal

# ── 3. Helper: trace a sample's path through one tree ─────────────────────────

def trace_path(estimator, x):
    """
    Walk x down a single isolation tree.
    Returns list of split dicts + the final leaf depth.
    """
    tree   = estimator.tree_
    node   = 0
    splits = []

    while tree.children_left[node] != TREE_LEAF:
        feat   = tree.feature[node]
        thresh = tree.threshold[node]
        val    = x[feat]

        splits.append({
            "node":      node,
            "feature":   FEATURE_NAMES[feat],
            "threshold": round(thresh, 3),
            "value":     round(val, 3),
            "went":      "LEFT  (val <= thresh)" if val <= thresh else "RIGHT (val > thresh)",
        })

        node = tree.children_left[node] if val <= thresh else tree.children_right[node]

    return splits, len(splits)   # splits list, path length (= depth of leaf)


# ── 4. Print: path lengths per point ─────────────────────────────────────────

print("\n" + "=" * 70)
print("  ISOLATION FOREST - EDUCATIONAL WALKTHROUGH")
print("=" * 70)

print("\n-- DATASET --------------------------------------------------------------")
df = pd.DataFrame(X, columns=FEATURE_NAMES)
df["label"]      = labels
df["IF_score"]   = scores.round(4)
df["prediction"] = ["ANOMALY" if p == -1 else "normal" for p in predictions]
print(df.to_string(index=True))

# -- 5. For each tree, print feature usage (how often each feature was split) --

print("\n\n-- FEATURE USAGE PER TREE -----------------------------------------------")
print(f"  (Which feature did the tree choose to split on, and how many times?)\n")

all_feature_counts = np.zeros((5, 3), dtype=int)

for t_idx, est in enumerate(model.estimators_):
    tree = est.tree_
    counts = np.zeros(3, dtype=int)
    for node in range(tree.node_count):
        if tree.children_left[node] != TREE_LEAF:   # internal node
            counts[tree.feature[node]] += 1
    all_feature_counts[t_idx] = counts

    bar = lambda n, mx: "#" * int(n / max(mx, 1) * 20)
    print(f"  Tree {t_idx+1}:")
    for f, name in enumerate(FEATURE_NAMES):
        print(f"    {name:<15}  {counts[f]:>3} splits  {bar(counts[f], counts.max())}")
    print()

total = all_feature_counts.sum(axis=0)
print(f"  TOTAL across all 5 trees:")
for f, name in enumerate(FEATURE_NAMES):
    print(f"    {name:<15}  {total[f]:>3} splits")

# ── 6. Deep-dive: trace ONE normal + all THREE anomalies through Tree 1 ────────

print("\n\n-- DECISION PATH INSIDE TREE 1 ------------------------------------------")
print("  KEY INSIGHT: anomalies are isolated in far fewer splits.\n")

showcase = [0, 10, 11, 12]   # indices: 1 normal + 3 anomalies

for idx in showcase:
    x      = X[idx]
    lbl    = labels[idx]
    splits, depth = trace_path(model.estimators_[0], x)

    print(f"  {'-'*60}")
    print(f"  Point #{idx:>2}  [{lbl}]")
    print(f"  Values  ->  ports={x[0]:.0f}  bytes={x[1]:.0f} KB  fail={x[2]:.2f}")
    print(f"  Path length (depth to isolation) = {depth}")
    print()
    for step, s in enumerate(splits, 1):
        print(f"    Step {step}: split on  {s['feature']:<15}  "
              f"(tree threshold = {s['threshold']})  "
              f"|  sample value = {s['value']}")
        print(f"             -> went {s['went']}")
    print()

# -- 7. Path-length summary ---------------------------------------------------

print("\n-- AVERAGE PATH LENGTH (all 5 trees) -----------------------------------")
print("  Shorter path -> isolated faster -> more anomalous\n")

path_lengths = []
for idx in range(len(X)):
    depths = [trace_path(est, X[idx])[1] for est in model.estimators_]
    path_lengths.append(np.mean(depths))

for idx in range(len(X)):
    bar = "#" * int(path_lengths[idx])
    flag = "  <- ANOMALY" if predictions[idx] == -1 else ""
    print(f"  #{idx:>2} {labels[idx]:<15} avg depth={path_lengths[idx]:.1f}  {bar}{flag}")

print("\n" + "=" * 70)


# ── 8. Visualisation ──────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("#F4F6FA")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

ax_ports  = fig.add_subplot(gs[0, 0])   # n_dst_ports vs n_bytes_kb
ax_fail   = fig.add_subplot(gs[0, 1])   # n_dst_ports vs failed_ratio
ax_bytes  = fig.add_subplot(gs[0, 2])   # n_bytes_kb  vs failed_ratio
ax_depth  = fig.add_subplot(gs[1, 0])   # avg path length bar chart
ax_score  = fig.add_subplot(gs[1, 1])   # IF anomaly score bar chart
ax_feat   = fig.add_subplot(gs[1, 2])   # feature split count heatmap

BG = "#EAEFF7"

def scatter2d(ax, xi, yi, title, xlabel, ylabel):
    ax.set_facecolor(BG)
    for i, (xx, yy, c, lbl) in enumerate(zip(X[:, xi], X[:, yi], colors, labels)):
        ax.scatter(xx, yy, color=c, s=120, zorder=3,
                   edgecolors="white", linewidths=0.8)
        if lbl != "Normal":
            ax.annotate(lbl, (xx, yy), textcoords="offset points",
                        xytext=(6, 4), fontsize=7.5, color=c, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
    ax.grid(True, alpha=0.3, linestyle="--")

scatter2d(ax_ports, 0, 1,
          "Port scan visible here\n(n_dst_ports vs n_bytes_kb)",
          "n_dst_ports", "n_bytes_kb")

scatter2d(ax_fail, 0, 2,
          "Brute Force visible here\n(n_dst_ports vs failed_ratio)",
          "n_dst_ports", "failed_ratio")

scatter2d(ax_bytes, 1, 2,
          "Exfiltration visible here\n(n_bytes_kb vs failed_ratio)",
          "n_bytes_kb", "failed_ratio")

# Legend for scatter plots
handles = [
    mpatches.Patch(color="#2196F3", label="Normal"),
    mpatches.Patch(color="#FB8C00", label="Port Scan"),
    mpatches.Patch(color="#4E342E", label="Exfiltration"),
    mpatches.Patch(color="#8E24AA", label="Brute Force"),
]
ax_ports.legend(handles=handles, fontsize=8, loc="upper right")

# Bar chart: avg path length
ax_depth.set_facecolor(BG)
bar_colors = [("#E53935" if p == -1 else "#42A5F5") for p in predictions]
bars = ax_depth.barh(range(len(labels)), path_lengths, color=bar_colors, alpha=0.85)
ax_depth.set_yticks(range(len(labels)))
ax_depth.set_yticklabels(
    [f"#{i} {lbl}" for i, lbl in enumerate(labels)], fontsize=7.5
)
ax_depth.set_xlabel("Avg path length (depth to isolation)", fontsize=9)
ax_depth.set_title("Path Length per Point\n(shorter = anomalous)", fontsize=10,
                   fontweight="bold", pad=5)
ax_depth.axvline(np.mean(path_lengths[:10]), color="#37474F",
                 linestyle="--", linewidth=1.2, label="Normal avg")
ax_depth.legend(fontsize=8)
ax_depth.grid(True, alpha=0.25, axis="x", linestyle="--")

# Bar chart: IF scores
ax_score.set_facecolor(BG)
ax_score.barh(range(len(labels)), scores, color=bar_colors, alpha=0.85)
ax_score.set_yticks(range(len(labels)))
ax_score.set_yticklabels(
    [f"#{i} {lbl}" for i, lbl in enumerate(labels)], fontsize=7.5
)
ax_score.set_xlabel("IF anomaly score (lower = more anomalous)", fontsize=9)
ax_score.set_title("Anomaly Score per Point\n(lower score = more isolated)", fontsize=10,
                   fontweight="bold", pad=5)
ax_score.axvline(0, color="#E53935", linestyle=":", linewidth=1.3, label="Decision boundary")
ax_score.legend(fontsize=8)
ax_score.grid(True, alpha=0.25, axis="x", linestyle="--")

# Heatmap: feature split counts per tree
ax_feat.set_facecolor(BG)
im = ax_feat.imshow(all_feature_counts, cmap="YlOrRd", aspect="auto")
ax_feat.set_xticks(range(3))
ax_feat.set_xticklabels(FEATURE_NAMES, fontsize=8, rotation=15, ha="right")
ax_feat.set_yticks(range(5))
ax_feat.set_yticklabels([f"Tree {i+1}" for i in range(5)], fontsize=8)
ax_feat.set_title("Feature Split Count per Tree\n(brighter = more splits on that feature)",
                  fontsize=10, fontweight="bold", pad=5)
for i in range(5):
    for j in range(3):
        ax_feat.text(j, i, str(all_feature_counts[i, j]),
                     ha="center", va="center", fontsize=11, fontweight="bold",
                     color="white" if all_feature_counts[i, j] > all_feature_counts.max() * 0.5 else "#333")
plt.colorbar(im, ax=ax_feat, shrink=0.8)

fig.suptitle(
    "Isolation Forest — How It Works on 13 Points\n"
    "(10 normal  +  Port Scan  +  Exfiltration  +  Brute Force)",
    fontsize=13, fontweight="bold", y=0.98,
)

import os; os.makedirs("graphs", exist_ok=True)
plt.savefig("graphs/isolation_forest_explained.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("\n[+] Plot saved -> graphs/isolation_forest_explained.png")
plt.show()

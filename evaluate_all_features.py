#!/usr/bin/env python3
"""
Evaluate Isolation Forest on CTU-13: All 57 Features vs Top 13 Features
========================================================================
This script:
1. Loads and merges the CTU-13 Attack and Normal datasets.
2. Identifies all 57 features.
3. Finds the top 13 features most correlated with the ground truth Label.
4. Trains/predicts Isolation Forest using:
   a) All 57 features.
   b) Only the top 13 features.
5. Saves the precision, recall, and confusion matrices for both to evaluation_metrics.md.
6. Generates a comparison dashboard image (all_features_analysis.png).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
ATTACK_CSV = "sample_logs/CTU13_Attack_Traffic.csv"
NORMAL_CSV = "sample_logs/CTU13_Normal_Traffic.csv"
OUTPUT_PNG = "all_features_analysis.png"
REPORT_MD = "evaluation_metrics.md"

def load_data():
    """Loads normal and attack traffic files and merges them."""
    for path in (ATTACK_CSV, NORMAL_CSV):
        if not os.path.exists(path):
            sys.exit(f"[-] File not found: {path}\n"
                     f"    Please ensure the CTU-13 CSV datasets are downloaded and placed in the 'sample_logs' folder.")

    print("[*] Loading CTU-13 datasets...")
    attack_df = pd.read_csv(ATTACK_CSV)
    normal_df = pd.read_csv(NORMAL_CSV)

    print(f"    Attack flows loaded : {len(attack_df):>7,}")
    print(f"    Normal flows loaded : {len(normal_df):>7,}")

    attack_df["true_label"] = 1
    normal_df["true_label"] = 0

    combined_df = pd.concat([attack_df, normal_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return combined_df

def plot_confusion_matrix(ax, cm, title, labels=["Normal", "Attack"]):
    """Plots a beautiful heatmap of a confusion matrix."""
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, alpha=0.3)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    
    ax.set_xlabel('Predicted Label', fontsize=9, labelpad=4)
    ax.set_ylabel('True Label', fontsize=9, labelpad=4)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    
    thresh = cm.max() / 2.
    total = np.sum(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = (count / total) * 100
            label_text = f"{count:,}\n({pct:.1f}%)"
            color = "white" if cm[i, j] > thresh else "black"
            
            if i == j:
                cell_bg = "#E8F5E9" if i == 0 else "#E3F2FD"
            else:
                cell_bg = "#FFEBEE"
                
            rect = mpatches.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color=cell_bg, zorder=-1)
            ax.add_patch(rect)
            
            ax.text(j, i, label_text, ha="center", va="center",
                    color=color if i == j and cm[i, j] > thresh else "black",
                    fontsize=10, fontweight="bold")
            
            shorthand = ""
            if i == 0 and j == 0: shorthand = "TN"
            elif i == 0 and j == 1: shorthand = "FP"
            elif i == 1 and j == 0: shorthand = "FN"
            elif i == 1 and j == 1: shorthand = "TP"
            ax.text(j, i+0.28, shorthand, ha="center", va="center", color="#757575", fontsize=8)

def main():
    # 1. Load data
    df = load_data()

    # 2. Extract features
    raw_feature_cols = [c for c in df.columns if c not in ["Unnamed: 0", "Label", "true_label"]]
    n_all_features = len(raw_feature_cols)

    # 3. Find top 13 most correlated features
    print("[*] Finding top 13 features most correlated with label...")
    corr_df = df[raw_feature_cols].copy()
    corr_df["true_label"] = df["true_label"]
    
    correlations = corr_df.corr()["true_label"].abs().sort_values(ascending=False)
    top_13_features = correlations.index[1:14].tolist()
    
    print("\n" + "=" * 80)
    print("  TOP 13 FEATURES IDENTIFIED BY TARGET CORRELATION")
    print("=" * 80)
    for idx, col in enumerate(top_13_features, 1):
        print(f"   #{idx:02d}  {col:<26}  (Correlation: {correlations[col]:.4f})")
    print("=" * 80 + "\n")

    # 4. Clean and scale features
    X_all = df[raw_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_13 = df[top_13_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = df["true_label"].values

    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)
    X_13_scaled = scaler.fit_transform(X_13)

    contamination_rate = np.mean(y) # proportion of attacks (~42.2%)

    # 5. Run Isolation Forest using ALL 57 Features
    print(f"[*] Fitting Isolation Forest on ALL {n_all_features} Features...")
    if_all = IsolationForest(n_estimators=150, contamination=contamination_rate, random_state=42, n_jobs=-1)
    preds_all = if_all.fit_predict(X_all_scaled)
    scores_all = if_all.score_samples(X_all_scaled)
    y_pred_all = np.where(preds_all == -1, 1, 0)
    
    cm_all = confusion_matrix(y, y_pred_all)
    tn_all, fp_all, fn_all, tp_all = cm_all.ravel()
    
    print(classification_report(y, y_pred_all, target_names=["Normal (0)", "Attack (1)"], digits=4))

    # 6. Run Isolation Forest using TOP 13 Features
    print(f"[*] Fitting Isolation Forest on TOP 13 Features...")
    if_13 = IsolationForest(n_estimators=150, contamination=contamination_rate, random_state=42, n_jobs=-1)
    preds_13 = if_13.fit_predict(X_13_scaled)
    scores_13 = if_13.score_samples(X_13_scaled)
    y_pred_13 = np.where(preds_13 == -1, 1, 0)
    
    cm_13 = confusion_matrix(y, y_pred_13)
    tn_13, fp_13, fn_13, tp_13 = cm_13.ravel()
    
    print(classification_report(y, y_pred_13, target_names=["Normal (0)", "Attack (1)"], digits=4))

    # 7. Write results to evaluation_metrics.md
    with open(REPORT_MD, "w") as f:
        f.write(f"""# CTU-13 Network Anomaly Detection — Isolation Forest Metrics Report

This document reports the performance metrics of the **Isolation Forest** model comparing the use of **all 57 features** vs. a subset of the **top 13 features**.

---

## Dataset Characteristics
*   **Total Dataset Size**: {len(df):,} network flows
*   **Normal Flows**: {len(df[df["true_label"] == 0]):,}
*   **Attack/Anomaly Flows**: {len(df[df["true_label"] == 1]):,}
*   **Contamination Rate**: {contamination_rate:.4%}

---

## Feature Comparison
*   **All Features Config**: Using all **57 numerical features** present in the dataset.
*   **13 Features Config**: Using only the top 13 features most correlated with the attack labels:
{chr(10).join([f"    {idx}. `{col}` (correlation: {correlations[col]:.4f})" for idx, col in enumerate(top_13_features, 1)])}

---

## Model Performance Summary

| Configuration | Recall (Attack) | Precision (Attack) | F1-Score (Attack) | Overall Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **All 57 Features** | **{tp_all / (tp_all + fn_all):.2%}** | **{tp_all / (tp_all + fp_all):.2%}** | **{2 * (tp_all / (tp_all + fn_all)) * (tp_all / (tp_all + fp_all)) / ((tp_all / (tp_all + fn_all)) + (tp_all / (tp_all + fp_all))):.2%}** | **{(tn_all + tp_all) / len(df):.2%}** |
| **Top 13 Features** | **{tp_13 / (tp_13 + fn_13):.2%}** | **{tp_13 / (tp_13 + fp_13):.2%}** | **{2 * (tp_13 / (tp_13 + fn_13)) * (tp_13 / (tp_13 + fp_13)) / ((tp_13 / (tp_13 + fn_13)) + (tp_13 / (tp_13 + fp_13))):.2%}** | **{(tn_13 + tp_13) / len(df):.2%}** |

---

## Detailed Isolation Forest Reports

### 1. Configuration: All 57 Features

#### Classification Report:
```text
{classification_report(y, y_pred_all, target_names=["Normal (0)", "Attack (1)"], digits=4)}
```

#### Confusion Matrix:
*   **True Negatives (TN)**: {tn_all:,}
*   **False Positives (FP)**: {fp_all:,}
*   **False Negatives (FN)**: {fn_all:,}
*   **True Positives (TP)**: {tp_all:,}

---

### 2. Configuration: Top 13 Features

#### Classification Report:
```text
{classification_report(y, y_pred_13, target_names=["Normal (0)", "Attack (1)"], digits=4)}
```

#### Confusion Matrix:
*   **True Negatives (TN)**: {tn_13:,}
*   **False Positives (FP)**: {fp_13:,}
*   **False Negatives (FN)**: {fn_13:,}
*   **True Positives (TP)**: {tp_13:,}
""")
    print(f"[+] Written detailed metrics comparison to: {REPORT_MD}")

    # 8. Create Visualization Comparison Dashboard
    print(f"[*] Generating visual dashboard -> {OUTPUT_PNG}...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle("CTU-13 Anomaly Detection — Isolation Forest Feature Configuration Comparison", 
                 fontsize=15, fontweight="bold", y=0.98, color="#1A252C")

    # 8.1. Plot Subplot 1: Heatmap of the Top 13 Features Correlation Matrix
    ax_corr = axes[0, 0]
    top_13_with_target = top_13_features + ["true_label"]
    corr_matrix = corr_df[top_13_with_target].corr()
    im = ax_corr.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    
    ax_corr.set_xticks(np.arange(len(top_13_with_target)))
    ax_corr.set_yticks(np.arange(len(top_13_with_target)))
    ax_corr.set_xticklabels(top_13_with_target, rotation=45, ha="right", fontsize=8)
    ax_corr.set_yticklabels(top_13_with_target, fontsize=8)
    ax_corr.set_title("Correlation Heatmap of Top 13 Features + Target", fontsize=11, fontweight="bold", pad=8)
    cbar = fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    
    # Annotate matrix
    for i in range(len(top_13_with_target)):
        for j in range(len(top_13_with_target)):
            val = corr_matrix.values[i, j]
            ax_corr.text(j, i, f"{val:.2f}", ha="center", va="center", 
                         color="white" if abs(val) > 0.6 else "black", fontsize=8)

    # 8.2. Plot Subplot 2: Grouped Bar Chart of Performance Metrics
    ax_bar = axes[0, 1]
    metrics = ["Precision", "Recall", "F1-Score", "Accuracy"]
    
    # Calculate percentages for All 57 Features
    p_all = (tp_all / (tp_all + fp_all)) * 100
    r_all = (tp_all / (tp_all + fn_all)) * 100
    f1_all = (2 * p_all * r_all / (p_all + r_all))
    acc_all = ((tn_all + tp_all) / len(df)) * 100
    
    # Calculate percentages for Top 13 Features
    p_13 = (tp_13 / (tp_13 + fp_13)) * 100
    r_13 = (tp_13 / (tp_13 + fn_13)) * 100
    f1_13 = (2 * p_13 * r_13 / (p_13 + r_13))
    acc_13 = ((tn_13 + tp_13) / len(df)) * 100
    
    all_vals = [p_all, r_all, f1_all, acc_all]
    vals_13 = [p_13, r_13, f1_13, acc_13]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    rects1 = ax_bar.bar(x - width/2, all_vals, width, label='All 57 Features', color='#B0BEC5', edgecolor='black', linewidth=0.5)
    rects2 = ax_bar.bar(x + width/2, vals_13, width, label='Top 13 Features', color='#4CAF50', edgecolor='black', linewidth=0.5)
    
    ax_bar.set_ylabel('Percentage (%)', fontsize=9)
    ax_bar.set_title('Performance Comparison: All 57 Features vs Top 13 Features', fontsize=11, fontweight="bold", pad=8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics, fontsize=9)
    ax_bar.set_ylim(0, 110)
    ax_bar.grid(True, linestyle="--", alpha=0.3, axis='y')
    ax_bar.legend(fontsize=9, loc="lower right")
    
    # Annotate bar heights
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax_bar.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
                        
    autolabel(rects1)
    autolabel(rects2)


    # 8.3. Plot Subplot 3: Confusion Matrix for All 57 Features
    plot_confusion_matrix(axes[1, 0], cm_all, "Isolation Forest (All 57 Features) Confusion Matrix")

    # 8.4. Plot Subplot 4: Confusion Matrix for Top 13 Features
    plot_confusion_matrix(axes[1, 1], cm_13, "Isolation Forest (Top 13 Features) Confusion Matrix")

    # Layout adjustment and save
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"[+] Visualization updated: {OUTPUT_PNG}")
    print("\n" + "=" * 80)
    print("  SUMMARY OF ISOLATION FOREST RESULTS")
    print("=" * 80)
    print(f"  All 57 Features (Recall)    : {tp_all / (tp_all + fn_all):.2%}")
    print(f"  All 57 Features (Precision) : {tp_all / (tp_all + fp_all):.2%}")
    print(f"  Top 13 Features (Recall)    : {tp_13 / (tp_13 + fn_13):.2%}")
    print(f"  Top 13 Features (Precision) : {tp_13 / (tp_13 + fp_13):.2%}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

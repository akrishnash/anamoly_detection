# CTU-13 Network Anomaly Detection — Isolation Forest Metrics Report

This document reports the performance metrics of the **Isolation Forest** model comparing the use of **all 57 features** vs. a subset of the **top 13 features**.

---

## Dataset Characteristics
*   **Total Dataset Size**: 92,212 network flows
*   **Normal Flows**: 53,314
*   **Attack/Anomaly Flows**: 38,898
*   **Contamination Rate**: 42.1832%

---

## Feature Comparison
*   **All Features Config**: Using all **57 numerical features** present in the dataset.
*   **13 Features Config**: Using only the top 13 features most correlated with the attack labels:
    1. `SYN Flag Cnt` (correlation: 0.3993)
    2. `Fwd Pkts/s` (correlation: 0.3218)
    3. `Flow Pkts/s` (correlation: 0.3216)
    4. `Bwd Pkts/s` (correlation: 0.3163)
    5. `Init Bwd Win Byts` (correlation: 0.2722)
    6. `Active Min` (correlation: 0.2334)
    7. `Active Mean` (correlation: 0.2148)
    8. `Fwd Pkt Len Min` (correlation: 0.1781)
    9. `Bwd IAT Tot` (correlation: 0.1694)
    10. `FIN Flag Cnt` (correlation: 0.1627)
    11. `Bwd Pkt Len Std` (correlation: 0.1408)
    12. `Active Max` (correlation: 0.1363)
    13. `Pkt Size Avg` (correlation: 0.1351)

---

## Model Performance Summary

| Configuration | Recall (Attack) | Precision (Attack) | F1-Score (Attack) | Overall Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **All 57 Features** | **62.37%** | **62.49%** | **62.43%** | **68.34%** |
| **Top 13 Features** | **67.83%** | **70.00%** | **68.90%** | **74.17%** |

---

## Detailed Isolation Forest Reports

### 1. Configuration: All 57 Features

#### Classification Report:
```text
              precision    recall  f1-score   support

  Normal (0)     0.7258    0.7269    0.7264     53314
  Attack (1)     0.6249    0.6237    0.6243     38898

    accuracy                         0.6834     92212
   macro avg     0.6754    0.6753    0.6753     92212
weighted avg     0.6833    0.6834    0.6833     92212

```

#### Confusion Matrix:
*   **True Negatives (TN)**: 38,755
*   **False Positives (FP)**: 14,559
*   **False Negatives (FN)**: 14,639
*   **True Positives (TP)**: 24,259

---

### 2. Configuration: Top 13 Features

#### Classification Report:
```text
              precision    recall  f1-score   support

  Normal (0)     0.7705    0.7879    0.7791     53314
  Attack (1)     0.7000    0.6783    0.6890     38898

    accuracy                         0.7417     92212
   macro avg     0.7352    0.7331    0.7340     92212
weighted avg     0.7407    0.7417    0.7411     92212

```

#### Confusion Matrix:
*   **True Negatives (TN)**: 42,005
*   **False Positives (FP)**: 11,309
*   **False Negatives (FN)**: 12,513
*   **True Positives (TP)**: 26,385

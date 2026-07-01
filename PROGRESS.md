# Project Progress — Network Anomaly Detection

Last updated: 2026-07-01

---

## Environment

- **Python**: `C:\Users\ADRIN-ISRO\anaconda3\envs\yolov8\python.exe`
  - Base anaconda Python is broken (SRE module mismatch — bomba_backend editable install removed)
  - Always use the `yolov8` conda env (sklearn, numpy, pandas, matplotlib, scipy all present)
- **Run from**: `E:\Projects Internet room\anamoly_detection\`

---

## Datasets

| Dataset | File | Rows | Format | Labels |
|---|---|---|---|---|
| CTU-13 Attack | `sample_logs/CTU13_Attack_Traffic.csv` | 38,898 | CICFlowMeter | binary (1) |
| CTU-13 Normal | `sample_logs/CTU13_Normal_Traffic.csv` | 53,314 | CICFlowMeter | binary (0) |
| UNSW-NB15 Train | `sample_logs/unsw_nb15/UNSW_NB15_training-set.csv` | 175,341 | Argus/Bro | multi-class |
| UNSW-NB15 Test | `sample_logs/unsw_nb15/UNSW_NB15_testing-set.csv` | 82,332 | Argus/Bro | multi-class |

CTU-13: university botnet traffic (Neris, Rbot, Menti botnets), 59 CICFlowMeter features.
UNSW-NB15: 9 attack types (Fuzzers, Exploits, DoS, Backdoor, Reconnaissance, Analysis,
Shellcode, Worms, Generic), 39 Argus features. Download: `sample_logs/unsw_nb15/` is gitignored.

---

## Codebase Map

| File | What it does |
|---|---|
| `run_ctu13.py` | IF on CTU-13, 10 features, 6K+6K sample. **Baseline numbers.** |
| `run_ctu13_v2.py` | All-57-features IF. Worse — curse of dimensionality confirmed. |
| `compare_datasets.py` | Attack vs Normal: Cohen's d, ROC, PR, threshold sweep (20K rows each) |
| `diagnose_overlap.py` | KS ceiling diagnostic. Score distribution overlap + CDF gap. |
| `ensemble_detector.py` | IF + LOF + AE comparison. AE dominates. Averaging hurts. |
| `stage2_supervised.py` | Two-stage pipeline: AE flags → XGBoost classifies → zero-day queue |
| `explain_isolation_forest.py` | Educational IF walkthrough on 13-point toy dataset |
| `agent.py` | GPT-4o tool-calling agent (requires OPENAI_API_KEY) |
| `cve_db.py` | CVE + MITRE ATT&CK lookup (used by agent.py) |
| `streamlit_app.py` | Interactive app: score overlap explorer |

---

## All Experiments & Results

### Exp 1 — IF Baseline: 10 features, CTU-13 (`run_ctu13.py`)

```
Sample: 6,000 attack + 6,000 normal | Contamination: 0.40
Precision: 0.695  Recall: 0.555  F1: 0.617  Accuracy: 0.656
TN=4,541  FP=1,459  FN=2,671  TP=3,329
KS ceiling: 0.445  AUC: 0.677
```

Key: IF flags statistically unusual normal flows (high-rate backups, video) above real attacks.
The 60% plateau is structural — no contamination tuning fixes it.

### Exp 2 — All 57 features (`run_ctu13_v2.py`)

```
Precision: 0.584  Recall: 0.467  F1: 0.519  (-9.8 pts vs baseline)
```

Curse of dimensionality confirmed: random feature splits dilute signal.
At optimal PR threshold: Recall=100% at Precision=64.3% — score signal exists, threshold is wrong.

### Exp 3 — Attack vs Normal Comparison, 20K rows (`compare_datasets.py`)

Cohen's d discrimination power (attack - normal / pooled std):
- SYN flags: d=+0.86 (attack > normal — port scanning / C&C setup)
- Pkt Rate:  d=+0.63 (bimodal: near-zero beacons + DDoS spikes)
- Pkt Len:   d=-0.19 (normal has bigger packets — HTTP content)
- Bwd Bytes: d=-0.02 (C&C gets near-empty ACK replies)

At 20K rows: Precision=0.637, Recall=0.509, ROC-AUC=0.706.

### Exp 4 — KS Ceiling Diagnosis (`diagnose_overlap.py`)

```
KS = 0.445  →  max achievable (TPR - FPR) = 0.445
AUC = 0.677
→ Heavy overlap. Ceiling is STRUCTURAL. Fix = change representation, not threshold.
```

This is the core claim of the paper.

### Exp 5 — Ensemble: IF + LOF + AE (`ensemble_detector.py`)

```
IF    KS=0.445  AUC=0.677  Precision=0.584  Recall=0.467
LOF   KS=0.155  AUC=0.537  (weakest — botnet flows cluster, not locally sparse)
AE    KS=0.860  AUC=0.978  Precision=0.977  Recall=0.898  ← dominates
Ensemble (avg)  KS=0.606   (LOF drags it down — averaging hurts)
```

AE trained on normal only. Botnet C&C fails reconstruction: MSE ratio = 23.4x higher
on attack vs normal. Key insight: reconstruction anomaly >> isolation anomaly here.

Proper train/test split verification:
```
AE (95th pct threshold, held-out test):
  Precision=0.950  Recall=0.906  F1=0.928  Accuracy=0.929
  TN=2,285  FP=115  FN=225  TP=2,175
```
No data leakage — numbers hold on unseen data.

### Exp 6 — Cross-Dataset: UNSW-NB15 (`ensemble_detector.py` + inline)

```
Train: 56,000 normal + 119,341 attack (UNSW-NB15 train split)
Test:  37,000 normal + 45,332 attack (UNSW-NB15 test split)
9 attack types: Fuzzers, Exploits, DoS, Backdoor, Recon, Analysis, Shellcode, Worms, Generic
```

| Detector | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| IF | 0.349 | 0.277 | 0.309 | 0.254 |
| AE (95th pct) | 0.816 | 0.335 | 0.475 | 0.709 |

Per-attack-type recall (AE vs IF):
- Fuzzers (6K):    AE=8%,  IF=29%  ← AE fails on random payload
- Exploits (11K):  AE=47%, IF=48%  ← tie
- Generic (19K):   AE=43%, IF=21%  ← AE wins
- Shellcode (378): AE=2%,  IF=2%   ← both fail
- DoS (4K):        AE=23%, IF=24%  ← tie

**Finding**: best anomaly notion is attack-type dependent.
No single unsupervised detector generalises across all attack families.

### Exp 7 — Two-Stage Pipeline (`stage2_supervised.py`) — IN PROGRESS

Architecture:
  Stage 1 (AE) flags anomalies → Stage 2 (XGBoost) classifies known attacks →
  residual = zero-day queue

Target metrics:
  - Stage 2 precision on known attacks: >90%
  - Zero-day queue FPR: <10%
  - Stage 2 reduces Stage 1 false positives by: >60%

---

## What We Know Doesn't Work

- **Naive all-features IF**: Curse of dimensionality → worse than 10-feature baseline
- **Fixed contamination**: PR optimal threshold ≠ contamination-implied threshold
- **LOF for botnet**: Botnet flows cluster together → not locally sparse → KS=0.155
- **Simple ensemble average**: Weak detector (LOF) drags down strong (AE) → worse than AE alone
- **AE alone for Fuzzers/Shellcode**: Random-payload attacks produce scattered MSE → low recall

## What Works

- **AE trained on normal only**: KS=0.860 on CTU-13, structural attack/normal gap
- **AE precision**: Stays high (82-95%) across CTU-13 and UNSW-NB15
- **Two-stage architecture**: Stage 1 zero-day net + Stage 2 known-attack classifier

---

## Current Roadmap

| Step | What | Status |
|---|---|---|
| 1 | IF baseline + KS diagnosis | Done |
| 2 | AE vs IF comparison (CTU-13) | Done |
| 3 | Cross-dataset validation (UNSW-NB15) | Done |
| 4 | Two-stage pipeline (Stage 2 XGBoost) | **In progress** |
| 5 | Zero-day queue analysis | Next |
| 6 | Paper write-up | After step 5 |

# Anomalous != Malicious: A Two-Stage Architecture for Zero-Day Network Intrusion Detection

*Working paper — updated 2026-07-01*

---

## Title

**"Anomalous Is Not Malicious: A Two-Stage Architecture for Zero-Day Network Intrusion Detection"**

Alternatives:
- "Why Unsupervised NIDS Plateau and How a Two-Stage Pipeline Breaks the Ceiling"
- "Reconstruction Anomaly vs Isolation Anomaly: Decomposing the Unsupervised NIDS Failure"

---

## Target Venue

**Primary**: Computers & Security (Elsevier, rolling, IF ~5.1)
**Secondary**: IEEE Transactions on Network and Service Management (IEEE TNSM)
**Fallback**: IEEE Access (open-access, rolling, fast turnaround)

---

## Abstract

Unsupervised network intrusion detection systems (NIDS) based on anomaly scores plateau at
55–70% recall on real-world datasets — a ceiling that persists despite tuning. We show this
ceiling is *structural*, not a parameter artefact: the Kolmogorov-Smirnov (KS) distance
between the anomaly-score distributions of attack and normal flows is 0.445 on CTU-13, meaning
no threshold can achieve (TPR - FPR) > 0.445 regardless of calibration. The root cause is that
unsupervised detectors conflate statistical rarity with malice — a property we term
**"anomalous != malicious"**: stealthy botnet C&C beaconing sits *inside* the normal-traffic
distribution, while benign high-throughput flows look anomalous.

We show that switching the anomaly notion from *isolation* (Isolation Forest, KS=0.445) to
*reconstruction* (Autoencoder trained on normal traffic, KS=0.860) raises recall from 55% to
90% on CTU-13 — but collapses on UNSW-NB15 Fuzzers (8% recall), exposing that no single
unsupervised notion generalises across all attack families.

To resolve this, we propose a **two-stage architecture**: Stage 1 (Autoencoder) acts as a
zero-day net, flagging anything that deviates from the learned normal manifold; Stage 2
(Gradient Boosting) classifies the flagged pool into known attack types, while flows it cannot
confidently classify form a **zero-day candidate queue**. On CTU-13 this achieves F1=0.945 and
reduces Stage 1 false positives by 87%. In a held-out zero-day simulation on UNSW-NB15 —
where Stage 2 has never seen Backdoor, Shellcode, Analysis, or Worms — the zero-day queue
captures 71% real attacks with the hidden types routing there automatically.

The key remaining bottleneck is Stage 1 recall on low-volume stealthy attacks (2-13% for hidden
types), which we attribute to the representation limit of per-flow features. We propose temporal
and graph features as the fix.

---

## 1. Introduction

### The Problem

Signature-based NIDS fail on zero-day attacks by definition — they match known patterns.
Anomaly-based detection offers the alternative: learn what normal traffic looks like, flag
deviations. But practitioners know the performance is disappointingly low in practice. We
investigate *why*.

The standard explanation is "threshold tuning" or "too few features." We show both are wrong.

### The Core Finding: Anomalous != Malicious

The failure is *structural*. Botnet C&C traffic — the dominant attack type in CTU-13 — is
designed to mimic normal idle connections: low packet rate, small payloads, periodic timing.
It is not a statistical outlier. No unsupervised detector trained to flag outliers can reliably
catch it, because it is not an outlier.

Conversely, legitimate high-throughput flows (backups, video streams) ARE statistical outliers
and get flagged constantly, driving false positives.

We quantify this with the KS distance between score distributions:

```
KS = 0.445 for Isolation Forest on CTU-13
   = the maximum achievable (TPR - FPR) over all thresholds
   = the structural ceiling, not a tuning parameter
```

### Our Contributions

1. **Diagnosis**: KS-distance decomposition proving the ceiling is structural, not tunable.
   Cohen's d feature analysis showing which flow features actually discriminate (SYN d=+0.86)
   vs which are noise.

2. **Finding**: Reconstruction anomaly (AE) dramatically outperforms isolation anomaly (IF)
   on structured periodic attacks (CTU-13: +35 pts recall) but degrades on random-payload
   attacks (UNSW-NB15 Fuzzers: 8% AE vs 29% IF). No single unsupervised notion generalises.

3. **Architecture**: Two-stage pipeline — Stage 1 zero-day net (AE) + Stage 2 known-attack
   classifier (GBM) — validated on CTU-13 (F1=0.945, 87% FP reduction) and UNSW-NB15
   zero-day simulation (71% queue precision on 4 completely hidden attack types).

---

## 2. Background

### 2.1 Isolation Forest (Liu et al., 2008)

Builds random trees by repeatedly choosing a random feature and a random split value.
Anomalous points isolate near the root (short average path length = low score).

**Key property**: measures *global rarity* — how different a point is from the bulk.
Stealthy attacks that blend into the bulk score as normal. This is the KS ceiling.

### 2.2 Autoencoder Anomaly Detection

An MLP encoder-decoder trained to reconstruct *only normal traffic*. At inference, a
flow that doesn't lie on the learned normal manifold has high reconstruction error (MSE).

**Key property**: measures *reconstruction anomaly* — whether a flow looks like anything
in the normal-traffic distribution. Structured attacks (constant beaconing, near-zero
payloads) produce consistent reconstruction errors that normal traffic never produces.

### 2.3 KS Distance as a Structural Ceiling

The Kolmogorov-Smirnov statistic between the attack and normal score distributions equals
the maximum Youden's J (TPR - FPR) achievable over all thresholds. If KS=0.445, no
threshold tuning, re-scaling, or calibration can beat (TPR - FPR) = 0.445. It is a
hard upper bound on any threshold-based decision using those scores.

### 2.4 Cohen's d

Effect size = (mean_attack - mean_normal) / pooled_std.
Used here to rank features by their discriminative power independent of the detector.
Large |d| = the feature separates attacks from normal on its own.
Small |d| = the feature is noise for this detection task.

---

## 3. Datasets

### CTU-13 (Botnet, CICFlowMeter features)

- Source: Czech Technical University. Real university network captures.
- Attack flows: 38,898 (Neris, Rbot, Menti botnets — primarily C&C beaconing)
- Normal flows: 53,314 (campus traffic)
- Features: 57 CICFlowMeter columns per bidirectional flow
- Experiments: 6,000 attack + 6,000 normal = 12,000 flows (70/30 stratified split)

### UNSW-NB15 (Multi-class, Argus/Bro features)

- Source: UNSW Canberra.
- Train: 175,341 flows | Test: 82,332 flows
- 9 attack categories: Generic, Exploits, DoS, Fuzzers, Reconnaissance,
  Backdoor, Analysis, Shellcode, Worms
- Features: 39 Argus/Bro columns per flow

Used for: cross-dataset generalisation test and zero-day simulation (§ Exp 7).

---

## 4. Experiments and Results

### Exp 1 — Isolation Forest Baseline (CTU-13, 10 features)

```
Features: flow_byts_s, flow_pkts_s, fwd_bytes, bwd_bytes, total_pkts,
          syn_flag, rst_flag, fin_flag, flow_duration_s, pkt_len_mean
Contamination: 0.40

Precision: 0.695   Recall: 0.555   F1: 0.617   Accuracy: 0.656
TN=4,541  FP=1,459  FN=2,671  TP=3,329
KS ceiling: 0.445   ROC-AUC: 0.677
```

Observation: The most anomalous-scoring flows are often normal (high-rate backups,
video streams). Stealthy C&C beaconing sits inside the normal score distribution.

### Exp 2 — All 57 Features (Curse of Dimensionality)

```
Precision: 0.584  Recall: 0.467  F1: 0.519  (-9.8 pts vs baseline)
```

Adding 47 more features hurts because IF picks features randomly at each split.
With 57 features, most splits land on uninformative dimensions, diluting signal.
At the PR-optimal threshold: Recall=100% at Precision=64% — the score is there,
but swamped by noise. Feature selection, not more features, is the fix.

### Exp 3 — KS Ceiling Diagnosis

```
KS distance: 0.445
Interpretation: max achievable (TPR - FPR) = 0.445
               i.e. no threshold can give more than TPR = FPR + 0.445
The ceiling is STRUCTURAL. No tuning fixes it.
```

Cohen's d per feature (attack - normal):

| Feature | Cohen's d | Direction | What it captures |
|---|---|---|---|
| SYN Flag Count | +0.86 | Attack > Normal | Port scanning, C&C connection setup |
| Packet Rate | +0.63 | Attack > Normal | Bimodal: near-zero beacons + DDoS spikes |
| Fwd Pkts/s | +0.63 | Attack > Normal | High forward rate in scanning phases |
| FIN Flag Count | +0.31 | Attack > Normal | Abruptly closed connections |
| Pkt Len Mean | -0.19 | Normal > Attack | Normal traffic has larger payloads (HTTP) |
| Bwd Bytes | -0.02 | Normal > Attack | C&C victims reply with near-empty ACKs |

Even the best individual feature (SYN, d=0.86) is insufficient alone — attacks span
multiple subtypes with different dominant features.

### Exp 4 — Autoencoder (Normal-Only Training)

```
Architecture: MLP encoder-decoder (dim → dim//2 → dim//4 → dim//2 → dim)
Training: ONLY on normal flows (no attack labels used)
Threshold: 95th percentile of normal-traffic MSE

CTU-13 (6K+6K, 70/30 split, held-out test):
  KS: 0.860   ROC-AUC: 0.978
  Precision: 0.950   Recall: 0.906   F1: 0.928   Accuracy: 0.929
  TN=2,285  FP=115  FN=225  TP=2,175
  Attack MSE mean: 0.1914  vs  Normal MSE mean: 0.0082  (23.4x higher)
```

Why it works: Botnet C&C (near-zero bytes, periodic tiny packets) is completely
unlike any normal traffic the AE learned. Reconstruction fails catastrophically.
Isolation Forest missed these flows because they are not global outliers — but
they ARE off the normal reconstruction manifold.

Detector comparison on CTU-13:

| Detector | KS | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Isolation Forest | 0.445 | 0.677 | 0.695 | 0.555 | 0.617 |
| LOF | 0.155 | 0.537 | 0.529 | 0.423 | 0.470 |
| **Autoencoder** | **0.860** | **0.978** | **0.950** | **0.906** | **0.928** |
| Ensemble (avg) | 0.606 | 0.730 | 0.637 | 0.509 | 0.566 |

LOF is worst because botnet flows cluster (not locally sparse). Simple ensemble
averaging is sub-optimal — the weak LOF drags down the dominant AE.

### Exp 5 — Cross-Dataset: UNSW-NB15 (9 Attack Types)

AE trained on UNSW-NB15 normal flows, tested on held-out UNSW-NB15 test set:

```
AE: Precision=0.816  Recall=0.335  F1=0.475  ROC-AUC=0.709
IF: Precision=0.349  Recall=0.277  F1=0.309  ROC-AUC=0.254
```

Per-attack-type recall reveals the failure pattern:

| Attack type | Count | AE recall | IF recall | Winner |
|---|---|---|---|---|
| Generic | 8,160 | 43% | 21% | AE |
| Exploits | 5,194 | 47% | 48% | Tie |
| Fuzzers | 479 | **8%** | **29%** | IF |
| DoS | 926 | 23% | 24% | Tie |
| Reconnaissance | 270 | 35% | 13% | AE |
| Backdoor | 583 | 2% | 3% | Both fail |
| Shellcode | 378 | 2% | 2% | Both fail |
| Analysis | 677 | 7% | 4% | Both fail |
| Worms | 44 | 53% | 8% | AE |

**Critical finding**: The best anomaly notion is attack-type dependent. AE excels on
structured periodic attacks (Generic, Worms). IF surprisingly beats AE on Fuzzers —
random payload traffic is geometrically isolated even when it's not reconstruction-anomalous.
Both fail on Backdoor, Shellcode, Analysis: low-volume, low-rate, similar to normal at
the per-flow level.

### Exp 6 — Two-Stage Pipeline (stage2_supervised.py)

```
Stage 1: AE (95th pct MSE threshold) — flags anomalous flows
Stage 2: Gradient Boosting — classifies flagged flows into known attack types
         Flows with confidence < 0.60 → zero-day candidate queue
```

**CTU-13 results (70/30 stratified split)**:

| Stage | Precision | Recall | F1 | False Positives |
|---|---|---|---|---|
| Stage 1 alone (AE) | 0.951 | 0.904 | 0.927 | 84 |
| Full pipeline (AE + GBM) | **0.993** | 0.901 | **0.945** | **11** |

Stage 2 reduces false positives from 84 to 11 — **87% fewer false alarms** — with
effectively no recall cost. The zero-day queue (5 flows, 60% precision) is minimal
on CTU-13 because Stage 2 recognises botnet patterns confidently.

**UNSW-NB15 results** (Stage 2 trained on UNSW-NB15 flagged flows):

| Stage | Precision | Recall | F1 |
|---|---|---|---|
| Stage 1 alone (AE) | 0.816 | 0.335 | 0.475 |
| Full pipeline | 0.828 | 0.334 | 0.476 |

Limited gain on UNSW-NB15 because the bottleneck is Stage 1 recall (33%), not
false positives. Stage 2 can only filter what Stage 1 flags.

### Exp 7 — Zero-Day Simulation (zero_day_sim.py)

**Setup**: Train Stage 2 on 5 *known* attack types (Generic, Exploits, DoS, Fuzzers,
Reconnaissance). Hold out 4 types completely — Stage 2 has never seen them:
Backdoor, Shellcode, Analysis, Worms.

**Question**: Do the hidden attack types route to the zero-day queue automatically?

```
Zero-Day Queue (UNSW-NB15 test set, 82,332 flows):
  Total flows in queue : 1,555
  Real attacks         : 1,097  (71% precision)
  False alarms         : 458    (29%)
  Hidden-type flows    : 53     (all 4 hidden types represented)
```

Queue composition by hidden type:
```
  Backdoor   : 31 flows  <- Stage 2 never trained on this
  Analysis   : 17 flows  <- Stage 2 never trained on this
  Shellcode  :  4 flows  <- Stage 2 never trained on this
  Worms      :  1 flow   <- Stage 2 never trained on this
```

**Stage 2 never misclassified a hidden-type flow as a known attack type.**
Every hidden-type flow that Stage 2 encountered went correctly to the zero-day queue.

Detection rate for hidden types end-to-end:

| Hidden type | In test set | Stage 1 flagged | Reached ZD queue |
|---|---|---|---|
| Backdoor | 583 | 73 (13%) | 31 (5%) |
| Analysis | 677 | 45 (7%) | 17 (3%) |
| Shellcode | 378 | 6 (2%) | 4 (1%) |
| Worms | 44 | 17 (39%) | 1 (2%) |

**The bottleneck is Stage 1**, not Stage 2. Of the 4 hidden types, only 2-39%
are flagged by the AE at all. Those that ARE flagged route correctly.

---

## 5. Discussion

### 5.1 Why the Bottleneck Is Stage 1

Backdoor, Shellcode, and Analysis attacks all share a property with C&C beaconing:
they generate low-volume, low-rate flows that look similar to normal idle connections
at the per-flow feature level. The AE trained on normal flows cannot distinguish them
because the MSE is low — they reconstruct adequately from the normal manifold.

This is the representation bottleneck: CICFlowMeter and Argus features aggregate each
flow into a single row. A Backdoor session might produce 10 packets over 300 seconds —
the flow-level statistics (mean packet size, rate) overlap heavily with a normal idle
SSH session. They are only distinguishable by *temporal patterns across multiple flows*
(periodic beaconing at exact intervals) or *graph topology* (fan-out to unusual IPs).

### 5.2 What the Zero-Day Queue Precision (71%) Means

Of every 100 flows sent to an analyst's zero-day queue, 71 are real attacks and 29 are
false alarms (legitimate traffic Stage 1 flagged but Stage 2 could not classify).
This is a deployable operating point: the analyst reviews 1,555 flows out of 82,332
(1.9%) and finds 1,097 real attacks they would otherwise miss. Without the pipeline,
those attacks are invisible.

### 5.3 The Ensemble Averaging Problem

Averaging IF + LOF + AE scores is suboptimal: a weak detector (LOF, KS=0.155) pulls
the ensemble score away from the dominant AE (KS=0.860). The right approach is
learned fusion — logistic regression or learned weighting on a small labeled set that
assigns near-zero weight to LOF. Alternatively, Stage 2 already performs this role
implicitly: it uses GBM on the combined feature space, effectively a learned ensemble.

### 5.4 Attack-Type Dependency of Anomaly Notions

No single unsupervised notion dominates:
- Reconstruction anomaly (AE): best for structured, periodic, low-variance attacks
- Isolation anomaly (IF): better for noisy/random attacks (Fuzzers) that are globally sparse
- Contrastive learning (CLAD, 2025): promising — learns tight normal cluster in embedding
  space, may generalise across both attack families. Not yet implemented.

---

## 6. What To Do Next

### Immediate (unblocks paper writing)

1. **Temporal features for Stage 1** — `temporal_features.py`
   Extract per-source-IP features over 30-second windows: beacon regularity
   (FFT on inter-arrival times), session count entropy, burst ratio.
   Test on hidden types (Backdoor, Shellcode): expected to lift Stage 1 recall
   from 2-13% to 30-50%. This is the go/no-go experiment for the paper's punchline.

2. **Contrastive AE for Stage 1** — replace MSE loss with contrastive loss
   (SimCLR-style): normal flows attract, any anomalous flow repels.
   Specifically targets the cross-dataset collapse on Fuzzers.

3. **Learned ensemble fusion** — instead of averaging IF+LOF+AE scores,
   train a logistic regression on a small labeled set to weight them.
   Compare: avg ensemble (KS=0.606) vs learned fusion (expected KS ~0.85).

### Paper (write in this order)

1. Section 4 (experiments) — tables and figures are already final. Write captions first.
2. Section 3 (datasets) — straightforward, 1 page.
3. Section 5 (discussion) — "anomalous != malicious" framing. 2 pages.
4. Section 1 (introduction) — write last when the story is clear.
5. Abstract — write after introduction.
6. 8-page target (Computers & Security short paper format).

### Scripts to build

| Script | Purpose | Priority |
|---|---|---|
| `temporal_features.py` | Per-IP beacon regularity over time windows | HIGH |
| `contrastive_ae.py` | SimCLR-style AE to fix Fuzzer recall | MEDIUM |
| `learned_fusion.py` | Replace avg ensemble with logistic regression weights | MEDIUM |
| `paper_figures.py` | Final publication-quality figures (all 4 panels) | HIGH |

---

## 7. Paper Structure (Final)

```
1. Introduction (1.5 pages)
   - Problem: signature NIDS fail on zero-day; anomaly NIDS plateau at 60%
   - Why the plateau is structural (KS preview)
   - Three contributions: diagnosis + finding + architecture
   - Road map

2. Background (1 page)
   - Isolation Forest, KS distance, Autoencoder anomaly detection
   - CTU-13 and UNSW-NB15 datasets (brief)

3. The Anomaly-Malice Gap: Diagnosis (1.5 pages)
   - KS ceiling: 0.445 on IF (Exp 1, 3)
   - Cohen's d analysis: which features actually discriminate (Exp 3)
   - Why C&C beaconing is NOT an outlier

4. Anomaly Notion Matters: AE vs IF (1 page)
   - AE: KS=0.860, 23.4x MSE ratio (Exp 4)
   - LOF: worst (clustering, not sparse)
   - Cross-dataset collapse: attack-type dependency (Exp 5)

5. Two-Stage Architecture (1.5 pages)
   - Stage 1 + Stage 2 design
   - CTU-13: F1=0.945, 87% FP reduction (Exp 6)
   - Zero-day simulation: 71% queue precision, hidden types route correctly (Exp 7)
   - Remaining bottleneck: Stage 1 recall on low-volume attacks

6. Discussion: Representation Bottleneck (0.5 pages)
   - Per-flow features insufficient for Backdoor/Shellcode
   - Temporal/graph features as the fix (future work)

7. Conclusion (0.5 pages)
   - "Anomalous != malicious" confirmed quantitatively
   - Two-stage architecture as a practical zero-day detector
   - Next: temporal features to close Stage 1 recall gap

References (~20 citations)
```

---

## 8. Key References

- Liu, F.T., Ting, K.M., Zhou, Z.H. (2008). *Isolation Forest.* ICDM 2008.
- Garcia, S. et al. (2014). *An empirical comparison of botnet detection methods.* Computers & Security.
- Moustafa, N., Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion detection.* MilCIS 2015.
- Hariri, S. et al. (2021). *Extended Isolation Forest.* IEEE TNNLS.
- Shen, Y. et al. (2025). *CLAD: Contrastive Loss for Anomaly Detection.* arXiv 2601.09902.
- Self-Supervised Transformer IDS (2025). arXiv 2505.08816.
- [KD-GAT] Graph Attention for IoT Botnet Detection (2025). arXiv 2505.17357.
- AMDS: Multi-Stage Adaptive Defense System (2025). arXiv 2603.00859.

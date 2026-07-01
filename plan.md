# Research Plan — Two-Stage Zero-Day Detector

*Started: 2026-06-27. Updated: 2026-07-01.*
*Scope: complete, honest paper on why unsupervised NIDS plateau and how a two-stage
architecture breaks the ceiling — validated on two real-world datasets (CTU-13 + UNSW-NB15).*

---

## The Thesis (updated)

Unsupervised network anomaly detectors plateau at ~60% because **"anomalous ≠ malicious"**:
stealthy C&C beaconing sits inside the normal distribution, while benign high-throughput
flows look anomalous. The failure is structural (KS ceiling = 0.445), not a tuning bug.

The fix is not a better unsupervised algorithm — it is **a two-stage architecture**:
1. **Stage 1 (unsupervised AE)**: catches everything that doesn't look like normal traffic,
   including zero-day attacks the supervised layer has never seen.
2. **Stage 2 (supervised XGBoost)**: classifies the flagged pool into known attack types;
   what remains unclassified is the zero-day candidate queue.

**Working title:**
*"Anomalous ≠ Malicious: A Two-Stage Architecture for Zero-Day Network Intrusion Detection."*

**Three contributions:**
1. **(Diagnosis)** KS-distance decomposition proving the unsupervised ceiling is structural.
2. **(Finding)** Autoencoder reconstruction anomaly outperforms IF on structured attacks
   (CTU-13: +43 pts recall) but degrades on random-payload attacks (UNSW-NB15 Fuzzers).
3. **(Architecture)** Two-stage pipeline: Stage 1 (zero-day net) + Stage 2 (known classifier)
   validated on CTU-13 and UNSW-NB15 — showing per-stage precision/recall and zero-day queue.

---

## Two-Stage Architecture

```
All traffic
    │
    ▼
Stage 1 — Unsupervised AE (trained on normal only)
    │
    ├── Low anomaly score  ──→  NORMAL  (pass through)
    │
    └── High anomaly score ──→  SUSPICIOUS POOL
                                      │
                                      ▼
                              Stage 2 — XGBoost
                              trained on known labeled attacks
                              (CTU-13 + UNSW-NB15)
                                      │
                                      ├── High confidence match ──→  KNOWN ATTACK
                                      │                               (classify + alert)
                                      │
                                      └── Low confidence / no match ──→  ZERO-DAY QUEUE
                                                                          (send to analyst)
```

Key insight: Stage 2 filters Stage 1's false positives (known-normal flows that AE
flagged) while preserving genuinely novel attacks in the zero-day queue.

---

## What We Know (completed experiments)

### Unsupervised baselines (CTU-13, 6K+6K flows)

| Detector | Precision | Recall | F1 | KS ceiling |
|---|---|---|---|---|
| IF (10 features) | 0.695 | 0.555 | 0.617 | 0.445 |
| IF (57 features) | 0.584 | 0.467 | 0.519 | — |
| LOF | 0.529 | 0.423 | 0.470 | 0.155 |
| **AE (normal only)** | **0.977** | **0.898** | **0.868** | **0.860** |
| Ensemble avg | 0.637 | 0.509 | 0.566 | 0.606 |

### Cross-dataset (UNSW-NB15, 175K train / 82K test, 9 attack types)

| Detector | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| IF | 0.349 | 0.277 | 0.309 | 0.254 |
| AE (95th pct) | **0.816** | 0.335 | 0.475 | **0.709** |

AE collapses on Fuzzers (8% recall) and Shellcode (2%) — random-payload attacks.
IF beats AE on Fuzzers (29% vs 8%). Neither is a complete detector alone.

### Key finding
Best anomaly notion is attack-type dependent — no single unsupervised detector
generalises across all categories. Two-stage architecture resolves this.

---

## 4-Act Plan

### Act 1 — Diagnosis (DONE)
- [x] Honest baseline: IF at 40% contamination, CTU-13 train/test split
- [x] KS distance ceiling: KS=0.445, AUC=0.677 → structural, not tunable
- [x] Cohen's d per feature → identifies real discriminators (SYN flags d=0.86)
- [x] Overlap diagnosis figure (`diagnose_overlap.py`, `graphs/overlap_ctu13.png`)

### Act 2 — Finding: anomaly notion matters (DONE)
- [x] AE trained on normal only → KS=0.860, Precision=0.977, Recall=0.898 on CTU-13
- [x] Ensemble (IF + LOF + AE) — averaging hurts: LOF drags to KS=0.606
- [x] Cross-dataset test on UNSW-NB15 → AE recall collapses on Fuzzers/Shellcode
- [x] Per-attack-type breakdown confirms attack-type dependency
- **Finding**: reconstruction anomaly >> isolation anomaly for structured attacks;
  neither dominates across all attack families.

### Act 3 — Two-Stage Architecture (IN PROGRESS)
- [x] Design: Stage 1 (AE) + Stage 2 (XGBoost on flagged pool)
- [ ] `stage2_supervised.py` — train/test on CTU-13, show per-stage metrics
- [ ] Cross-dataset: train Stage 2 on CTU-13, test on UNSW-NB15 → zero-day queue
- [ ] Key metric: **zero-day queue precision** (how many in the queue are real novel attacks)
- **Target**: Stage 2 reduces Stage 1 false positives by >60%; zero-day queue FPR < 10%

### Act 4 — Write + figures
- [ ] Paper: 8 pages, 3 core figures (KS diagram, AE vs IF, two-stage pipeline results)
- [ ] Venue: Computers & Security (Elsevier) or IEEE TNSM
- [ ] Honest limitations: payload features not used, CTU-13 is 2011-era traffic

---

## Scripts Map

| Script | Role | Status |
|---|---|---|
| `run_ctu13.py` | IF baseline (10 features) | Done |
| `run_ctu13_v2.py` | All-features IF | Done |
| `compare_datasets.py` | Cohen's d + AUC comparison | Done |
| `diagnose_overlap.py` | KS ceiling diagnostic | Done |
| `ensemble_detector.py` | IF + LOF + AE comparison | Done |
| `stage2_supervised.py` | Two-stage pipeline | **Building now** |
| `explain_isolation_forest.py` | Educational IF walkthrough | Done |

---

## Venue Targets

- **Computers & Security** (Elsevier, rolling, IF ~5.1) — primary target
- **IEEE TNSM** — secondary
- Workshop @ IEEE S&P or USENIX — fast checkpoint if needed

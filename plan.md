# Research Plan — 1-Month Paper

*Decided 2026-06-27. Scope: a complete, honest, single-dataset paper achievable in ~1 month of part-time work (~40–60 focused hours alongside the day job), that also seeds a larger multi-dataset journal paper later.*

---

## The Thesis

Unsupervised network anomaly detectors (Isolation Forest et al.) plateau at ~60% because
**"anomalous ≠ malicious"** — stealthy C&C beaconing sits *inside* the normal distribution,
while benign high-throughput flows look anomalous. The failure is structural, not a tuning bug.

**Working title:** *"Why Unsupervised NIDS Plateau: A Failure Decomposition, and Evidence
that Representation — Not Algorithm — Is the Bottleneck."*

**Two contributions:**
1. **(Act 1 — measurement)** A principled decomposition of *why* unsupervised NIDS fail, into:
   (a) anomaly–malice mismatch, (b) feature inadequacy, (c) threshold mis-selection.
2. **(Act 2 — finding)** Preliminary evidence that switching from **per-flow** features to
   **temporal/graph** representations recovers attacks the detector missed — i.e. the
   bottleneck is the *representation*, not the algorithm.

SHAP is demoted from headline to **one instrument inside (b)**, not the paper's point.

---

## Why this scope (and not the bigger versions)

- **Direction #1 (measure the ceiling)** is achievable in a month — mostly analysis on the
  existing pipeline, low engineering, low risk. Even a "boring" confirming result is publishable.
- **Direction #2 (break the ceiling, full multi-dataset)** is a 2–3 month journal paper —
  needs temporal/graph feature engineering across 3 datasets. Too big for a month.
- **This plan = all of #1 + ONE decisive #2 experiment** on CTU-13 only, as the punchline.
  Reviewers accept "full multi-dataset study is future work."

---

## Novelty (what actually qualifies)

- ❌ NOT novel: IF on CTU-13, SHAP-on-IF, feature selection recovers F1, IF-vs-OCSVM-vs-LOF,
  curse of dimensionality (known/expected → demote to *motivation*).
- ✅ Candidate novelty: the **error decomposition** (nobody cleanly separates the 3 failure
  sources) and the **representation-beats-algorithm finding** (a result whose magnitude an
  expert could not predict).
- Rule: novelty is confirmed only when the result **could have gone the other way and didn't**.
  → the Week-3 FN-recovery number is the go/no-go.

---

## 4-Week Plan

### Week 1 — Fix the setup, honest baselines
- [ ] Re-run IF at **realistic contamination (1–5%)**, NOT the balanced 40% — the "60%" may
      shift, and we need the real number.
- [ ] Lock train/eval protocol; capture confusion matrix + PR curve.
- [ ] Confirm CTU-13 **botnet subtype labels** survived preprocessing (needed for per-family
      analysis); re-derive if lost.
- **Deliverable:** defensible baseline table.

### Week 2 — Act 1: decompose the failure (core contribution)
- [ ] Measure anomaly-score ↔ true-label correlation → quantify "anomalous ≠ malicious".
- [ ] Decompose error into (a) anomaly–malice mismatch, (b) feature inadequacy, (c) threshold.
- [ ] SHAP used here only as an instrument inside (b); add U-tests / effect sizes for rigor.
- **Deliverable:** the decomposition figure (the headline).

### Week 3 — Act 2: the decisive experiment (RUN AS EARLY AS POSSIBLE)
- [ ] Take the false-negatives (attacks IF missed per-flow).
- [ ] Engineer host-level **temporal periodicity** (autocorrelation/FFT on inter-arrival,
      beacon regularity, burstiness) + a couple of **graph** features (fan-out, conn entropy).
- [ ] Re-score those flows; report the **flip rate** (how many missed attacks become detected).
- **Deliverable:** the punchline number. Go/no-go for the paper's excitement level.
  - Strong flip (~70%) → strong conference paper.
  - Weak flip (~10%) → "the ceiling is deeper than representation" — still a finding, pivot framing.

### Week 4 — Write + figures
- [ ] Workshop length (6–8 pages); tighten the 3 figures.
- [ ] Honest limitations: single dataset for Act 2, contamination caveats.
- **Deliverable:** submittable draft.

---

## Venue Targets (realistic — NOT top-tier)

- Top-tier (S&P / USENIX / CCS / NDSS): ❌ out of scope, do not aim here.
- Realistic: **Computers & Security** (Elsevier, journal, rolling) or **IEEE TNSM**; or a
  **workshop co-located with a big IEEE/ACM conf** as a fast first checkpoint.

---

## Hard Rules

1. **Run the Week-3 experiment first** if at all possible — highest-information, highest-risk.
   Knowing the flip rate early lets us write Act 1 with confidence (or pivot).
2. **Re-run at realistic contamination before committing the narrative** — don't explain an
   artifact of the balanced/40%-contamination eval setup.
3. Every claimed contribution must complete: *"Prior work shows ___. We show ___, which
   surprises that because ___."* If it can't, it's background, not a contribution.

---

## Next Scripts To Build

- `baseline_realistic.py` — IF at realistic contamination, confusion matrix + PR curve (Week 1)
- `fn_recovery.py` — temporal/graph features on the missed false-negatives, report flip rate (Week 3)

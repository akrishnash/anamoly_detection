# SecureAI Agent — AI Threat Triage on Real Botnet Traffic

> A **GPT-4o tool-calling agent** that triages network traffic for threats: it runs an
> unsupervised **Isolation Forest** detector, then enriches every finding with CVE and
> **MITRE ATT&CK** context to return a grounded, analyst-grade report.
>
> The detection engine is validated on the real-world **CTU-13 botnet dataset**
> (38,898 attack flows + 53,314 benign): it **catches 55% of stealthy botnet attacks with
> zero labelled training data** (69% precision) — and is tunable to higher recall at the
> optimal threshold. Runs in the cloud on GPT-4o or fully offline via Ollama.

---

## What the agent does

```
PCAP / network logs
        │
        ▼
┌─────────────────────────────┐
│   agent.py  (GPT-4o loop)   │   bounded tool-calling loop, full audit trail
└──────┬───────────────┬──────┘
       │               │
       ▼               ▼
 analyze_traffic()   lookup_cve()
 (Isolation Forest)  (CVE + MITRE ATT&CK KB)
       │               │
       ▼               ▼
┌─────────────────────────────┐
│  Grounded threat report:    │
│  findings · severity · CVEs │
│  · ATT&CK · mitigations     │
└─────────────────────────────┘
```

The agent is **advisory only** — it never takes automated action. Every tool call is
logged; human review is always the final step.

---

## Quick Start

```bash
git clone https://github.com/akrishnash/anomaly-detection.git
cd anomaly-detection
pip install -r requirements.txt
```

### Run the agent

```bash
export OPENAI_API_KEY=sk-...
python agent.py
```

**Local / air-gapped mode** — swap two lines in `agent.py`, no internet required:

```python
# Cloud
client = OpenAI()
MODEL  = "gpt-4o"

# Local via Ollama
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL  = "llama3.1"
```

### Run the detection engine directly

```bash
python run_ctu13.py            # 10-feature Isolation Forest on CTU-13
python run_ctu13_v2.py         # all-features version (all CICFlowMeter columns)
python compare_datasets.py     # Attack vs Normal — Cohen's d, ROC, PR, threshold sweep
python diagnose_overlap.py     # score-overlap diagnosis (AUC + KS ceiling)
python explain_isolation_forest.py  # step-by-step IF walkthrough on toy data
```

Graphs are saved to `graphs/`. (Place the CTU-13 CSVs in `sample_logs/` first — source:
[imfaisalmalik/CTU13-CSV-Dataset](https://github.com/imfaisalmalik/CTU13-CSV-Dataset).)

---

## The Detection Engine

Fully **unsupervised** — no labelled attack data required. Validated on CTU-13:

### `run_ctu13.py` (6,000 flows per class, 10 features)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 0.630 | 0.757 | 0.687 |
| Attack | **0.695** | **0.555** | **0.617** |
| Overall accuracy | | | **0.656** |

```
Confusion Matrix:
  TN = 4,541   FP = 1,459
  FN = 2,671   TP = 3,329
```

### `compare_datasets.py` (20,000 flows per class)

| Metric | Value |
|---|---|
| Attack Precision | 0.637 |
| Attack Recall | 0.509 |
| ROC-AUC | **0.706** |
| Avg Precision | 0.612 |

> **On operating points:** the default uses a fixed contamination threshold. Sweeping the
> decision threshold along the PR curve reaches **~78% F1 (100% recall at 64% precision)** —
> i.e. *every* attack can be caught at the cost of more false positives. Threshold selection
> is a deployment choice, not a model limit. See `diagnose_overlap.py` and `paper.md`.

### Why stealthy attack traffic still signals attack

CTU-13 is predominantly **botnet C&C** — deliberately quiet and low-volume. The median
byte rate for attack flows is near zero: not "nothing happening," but suspiciously *too
quiet* for real user traffic. The discriminating signals (Cohen's d effect size):

| Feature | Cohen's d | Direction | Meaning |
|---|---|---|---|
| SYN Flag Count | **+0.86** | Attack > Normal | Constant connection attempts — port scans / C&C setup |
| Packet Rate | **+0.63** | Attack > Normal | Bimodal: near-zero beacons AND DDoS burst spikes |
| Fwd Pkts/s | **+0.63** | Attack > Normal | High forward rate in scanning phases |
| FIN Flag Count | **+0.31** | Attack > Normal | Many abruptly closed connections |
| Pkt Len Mean | **-0.19** | Normal > Attack | Normal has larger packets (HTTP/file data) |
| Bwd Bytes | **-0.02** | Normal > Attack | C&C victims reply with near-empty ACKs |

Isolation Forest flags anomalies in the **joint feature space** — a flow with near-zero
bytes + elevated SYN + specific timing sits in an isolated region a random tree cuts off
quickly, earning a low (anomalous) score.

---

## Research

This project doubles as an active research effort on *why* unsupervised detectors plateau
and how to push past it:

- **`paper.md`** — working paper: "anomalous ≠ malicious," the failure decomposition, and the
  representation-bottleneck experiment.
- **`diagnose_overlap.py`** — the core diagnostic: benign vs malicious score distributions,
  with AUC and the KS-distance ceiling.
- **`RESEARCH.md`** — four directions (SHAP explainability, benchmarking, federated, streaming).
- **`plan.md`** — the 1-month paper plan.

---

## How Isolation Forest Works

1. Builds `n_estimators` random trees, each on a random feature subset.
2. At each node, picks a random feature and a random split value.
3. **Anomalous points isolate near the root** — they need fewer splits.
4. Anomaly score = average path length across trees (normalised); short path → low score → flagged.

`explain_isolation_forest.py` walks through this on a 13-point toy dataset with full
visualisation of every tree split.

---

## File Structure

```
anomaly-detection/
│
├── agent.py                   ← SecureAI Agent (OpenAI tool-calling loop)
├── cve_db.py                  ← CVE + MITRE ATT&CK knowledge base
│
├── run_ctu13.py               ← Detection engine (10 features, 6K rows/class)
├── run_ctu13_v2.py            ← All-features version
├── compare_datasets.py        ← Attack vs Normal comparison (20K rows each)
├── diagnose_overlap.py        ← Score-overlap diagnosis (AUC + KS ceiling)
├── explain_isolation_forest.py← IF explainability on toy data
│
├── graphs/                    ← Output PNGs
├── requirements.txt
├── paper.md                   ← Draft research paper
├── plan.md                    ← 1-month paper plan
├── RESEARCH.md                ← Research directions
└── PROGRESS.md                ← Experiment log
```

---

## Dataset

**CTU-13** — Sebastián García, Martin Grill, Jan Stiborek, Alejandro Zunino.
*"An empirical comparison of botnet detection methods"*, Computers & Security, 2014.
[stratosphereips.org/datasets-ctu13](https://www.stratosphereips.org/datasets-ctu13)

13 scenarios of real botnet traffic (Neris, Rbot, Menti, Sogou, Murlo, NSIS.ay) captured
on the CTU university network, mixed with normal campus background traffic.

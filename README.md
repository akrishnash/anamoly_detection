# Network Anomaly Detection — CTU-13 Botnet Dataset

> **Unsupervised anomaly detection** on real-world botnet traffic using **Isolation Forest**.
> Validated on the CTU-13 dataset (38,898 attack flows + 53,314 normal flows).
> Includes a dataset comparison pipeline, explainability walkthrough, and an optional AI agent layer.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/akrishnash/anamoly_detection.git
cd anamoly_detection
pip install -r requirements.txt
```

### 2. Download the CTU-13 dataset

Place both CSVs in `sample_logs/`:

```
sample_logs/
├── CTU13_Attack_Traffic.csv    # 38,898 botnet flows
└── CTU13_Normal_Traffic.csv    # 53,314 benign campus flows
```

Download source: [imfaisalmalik/CTU13-CSV-Dataset](https://github.com/imfaisalmalik/CTU13-CSV-Dataset)

### 3. Run the scripts

```bash
# Main detector — 10-feature Isolation Forest on CTU-13 (6K rows each)
python run_ctu13.py

# All-features version — all CICFlowMeter columns + log1p transform
python run_ctu13_v2.py

# Attack vs Normal comparison — 20K rows each, Cohen's d, ROC, PR, threshold sweep
python compare_datasets.py

# Isolation Forest explainability walkthrough (tiny 13-point dataset)
python explain_isolation_forest.py

# SecureAI Agent — GPT-4o tool-calling loop over the detector (requires OpenAI key)
export OPENAI_API_KEY=sk-...
python agent.py
```

All graphs are saved to the `graphs/` folder.

---

## Results

### run_ctu13.py  (6,000 flows per class, 10 features)

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

### compare_datasets.py  (20,000 flows per class, same 10 features)

| Metric | Value |
|---|---|
| Attack Precision | 0.637 |
| Attack Recall | 0.509 |
| ROC-AUC | **0.706** |
| Avg Precision | 0.612 |

---

## Why Attack Traffic Has Low Byte Rates (and Still Signals Attack)

CTU-13 is predominantly **botnet C&C traffic** — deliberately stealthy and low-volume.
The median byte rate for attack flows is near zero. This is not "nothing happening":
it is suspiciously *too quiet* for real user traffic.

The actual discriminating signals (measured by Cohen's d effect size):

| Feature | Cohen's d | Direction | Meaning |
|---|---|---|---|
| SYN Flag Count | **+0.86** | Attack > Normal | Constant connection attempts = port scans / C&C setup |
| Packet Rate | **+0.63** | Attack > Normal | Bimodal: near-zero beacons AND DDoS burst spikes |
| Fwd Pkts/s | **+0.63** | Attack > Normal | High forward rate in scanning phases |
| FIN Flag Count | **+0.31** | Attack > Normal | Many abruptly closed connections (scan and move on) |
| Pkt Len Mean | **-0.19** | Normal > Attack | Normal has larger packets (HTTP content, file data) |
| Bwd Bytes | **-0.02** | Normal > Attack | C&C victims reply with near-empty ACKs |

Isolation Forest detects anomalies in the **joint feature space** — a flow with near-zero bytes
+ elevated SYN + specific timing occupies an isolated region that a random tree cuts off
very quickly, earning a low (anomalous) score.

---

## File Structure

```
anamoly_detection/
│
├── run_ctu13.py               ← Main detector  (10 features, 6K rows per class)
├── run_ctu13_v2.py            ← All-features version (all CICFlowMeter columns)
├── compare_datasets.py        ← Attack vs Normal comparison (20K rows each)
├── explain_isolation_forest.py← Step-by-step IF explainability on toy data
│
├── agent.py                   ← SecureAI Agent (OpenAI tool-calling loop)
├── cve_db.py                  ← CVE + MITRE ATT&CK knowledge base
│
├── sample_logs/
│   ├── CTU13_Attack_Traffic.csv
│   └── CTU13_Normal_Traffic.csv
│
├── graphs/                    ← All output PNGs saved here
│   ├── anomaly_report_ctu13.png
│   ├── ctu13_all_features.png
│   ├── ctu13_comparison.png
│   └── isolation_forest_explained.png
│
├── requirements.txt
├── RESEARCH.md                ← 4 research directions to elevate this project
├── PROGRESS.md                ← Experiment log and next steps
└── paper.md                   ← Draft paper / write-up
```

---

## Architecture

```
CTU-13 CSV (attack + normal)
          │
          ▼
    load_and_merge()           sample N rows per class, assign true_label
          │
          ▼
    featurise()                map CICFlowMeter columns → 10 feature cols
          │
          ▼
  StandardScaler + IsolationForest (n_estimators=200, contamination=0.40)
          │
          ├── score_samples()  → continuous IF score (lower = more anomalous)
          └── fit_predict()    → binary flag (-1 anomaly / +1 normal)
                    │
                    ▼
            classify()         heuristic rules (z-score per feature)
                    │
                    ▼
          Console report + 4-panel PNG  (saved to graphs/)
```

---

## How Isolation Forest Works

1. Builds `n_estimators` random trees, each grown on a random feature subset.
2. At each node it picks a random feature and a random split value.
3. **Anomalous points are isolated near the root** — they need fewer splits.
4. The anomaly score = average path length across all trees (normalised).
5. Points with short average path length get a low (negative) score → flagged.

The `explain_isolation_forest.py` script walks through this on a 13-point toy dataset
with full visualisation of every tree split.

---

## SecureAI Agent (Optional)

`agent.py` wraps the detector in a GPT-4o tool-calling loop that:

- Calls `analyze_traffic()` to run the IF detector and return anomaly episodes
- Calls `lookup_cve()` to enrich each finding with CVEs and MITRE ATT&CK techniques
- Returns a structured analyst-grade threat report

Requires `OPENAI_API_KEY`. The agent is **advisory only** — it never takes automated
action. Every tool call is logged. Human review is always the final step.

**Local / air-gapped mode** — swap two lines in `agent.py`:

```python
# Cloud
client = OpenAI()
MODEL  = "gpt-4o"

# Local via Ollama (no internet required)
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL  = "llama3.1"
```

---

## Dataset

**CTU-13** — Sebastian Garcia, Martin Grill, Jan Stiborek, Alejandro Zunino.
*"An empirical comparison of botnet detection methods"*, Computers & Security, 2014.
[https://www.stratosphereips.org/datasets-ctu13](https://www.stratosphereips.org/datasets-ctu13)

13 scenarios of real botnet traffic (Neris, Rbot, Menti, Sogou, Murlo, NSIS.ay botnets)
captured on the CTU university network, mixed with normal campus background traffic.

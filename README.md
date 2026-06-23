# SecureAI Agent — Network Threat Triage

> **AI-powered cybersecurity agent** that ingests PCAP captures and network logs,
> detects anomalies with Isolation Forest, and returns grounded threat explanations,
> CVE context, and analyst-grade mitigations — via an OpenAI tool-calling loop.

---

## Architecture

```
User Query ("analyse capture.pcap for threats")
          │
          ▼
  ┌─────────────────────────────────────┐
  │          agent.py (GPT-4o)          │
  │   OpenAI tool-calling loop          │
  └────────┬──────────────┬────────────┘
           │              │
           ▼              ▼
  analyze_traffic()   lookup_cve()
           │              │
           ▼              ▼
  ┌─────────────────┐  ┌──────────────────────┐
  │ Isolation Forest│  │ CVE + MITRE ATT&CK   │
  │ anomaly detector│  │ static knowledge base │
  │ (anomaly_       │  │ (cve_db.py)          │
  │  detector_v2.py)│  └──────────────────────┘
  └─────────────────┘
           │
           ▼
  ┌─────────────────────────────────────┐
  │  Structured Threat Report           │
  │  • Executive summary                │
  │  • Findings table (IP, type, sev.)  │
  │  • CVEs + ATT&CK per attack type    │
  │  • Prioritised mitigations          │
  └─────────────────────────────────────┘
```

The agent is **bounded** — it analyses and advises; it never takes automated action.
Every tool call is logged. Human review is always the final step.

---

## Tools

| Tool | What it does |
|---|---|
| `analyze_traffic(file, contamination, window)` | Runs Isolation Forest on a PCAP or CSV log. Returns anomaly episodes with type, severity, source IP, IF score, and traffic stats. |
| `lookup_cve(attack_type)` | Returns CVEs, MITRE ATT&CK technique, and mitigations for a detected attack type. |

---

## Running with a Local Model (Air-Gapped / Offline)

The agent uses the OpenAI-compatible API, so swapping to a local model via
[Ollama](https://ollama.com) requires two lines in `agent.py`:

```python
# Default — OpenAI cloud
client = OpenAI()
MODEL  = "gpt-4o"

# Local model via Ollama (no internet required)
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL  = "llama3.1"  # or qwen2.5, mistral-nemo
```

Everything else — tool definitions, the calling loop, CVE enrichment — works unchanged.

```bash
# Install and run Ollama
ollama pull llama3.1
ollama serve

# Run the agent fully offline
python agent.py capture.pcap
```

**Model compatibility for tool calling:**

| Model | Tool calling | Notes |
|---|---|---|
| `llama3.1` | Reliable | Recommended for local use |
| `qwen2.5` | Reliable | Strong at structured output |
| `mistral-nemo` | Good | Lighter, faster |
| `phi3` / `phi3.5` | Unreliable | Often ignores tool call schema |

This means the same agent code works in a cloud environment (GPT-4o) *and* in an
air-gapped network with a local model — no code changes, no internet dependency.

---

## Detection Engine

The underlying detector (`anomaly_detector_v2.py`) uses **scikit-learn Isolation Forest** —
fully unsupervised, no labelled data required. Validated on real botnet traffic:

| Dataset | Precision | Recall | Notes |
|---|---|---|---|
| CTU-13 (Neris, Rbot, Menti botnets) | 69.5 % | 55.5 % | Zero labels used during training |

**Detected anomaly types:**

| Type | Detection Signal |
|---|---|
| Port Scan | Many distinct destination ports from one source IP |
| Brute Force | High failed-connection ratio, single target port |
| Traffic Spike / DDoS | Z-score spike on byte-rate per time window |
| Data Exfiltration | Sustained high outbound volume (rolling Z-score) |
| Night Activity | Unusual connection count during 00:00–05:59 |
| Botnet C&C | Long idle flows, low packet rate, irregular timing |

---

## Quick Start

```bash
git clone https://github.com/akrishnash/anamoly_detection.git
cd anamoly_detection
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...

# Analyse a log file
python agent.py sample_logs/network_traffic.log

# Analyse a PCAP
python agent.py capture.pcap --contamination 0.03

# Interactive chat mode
python agent.py --interactive
```

### Generate sample data

```bash
python generate_sample_log.py       # creates sample_logs/network_traffic.log
python generate_sample_pcap.py      # creates sample_logs/network_traffic.pcap
```

---

## Sample Output

```
================================================================
  SecureAI Agent  —  Cybersecurity Triage
  Isolation Forest + GPT-4o Tool Calling
================================================================

## Threat Report — sample_logs/network_traffic.log

### Executive Summary
Analysis of 7,722 network records spanning 2024-01-15 identified **5 anomaly
episodes** across 4 distinct attack types. Two critical-severity events require
immediate investigation: a data exfiltration attempt from 192.168.1.25 and an
off-hours traffic spike from 192.168.1.50.

### Findings

| # | Time  | Source IP       | Type               | Severity |
|---|-------|-----------------|-------------------|----------|
| 1 | 19:43 | 192.168.1.25    | Data Exfiltration  | CRITICAL |
| 2 | 02:20 | 192.168.1.50    | Night Activity     | CRITICAL |
| 3 | 10:28 | 45.33.32.156    | Traffic Spike      | HIGH     |
| 4 | 16:27 | 192.168.200.1   | Brute Force        | HIGH     |
| 5 | 13:58 | 192.168.100.50  | Port Scan          | HIGH     |

### CVE Context

Data Exfiltration — MITRE T1048
Relevant CVEs: CVE-2023-0669 (GoAnywhere MFT RCE, CVSS 7.2),
CVE-2021-26855 (ProxyLogon, CVSS 9.8)

Brute Force — MITRE T1110
Relevant CVEs: CVE-2023-38408 (OpenSSH RCE, CVSS 9.8),
CVE-2019-0708 (BlueKeep RDP, CVSS 9.8)

### Prioritised Mitigations
1. [CRITICAL] Block and investigate 192.168.1.25 — egress >500 MB in single window
2. [CRITICAL] Review off-hours activity from 192.168.1.50 — possible C2 beacon
3. [HIGH] Enforce account lockout on SSH/RDP targets of 192.168.200.1
```

---

## File Structure

```
anamoly_detection/
│
├── agent.py                   ← SecureAI Agent (OpenAI tool-calling loop)
├── cve_db.py                  ← CVE + MITRE ATT&CK knowledge base
│
├── anomaly_detector_v2.py     ← Isolation Forest detection engine (PCAP + CSV)
├── anomaly_detector.py        ← v1 statistical detector (Z-score)
├── run_ctu13.py               ← Real-world CTU-13 botnet evaluation runner
│
├── generate_sample_log.py     ← Generates synthetic log with 5 planted anomalies
├── generate_sample_pcap.py    ← Generates synthetic PCAP
│
├── requirements.txt
│
├── anomaly_report_v2.png      ← Sample 4-panel detection graph (CSV input)
├── anomaly_report_pcap.png    ← Sample detection graph (PCAP input)
└── anomaly_report_ctu13.png   ← CTU-13 real botnet detection results
```

---

## Design Choices

**Why Isolation Forest?** Fully unsupervised — no labelled attack data required.
Works on any network environment without retraining. The 55% recall on CTU-13 botnet
traffic (zero labels) demonstrates real-world utility.

**Why bounded tool calling?** The agent loop is deliberately constrained: it can read
files and query a knowledge base, nothing else. No automated firewall rules, no ticket
creation, no blocking. This is an *advisory* agent — the analyst stays in the loop.

**Why save reports to disk?** Every run produces a timestamped Markdown file. In
regulated environments audit trails are non-negotiable.

---

## Dataset Credit

- **CTU-13**: Sebastián García, Martin Grill, Jan Stiborek, Alejandro Zunino.
  *"An empirical comparison of botnet detection methods"*, Computers & Security, 2014.
  [https://www.stratosphereips.org/datasets-ctu13](https://www.stratosphereips.org/datasets-ctu13)

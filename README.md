# ML-Based Network Intrusion Detection System (NIDS)

**Author:** Krishkumar (2401CS83)

## Overview

A machine learning system that classifies network traffic flows as benign or
as a specific attack type (DoS, Port Scan, Brute Force, Botnet), using
flow-based statistical features. Built as an end-to-end project: dataset →
trained model → API → simulated live traffic → dashboard.

## Problem

Traditional signature-based intrusion detection can't catch novel or slightly
modified attacks. A model trained on flow statistics (packet timing, size,
rate) can generalize better to attack patterns it hasn't seen in that exact
form before.

## Planned Architecture

```
Dataset
  |
Cleaning
  |
Preprocessing
  |
ML Model (Random Forest / XGBoost)
  |
FastAPI inference service
  |
Simulated live traffic (pcap replay)
  |
Streamlit dashboard
```

## Current Status: Day 1-2 (Setup + Dataset + EDA)

Implemented so far:
- Project structure and configuration
- Dataset loading (`src/data/load_dataset.py`)
- Exploratory data analysis: missing values, infinite values, duplicates,
  class distribution, correlation analysis, leakage-column detection
  (`src/data/eda.py`)
- Generated reports: `reports/EDA_REPORT.md`, `reports/eda_summary.json`
- Generated plots: `reports/figures/`

Not yet implemented (see project plan for Day 3-21): preprocessing pipeline,
model training, FastAPI service, live-replay demo, Streamlit dashboard.

## Dataset

**Primary choice: CICIDS2017** (Canadian Institute for Cybersecurity).
Use the daily CSV subset (e.g. Wednesday or Friday), not the full multi-GB
archive, to keep Day 1-2 fast.

- Source: obtain from the official CICIDS2017 distribution (search
  "CICIDS2017 dataset download" — the download link changes hosts over
  time, so get it from the current official source rather than a fixed URL).
- Expected files: one or more `*.csv` files such as
  `Wednesday-workingHours.pcap_ISCX.csv`
- Expected location: `data/raw/`
- Approx. disk space: ~250-400 MB per daily CSV
- Verify: after placing the file, run `python scripts/run_eda.py` — if it
  loads and prints a row/column count, the file is good.

**Fallback: NSL-KDD** — smaller and cleaner if CICIDS2017 gives you loading
trouble. If you switch, update `dataset.name` and `dataset.target_column`
in `configs/config.yaml` (NSL-KDD's label column is typically named
differently — check the file header and set it there).

**Smoke-test option:** before downloading the real dataset, run
`python scripts/generate_sample_data.py` to generate a small synthetic
CICIDS-shaped CSV in `data/raw/`. This lets you confirm the whole pipeline
runs before committing to a multi-hundred-MB download. Delete the generated
file once you have real data.

## Installation

```
git clone <your-repo-url>
cd nids-project
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Project Structure

```
nids-project/
├── data/
│   ├── raw/            # original CSVs, never modified
│   ├── interim/        # intermediate cleaned data (Day 3+)
│   └── processed/      # final model-ready data (Day 3+)
├── notebooks/
├── src/
│   ├── data/           # load_dataset.py, eda.py
│   ├── preprocessing/  # Day 3-4
│   ├── models/         # Day 5-7
│   ├── api/            # Day 8-9
│   ├── dashboard/      # Day 13-15
│   └── utils/          # paths.py, logging_config.py
├── scripts/
│   ├── run_eda.py
│   └── generate_sample_data.py
├── tests/
├── configs/
│   └── config.yaml
├── models/             # saved model files (Day 5+)
├── reports/
│   ├── figures/
│   └── metrics/
├── logs/
├── README.md
├── requirements.txt
└── .gitignore
```

## Running Day 1-2

```
# optional: generate fake data to smoke-test first
python scripts/generate_sample_data.py

# place real CICIDS2017 CSV(s) in data/raw/, then:
python scripts/run_eda.py

# run tests
pytest tests/
```

Outputs land in `reports/EDA_REPORT.md`, `reports/eda_summary.json`, and
`reports/figures/`.

## Future Work

- Day 3-4: preprocessing pipeline (cleaning, encoding, scaling, class balancing)
- Day 5-7: train Random Forest / XGBoost, evaluate per-class precision/recall/F1
- Day 8-9: FastAPI inference service
- Day 10-12: pcap-replay live-traffic simulation
- Day 13-15: Streamlit dashboard
- Day 16-18: documentation polish, demo recording
- Day 19-21: buffer, optional SHAP explainability / alerting

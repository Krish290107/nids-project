# Network Intrusion Detection System (NIDS)

A machine-learning system that watches network traffic and says whether it is normal or an attack, and if it is an attack, *which kind*. I built it end to end as a student project at IIT Patna: data analysis, model training, a live prediction API, a dashboard that shows alerts as they arrive, and explanations of *why* a flow was flagged.

![Architecture](docs/architecture.png)

## Results

<!-- RESULTS:START -->
Results appear here once you have trained the model. They are also in `reports/MODEL_REPORT.md` and on the dashboard's **Model Performance** tab.
<!-- RESULTS:END -->

Near-perfect scores are normal on this dataset because DoS attacks leave very distinctive traffic patterns. They show the pipeline is sound on this data, not that the model would work on another network (see [Limitations](#limitations)).

## How it works

Traffic is summarised as **flows**: one flow is one conversation between two computers, described by about 78 numbers (duration, packets per second, average packet size, and so on). The model learns which numbers look like which kind of traffic.

1. **Look first (EDA).** Count missing values, infinities, duplicates and class sizes. This only reads the data.
2. **Prepare it carefully.** Remove unwanted classes and duplicate rows, split 80/20, fill gaps with medians, drop near-duplicate columns, scale the numbers, and balance rare attack classes with **SMOTE** (it creates extra synthetic examples). The golden rule: everything *learned* from data (medians, scaler, SMOTE) is learned from the training part only, and the test part is never touched. Otherwise test information leaks into training and the scores look better than they really are.
3. **Train two models and pick the better one.** Random Forest and XGBoost are compared on **macro F1**, which counts every class equally. Plain accuracy would hide a missed rare attack, because most traffic is normal.
4. **Serve it.** A FastAPI service loads the model and the saved preparation steps, so live flows are treated exactly like training flows. Every prediction is logged to a small SQLite database.
5. **Replay traffic.** Real packet sniffing is unreliable, so held-out test flows are sent to the API one at a time, like live traffic.
6. **Watch and explain.** A Streamlit dashboard reads the log. **SHAP** shows which features pushed a flow toward its predicted class, and the contributions add up exactly to the model's output.

## Dataset

[CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) (Canadian Institute for Cybersecurity), the **Wednesday** file: 692,703 flows, 79 columns. It holds normal traffic, four DoS attack types (Hulk, GoldenEye, slowloris, Slowhttptest) and **Heartbleed**.

**Heartbleed is removed.** The file has only 11 Heartbleed flows, too few to train on or test on honestly (a "perfect" score on 2 test flows means nothing). It is excluded by one line in `configs/config.yaml`: `drop_classes: [Heartbleed]`. Set it to `[]` to keep it. The EDA report describes the raw file, which is where you can still see those 11 rows.

## Setup

You need **Python 3.12 or newer** from [python.org](https://www.python.org/downloads/) (tick **Add python.exe to PATH** when installing; tested on 3.12 and 3.14), an internet connection to install the packages, and a few minutes for training.

**1. Install**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
On Mac/Linux use `source venv/bin/activate`. The virtual environment is optional: if it won't create, run `python -m pip install -r requirements.txt` and skip the activate line. If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first.

**2. Get the data.** From the [CICIDS2017 page](https://www.unb.ca/cic/datasets/ids-2017.html) download `MachineLearningCSV.zip` (you may need to fill in a short form). Put **only the Wednesday CSV** in `data/raw/`. Every CSV in that folder is loaded, so extra files change the results. No data yet? `python scripts/generate_sample_data.py` makes a small fake CSV to try things out; delete it before using real data.

**3. Check it works:** `pytest tests/` should end with `passed` and no failures.

## Use

**Build the model (once).**
```
python scripts/run_eda.py            # optional: reports on the raw data
python scripts/run_preprocessing.py  # clean, split, scale, balance
python scripts/run_training.py       # train both models, keep the best
python scripts/update_readme.py      # optional: write your results into this README
```

**Run the live system.** Open three terminals (each with the environment activated):
```
python scripts/run_api.py                                    # API docs: http://127.0.0.1:8000/docs
python scripts/replay_traffic.py --count 200 --interval 0.1  # send test flows to the API
python scripts/run_dashboard.py                              # dashboard: http://localhost:8501
```
`python scripts/run_demo.py` starts all three together; Ctrl+C stops them. On the dashboard, tick **Auto-refresh** to watch alerts arrive.

**Explain predictions.** `python scripts/explain_flow.py` prints, for one example flow per class, which features pushed the model to its answer, and saves a chart to `reports/figures/`. Use `--class "DoS Hulk"` for one class.

**API endpoints** (try them on the `/docs` page):

| Endpoint | What it does |
|---|---|
| `GET /health` | Is the service up and a model loaded |
| `GET /model-info` | Classes and the exact feature names the model needs |
| `POST /predict` | Classify one flow: `{"features": {...}}` |
| `POST /predict/batch` | Classify up to 1000 flows: `{"flows": [...]}` |
| `POST /explain` | Classify one flow and show what drove the answer (SHAP) |
| `GET /predictions/recent`, `GET /stats` | The logged predictions and counts per class |

Missing features give a clear 422 error; missing or infinite values are filled with training medians. `python scripts/test_api_client.py` sends real test flows to the API and checks it agrees with the model run offline.

**Settings** live in `configs/config.yaml`. Top-level names (`preprocessing:`, `training:`) must start at the left edge, with their settings indented underneath.

| Setting | Meaning |
|---|---|
| `preprocessing.drop_classes` | Classes to remove before training |
| `preprocessing.balancing_method` | `smote`, `class_weight` or `none` |
| `training.max_train_rows` | Set e.g. `300000` if training is slow or runs out of memory |
| `training.selection_metric` | How the best model is chosen (`macro_f1` by default) |
| `replay.interval_seconds`, `api.port`, `dashboard.port` | Replay speed and ports |

## Troubleshooting

| Problem | Fix |
|---|---|
| `Error: [WinError 2]` when creating the venv | Delete the half-made `venv` folder and try `py -m venv venv`. If it still fails, reinstall Python from python.org (a Microsoft Store or damaged install can cause it), or skip the venv |
| `KeyError: 'training'` (or `'api'`, `'replay'`) | That section is missing or indented in `configs/config.yaml` |
| `Required file not found ... model.pkl` | Run `run_preprocessing.py`, then `run_training.py` |
| Results contain fake classes (BENIGN, PortScan, Botnet...) | Delete the sample CSV from `data/raw/` and rerun |
| Dashboard says "No predictions logged yet" | Start the API, then run `replay_traffic.py` |
| Dashboard charts or `/docs` are blank | They load libraries from the internet; connect and refresh |
| `Address already in use` | An old run is still going; close it or change the port in the config |
| Training is very slow or out of memory | Set `training.max_train_rows: 300000` |

## Project layout

```
configs/config.yaml    every path and setting
data/                  raw/ (put the CSV here), processed/ (created for you)
models/                saved model and preprocessing steps (created for you)
reports/               EDA and model reports, figures, metrics
scripts/               the commands above, plus test_api_client.py and generate_sample_data.py
src/                   data/, preprocessing/, models/, api/, dashboard/, utils/
tests/                 pytest tests for all of it
```

## Limitations

- **One day of one dataset.** It has not been tested on other days, other attacks or a real network.
- **The best model is chosen on the same test set it is scored on** (no separate validation split), which adds a small optimistic bias.
- **The replay is a simulation.** There is no packet capture or flow extractor; it streams recorded flow records.
- **SHAP explains the model, not the cause of an attack.** Explaining one flow takes longer on a bigger forest.
- **The API is for local use:** it listens on 127.0.0.1 and has no authentication.

## Credits

Author: Krishkumar (2401CS83), B.Tech CSE, IIT Patna. Dataset: Sharafaldin, Lashkari and Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", ICISSP 2018.

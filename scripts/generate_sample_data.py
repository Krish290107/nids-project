"""Generate a small, fake CICIDS2017-shaped CSV so you can smoke-test the
Day 1-2 pipeline before downloading the real (multi-GB) dataset.

This is NOT real network data — it exists only to prove load_dataset.py and
eda.py run correctly end-to-end on your machine. Delete it once the real
dataset is in place.

Run from the project root:
    python scripts/generate_sample_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.utils.paths import resolve_path, ensure_dir

N_ROWS = 5000
RNG = np.random.default_rng(42)

CLASSES = ["BENIGN", "DoS", "PortScan", "BruteForce", "Botnet"]
CLASS_WEIGHTS = [0.80, 0.08, 0.06, 0.04, 0.02]  # realistic-ish imbalance


def make_sample() -> pd.DataFrame:
    n = N_ROWS
    labels = RNG.choice(CLASSES, size=n, p=CLASS_WEIGHTS)

    df = pd.DataFrame({
        "Flow ID": [f"192.168.1.{i%255}-10.0.0.{i%255}-{1000+i%500}-80-6" for i in range(n)],
        "Source IP": [f"192.168.1.{i % 255}" for i in range(n)],
        "Source Port": RNG.integers(1024, 65535, n),
        "Destination IP": [f"10.0.0.{i % 255}" for i in range(n)],
        "Destination Port": RNG.choice([80, 443, 22, 21, 3389], n),
        "Protocol": RNG.choice([6, 17], n),
        "Timestamp": pd.date_range("2017-07-05", periods=n, freq="s").astype(str),
        "Flow Duration": RNG.exponential(scale=100000, size=n),
        "Total Fwd Packets": RNG.poisson(10, n),
        "Total Backward Packets": RNG.poisson(8, n),
        "Total Length of Fwd Packets": RNG.exponential(500, n),
        "Total Length of Bwd Packets": RNG.exponential(400, n),
        "Fwd Packet Length Max": RNG.exponential(200, n),
        "Fwd Packet Length Mean": RNG.exponential(100, n),
        "Bwd Packet Length Max": RNG.exponential(180, n),
        "Bwd Packet Length Mean": RNG.exponential(90, n),
        "Flow Bytes/s": RNG.exponential(1000, n),
        "Flow Packets/s": RNG.exponential(50, n),
        "Flow IAT Mean": RNG.exponential(1000, n),
        "Flow IAT Std": RNG.exponential(500, n),
        "Label": labels,
    })

    # Duplicate a few rows, like real captures sometimes have.
    dup_idx = RNG.choice(n, size=20, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # Inject some missing values.
    for col in ["Flow Bytes/s", "Fwd Packet Length Mean"]:
        na_idx = RNG.choice(df.index, size=15, replace=False)
        df.loc[na_idx, col] = np.nan

    # Inject some infinite values, mimicking division-by-zero in real flow stats.
    inf_idx = RNG.choice(df.index, size=10, replace=False)
    df.loc[inf_idx, "Flow Bytes/s"] = np.inf

    # A near-duplicate column pair, to test correlation detection.
    df["Fwd Packet Length Mean 2"] = df["Fwd Packet Length Mean"] * 1.001

    return df


def main() -> None:
    raw_dir = ensure_dir(resolve_path("data/raw"))
    out_path = raw_dir / "sample_day01_wednesday.csv"
    df = make_sample()
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} synthetic rows to {out_path}")
    print("This is fake data for pipeline testing only — replace with real CICIDS2017 CSVs.")


if __name__ == "__main__":
    main()

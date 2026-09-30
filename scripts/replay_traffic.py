"""Day 10-12: simulated live traffic.

The API only ever sees one flow at a time, arriving on a delay, the same
shape of input it would get from a real flow extractor. This is deliberately
NOT real packet sniffing: sniffing live traffic during a demo is unreliable
(permissions, timing, network conditions), while replaying held-out test
flows gives a genuine live-style demo every time.

Terminal 1:  python scripts/run_api.py
Terminal 2:  python scripts/replay_traffic.py

Options:
    --count 300          how many flows to send (default from config)
    --interval 0.2        seconds between flows (default from config)
    --loop                stream forever, cycling through the flows, until Ctrl+C
    --url http://...      override the API address
"""
import argparse
import itertools
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.api.replay_utils import format_console_line, prepare_replay_data, summarize_session
from src.utils.logging_config import get_logger
from src.utils.paths import ensure_dir, load_config, resolve_path

logger = get_logger(__name__)


def http_post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method="POST", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())
    except urllib.error.URLError:
        raise SystemExit(
            f"Could not reach the API at {url}.\n"
            "Start it first in another terminal:  python scripts/run_api.py"
        )


def main() -> None:
    config = load_config()
    replay_cfg = config["replay"]
    api_cfg = config["api"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=replay_cfg["default_count"])
    parser.add_argument("--interval", type=float, default=replay_cfg["interval_seconds"])
    parser.add_argument("--loop", action="store_true", help="stream forever until Ctrl+C")
    parser.add_argument("--url", default=f"http://{api_cfg['host']}:{api_cfg['port']}")
    args = parser.parse_args()

    processed_dir = resolve_path(config["preprocessing"]["processed_dir"])
    artifacts = joblib.load(resolve_path(config["preprocessing"]["artifact_path"]))
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_test = pd.read_csv(processed_dir / "y_test.csv")["label"].to_numpy()

    records = prepare_replay_data(
        X_test, y_test,
        scaler=artifacts["scaler"],
        feature_columns=artifacts["feature_columns"],
        class_names=artifacts["label_encoder"].classes_.tolist(),
        shuffle=replay_cfg["shuffle"],
        random_state=replay_cfg["random_state"],
        count=None if args.loop else args.count,
    )
    logger.info(f"Prepared {len(records)} flow(s) to replay against {args.url}")
    if args.loop:
        logger.info("Looping forever - press Ctrl+C to stop")

    stream = itertools.cycle(records) if args.loop else records
    total_label = "\u221e" if args.loop else str(len(records))  # infinity symbol for --loop

    true_classes, predictions = [], []
    start = time.perf_counter()
    try:
        for i, record in enumerate(stream, start=1):
            prediction = http_post(f"{args.url}/predict", {"features": record["raw_features"]})
            true_classes.append(record["true_class"])
            predictions.append(prediction)
            print(format_console_line(i, total_label, record["true_class"], prediction))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped by user.")

    duration = time.perf_counter() - start
    if not predictions:
        return

    summary = summarize_session(true_classes, predictions, duration)
    print("\n--- Session summary ---")
    for key, value in summary.items():
        print(f"{key}: {value}")

    summary_path = resolve_path(replay_cfg["summary_path"])
    ensure_dir(summary_path.parent)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved {summary_path}")


if __name__ == "__main__":
    main()

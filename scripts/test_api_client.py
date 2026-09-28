"""Try the running API with real flows from your held-out test set.

Terminal 1:  python scripts/run_api.py
Terminal 2:  python scripts/test_api_client.py

For each class it picks a few test flows, converts them back to RAW feature
values (the API expects unscaled data), sends them to /predict/batch, and prints
true class vs predicted class. It also compares the API's answers with the model
run directly on the scaled test data. They should match, which proves the API
preprocesses traffic exactly like training did.

Options:
    --url http://127.0.0.1:8000   where the API is running
    --per-class 3                 how many flows to send per class
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.utils.paths import load_config, resolve_path


def http(method: str, url: str, payload: dict | None = None) -> dict | list:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"API returned {exc.code}: {exc.read().decode()}")
    except urllib.error.URLError:
        raise SystemExit(
            f"Could not reach the API at {url}.\n"
            "Start it first in another terminal:  python scripts/run_api.py"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--per-class", type=int, default=3)
    args = parser.parse_args()

    config = load_config()
    processed = resolve_path(config["preprocessing"]["processed_dir"])
    artifacts = joblib.load(resolve_path(config["preprocessing"]["artifact_path"]))
    bundle = joblib.load(resolve_path(config["training"]["model_path"]))

    print("Health:", http("GET", f"{args.url}/health"))

    X_test = pd.read_csv(processed / "X_test.csv")
    y_test = pd.read_csv(processed / "y_test.csv")["label"]
    class_names = artifacts["label_encoder"].classes_.tolist()

    picked = (
        pd.DataFrame({"idx": X_test.index, "label": y_test})
        .groupby("label", group_keys=False)
        .head(args.per_class)["idx"]
        .tolist()
    )
    scaled = X_test.loc[picked, artifacts["feature_columns"]]

    # undo the scaling so the API receives raw values, like a real flow extractor would send
    raw = pd.DataFrame(
        artifacts["scaler"].inverse_transform(scaled), columns=artifacts["feature_columns"]
    )
    flows = raw.to_dict(orient="records")

    result = http("POST", f"{args.url}/predict/batch", {"flows": flows})

    offline = bundle["model"].predict(scaled.astype("float32"))

    print(f"\n{'true class':<14} {'API prediction':<16} {'confidence':<11} {'attack?':<8} matches offline?")
    print("-" * 70)
    matches = 0
    for idx, pred, off in zip(picked, result["predictions"], offline):
        true_name = class_names[int(y_test.loc[idx])]
        same = pred["predicted_class"] == class_names[int(off)]
        matches += same
        print(
            f"{true_name:<14} {pred['predicted_class']:<16} {pred['confidence']:<11} "
            f"{str(pred['is_attack']):<8} {'yes' if same else 'NO'}"
        )

    print(f"\nAPI matched the offline model on {matches}/{len(picked)} flows.")
    print(f"Attacks flagged by the API: {result['attacks_detected']}/{result['count']}")
    print("Stats so far:", http("GET", f"{args.url}/stats"))


if __name__ == "__main__":
    main()

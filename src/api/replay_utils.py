"""Logic for the Day 10-12 traffic replay, kept separate from the network
loop in scripts/replay_traffic.py so it can be unit tested without a live
server or real time delays.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_replay_data(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    scaler,
    feature_columns: list[str],
    class_names: list[str],
    shuffle: bool,
    random_state: int,
    count: int | None,
) -> list[dict]:
    """Turn the held-out (scaled) test set back into "raw" flows to send to the API,
    the same way a real flow extractor would hand off unscaled numbers.

    Returns a list of {"raw_features": {...}, "true_class": str}, one per flow.
    """
    X = X_test[feature_columns]
    order = np.arange(len(X))
    if shuffle:
        # real traffic isn't neatly grouped by attack type; mixing classes
        # makes the console stream and any live dashboard look realistic
        rng = np.random.default_rng(random_state)
        rng.shuffle(order)

    if count is not None and count < len(order):
        order = order[:count]

    raw_values = scaler.inverse_transform(X.iloc[order])
    raw_df = pd.DataFrame(raw_values, columns=feature_columns)

    records = []
    for pos, idx in enumerate(order):
        records.append(
            {
                "raw_features": raw_df.iloc[pos].to_dict(),
                "true_class": class_names[int(y_test[idx])],
            }
        )
    return records


def format_console_line(seq: int, total: int, true_class: str, prediction: dict) -> str:
    marker = "  *** ATTACK DETECTED ***" if prediction["is_attack"] else ""
    correct = "" if prediction["predicted_class"] == true_class else "  (missed)"
    return (
        f"[{seq:>4}/{total}] true={true_class:<16} pred={prediction['predicted_class']:<16} "
        f"conf={prediction['confidence']:.2f}{correct}{marker}"
    )


def summarize_session(
    true_classes: list[str], predictions: list[dict], duration_seconds: float
) -> dict:
    total = len(predictions)
    attacks = sum(p["is_attack"] for p in predictions)
    correct = sum(t == p["predicted_class"] for t, p in zip(true_classes, predictions))
    per_class: dict[str, int] = {}
    for p in predictions:
        per_class[p["predicted_class"]] = per_class.get(p["predicted_class"], 0) + 1

    return {
        "total_flows": total,
        "attacks_detected": attacks,
        "correct_predictions": correct,
        "accuracy_vs_true_label": round(correct / total, 4) if total else 0.0,
        "predicted_class_counts": per_class,
        "duration_seconds": round(duration_seconds, 2),
        "flows_per_second": round(total / duration_seconds, 2) if duration_seconds > 0 else None,
    }

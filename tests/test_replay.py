import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.api.replay_utils import format_console_line, prepare_replay_data, summarize_session

CLASS_NAMES = ["BENIGN", "DoS"]
FEATURES = ["f1", "f2"]


def make_fixture(n=20):
    rng = np.random.default_rng(0)
    raw = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    scaler = StandardScaler().fit(raw)
    scaled = pd.DataFrame(scaler.transform(raw), columns=FEATURES)
    y = rng.integers(0, 2, size=n)
    return scaled, y, scaler, raw


def test_prepare_replay_data_unscales_back_to_raw_values():
    scaled, y, scaler, raw = make_fixture()
    records = prepare_replay_data(
        scaled, y, scaler, FEATURES, CLASS_NAMES, shuffle=False, random_state=0, count=None
    )
    assert len(records) == len(scaled)
    first = records[0]["raw_features"]
    assert abs(first["f1"] - raw.iloc[0]["f1"]) < 1e-6
    assert records[0]["true_class"] in CLASS_NAMES


def test_prepare_replay_data_respects_count():
    scaled, y, scaler, _ = make_fixture()
    records = prepare_replay_data(
        scaled, y, scaler, FEATURES, CLASS_NAMES, shuffle=False, random_state=0, count=5
    )
    assert len(records) == 5


def test_prepare_replay_data_shuffle_is_reproducible_with_seed():
    scaled, y, scaler, _ = make_fixture()
    a = prepare_replay_data(scaled, y, scaler, FEATURES, CLASS_NAMES, True, 42, None)
    b = prepare_replay_data(scaled, y, scaler, FEATURES, CLASS_NAMES, True, 42, None)
    assert [r["true_class"] for r in a] == [r["true_class"] for r in b]


def test_prepare_replay_data_unshuffled_keeps_order():
    scaled, y, scaler, _ = make_fixture()
    records = prepare_replay_data(scaled, y, scaler, FEATURES, CLASS_NAMES, False, 0, None)
    expected = [CLASS_NAMES[label] for label in y]
    assert [r["true_class"] for r in records] == expected


def test_format_console_line_flags_attack_and_miss():
    attack_line = format_console_line(
        1, "10", "DoS", {"predicted_class": "DoS", "confidence": 0.91, "is_attack": True}
    )
    assert "ATTACK DETECTED" in attack_line and "missed" not in attack_line

    miss_line = format_console_line(
        2, "10", "DoS", {"predicted_class": "BENIGN", "confidence": 0.55, "is_attack": False}
    )
    assert "missed" in miss_line


def test_summarize_session_counts_and_accuracy():
    true_classes = ["BENIGN", "DoS", "DoS"]
    predictions = [
        {"predicted_class": "BENIGN", "is_attack": False, "confidence": 0.9},
        {"predicted_class": "DoS", "is_attack": True, "confidence": 0.8},
        {"predicted_class": "BENIGN", "is_attack": False, "confidence": 0.6},
    ]
    summary = summarize_session(true_classes, predictions, duration_seconds=2.0)
    assert summary["total_flows"] == 3
    assert summary["attacks_detected"] == 1
    assert summary["correct_predictions"] == 2
    assert summary["accuracy_vs_true_label"] == round(2 / 3, 4)
    assert summary["predicted_class_counts"] == {"BENIGN": 2, "DoS": 1}
    assert summary["flows_per_second"] == 1.5


def test_summarize_session_handles_zero_duration():
    summary = summarize_session(["BENIGN"], [{"predicted_class": "BENIGN", "is_attack": False, "confidence": 1.0}], 0.0)
    assert summary["flows_per_second"] is None

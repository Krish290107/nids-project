import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

from src.utils.paths import load_config, resolve_path

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py")


def _paths():
    config = load_config()
    return {
        "db": resolve_path(config["api"]["db_path"]),
        "metrics": resolve_path(config["training"]["metrics_path"]),
    }


def test_app_runs_with_no_data():
    paths = _paths()
    for p in paths.values():  # start from a clean slate, whatever earlier tests left behind
        if p.exists():
            p.unlink()

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert not at.exception
    assert "Network Intrusion Detection" in at.title[0].value
    assert any("No predictions logged yet" in info.value for info in at.info)


def test_app_runs_with_data():
    paths = _paths()
    paths["db"].parent.mkdir(parents=True, exist_ok=True)
    paths["metrics"].parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(paths["db"])
    conn.execute(
        "CREATE TABLE predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, "
        "predicted_class TEXT, confidence REAL, is_attack INTEGER, features_json TEXT)"
    )
    conn.executemany(
        "INSERT INTO predictions (timestamp, predicted_class, confidence, is_attack, features_json) "
        "VALUES (?, ?, ?, ?, ?)",
        [("t", "BENIGN", 0.9, 0, "{}"), ("t", "DoS", 0.8, 1, "{}")],
    )
    conn.commit()
    conn.close()

    metrics = {
        "best_model": "random_forest",
        "selection_metric": "macro_f1",
        "models": {
            "random_forest": {
                "accuracy": 0.95, "macro_f1": 0.9, "weighted_f1": 0.95,
                "per_class": {
                    "BENIGN": {"precision": 0.95, "recall": 0.95, "f1": 0.95, "support": 100}
                },
                "confusion_matrix": [[100]],
            }
        },
        "top_features": {},
    }
    paths["metrics"].write_text(json.dumps(metrics))

    try:
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        assert not at.exception

        metric_values = {m.label: m.value for m in at.metric}
        assert metric_values["Total flows logged"] == "2"
        assert metric_values["Attacks detected"] == "1"
        assert metric_values["Benign"] == "1"
        assert any("random_forest" in md.value for md in at.subheader)
    finally:
        paths["db"].unlink(missing_ok=True)
        paths["metrics"].unlink(missing_ok=True)


def test_app_reruns_cleanly():
    # simulates the sidebar "Refresh now" button: the script must tolerate running twice
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    at.run()
    assert not at.exception

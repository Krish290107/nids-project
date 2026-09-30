import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

import src.utils.paths as paths_module

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py")


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Point the dashboard at a temporary folder so these tests never touch
    your real logs/predictions.db, reports/metrics/*.json or reports/figures/*.

    resolve_path() joins project_root / value, and an absolute value wins, so
    giving the app absolute temp paths redirects every file it reads.
    """
    real_config = paths_module.load_config()
    config = json.loads(json.dumps(real_config))  # deep copy
    config["api"]["db_path"] = str(tmp_path / "predictions.db")
    config["training"]["metrics_path"] = str(tmp_path / "model_metrics.json")
    config["replay"]["summary_path"] = str(tmp_path / "replay_session_summary.json")
    config["reports"]["figures_dir"] = str(tmp_path / "figures")
    monkeypatch.setattr(paths_module, "load_config", lambda *a, **k: config)
    return tmp_path


def _write_db(path: Path) -> None:
    conn = sqlite3.connect(path)
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


def _metric_values(at) -> dict:
    # labels compared case-insensitively so restyling the wording doesn't break the test
    return {m.label.lower(): m.value for m in at.metric}


def test_app_runs_with_no_data(sandbox):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert not at.exception
    assert "Network Intrusion Detection" in at.title[0].value
    assert any("No predictions logged yet" in info.value for info in at.info)


def test_app_runs_with_data(sandbox):
    _write_db(sandbox / "predictions.db")
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
    (sandbox / "model_metrics.json").write_text(json.dumps(metrics))

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    values = _metric_values(at)
    assert values["total flows logged"] == "2"
    assert values["attacks detected"] == "1"
    assert values["benign"] == "1"
    # best model name is shown in a styled badge (markdown); the per-class table has its own subheader
    assert any("random_forest" in md.value for md in at.markdown)
    assert any("Per-Class Results" in sh.value for sh in at.subheader)


def test_app_reruns_cleanly(sandbox):
    # simulates the sidebar "Refresh now" button: the script must tolerate running twice
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    at.run()
    assert not at.exception


def test_tests_do_not_touch_real_project_files(sandbox):
    real = paths_module.PROJECT_ROOT
    before = {p: p.exists() for p in (real / "logs" / "predictions.db", real / "reports" / "metrics" / "model_metrics.json")}
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    assert before == {p: p.exists() for p in before}

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dashboard.data_access import overall_stats, read_json, recent_predictions_df


def make_db(path: Path, rows: list[tuple]) -> None:
    """rows: (timestamp, predicted_class, confidence, is_attack, features_json)"""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, "
        "predicted_class TEXT, confidence REAL, is_attack INTEGER, features_json TEXT)"
    )
    conn.executemany(
        "INSERT INTO predictions (timestamp, predicted_class, confidence, is_attack, features_json) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_recent_predictions_df_missing_db_returns_empty(tmp_path):
    df = recent_predictions_df(tmp_path / "nope.db", limit=10)
    assert df.empty
    assert list(df.columns) == ["id", "timestamp", "predicted_class", "confidence", "is_attack"]


def test_recent_predictions_df_reads_rows_newest_first(tmp_path):
    db = tmp_path / "preds.db"
    make_db(db, [
        ("2026-01-01T00:00:00", "BENIGN", 0.9, 0, "{}"),
        ("2026-01-01T00:00:01", "DoS", 0.8, 1, "{}"),
    ])
    df = recent_predictions_df(db, limit=10)
    assert len(df) == 2
    assert df.iloc[0]["predicted_class"] == "DoS"  # most recent (highest id) first
    assert df["is_attack"].dtype == bool


def test_recent_predictions_df_respects_limit(tmp_path):
    db = tmp_path / "preds.db"
    make_db(db, [("t", "BENIGN", 0.5, 0, "{}") for _ in range(20)])
    df = recent_predictions_df(db, limit=5)
    assert len(df) == 5


def test_overall_stats_missing_db_returns_zeros(tmp_path):
    stats = overall_stats(tmp_path / "nope.db")
    assert stats == {"total": 0, "per_class": {}, "attacks": 0}


def test_overall_stats_counts_correctly(tmp_path):
    db = tmp_path / "preds.db"
    make_db(db, [
        ("t", "BENIGN", 0.9, 0, "{}"),
        ("t", "BENIGN", 0.9, 0, "{}"),
        ("t", "DoS", 0.8, 1, "{}"),
    ])
    stats = overall_stats(db)
    assert stats["total"] == 3
    assert stats["per_class"] == {"BENIGN": 2, "DoS": 1}
    assert stats["attacks"] == 1


def test_reading_does_not_create_db_file(tmp_path):
    db = tmp_path / "should_not_exist.db"
    recent_predictions_df(db, limit=10)
    overall_stats(db)
    assert not db.exists()


def test_read_json_missing_file_returns_none(tmp_path):
    assert read_json(tmp_path / "nope.json") is None


def test_read_json_reads_existing_file(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"a": 1}))
    assert read_json(path) == {"a": 1}

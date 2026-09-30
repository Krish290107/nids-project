"""Read-only data access for the Day 13-15 dashboard.

Everything here reads files that Day 5-7 (training) and Day 8-9 (the API)
already produce. Nothing here writes anything, and nothing here requires
the API to be running - the dashboard reads logs/predictions.db and the
reports/ files directly from disk.

The SQLite connection opens in read-only mode and does NOT create the
database file if it's missing, so simply opening the dashboard before the
API has ever run leaves no trace on disk.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

RECENT_COLUMNS = ["id", "timestamp", "predicted_class", "confidence", "is_attack"]


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _connect_readonly(db_path: Path) -> sqlite3.Connection | None:
    if not Path(db_path).exists():
        return None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def recent_predictions_df(db_path: Path, limit: int = 100) -> pd.DataFrame:
    conn = _connect_readonly(db_path)
    if conn is None:
        return pd.DataFrame(columns=RECENT_COLUMNS)
    try:
        rows = conn.execute(
            "SELECT id, timestamp, predicted_class, confidence, is_attack "
            "FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except sqlite3.OperationalError:
        return pd.DataFrame(columns=RECENT_COLUMNS)
    finally:
        conn.close()

    df = pd.DataFrame([dict(r) for r in rows], columns=RECENT_COLUMNS)
    if not df.empty:
        df["is_attack"] = df["is_attack"].astype(bool)
    return df


def overall_stats(db_path: Path) -> dict:
    conn = _connect_readonly(db_path)
    if conn is None:
        return {"total": 0, "per_class": {}, "attacks": 0}
    try:
        rows = conn.execute(
            "SELECT predicted_class, COUNT(*) AS n, SUM(is_attack) AS attacks "
            "FROM predictions GROUP BY predicted_class"
        ).fetchall()
    except sqlite3.OperationalError:
        return {"total": 0, "per_class": {}, "attacks": 0}
    finally:
        conn.close()

    per_class = {r["predicted_class"]: r["n"] for r in rows}
    attacks = sum((r["attacks"] or 0) for r in rows)
    return {"total": sum(per_class.values()), "per_class": per_class, "attacks": attacks}

"""SQLite log of every prediction the API makes.

Uses only the standard library. A fresh connection is opened per call, which
keeps things safe when FastAPI serves requests from several threads.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    is_attack INTEGER NOT NULL,
    features_json TEXT NOT NULL
)
"""


class PredictionStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(CREATE_TABLE)
        logger.info(f"Prediction log ready: {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def log(self, features: list[dict], predictions: list[dict]) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = [
            (
                now,
                pred["predicted_class"],
                pred["confidence"],
                int(pred["is_attack"]),
                json.dumps(feats),
            )
            for feats, pred in zip(features, predictions)
        ]
        with closing(self._connect()) as conn, conn:
            conn.executemany(
                "INSERT INTO predictions "
                "(timestamp, predicted_class, confidence, is_attack, features_json) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def recent(self, limit: int = 50) -> list[dict]:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "SELECT id, timestamp, predicted_class, confidence, is_attack "
                "FROM predictions ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [{**dict(r), "is_attack": bool(r["is_attack"])} for r in cursor]

    def stats(self) -> dict:
        with closing(self._connect()) as conn:
            per_class = {
                r["predicted_class"]: r["n"]
                for r in conn.execute(
                    "SELECT predicted_class, COUNT(*) AS n FROM predictions GROUP BY predicted_class"
                )
            }
        total = sum(per_class.values())
        return {"total": total, "per_class": per_class}

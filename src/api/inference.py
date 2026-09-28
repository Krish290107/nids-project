"""Turns raw flow features into a prediction.

The steps mirror Day 3-4 exactly, using the artifacts saved there:
    raw features -> inf -> NaN -> fill with TRAIN medians -> pick final columns
                 -> scale with the TRAIN scaler -> model.predict_proba

If this drifts from the training-time steps, predictions get quietly wrong,
which is why the medians, scaler and column order all come from
models/preprocessing_pipeline.joblib rather than being re-derived here.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class MissingFeaturesError(ValueError):
    """Raised when a request doesn't contain every feature the model needs."""

    def __init__(self, missing: list[str]):
        self.missing = missing
        shown = ", ".join(missing[:10])
        more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        super().__init__(f"Missing {len(missing)} required feature(s): {shown}{more}")


class InferenceService:
    def __init__(self, model_bundle: dict, preprocessing: dict, benign_label: str = "BENIGN"):
        self.model = model_bundle["model"]
        self.model_name: str = model_bundle["model_name"]
        self.feature_columns: list[str] = list(model_bundle["feature_columns"])
        self.class_names: list[str] = list(model_bundle["class_names"])
        self.benign_label = benign_label

        self.scaler = preprocessing["scaler"]
        self.medians: dict = preprocessing["medians"]

        if self.feature_columns != list(preprocessing["feature_columns"]):
            raise ValueError(
                "Feature columns in model.pkl and preprocessing_pipeline.joblib differ. "
                "Re-run 'python scripts/run_preprocessing.py' and then "
                "'python scripts/run_training.py' so both files come from the same run."
            )

    @classmethod
    def from_files(cls, model_path: Path, preprocessing_path: Path, benign_label: str):
        for path, hint in [
            (model_path, "python scripts/run_training.py"),
            (preprocessing_path, "python scripts/run_preprocessing.py"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}\nCreate it with: {hint}")
        service = cls(joblib.load(model_path), joblib.load(preprocessing_path), benign_label)
        logger.info(
            f"Loaded '{service.model_name}' model: {len(service.feature_columns)} features, "
            f"classes {service.class_names}"
        )
        return service

    def prepare(self, rows: list[dict]) -> pd.DataFrame:
        """Raw feature dicts -> scaled DataFrame in the exact column order the model expects."""
        df = pd.DataFrame(rows)

        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise MissingFeaturesError(missing)

        df = df[self.feature_columns].apply(pd.to_numeric, errors="coerce").astype(float)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna({c: self.medians.get(c, 0.0) for c in self.feature_columns})
        df = df.fillna(0.0)  # a median can itself be NaN if a column was all-NaN in training

        scaled = self.scaler.transform(df)
        return pd.DataFrame(scaled, columns=self.feature_columns).astype(np.float32)

    def predict(self, rows: list[dict]) -> list[dict]:
        X = self.prepare(rows)
        probabilities = self.model.predict_proba(X)
        model_classes = [int(c) for c in self.model.classes_]

        results = []
        for row in probabilities:
            best = int(np.argmax(row))
            predicted_class = self.class_names[model_classes[best]]
            results.append(
                {
                    "predicted_class": predicted_class,
                    "confidence": round(float(row[best]), 4),
                    "is_attack": predicted_class.lower() != self.benign_label.lower(),
                    "probabilities": {
                        self.class_names[c]: round(float(p), 4)
                        for c, p in zip(model_classes, row)
                    },
                }
            )
        return results

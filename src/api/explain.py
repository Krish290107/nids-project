"""Explain individual predictions with SHAP (SHapley Additive exPlanations).

For one flow, SHAP answers: "which features pushed the model toward the class it
predicted, and by how much?" Each feature gets a contribution, and

    baseline + sum(all contributions) = the model's output for the predicted class

so the explanation always adds up exactly to the prediction.

* Random Forest: contributions are in probability units, so the right-hand side is
  the model's confidence (0 to 1).
* XGBoost: contributions are in log-odds units (the model's raw score). The
  response says which one it is in `explanation_space`.

SHAP explains what the MODEL does, not what causes an attack. A feature that
pushes toward "DoS Hulk" is something the model relies on, not proof of anything
about the network.
"""
from __future__ import annotations

import numpy as np

from src.api.inference import InferenceService
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_TOP_K = 8


def _to_samples_features_classes(shap_values, n_classes: int) -> np.ndarray:
    """SHAP returns different shapes depending on version and model.
    Normalize to an array of shape (samples, features, classes)."""
    if isinstance(shap_values, list):  # older SHAP: one (samples, features) array per class
        return np.stack(shap_values, axis=-1)
    values = np.asarray(shap_values)
    if values.ndim == 3:
        if values.shape[-1] == n_classes:
            return values
        if values.shape[0] == n_classes:  # (classes, samples, features)
            return np.transpose(values, (1, 2, 0))
    if values.ndim == 2:  # single output: treat as the positive class of a binary model
        return np.stack([-values, values], axis=-1)
    raise ValueError(f"Unexpected SHAP output shape {values.shape} for {n_classes} classes")


class FlowExplainer:
    def __init__(self, service: InferenceService):
        import shap  # imported here so the API starts fast and works without SHAP installed

        self.service = service
        self.explainer = shap.TreeExplainer(service.model)
        self.n_classes = len(service.model.classes_)
        base = np.atleast_1d(np.asarray(self.explainer.expected_value, dtype=float))
        if base.size == 1 and self.n_classes > 1:
            base = np.repeat(base, self.n_classes)
        self.base_values = base
        self.space = "log-odds" if "XGB" in type(service.model).__name__ else "probability"
        logger.info(f"SHAP explainer ready for '{service.model_name}' ({self.space} space)")

    def explain(self, rows: list[dict], top_k: int = DEFAULT_TOP_K) -> list[dict]:
        service = self.service
        scaled = service.prepare(rows)  # same preprocessing as prediction
        probabilities = service.model.predict_proba(scaled)
        contributions = _to_samples_features_classes(self.explainer.shap_values(scaled), self.n_classes)
        model_classes = [int(c) for c in service.model.classes_]
        features = service.feature_columns

        results = []
        for i, row in enumerate(rows):
            class_pos = int(np.argmax(probabilities[i]))
            predicted_class = service.class_names[model_classes[class_pos]]
            values = contributions[i, :, class_pos]

            order = np.argsort(-np.abs(values))
            top = order[:top_k]
            top_features = []
            for j in top:
                raw = row.get(features[j])
                top_features.append(
                    {
                        "feature": features[j],
                        "raw_value": _clean_number(raw),
                        "z_score": round(float(scaled.iloc[i, j]), 3),
                        "contribution": round(float(values[j]), 5),
                        "direction": "toward" if values[j] >= 0 else "against",
                    }
                )

            results.append(
                {
                    "predicted_class": predicted_class,
                    "confidence": round(float(probabilities[i, class_pos]), 4),
                    "is_attack": predicted_class.lower() != service.benign_label.lower(),
                    "explanation_space": self.space,
                    "baseline": round(float(self.base_values[class_pos]), 5),
                    "top_features": top_features,
                    "other_features_contribution": round(float(values[order[top_k:]].sum()), 5),
                }
            )
        return results


def _clean_number(value):
    """Raw value as a plain float, or None if it was missing / not finite / not numeric."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None

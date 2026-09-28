"""Request and response shapes for the NIDS inference API.

FastAPI uses these to validate incoming JSON and to build the /docs page.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Values can be numbers, null (filled with the training medians), or strings for the
# identifier fields a flow extractor sends along (Flow ID, Source IP, ...). Only the
# columns the model needs are read, and anything non-numeric there becomes null.
FeatureValue = Any


class FlowRequest(BaseModel):
    features: dict[str, FeatureValue] = Field(
        ...,
        description="Raw (unscaled) flow features, keyed by the exact CICIDS2017 column names "
        "the model was trained on. Extra keys are ignored.",
    )


class BatchRequest(BaseModel):
    flows: list[dict[str, FeatureValue]] = Field(
        ..., min_length=1, description="One dict of raw features per flow."
    )


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(..., description="Probability of the predicted class (0 to 1).")
    is_attack: bool
    probabilities: dict[str, float]


class BatchResponse(BaseModel):
    count: int
    attacks_detected: int
    predictions: list[PredictionResponse]

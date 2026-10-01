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


class FeatureContribution(BaseModel):
    feature: str
    raw_value: float | None = Field(..., description="The value in the request (null if it was missing or not finite).")
    z_score: float = Field(..., description="How unusual the value is vs. the training data (standard deviations from the mean).")
    contribution: float = Field(..., description="How much this feature moved the output for the predicted class.")
    direction: str = Field(..., description="'toward' or 'against' the predicted class.")


class ExplanationResponse(BaseModel):
    predicted_class: str
    confidence: float
    is_attack: bool
    explanation_space: str = Field(..., description="'probability' (Random Forest) or 'log-odds' (XGBoost).")
    baseline: float = Field(..., description="The model's average output for this class. baseline + all contributions = the model's output.")
    top_features: list[FeatureContribution]
    other_features_contribution: float = Field(..., description="Combined contribution of all features not listed above.")

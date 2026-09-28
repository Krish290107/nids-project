import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.api.inference import InferenceService, MissingFeaturesError
from src.api.main import create_app

CLASS_NAMES = ["BENIGN", "DoS", "PortScan"]
FEATURES = ["f1", "f2", "f3"]


@pytest.fixture()
def paths(tmp_path):
    """Train a tiny model and save it the same way Day 3-4 and Day 5-7 do."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=300)
    raw = pd.DataFrame({
        "f1": y * 10 + rng.normal(size=300),   # class is easy to tell from f1
        "f2": rng.normal(size=300),
        "f3": rng.normal(size=300),
    })

    scaler = StandardScaler().fit(raw)
    scaled = pd.DataFrame(scaler.transform(raw), columns=FEATURES).astype(np.float32)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(scaled, y)

    encoder = LabelEncoder().fit(CLASS_NAMES)
    preprocessing = {
        "scaler": scaler,
        "medians": raw.median().to_dict(),
        "feature_columns": FEATURES,
        "label_encoder": encoder,
    }
    model_bundle = {
        "model": model,
        "model_name": "random_forest",
        "feature_columns": FEATURES,
        "class_names": CLASS_NAMES,
    }
    model_path = tmp_path / "model.pkl"
    prep_path = tmp_path / "prep.joblib"
    joblib.dump(model_bundle, model_path)
    joblib.dump(preprocessing, prep_path)
    return {"model": model_path, "prep": prep_path, "db": tmp_path / "preds.db"}


@pytest.fixture()
def client(paths):
    app = create_app(
        model_path=paths["model"],
        preprocessing_path=paths["prep"],
        db_path=paths["db"],
        benign_label="BENIGN",
        max_batch_size=5,
    )
    with TestClient(app) as test_client:   # the "with" runs startup, which loads the model
        yield test_client


def flow(f1: float) -> dict:
    return {"f1": f1, "f2": 0.0, "f3": 0.0}


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"
    assert client.get("/favicon.ico").status_code == 204


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["classes"] == CLASS_NAMES
    assert body["feature_columns"] == FEATURES


def test_predict_returns_expected_class(client):
    # f1 near 0 -> BENIGN, near 10 -> DoS, near 20 -> PortScan (see the fixture)
    assert client.post("/predict", json={"features": flow(0.0)}).json()["predicted_class"] == "BENIGN"
    dos = client.post("/predict", json={"features": flow(10.0)}).json()
    assert dos["predicted_class"] == "DoS"
    assert dos["is_attack"] is True
    assert 0 <= dos["confidence"] <= 1
    assert set(dos["probabilities"]) == set(CLASS_NAMES)


def test_missing_feature_gives_422(client):
    response = client.post("/predict", json={"features": {"f1": 1.0}})
    assert response.status_code == 422
    assert "f2" in response.json()["detail"]


def test_null_feature_is_imputed(client):
    response = client.post("/predict", json={"features": {"f1": 10.0, "f2": None, "f3": None}})
    assert response.status_code == 200


def test_extra_features_are_ignored(client):
    payload = {"features": {**flow(10.0), "Flow ID": "abc", "Source IP": "1.2.3.4"}}
    assert client.post("/predict", json=payload).status_code == 200


def test_batch_predict(client):
    body = client.post(
        "/predict/batch", json={"flows": [flow(0.0), flow(10.0), flow(20.0)]}
    ).json()
    assert body["count"] == 3
    assert body["attacks_detected"] == 2
    assert [p["predicted_class"] for p in body["predictions"]] == CLASS_NAMES


def test_batch_too_large_gives_413(client):
    response = client.post("/predict/batch", json={"flows": [flow(0.0)] * 6})
    assert response.status_code == 413


def test_empty_batch_gives_422(client):
    assert client.post("/predict/batch", json={"flows": []}).status_code == 422


def test_predictions_are_logged_and_counted(client):
    client.post("/predict/batch", json={"flows": [flow(0.0), flow(10.0), flow(10.0)]})
    recent = client.get("/predictions/recent?limit=10").json()
    assert len(recent) == 3
    stats = client.get("/stats").json()
    assert stats["total"] == 3
    assert stats["per_class"]["DoS"] == 2
    assert stats["attacks"] == 2


def test_prepare_handles_infinity(paths):
    service = InferenceService.from_files(paths["model"], paths["prep"], "BENIGN")
    X = service.prepare([{"f1": np.inf, "f2": 1.0, "f3": -np.inf}])
    assert not np.isinf(X.values).any()
    assert not X.isna().values.any()
    assert list(X.columns) == FEATURES


def test_prepare_raises_for_missing_columns(paths):
    service = InferenceService.from_files(paths["model"], paths["prep"], "BENIGN")
    with pytest.raises(MissingFeaturesError):
        service.prepare([{"f1": 1.0}])


def test_startup_fails_clearly_without_model(paths, tmp_path):
    app = create_app(
        model_path=tmp_path / "nope.pkl",
        preprocessing_path=paths["prep"],
        db_path=paths["db"],
    )
    with pytest.raises(FileNotFoundError, match="run_training"):
        with TestClient(app):
            pass

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
from xgboost import XGBClassifier

import src.api.main as api_main
from src.api.explain import FlowExplainer, _to_samples_features_classes
from src.api.inference import InferenceService, MissingFeaturesError

CLASS_NAMES = ["BENIGN", "DoS", "PortScan"]
FEATURES = [f"f{i}" for i in range(8)]


def build(model_factory):
    """A tiny trained model + the artifacts the real pipeline would save."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 900)
    raw = pd.DataFrame(rng.normal(size=(900, 8)), columns=FEATURES)
    raw["f0"] += y * 4  # f0 carries the class signal
    scaler = StandardScaler().fit(raw)
    scaled = pd.DataFrame(scaler.transform(raw), columns=FEATURES).astype(np.float32)
    model = model_factory().fit(scaled, y)
    bundle = {"model": model, "model_name": "m", "feature_columns": FEATURES, "class_names": CLASS_NAMES}
    prep = {
        "scaler": scaler,
        "medians": raw.median().to_dict(),
        "feature_columns": FEATURES,
        "label_encoder": LabelEncoder().fit(CLASS_NAMES),
    }
    return bundle, prep, raw


@pytest.fixture(scope="module")
def rf_service():
    bundle, prep, raw = build(lambda: RandomForestClassifier(30, random_state=0))
    return InferenceService(bundle, prep, "BENIGN"), raw


def test_contributions_add_up_to_the_confidence_for_random_forest(rf_service):
    service, raw = rf_service
    explainer = FlowExplainer(service)
    assert explainer.space == "probability"
    for result in explainer.explain(raw.iloc[:25].to_dict(orient="records"), top_k=3):
        total = (
            result["baseline"]
            + sum(f["contribution"] for f in result["top_features"])
            + result["other_features_contribution"]
        )
        assert total == pytest.approx(result["confidence"], abs=1e-3)


def test_explanation_agrees_with_prediction(rf_service):
    service, raw = rf_service
    rows = raw.iloc[:20].to_dict(orient="records")
    explained = FlowExplainer(service).explain(rows)
    predicted = service.predict(rows)
    assert [e["predicted_class"] for e in explained] == [p["predicted_class"] for p in predicted]
    assert [e["is_attack"] for e in explained] == [p["is_attack"] for p in predicted]


def test_the_informative_feature_ranks_first(rf_service):
    service, raw = rf_service
    results = FlowExplainer(service).explain(raw.iloc[:40].to_dict(orient="records"))
    firsts = [r["top_features"][0]["feature"] for r in results]
    assert firsts.count("f0") >= 0.8 * len(firsts)  # f0 is where the class signal was planted


def test_top_k_limits_the_list_and_sorts_by_size(rf_service):
    service, raw = rf_service
    result = FlowExplainer(service).explain([raw.iloc[0].to_dict()], top_k=4)[0]
    sizes = [abs(f["contribution"]) for f in result["top_features"]]
    assert len(sizes) == 4 and sizes == sorted(sizes, reverse=True)
    assert {f["direction"] for f in result["top_features"]} <= {"toward", "against"}


def test_missing_or_infinite_raw_values_come_back_as_none(rf_service):
    service, raw = rf_service
    row = raw.iloc[0].to_dict()
    row["f0"] = float("inf")
    result = FlowExplainer(service).explain([row], top_k=8)[0]
    f0 = next(f for f in result["top_features"] if f["feature"] == "f0")
    assert f0["raw_value"] is None  # was inf; the model used the training median instead


def test_missing_required_features_raise(rf_service):
    service, _ = rf_service
    with pytest.raises(MissingFeaturesError):
        FlowExplainer(service).explain([{"f0": 1.0}])


def test_xgboost_is_explained_in_log_odds_space():
    bundle, prep, raw = build(lambda: XGBClassifier(n_estimators=30, max_depth=3))
    explainer = FlowExplainer(InferenceService(bundle, prep, "BENIGN"))
    assert explainer.space == "log-odds"
    result = explainer.explain([raw.iloc[0].to_dict()])[0]
    assert result["explanation_space"] == "log-odds" and result["top_features"]


def test_shap_output_shapes_are_normalized():
    n, f, c = 5, 4, 3
    as_array = np.random.default_rng(0).normal(size=(n, f, c))
    as_list = [as_array[:, :, k] for k in range(c)]  # older SHAP: list of per-class arrays
    as_classes_first = np.transpose(as_array, (2, 0, 1))
    for shape_variant in (as_array, as_list, as_classes_first):
        assert _to_samples_features_classes(shape_variant, c).shape == (n, f, c)
    assert _to_samples_features_classes(as_array[:, :, 0], 2).shape == (n, f, 2)  # single output
    with pytest.raises(ValueError):
        _to_samples_features_classes(np.zeros((2, 2, 2, 2)), 3)


# ---- the API endpoint -------------------------------------------------------

@pytest.fixture()
def client(tmp_path):
    bundle, prep, raw = build(lambda: RandomForestClassifier(30, random_state=0))
    joblib.dump(bundle, tmp_path / "m.pkl")
    joblib.dump(prep, tmp_path / "p.joblib")
    app = api_main.create_app(
        model_path=tmp_path / "m.pkl",
        preprocessing_path=tmp_path / "p.joblib",
        db_path=tmp_path / "d.db",
        benign_label="BENIGN",
    )
    with TestClient(app) as test_client:
        test_client.raw = raw
        yield test_client


def test_explain_endpoint_returns_prediction_and_reasons(client):
    flow = client.raw.iloc[0].to_dict()
    response = client.post("/explain?top_k=3", json={"features": flow})
    assert response.status_code == 200
    body = response.json()
    assert len(body["top_features"]) == 3
    assert body["explanation_space"] == "probability"
    assert body["predicted_class"] == client.post("/predict", json={"features": flow}).json()["predicted_class"]


def test_explain_is_lazy_and_not_logged(client):
    assert client.app.state.explainer is None  # nothing built until first use
    client.post("/explain", json={"features": client.raw.iloc[0].to_dict()})
    assert client.app.state.explainer is not None
    assert client.get("/stats").json()["total"] == 0  # explanations don't go in the prediction log


def test_explain_missing_features_gives_422(client):
    response = client.post("/explain", json={"features": {"f0": 1.0}})
    assert response.status_code == 422 and "f1" in response.json()["detail"]


def test_explain_top_k_is_clamped(client):
    flow = client.raw.iloc[0].to_dict()
    assert len(client.post("/explain?top_k=0", json={"features": flow}).json()["top_features"]) == 1
    assert len(client.post("/explain?top_k=999", json={"features": flow}).json()["top_features"]) == 8  # only 8 exist


def test_explain_gives_501_if_shap_is_missing(client, monkeypatch):
    def no_shap(service):
        raise ImportError("No module named 'shap'")

    monkeypatch.setattr(api_main, "FlowExplainer", no_shap)
    response = client.post("/explain", json={"features": client.raw.iloc[0].to_dict()})
    assert response.status_code == 501 and "pip install shap" in response.json()["detail"]

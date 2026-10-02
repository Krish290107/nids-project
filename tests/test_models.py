import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models import train as trn
from src.models import evaluate as ev

CLASS_NAMES = ["BENIGN", "DoS", "PortScan"]


def make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    X = pd.DataFrame({
        "f1": y + rng.normal(scale=0.3, size=n),   # informative
        "f2": rng.normal(size=n),
        "f3": rng.normal(size=n),
    }).astype(np.float32)
    return X, y


def test_random_forest_learns_and_evaluates():
    X, y = make_data()
    model, elapsed = trn.train_random_forest(
        X, y, {"n_estimators": 20}, random_state=0, use_class_weight=False
    )
    assert elapsed >= 0
    res = ev.evaluate_model(model, X, y, CLASS_NAMES)
    assert res["accuracy"] > 0.8
    assert set(res["per_class"]) == set(CLASS_NAMES)
    assert np.array(res["confusion_matrix"]).shape == (3, 3)


def test_xgboost_trains():
    pytest.importorskip("xgboost")
    X, y = make_data()
    model, _ = trn.train_xgboost(
        X, y, {"n_estimators": 20, "max_depth": 3}, random_state=0, use_class_weight=False
    )
    res = ev.evaluate_model(model, X, y, CLASS_NAMES)
    assert 0 <= res["macro_f1"] <= 1


def test_evaluate_handles_class_missing_from_test():
    X, y = make_data()
    model, _ = trn.train_random_forest(X, y, {"n_estimators": 10}, 0, False)
    mask = y != 2  # test set with no PortScan rows
    res = ev.evaluate_model(model, X[mask], y[mask], CLASS_NAMES)
    assert res["per_class"]["PortScan"]["support"] == 0


def test_subsample_training_data():
    X, y = make_data(n=200)
    X_small, y_small = trn.subsample_training_data(X, y, max_rows=50, random_state=0)
    assert len(X_small) == 50 and len(y_small) == 50
    X_same, _ = trn.subsample_training_data(X, y, max_rows=None, random_state=0)
    assert len(X_same) == 200


def test_save_model_bundle_roundtrip(tmp_path):
    X, y = make_data()
    model, _ = trn.train_random_forest(X, y, {"n_estimators": 5}, 0, False)
    path = tmp_path / "model.pkl"
    trn.save_model_bundle(path, model, "random_forest", list(X.columns), CLASS_NAMES)
    bundle = joblib.load(path)
    assert bundle["model_name"] == "random_forest"
    assert bundle["feature_columns"] == ["f1", "f2", "f3"]
    assert bundle["model"].predict(X).shape == (len(X),)


def test_load_processed_data_missing_files_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_preprocessing"):
        trn.load_processed_data(tmp_path)


def test_top_feature_importances():
    X, y = make_data()
    model, _ = trn.train_random_forest(X, y, {"n_estimators": 20}, 0, False)
    top = ev.top_feature_importances(model, list(X.columns), top_n=2)
    assert len(top) == 2
    assert next(iter(top)) == "f1"


def test_low_support_classes_flags_tiny_classes():
    from src.models.evaluate import low_support_classes

    per_class = {
        "BENIGN": {"support": 5000},
        "RareClass": {"support": 2},
        "Absent": {"support": 0},  # not in the test set at all: not a "low support" score
    }
    assert low_support_classes(per_class) == {"RareClass": 2}


def test_model_report_includes_low_support_warning(tmp_path):
    from src.models.evaluate import write_model_report

    per_class = {
        "BENIGN": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 5000},
        "RareClass": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2},
    }
    results = {"rf": {"accuracy": 1.0, "macro_f1": 1.0, "weighted_f1": 1.0, "per_class": per_class}}
    out = tmp_path / "report.md"
    write_model_report(results, {"rf": 1.0}, "rf", "macro_f1", {}, out)
    text = out.read_text(encoding="utf-8")
    assert "Low-support warning" in text and "RareClass (2 test flows)" in text
    assert "no separate validation split" in text

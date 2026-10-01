import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.models.explain_report import format_explanation, pick_example_indices, plot_explanations

CLASS_NAMES = ["BENIGN", "DoS", "Heartbleed", "Absent"]
Y = np.array([0] * 50 + [1] * 30 + [2] * 2)  # class 3 ("Absent") has no test rows


def make_result(predicted="DoS"):
    return {
        "predicted_class": predicted, "confidence": 0.93, "is_attack": predicted != "BENIGN",
        "explanation_space": "probability", "baseline": 0.31,
        "top_features": [
            {"feature": "Flow Duration", "raw_value": 1234.5, "z_score": 2.1, "contribution": 0.4, "direction": "toward"},
            {"feature": "Fwd IAT Mean", "raw_value": None, "z_score": -0.2, "contribution": -0.05, "direction": "against"},
        ],
        "other_features_contribution": 0.02,
    }


def test_pick_one_per_class_skips_classes_with_no_test_rows():
    picks = pick_example_indices(Y, CLASS_NAMES, per_class=1, seed=0)
    assert [name for _, name in picks] == ["BENIGN", "DoS", "Heartbleed"]
    for index, name in picks:
        assert CLASS_NAMES[Y[index]] == name


def test_pick_is_reproducible_and_caps_at_available_rows():
    a = pick_example_indices(Y, CLASS_NAMES, per_class=5, seed=1)
    b = pick_example_indices(Y, CLASS_NAMES, per_class=5, seed=1)
    assert a == b
    assert sum(1 for _, name in a if name == "Heartbleed") == 2  # only 2 exist


def test_pick_only_selected_classes_and_rejects_unknown():
    assert [n for _, n in pick_example_indices(Y, CLASS_NAMES, only=["DoS"])] == ["DoS"]
    with pytest.raises(ValueError, match="Unknown"):
        pick_example_indices(Y, CLASS_NAMES, only=["Nope"])


def test_format_explanation_contains_the_key_facts():
    text = format_explanation(make_result(), true_class="DoS")
    assert "true: DoS -> predicted DoS" in text and "ATTACK" in text
    assert "Flow Duration" in text and "toward" in text and "against" in text
    assert "missing" in text  # the None raw value
    assert "all other features" in text


def test_plot_explanations_writes_a_png(tmp_path):
    out = tmp_path / "figs" / "shap.png"
    plot_explanations([make_result(), make_result("BENIGN"), make_result()], ["DoS", "BENIGN", "DoS"], out)
    assert out.exists() and out.stat().st_size > 1000

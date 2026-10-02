import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.preprocessing import pipeline as prep


def make_df():
    n = 60
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Flow ID": [f"id-{i}" for i in range(n)],
        "Source IP": [f"192.168.0.{i}" for i in range(n)],
        "feat_a": rng.normal(size=n),
        "feat_b": rng.normal(size=n),
        "Label": rng.choice(["BENIGN", "DoS"], size=n, p=[0.8, 0.2]),
    })
    df["feat_c"] = df["feat_a"] * 1.0001  # near-perfect correlation with feat_a
    df.loc[0, "feat_b"] = np.inf
    df.loc[1, "feat_b"] = np.nan
    df = pd.concat([df, df.iloc[[2]]], ignore_index=True)  # inject one duplicate
    return df


def test_drop_duplicate_rows():
    df = make_df()
    before = len(df)
    cleaned, removed = prep.drop_duplicate_rows(df)
    assert removed == 1
    assert len(cleaned) == before - 1


def test_drop_leakage_columns():
    df = make_df()
    cleaned, dropped = prep.drop_leakage_columns(df, target_col="Label")
    assert "Flow ID" in dropped
    assert "Source IP" in dropped
    assert "Flow ID" not in cleaned.columns


def test_median_imputation_removes_inf_and_nan():
    df = make_df()
    X = df[["feat_a", "feat_b"]]
    medians = prep.compute_medians(X)
    cleaned = prep.apply_median_imputation(X, medians)
    assert not np.isinf(cleaned.values).any()
    assert not cleaned.isna().values.any()


def test_correlated_drop_list_finds_pair():
    df = make_df()
    X = df[["feat_a", "feat_b", "feat_c"]].replace([np.inf, -np.inf], np.nan).fillna(0)
    to_drop = prep.get_correlated_drop_list(X, threshold=0.99)
    assert "feat_c" in to_drop or "feat_a" in to_drop


def test_scale_features_zero_mean_on_train():
    df = make_df()
    X = df[["feat_a", "feat_b"]].fillna(0).replace([np.inf, -np.inf], 0)
    X_train, X_test = X.iloc[:40], X.iloc[40:]
    X_train_s, X_test_s, scaler = prep.scale_features(X_train, X_test)
    assert abs(X_train_s["feat_a"].mean()) < 1e-8
    assert X_test_s.shape == X_test.shape


def test_encode_labels_roundtrip():
    y = pd.Series(["BENIGN", "DoS", "BENIGN"])
    y_encoded, encoder = prep.encode_labels(y)
    assert set(y_encoded) == {0, 1}
    assert list(encoder.inverse_transform(y_encoded)) == list(y)


def test_smote_balances_classes():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    y = np.array([0] * 90 + [1] * 10)
    X_bal, y_bal = prep.balance_training_data(
        X, y, method="smote", random_state=42, k_neighbors=5
    )
    counts = pd.Series(y_bal).value_counts()
    assert counts[0] == counts[1]


def test_drop_classes_removes_rows_and_reports_counts():
    df = pd.DataFrame({"f": range(10), "Label": ["BENIGN"] * 6 + ["DoS"] * 3 + ["Rare"]})
    kept, removed = prep.drop_classes(df, "Label", ["Rare"])
    assert removed == {"Rare": 1}
    assert len(kept) == 9 and "Rare" not in set(kept["Label"])
    assert list(kept.index) == list(range(9))  # index reset


def test_drop_classes_ignores_case_and_spaces():
    df = pd.DataFrame({"f": range(8), "Label": ["BENIGN"] * 4 + ["DoS"] * 2 + [" rare ", "RARE"]})
    kept, removed = prep.drop_classes(df, "Label", ["Rare"])
    assert sum(removed.values()) == 2 and len(kept) == 6


def test_drop_classes_unknown_name_is_only_a_warning():
    df = pd.DataFrame({"f": range(4), "Label": ["BENIGN", "BENIGN", "DoS", "DoS"]})
    kept, removed = prep.drop_classes(df, "Label", ["Heartbleed"])
    assert removed == {} and len(kept) == 4


def test_drop_classes_empty_or_none_changes_nothing():
    df = pd.DataFrame({"f": range(4), "Label": ["BENIGN", "BENIGN", "DoS", "DoS"]})
    for setting in ([], None):
        kept, removed = prep.drop_classes(df, "Label", setting)
        assert removed == {} and len(kept) == 4


def test_drop_classes_refuses_to_leave_fewer_than_two_classes():
    import pytest

    df = pd.DataFrame({"f": range(4), "Label": ["BENIGN", "BENIGN", "DoS", "DoS"]})
    with pytest.raises(ValueError, match="fewer than 2 classes"):
        prep.drop_classes(df, "Label", ["DoS"])

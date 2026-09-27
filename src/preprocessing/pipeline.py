"""Day 3-4 preprocessing pipeline.

Order of operations matters here — everything that "learns" a statistic
(medians, correlated-column list, scaler, SMOTE) is fit on the TRAINING
split only, then applied to test. Fitting on the full dataset before
splitting would leak test-set information into training, which quietly
inflates your reported metrics later.

Pipeline:
    raw df
      -> drop exact duplicate rows
      -> drop leakage-prone columns (Flow ID, IPs, Timestamp)
      -> split into train/test (stratified on label)
      -> [fit on train] replace inf -> NaN, impute NaN with train medians
      -> [fit on train] drop one column per highly-correlated pair
      -> [fit on train] scale numeric features
      -> [fit on train] SMOTE-balance the training set only (never test)
      -> save artifacts (medians, dropped columns, scaler, label encoder)
      -> save processed train/test CSVs
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.data.eda import identify_leakage_columns
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def drop_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    logger.info(f"Dropped {removed:,} duplicate row(s); {len(df):,} rows remain")
    return df, removed


def drop_leakage_columns(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, list[str]]:
    flagged = identify_leakage_columns(df)
    flagged = [c for c in flagged if c != target_col]
    df = df.drop(columns=flagged, errors="ignore")
    logger.info(f"Dropped {len(flagged)} leakage-prone column(s): {flagged}")
    return df, flagged


def encode_labels(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    logger.info(f"Label encoding: {mapping}")
    return y_encoded, encoder


def split_data(
    X: pd.DataFrame, y: np.ndarray, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Split: {len(X_train):,} train rows / {len(X_test):,} test rows")
    return X_train, X_test, y_train, y_test


def compute_medians(df: pd.DataFrame) -> dict:
    """Replace inf with NaN first, then compute per-column medians. Fit on TRAIN only."""
    numeric_df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    medians = numeric_df.median(numeric_only=True).to_dict()
    return medians


def apply_median_imputation(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    for col in numeric_cols:
        if col in medians:
            df[col] = df[col].fillna(medians[col])
    remaining_na = df[numeric_cols].isna().sum().sum()
    if remaining_na:
        # a column with no train-set median (e.g. all-NaN) — fall back to 0 rather than crash
        df[numeric_cols] = df[numeric_cols].fillna(0)
        logger.warning(f"{remaining_na} values had no median available; filled with 0")
    return df


def get_correlated_drop_list(df: pd.DataFrame, threshold: float) -> list[str]:
    """Return one column name per highly-correlated pair, to be dropped. Fit on TRAIN only."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop: set[str] = set()
    for col in upper.columns:
        if col in to_drop:
            continue
        correlated_with = upper.index[upper[col] >= threshold].tolist()
        for other in correlated_with:
            if other not in to_drop:
                to_drop.add(other)
    to_drop_list = sorted(to_drop)
    logger.info(f"Dropping {len(to_drop_list)} column(s) for high correlation: {to_drop_list}")
    return to_drop_list


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler


def balance_training_data(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    method: Literal["smote", "class_weight", "none"],
    random_state: int,
    k_neighbors: int = 5,
) -> tuple[pd.DataFrame, np.ndarray]:
    if method == "none":
        return X_train, y_train

    if method == "class_weight":
        # no resampling — model training (Day 5-7) should pass class_weight='balanced' instead
        logger.info("balancing_method=class_weight: no resampling done here, "
                     "pass class_weight='balanced' when training the model")
        return X_train, y_train

    if method == "smote":
        from imblearn.over_sampling import SMOTE

        counts = pd.Series(y_train).value_counts()
        safe_k = min(k_neighbors, counts.min() - 1) if counts.min() > 1 else 1
        if safe_k < 1:
            logger.warning(
                f"Smallest class has only {counts.min()} sample(s) — SMOTE needs at least 2. "
                "Skipping balancing for this run."
            )
            return X_train, y_train
        if safe_k != k_neighbors:
            logger.warning(f"Reducing SMOTE k_neighbors from {k_neighbors} to {safe_k} "
                            "because the smallest class is small")

        smote = SMOTE(random_state=random_state, k_neighbors=safe_k)
        X_bal, y_bal = smote.fit_resample(X_train, y_train)
        logger.info(f"SMOTE: {len(X_train):,} -> {len(X_bal):,} training rows after balancing")
        return pd.DataFrame(X_bal, columns=X_train.columns), y_bal

    raise ValueError(f"Unknown balancing_method: {method}")


def save_artifacts(
    artifact_path: Path,
    label_encoder: LabelEncoder,
    scaler: StandardScaler,
    medians: dict,
    feature_columns: list[str],
    dropped_leakage_columns: list[str],
    dropped_correlated_columns: list[str],
) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "label_encoder": label_encoder,
        "scaler": scaler,
        "medians": medians,
        "feature_columns": feature_columns,  # exact order the model/API must feed features in
        "dropped_leakage_columns": dropped_leakage_columns,
        "dropped_correlated_columns": dropped_correlated_columns,
    }
    joblib.dump(bundle, artifact_path)
    logger.info(f"Saved preprocessing artifact bundle -> {artifact_path}")


def save_processed_data(
    processed_dir: Path,
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    pd.Series(y_train, name="label").to_csv(processed_dir / "y_train.csv", index=False)
    pd.Series(y_test, name="label").to_csv(processed_dir / "y_test.csv", index=False)
    logger.info(f"Saved processed train/test CSVs -> {processed_dir}")

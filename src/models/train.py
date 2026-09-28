"""Day 5-7 model training helpers.

Reads the train/test CSVs produced by Day 3-4 and trains the models.
Nothing here re-does preprocessing: the data is already cleaned, scaled and
(for the training split) balanced.
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_processed_data(
    processed_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    names = ["X_train", "X_test", "y_train", "y_test"]
    paths = {n: processed_dir / f"{n}.csv" for n in names}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Processed data not found:\n  " + "\n  ".join(missing) + "\n"
            "Run 'python scripts/run_preprocessing.py' (Day 3-4) first."
        )

    # float32 halves memory use compared to the default float64
    X_train = pd.read_csv(paths["X_train"]).astype(np.float32)
    X_test = pd.read_csv(paths["X_test"]).astype(np.float32)
    y_train = pd.read_csv(paths["y_train"])["label"].to_numpy()
    y_test = pd.read_csv(paths["y_test"])["label"].to_numpy()

    logger.info(f"Loaded train {X_train.shape} and test {X_test.shape}")
    return X_train, X_test, y_train, y_test


def subsample_training_data(
    X: pd.DataFrame, y: np.ndarray, max_rows: int | None, random_state: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Optional random subsample so training fits on a student laptop."""
    if max_rows is None or len(X) <= max_rows:
        return X, y
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=max_rows, replace=False)
    logger.info(f"Subsampled training data: {len(X):,} -> {max_rows:,} rows")
    return X.iloc[idx].reset_index(drop=True), y[idx]


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: dict,
    random_state: int,
    use_class_weight: bool,
) -> tuple[RandomForestClassifier, float]:
    model = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth"),
        class_weight="balanced" if use_class_weight else None,
        n_jobs=-1,
        random_state=random_state,
    )
    logger.info(f"Training Random Forest ({model.n_estimators} trees) ...")
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    logger.info(f"Random Forest trained in {elapsed:.1f}s")
    return model, elapsed


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: dict,
    random_state: int,
    use_class_weight: bool,
):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is not installed. Run: pip install -r requirements.txt"
        ) from exc

    model = XGBClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 8),
        learning_rate=params.get("learning_rate", 0.1),
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=random_state,
    )
    sample_weight = (
        compute_sample_weight("balanced", y_train) if use_class_weight else None
    )
    logger.info(f"Training XGBoost ({params.get('n_estimators', 100)} rounds) ...")
    start = time.perf_counter()
    model.fit(X_train, y_train, sample_weight=sample_weight)
    elapsed = time.perf_counter() - start
    logger.info(f"XGBoost trained in {elapsed:.1f}s")
    return model, elapsed


def save_model_bundle(
    path: Path,
    model,
    model_name: str,
    feature_columns: list[str],
    class_names: list[str],
) -> None:
    """Save the model together with what the API (Day 8-9) needs to use it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "model_name": model_name,
        "feature_columns": feature_columns,
        "class_names": class_names,
    }
    joblib.dump(bundle, path)
    logger.info(f"Saved model bundle ({model_name}) -> {path}")

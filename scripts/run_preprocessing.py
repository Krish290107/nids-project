"""Day 3-4 entry point.

Run from the project root, after Day 1-2 has confirmed data/raw/ loads cleanly:
    python scripts/run_preprocessing.py

Produces:
    models/preprocessing_pipeline.joblib   (scaler, label encoder, medians, column order)
    data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv
    reports/metrics/preprocessing_summary.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import load_config, resolve_path, ensure_dir
from src.utils.logging_config import get_logger
from src.data.load_dataset import load_dataset, detect_target_column
from src.preprocessing import pipeline as prep

logger = get_logger(__name__)


def main() -> None:
    config = load_config()
    ds_cfg = config["dataset"]
    pp_cfg = config["preprocessing"]

    logger.info("=== Day 3-4: Preprocessing ===")
    df = load_dataset()
    target_col = detect_target_column(df, ds_cfg["target_column"])

    df, n_duplicates_removed = prep.drop_duplicate_rows(df)

    dropped_leakage: list[str] = []
    if pp_cfg["drop_leakage_columns"]:
        df, dropped_leakage = prep.drop_leakage_columns(df, target_col)

    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    y_encoded, label_encoder = prep.encode_labels(y_raw)

    X_train, X_test, y_train, y_test = prep.split_data(
        X, y_encoded, test_size=pp_cfg["test_size"], random_state=pp_cfg["random_state"]
    )

    # --- everything below is fit on X_train only, then applied to X_test ---
    medians = prep.compute_medians(X_train)
    X_train = prep.apply_median_imputation(X_train, medians)
    X_test = prep.apply_median_imputation(X_test, medians)

    dropped_correlated: list[str] = []
    if pp_cfg["drop_correlated_features"]:
        dropped_correlated = prep.get_correlated_drop_list(X_train, pp_cfg["correlation_threshold"])
        X_train = X_train.drop(columns=dropped_correlated, errors="ignore")
        X_test = X_test.drop(columns=dropped_correlated, errors="ignore")

    X_train, X_test, scaler = prep.scale_features(X_train, X_test)

    X_train_bal, y_train_bal = prep.balance_training_data(
        X_train, y_train,
        method=pp_cfg["balancing_method"],
        random_state=pp_cfg["random_state"],
        k_neighbors=pp_cfg["smote_k_neighbors"],
    )

    feature_columns = X_train_bal.columns.tolist()

    prep.save_artifacts(
        artifact_path=resolve_path(pp_cfg["artifact_path"]),
        label_encoder=label_encoder,
        scaler=scaler,
        medians=medians,
        feature_columns=feature_columns,
        dropped_leakage_columns=dropped_leakage,
        dropped_correlated_columns=dropped_correlated,
    )

    prep.save_processed_data(
        processed_dir=resolve_path(pp_cfg["processed_dir"]),
        X_train=X_train_bal, y_train=y_train_bal,
        X_test=X_test, y_test=y_test,
    )

    summary = {
        "rows_after_dedup": int(len(df)),
        "duplicates_removed": int(n_duplicates_removed),
        "dropped_leakage_columns": dropped_leakage,
        "dropped_correlated_columns": dropped_correlated,
        "train_rows_before_balancing": int(len(X_train)),
        "train_rows_after_balancing": int(len(X_train_bal)),
        "test_rows": int(len(X_test)),
        "final_feature_count": len(feature_columns),
        "final_feature_columns": feature_columns,
        "label_classes": label_encoder.classes_.tolist(),
        "balancing_method": pp_cfg["balancing_method"],
    }
    summary_path = resolve_path(pp_cfg["summary_path"])
    ensure_dir(summary_path.parent)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved {summary_path}")

    logger.info("=== Day 3-4 complete ===")
    logger.info(f"Final feature count: {len(feature_columns)}")
    logger.info(f"Train rows (post-balancing): {len(X_train_bal):,} | Test rows: {len(X_test):,}")


if __name__ == "__main__":
    main()

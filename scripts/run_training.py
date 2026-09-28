"""Day 5-7 entry point.

Run from the project root, after Day 3-4 has produced data/processed/*.csv:
    python scripts/run_training.py

Produces:
    models/model.pkl                        (best model + feature order + class names)
    reports/metrics/model_metrics.json      (all metrics, incl. confusion matrices)
    reports/MODEL_REPORT.md                 (readable comparison + per-class table)
    reports/figures/confusion_matrix_*.png, feature_importance.png, model_comparison.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib

from src.utils.paths import load_config, resolve_path, ensure_dir
from src.utils.logging_config import get_logger
from src.models import train as trn
from src.models import evaluate as ev

logger = get_logger(__name__)

TRAINERS = {
    "random_forest": trn.train_random_forest,
    "xgboost": trn.train_xgboost,
}


def main() -> None:
    config = load_config()
    pp_cfg = config["preprocessing"]
    tr_cfg = config["training"]

    logger.info("=== Day 5-7: Model training ===")

    artifacts = joblib.load(resolve_path(pp_cfg["artifact_path"]))
    feature_columns = artifacts["feature_columns"]
    class_names = artifacts["label_encoder"].classes_.tolist()

    X_train, X_test, y_train, y_test = trn.load_processed_data(
        resolve_path(pp_cfg["processed_dir"])
    )
    if X_train.columns.tolist() != feature_columns:
        raise ValueError(
            "Column order in X_train.csv does not match the saved preprocessing "
            "artifact. Re-run 'python scripts/run_preprocessing.py' and try again."
        )

    X_train, y_train = trn.subsample_training_data(
        X_train, y_train, tr_cfg["max_train_rows"], tr_cfg["random_state"]
    )

    # if Day 3-4 didn't balance with SMOTE, let the models handle imbalance themselves
    use_class_weight = pp_cfg["balancing_method"] == "class_weight"

    models, results, train_times = {}, {}, {}
    for name in tr_cfg["models_to_train"]:
        if name not in TRAINERS:
            raise ValueError(f"Unknown model '{name}'. Choose from: {list(TRAINERS)}")
        model, elapsed = TRAINERS[name](
            X_train, y_train, tr_cfg.get(name, {}), tr_cfg["random_state"], use_class_weight
        )
        models[name], train_times[name] = model, elapsed
        results[name] = ev.evaluate_model(model, X_test, y_test, class_names)
        logger.info(
            f"{name}: accuracy={results[name]['accuracy']} "
            f"macro_f1={results[name]['macro_f1']} weighted_f1={results[name]['weighted_f1']}"
        )

    metric = tr_cfg["selection_metric"]
    best_name = max(results, key=lambda n: results[n][metric])
    logger.info(f"Best model by {metric}: {best_name}")

    figures_dir = ensure_dir(resolve_path(config["reports"]["figures_dir"]))
    for name, res in results.items():
        ev.plot_confusion_matrix(
            res["confusion_matrix"], class_names, name,
            figures_dir / f"confusion_matrix_{name}.png",
        )
    ev.plot_model_comparison(results, figures_dir / "model_comparison.png")

    importances = ev.top_feature_importances(models[best_name], feature_columns)
    ev.plot_feature_importance(
        importances, f"Top features - {best_name}", figures_dir / "feature_importance.png"
    )

    model_path = resolve_path(tr_cfg["model_path"])
    trn.save_model_bundle(model_path, models[best_name], best_name, feature_columns, class_names)
    if tr_cfg["save_all_models"]:
        for name, model in models.items():
            trn.save_model_bundle(
                model_path.parent / f"{name}.pkl", model, name, feature_columns, class_names
            )

    metrics_path = resolve_path(tr_cfg["metrics_path"])
    ensure_dir(metrics_path.parent)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best_name,
                "selection_metric": metric,
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "train_time_seconds": {k: round(v, 2) for k, v in train_times.items()},
                "models": results,
                "top_features": importances,
            },
            f, indent=2,
        )
    logger.info(f"Saved {metrics_path}")

    ev.write_model_report(
        results, train_times, best_name, metric, importances, resolve_path(tr_cfg["report_path"])
    )
    logger.info("=== Day 5-7 complete ===")


if __name__ == "__main__":
    main()

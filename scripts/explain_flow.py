"""Explain example predictions with SHAP: which features pushed the model to its answer?

Run from the project root, after Day 3-7 has produced the model and test data:
    python scripts/explain_flow.py                       # one example flow per class
    python scripts/explain_flow.py --class "DoS Hulk"    # only this class (repeatable)
    python scripts/explain_flow.py --per-class 2 --top-k 10

Needs `pip install shap` (already in requirements.txt). Does not need the API running.

Outputs:
    reports/metrics/shap_examples.json     the explanations, machine-readable
    reports/figures/shap_examples.png      one bar chart per example flow

The same explanations are available live from the API: POST /explain.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.api.explain import DEFAULT_TOP_K, FlowExplainer
from src.api.inference import InferenceService
from src.models.explain_report import format_explanation, pick_example_indices, plot_explanations
from src.utils.logging_config import get_logger
from src.utils.paths import ensure_dir, load_config, resolve_path

logger = get_logger(__name__)


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="classes", action="append", help="explain only this class (repeatable)")
    parser.add_argument("--per-class", type=int, default=1, help="example flows per class")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="features to show per flow")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    service = InferenceService.from_files(
        resolve_path(config["training"]["model_path"]),
        resolve_path(config["preprocessing"]["artifact_path"]),
        config["api"]["benign_label"],
    )
    processed = resolve_path(config["preprocessing"]["processed_dir"])
    X_test = pd.read_csv(processed / "X_test.csv")
    y_test = pd.read_csv(processed / "y_test.csv")["label"].to_numpy()
    class_names = service.class_names

    try:
        picks = pick_example_indices(y_test, class_names, args.per_class, args.seed, args.classes)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if not picks:
        raise SystemExit("No matching flows found in the test set.")

    # the test set is stored scaled; undo the scaling so the explainer receives raw values,
    # exactly like a real flow would arrive at the API
    indices = [i for i, _ in picks]
    raw = pd.DataFrame(
        service.scaler.inverse_transform(X_test.loc[indices, service.feature_columns]),
        columns=service.feature_columns,
    )
    rows = raw.to_dict(orient="records")
    true_classes = [name for _, name in picks]

    logger.info(f"Building the SHAP explainer for '{service.model_name}' (can take a while for a large forest) ...")
    explainer = FlowExplainer(service)
    start = time.perf_counter()
    results = explainer.explain(rows, top_k=args.top_k)
    per_flow = (time.perf_counter() - start) / len(rows)

    print()
    for result, true_class in zip(results, true_classes):
        print(format_explanation(result, true_class))
        print()
    print(f"Explained {len(results)} flow(s), {per_flow:.2f} s per flow.")

    metrics_dir = ensure_dir(resolve_path(config["reports"]["metrics_dir"]))
    figures_dir = ensure_dir(resolve_path(config["reports"]["figures_dir"]))
    json_path = metrics_dir / "shap_examples.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"true_class": t, **r} for t, r in zip(true_classes, results)], f, indent=2
        )
    plot_path = figures_dir / "shap_examples.png"
    plot_explanations(results, true_classes, plot_path)
    logger.info(f"Saved {json_path}")
    logger.info(f"Saved {plot_path}")


if __name__ == "__main__":
    main()

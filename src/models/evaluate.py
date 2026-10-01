"""Day 5-7 model evaluation.

Accuracy alone is misleading on imbalanced attack data, so everything here
reports precision / recall / F1 per class and uses macro F1 to compare models.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def evaluate_model(model, X_test: pd.DataFrame, y_test: np.ndarray, class_names: list[str]) -> dict:
    y_pred = model.predict(X_test)
    labels = list(range(len(class_names)))

    report = classification_report(
        y_test, y_pred, labels=labels, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    per_class = {
        name: {
            "precision": round(report[name]["precision"], 4),
            "recall": round(report[name]["recall"], 4),
            "f1": round(report[name]["f1-score"], 4),
            "support": int(report[name]["support"]),
        }
        for name in class_names
    }
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def top_feature_importances(model, feature_columns: list[str], top_n: int = 15) -> dict[str, float]:
    if not hasattr(model, "feature_importances_"):
        return {}
    series = pd.Series(model.feature_importances_, index=feature_columns)
    return {k: round(float(v), 5) for k, v in series.sort_values(ascending=False).head(top_n).items()}


def plot_confusion_matrix(cm: list[list[int]], class_names: list[str], title: str, out_path: Path) -> None:
    cm_arr = np.array(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    normalized = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums != 0)

    size = max(5, 0.9 * len(class_names) + 3)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{title} (row-normalized)")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            colour = "white" if normalized[i, j] > 0.5 else "black"
            ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center",
                    color=colour, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_feature_importance(importances: dict[str, float], title: str, out_path: Path) -> None:
    if not importances:
        return
    names = list(importances.keys())[::-1]
    values = list(importances.values())[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    ax.barh(names, values, color="#0f3460")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_model_comparison(results: dict[str, dict], out_path: Path) -> None:
    names = list(results.keys())
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, [results[n][metric] for n in names], width, label=metric)
    ax.set_xticks(x + width)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison on the held-out test set")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


LOW_SUPPORT_THRESHOLD = 30  # test flows; below this, per-class scores are not statistically meaningful


def low_support_classes(per_class: dict, threshold: int = LOW_SUPPORT_THRESHOLD) -> dict[str, int]:
    """Classes with fewer than `threshold` test flows, e.g. {"Heartbleed": 2}."""
    return {cls: int(m["support"]) for cls, m in per_class.items() if 0 < m["support"] < threshold}


def write_model_report(
    results: dict[str, dict],
    train_times: dict[str, float],
    best_name: str,
    selection_metric: str,
    importances: dict[str, float],
    md_path: Path,
) -> None:
    lines = [
        "# Model Report",
        "",
        f"Best model: **{best_name}** (selected by `{selection_metric}` on the held-out test set).",
        "",
        "## Model Comparison",
        "",
        "| Model | Accuracy | Macro F1 | Weighted F1 | Train time (s) |",
        "|---|---|---|---|---|",
    ]
    for name, res in results.items():
        lines.append(
            f"| {name} | {res['accuracy']} | {res['macro_f1']} | "
            f"{res['weighted_f1']} | {train_times[name]:.1f} |"
        )

    lines += [
        "",
        f"## Per-Class Results — {best_name}",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for cls, m in results[best_name]["per_class"].items():
        lines.append(f"| {cls} | {m['precision']} | {m['recall']} | {m['f1']} | {m['support']:,} |")

    if importances:
        lines += ["", f"## Top Features — {best_name}", "", "| Feature | Importance |", "|---|---|"]
        for feat, val in importances.items():
            lines.append(f"| {feat} | {val} |")

    lines += [
        "",
        "## Notes",
        "",
        "- The test set was never resampled, so these numbers reflect the real class balance.",
        "- Macro F1 weights every class equally, so a model that misses a rare attack class is penalized.",
        "- The best model is chosen on the same held-out test set it is reported on (there is no separate "
        "validation split), so the headline numbers carry a small optimistic bias.",
        "- Figures: `reports/figures/confusion_matrix_*.png`, `feature_importance.png`, `model_comparison.png`.",
    ]
    rare = low_support_classes(results[best_name]["per_class"])
    if rare:
        listed = ", ".join(f"{cls} ({n} test flow{'s' if n != 1 else ''})" for cls, n in rare.items())
        lines += [
            "",
            f"> **Low-support warning:** {listed} — fewer than {LOW_SUPPORT_THRESHOLD} test flows. "
            "Their precision/recall/F1 (even a perfect 1.0) are not statistically meaningful.",
        ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved {md_path}")

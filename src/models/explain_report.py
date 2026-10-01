"""Helpers for scripts/explain_flow.py: choose example flows, print and plot explanations."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def pick_example_indices(
    y_test: np.ndarray,
    class_names: list[str],
    per_class: int = 1,
    seed: int = 42,
    only: list[str] | None = None,
) -> list[tuple[int, str]]:
    """Pick `per_class` random test rows for each class that has any test rows.
    Returns [(row_index, true_class_name), ...] in class order."""
    if only:
        unknown = [name for name in only if name not in class_names]
        if unknown:
            raise ValueError(f"Unknown class(es) {unknown}. Choose from: {class_names}")
    rng = np.random.default_rng(seed)
    picks: list[tuple[int, str]] = []
    for code, name in enumerate(class_names):
        if only and name not in only:
            continue
        candidates = np.flatnonzero(np.asarray(y_test) == code)
        if len(candidates) == 0:
            continue
        chosen = rng.choice(candidates, size=min(per_class, len(candidates)), replace=False)
        picks.extend((int(i), name) for i in sorted(chosen))
    return picks


def format_explanation(result: dict, true_class: str | None = None) -> str:
    unit = "probability" if result["explanation_space"] == "probability" else "log-odds"
    verdict = "ATTACK" if result["is_attack"] else "benign"
    true_part = f"true: {true_class} -> " if true_class else ""
    lines = [
        f"=== {true_part}predicted {result['predicted_class']} "
        f"(confidence {result['confidence']:.2f}, {verdict}) ===",
        f"baseline {result['baseline']:+.3f} + feature contributions = model output ({unit} units)",
    ]
    for f in result["top_features"]:
        value = "missing" if f["raw_value"] is None else f"{f['raw_value']:,.4g}"
        lines.append(
            f"  {f['feature']:<28} value={value:<12} z={f['z_score']:+6.2f}  "
            f"{f['direction']:<7} {f['contribution']:+.4f}"
        )
    lines.append(f"  {'all other features':<28} {'':<12} {'':<9}  {'':<7} {result['other_features_contribution']:+.4f}")
    return "\n".join(lines)


def plot_explanations(results: list[dict], true_classes: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.6 * rows), squeeze=False)

    for ax, result, true_class in zip(axes.flat, results, true_classes):
        features = result["top_features"][::-1]  # largest at the top
        values = [f["contribution"] for f in features]
        colors = ["#2E6FBD" if v >= 0 else "#D9822B" for v in values]
        ax.barh([f["feature"] for f in features], values, color=colors)
        ax.axvline(0, color="#444", lw=0.8)
        ax.set_title(f"true: {true_class}  |  predicted: {result['predicted_class']} ({result['confidence']:.2f})", fontsize=9.5)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlabel("contribution (blue = toward predicted class, orange = against)", fontsize=8)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")

    fig.suptitle("SHAP: which features pushed each example flow toward its predicted class", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)

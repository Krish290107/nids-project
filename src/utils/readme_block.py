"""Builds the README "results" block from the real files written by training.

The block lives between <!-- RESULTS:START --> and <!-- RESULTS:END --> in README.md.
Every number in it comes from reports/metrics/*.json, so the README can't drift out of
date: re-run training, then `python scripts/update_readme.py`.
"""
from __future__ import annotations

import re

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
LOW_SUPPORT = 30  # test flows; fewer than this and a class's scores aren't meaningful


def render_results(metrics: dict, prep: dict) -> str:
    best = metrics["best_model"]
    models = metrics["models"]
    test_rows = int(prep["test_rows"])
    others = [f"{name} {m['macro_f1']:.4f}" for name, m in models.items() if name != best]
    compared = f" (macro F1 of the other model: {', '.join(others)})" if others else ""

    lines = [
        f"Scored on **{test_rows:,} held-out flows** the model never saw, with the real class balance "
        f"(the test set is never resampled). Best model: **{best}** with accuracy "
        f"**{models[best]['accuracy']:.4f}** and macro F1 **{models[best]['macro_f1']:.4f}**{compared}.",
        "",
        "| Class | Precision | Recall | F1 | Test flows |",
        "|---|---|---|---|---|",
    ]
    for cls, m in models[best]["per_class"].items():
        lines.append(f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {int(m['support']):,} |")

    raw_features = prep["final_feature_count"] + len(prep["dropped_correlated_columns"])  # ID columns are not features
    loaded = prep.get("rows_loaded")
    flow = f"{loaded:,} flows loaded → " if loaded else ""
    lines += [
        "",
        f"Data preparation: {flow}{prep['duplicates_removed']:,} duplicates removed → "
        f"{prep['rows_after_dedup']:,} kept ({prep['train_rows_before_balancing']:,} train / {test_rows:,} test). "
        f"SMOTE grew the training set to {prep['train_rows_after_balancing']:,} rows. "
        f"Features: {raw_features} → {prep['final_feature_count']} after dropping near-duplicates.",
    ]
    dropped = prep.get("dropped_classes") or {}
    if dropped:
        names = ", ".join(f"{k} ({v:,} rows)" for k, v in dropped.items())
        lines.append(f"Classes removed before training (too few examples): {names}.")
    rare = [c for c, m in models[best]["per_class"].items() if 0 < m["support"] < LOW_SUPPORT]
    if rare:
        lines.append(f"**Caution:** {', '.join(rare)} has fewer than {LOW_SUPPORT} test flows, so its scores are not reliable.")
    return "\n".join(lines)


def replace_block(readme: str, new_block: str) -> str:
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(readme):
        raise ValueError(f"README has no {START} ... {END} section to fill in")
    return pattern.sub(lambda _: f"{START}\n{new_block}\n{END}", readme, count=1)

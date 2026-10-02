import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.utils.readme_block import END, START, render_results, replace_block

PER_CLASS = {
    "BENIGN": {"precision": 0.9999, "recall": 0.9998, "f1": 0.9998, "support": 80000},
    "DoS Hulk": {"precision": 0.9990, "recall": 0.9980, "f1": 0.9985, "support": 30000},
}
METRICS = {
    "best_model": "random_forest",
    "models": {
        "random_forest": {"accuracy": 0.9995, "macro_f1": 0.9974, "per_class": PER_CLASS},
        "xgboost": {"accuracy": 0.9990, "macro_f1": 0.9961, "per_class": PER_CLASS},
    },
}
PREP = {
    "rows_loaded": 1100, "dropped_classes": {"Rare": 11}, "duplicates_removed": 100, "rows_after_dedup": 989,
    "train_rows_before_balancing": 791, "train_rows_after_balancing": 1500, "test_rows": 198,
    "final_feature_count": 46, "dropped_correlated_columns": ["a"] * 32, "dropped_leakage_columns": [],
}


def test_render_contains_the_real_numbers():
    text = render_results(METRICS, PREP)
    assert "198 held-out flows" in text and "random_forest" in text
    assert "0.9995" in text and "0.9974" in text and "xgboost 0.9961" in text
    assert "| DoS Hulk | 0.9990 | 0.9980 | 0.9985 | 30,000 |" in text
    assert "Features: 78 → 46" in text
    assert "Rare (11 rows)" in text and "1,100 flows loaded" in text
    assert "Caution" not in text


def test_render_warns_about_tiny_classes():
    tiny = {"best_model": "m", "models": {"m": {"accuracy": 1, "macro_f1": 1, "per_class": {
        "A": {"precision": 1, "recall": 1, "f1": 1, "support": 5000},
        "B": {"precision": 1, "recall": 1, "f1": 1, "support": 2}}}}}
    assert "Caution" in render_results(tiny, {**PREP, "dropped_classes": {}}) and "B has fewer" in render_results(tiny, PREP)


def test_render_works_without_optional_fields():
    prep = {k: v for k, v in PREP.items() if k not in ("rows_loaded", "dropped_classes")}
    text = render_results(METRICS, prep)
    assert "flows loaded" not in text and "removed before training" not in text


def test_replace_block_swaps_only_the_marked_part():
    readme = f"# Title\n\nintro\n\n{START}\nOLD\nOLD2\n{END}\n\nfooter\n"
    out = replace_block(readme, "NEW")
    assert "OLD" not in out and f"{START}\nNEW\n{END}" in out
    assert out.startswith("# Title\n\nintro") and out.endswith("footer\n")
    assert replace_block(out, "NEWER").count(START) == 1  # safe to run repeatedly


def test_replace_block_errors_clearly_without_markers():
    with pytest.raises(ValueError, match="RESULTS:START"):
        replace_block("# no markers here", "x")


def test_the_real_readme_keeps_its_results_markers():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert readme.count(START) == 1 and readme.count(END) == 1
    assert readme.index(START) < readme.index(END)


def test_id_columns_are_not_counted_as_features():
    prep = {**PREP, "dropped_leakage_columns": ["Flow ID", "Source IP"], "dropped_correlated_columns": ["a"] * 4, "final_feature_count": 16}
    assert "Features: 20 → 16" in render_results(METRICS, prep)

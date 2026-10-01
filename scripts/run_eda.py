"""Day 1-2 entry point.

Run from the project root:
    python scripts/run_eda.py

Loads data/raw/*.csv, runs the full EDA, saves plots to reports/figures/
and a report to reports/EDA_REPORT.md + reports/eda_summary.json.
"""
import sys
from pathlib import Path

# allow "python scripts/run_eda.py" to find the src package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import load_config, resolve_path, ensure_dir
from src.utils.logging_config import get_logger
from src.data.load_dataset import load_dataset, detect_target_column, describe_dataset
from src.data import eda

logger = get_logger(__name__)


def main() -> None:
    config = load_config()
    ds_cfg = config["dataset"]
    report_cfg = config["reports"]
    analysis_cfg = config["analysis"]

    figures_dir = ensure_dir(resolve_path(report_cfg["figures_dir"]))
    ensure_dir(resolve_path(report_cfg["metrics_dir"]))

    logger.info("=== Day 1-2: Loading dataset ===")
    df = load_dataset()
    target_col = detect_target_column(df, ds_cfg["target_column"])
    describe_dataset(df, target_col)

    logger.info("=== Running data quality checks ===")
    missing = eda.missing_value_report(df)
    infinite = eda.infinite_value_report(df)
    duplicates = eda.duplicate_report(df)
    dist = eda.class_distribution(df, target_col)
    leakage_cols = eda.identify_leakage_columns(df)
    corr_pairs = eda.correlated_pairs(
        df, target_col,
        threshold=analysis_cfg["correlation_threshold"],
        top_n=analysis_cfg["top_correlated_pairs"],
    )

    logger.info("=== Generating plots ===")
    eda.plot_class_distribution(dist, figures_dir / "class_distribution.png")
    eda.plot_missing_values(missing, figures_dir / "missing_values.png")
    eda.plot_correlated_pairs(corr_pairs, figures_dir / "correlation_pairs.png")
    eda.plot_feature_distributions(
        df, target_col, figures_dir / "feature_distributions.png",
        top_n=analysis_cfg["top_features_to_plot"],
    )

    logger.info("=== Writing EDA report ===")
    eda.generate_eda_report(
        df=df,
        target_col=target_col,
        dataset_name=ds_cfg["name"],
        missing=missing,
        infinite=infinite,
        duplicates=duplicates,
        dist=dist,
        leakage_cols=leakage_cols,
        corr_pairs=corr_pairs,
        json_path=resolve_path(report_cfg["eda_summary_json"]),
        md_path=resolve_path(report_cfg["eda_report_md"]),
    )

    logger.info("=== Day 1-2 complete ===")
    logger.info(f"Check: {resolve_path(report_cfg['eda_report_md'])}")
    logger.info(f"Check: {figures_dir}")


if __name__ == "__main__":
    main()

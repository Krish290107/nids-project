"""Day 1-2 exploratory data analysis.

Every function here only READS the dataframe — nothing is dropped or
modified. Recommendations are recorded for Day 3-4 preprocessing to act on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, we only save figures
import matplotlib.pyplot as plt

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Substrings that flag a column as a likely identifier / leakage risk.
# Matched case-insensitively against column names.
LEAKAGE_HINTS = ["flow id", "src ip", "source ip", "dst ip", "destination ip", "timestamp"]


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    total = df.isna().sum()
    pct = (total / len(df)) * 100
    report = pd.DataFrame({"missing_count": total, "missing_pct": pct.round(3)})
    report = report[report["missing_count"] > 0].sort_values("missing_count", ascending=False)
    logger.info(f"Missing values: {len(report)} column(s) affected, {total.sum():,} cells total")
    return report


def infinite_value_report(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    inf_counts = np.isinf(numeric_df).sum()
    report = pd.DataFrame({"infinite_count": inf_counts})
    report = report[report["infinite_count"] > 0].sort_values("infinite_count", ascending=False)
    logger.info(f"Infinite values: {len(report)} column(s) affected, {inf_counts.sum():,} cells total")
    return report


def duplicate_report(df: pd.DataFrame) -> dict:
    dup_count = int(df.duplicated().sum())
    pct = round((dup_count / len(df)) * 100, 3) if len(df) else 0.0
    logger.info(f"Duplicate rows: {dup_count:,} ({pct}%)")
    return {"duplicate_count": dup_count, "duplicate_pct": pct}


def class_distribution(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    counts = df[target_col].value_counts()
    pct = (counts / len(df) * 100).round(3)
    report = pd.DataFrame({"count": counts, "percentage": pct}).sort_values("count", ascending=False)
    logger.info(f"Class distribution ({len(report)} classes):\n{report.to_string()}")
    return report


def identify_leakage_columns(df: pd.DataFrame) -> list[str]:
    flagged = [
        c for c in df.columns
        if any(hint in c.lower() for hint in LEAKAGE_HINTS)
    ]
    if flagged:
        logger.info(f"Potential leakage-prone columns flagged for Day 3-4 review: {flagged}")
    return flagged


def correlated_pairs(df: pd.DataFrame, target_col: str, threshold: float = 0.90, top_n: int = 25) -> pd.DataFrame:
    """Return the strongest numeric feature pairs, without building a dense matrix
    of hundreds of columns into a display."""
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors="ignore")
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

    corr = numeric_df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "correlation"})
    )
    pairs = pairs[pairs["correlation"] >= threshold].sort_values("correlation", ascending=False)
    pairs = pairs.head(top_n).reset_index(drop=True)
    logger.info(f"Found {len(pairs)} feature pair(s) with correlation >= {threshold}")
    return pairs


def plot_class_distribution(dist: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    dist["count"].plot(kind="bar", ax=ax, color="#0f3460")
    ax.set_title("Class Distribution")
    ax.set_ylabel("Number of flows")
    ax.set_xlabel("Class")
    ax.set_yscale("log")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_missing_values(missing: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values found", ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        missing["missing_pct"].plot(kind="barh", ax=ax, color="#e94560")
        ax.set_xlabel("% missing")
        ax.set_title("Missing Values by Column")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_correlated_pairs(pairs: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, max(4, len(pairs) * 0.3)))
    if pairs.empty:
        ax.text(0.5, 0.5, "No highly correlated pairs found", ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        labels = pairs["feature_1"] + " / " + pairs["feature_2"]
        ax.barh(labels, pairs["correlation"], color="#16213e")
        ax.set_xlabel("Correlation")
        ax.set_title("Top Correlated Feature Pairs")
        ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_feature_distributions(df: pd.DataFrame, target_col: str, out_path: Path, top_n: int = 6) -> None:
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors="ignore")
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    variances = numeric_df.var(numeric_only=True).sort_values(ascending=False)
    features = variances.head(top_n).index.tolist()

    fig, axes = plt.subplots(len(features), 1, figsize=(8, 2.2 * len(features)))
    if len(features) == 1:
        axes = [axes]
    for ax, feature in zip(axes, features):
        numeric_df[feature].dropna().clip(
            upper=numeric_df[feature].quantile(0.99)
        ).hist(bins=50, ax=ax, color="#0f3460")
        ax.set_title(feature, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def generate_eda_report(
    df: pd.DataFrame,
    target_col: str,
    dataset_name: str,
    missing: pd.DataFrame,
    infinite: pd.DataFrame,
    duplicates: dict,
    dist: pd.DataFrame,
    leakage_cols: list[str],
    corr_pairs: pd.DataFrame,
    json_path: Path,
    md_path: Path,
) -> None:
    summary = {
        "dataset_name": dataset_name,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "target_column": target_col,
        "class_distribution": dist["count"].to_dict(),
        "missing_value_columns": missing.index.tolist(),
        "infinite_value_columns": infinite.index.tolist(),
        "duplicate_rows": duplicates,
        "leakage_prone_columns": leakage_cols,
        "high_correlation_pair_count": int(len(corr_pairs)),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved {json_path}")

    lines = [
        f"# EDA Report — {dataset_name}",
        "",
        f"- Rows: {summary['row_count']:,}",
        f"- Columns: {summary['column_count']}",
        f"- Target column: `{target_col}`",
        "",
        "## Class Distribution",
        "",
        "| Class | Count | % |",
        "|---|---|---|",
    ]
    for cls, row in dist.iterrows():
        lines.append(f"| {cls} | {int(row['count']):,} | {row['percentage']}% |")

    lines += [
        "",
        "## Data Quality",
        "",
        f"- Columns with missing values: {len(missing)}",
        f"- Columns with infinite values: {len(infinite)}",
        f"- Duplicate rows: {duplicates['duplicate_count']:,} ({duplicates['duplicate_pct']}%)",
        "",
        "## Leakage-Prone Columns (flagged for Day 3-4 review)",
        "",
    ]
    lines += [f"- `{c}`" for c in leakage_cols] if leakage_cols else ["- None detected by name-based heuristic."]

    lines += [
        "",
        f"## Highly Correlated Feature Pairs (>= threshold)",
        "",
        "| Feature 1 | Feature 2 | Correlation |",
        "|---|---|---|",
    ]
    for _, row in corr_pairs.iterrows():
        lines.append(f"| {row['feature_1']} | {row['feature_2']} | {row['correlation']:.3f} |")

    lines += [
        "",
        "## Cleaning Recommendations for Day 3-4",
        "",
        "- Drop or transform leakage-prone identifier columns listed above before training.",
        "- Replace infinite values (commonly in rate-based flow features) before scaling — "
        "typically via clipping or dropping affected rows.",
        "- Address class imbalance seen in the distribution table (e.g. SMOTE or class_weight='balanced').",
        "- Consider dropping one feature from each highly-correlated pair to reduce redundancy.",
    ]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved {md_path}")

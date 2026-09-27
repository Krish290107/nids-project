"""Load and lightly normalize the raw CICIDS2017 CSV files.

This module does NOT clean or drop anything — it only loads the data and
makes column names consistent, since CICIDS2017 mirrors are inconsistent
about leading/trailing spaces in column headers (e.g. " Label" vs "Label").
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.utils.paths import load_config, resolve_path
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def find_csv_files(raw_dir: Path) -> list[Path]:
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}.\n"
            f"Expected location: {raw_dir}\n"
            "Download a CICIDS2017 daily CSV (e.g. the Wednesday or Friday file) "
            "and place it there. See README.md 'Dataset Setup' for the exact steps."
        )
    return csv_files


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names only. Never rename feature semantics."""
    df.columns = [c.strip() for c in df.columns]
    return df


def load_dataset(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load every CSV in data/raw and concatenate them into one DataFrame.

    Raises a clear error if files are missing or if the CSVs don't share a
    consistent column set (rather than silently misaligning columns).
    """
    config = load_config()
    raw_dir = raw_dir or resolve_path(config["dataset"]["raw_dir"])
    csv_files = find_csv_files(raw_dir)

    logger.info(f"Found {len(csv_files)} CSV file(s) in {raw_dir}")

    frames = []
    reference_columns = None
    for path in csv_files:
        logger.info(f"Loading {path.name} ...")
        try:
            df = pd.read_csv(path, low_memory=False, encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(f"{path.name} is not UTF-8, retrying with latin-1")
            df = pd.read_csv(path, low_memory=False, encoding="latin-1")

        df = _normalize_columns(df)

        if reference_columns is None:
            reference_columns = set(df.columns)
        elif set(df.columns) != reference_columns:
            missing = reference_columns - set(df.columns)
            extra = set(df.columns) - reference_columns
            raise ValueError(
                f"{path.name} has a different column set than the first file loaded.\n"
                f"Missing columns: {sorted(missing)}\n"
                f"Extra columns: {sorted(extra)}\n"
                "CICIDS2017 daily files are usually consistent — check you downloaded "
                "matching files, or load one file at a time for Day 1."
            )

        frames.append(df)
        logger.info(f"  -> {df.shape[0]:,} rows, {df.shape[1]} columns")

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined dataset: {combined.shape[0]:,} rows, {combined.shape[1]} columns")
    return combined


def detect_target_column(df: pd.DataFrame, configured_name: str = "Label") -> str:
    """Find the label column even if casing/whitespace differs from config."""
    if configured_name in df.columns:
        return configured_name

    normalized_lookup = {c.strip().lower(): c for c in df.columns}
    if configured_name.strip().lower() in normalized_lookup:
        return normalized_lookup[configured_name.strip().lower()]

    raise ValueError(
        "Unable to identify target column.\n"
        f"Configured target_column: '{configured_name}'\n"
        f"Detected columns:\n{list(df.columns)}\n"
        "Please set dataset.target_column in configs/config.yaml to the exact column name."
    )


def describe_dataset(df: pd.DataFrame, target_col: str) -> None:
    """Print the Day-1 sanity-check info: shape, dtypes, sample rows."""
    logger.info(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    logger.info(f"Target column: '{target_col}'")
    logger.info("Column dtypes:\n" + df.dtypes.value_counts().to_string())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())


if __name__ == "__main__":
    cfg = load_config()
    data = load_dataset()
    target = detect_target_column(data, cfg["dataset"]["target_column"])
    describe_dataset(data, target)

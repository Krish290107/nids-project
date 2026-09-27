import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.data.load_dataset import detect_target_column, find_csv_files


def test_detect_target_column_exact_match():
    df = pd.DataFrame({"Label": ["BENIGN", "DoS"]})
    assert detect_target_column(df, "Label") == "Label"


def test_detect_target_column_handles_whitespace_casing():
    df = pd.DataFrame({" label ": ["BENIGN", "DoS"]})
    # note: real loader strips column whitespace before this is called;
    # this test checks the case-insensitive fallback specifically.
    df.columns = [c.strip() for c in df.columns]
    assert detect_target_column(df, "LABEL") == "label"


def test_detect_target_column_raises_clear_error():
    df = pd.DataFrame({"SomeColumn": [1, 2]})
    with pytest.raises(ValueError, match="Unable to identify target column"):
        detect_target_column(df, "Label")


def test_find_csv_files_raises_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError, match="No CSV files found"):
        find_csv_files(tmp_path)


def test_find_csv_files_finds_files(tmp_path):
    (tmp_path / "a.csv").write_text("col\n1\n")
    (tmp_path / "b.csv").write_text("col\n2\n")
    files = find_csv_files(tmp_path)
    assert len(files) == 2

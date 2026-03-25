"""Fast tests for preprocessing utilities."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.preprocessing.main import (
    drop_rows_with_missing_files,
    preprocess_dataframe,
    validate_required_columns,
)


def test_validate_required_columns_raises_for_missing_columns() -> None:
    dataframe = pd.DataFrame({"objid": [1], "gz2_class": ["Ei"]})

    with pytest.raises(KeyError):
        validate_required_columns(dataframe)


def test_preprocess_dataframe_drops_missing_empty_and_duplicates() -> None:
    dataframe = pd.DataFrame(
        {
            "objid": [1, 1, 2, 3, 4],
            "gz2_class": [" Ei ", "Ei", None, " ", "Er"],
            "image_path": ["img1.jpg", "img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"],
            "extra_column": [10, 11, 12, 13, 14],
        }
    )

    cleaned_df, stats = preprocess_dataframe(dataframe)

    assert list(cleaned_df.columns) == ["objid", "gz2_class", "image_path"]
    assert cleaned_df.to_dict(orient="records") == [
        {"objid": 1, "gz2_class": "Ei", "image_path": "img1.jpg"},
        {"objid": 4, "gz2_class": "Er", "image_path": "img4.jpg"},
    ]
    assert stats["rows_input"] == 5
    assert stats["rows_removed_missing"] == 1
    assert stats["rows_removed_empty"] == 1
    assert stats["rows_removed_duplicates"] == 1
    assert stats["rows_output"] == 2


def test_drop_rows_with_missing_files_filters_non_existing_paths(tmp_path: Path) -> None:
    existing_file = tmp_path / "sample.jpg"
    existing_file.write_bytes(b"test")

    dataframe = pd.DataFrame(
        {
            "objid": [1, 2],
            "gz2_class": ["Ei", "Er"],
            "image_path": [str(existing_file), str(tmp_path / "missing.jpg")],
        }
    )

    filtered_df, removed_rows = drop_rows_with_missing_files(dataframe)

    assert removed_rows == 1
    assert filtered_df["image_path"].tolist() == [str(existing_file)]

"""Tests for the reproducible train/validation/test protocol."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from galaxy_morphology_classification.training import (
    compute_class_weights,
    prepare_training_dataframe,
    save_dataset_splits,
    split_dataframe,
)


def _build_dataframe_with_files(tmp_path: Path) -> pd.DataFrame:
    records = []
    for index in range(12):
        image_path = tmp_path / f"sample_{index}.jpg"
        image_path.write_bytes(b"test")
        records.append(
            {
                "objid": index,
                "gz2_class": "Ei" if index < 6 else "Er",
                "image_path": str(image_path),
            }
        )
    return pd.DataFrame(records)


def test_prepare_training_dataframe_removes_missing_files_and_duplicates(tmp_path: Path) -> None:
    existing_file = tmp_path / "existing.jpg"
    existing_file.write_bytes(b"test")

    dataframe = pd.DataFrame(
        [
            {"objid": 1, "gz2_class": " Ei ", "image_path": str(existing_file)},
            {"objid": 1, "gz2_class": "Ei", "image_path": str(existing_file)},
            {"objid": 2, "gz2_class": "Er", "image_path": str(tmp_path / "missing.jpg")},
        ]
    )

    cleaned_df, stats = prepare_training_dataframe(dataframe)

    assert cleaned_df.to_dict(orient="records") == [
        {"objid": 1, "gz2_class": "Ei", "image_path": str(existing_file)}
    ]
    assert stats["rows_removed_duplicates"] == 1
    assert stats["rows_removed_missing_files"] == 1


def test_split_dataframe_creates_three_disjoint_stratified_splits(tmp_path: Path) -> None:
    dataframe = _build_dataframe_with_files(tmp_path)

    split_dataframes = split_dataframe(
        dataframe,
        validation_size=0.25,
        test_size=0.25,
        random_state=7,
    )

    assert sorted(split_dataframes.keys()) == ["test", "train", "validation"]
    assert sum(len(dataframe) for dataframe in split_dataframes.values()) == len(dataframe)

    train_ids = set(split_dataframes["train"]["objid"])
    validation_ids = set(split_dataframes["validation"]["objid"])
    test_ids = set(split_dataframes["test"]["objid"])
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

    assert set(split_dataframes["train"]["gz2_class"]) == {"Ei", "Er"}
    assert set(split_dataframes["validation"]["gz2_class"]) == {"Ei", "Er"}
    assert set(split_dataframes["test"]["gz2_class"]) == {"Ei", "Er"}


def test_save_dataset_splits_writes_metadata(tmp_path: Path) -> None:
    dataframe = _build_dataframe_with_files(tmp_path)
    split_dataframes = split_dataframe(dataframe)
    output_dir = tmp_path / "splits"

    save_dataset_splits(
        split_dataframes,
        output_dir,
        source_csv=tmp_path / "input.csv",
        validation_size=0.15,
        test_size=0.15,
        random_state=42,
    )

    assert (output_dir / "train.csv").exists()
    assert (output_dir / "validation.csv").exists()
    assert (output_dir / "test.csv").exists()
    assert (output_dir / "metadata.json").exists()


def test_compute_class_weights_prioritizes_minority_class(tmp_path: Path) -> None:
    dataframe = _build_dataframe_with_files(tmp_path).iloc[:9].copy()
    dataframe.loc[dataframe.index[:7], "gz2_class"] = "Ei"
    dataframe.loc[dataframe.index[7:], "gz2_class"] = "Er"

    class_weights = compute_class_weights(
        dataframe,
        class_to_index={"Ei": 0, "Er": 1},
    )

    assert class_weights[1] > class_weights[0]

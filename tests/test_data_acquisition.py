"""Fast tests for deterministic data-acquisition utilities."""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("kagglehub")

from scripts.data_acquisition.main import add_image_paths, filter_top_classes, merge_mapping_and_labels


def test_merge_mapping_and_labels_renames_and_merges_on_objid() -> None:
    mapping_df = pd.DataFrame(
        {
            "objid": [101, 102, 103],
            "asset_id": [1001, 1002, 1003],
        }
    )
    hart_df = pd.DataFrame(
        {
            "dr7objid": [102, 103, 999],
            "gz2_class": ["Ei", "Er", "Ec"],
        }
    )

    merged_df = merge_mapping_and_labels(mapping_df, hart_df)

    assert sorted(merged_df["objid"].tolist()) == [102, 103]
    assert "dr7objid" not in merged_df.columns
    assert "gz2_class" in merged_df.columns


def test_add_image_paths_creates_expected_jpg_paths() -> None:
    merged_df = pd.DataFrame(
        {
            "objid": [1, 2],
            "asset_id": [11, 22],
            "gz2_class": ["Ei", "Er"],
        }
    )
    images_root = Path("C:/datasets/images")

    result_df = add_image_paths(merged_df, images_root)

    assert result_df["image_path"].tolist() == [
        str(images_root / "11.jpg"),
        str(images_root / "22.jpg"),
    ]


def test_filter_top_classes_keeps_only_most_frequent_labels() -> None:
    merged_df = pd.DataFrame(
        {
            "objid": [1, 2, 3, 4, 5, 6],
            "gz2_class": ["Ei", "Ei", "Er", "Er", "Er", "Ec"],
            "image_path": [f"img_{index}.jpg" for index in range(6)],
        }
    )

    filtered_df = filter_top_classes(merged_df, target_column="gz2_class", top_n=2)

    assert list(filtered_df.columns) == ["objid", "gz2_class", "image_path"]
    assert set(filtered_df["gz2_class"].unique()) == {"Ei", "Er"}
    assert len(filtered_df) == 5

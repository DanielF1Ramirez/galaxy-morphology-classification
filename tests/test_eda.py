"""Fast tests for EDA helpers."""

import pandas as pd
import pytest

from scripts.eda.main import compute_class_distribution, load_filtered_dataset, save_class_distribution_plot


def test_load_filtered_dataset_raises_for_missing_file(tmp_path) -> None:
    missing_path = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        load_filtered_dataset(missing_path)


def test_compute_class_distribution_returns_expected_counts() -> None:
    dataframe = pd.DataFrame({"gz2_class": ["Ei", "Ei", "Er", "Ec", "Er", "Er"]})

    counts = compute_class_distribution(dataframe)

    assert counts.to_dict() == {"Er": 3, "Ei": 2, "Ec": 1}


def test_save_class_distribution_plot_creates_png(tmp_path) -> None:
    class_counts = pd.Series({"Ei": 10, "Er": 8, "Ec": 4})
    output_path = tmp_path / "class_distribution.png"

    save_class_distribution_plot(class_counts, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0

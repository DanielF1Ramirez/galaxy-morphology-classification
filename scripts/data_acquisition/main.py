"""Acquire and prepare the Galaxy Zoo 2 dataset for downstream modeling.

This script is responsible for:
1. Downloading the Galaxy Zoo 2 image dataset from Kaggle.
2. Downloading the Hart 2016 label file if it is not available locally.
3. Loading the filename-to-object-id mapping.
4. Merging the mapping with the Hart morphological labels.
5. Building the `image_path` column.
6. Keeping only the top-N most frequent morphology classes.
7. Saving the filtered dataset to `data/interim/merged_filtered.csv`.

Run from the repository root:

    python scripts/data_acquisition/main.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import kagglehub
import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

DATASET_NAME = "jaimetrickz/galaxy-zoo-2-images"
HART_LABELS_URL = "https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz"
DEFAULT_TOP_N_CLASSES = 5


def configure_logging() -> None:
    """Configure application logging for the data acquisition workflow."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )


def get_project_root() -> Path:
    """Return the repository root path."""
    return Path(__file__).resolve().parents[2]


def get_data_directories(project_root: Path) -> Tuple[Path, Path]:
    """Return and create the raw and interim data directories.

    Parameters
    ----------
    project_root : Path
        Repository root directory.

    Returns
    -------
    Tuple[Path, Path]
        Raw data directory and interim data directory.
    """
    raw_dir = project_root / "data" / "raw"
    interim_dir = project_root / "data" / "interim"

    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    return raw_dir, interim_dir


def download_hart_labels(hart_url: str, destination_path: Path) -> Path:
    """Download the Hart 2016 labels file if it is not already present.

    Parameters
    ----------
    hart_url : str
        Remote URL of the compressed CSV label file.
    destination_path : Path
        Local destination path for the downloaded file.

    Returns
    -------
    Path
        Path to the downloaded or existing label file.
    """
    if destination_path.exists():
        LOGGER.info("Hart 2016 labels already exist at %s", destination_path)
        return destination_path

    LOGGER.info("Downloading Hart 2016 labels from %s", hart_url)
    response = requests.get(hart_url, timeout=60)
    response.raise_for_status()

    destination_path.write_bytes(response.content)
    LOGGER.info("Hart 2016 labels saved to %s", destination_path)
    return destination_path


def download_kaggle_dataset(dataset_name: str) -> Path:
    """Download the Galaxy Zoo 2 image dataset from Kaggle.

    Parameters
    ----------
    dataset_name : str
        Kaggle dataset identifier.

    Returns
    -------
    Path
        Local path where the dataset is stored.
    """
    LOGGER.info("Downloading Kaggle dataset: %s", dataset_name)
    dataset_path = Path(kagglehub.dataset_download(dataset_name))
    LOGGER.info("Kaggle dataset available at %s", dataset_path)
    return dataset_path


def load_filename_mapping(mapping_path: Path) -> pd.DataFrame:
    """Load the filename mapping file.

    Parameters
    ----------
    mapping_path : Path
        Path to the mapping CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing object identifiers and file-related metadata.
    """
    if not mapping_path.exists():
        raise FileNotFoundError(f"Filename mapping file not found: {mapping_path}")

    LOGGER.info("Loading filename mapping from %s", mapping_path)
    dataframe = pd.read_csv(mapping_path)
    LOGGER.info("Filename mapping loaded with shape %s", dataframe.shape)
    return dataframe


def load_hart_dataframe(hart_path: Path) -> pd.DataFrame:
    """Load the compressed Hart 2016 morphological label file.

    Parameters
    ----------
    hart_path : Path
        Path to the compressed `.csv.gz` file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the Hart labels.
    """
    if not hart_path.exists():
        raise FileNotFoundError(f"Hart label file not found: {hart_path}")

    LOGGER.info("Loading Hart labels from %s", hart_path)
    dataframe = pd.read_csv(hart_path, compression="gzip")
    LOGGER.info("Hart labels loaded with shape %s", dataframe.shape)
    return dataframe


def merge_mapping_and_labels(mapping_df: pd.DataFrame, hart_df: pd.DataFrame) -> pd.DataFrame:
    """Merge filename mapping data with Hart labels.

    Notes
    -----
    The Hart file uses `dr7objid`, while the project pipeline expects `objid`.
    This function standardizes the merge key before performing the join.

    Parameters
    ----------
    mapping_df : pd.DataFrame
        Mapping DataFrame.
    hart_df : pd.DataFrame
        Hart labels DataFrame.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.
    """
    standardized_hart_df = hart_df.rename(columns={"dr7objid": "objid"})

    LOGGER.info("Merging filename mapping with Hart labels")
    merged_df = pd.merge(mapping_df, standardized_hart_df, on="objid", how="inner")
    LOGGER.info("Merged DataFrame shape: %s", merged_df.shape)
    return merged_df


def add_image_paths(merged_df: pd.DataFrame, images_root: Path) -> pd.DataFrame:
    """Create the `image_path` column for each asset.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged DataFrame containing the `asset_id` column.
    images_root : Path
        Root directory containing Galaxy Zoo 2 images.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional `image_path` column.
    """
    if "asset_id" not in merged_df.columns:
        raise KeyError("The merged DataFrame must contain an 'asset_id' column.")

    LOGGER.info("Building image paths using root directory %s", images_root)

    result_df = merged_df.copy()
    result_df["image_path"] = result_df["asset_id"].apply(
        lambda asset_id: str(images_root / f"{asset_id}.jpg")
    )

    return result_df


def filter_top_classes(
    merged_df: pd.DataFrame,
    target_column: str = "gz2_class",
    top_n: int = DEFAULT_TOP_N_CLASSES,
) -> pd.DataFrame:
    """Keep only the top-N most frequent morphology classes.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Input DataFrame containing labels and image paths.
    target_column : str, optional
        Name of the target label column.
    top_n : int, optional
        Number of most frequent classes to keep.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with `objid`, target column, and `image_path`.
    """
    if target_column not in merged_df.columns:
        raise KeyError(f"Target column not found: {target_column}")

    LOGGER.info("Selecting the top %d classes from column '%s'", top_n, target_column)
    top_classes = merged_df[target_column].value_counts().nlargest(top_n).index.tolist()
    LOGGER.info("Selected classes: %s", top_classes)

    selected_columns = ["objid", target_column, "image_path"]
    filtered_df = merged_df[selected_columns].copy()
    filtered_df = filtered_df[filtered_df[target_column].isin(top_classes)].reset_index(drop=True)

    LOGGER.info("Filtered DataFrame shape: %s", filtered_df.shape)
    return filtered_df


def save_filtered_dataset(filtered_df: pd.DataFrame, output_path: Path) -> None:
    """Save the filtered dataset to disk.

    Parameters
    ----------
    filtered_df : pd.DataFrame
        Final filtered dataset.
    output_path : Path
        Output CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_path, index=False)
    LOGGER.info("Filtered dataset saved to %s", output_path)


def resolve_dataset_paths(dataset_path: Path) -> Tuple[Path, Path]:
    """Resolve expected internal paths within the downloaded Kaggle dataset.

    Parameters
    ----------
    dataset_path : Path
        Root path returned by kagglehub.

    Returns
    -------
    Tuple[Path, Path]
        Mapping CSV path and images root directory.
    """
    mapping_path = dataset_path / "gz2_filename_mapping.csv"
    images_root = dataset_path / "images_gz2" / "images"

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Expected filename mapping file was not found: {mapping_path}"
        )

    if not images_root.exists():
        raise FileNotFoundError(
            f"Expected image directory was not found: {images_root}"
        )

    return mapping_path, images_root


def main() -> None:
    """Run the full data acquisition and preparation workflow."""
    configure_logging()

    project_root = get_project_root()
    raw_dir, interim_dir = get_data_directories(project_root)

    dataset_path = download_kaggle_dataset(DATASET_NAME)
    mapping_path, images_root = resolve_dataset_paths(dataset_path)

    hart_labels_path = raw_dir / "gz2_hart16.csv.gz"
    download_hart_labels(HART_LABELS_URL, hart_labels_path)

    mapping_df = load_filename_mapping(mapping_path)
    hart_df = load_hart_dataframe(hart_labels_path)

    merged_df = merge_mapping_and_labels(mapping_df, hart_df)
    merged_df = add_image_paths(merged_df, images_root)
    filtered_df = filter_top_classes(
        merged_df,
        target_column="gz2_class",
        top_n=DEFAULT_TOP_N_CLASSES,
    )

    output_csv = interim_dir / "merged_filtered.csv"
    save_filtered_dataset(filtered_df, output_csv)

    LOGGER.info("Data acquisition pipeline completed successfully.")


if __name__ == "__main__":
    main()
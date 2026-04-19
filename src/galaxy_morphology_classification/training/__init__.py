"""Training data and TensorFlow dataset utilities.

This module centralizes the repository training protocol:
- load and clean the filtered tabular dataset,
- generate deterministic train/validation/test splits,
- persist split metadata for reproducibility,
- build TensorFlow datasets with model-aware preprocessing,
- compute class weights for imbalanced training.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - exercised in CI without TensorFlow.
    tf = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

IMG_SIZE: tuple[int, int] = (128, 128)
REQUIRED_COLUMNS: tuple[str, ...] = ("objid", "gz2_class", "image_path")

DEFAULT_RANDOM_STATE = 42
DEFAULT_VALIDATION_SPLIT = 0.15
DEFAULT_TEST_SPLIT = 0.15

DEFAULT_INPUT_SCALING = "zero_one"
EFFICIENTNET_INPUT_SCALING = "efficientnet"

SPLIT_FILE_NAMES: dict[str, str] = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}
SPLIT_METADATA_FILE = "metadata.json"


@dataclass
class DatasetBundle:
    """Container with datasets, split metadata, and label mappings."""

    train_dataset: Any
    validation_dataset: Any
    test_dataset: Any
    class_to_index: dict[str, int]
    index_to_class: dict[int, str]
    split_dataframes: dict[str, pd.DataFrame]
    split_summary: dict[str, dict[str, Any]]
    train_class_weights: dict[int, float]
    input_scaling: str


def get_project_root() -> Path:
    """Return the repository root path."""
    return Path(__file__).resolve().parents[3]


def get_default_input_csv_path(project_root: Path | None = None) -> Path:
    """Return the preferred cleaned dataset path.

    The processed cleaned dataset is preferred when available; otherwise the
    interim merged dataset is used as the fallback input.
    """

    project_root = project_root or get_project_root()
    processed_csv = project_root / "data" / "processed" / "merged_filtered_clean.csv"
    if processed_csv.exists():
        return processed_csv
    return project_root / "data" / "interim" / "merged_filtered.csv"


def get_default_splits_dir(project_root: Path | None = None) -> Path:
    """Return the default directory where reproducible split CSVs are stored."""

    project_root = project_root or get_project_root()
    return project_root / "data" / "processed" / "splits"


def get_model_input_scaling(model_name: str) -> str:
    """Return the image scaling policy associated with a model family."""

    if model_name == "efficientnet":
        return EFFICIENTNET_INPUT_SCALING
    return DEFAULT_INPUT_SCALING


def validate_split_configuration(validation_size: float, test_size: float) -> None:
    """Validate the holdout split configuration."""

    if validation_size <= 0.0 or test_size <= 0.0:
        raise ValueError("validation_size and test_size must both be greater than 0.")

    if validation_size + test_size >= 1.0:
        raise ValueError("validation_size + test_size must be less than 1.")


def validate_training_dataframe(
    dataframe: pd.DataFrame,
    required_columns: Sequence[str] = REQUIRED_COLUMNS,
) -> None:
    """Ensure the dataframe exposes the columns required for training."""

    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")


def load_filtered_dataframe(csv_path: Path | None = None) -> pd.DataFrame:
    """Load the filtered dataset used for training and split generation."""

    csv_path = csv_path or get_default_input_csv_path()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Filtered dataset not found at {csv_path}. "
            "Run scripts/data_acquisition/main.py and scripts/preprocessing/main.py first."
        )

    dataframe = pd.read_csv(csv_path)
    validate_training_dataframe(dataframe)
    return dataframe


def prepare_training_dataframe(
    dataframe: pd.DataFrame,
    *,
    drop_missing_files: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply deterministic cleanup before split generation or training."""

    validate_training_dataframe(dataframe)

    cleaned_df = dataframe[list(REQUIRED_COLUMNS)].copy()
    rows_input = len(cleaned_df)

    cleaned_df = cleaned_df.dropna(subset=list(REQUIRED_COLUMNS))
    rows_after_dropna = len(cleaned_df)

    cleaned_df["gz2_class"] = cleaned_df["gz2_class"].astype(str).str.strip()
    cleaned_df["image_path"] = cleaned_df["image_path"].astype(str).str.strip()

    cleaned_df = cleaned_df[
        (cleaned_df["gz2_class"] != "") & (cleaned_df["image_path"] != "")
    ]
    rows_after_empty_filter = len(cleaned_df)

    cleaned_df = cleaned_df.drop_duplicates(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
    rows_after_dedup = len(cleaned_df)

    rows_removed_missing_files = 0
    if drop_missing_files:
        existing_mask = cleaned_df["image_path"].map(lambda path: Path(path).is_file())
        rows_removed_missing_files = int((~existing_mask).sum())
        cleaned_df = cleaned_df[existing_mask].reset_index(drop=True)

    if cleaned_df.empty:
        raise ValueError("No valid rows remain after applying training data preparation.")

    stats = {
        "rows_input": rows_input,
        "rows_removed_missing": rows_input - rows_after_dropna,
        "rows_removed_empty": rows_after_dropna - rows_after_empty_filter,
        "rows_removed_duplicates": rows_after_empty_filter - rows_after_dedup,
        "rows_removed_missing_files": rows_removed_missing_files,
        "rows_output": len(cleaned_df),
    }
    return cleaned_df, stats


def build_class_mappings(dataframe: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    """Build forward and reverse label mappings from class names."""

    class_names = sorted(dataframe["gz2_class"].dropna().unique())
    class_to_index = {name: index for index, name in enumerate(class_names)}
    index_to_class = {index: name for name, index in class_to_index.items()}
    return class_to_index, index_to_class


def summarize_class_distribution(dataframe: pd.DataFrame) -> dict[str, int]:
    """Return a JSON-serializable per-class frequency summary."""

    return {
        str(class_name): int(count)
        for class_name, count in dataframe["gz2_class"].value_counts().sort_index().items()
    }


def build_split_summary(split_dataframes: Mapping[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    """Build a compact summary for each persisted split."""

    summary: dict[str, dict[str, Any]] = {}
    for split_name, dataframe in split_dataframes.items():
        summary[split_name] = {
            "rows": int(len(dataframe)),
            "class_distribution": summarize_class_distribution(dataframe),
        }
    return summary


def split_dataframe(
    dataframe: pd.DataFrame,
    *,
    validation_size: float = DEFAULT_VALIDATION_SPLIT,
    test_size: float = DEFAULT_TEST_SPLIT,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, pd.DataFrame]:
    """Create deterministic train/validation/test splits."""

    validate_split_configuration(validation_size, test_size)

    cleaned_df, _ = prepare_training_dataframe(dataframe, drop_missing_files=True)
    class_to_index, _ = build_class_mappings(cleaned_df)

    working_df = cleaned_df.copy()
    working_df["label_idx"] = working_df["gz2_class"].map(class_to_index)

    holdout_size = validation_size + test_size
    try:
        train_df, holdout_df = train_test_split(
            working_df,
            test_size=holdout_size,
            random_state=random_state,
            stratify=working_df["label_idx"],
        )
    except ValueError as exc:
        raise ValueError(
            "Unable to create stratified splits. Check whether every class has enough samples "
            "after preprocessing and missing-file filtering."
        ) from exc

    relative_test_size = test_size / holdout_size
    validation_df, test_df = train_test_split(
        holdout_df,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=holdout_df["label_idx"],
    )

    split_dataframes = {
        "train": train_df.drop(columns="label_idx").reset_index(drop=True),
        "validation": validation_df.drop(columns="label_idx").reset_index(drop=True),
        "test": test_df.drop(columns="label_idx").reset_index(drop=True),
    }
    return split_dataframes


def save_dataset_splits(
    split_dataframes: Mapping[str, pd.DataFrame],
    output_dir: Path,
    *,
    source_csv: Path,
    validation_size: float,
    test_size: float,
    random_state: int,
) -> None:
    """Persist split CSV files plus reproducibility metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, filename in SPLIT_FILE_NAMES.items():
        split_dataframes[split_name].to_csv(output_dir / filename, index=False)

    metadata = {
        "source_csv": str(source_csv),
        "validation_size": validation_size,
        "test_size": test_size,
        "random_state": random_state,
        "summary": build_split_summary(split_dataframes),
    }
    with (output_dir / SPLIT_METADATA_FILE).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


def load_split_metadata(splits_dir: Path | None = None) -> dict[str, Any]:
    """Load saved split metadata from disk."""

    splits_dir = splits_dir or get_default_splits_dir()
    metadata_path = splits_dir / SPLIT_METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(f"Split metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_dataset_splits(splits_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load train/validation/test CSVs from a split directory."""

    splits_dir = splits_dir or get_default_splits_dir()
    split_dataframes: dict[str, pd.DataFrame] = {}
    for split_name, filename in SPLIT_FILE_NAMES.items():
        split_path = splits_dir / filename
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")

        split_dataframe = pd.read_csv(split_path)
        validate_training_dataframe(split_dataframe)
        split_dataframes[split_name] = split_dataframe

    return split_dataframes


def ensure_dataset_splits(
    *,
    csv_path: Path | None = None,
    splits_dir: Path | None = None,
    validation_size: float = DEFAULT_VALIDATION_SPLIT,
    test_size: float = DEFAULT_TEST_SPLIT,
    random_state: int = DEFAULT_RANDOM_STATE,
    force_rebuild: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load existing splits or create them deterministically if missing."""

    csv_path = csv_path or get_default_input_csv_path()
    splits_dir = splits_dir or get_default_splits_dir()

    split_files_exist = all((splits_dir / filename).exists() for filename in SPLIT_FILE_NAMES.values())
    metadata_exists = (splits_dir / SPLIT_METADATA_FILE).exists()

    if split_files_exist and metadata_exists and not force_rebuild:
        return load_dataset_splits(splits_dir)

    dataframe = load_filtered_dataframe(csv_path)
    split_dataframes = split_dataframe(
        dataframe,
        validation_size=validation_size,
        test_size=test_size,
        random_state=random_state,
    )
    save_dataset_splits(
        split_dataframes,
        splits_dir,
        source_csv=csv_path,
        validation_size=validation_size,
        test_size=test_size,
        random_state=random_state,
    )
    return split_dataframes


def compute_class_weights(
    training_dataframe: pd.DataFrame,
    class_to_index: Mapping[str, int],
) -> dict[int, float]:
    """Compute inverse-frequency class weights from the training split."""

    total_samples = len(training_dataframe)
    num_classes = len(class_to_index)
    if total_samples == 0 or num_classes == 0:
        raise ValueError("Cannot compute class weights from an empty training dataframe.")

    counts = training_dataframe["gz2_class"].value_counts()
    class_weights: dict[int, float] = {}
    for class_name, class_index in class_to_index.items():
        class_count = int(counts[class_name])
        class_weights[class_index] = float(total_samples / (num_classes * class_count))
    return class_weights


def _require_tensorflow() -> Any:
    """Return TensorFlow or raise an actionable ImportError."""

    if tf is None:
        raise ImportError(
            "TensorFlow is required for dataset creation and model execution. "
            "Install project dependencies from requirements.txt or environment.yml."
        )
    return tf


def _build_augmentation_layer(tf_lib: Any) -> Any:
    """Return a lightweight image augmentation stack for training only."""

    return tf_lib.keras.Sequential(
        [
            tf_lib.keras.layers.RandomFlip("horizontal"),
            tf_lib.keras.layers.RandomRotation(0.05),
            tf_lib.keras.layers.RandomZoom(0.1),
        ],
        name="train_augmentation",
    )


def _build_image_loader(
    *,
    normalization_mode: str,
    augment: bool,
) -> Any:
    """Build a `tf.data` mapping function with model-aware preprocessing."""

    tf_lib = _require_tensorflow()
    augmentation_layer = _build_augmentation_layer(tf_lib) if augment else None

    if normalization_mode not in {DEFAULT_INPUT_SCALING, EFFICIENTNET_INPUT_SCALING}:
        raise ValueError(f"Unsupported normalization_mode: {normalization_mode}")

    def _load_and_preprocess_image(image_path: Any, label: Any) -> tuple[Any, Any]:
        image_bytes = tf_lib.io.read_file(image_path)
        image = tf_lib.image.decode_jpeg(image_bytes, channels=3)
        image = tf_lib.image.resize(image, IMG_SIZE)
        image = tf_lib.cast(image, tf_lib.float32)

        if normalization_mode == DEFAULT_INPUT_SCALING:
            image = image / 255.0

        if augmentation_layer is not None:
            image = augmentation_layer(image, training=True)

        return image, label

    return _load_and_preprocess_image


def _make_dataset(
    image_paths: Sequence[str],
    labels: Sequence[int],
    *,
    batch_size: int,
    training: bool,
    normalization_mode: str,
    augment: bool = False,
) -> Any:
    """Create a TensorFlow dataset from aligned image paths and labels."""

    tf_lib = _require_tensorflow()
    if not image_paths:
        raise ValueError("Cannot create a TensorFlow dataset from zero samples.")

    dataset = tf_lib.data.Dataset.from_tensor_slices((list(image_paths), list(labels)))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        _build_image_loader(
            normalization_mode=normalization_mode,
            augment=augment,
        ),
        num_parallel_calls=tf_lib.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size).prefetch(tf_lib.data.AUTOTUNE)
    return dataset


def create_dataset_bundle(
    *,
    batch_size: int = 32,
    validation_size: float = DEFAULT_VALIDATION_SPLIT,
    test_size: float = DEFAULT_TEST_SPLIT,
    random_state: int = DEFAULT_RANDOM_STATE,
    csv_path: Path | None = None,
    splits_dir: Path | None = None,
    force_rebuild_splits: bool = False,
    input_scaling: str = DEFAULT_INPUT_SCALING,
    use_augmentation: bool = False,
) -> DatasetBundle:
    """Create train/validation/test datasets plus split metadata."""

    split_dataframes = ensure_dataset_splits(
        csv_path=csv_path,
        splits_dir=splits_dir,
        validation_size=validation_size,
        test_size=test_size,
        random_state=random_state,
        force_rebuild=force_rebuild_splits,
    )
    combined_df = pd.concat(split_dataframes.values(), ignore_index=True)
    class_to_index, index_to_class = build_class_mappings(combined_df)

    train_df = split_dataframes["train"].copy()
    validation_df = split_dataframes["validation"].copy()
    test_df = split_dataframes["test"].copy()

    train_labels = train_df["gz2_class"].map(class_to_index).tolist()
    validation_labels = validation_df["gz2_class"].map(class_to_index).tolist()
    test_labels = test_df["gz2_class"].map(class_to_index).tolist()

    train_dataset = _make_dataset(
        train_df["image_path"].tolist(),
        train_labels,
        batch_size=batch_size,
        training=True,
        normalization_mode=input_scaling,
        augment=use_augmentation,
    )
    validation_dataset = _make_dataset(
        validation_df["image_path"].tolist(),
        validation_labels,
        batch_size=batch_size,
        training=False,
        normalization_mode=input_scaling,
    )
    test_dataset = _make_dataset(
        test_df["image_path"].tolist(),
        test_labels,
        batch_size=batch_size,
        training=False,
        normalization_mode=input_scaling,
    )

    return DatasetBundle(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        class_to_index=class_to_index,
        index_to_class=index_to_class,
        split_dataframes={
            "train": train_df,
            "validation": validation_df,
            "test": test_df,
        },
        split_summary=build_split_summary(split_dataframes),
        train_class_weights=compute_class_weights(train_df, class_to_index),
        input_scaling=input_scaling,
    )


def create_datasets(
    *,
    batch_size: int = 32,
    val_split: float = DEFAULT_VALIDATION_SPLIT,
    test_split: float = DEFAULT_TEST_SPLIT,
    random_state: int = DEFAULT_RANDOM_STATE,
    csv_path: Path | None = None,
    splits_dir: Path | None = None,
    force_rebuild_splits: bool = False,
    input_scaling: str = DEFAULT_INPUT_SCALING,
    use_augmentation: bool = False,
) -> tuple[Any, Any, dict[str, int], dict[int, str]]:
    """Backward-compatible wrapper that returns train and validation datasets."""

    bundle = create_dataset_bundle(
        batch_size=batch_size,
        validation_size=val_split,
        test_size=test_split,
        random_state=random_state,
        csv_path=csv_path,
        splits_dir=splits_dir,
        force_rebuild_splits=force_rebuild_splits,
        input_scaling=input_scaling,
        use_augmentation=use_augmentation,
    )
    return (
        bundle.train_dataset,
        bundle.validation_dataset,
        bundle.class_to_index,
        bundle.index_to_class,
    )

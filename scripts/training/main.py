"""Train image classification models for the Galaxy Morphology Classification project.

Usage examples from the repository root:

    python scripts/training/main.py --model baseline --epochs 5
    python scripts/training/main.py --model efficientnet --epochs 5 --fine-tune-epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import tensorflow as tf

from galaxy_morphology_classification.evaluation import (
    build_model_comparison,
    evaluate_model,
    extract_split_metrics,
    load_metrics_payload,
)
from galaxy_morphology_classification.models import build_baseline_cnn
from galaxy_morphology_classification.models.model_efficient import (
    build_efficientnet_b0,
    configure_efficientnet_fine_tuning,
)
from galaxy_morphology_classification.training import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SPLIT,
    DEFAULT_VALIDATION_SPLIT,
    IMG_SIZE,
    create_dataset_bundle,
    get_model_input_scaling,
    get_project_root,
)

LOGGER = logging.getLogger(__name__)
PRIMARY_SELECTION_METRIC = "macro_f1"


def configure_logging() -> None:
    """Configure the logging system for the training script."""

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train galaxy morphology classification models.",
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "efficientnet"],
        default="baseline",
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of head-training epochs.",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=3,
        help="Additional fine-tuning epochs for transfer learning models.",
    )
    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=20,
        help="Number of EfficientNet backbone layers to unfreeze during fine-tuning.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=DEFAULT_VALIDATION_SPLIT,
        help="Validation split fraction.",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=DEFAULT_TEST_SPLIT,
        help="Blind test split fraction.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the initial training stage.",
    )
    parser.add_argument(
        "--fine-tune-learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the fine-tuning stage.",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.2,
        help="Dropout rate used by the transfer-learning classifier head.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for deterministic split generation.",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Optional path to the cleaned dataset CSV.",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        help="Optional path to the directory containing split CSVs.",
    )
    parser.add_argument(
        "--force-rebuild-splits",
        action="store_true",
        help="Rebuild train/validation/test split CSVs before training.",
    )
    parser.add_argument(
        "--disable-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weighting during training.",
    )
    parser.add_argument(
        "--disable-augmentation",
        action="store_true",
        help="Disable training-time image augmentation.",
    )
    parser.add_argument(
        "--use-mixed-precision",
        action="store_true",
        help="Enable mixed precision when the local TensorFlow runtime supports it.",
    )
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace) -> None:
    """Apply runtime configuration such as mixed precision and seeds."""

    tf.keras.utils.set_random_seed(args.random_state)
    if args.use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        LOGGER.info("Mixed precision enabled with policy mixed_float16.")


def build_model(
    model_name: Literal["baseline", "efficientnet"],
    *,
    input_shape: tuple[int, int, int],
    num_classes: int,
    dropout_rate: float,
) -> tf.keras.Model:
    """Build the requested model architecture."""

    if model_name == "baseline":
        return build_baseline_cnn(input_shape=input_shape, num_classes=num_classes)

    if model_name == "efficientnet":
        return build_efficientnet_b0(
            input_shape=input_shape,
            num_classes=num_classes,
            train_base=False,
            dropout_rate=dropout_rate,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def compile_model(model: tf.keras.Model, *, learning_rate: float) -> None:
    """Compile a model with the standard classification setup."""

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def build_callbacks(checkpoint_path: Path) -> list[tf.keras.callbacks.Callback]:
    """Build callbacks shared across training phases."""

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def serialize_history(history: tf.keras.callbacks.History | None) -> dict[str, list[float]]:
    """Convert a Keras history object into JSON-serializable values."""

    if history is None:
        return {}

    return {
        metric_name: [float(value) for value in values]
        for metric_name, values in history.history.items()
    }


def update_selection_artifacts(metrics_dir: Path) -> dict[str, Any] | None:
    """Recompute the selected model and comparison summary from tracked metrics."""

    candidate_payloads: dict[str, dict[str, Any]] = {}
    metric_paths = sorted(metrics_dir.glob("*_metrics.json"))
    for metrics_path in metric_paths:
        payload = load_metrics_payload(metrics_path)
        try:
            validation_metrics = extract_split_metrics(payload, "validation")
        except KeyError:
            continue
        candidate_payloads[payload["model"]] = validation_metrics

    if not candidate_payloads:
        return None

    comparison_summary = build_model_comparison(
        candidate_payloads,
        primary_metric=PRIMARY_SELECTION_METRIC,
    )
    selected_model_name = comparison_summary["selected_model"]

    for metrics_path in metric_paths:
        payload = load_metrics_payload(metrics_path)
        payload["is_selected_model"] = payload.get("model") == selected_model_name
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

    comparison_path = metrics_dir / "model_comparison.json"
    with comparison_path.open("w", encoding="utf-8") as file:
        json.dump(comparison_summary, file, indent=2, ensure_ascii=False)

    return comparison_summary


def main() -> None:
    """Run the full training workflow."""

    configure_logging()
    args = parse_args()
    configure_runtime(args)

    project_root = get_project_root()
    models_dir = project_root / "models"
    metrics_dir = project_root / "reports" / "metrics"
    csv_path = Path(args.input_csv) if args.input_csv else None
    splits_dir = Path(args.splits_dir) if args.splits_dir else None

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    input_scaling = get_model_input_scaling(args.model)
    LOGGER.info("Preparing reproducible train/validation/test splits and datasets.")
    bundle = create_dataset_bundle(
        batch_size=args.batch_size,
        validation_size=args.validation_split,
        test_size=args.test_split,
        random_state=args.random_state,
        csv_path=csv_path,
        splits_dir=splits_dir,
        force_rebuild_splits=args.force_rebuild_splits,
        input_scaling=input_scaling,
        use_augmentation=not args.disable_augmentation,
    )

    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    num_classes = len(bundle.class_to_index)

    LOGGER.info("Building '%s' model.", args.model)
    model = build_model(
        model_name=args.model,
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=args.dropout_rate,
    )
    compile_model(model, learning_rate=args.learning_rate)

    model_name = "baseline_cnn" if args.model == "baseline" else "efficientnet_b0"
    checkpoint_path = models_dir / f"{model_name}.keras"
    class_weights = None if args.disable_class_weights else bundle.train_class_weights
    callbacks = build_callbacks(checkpoint_path)

    LOGGER.info("Starting stage 1 training for %d epochs.", args.epochs)
    head_history = model.fit(
        bundle.train_dataset,
        validation_data=bundle.validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    fine_tune_history = None
    if args.model == "efficientnet" and args.fine_tune_epochs > 0:
        LOGGER.info(
            "Starting stage 2 fine-tuning for %d epochs over the last %d backbone layers.",
            args.fine_tune_epochs,
            args.fine_tune_layers,
        )
        configure_efficientnet_fine_tuning(model, args.fine_tune_layers)
        compile_model(model, learning_rate=args.fine_tune_learning_rate)
        fine_tune_history = model.fit(
            bundle.train_dataset,
            validation_data=bundle.validation_dataset,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
            class_weight=class_weights,
        )

    LOGGER.info("Evaluating the trained model on the validation split.")
    validation_metrics = evaluate_model(
        model,
        split_name="validation",
        batch_size=args.batch_size,
        validation_size=args.validation_split,
        test_size=args.test_split,
        random_state=args.random_state,
        csv_path=csv_path,
        splits_dir=splits_dir,
        input_scaling=input_scaling,
    )

    metrics = {
        "model": model_name,
        "model_family": args.model,
        "primary_selection_metric": PRIMARY_SELECTION_METRIC,
        "epochs": args.epochs,
        "fine_tune_epochs": args.fine_tune_epochs if args.model == "efficientnet" else 0,
        "fine_tune_layers": args.fine_tune_layers if args.model == "efficientnet" else 0,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "fine_tune_learning_rate": (
            args.fine_tune_learning_rate if args.model == "efficientnet" else None
        ),
        "dropout_rate": args.dropout_rate,
        "validation_split": args.validation_split,
        "test_split": args.test_split,
        "random_state": args.random_state,
        "input_scaling": input_scaling,
        "class_weights_enabled": not args.disable_class_weights,
        "augmentation_enabled": not args.disable_augmentation,
        "class_to_index": bundle.class_to_index,
        "index_to_class": {
            str(index): class_name for index, class_name in bundle.index_to_class.items()
        },
        "split_summary": bundle.split_summary,
        "validation_metrics": validation_metrics,
        "history": {
            "head_training": serialize_history(head_history),
            "fine_tuning": serialize_history(fine_tune_history),
        },
        "test_locked": True,
        "model_path": str(checkpoint_path),
    }

    metrics_path = metrics_dir / f"{model_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    comparison_summary = update_selection_artifacts(metrics_dir)

    LOGGER.info("Training completed successfully.")
    LOGGER.info("Model saved to: %s", checkpoint_path)
    LOGGER.info("Metrics saved to: %s", metrics_path)
    if comparison_summary is not None:
        LOGGER.info("Selected model after comparison: %s", comparison_summary["selected_model"])


if __name__ == "__main__":
    main()

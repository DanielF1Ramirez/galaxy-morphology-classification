"""Train image classification models for the Galaxy Morphology Classification project.

Usage examples from the repository root:

    python scripts/training/main.py --model baseline --epochs 5
    python scripts/training/main.py --model efficientnet --epochs 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Literal

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from galaxy_morphology_classification.models import build_baseline_cnn
from galaxy_morphology_classification.models.model_efficient import build_efficientnet_b0
from galaxy_morphology_classification.training import IMG_SIZE, create_datasets, get_project_root

LOGGER = logging.getLogger(__name__)


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
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split fraction.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the Adam optimizer.",
    )
    return parser.parse_args()


def build_model(
    model_name: Literal["baseline", "efficientnet"],
    input_shape: tuple[int, int, int],
    num_classes: int,
) -> tf.keras.Model:
    """Build the requested model architecture.

    Parameters
    ----------
    model_name : Literal["baseline", "efficientnet"]
        Name of the model architecture.
    input_shape : tuple[int, int, int]
        Input image shape.
    num_classes : int
        Number of output classes.

    Returns
    -------
    tf.keras.Model
        Uncompiled Keras model.
    """
    if model_name == "baseline":
        return build_baseline_cnn(input_shape=input_shape, num_classes=num_classes)

    if model_name == "efficientnet":
        return build_efficientnet_b0(
            input_shape=input_shape,
            num_classes=num_classes,
            train_base=False,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def main() -> None:
    """Run the full training workflow."""
    configure_logging()
    args = parse_args()

    project_root = get_project_root()
    models_dir = project_root / "models"
    metrics_dir = project_root / "reports" / "metrics"

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Creating training and validation datasets.")
    train_dataset, validation_dataset, class_to_index, index_to_class = create_datasets(
        batch_size=args.batch_size,
        val_split=args.val_split,
    )

    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    num_classes = len(class_to_index)

    LOGGER.info("Building '%s' model.", args.model)
    model = build_model(
        model_name=args.model,
        input_shape=input_shape,
        num_classes=num_classes,
    )

    LOGGER.info("Compiling model.")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_name = "baseline_cnn" if args.model == "baseline" else "efficientnet_b0"
    checkpoint_path = models_dir / f"{model_name}.keras"

    callbacks = [
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
    ]

    LOGGER.info("Starting training for %d epochs.", args.epochs)
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    LOGGER.info("Evaluating final model on the validation split.")
    val_loss, val_accuracy = model.evaluate(validation_dataset, verbose=0)

    metrics = {
        "model": model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "val_split": args.val_split,
        "val_loss": float(val_loss),
        "val_accuracy": float(val_accuracy),
        "class_to_index": class_to_index,
        "index_to_class": {str(index): name for index, name in index_to_class.items()},
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
    }

    metrics_path = metrics_dir / f"{model_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    LOGGER.info("Training completed successfully.")
    LOGGER.info("Model saved to: %s", checkpoint_path)
    LOGGER.info("Metrics saved to: %s", metrics_path)


if __name__ == "__main__":
    main()
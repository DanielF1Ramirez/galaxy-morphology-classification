"""Evaluation utilities for trained galaxy morphology classification models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from galaxy_morphology_classification.training import create_datasets


def load_model(model_path: Path) -> tf.keras.Model:
    """Load a serialized Keras model from disk.

    Parameters
    ----------
    model_path : Path
        Path to a `.keras` model file.

    Returns
    -------
    tf.keras.Model
        Loaded Keras model.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return tf.keras.models.load_model(model_path)


def evaluate_model(
    model: tf.keras.Model,
    batch_size: int = 32,
    val_split: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """Run inference on the validation split and return labels and predictions.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model.
    batch_size : int, optional
        Batch size used to recreate the validation dataset.
    val_split : float, optional
        Validation fraction. It must match the training split configuration.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict[int, str]]
        True labels, predicted labels, and the index-to-class mapping.
    """
    _, validation_dataset, _, index_to_class = create_datasets(
        batch_size=batch_size,
        val_split=val_split,
    )

    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in validation_dataset:
        predictions = model.predict(images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predicted_classes.tolist())

    return np.array(y_true), np.array(y_pred), index_to_class


def generate_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index_to_class: Dict[int, str],
) -> str:
    """Generate a human-readable classification report.

    Parameters
    ----------
    y_true : np.ndarray
        True encoded labels.
    y_pred : np.ndarray
        Predicted encoded labels.
    index_to_class : Dict[int, str]
        Mapping from encoded labels to class names.

    Returns
    -------
    str
        Formatted classification report.
    """
    label_order = sorted(index_to_class.keys())
    target_names = [index_to_class[index] for index in label_order]

    return classification_report(
        y_true,
        y_pred,
        labels=label_order,
        target_names=target_names,
        zero_division=0,
    )


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Compute the confusion matrix for a classification task.

    Parameters
    ----------
    y_true : np.ndarray
        True encoded labels.
    y_pred : np.ndarray
        Predicted encoded labels.

    Returns
    -------
    np.ndarray
        Confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)
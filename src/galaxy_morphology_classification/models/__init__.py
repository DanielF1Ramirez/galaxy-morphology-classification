"""Baseline convolutional model definitions for galaxy morphology classification."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models


def build_baseline_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
) -> tf.keras.Model:
    """Build a simple CNN baseline for image classification.

    Parameters
    ----------
    input_shape : Tuple[int, int, int]
        Shape of the input images as (height, width, channels).
    num_classes : int
        Number of target classes.

    Returns
    -------
    tf.keras.Model
        Uncompiled Keras model.
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="baseline_cnn",
    )
    return model
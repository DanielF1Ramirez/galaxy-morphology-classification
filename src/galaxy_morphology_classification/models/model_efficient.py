"""Transfer learning model definitions based on EfficientNetB0."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers


def build_efficientnet_b0(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    train_base: bool = False,
    dropout_rate: float = 0.2,
) -> tf.keras.Model:
    """Build an EfficientNetB0-based classifier.

    Parameters
    ----------
    input_shape : Tuple[int, int, int]
        Shape of the input images as (height, width, channels).
    num_classes : int
        Number of target classes.
    train_base : bool, optional
        Whether to unfreeze the EfficientNet backbone for fine-tuning.
    dropout_rate : float, optional
        Dropout rate applied before the classification head.

    Returns
    -------
    tf.keras.Model
        Uncompiled Keras model.
    """
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = train_base
    base_model._name = "efficientnet_backbone"

    inputs = tf.keras.Input(shape=input_shape, name="image")
    features = base_model(inputs, training=False)
    features = layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    features = layers.Dropout(dropout_rate, name="dropout")(features)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="classification_head",
    )(features)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="efficientnet_b0_classifier",
    )
    return model


def configure_efficientnet_fine_tuning(
    model: tf.keras.Model,
    fine_tune_layers: int,
) -> None:
    """Unfreeze the last N backbone layers while keeping batch norm frozen."""

    if fine_tune_layers <= 0:
        raise ValueError("fine_tune_layers must be greater than 0.")

    backbone = model.get_layer("efficientnet_backbone")
    backbone.trainable = True

    frozen_layers = max(len(backbone.layers) - fine_tune_layers, 0)
    for layer_index, layer in enumerate(backbone.layers):
        layer.trainable = layer_index >= frozen_layers
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

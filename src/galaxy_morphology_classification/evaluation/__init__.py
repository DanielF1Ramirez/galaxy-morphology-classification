"""Evaluation helpers for comparable offline model assessment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from galaxy_morphology_classification.training import (
    DEFAULT_INPUT_SCALING,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SPLIT,
    DEFAULT_VALIDATION_SPLIT,
    create_dataset_bundle,
)

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - exercised in CI without TensorFlow.
    tf = None  # type: ignore[assignment]


def _require_tensorflow() -> Any:
    """Return TensorFlow or raise an actionable ImportError."""

    if tf is None:
        raise ImportError(
            "TensorFlow is required to load trained models and run offline evaluation."
        )
    return tf


def load_model(model_path: Path) -> Any:
    """Load a serialized Keras model from disk."""

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    tf_lib = _require_tensorflow()
    return tf_lib.keras.models.load_model(model_path)


def load_metrics_payload(metrics_path: Path) -> dict[str, Any]:
    """Load a JSON metrics payload from disk."""

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def generate_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index_to_class: Mapping[int, str],
) -> str:
    """Generate a human-readable classification report."""

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
    """Compute the confusion matrix for a classification task."""

    return confusion_matrix(y_true, y_pred)


def compute_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index_to_class: Mapping[int, str],
) -> dict[str, Any]:
    """Compute comparable classification metrics and per-class details."""

    label_order = sorted(index_to_class.keys())
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_order,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_order,
        average="weighted",
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=label_order,
        target_names=[index_to_class[index] for index in label_order],
        output_dict=True,
        zero_division=0,
    )

    confusion = compute_confusion_matrix(y_true, y_pred)
    return {
        "num_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": confusion.tolist(),
        "classification_report": report_dict,
        "classification_report_text": generate_report(y_true, y_pred, index_to_class),
    }


def evaluate_model(
    model: Any,
    *,
    split_name: str = "validation",
    batch_size: int = 32,
    validation_size: float = DEFAULT_VALIDATION_SPLIT,
    test_size: float = DEFAULT_TEST_SPLIT,
    random_state: int = DEFAULT_RANDOM_STATE,
    csv_path: Path | None = None,
    splits_dir: Path | None = None,
    input_scaling: str = DEFAULT_INPUT_SCALING,
) -> dict[str, Any]:
    """Run offline evaluation against a named split."""

    if split_name not in {"validation", "test"}:
        raise ValueError("split_name must be either 'validation' or 'test'.")

    bundle = create_dataset_bundle(
        batch_size=batch_size,
        validation_size=validation_size,
        test_size=test_size,
        random_state=random_state,
        csv_path=csv_path,
        splits_dir=splits_dir,
        input_scaling=input_scaling,
        use_augmentation=False,
    )
    dataset = bundle.validation_dataset if split_name == "validation" else bundle.test_dataset

    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predicted_classes.tolist())

    metrics = compute_prediction_metrics(
        np.array(y_true),
        np.array(y_pred),
        bundle.index_to_class,
    )
    metrics["split_name"] = split_name
    metrics["class_to_index"] = bundle.class_to_index
    metrics["index_to_class"] = {
        str(index): class_name for index, class_name in bundle.index_to_class.items()
    }
    return metrics


def save_confusion_matrix_figure(
    confusion: np.ndarray | list[list[int]],
    index_to_class: Mapping[int, str],
    output_path: Path,
    *,
    title: str,
) -> None:
    """Save a confusion matrix heatmap to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_array = np.array(confusion)
    ordered_labels = [index_to_class[index] for index in sorted(index_to_class.keys())]

    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(confusion_array, cmap="Blues")
    axis.figure.colorbar(image, ax=axis)

    axis.set_title(title)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(range(len(ordered_labels)))
    axis.set_xticklabels(ordered_labels, rotation=45, ha="right")
    axis.set_yticks(range(len(ordered_labels)))
    axis.set_yticklabels(ordered_labels)

    for row_index in range(confusion_array.shape[0]):
        for column_index in range(confusion_array.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(confusion_array[row_index, column_index]),
                ha="center",
                va="center",
                color="black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def build_model_comparison(
    named_payloads: Mapping[str, Mapping[str, Any]],
    *,
    primary_metric: str = "macro_f1",
) -> dict[str, Any]:
    """Build a comparable ranking summary from multiple metrics payloads."""

    comparison_rows: list[dict[str, Any]] = []
    for model_name, payload in named_payloads.items():
        comparison_rows.append(
            {
                "model": model_name,
                "primary_metric": primary_metric,
                "primary_score": float(payload[primary_metric]),
                "accuracy": float(payload["accuracy"]),
                "macro_f1": float(payload["macro_f1"]),
                "weighted_f1": float(payload["weighted_f1"]),
                "num_samples": int(payload["num_samples"]),
                "split_name": payload.get("split_name", "validation"),
            }
        )

    ranking = sorted(
        comparison_rows,
        key=lambda item: (item["primary_score"], item["accuracy"]),
        reverse=True,
    )
    return {
        "primary_metric": primary_metric,
        "ranking": ranking,
        "selected_model": ranking[0]["model"] if ranking else None,
    }


def extract_split_metrics(payload: Mapping[str, Any], split_name: str) -> dict[str, Any]:
    """Extract metrics for a given split from a training or evaluation payload."""

    if payload.get("split_name") == split_name:
        return dict(payload)

    key = f"{split_name}_metrics"
    if key in payload:
        return dict(payload[key])

    raise KeyError(f"Payload does not contain metrics for split '{split_name}'.")

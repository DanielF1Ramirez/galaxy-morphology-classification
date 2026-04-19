"""Tests for comparable evaluation helpers."""

from __future__ import annotations

import numpy as np

from galaxy_morphology_classification.evaluation import (
    build_model_comparison,
    compute_prediction_metrics,
    extract_split_metrics,
)


def test_compute_prediction_metrics_returns_expected_core_scores() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    index_to_class = {0: "Ei", 1: "Er"}

    metrics = compute_prediction_metrics(y_true, y_pred, index_to_class)

    assert metrics["num_samples"] == 4
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] > 0
    assert metrics["weighted_f1"] > 0
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert "Ei" in metrics["classification_report"]


def test_build_model_comparison_ranks_by_primary_metric() -> None:
    comparison = build_model_comparison(
        {
            "baseline_cnn": {
                "accuracy": 0.80,
                "macro_f1": 0.78,
                "weighted_f1": 0.79,
                "num_samples": 100,
                "split_name": "validation",
            },
            "efficientnet_b0": {
                "accuracy": 0.83,
                "macro_f1": 0.81,
                "weighted_f1": 0.82,
                "num_samples": 100,
                "split_name": "validation",
            },
        }
    )

    assert comparison["selected_model"] == "efficientnet_b0"
    assert comparison["ranking"][0]["model"] == "efficientnet_b0"


def test_extract_split_metrics_supports_training_payload_shape() -> None:
    payload = {
        "model": "baseline_cnn",
        "validation_metrics": {
            "accuracy": 0.8,
            "macro_f1": 0.79,
            "weighted_f1": 0.8,
            "num_samples": 100,
        },
    }

    metrics = extract_split_metrics(payload, "validation")

    assert metrics["accuracy"] == 0.8
    assert metrics["macro_f1"] == 0.79

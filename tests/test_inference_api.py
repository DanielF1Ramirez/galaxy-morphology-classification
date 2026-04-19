"""Tests for the FastAPI inference service wiring."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from scripts.evaluation.main import (
    RuntimeArtifacts,
    create_app,
    select_model_artifacts,
)


class _StubModel:
    def predict(self, inputs, verbose: int = 0) -> np.ndarray:
        del inputs
        del verbose
        return np.array([[0.1, 0.9]], dtype=np.float32)


def _make_runtime() -> RuntimeArtifacts:
    return RuntimeArtifacts(
        model_name="baseline_cnn",
        model_path=Path("models/baseline_cnn.keras"),
        metrics_path=Path("reports/metrics/baseline_cnn_metrics.json"),
        input_scaling="zero_one",
        index_to_class={0: "Ei", 1: "Er"},
        model=_StubModel(),
    )


def test_select_model_artifacts_prefers_selected_model_flag(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    metrics_dir = tmp_path / "metrics"
    models_dir.mkdir()
    metrics_dir.mkdir()

    (models_dir / "baseline_cnn.keras").write_bytes(b"baseline")
    (models_dir / "efficientnet_b0.keras").write_bytes(b"efficient")
    (metrics_dir / "baseline_cnn_metrics.json").write_text(
        json.dumps({"model": "baseline_cnn", "is_selected_model": False}),
        encoding="utf-8",
    )
    (metrics_dir / "efficientnet_b0_metrics.json").write_text(
        json.dumps({"model": "efficientnet_b0", "is_selected_model": True}),
        encoding="utf-8",
    )

    model_name, model_path, metrics_path = select_model_artifacts(models_dir, metrics_dir)

    assert model_name == "efficientnet_b0"
    assert model_path.name == "efficientnet_b0.keras"
    assert metrics_path.name == "efficientnet_b0_metrics.json"


def test_api_predict_endpoint_returns_probabilities() -> None:
    app = create_app(runtime_getter=_make_runtime)
    client = TestClient(app)

    image = Image.new("RGB", (32, 32), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("sample.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_class"] == "Er"
    assert payload["predicted_index"] == 1
    assert set(payload["probabilities"]) == {"Ei", "Er"}


def test_api_rejects_non_image_upload() -> None:
    app = create_app(runtime_getter=_make_runtime)
    client = TestClient(app)

    response = client.post(
        "/predict",
        files={"file": ("sample.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415

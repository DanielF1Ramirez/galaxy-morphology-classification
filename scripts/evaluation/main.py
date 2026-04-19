"""Serve a trained galaxy morphology classification model through FastAPI.

Recommended usage from the repository root:

    uvicorn scripts.evaluation.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import logging
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - exercised in CI without TensorFlow.
    tf = None  # type: ignore[assignment]

from galaxy_morphology_classification.evaluation import load_metrics_payload
from galaxy_morphology_classification.training import (
    DEFAULT_INPUT_SCALING,
    EFFICIENTNET_INPUT_SCALING,
    IMG_SIZE,
)

LOGGER = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"


@dataclass(frozen=True)
class RuntimeArtifacts:
    """Loaded model metadata and runtime dependencies for the API."""

    model_name: str
    model_path: Path
    metrics_path: Path
    input_scaling: str
    index_to_class: dict[int, str]
    model: Any


class PredictionResponse(BaseModel):
    """Schema returned by the prediction endpoint."""

    predicted_class: str
    predicted_index: int
    probabilities: dict[str, float]


def _require_tensorflow() -> Any:
    """Return TensorFlow or raise an actionable startup error."""

    if tf is None:
        raise RuntimeError(
            "TensorFlow is required to run the FastAPI inference service. "
            "Install project dependencies before starting the API."
        )
    return tf


def _candidate_model_names(metrics_dir: Path) -> list[str]:
    """Return candidate model names ordered by selection priority."""

    preferred_metrics = sorted(metrics_dir.glob("*_metrics.json"))
    selected_names: list[str] = []
    for metrics_path in preferred_metrics:
        payload = load_metrics_payload(metrics_path)
        if payload.get("is_selected_model"):
            selected_names.append(str(payload["model"]))

    fallback_names = ["efficientnet_b0", "baseline_cnn"]
    for name in fallback_names:
        if name not in selected_names:
            selected_names.append(name)
    return selected_names


def select_model_artifacts(
    models_dir: Path = MODELS_DIR,
    metrics_dir: Path = METRICS_DIR,
) -> tuple[str, Path, Path]:
    """Select the active model and its metrics payload."""

    for model_name in _candidate_model_names(metrics_dir):
        model_path = models_dir / f"{model_name}.keras"
        metrics_path = metrics_dir / f"{model_name}_metrics.json"
        if model_path.exists() and metrics_path.exists():
            return model_name, model_path, metrics_path

    raise RuntimeError(
        "No valid model/metrics pair was found. Run scripts/training/main.py "
        "and optionally scripts/evaluation/offline.py before starting the API."
    )


def load_runtime_artifacts(
    *,
    model_loader: Callable[[str], Any] | None = None,
) -> RuntimeArtifacts:
    """Load the selected model plus its metadata."""

    model_name, model_path, metrics_path = select_model_artifacts()
    payload = load_metrics_payload(metrics_path)

    raw_index_to_class = payload.get("index_to_class")
    if raw_index_to_class is None:
        raise RuntimeError(
            f"Metrics file {metrics_path} does not contain the 'index_to_class' mapping."
        )

    input_scaling = payload.get("input_scaling", DEFAULT_INPUT_SCALING)
    if input_scaling not in {DEFAULT_INPUT_SCALING, EFFICIENTNET_INPUT_SCALING}:
        raise RuntimeError(f"Unsupported input_scaling value: {input_scaling}")

    loader = model_loader
    if loader is None:
        tf_lib = _require_tensorflow()
        loader = tf_lib.keras.models.load_model

    LOGGER.info("Loading model from %s", model_path)
    model = loader(str(model_path))
    return RuntimeArtifacts(
        model_name=model_name,
        model_path=model_path,
        metrics_path=metrics_path,
        input_scaling=input_scaling,
        index_to_class={
            int(index): class_name for index, class_name in raw_index_to_class.items()
        },
        model=model,
    )


@lru_cache(maxsize=1)
def get_runtime() -> RuntimeArtifacts:
    """Return a cached runtime instance."""

    return load_runtime_artifacts()


def _preprocess_image(image: Image.Image, *, input_scaling: str) -> np.ndarray:
    """Convert an input image into a model-ready NumPy array."""

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    array = np.array(image).astype("float32")
    if input_scaling == DEFAULT_INPUT_SCALING:
        array = array / 255.0
    elif input_scaling != EFFICIENTNET_INPUT_SCALING:
        raise ValueError(f"Unsupported input_scaling value: {input_scaling}")

    return np.expand_dims(array, axis=0)


def _predict_from_bytes(
    image_bytes: bytes,
    *,
    runtime: RuntimeArtifacts,
) -> PredictionResponse:
    """Run inference from raw image bytes."""

    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be interpreted as a valid image.",
        ) from exc

    input_array = _preprocess_image(pil_image, input_scaling=runtime.input_scaling)
    predictions = runtime.model.predict(input_array, verbose=0)
    probabilities = predictions[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = runtime.index_to_class.get(predicted_index, str(predicted_index))

    probabilities_by_class: dict[str, float] = {}
    for index, probability in enumerate(probabilities):
        class_name = runtime.index_to_class.get(index, str(index))
        probabilities_by_class[class_name] = float(probability)

    return PredictionResponse(
        predicted_class=predicted_class,
        predicted_index=predicted_index,
        probabilities=probabilities_by_class,
    )


def create_app(
    *,
    runtime_getter: Callable[[], RuntimeArtifacts] = get_runtime,
) -> FastAPI:
    """Create the FastAPI application."""

    app = FastAPI(
        title="Galaxy Morphology Classification API",
        description="Inference service for galaxy morphology classification models.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup() -> None:
        runtime_getter()

    @app.get("/")
    def read_root() -> dict[str, str]:
        runtime = runtime_getter()
        return {
            "status": "ok",
            "message": "Galaxy morphology inference API is running.",
            "model_name": runtime.model_name,
            "model_path": str(runtime.model_path),
            "metrics_path": str(runtime.metrics_path),
        }

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)) -> PredictionResponse:
        if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
            raise HTTPException(
                status_code=415,
                detail="Unsupported file type. Only JPEG and PNG images are allowed.",
            )

        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        return _predict_from_bytes(image_bytes, runtime=runtime_getter())

    return app


app = create_app()

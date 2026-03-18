"""Serve a trained galaxy morphology classification model through FastAPI.

This script loads a trained Keras model and exposes a simple REST API for
image-based inference.

Recommended usage from the repository root:

    uvicorn scripts.evaluation.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from galaxy_morphology_classification.training import IMG_SIZE

LOGGER = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

EFFICIENTNET_MODEL = MODELS_DIR / "efficientnet_b0.keras"
BASELINE_MODEL = MODELS_DIR / "baseline_cnn.keras"

if EFFICIENTNET_MODEL.exists():
    MODEL_NAME = "efficientnet_b0"
elif BASELINE_MODEL.exists():
    MODEL_NAME = "baseline_cnn"
else:
    raise RuntimeError(
        f"No trained model was found in {MODELS_DIR}. "
        "Run scripts/training/main.py before starting the API."
    )

MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.keras"
METRICS_PATH = METRICS_DIR / f"{MODEL_NAME}_metrics.json"

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

if not METRICS_PATH.exists():
    raise RuntimeError(f"Metrics file not found: {METRICS_PATH}")

LOGGER.info("Loading model from %s", MODEL_PATH)
MODEL = tf.keras.models.load_model(MODEL_PATH)

with METRICS_PATH.open("r", encoding="utf-8") as file:
    metrics = json.load(file)

raw_index_to_class = metrics.get("index_to_class")
if raw_index_to_class is None:
    raise RuntimeError(
        "The metrics file does not contain the 'index_to_class' field."
    )

INDEX_TO_CLASS: Dict[int, str] = {
    int(index): class_name for index, class_name in raw_index_to_class.items()
}


class PredictionResponse(BaseModel):
    """Schema returned by the prediction endpoint."""

    predicted_class: str
    predicted_index: int
    probabilities: Dict[str, float]


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


def _preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert an input image into a model-ready NumPy array.

    Parameters
    ----------
    image : Image.Image
        Input image uploaded by the client.

    Returns
    -------
    np.ndarray
        Preprocessed batch of shape (1, height, width, 3).
    """
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    array = np.array(image).astype("float32") / 255.0
    array = np.expand_dims(array, axis=0)
    return array


def _predict_from_bytes(image_bytes: bytes) -> PredictionResponse:
    """Run inference from raw image bytes.

    Parameters
    ----------
    image_bytes : bytes
        Raw uploaded image content.

    Returns
    -------
    PredictionResponse
        Structured prediction output.
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be interpreted as a valid image.",
        ) from exc

    input_array = _preprocess_image(pil_image)
    predictions = MODEL.predict(input_array, verbose=0)
    probabilities = predictions[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = INDEX_TO_CLASS.get(predicted_index, str(predicted_index))

    probabilities_by_class: Dict[str, float] = {}
    for index, probability in enumerate(probabilities):
        class_name = INDEX_TO_CLASS.get(index, str(index))
        probabilities_by_class[class_name] = float(probability)

    return PredictionResponse(
        predicted_class=predicted_class,
        predicted_index=predicted_index,
        probabilities=probabilities_by_class,
    )


@app.get("/")
def read_root() -> Dict[str, str]:
    """Health-check endpoint for the API service."""
    return {
        "status": "ok",
        "message": "Galaxy morphology inference API is running.",
        "model_name": MODEL_NAME,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Predict the morphological class of an uploaded image.

    Parameters
    ----------
    file : UploadFile
        Uploaded image file in JPEG or PNG format.

    Returns
    -------
    PredictionResponse
        Predicted class index, class name, and per-class probabilities.
    """
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Only JPEG and PNG images are allowed.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    return _predict_from_bytes(image_bytes)
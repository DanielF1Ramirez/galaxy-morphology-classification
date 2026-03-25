# Deployment Guide

## Scope

This document describes how to run the FastAPI inference service defined in `scripts/evaluation/main.py`.

## Runtime Requirements

- Python 3.11+
- Installed dependencies from `requirements.txt` or `environment.yml`
- At least one trained model file in `models/`
- Matching metrics JSON file in `reports/metrics/`

## Model and Metrics Contract

The API expects one of these model artifacts:

- `models/efficientnet_b0.keras` (preferred if present)
- `models/baseline_cnn.keras` (fallback)

And the corresponding metrics file:

- `reports/metrics/efficientnet_b0_metrics.json`
- `reports/metrics/baseline_cnn_metrics.json`

The metrics file must include `index_to_class` for decoding predictions.

## Run Locally

From the repository root:

```bash
uvicorn scripts.evaluation.main:app --host 127.0.0.1 --port 8000 --reload
```

Access:

- API root: `http://127.0.0.1:8000`
- Interactive docs: `http://127.0.0.1:8000/docs`

## Endpoints

### `GET /`

Health and model metadata endpoint.

### `POST /predict`

Accepts a multipart file upload (`image/jpeg`, `image/jpg`, or `image/png`) and returns:

- `predicted_class`
- `predicted_index`
- `probabilities` (class-to-probability mapping)

Example using curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/galaxy.jpg"
```

## Operational Notes

- The current service loads the model at import/startup time.
- If no valid model/metrics pair exists, API startup fails by design.
- For production usage, place the API behind HTTPS and add authentication, request size limits, and rate limiting.

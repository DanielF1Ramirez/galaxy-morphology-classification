# Deployment Guide

## Scope

This document describes how to run the FastAPI inference service defined in `scripts/evaluation/main.py`.

## Runtime Requirements

- Python 3.11+
- Installed dependencies from `requirements.txt` or `environment.yml`
- A trained `.keras` model inside `models/`
- Matching metrics JSON in `reports/metrics/`

## Model and Metrics Contract

The API expects:

- `models/<model_name>.keras`
- `reports/metrics/<model_name>_metrics.json`

The metrics file must include:

- `index_to_class`
- `input_scaling`
- optionally `is_selected_model`

If multiple models exist, the API chooses the one flagged as `is_selected_model`. If no model is flagged, it falls back to EfficientNetB0 and then to the baseline CNN when corresponding artifacts exist.

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

Returns:

- runtime status
- active model name
- active model path
- active metrics path

### `POST /predict`

Accepts a multipart file upload (`image/jpeg`, `image/jpg`, or `image/png`) and returns:

- `predicted_class`
- `predicted_index`
- `probabilities`

Example using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/galaxy.jpg"
```

## Operational Notes

- The API now loads the selected runtime lazily and caches it after startup.
- Image preprocessing respects the `input_scaling` stored in the selected metrics payload.
- If no valid model/metrics pair exists, startup fails with an actionable error.
- For production usage, place the API behind HTTPS and add authentication, request-size limits, and rate limiting.

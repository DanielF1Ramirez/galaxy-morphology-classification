# Galaxy Morphology Classification

A professional machine learning repository for galaxy morphology classification using Galaxy Zoo 2 images and Hart (2016) labels.

## Overview

This project consolidates prior academic work into a reproducible, software-oriented workflow:

- data acquisition
- tabular preprocessing
- exploratory data analysis
- baseline CNN training
- transfer learning with EfficientNetB0
- reusable evaluation utilities
- inference through a FastAPI service

## Repository Structure

```text
galaxy-morphology-classification/
|- .github/
|  |- workflows/
|- data/
|  |- raw/              # local-only, ignored
|  |- interim/          # local-only, ignored
|  |- README.md
|- docs/
|  |- acceptance/
|  |- business_understanding/
|  |- data/
|  |- deployment/
|  |- modeling/
|- notebooks/
|  |- README.md
|- reports/
|  |- metrics/
|  |- README.md
|- scripts/
|  |- data_acquisition/
|  |- preprocessing/
|  |- eda/
|  |- training/
|  |- evaluation/
|- src/
|  |- galaxy_morphology_classification/
|- tests/
|- .gitignore
|- AGENTS.md
|- environment.yml
|- LICENSE
|- pyproject.toml
|- README.md
|- requirements.txt
```

## Environment Setup

### Option A: pip + virtual environment

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate galaxy-morphology-classification
```

## Quickstart Pipeline

Run from the repository root.

### 1) Data acquisition

```bash
python scripts/data_acquisition/main.py
```

### 2) Preprocessing

```bash
python scripts/preprocessing/main.py
```

### 3) EDA

```bash
python scripts/eda/main.py
```

### 4) Training (fast smoke run)

```bash
python scripts/training/main.py --model baseline --epochs 1
```

### 5) API inference service

```bash
uvicorn scripts.evaluation.main:app --host 127.0.0.1 --port 8000 --reload
```

Open:
- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`

## Testing

```bash
python -m pytest -q
```

CI is configured in `.github/workflows/tests.yml`.

## Artifacts

- Trained models: `models/*.keras` (local, not committed by default).
- Training metrics: `reports/metrics/*.json`.
- EDA figure: `docs/data/figures/class_distribution.png`.

## License

MIT (see `LICENSE`).

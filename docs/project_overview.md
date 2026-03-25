# Project Overview

## Objective

Build a reproducible machine learning workflow for classifying galaxy morphology from Galaxy Zoo 2 images.

## End-to-End Workflow

1. Data acquisition (`scripts/data_acquisition/main.py`)
2. Tabular preprocessing (`scripts/preprocessing/main.py`)
3. Exploratory data analysis (`scripts/eda/main.py`)
4. Training (`scripts/training/main.py`)
5. Inference service (`scripts/evaluation/main.py`)

## Core Source Modules

- `src/galaxy_morphology_classification/models/`
  - baseline CNN builder
  - EfficientNetB0 transfer model builder
- `src/galaxy_morphology_classification/training/`
  - dataset loading, split, and `tf.data` pipeline
- `src/galaxy_morphology_classification/evaluation/`
  - reusable evaluation helpers

## Outputs

- Preprocessed data: `data/interim/merged_filtered.csv`
- Metrics: `reports/metrics/*.json`
- Figures: `docs/data/figures/*.png`
- Models (local artifacts): `models/*.keras`

## Engineering Quality

- Automated tests in `tests/`
- CI workflow in `.github/workflows/tests.yml`
- Reproducible environments via `requirements.txt`, `environment.yml`, and `pyproject.toml`

## Definition of Done

The repository is considered professionally ready when:

- scripts are runnable end-to-end with documented commands
- source code is modular and reusable under `src/`
- fast automated tests pass locally and in CI
- documentation reflects actual behavior and structure
- no unnecessary legacy clutter or heavy tracked artifacts

# Project Overview

## Objective

Build a reproducible machine learning workflow for classifying galaxy morphology from Galaxy Zoo 2 images while preserving a professional repository structure suitable for GitHub review.

## End-to-End Workflow

1. Data acquisition (`scripts/data_acquisition/main.py`)
2. Deterministic preprocessing (`scripts/preprocessing/main.py`)
3. Reproducible split generation (`train/validation/test`)
4. Exploratory data analysis (`scripts/eda/main.py`)
5. Baseline CNN training (`scripts/training/main.py --model baseline`)
6. EfficientNetB0 transfer learning with staged fine tuning (`scripts/training/main.py --model efficientnet`)
7. Offline evaluation for validation/test (`scripts/evaluation/offline.py`)
8. Inference service (`scripts/evaluation/main.py`)

## Core Source Modules

- `src/galaxy_morphology_classification/models/`
  - baseline CNN builder
  - EfficientNetB0 builder
  - EfficientNet fine-tuning helper
- `src/galaxy_morphology_classification/training/`
  - cleaned dataset loading
  - split persistence and metadata
  - TensorFlow dataset generation
  - class-weight computation
- `src/galaxy_morphology_classification/evaluation/`
  - comparable classification metrics
  - confusion matrix generation
  - model comparison summary

## Deliverables and Artifacts

- Preprocessed dataset: `data/processed/merged_filtered_clean.csv` (local)
- Reproducible split CSVs: `data/processed/splits/*.csv` (local)
- Training metrics: `reports/metrics/*.json`
- Confusion matrices: `reports/figures/*.png`
- Models: `models/*.keras` (local by default)

## Engineering Quality

- Automated tests in `tests/`
- CI workflow in `.github/workflows/tests.yml`
- Lightweight linting with `ruff`
- Reproducible environments via `requirements.txt`, `environment.yml`, and `pyproject.toml`

## Definition of Done

The repository is considered professionally closed when:

- scripts are runnable end-to-end with documented commands
- train/validation/test splits are reproducible
- the baseline and transfer-learning paths are comparable under the same protocol
- final evaluation is performed on a blind test split
- documentation reflects actual repository behavior
- no unnecessary local artifacts or hidden blockers remain in the publishable tree

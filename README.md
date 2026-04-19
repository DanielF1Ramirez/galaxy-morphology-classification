# Galaxy Morphology Classification

A professional machine learning repository for galaxy morphology classification using Galaxy Zoo 2 images and Hart (2016) labels.

## Overview

This repository consolidates prior academic work into a reproducible, software-oriented workflow that now includes:

- data acquisition from Kaggle and Hart labels
- deterministic preprocessing and quality checks
- reproducible `train/validation/test` split generation
- baseline CNN training
- EfficientNetB0 transfer learning with staged fine tuning
- offline evaluation with comparable metrics and confusion matrices
- inference through a FastAPI service

## Repository Structure

```text
galaxy-morphology-classification/
|- .github/
|  |- workflows/
|- data/
|  |- raw/              # local-only, ignored
|  |- interim/          # local-only, ignored
|  |- processed/        # local-only, ignored
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
|  |- figures/
|  |- metrics/
|  |- README.md
|- scripts/
|  |- data_acquisition/
|  |- preprocessing/
|  |- training/
|  |- evaluation/
|  |- eda/
|- src/
|  |- galaxy_morphology_classification/
|- tests/
|- .gitignore
|- environment.yml
|- pyproject.toml
|- requirements.txt
```

## Environment Setup

### Option A: pip + virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Option B: editable install with dev extras

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

### Option C: conda

```bash
conda env create -f environment.yml
conda activate galaxy-morphology-classification
```

## External Prerequisites

The data-acquisition stage requires authenticated Kaggle access because the image dataset is downloaded through `kagglehub`.

Before running acquisition, make sure:

- you have a Kaggle account
- Kaggle API credentials are configured on the local machine
- the local session can access the Kaggle dataset endpoint and the Hart label URL

If Kaggle authentication is missing, `scripts/data_acquisition/main.py` will fail before the rest of the pipeline can run.

## End-to-End Pipeline

Run all commands from the repository root.

### 1) Data acquisition

```bash
python scripts/data_acquisition/main.py
```

Expected output:

- `data/interim/merged_filtered.csv`
- local raw Hart labels in `data/raw/`

### 2) Preprocessing and reproducible split generation

```bash
python scripts/preprocessing/main.py --drop-missing-files
```

Expected output:

- `data/processed/merged_filtered_clean.csv`
- `data/processed/splits/train.csv`
- `data/processed/splits/validation.csv`
- `data/processed/splits/test.csv`
- `data/processed/splits/metadata.json`

### 3) Exploratory data analysis

```bash
python scripts/eda/main.py
```

Expected output:

- `docs/data/figures/class_distribution.png`

### 4) Baseline training

```bash
python scripts/training/main.py --model baseline --epochs 1
```

### 5) EfficientNetB0 training with staged fine tuning

```bash
python scripts/training/main.py --model efficientnet --epochs 3 --fine-tune-epochs 2 --fine-tune-layers 20
```

The transfer-learning path uses:

- initial frozen-backbone training
- optional partial fine tuning of the last backbone layers
- class weights
- lightweight data augmentation
- validation-based model selection via `macro_f1`

### 6) Offline evaluation

Validation comparison:

```bash
python scripts/evaluation/offline.py --split validation
```

Blind test evaluation of the selected model:

```bash
python scripts/evaluation/offline.py --split test
```

Expected output:

- `reports/metrics/*_validation_evaluation.json`
- `reports/metrics/*_test_evaluation.json`
- `reports/figures/*_confusion_matrix.png`
- `reports/metrics/model_comparison.json`

If you want a baseline-versus-transfer comparison summary, rerun both models with the new training pipeline first so both metrics payloads share the same schema.

### 7) FastAPI inference service

```bash
uvicorn scripts.evaluation.main:app --host 127.0.0.1 --port 8000 --reload
```

The API selects the model flagged as `is_selected_model` in the training metrics when available.

Open:

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`

## Evaluation Protocol

This repository uses a reproducible `train/validation/test` protocol:

- `train`: optimization and augmentation
- `validation`: architecture and hyperparameter comparison
- `test`: blind final evaluation only after model selection

Primary selection metric:

- `macro_f1`

Reported metrics:

- accuracy
- macro precision / recall / F1
- weighted precision / recall / F1
- confusion matrix
- classification report per class

## Testing and Quality Checks

```bash
ruff check .
pytest -q
```

The automated suite is intentionally lightweight:

- preprocessing utilities
- split-generation protocol
- evaluation helpers
- FastAPI wiring with mocked runtime

Long-running training is validated through smoke runs rather than full experiments.

## Current Deliverables

Currently versioned lightweight artifacts include:

- `docs/data/figures/class_distribution.png`
- `reports/metrics/baseline_cnn_metrics.json`

Generated local artifacts that should remain out of version control:

- `data/raw/`
- `data/interim/`
- `data/processed/`
- `models/`

## Limitations

- The repository does not version heavy datasets or trained model binaries by default.
- Final blind-test metrics depend on local training and evaluation runs.
- EfficientNetB0 is the main transfer-learning path; broader architecture search is intentionally out of scope.

## License

MIT (see `LICENSE`).

# Reports Directory

This directory stores lightweight reporting artifacts generated during training and evaluation.

## Expected Contents

- `figures/`: confusion matrices and other lightweight visual outputs
- `metrics/`: JSON summaries for training, validation, and blind-test evaluation
- concise written summaries when needed for the delivery package

## Current Reporting Contract

Training produces:

- `reports/metrics/<model_name>_metrics.json`

Offline evaluation produces:

- `reports/metrics/<model_name>_validation_evaluation.json`
- `reports/metrics/<model_name>_test_evaluation.json`
- `reports/metrics/model_comparison.json`
- `reports/figures/<model_name>_<split>_confusion_matrix.png`

## Versioning Policy

Commit concise, reproducible artifacts only. Avoid heavy intermediate outputs, duplicate experiment dumps, or notebook scratch files.

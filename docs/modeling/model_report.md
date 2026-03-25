# Model Report

## Summary

This repository supports two training paths:

- Baseline CNN (`--model baseline`)
- EfficientNetB0 transfer learning (`--model efficientnet`)

The baseline path has committed metrics. EfficientNet training code is available but no committed EfficientNet metrics artifact is currently present.

## Experimental Setup

- Data source: `data/interim/merged_filtered.csv`
- Classes: top 5 morphology classes from acquisition stage
- Image size: `128 x 128`
- Data split: stratified train/validation (`0.8 / 0.2`)
- Data pipeline: `src/galaxy_morphology_classification/training/__init__.py`

## Results Snapshot

| Model | Status | Val Loss | Val Accuracy | Metrics File |
|---|---|---:|---:|---|
| Baseline CNN | Available | 0.5599 | 0.7993 | `reports/metrics/baseline_cnn_metrics.json` |
| EfficientNetB0 | Training code available | N/A | N/A | Not committed yet |

## Recommendation

Use the baseline result as the reproducible reference and run a short EfficientNet smoke training (`--epochs 1`) before full experimentation.

## Next Evaluation Artifacts to Add

- EfficientNet metrics JSON
- Confusion matrix image
- Classification report (per-class precision/recall/F1)

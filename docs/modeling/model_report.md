# Model Report

## Summary

The repository now supports a closed comparison protocol between:

- Baseline CNN (`--model baseline`)
- EfficientNetB0 transfer learning (`--model efficientnet`)

Both training paths share the same deterministic preprocessing, the same split protocol, and the same offline evaluation utilities.

## Final Training Protocol

- Data source: cleaned CSV produced by `scripts/preprocessing/main.py`
- Split strategy: stratified `train/validation/test`
- Validation usage: model comparison and tuning
- Test usage: blind final evaluation only after model selection
- Primary selection metric: `macro_f1`

## Baseline Path

The baseline CNN remains the reference architecture.

Current committed reference:

- Validation loss: `0.5599`
- Validation accuracy: `0.7993`
- Metrics file: `reports/metrics/baseline_cnn_metrics.json`

## Transfer-Learning Path

EfficientNetB0 is the main improvement path because it offers a strong time/performance tradeoff without expanding repository scope.

Implemented improvements:

- frozen-backbone warm-up stage
- partial fine tuning of the last backbone layers
- class weights for imbalance handling
- lightweight augmentation for the training split only
- lower fine-tuning learning rate
- validation-driven model selection

## Offline Evaluation Outputs

The offline evaluation command produces:

- accuracy
- macro precision / recall / F1
- weighted precision / recall / F1
- confusion matrix
- per-class classification report
- comparison summary between models

Expected artifact locations:

- `reports/metrics/*_validation_evaluation.json`
- `reports/metrics/*_test_evaluation.json`
- `reports/figures/*_confusion_matrix.png`
- `reports/metrics/model_comparison.json`

## Interpretation Policy

- Use validation metrics to compare candidate models.
- Select the final delivery model by `macro_f1`.
- Evaluate on the blind test split only after the final candidate is chosen.

## Current Limitations

- The repository does not version heavy model binaries by default.
- Final blind-test metrics depend on local execution of the new offline evaluation flow.
- The current committed baseline result should be treated as a reproducible reference, not as the final delivery score.

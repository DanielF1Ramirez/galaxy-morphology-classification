# Baseline CNN Report

## Objective

Establish a reproducible baseline for 5-class galaxy morphology classification using images from Galaxy Zoo 2.

## Architecture

Implemented in `src/galaxy_morphology_classification/models/__init__.py` via `build_baseline_cnn`:

- Input: `128 x 128 x 3`
- Conv block 1: `Conv2D(32)` -> `MaxPooling2D` -> `BatchNormalization`
- Conv block 2: `Conv2D(64)` -> `MaxPooling2D` -> `BatchNormalization`
- Conv block 3: `Conv2D(128)` -> `MaxPooling2D` -> `BatchNormalization`
- Classifier head: `Flatten` -> `Dense(128, relu)` -> `Dropout(0.5)` -> `Dense(num_classes, softmax)`

## Training Setup

Training entrypoint: `scripts/training/main.py`

Typical command:

```bash
python scripts/training/main.py --model baseline --epochs 5
```

Default configuration:

- Optimizer: Adam
- Learning rate: `1e-3`
- Loss: `sparse_categorical_crossentropy`
- Batch size: `32`
- Validation split: `0.15`
- Test split: `0.15`
- Selection metric: `macro_f1`

## Current Baseline Result

From `reports/metrics/baseline_cnn_metrics.json`:

- Validation loss: `0.5599`
- Validation accuracy: `0.7993`
- Epochs: `5`

## Interpretation

The baseline model provides a stable reference point for subsequent transfer learning experiments.

## Limitations

- The baseline is the reference model, not necessarily the final delivery model.
- Offline evaluation artifacts depend on running `scripts/evaluation/offline.py`.
- EfficientNet comparison requires a local transfer-learning run under the same split protocol.

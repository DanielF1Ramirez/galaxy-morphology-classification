# Data Directory

This directory stores lightweight data artifacts and dataset access instructions for the Galaxy Morphology Classification project.

## Directory purpose
The repository should only include small, shareable, and reproducible data-related assets.

## Suggested structure
- `sample/`: Small example files that are safe to version.
- `raw/`: Original downloaded files. This directory should remain local and should not be committed.
- `interim/`: Intermediate files generated during preprocessing. This directory should remain local and should not be committed.
- `processed/`: Final processed artifacts when they are lightweight and useful for reproducibility.

## Versioning policy
Do not commit:
- large image datasets
- raw Kaggle downloads
- generated heavy CSV files
- temporary preprocessing outputs
- model checkpoints stored together with data

## Reproducibility note
All large data assets should be recreated through documented acquisition and preprocessing scripts rather than being stored directly in the repository.
# Changelog

## Unreleased

### Added

- deterministic `train/validation/test` split generation with persisted metadata
- offline evaluation script for validation and blind-test runs
- model-comparison summary support driven by `macro_f1`
- API runtime selection based on versioned metrics metadata
- tests for split generation, evaluation utilities, and FastAPI wiring
- lightweight linting with `ruff`

### Changed

- preprocessing now writes a cleaned dataset to `data/processed/`
- training now supports staged EfficientNetB0 fine tuning
- documentation now reflects the final split/evaluation protocol
- `.gitignore` now ignores only top-level model artifacts instead of source-code packages under `src/`

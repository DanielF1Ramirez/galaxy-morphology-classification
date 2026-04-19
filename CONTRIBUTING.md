# Contributing Guide

Thanks for contributing to `galaxy-morphology-classification`.

## Scope

- Keep work focused on this repository.
- Preserve the current project structure unless there is a strong technical reason.
- Prefer small, reviewable pull requests.

## Development Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Alternative:

```bash
pip install -r requirements.txt
```

## Required Checks

Run before submitting changes:

```bash
ruff check .
pytest -q
```

When touching training or evaluation flows, also run the smallest realistic smoke commands you can justify.

## Data and Model Policy

- Do not commit heavy raw datasets, split CSVs, or local model binaries by default.
- Keep generated artifacts in `data/raw/`, `data/interim/`, `data/processed/`, and `models/` local unless the task explicitly requires something lighter to be versioned.
- Use the deterministic split protocol already implemented by `scripts/preprocessing/main.py`.

## Coding Guidelines

- Keep comments and docstrings in English.
- Avoid speculative refactors.
- Do not edit notebooks unless required for the task.
- Update documentation when behavior changes.
- Prefer reproducible, testable utilities under `src/`.

## Commit and PR Guidance

- Use concise English commit messages.
- Keep commits logically scoped.
- Include validation details in PR descriptions:
  - what changed
  - what was validated
  - what remains unverified

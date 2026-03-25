# Contributing Guide

Thanks for contributing to `galaxy-morphology-classification`.

## Scope

- Keep work focused on this repository.
- Preserve the current project structure unless there is a strong technical reason.
- Prefer small, reviewable pull requests.

## Development Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Alternative (editable package install with declared dev extras):

```bash
pip install -e ".[dev]"
```

## Quality Checks

Run before submitting changes:

```bash
python -m pytest -q
```

For workflow sanity checks, prefer fast commands and smoke runs.

## Coding Guidelines

- Keep comments and docstrings in English.
- Avoid large, speculative refactors.
- Do not edit notebooks unless required for the task.
- Update documentation when behavior changes.
- Avoid unnecessary dependencies.

## Commit and PR Guidance

- Use concise English commit messages.
- Keep commits logically scoped.
- Include validation details in PR descriptions:
  - what was changed
  - what was validated
  - what remains unverified

## Data and Artifacts

Do not commit heavy raw datasets or large model artifacts. Keep tracked outputs lightweight and reproducible.

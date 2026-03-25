# AGENTS.md

## Project identity
- Repository name: `galaxy-morphology-classification`
- Repository purpose: professionalize and consolidate the original `DeepLearning` and `MetodologiasDesarrollo` work into a clean, reproducible, portfolio-ready machine learning repository.
- Main domain: galaxy morphology classification from astronomical image data.

## Working directory rules
- Treat the repository root as the only editable workspace.
- Current repository root:
  - `C:\Users\danie_zxu\Downloads\galaxy-morphology-classification`
- External reference material exists at:
  - `C:\Users\danie_zxu\OneDrive\Documentos\ProyectosGitHub`
- Use the following folders only as reference unless explicitly asked to copy or migrate files:
  - `C:\Users\danie_zxu\OneDrive\Documentos\ProyectosGitHub\DeepLearning`
  - `C:\Users\danie_zxu\OneDrive\Documentos\ProyectosGitHub\MetodologiasDesarrollo\ProyectoUNAL-main`
- Do not reorganize, rename, or modify unrelated external projects.

## Primary objective
Finish this first repository professionally before starting any other repository.

## Current repository scope
The repository should include and maintain:
- data acquisition pipeline
- exploratory data analysis workflow
- baseline CNN training
- EfficientNetB0 transfer learning
- evaluation utilities
- FastAPI inference service
- reproducible environment files
- professional documentation
- tests and CI scaffolding

## Project structure expectations
Preserve this structure unless there is a strong technical reason to change it:

- `docs/`: formal project documentation
- `data/`: lightweight samples and data instructions only
- `notebooks/`: analytical and experimental notebooks
- `scripts/`: runnable entry-point scripts
- `src/galaxy_morphology_classification/`: reusable source code
- `reports/`: figures, metrics, and concise reporting artifacts
- `tests/`: automated tests
- `.github/`: CI and repo templates

## Editing rules
1. Prefer small, safe, reviewable changes.
2. Explain the plan before making multi-file or high-impact changes.
3. Keep comments and docstrings in English.
4. Keep commit messages in English.
5. Prefer professional naming and modular code.
6. Avoid unnecessary dependencies.
7. Do not edit notebooks unless explicitly requested.
8. When changing behavior, update documentation too.
9. Keep external datasets, generated heavy artifacts, and large model files out of version control.
10. Do not replace working code with speculative refactors.

## Validation rules
- Prefer fast local validation.
- Avoid long-running training unless explicitly approved.
- For training validation, prefer smoke tests such as:
  - one epoch
  - reduced dataset checks if available
  - import validation
  - dataset construction validation
- Be explicit about what was validated and what remains unverified.
- If a command is long-running, warn before executing it.

## Source-of-truth guidance
When there is tension between old academic code and the new repository structure:
- preserve the project intent
- normalize code quality professionally
- move reusable logic into `src/`
- keep execution entry points in `scripts/`
- keep notebooks for traceability, not as the only execution path

## Documentation expectations
The repository should remain suitable for:
- GitHub portfolio presentation
- technical review
- reproducibility
- future extension

Whenever you make meaningful code changes, also check whether any of the following should be updated:
- `README.md`
- `docs/project_overview.md`
- `CONTRIBUTING.md`
- usage instructions
- dependency files

## What to do first when starting a task
1. Inspect the current repository state.
2. Summarize what is already well structured.
3. Identify what is missing or weak.
4. Propose the smallest high-value next step.
5. Only then implement changes.

## Definition of done for this repository
The repository can be considered professionally finished when it has:
- clear documentation
- coherent structure
- reusable source modules
- stable training and inference entry points
- dependency files aligned with actual code
- lightweight validation or tests
- no unnecessary legacy clutter
- clean Git history with clear commit messages

# Final Delivery Checklist

Use this checklist before the final GitHub publication or handoff.

## Repository State

- `git status` is clean
- no local virtual environments are tracked
- no raw or processed datasets are tracked accidentally
- no local model binaries are tracked accidentally unless explicitly intended

## Environment and Quality

- dependencies install correctly from the documented method
- `ruff check .` passes
- `pytest -q` passes
- CI is green on the final branch

## Data and Training

- preprocessing completes successfully
- split metadata is generated and consistent
- baseline training smoke run completes
- EfficientNetB0 smoke run completes
- validation comparison artifacts are generated

## Final Evaluation

- blind test evaluation is executed only after selecting the final candidate
- confusion matrix figure is generated
- per-class classification report is generated
- selected model and metrics are consistent with the API runtime

## Documentation and Delivery

- README matches the actual commands and outputs
- deployment guide matches the selected-model loading behavior
- model report reflects the current evaluation protocol
- changelog summarizes the final intervention

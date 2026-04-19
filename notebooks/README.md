# Notebooks Directory

This directory contains research, analysis, and experimentation notebooks related to the galaxy morphology classification project.

## Purpose
Notebooks are used to document exploratory work, business understanding, data analysis, and model experimentation.

## Conventions
- Numbered notebooks should represent the main analytical flow.
- Experimental or transitional work should be stored inside `experiments/`.
- Reusable logic should be migrated to `src/` and `scripts/`.
- Final repository execution should not depend exclusively on notebooks.

## Expected contents
Only keep notebooks that add traceability or explain experiments that are not already reproducible through `scripts/` and `src/`.

The current delivery does not depend on notebooks to run the end-to-end pipeline.

## Good practice
A notebook should explain analysis and experimentation, while maintainable and reusable logic should live in source modules and scripts.

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
This repository will later integrate notebooks from the Deep Learning project to preserve:
- business understanding
- exploratory data analysis
- integrated training pipeline exploration
- transition from baseline CNN to transfer learning

## Good practice
A notebook should explain analysis and experimentation, while maintainable and reusable logic should live in source modules and scripts.
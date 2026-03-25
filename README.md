# Galaxy Morphology Classification

A deep learning project for galaxy morphology classification using Galaxy Zoo 2 image data. This repository consolidates work from the original Deep Learning and Software Development phases into a clean, reproducible, and portfolio-ready machine learning repository.

## Overview

The goal of this project is to classify galaxy morphology from astronomical images by building a structured workflow that covers:

- data acquisition
- data preparation
- exploratory data analysis
- baseline convolutional neural networks
- transfer learning with EfficientNetB0
- model evaluation
- inference through a FastAPI service

This repository was reorganized to transform previous academic project phases into a professional software-oriented machine learning project.

## Problem Statement

Galaxy morphology classification is a relevant task in astronomy because galaxy shape is strongly related to physical structure and evolutionary history. Manual labeling is expensive and time-consuming, so image classification models can support scalable morphological categorization.

This project uses Galaxy Zoo 2 related resources to build a supervised classification pipeline for the most frequent morphology classes.

## Main Objectives

- Build a reproducible machine learning workflow for galaxy morphology classification.
- Consolidate exploratory and modeling work into a professional repository structure.
- Compare a baseline CNN against a transfer learning architecture.
- Preserve the analytical value of the original notebooks while moving reusable logic into source modules.
- Expose trained models through a lightweight inference API.

## Repository Structure

```text
galaxy-morphology-classification/
├─ .github/
│  └─ workflows/
├─ data/
│  ├─ sample/
│  └─ README.md
├─ docs/
│  ├─ acceptance/
│  ├─ business_understanding/
│  ├─ data/
│  ├─ deployment/
│  └─ modeling/
├─ notebooks/
│  ├─ experiments/
│  └─ README.md
├─ reports/
│  ├─ figures/
│  ├─ metrics/
│  └─ README.md
├─ scripts/
│  ├─ data_acquisition/
│  ├─ eda/
│  ├─ evaluation/
│  ├─ preprocessing/
│  └─ training/
├─ src/
│  └─ galaxy_morphology_classification/
├─ tests/
├─ .gitignore
├─ LICENSE
├─ environment.yml
├─ pyproject.toml
├─ README.md
└─ requirements.txt
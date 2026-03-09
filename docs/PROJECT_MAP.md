# Project Map

This document explains what each retained folder/file does and how data flows through the project.

## Core Folders

- `notebooks/`
  - `Aerial_Object_Classification.ipynb`
  - Main end-to-end experimentation notebook (inspection, model training, evaluation, deployment helper cells).

- `scripts/`
  - `train_classification.py`:
    - Trains either `custom` CNN or `transfer` (MobileNetV2) model.
    - Input: `classification_dataset/`
    - Output: `.h5` model file in `artifacts/models/`
  - `evaluate_model.py`:
    - Evaluates a trained model on test split.
    - Input: model path + `classification_dataset/test/`
    - Output: `classification_report.txt` + `confusion_matrix.png` in `reports/evaluation/`
  - `dummy_classification_baseline.py`:
    - Fast baseline (RandomForest over color histograms).
    - Output: baseline artifacts in `reports/results_dummy/` and a filled comparison markdown.

- `artifacts/models/`
  - Stores trained model artifacts (`.h5`).

- `reports/`
  - Contains generated performance reports and report templates.

- `classification_dataset/`
  - Classification dataset organized into `train/`, `valid/`, `test/` class folders.

- `object_detection_dataset/`
  - Detection dataset for YOLO pipeline (`train/valid/test` with `images/labels`).
  - Includes canonical YOLO config at `object_detection_dataset/data.yaml`.

- `docs/dataset_sources/`
  - Original dataset-source text files preserved for provenance.

- `archive/`
  - Historical notebook/script variants kept for reference; not part of primary workflow.

## Root Files

- `app.py`
  - Streamlit app for Bird/Drone prediction.
  - Reads model from `artifacts/models/best_custom_cnn.h5`.

- `requirements.txt`
  - Main dependency list for development and running workflows.

- `requirements-lock.txt`
  - Additional locked dependency snapshot.

## End-to-End Flow

1. Prepare environment from `requirements.txt`.
2. Train using notebook or `scripts/train_classification.py`.
3. Save trained model to `artifacts/models/`.
4. Evaluate with `scripts/evaluate_model.py` (writes to `reports/`).
5. Deploy/demo with `app.py`.

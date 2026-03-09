# Aerial Object Classification & Detection

Internship project for two tasks:
- **Image classification**: Bird vs Drone (TensorFlow/Keras)
- **Object detection**: YOLO-ready dataset and config

## Clean Project Layout

```
Aerial Object Classification & Detection/
├─ notebooks/
│  └─ Aerial_Object_Classification.ipynb
├─ scripts/
│  ├─ train_classification.py
│  ├─ evaluate_model.py
│  └─ dummy_classification_baseline.py
├─ artifacts/
│  └─ models/
├─ reports/
│  ├─ model_comparison_report.md
│  ├─ model_comparison_report_filled.md
│  └─ results_dummy/
├─ classification_dataset/
├─ object_detection_dataset/
│  └─ data.yaml
├─ docs/
│  └─ dataset_sources/
├─ archive/
│  ├─ legacy_notebooks/
│  └─ legacy_scripts/
├─ app.py
├─ requirements.txt
└─ requirements-lock.txt
```

## Quick Start (Windows PowerShell)

1. Create and activate environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Run notebook workflow:

```powershell
code notebooks/Aerial_Object_Classification.ipynb
```

3. Train a classification model:

```powershell
python scripts/train_classification.py --model custom --epochs 10
```

4. Evaluate a saved model:

```powershell
python scripts/evaluate_model.py --model_path artifacts/models/best_custom_cnn.h5
```

5. Start Streamlit app:

```powershell
streamlit run app.py
```

## How Components Connect

- `classification_dataset/` is the source for training and evaluation.
- `scripts/train_classification.py` reads `classification_dataset/` and writes `.h5` models to `artifacts/models/`.
- `scripts/evaluate_model.py` loads a model from `artifacts/models/`, evaluates on test data, and writes metrics/plots to `reports/evaluation/`.
- `app.py` loads the trained model and serves image predictions via Streamlit.
- `reports/` stores generated analysis outputs and comparison documents.
- `object_detection_dataset/` (with `object_detection_dataset/data.yaml`) supports optional YOLO training.

## Notes

- Legacy experiments were moved to `archive/` to keep the root clean.
- Source dataset readme files are in `docs/dataset_sources/`.
- The environment used by this editor may not allow pip installs; if package installs fail, run the install commands locally on your machine.
- The notebook cells include guards that skip TF-dependent code if TensorFlow is not present — this allows inspection and visualization to run even when heavy packages are missing.

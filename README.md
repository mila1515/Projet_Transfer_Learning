# AI-Powered Pneumonia Detection from Chest X-rays

End-to-end Deep Learning system for binary medical image classification with experiment tracking, explainability, and API deployment.

## 1. Business Problem
Pneumonia diagnosis from chest X-rays is time-critical and can be challenging under high clinical workload.
This project demonstrates how an AI assistant can support radiology triage by detecting likely pneumonia cases quickly, while keeping model behavior observable and explainable.

## 2. Project Goal
Build a production-style AI pipeline that:
- Trains transfer learning models (VGG16, ResNet50V2, or MobileNetV2).
- Tracks experiments with MLflow.
- Exposes inference via FastAPI.
- Generates explainability artifacts with Grad-CAM.
- Produces recruiter-ready evaluation reports.

## 3. Professional Architecture

```text
Projet_Transfer_Learning/
├── api/
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_training_experiments.ipynb
├── reports/
│   └── figures/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── gradcam.py
│   ├── inference.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
├── config.yaml
├── Dockerfile
├── requirements-api.txt
└── requirements.txt
```

## 4. Machine Learning Pipeline
1. Data loading from folder-based dataset (`train`, `val`, `test`, classes `NORMAL` and `PNEUMONIA`).
2. Reproducible 85/15 train/validation split, preprocessing, and augmentation
   (flip, rotation, zoom, translation). The tiny validation folder shipped with
   the source dataset is intentionally not used by default.
3. Transfer learning model construction (VGG16/ResNet50V2/MobileNetV2 + fine-tuning).
4. Imbalance-aware training with class weights and callbacks (EarlyStopping,
   ReduceLROnPlateau, ModelCheckpoint).
5. Evaluation with medical-priority metrics.
6. Artifact generation (metrics JSON, confusion matrix, ROC curve, error analysis).
7. Inference pipeline for CLI and API.

## 5. MLOps and Experiment Tracking (MLflow)
The training script logs:
- Parameters: learning rate, batch size, backbone, seed, threshold, epochs.
- Metrics: accuracy, precision, recall, F1, AUC, specificity.
- Artifacts: reports folder (metrics + figures), trained model.
- Run metadata: project/stage/model_version tags for clear run comparison.

Launch MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 6. Explainable AI (Grad-CAM)
Grad-CAM heatmaps are supported for inference results to highlight image regions driving predictions.
This improves interpretability and makes the project stand out in interviews focused on trustworthy AI.

## 7. Model Evaluation (Medical Focus)
Generated outputs include:
- Confusion matrix.
- ROC curve.
- Classification report.
- False positives / false negatives sample paths.

Medical interpretation note:
- In screening scenarios, recall for the `PNEUMONIA` class is critical to reduce missed pneumonia cases (false negatives).

Current MobileNetV2 fast-profile baseline on the 624-image test split at a
0.5 threshold:

- Accuracy: 80.13%
- Precision (`PNEUMONIA`): 76.08%
- Recall (`PNEUMONIA`): 99.49% (2 false negatives)
- Specificity: 47.86% (122 false positives)
- F1: 86.22%
- ROC AUC: 96.01%

This is a screening-oriented operating point: it catches nearly all pneumonia
cases but over-flags normal images. Threshold selection requires validation for
the intended use and must not be treated as clinical approval.

## 8. FastAPI Inference Service
### Endpoint: `POST /predict`
Input:
- Image file (`multipart/form-data`).

Output example:

```json
{
  "prediction": "PNEUMONIA",
  "probability": 0.94,
  "threshold": 0.5
}
```

Health check:
- `GET /health`

The health response reports `model_missing` until a trained `.keras` model is
available. Runtime settings can be overridden with `MODEL_PATH`,
`CONFIG_PATH`, `PREDICTION_THRESHOLD`, and `MAX_UPLOAD_BYTES` environment
variables. Use `CONFIG_PATH=config_fast.yaml` together with the MobileNetV2
model so the API applies its 160×160 input size.

## 9. Setup and Usage
### 9.1 Install
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 9.2 Dataset layout
Place Kaggle Chest X-ray dataset in:

```text
data/raw/chest_xray/
├── train/
├── val/      # optional with the default configuration
└── test/
```

Each split must contain:
- `NORMAL/`
- `PNEUMONIA/`

By default, validation is sampled from `train/` using the seeded split defined
in `config.yaml`. Set `data.validation_source` to `folder` to use `val/`
instead.

In-memory dataset caching is disabled by default because the resized float32
dataset requires roughly 3.3 GiB before TensorFlow and model overhead. Enable
`data.cache_in_memory` only on machines with sufficient RAM.

### 9.3 Train
```bash
python -m src.train --config config.yaml
```

Train with ResNet50V2:
```bash
python -m src.train --config config.yaml --backbone resnet50v2
```

Fast CPU-friendly experiment with MobileNetV2, 160×160 inputs, and 5 epochs:

```bash
python -m src.train --config config_fast.yaml
```

This profile writes `models/pneumonia_mobilenetv2_fast.keras` and keeps its
evaluation artifacts separate in `reports/fast/`. Compare its recall,
specificity, and AUC against the full profile before selecting a production
model.

Run the API with the trained fast model in PowerShell:

```powershell
$env:CONFIG_PATH = "config_fast.yaml"
$env:MODEL_PATH = "models/pneumonia_mobilenetv2_fast.keras"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 9.4 Evaluate
```bash
python -m src.evaluate --config config.yaml
```

### 9.5 Predict from CLI
```bash
python -m src.predict --image path/to/xray.jpg --model models/pneumonia_model.keras --gradcam
```

### 9.6 Run API
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## 10. Docker Deployment (API)
Build and run:

```bash
docker build -t pneumonia-api .
docker run --rm -p 8000:8000 pneumonia-api
```

The API image installs the smaller, pinned `requirements-api.txt` dependency
set and runs as an unprivileged user. Training and notebook dependencies remain
in `requirements.txt`. When the trained fast model is present under `models/`,
the image starts with `config_fast.yaml` and is ready to serve immediately.

## 11. Configuration
All key hyperparameters are centralized in `config.yaml`:
- Seed for reproducibility.
- Backbone selection.
- Learning rate, epochs, batch size.
- Decision threshold.
- MLflow tracking settings.

## 12. Tests

Run the lightweight unit tests without starting a full training job:

```bash
python -m unittest discover -s tests -v
```

The suite includes API contract tests, a real Keras save/load inference test,
Grad-CAM coverage, metric checks, and transfer-learning layer checks. The same
suite runs in GitHub Actions.

## 13. Current Limitations
- Binary classification only (`NORMAL` vs `PNEUMONIA`).
- No lesion segmentation.
- Trained on one dataset; external validation remains necessary.
- Not a clinical device and not intended for standalone diagnosis.

## 14. Roadmap / Future Improvements
- Add calibration and threshold optimization by medical objective.
- Add test-time augmentation and confidence intervals.
- Add static analysis and container scanning to the existing CI pipeline.
- Add model registry workflow with automated staging/production promotion.
- Extend to multi-class thoracic disease classification.

## 15. Tech Stack
- Python
- TensorFlow / Keras
- Scikit-learn
- MLflow
- FastAPI
- OpenCV
- Matplotlib / Seaborn

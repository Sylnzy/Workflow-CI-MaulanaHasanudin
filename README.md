# Workflow CI/CD for Crop Recommendation

Repository: Workflow-CI-MaulanaHasanudin (GitHub: Sylnzy)

## Overview
- MLflow Project for crop recommendation training (Random Forest).
- GitHub Actions CI to run MLflow Project, upload artifacts, and build/push Docker image (Advanced target).
- Dataset: crop_preprocessing.csv (from Kriteria 1 preprocessing output).

## Structure
```
Workflow-CI-MaulanaHasanudin/
├── .github/workflows/mlflow_training.yml   # CI pipeline (advanced)
├── MLProject/
│   ├── MLproject                           # MLflow project config
│   ├── conda.yaml                          # Conda environment spec
│   ├── requirements.txt                    # Python deps
│   ├── modelling_tuning.py                 # CLI training script
│   └── crop_preprocessing.csv              # Dataset (copy from preprocessing)
├── .gitignore
├── README.md
└── Workflow-CI.txt (fill link after repo public)
```

## Local Run
```bash
cd MLProject
mlflow run . --no-conda -P n_estimators=200 -P max_depth=20 -P test_size=0.2 -P data_path=crop_preprocessing.csv
```
Artifacts appear under `MLProject/mlruns/` and include model, confusion matrix, feature importance, classification report.

## CI/CD (Advanced)
- Trigger: push to `main` (MLProject/**), manual dispatch, weekly cron.
- Steps: checkout → setup Python 3.12 → install deps → run MLflow Project → gather artifacts → upload artifact → build Docker from latest MLflow model → push to Docker Hub (needs secrets).

### Required Secrets
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_TOKEN`: Docker Hub access token

Optional: adjust image name/tag in workflow if needed.

## Notes
- Ensure `crop_preprocessing.csv` is present under MLProject before running locally or in CI.
- For grading, keep repository public.

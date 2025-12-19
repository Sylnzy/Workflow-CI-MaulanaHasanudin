"""
Crop Recommendation Model with Hyperparameter Tuning - MLflow Project Version
Author: Maulana Hasanudin (GitHub: Sylnzy)
Description: CLI-enabled training script for MLflow Project CI/CD.
"""

import argparse
import json
import logging
import os
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Crop Recommendation with MLflow Project")
    parser.add_argument("--data_path", type=str, default="crop_preprocessing.csv", help="Path to preprocessed CSV")
    parser.add_argument("--n_estimators", type=int, default=300, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=30, help="Max depth")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size fraction")
    return parser.parse_args()


def load_preprocessed_data(filepath: str, test_size: float):
    logger.info("Loading preprocessed data from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Data loaded: %d rows, %d columns", df.shape[0], df.shape[1])

    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    logger.info("Train shape: %s | Test shape: %s | Classes: %d", X_train.shape, X_test.shape, y.nunique())
    return X_train, X_test, y_train, y_test


def hyperparameter_tuning(X_train, y_train, n_estimators: int, max_depth: int):
    """Simplified grid for CI speed."""
    logger.info("Starting hyperparameter tuning (CI-friendly grid)")
    param_grid = {
        "n_estimators": [n_estimators],
        "max_depth": [max_depth, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    logger.info("Best params: %s | Best CV F1: %.4f", grid_search.best_params_, grid_search.best_score_)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def evaluate_model(model, X_train, X_test, y_train, y_test):
    logger.info("Evaluating model")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "train_f1": f1_score(y_train, y_pred_train, average="weighted"),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_precision": precision_score(y_test, y_pred_test, average="weighted", zero_division=0),
        "test_recall": recall_score(y_test, y_pred_test, average="weighted", zero_division=0),
        "test_f1": f1_score(y_test, y_pred_test, average="weighted", zero_division=0),
    }
    return metrics, y_pred_test


def plot_confusion_matrix(y_true, y_pred, filename: str = "confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True)
    plt.title("Confusion Matrix - Crop Recommendation", fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, feature_names, filename: str = "feature_importance.png"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importances)), importances[indices], color="skyblue", edgecolor="black")
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance - Random Forest", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    logger.info(
        "CROP RECOMMENDATION CI RUN | data_path=%s | n_estimators=%d | max_depth=%s | test_size=%.2f",
        args.data_path,
        args.n_estimators,
        args.max_depth,
        args.test_size,
    )

    mlflow.set_experiment("crop_recommendation_ci")

    # If running under MLflow Project, a parent run exists; use nested run to avoid ID conflicts
    parent_run = mlflow.active_run()
    with mlflow.start_run(run_name="ci_automated_training", nested=bool(parent_run)):
        X_train, X_test, y_train, y_test = load_preprocessed_data(args.data_path, args.test_size)

        mlflow.log_param("data_path", args.data_path)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_classes", len(np.unique(y_train)))

        model, best_params, cv_score = hyperparameter_tuning(X_train, y_train, args.n_estimators, args.max_depth)
        for p, v in best_params.items():
            mlflow.log_param(f"best_{p}", v)
        mlflow.log_metric("cv_f1", cv_score)

        metrics, y_pred_test = evaluate_model(model, X_train, X_test, y_train, y_test)
        mlflow.log_metrics(metrics)

        plot_confusion_matrix(y_test, y_pred_test, "confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        plot_feature_importance(model, X_train.columns, "feature_importance.png")
        mlflow.log_artifact("feature_importance.png")

        report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
        with open("classification_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact("classification_report.json")

        mlflow.sklearn.log_model(model, "model")
        joblib.dump(model, "crop_model.pkl")
        mlflow.log_artifact("crop_model.pkl")

        logger.info("Training completed | test_acc=%.2f%% | test_f1=%.2f%%", metrics["test_accuracy"] * 100, metrics["test_f1"] * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

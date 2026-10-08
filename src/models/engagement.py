"""High Engagement Classification (Étape 6).

Target: high_engagement = 1 if vote_count > 70th percentile, else 0
Models: Logistic Regression, Random Forest, Linear SVM
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from config import MODELS_DIR, ensure_directories
from src.models.common import RANDOM_STATE, TEST_SIZE, load_data, prepare_features, split_data

logger = logging.getLogger(__name__)

THRESHOLD_PERCENTILE = 70


def build_high_engagement_target(df: pd.DataFrame, percentile: int = 70) -> pd.Series:
    """
    Create high_engagement target from vote_count.

    high_engagement = 1 if vote_count > percentile-threshold, else 0

    Important: vote_count is NOT used as a feature - it only defines the target.
    """
    logger.info(f"Creating high_engagement target (>{percentile}th percentile of vote_count)...")

    threshold = df["vote_count"].quantile(percentile / 100.0)
    y_high_engagement = (df["vote_count"] > threshold).astype(int)

    logger.info(f"Threshold vote_count: {threshold:.0f}")
    logger.info(f"high_engagement distribution: {pd.Series(y_high_engagement).value_counts().to_dict()}")
    logger.info(
        f"Movies above threshold: {(y_high_engagement == 1).sum()} / {len(y_high_engagement)} ({100*(y_high_engagement==1).mean():.1f}%)"
    )

    return pd.Series(y_high_engagement, name="high_engagement")


def train_engagement_models(X_train, y_train, X_test, y_test):
    """
    Train high engagement classifier comparing 3 models:
    - Logistic Regression
    - Random Forest
    - Linear SVM

    Evaluates with Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix.
    """
    logger.info("Training High Engagement Classification (comparing 3 models)...")
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    models = {
        "Logistic Regression": Pipeline(steps=[("classifier", LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=RANDOM_STATE, solver="liblinear", C=1.0))]),
        "Random Forest": Pipeline(steps=[("classifier", RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1))]),
        "Linear SVM": Pipeline(steps=[("classifier", SVC(
            kernel="linear", probability=True, random_state=RANDOM_STATE, class_weight="balanced"))]),
    }

    results = {}

    for name, pipeline in models.items():
        logger.info(f"Training: {name}")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred = np.array(y_pred).reshape(-1)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            if hasattr(pipeline["classifier"], "predict_proba"):
                proba = pipeline["classifier"].predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, proba)
            else:
                decision = pipeline["classifier"].decision_function(X_test)
                auc = roc_auc_score(y_test, decision)
        except Exception:
            auc = float("nan")

        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "predictions": y_pred,
            "pipeline": pipeline,
        }

        logger.info(f"  Accuracy:  {acc:.4f}")
        logger.info(f"  Precision: {prec:.4f}")
        logger.info(f"  Recall:    {rec:.4f}")
        logger.info(f"  F1-score:  {f1:.4f}")
        logger.info(f"  ROC-AUC:   {auc:.4f}")
        logger.info(f"  Confusion Matrix:\n{cm}")

        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Engagement", "High Engagement"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"High Engagement - {name}")
        plt.tight_layout()
        plt.savefig(f"reports/figures/8_high_engagement_{name.replace(' ', '_')}.png", dpi=150)
        plt.close()

    logger.info("=" * 60)
    logger.info("HIGH ENGAGEMENT MODEL COMPARISON")
    logger.info("=" * 60)
    logger.info(f"{'Model':<20} {'Acc':>6} {'Prec':>6} {'Recall':>6} {'F1':>6} {'ROC-AUC':>8}")
    logger.info("-" * 60)
    for name, res in results.items():
        logger.info(
            f"{name:<20} {res['accuracy']:>6.4f} {res['precision']:>6.4f} "
            f"{res['recall']:>6.4f} {res['f1']:>6.4f} {res['roc_auc']:>8.4f}"
        )
    logger.info("=" * 60)

    return results


def main():
    """Run high engagement classification pipeline."""
    ensure_directories()
    logger.info("Starting High Engagement Classification Pipeline...")

    tfidf, tfidf_matrix, movie_index, df = load_data()

    y_high_engagement = build_high_engagement_target(df, THRESHOLD_PERCENTILE)

    X = prepare_features(tfidf_matrix, df)

    train_idx, test_idx = split_data(X, df)

    X_train, X_test = X[train_idx], X[test_idx]
    y_high_train, y_high_test = y_high_engagement.iloc[train_idx], y_high_engagement.iloc[test_idx]

    logger.info(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

    results = train_engagement_models(X_train, y_high_train, X_test, y_high_test)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(results, MODELS_DIR / "high_engagement_results.joblib")

    logger.info("High Engagement Classification Pipeline Complete!")
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
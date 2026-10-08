"""Revenue Tier Classification.

Model: RandomForest (no budget/revenue leakage)
"""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from config import MODELS_DIR, ensure_directories
from src.models.common import RANDOM_STATE, load_data, prepare_features, split_data

logger = logging.getLogger(__name__)


def prepare_revenue_tier_target(df: pd.DataFrame) -> np.ndarray:
    """Prepare revenue tier target (already categorical)."""
    logger.info("Preparing revenue tier target...")
    y_revenue = df["revenue_tier"].cat.codes.values
    logger.info(f"Revenue tier distribution: {pd.Series(y_revenue).value_counts().sort_index().to_dict()}")
    return y_revenue


def train_revenue_classifier(X_train, y_train, X_test, y_test):
    """Train revenue tier classifier (no budget/revenue leakage)."""
    logger.info("Training Revenue Tier Classification (no leakage)...")

    clf = Pipeline(steps=[("classifier", RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1))])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    logger.info(f"Revenue Tier - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    cm = confusion_matrix(y_test, y_pred)

    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Tier {i}" for i in range(5)])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Revenue Tier Classification")
    plt.tight_layout()
    plt.savefig("reports/figures/9_revenue_tier_cm.png", dpi=150)
    plt.close()

    return clf, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}


def main():
    """Run revenue tier classification pipeline."""
    ensure_directories()
    logger.info("Starting Revenue Tier Classification Pipeline...")

    tfidf, tfidf_matrix, movie_index, df = load_data()

    X = prepare_features(tfidf_matrix, df)

    train_idx, test_idx = split_data(X, df)

    X_train, X_test = X[train_idx], X[test_idx]

    y_revenue = prepare_revenue_tier_target(df)
    y_rev_train, y_rev_test = y_revenue[train_idx], y_revenue[test_idx]

    logger.info(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

    clf, metrics = train_revenue_classifier(X_train, y_rev_train, X_test, y_rev_test)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODELS_DIR / "revenue_tier_classifier.joblib")

    logger.info("Revenue Tier Classification Pipeline Complete!")
    return clf, metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
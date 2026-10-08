"""Multi-label Genre Classification.

Model: OneVsRest + LogisticRegression
"""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from config import DATA_ENRICHED, MODELS_DIR, ensure_directories
from src.models.common import RANDOM_STATE, load_data, prepare_features, split_data

logger = logging.getLogger(__name__)


def prepare_genre_targets(df: pd.DataFrame, genres_df: pd.DataFrame) -> tuple[np.ndarray, list]:
    """Create multi-label genre targets."""
    logger.info("Preparing multi-label genre targets...")

    genre_map = genres_df.groupby("movie_id")["genre_name"].apply(list).to_dict()
    y_genres = df["id"].map(genre_map).apply(lambda x: x if isinstance(x, list) else []).tolist()

    mlb = MultiLabelBinarizer()
    y_genres_bin = mlb.fit_transform(y_genres)

    logger.info(f"Genre classes: {mlb.classes_.tolist()}")
    logger.info(f"Multi-label shape: {y_genres_bin.shape}")
    return y_genres_bin, mlb.classes_.tolist()


def train_genre_classifier(X_train, y_train, X_test, y_test, genre_names):
    """Train multi-label genre classifier."""
    logger.info("Training Genre Classification (OneVsRest + LogisticRegression)...")

    clf = Pipeline(steps=[
        ("classifier", OneVsRestClassifier(
            LogisticRegression(
                max_iter=500, class_weight="balanced", random_state=RANDOM_STATE, solver="liblinear", C=1.0
            ), n_jobs=1))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_arr = np.array(y_pred)

    f1_scores = f1_score(y_test, y_pred_arr, average=None, zero_division=0)
    macro_f1 = f1_score(y_test, y_pred_arr, average="macro", zero_division=0)
    micro_f1 = f1_score(y_test, y_pred_arr, average="micro", zero_division=0)

    logger.info(f"Genre Classification - Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}")
    for i, name in enumerate(genre_names):
        logger.info(f"  {name}: F1={f1_scores[i]:.4f}")

    return clf, {"macro_f1": macro_f1, "micro_f1": micro_f1, "per_class_f1": dict(zip(genre_names, f1_scores))}


def main():
    """Run genre classification pipeline."""
    ensure_directories()
    logger.info("Starting Genre Classification Pipeline...")

    tfidf, tfidf_matrix, movie_index, df = load_data()

    X = prepare_features(tfidf_matrix, df)

    train_idx, test_idx = split_data(X, df)

    X_train, X_test = X[train_idx], X[test_idx]

    genres_df = pd.read_parquet(DATA_ENRICHED / "genres.parquet")
    y_genres, genre_names = prepare_genre_targets(df, genres_df)

    y_genres_train, y_genres_test = y_genres[train_idx], y_genres[test_idx]

    logger.info(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

    clf, metrics = train_genre_classifier(X_train, y_genres_train, X_test, y_genres_test, genre_names)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODELS_DIR / "genre_classifier.joblib")

    logger.info("Genre Classification Pipeline Complete!")
    return clf, metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
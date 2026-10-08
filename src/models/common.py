"""Shared utilities for classification models."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_ENRICHED, MODELS_DIR

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data():
    """Load TF-IDF artifacts and enriched movie data."""
    logger.info("Loading TF-IDF artifacts and enriched data...")

    tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    tfidf_matrix = load_npz(MODELS_DIR / "tfidf_matrix.npz")
    movie_index = joblib.load(MODELS_DIR / "movie_index.joblib")

    df = pd.read_parquet(DATA_ENRICHED / "movies_enriched.parquet")

    logger.info(f"Loaded {len(df)} movies, TF-IDF matrix shape: {tfidf_matrix.shape}")
    return tfidf, tfidf_matrix, movie_index, df


def prepare_features(tfidf_matrix, df: pd.DataFrame) -> np.ndarray:
    """
    Prepare feature matrix using Pipeline + ColumnTransformer.

    Key: vote_count is NOT included as a feature (prevents data leakage).
    Only TF-IDF text features + other non-leaky numeric features.
    """
    logger.info("Preparing feature matrix with Pipeline + ColumnTransformer...")

    numeric_cols = ["runtime", "popularity", "release_year"]
    available_numeric = [c for c in numeric_cols if c in df.columns]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, available_numeric)]
    )

    if available_numeric:
        X_numeric_scaled = preprocessor.fit_transform(df[available_numeric].fillna(0))
    else:
        X_numeric_scaled = np.empty((len(df), 0))

    tfidf_dense = tfidf_matrix.toarray() if hasattr(tfidf_matrix, "toarray") else tfidf_matrix

    X = np.hstack([tfidf_dense, X_numeric_scaled])

    logger.info(f"Combined feature shape: {X.shape}")
    logger.info(f"  TF-IDF components: {tfidf_matrix.shape[1] if hasattr(tfidf_matrix, 'shape') else 'sparse'}")
    logger.info(f"  Numeric features: {len(available_numeric)} ({available_numeric})")

    return X


def split_data(X, df: pd.DataFrame):
    """Split data into train/test sets."""
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return train_idx, test_idx
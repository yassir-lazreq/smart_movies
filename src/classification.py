"""
Classification & Regression Models (J5).

3 Models:
1. Multi-label Genre Classification (OneVsRest + LogisticRegression)
2. Revenue Tier Classification (RandomForest)
3. Vote Average Regression (Ridge)
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logging.warning("XGBoost not installed, using RandomForest/Ridge fallback")

from config import DATA_ENRICHED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data():
    """Load TF-IDF artifacts and enriched movie data."""
    logger.info("Loading TF-IDF artifacts and enriched data...")
    
    tfidf = joblib.load("models/tfidf_vectorizer.joblib")
    tfidf_matrix = load_npz("models/tfidf_matrix.npz")
    movie_index = joblib.load("models/movie_index.joblib")
    
    df = pd.read_parquet("data/enriched/movies_enriched.parquet")
    
    # Load genres for multi-label targets
    genres = pd.read_parquet("data/enriched/genres.parquet")
    
    logger.info(f"Loaded {len(df)} movies, TF-IDF matrix shape: {tfidf_matrix.shape}")
    return tfidf, tfidf_matrix, movie_index, df, genres


def prepare_genre_targets(df: pd.DataFrame, genres_df: pd.DataFrame) -> tuple[np.ndarray, list]:
    """Create multi-label genre targets."""
    logger.info("Preparing multi-label genre targets...")
    
    # Get genre names per movie
    genre_map = genres_df.groupby("movie_id")["genre_name"].apply(list).to_dict()
    y_genres = df["id"].map(genre_map).apply(lambda x: x if isinstance(x, list) else []).tolist()
    
    # Binarize multi-label - return numpy array
    mlb = MultiLabelBinarizer()
    y_genres_bin = mlb.fit_transform(y_genres)
    
    logger.info(f"Genre classes: {mlb.classes_.tolist()}")
    logger.info(f"Multi-label shape: {y_genres_bin.shape}")
    return y_genres_bin, mlb.classes_.tolist()


def prepare_revenue_tier_target(df: pd.DataFrame) -> np.ndarray:
    """Prepare revenue tier target (already categorical)."""
    logger.info("Preparing revenue tier target...")
    # revenue_tier already exists from features.py
    y_revenue = df["revenue_tier"].cat.codes.values  # 0-4
    logger.info(f"Revenue tier distribution: {pd.Series(y_revenue).value_counts().sort_index().to_dict()}")
    return y_revenue


def prepare_vote_average_target(df: pd.DataFrame) -> np.ndarray:
    """Prepare vote average target for regression."""
    logger.info("Preparing vote average target...")
    y_vote = df["vote_average"].values
    logger.info(f"Vote average range: {y_vote.min():.2f} - {y_vote.max():.2f}, mean: {y_vote.mean():.2f}")
    return y_vote


def prepare_features(tfidf_matrix, df: pd.DataFrame) -> np.ndarray:
    """Combine TF-IDF with numeric features (no data leakage)."""
    logger.info("Preparing combined feature matrix...")
    
    # Only use numeric features that don't leak target info
    numeric_cols = [
        "runtime", "popularity", "vote_count", "release_year"
    ]
    
    # Select available numeric columns
    available_numeric = [c for c in numeric_cols if c in df.columns]
    
    # Numeric features
    numeric_features = df[available_numeric].fillna(0).values
    
    # Scale numeric features (TF-IDF is already normalized)
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_features)
    
    # Combine sparse TF-IDF with dense features
    tfidf_dense = tfidf_matrix.toarray() if hasattr(tfidf_matrix, 'toarray') else tfidf_matrix
    
    X = np.hstack([tfidf_dense, numeric_features])
    
    logger.info(f"Combined feature shape: {X.shape} (TF-IDF: {tfidf_matrix.shape[1] if hasattr(tfidf_matrix, 'shape') else 'sparse'}, numeric: {len(available_numeric)})")
    return X


def train_genre_classifier(X_train, y_train, X_test, y_test, genre_names):
    """Train multi-label genre classifier."""
    logger.info("Training Genre Classification (OneVsRest + LogisticRegression)...")
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
# Use OneVsRest for multi-label - use liblinear solver for better convergence
    clf = MultiOutputClassifier(
        LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE, C=1.0, solver="liblinear"),
        n_jobs=1
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    y_pred_arr = np.array(y_pred)
    
    # Per-class F1
    f1_scores = f1_score(y_test, y_pred_arr, average=None, zero_division=0)
    macro_f1 = f1_score(y_test, y_pred_arr, average="macro", zero_division=0)
    micro_f1 = f1_score(y_test, y_pred_arr, average="micro", zero_division=0)
    
    logger.info(f"Genre Classification - Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}")
    for i, name in enumerate(genre_names):
        logger.info(f"  {name}: F1={f1_scores[i]:.4f}")
    
    return clf, {"macro_f1": macro_f1, "micro_f1": micro_f1, "per_class_f1": dict(zip(genre_names, f1_scores))}


def train_revenue_classifier(X_train, y_train, X_test, y_test):
    """Train revenue tier classifier (without budget/revenue leakage)."""
    logger.info("Training Revenue Tier Classification...")
    
    if HAS_XGB:
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    
    logger.info(f"Revenue Tier - Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    return clf, {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


def train_vote_regressor(X_train, y_train, X_test, y_test):
    """Train vote average regressor."""
    logger.info("Training Vote Average Regression...")
    
    if HAS_XGB:
        reg = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )
    else:
        # Use Ridge with higher alpha to avoid singular matrix
        reg = Ridge(alpha=10.0, random_state=RANDOM_STATE)
    
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Vote Average Regression - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    return reg, {"mae": mae, "rmse": rmse, "r2": r2}


def save_model(model, name: str, metrics: dict, models_dir: Path = Path("models")):
    """Save model and metrics."""
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / f"{name}.joblib")
    joblib.dump(metrics, models_dir / f"{name}_metrics.joblib")
    logger.info(f"Saved {name} to {models_dir}")


def run_classification_pipeline():
    """Full classification pipeline."""
    logger.info("Starting J5 Classification Pipeline...")
    
    # Load data
    tfidf, tfidf_matrix, movie_index, df, genres_df = load_data()
    
    # Prepare targets
    y_genres, genre_names = prepare_genre_targets(df, genres_df)
    y_revenue = prepare_revenue_tier_target(df)
    y_vote = prepare_vote_average_target(df)
    
    # Prepare features (TF-IDF only, no leakage)
    X = prepare_features(tfidf_matrix, df)
    
    # Split data (same split for all tasks)
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_genres_train, y_genres_test = y_genres[train_idx], y_genres[test_idx]
    y_rev_train, y_rev_test = y_revenue[train_idx], y_revenue[test_idx]
    y_vote_train, y_vote_test = y_vote[train_idx], y_vote[test_idx]
    
    logger.info(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
    
    # Train 3 models
    logger.info("=" * 50)
    logger.info("MODEL 1: Genre Classification")
    logger.info("=" * 50)
    genre_model, genre_metrics = train_genre_classifier(
        X_train, y_genres[train_idx], X_test, y_genres[test_idx], genre_names
    )
    
    logger.info("=" * 50)
    logger.info("MODEL 2: Revenue Tier Classification")
    logger.info("=" * 50)
    revenue_model, revenue_metrics = train_revenue_classifier(
        X_train, y_revenue[train_idx], X_test, y_revenue[test_idx]
    )
    
    logger.info("=" * 50)
    logger.info("MODEL 3: Vote Average Regression")
    logger.info("=" * 50)
    vote_model, vote_metrics = train_vote_regressor(
        X_train, y_vote[train_idx], X_test, y_vote[test_idx]
    )
    
    # Save all models
    save_model(genre_model, "genre_classifier", genre_metrics)
    save_model(revenue_model, "revenue_tier_classifier", revenue_metrics)
    save_model(vote_model, "vote_average_regressor", vote_metrics)
    
    logger.info("=" * 50)
    logger.info("J5 Classification Pipeline Complete!")
    logger.info("=" * 50)
    
    return {
        "genre_classifier": (genre_model, genre_metrics),
        "revenue_tier_classifier": (revenue_model, revenue_metrics),
        "vote_average_regressor": (vote_model, vote_metrics),
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_classification_pipeline()
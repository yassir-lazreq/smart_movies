"""
Classification & Regression Models (J5 + Étape 6).

3 Models compared:
1. Multi-label Genre Classification (OneVsRest + LogisticRegression)
2. High Engagement Classification (RandomForest + Pipeline + ColumnTransformer)
3. Revenue Tier Classification (RandomForest, no leakage)

Important: the variable used to create the target (high_engagement from vote_count)
must NOT be used as a feature.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

try:
    from xgboost import XGBClassifier
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
THRESHOLD_PERCENTILE = 70  # top 30% vote_count = high engagement


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


def load_data():
    """Load TF-IDF artifacts and enriched movie data."""
    logger.info("Loading TF-IDF artifacts and enriched data...")

    tfidf = joblib.load("models/tfidf_vectorizer.joblib")
    tfidf_matrix = load_npz("models/tfidf_matrix.npz")
    movie_index = joblib.load("models/movie_index.joblib")

    df = pd.read_parquet("data/enriched/movies_enriched.parquet")

    logger.info(f"Loaded {len(df)} movies, TF-IDF matrix shape: {tfidf_matrix.shape}")
    return tfidf, tfidf_matrix, movie_index, df


def prepare_features(tfidf_matrix, df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix using Pipeline + ColumnTransformer.

    Key: vote_count is NOT included as a feature (prevents data leakage).
    Only TF-IDF text features + other non-leaky numeric features.
    """
    logger.info("Preparing feature matrix with Pipeline + ColumnTransformer...")

    # Numeric features that don't leak target info
    # - runtime: production time
    # - popularity: TMDB popularity score
    # - release_year: year of release
    # NOT included: vote_count (defines the target!)
    numeric_cols = ["runtime", "popularity", "release_year"]
    available_numeric = [c for c in numeric_cols if c in df.columns]

    # Categorical features (if any)
    categorical_cols = [
        c for c in df.columns if df[c].dtype == "object" and c not in ["title", "overview", "genres"]
    ]
    available_categorical = [c for c in categorical_cols if c in df.columns]

    # Numeric pipeline: scale numeric features
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    # Categorical pipeline: one-hot encode
    if available_categorical:
        categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, available_numeric),
                ("cat", categorical_transformer, available_categorical),
            ],
            remainder="drop",
        )
    else:
        preprocessor = Pipeline(steps=[("scaler", numeric_transformer)])

    # Fit and transform numeric features
    if available_numeric:
        X_numeric = df[available_numeric].fillna(0).values
        X_numeric_scaled = preprocessor.fit_transform(X_numeric)

        # If only numeric, squeeze dimensions
        if not available_categorical:
            X_numeric_scaled = X_numeric_scaled.reshape(X_numeric_scaled.shape[0], -1)
    else:
        X_numeric_scaled = np.empty((len(df), 0))

    # Combine TF-IDF with processed features
    tfidf_dense = tfidf_matrix.toarray() if hasattr(tfidf_matrix, "toarray") else tfidf_matrix

    X = np.hstack([tfidf_dense, X_numeric_scaled])

    logger.info(f"Combined feature shape: {X.shape}")
    logger.info(f"  TF-IDF components: {tfidf_matrix.shape[1] if hasattr(tfidf_matrix, 'shape') else 'sparse'}")
    logger.info(f"  Numeric features: {len(available_numeric)} ({available_numeric})")
    logger.info(f"  Categorical features: {len(available_categorical)} ({available_categorical})")

    return X


def train_high_engagement_classifier(X_train, y_train, X_test, y_test):
    """
    Train high engagement classifier comparing 3 models:
    - Logistic Regression
    - Random Forest
    - Linear SVM

    Uses Pipeline with ColumnTransformer.
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

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # ROC-AUC (binary classification)
        try:
            if hasattr(pipeline["classifier"], "predict_proba"):
                proba = pipeline["classifier"].predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, proba)
            else:
                decision = pipeline["classifier"].decision_function(X_test)
                auc = roc_auc_score(y_test, decision)
        except Exception:
            auc = float("nan")

        # Confusion matrix
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

        # Plot confusion matrix
        import matplotlib.pyplot as plt

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Engagement", "High Engagement"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"High Engagement - {name}")
        plt.tight_layout()
        plt.savefig(f"reports/figures/8_high_engagement_{name.replace(' ', '_')}.png", dpi=150)
        plt.close()

    # Display comparison table
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


def prepare_genre_targets(df: pd.DataFrame, genres_df: pd.DataFrame) -> tuple:
    """Create multi-label genre targets."""
    logger.info("Preparing multi-label genre targets...")

    # Get genre names per movie
    genre_map = genres_df.groupby("movie_id")["genre_name"].apply(list).to_dict()
    y_genres = df["id"].map(genre_map).apply(lambda x: x if isinstance(x, list) else []).tolist()

    # Binarize multi-label - return numpy array
    from sklearn.preprocessing import MultiLabelBinarizer

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

    # Per-class F1
    from sklearn.metrics import f1_score

    f1_scores = f1_score(y_test, y_pred_arr, average=None, zero_division=0)
    macro_f1 = f1_score(y_test, y_pred_arr, average="macro", zero_division=0)
    micro_f1 = f1_score(y_test, y_pred_arr, average="micro", zero_division=0)

    logger.info(f"Genre Classification - Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}")
    for i, name in enumerate(genre_names):
        logger.info(f"  {name}: F1={f1_scores[i]:.4f}")

    return clf, {"macro_f1": macro_f1, "micro_f1": micro_f1, "per_class_f1": dict(zip(genre_names, f1_scores))}


def train_revenue_classifier(X_train, y_train, X_test, y_test):
    """Train revenue tier classifier (no budget/revenue leakage)."""
    logger.info("Training Revenue Tier Classification (no leakage)...")

    if HAS_XGB:
        clf = Pipeline(steps=[("classifier", XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0))])
    else:
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

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    import matplotlib.pyplot as plt

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Tier {i}" for i in range(5)])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Revenue Tier Classification")
    plt.tight_layout()
    plt.savefig("reports/figures/9_revenue_tier_cm.png", dpi=150)
    plt.close()

    return clf, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}


def run_classification_pipeline():
    """Full classification pipeline implementing Étape 6 requirements."""
    logger.info("Starting J5+ Étape 6 Classification Pipeline...")

    # Load data
    tfidf, tfidf_matrix, movie_index, df = load_data()

    # Create high_engagement target (Étape 6 requirement)
    y_high_engagement = build_high_engagement_target(df, THRESHOLD_PERCENTILE)

    # Prepare features (vote_count is intentionally NOT included)
    X = prepare_features(tfidf_matrix, df)

    # Split data
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    X_train, X_test = X[train_idx], X[test_idx]
    y_high_train, y_high_test = y_high_engagement.iloc[train_idx], y_high_engagement.iloc[test_idx]

    # Get other targets
    genres_df = pd.read_parquet("data/enriched/genres.parquet")
    y_genres, genre_names = prepare_genre_targets(df, genres_df)
    y_revenue = prepare_revenue_tier_target(df)

    # Split same indices for all tasks
    y_genres_train, y_genres_test = y_genres[train_idx], y_genres[test_idx]
    y_rev_train, y_rev_test = y_revenue[train_idx], y_revenue[test_idx]

    logger.info(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

    # ==========================================
    # Étape 6: High Engagement Classification
    # ==========================================
    logger.info("=" * 60)
    logger.info("ÉTAPE 6: HIGH ENGAGEMENT CLASSIFICATION")
    logger.info("=" * 60)
    engagement_results = train_high_engagement_classifier(
        X_train, y_high_train, X_test, y_high_test
    )

    # ==========================================
    # Model 2: Genre Classification
    # ==========================================
    logger.info("=" * 60)
    logger.info("MODEL 2: GENRE CLASSIFICATION")
    logger.info("=" * 60)
    genre_model, genre_metrics = train_genre_classifier(
        X_train, y_genres_train, X_test, y_genres_test, genre_names
    )

    # ==========================================
    # Model 3: Revenue Tier Classification
    # ==========================================
    logger.info("=" * 60)
    logger.info("MODEL 3: REVENUE TIER CLASSIFICATION")
    logger.info("=" * 60)
    revenue_model, revenue_metrics = train_revenue_classifier(
        X_train, y_rev_train, X_test, y_rev_test
    )

    # Save all models and artifacts
    joblib.dump(engagement_results, "models/high_engagement_results.joblib")
    joblib.dump(genre_model, "models/genre_classifier.joblib")
    joblib.dump(revenue_model, "models/revenue_tier_classifier.joblib")

    # Save feature names info
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.text(0.1, 0.9, f"TF-IDF features: {tfidf_matrix.shape[1]}", fontsize=12)
    plt.text(0.1, 0.8, f"Combined feature matrix: {X.shape[1]} dimensions", fontsize=12)
    plt.text(0.1, 0.7, f"High engagement threshold: {THRESHOLD_PERCENTILE}th percentile of vote_count", fontsize=12)
    plt.text(0.1, 0.3, "vote_count is NOT used as a feature (target leakage prevention)", fontsize=12, color="red")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("reports/figures/6_pipeline_overview.png", dpi=150)
    plt.close()

    logger.info("=" * 60)
    logger.info("J5+ Étape 6 Classification Pipeline Complete!")
    logger.info("=" * 60)

    return {
        "high_engagement": engagement_results,
        "genre_classifier": (genre_model, genre_metrics),
        "revenue_tier_classifier": (revenue_model, revenue_metrics),
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_classification_pipeline()
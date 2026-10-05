"""
TF-IDF Vectorizer + Cosine Similarity Recommender (J4).

Builds TF-IDF on `search_text` (overview + tagline + genres + keywords)
and saves cosine similarity matrix for content-based recommendations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import DATA_ENRICHED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def build_tfidf(
    df: pd.DataFrame,
    text_col: str = "search_text",
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
) -> tuple[TfidfVectorizer, pd.DataFrame]:
    """Fit TF-IDF on text column and return vectorizer + transformed matrix."""
    logger.info(f"Building TF-IDF on '{text_col}' (max_features={max_features})")
    
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )
    
    texts = df[text_col].fillna("").astype(str)
    tfidf_matrix = tfidf.fit_transform(texts)
    
    logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    logger.info(f"Vocabulary size: {len(tfidf.vocabulary_)}")
    
    return tfidf, tfidf_matrix


def build_cosine_similarity(tfidf_matrix) -> pd.DataFrame:
    """Compute cosine similarity matrix."""
    logger.info("Computing cosine similarity...")
    cosine_sim = cosine_similarity(tfidf_matrix, dense_output=False)
    logger.info(f"Cosine similarity shape: {cosine_sim.shape}")
    return cosine_sim


def save_artifacts(
    tfidf: TfidfVectorizer,
    tfidf_matrix,
    cosine_sim,
    df: pd.DataFrame,
    models_dir: Path = MODELS_DIR,
) -> None:
    """Save all artifacts for later use."""
    models_dir.mkdir(exist_ok=True)
    
    joblib.dump(tfidf, models_dir / "tfidf_vectorizer.joblib")
    save_npz(models_dir / "tfidf_matrix.npz", tfidf_matrix)
    save_npz(models_dir / "cosine_similarity.npz", cosine_sim)
    
    # Save minimal movie index for lookup
    movie_index = df[["id", "title", "search_text"]].reset_index(drop=True)
    joblib.dump(movie_index, models_dir / "movie_index.joblib")
    
    logger.info(f"Artifacts saved to {models_dir}")


def load_artifacts(models_dir: Path = MODELS_DIR):
    """Load saved artifacts."""
    tfidf = joblib.load(models_dir / "tfidf_vectorizer.joblib")
    tfidf_matrix = load_npz(models_dir / "tfidf_matrix.npz")
    cosine_sim = load_npz(models_dir / "cosine_similarity.npz")
    movie_index = joblib.load(models_dir / "movie_index.joblib")
    return tfidf, tfidf_matrix, cosine_sim, movie_index


def recommend(
    movie_id: int,
    cosine_sim,
    movie_index: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return top-N similar movies for a given movie_id."""
    idx = movie_index.index[movie_index["id"] == movie_id]
    if len(idx) == 0:
        raise ValueError(f"Movie ID {movie_id} not found in index")
    idx = idx[0]
    
    sim_scores = list(enumerate(cosine_sim[idx].toarray().flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    
    results = movie_index.iloc[movie_indices].copy()
    results["similarity"] = scores
    return results[["id", "title", "similarity"]]


def run_tfidf_pipeline(
    enriched_path: Path = DATA_ENRICHED / "movies_enriched.parquet",
    models_dir: Path = MODELS_DIR,
) -> dict:
    """Full TF-IDF pipeline: load → vectorize → similarity → save."""
    logger.info("Loading enriched data...")
    df = pd.read_parquet(enriched_path)
    logger.info(f"Loaded {len(df)} movies")
    
    tfidf, tfidf_matrix = build_tfidf(df)
    cosine_sim = build_cosine_similarity(tfidf_matrix)
    save_artifacts(tfidf, tfidf_matrix, cosine_sim, df)
    
    return {"tfidf": tfidf, "tfidf_matrix": tfidf_matrix, "cosine_sim": cosine_sim, "df": df}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_tfidf_pipeline()
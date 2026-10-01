"""
Feature engineering & enrichment (Gold layer).

Reads silver parquet from data/processed/, writes enriched parquet to data/enriched/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
ENRICHED_DIR = Path("data/enriched")

# ISO 639-1 code -> full name
LANG_MAP = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German", "it": "Italian",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "hi": "Hindi", "pt": "Portuguese",
    "ru": "Russian", "ar": "Arabic", "nl": "Dutch", "pl": "Polish", "tr": "Turkish",
    "sv": "Swedish", "da": "Danish", "no": "Norwegian", "fi": "Finnish", "cs": "Czech",
    "hu": "Hungarian", "ro": "Romanian", "he": "Hebrew", "th": "Thai", "vi": "Vietnamese",
    "id": "Indonesian", "ms": "Malay", "el": "Greek", "hr": "Croatian", "sk": "Slovak",
    "bg": "Bulgarian", "lt": "Lithuanian", "lv": "Latvian", "et": "Estonian", "sl": "Slovenian",
    "sr": "Serbian", "uk": "Ukrainian", "fa": "Persian", "bn": "Bengali", "ta": "Tamil",
    "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "pa": "Punjabi", "or": "Odia", "as": "Assamese", "ne": "Nepali", "si": "Sinhala",
    "my": "Burmese", "km": "Khmer", "lo": "Lao", "ka": "Georgian", "am": "Amharic",
    "sw": "Swahili", "zu": "Zulu", "af": "Afrikaans", "sq": "Albanian", "eu": "Basque",
    "ca": "Catalan", "gl": "Galician", "is": "Icelandic", "ga": "Irish", "mt": "Maltese",
    "cy": "Welsh", "mk": "Macedonian", "bs": "Bosnian", "me": "Montenegrin",
}


def _cap_outliers(df: pd.DataFrame, cols: list[str], factor: float = 1.5) -> pd.DataFrame:
    """IQR-based outlier capping (winsorization)."""
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                lower, upper = q1 - factor * iqr, q3 + factor * iqr
                before = df[c].clip(lower, upper)
                changed = (df[c] != before).sum()
                if changed:
                    logger.info("  %s: capped %d outliers to [%.2f, %.2f]", c, changed, lower, upper)
                    df[c] = before
    return df


def _add_language_name(df: pd.DataFrame) -> pd.DataFrame:
    """Map original_language ISO code -> full name."""
    if "original_language" in df.columns:
        df["original_language_name"] = df["original_language"].map(LANG_MAP).fillna(df["original_language"])
    return df


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features."""
    # Profit & ROI
    if {"budget", "revenue"}.issubset(df.columns):
        df["profit"] = df["revenue"] - df["budget"]
        df["roi"] = df["profit"] / df["budget"].replace(0, np.nan)

    # Decade
    if "release_year" in df.columns:
        df["decade"] = (df["release_year"] // 10) * 10

    # Runtime buckets
    if "runtime" in df.columns:
        bins = [0, 60, 90, 120, 150, 180, 999]
        labels = ["<60", "60-90", "90-120", "120-150", "150-180", ">180"]
        df["runtime_bucket"] = pd.cut(df["runtime"], bins=bins, labels=labels, right=True)

    # Vote popularity score (weighted)
    if {"vote_average", "vote_count", "popularity"}.issubset(df.columns):
        # IMDB-style weighted rating
        m = df["vote_count"].quantile(0.8)
        C = df["vote_average"].mean()
        df["weighted_rating"] = (df["vote_count"] / (df["vote_count"] + m)) * df["vote_average"] + (m / (df["vote_count"] + m)) * C

    # Budget/revenue tiers
    for col in ["budget", "revenue"]:
        if col in df.columns:
            df[f"{col}_tier"] = pd.qcut(df[col].fillna(0), q=5, labels=["Very Low", "Low", "Medium", "High", "Very High"], duplicates="drop")

    return df


def _build_search_text(df: pd.DataFrame, genres_df: pd.DataFrame, keywords_df: pd.DataFrame) -> pd.DataFrame:
    """Create merged text column for full-text search / embeddings."""
    # Aggregate genres per movie
    genre_text = genres_df.groupby("movie_id")["genre_name"].apply(lambda x: " ".join(x.astype(str))).reset_index()
    genre_text.columns = ["movie_id", "genre_text"]

    # Aggregate keywords per movie
    kw_text = keywords_df.groupby("movie_id")["keyword_name"].apply(lambda x: " ".join(x.astype(str))).reset_index()
    kw_text.columns = ["movie_id", "keyword_text"]

    # Merge
    df = df.merge(genre_text, left_on="id", right_on="movie_id", how="left").drop(columns=["movie_id"])
    df = df.merge(kw_text, left_on="id", right_on="movie_id", how="left").drop(columns=["movie_id"])

    # Build search_text
    parts = [
        df["overview"].fillna(""),
        df["tagline"].fillna(""),
        df["genre_text"].fillna(""),
        df["keyword_text"].fillna(""),
    ]
    df["search_text"] = pd.concat(parts, axis=1).agg(" ".join, axis=1)
    df["search_text"] = df["search_text"].str.replace(r"\s+", " ", regex=True).str.strip()

    # Drop intermediates
    df = df.drop(columns=["genre_text", "keyword_text"])
    return df


def _log_volumes(df: pd.DataFrame, name: str) -> None:
    """Log data quality summary."""
    logger.info("=== Volume Report: %s ===", name)
    logger.info("  Rows: %d, Cols: %d", len(df), df.shape[1])
    logger.info("  Nulls per column:")
    for c in df.columns:
        null_pct = df[c].isnull().mean() * 100
        nunique = df[c].nunique()
        logger.info("    %-25s %7.2f%% null  |  %6d unique", c, null_pct, nunique)
    if "id" in df.columns:
        logger.info("  Unique IDs: %d", df["id"].nunique())
    logger.info("  Memory: %.2f MB", df.memory_usage(deep=True).sum() / 1024**2)


def run_features(
    processed_dir: Path = PROCESSED_DIR,
    enriched_dir: Path = ENRICHED_DIR,
) -> dict[str, pd.DataFrame]:
    """Orchestrate silver -> gold enrichment."""
    enriched_dir.mkdir(parents=True, exist_ok=True)

    # Load silver tables
    logger.info("Loading silver tables from %s", processed_dir)
    movies = pd.read_parquet(processed_dir / "movies.parquet")
    genres = pd.read_parquet(processed_dir / "genres.parquet")
    keywords = pd.read_parquet(processed_dir / "keywords.parquet")
    production_companies = pd.read_parquet(processed_dir / "production_companies.parquet")
    production_countries = pd.read_parquet(processed_dir / "production_countries.parquet")
    spoken_languages = pd.read_parquet(processed_dir / "spoken_languages.parquet")
    origin_country = pd.read_parquet(processed_dir / "origin_country.parquet")

    tables = {}

    # --- Enrich movies ---
    logger.info("Enriching movies table...")
    movies = _cap_outliers(movies, ["budget", "revenue", "runtime", "popularity", "vote_average", "vote_count"])
    movies = _add_language_name(movies)
    movies = _add_features(movies)
    movies = _build_search_text(movies, genres, keywords)
    tables["movies_enriched"] = movies

    # Other tables pass through (already clean)
    tables["genres"] = genres
    tables["keywords"] = keywords
    tables["production_companies"] = production_companies
    tables["production_countries"] = production_countries
    tables["spoken_languages"] = spoken_languages
    tables["origin_country"] = origin_country

    # --- Volume reports ---
    for name, df in tables.items():
        _log_volumes(df, name)

    # --- Write enriched parquet ---
    logger.info("Writing enriched tables to %s", enriched_dir)
    for name, df in tables.items():
        out = enriched_dir / f"{name}.parquet"
        df.to_parquet(out, index=False)
        logger.info("  %s  (%d rows × %d cols)", out, df.shape[0], df.shape[1])

    logger.info("Feature engineering complete.")
    return tables


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_features()
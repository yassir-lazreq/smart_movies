"""
Transform raw TMDB JSON files into cleaned, normalized parquet tables.

Raw    : data/raw/movie_*.json       (one file per movie)
Silver : data/processed/*.parquet

Usage:
    python -c "from src.transform import run_transform; run_transform()"

Design principles:
  - Idempotent: re-running produces identical output for identical input.
  - Schema-aware: validates required fields, warns on unexpected schema drift.
  - MongoDB-friendly: writes int (not int64) for ids, NaN (not pd.NA) for nulls.
  - Fail-fast on structural errors; degrade gracefully on per-file corruption.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- constants
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

REQUIRED_FIELDS = {"id", "title"}

# Scalar fields kept in the main movies table (flat, no nesting).
MOVIE_SCALAR_FIELDS = (
    "id", "imdb_id", "title", "original_title", "original_language",
    "overview", "tagline", "status", "release_date",
    "runtime", "budget", "revenue", "popularity", "vote_average", "vote_count",
    "adult", "video", "homepage", "poster_path", "backdrop_path",
)

# Boolean-ish fields we want normalized to real bools.
BOOL_FIELDS = ("adult", "video")

# Fields TMDB encodes with 0 == unknown. We convert these to NaN + has_* flag.
ZERO_AS_MISSING_FIELDS = ("budget", "revenue")

# Free-text fields that may contain control chars → strip before parquet.
FREE_TEXT_FIELDS = ("overview", "tagline", "title", "original_title")

# Nested list-of-dicts fields → extracted into their own long-format tables.
# `id_type` tells us whether the entity's id is int or str (BSON-safe coercion).
NESTED_FIELDS: dict[str, dict[str, str]] = {
    "genres": {
        "flat_id":   "genre_id",
        "flat_name": "genre_name",
        "src_id":    "id",
        "src_name":  "name",
        "id_type":   "int",
    },
    "production_companies": {
        "flat_id":   "company_id",
        "flat_name": "company_name",
        "src_id":    "id",
        "src_name":  "name",
        "id_type":   "int",
    },
    "production_countries": {
        "flat_id":   "country_code",
        "flat_name": "country_name",
        "src_id":    "iso_3166_1",
        "src_name":  "name",
        "id_type":   "str",
    },
    "spoken_languages": {
        "flat_id":   "language_code",
        "flat_name": "language_name",
        "src_id":    "iso_639_1",
        "src_name":  "name",
        "id_type":   "str",
    },
    # origin_country is a list of strings, not list-of-dicts → special-cased below.
}


# ---------------------------------------------------------------- helpers
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean_text(value: Any) -> Any:
    """Strip control characters from a string; pass through non-strings."""
    if isinstance(value, str):
        return _CONTROL_CHARS.sub("", value)
    return value


def _iter_raw_files(raw_dir: Path) -> Iterable[Path]:
    """Yield movie JSON paths in deterministic order."""
    return sorted(raw_dir.glob("movie_*.json"))


def _load_raw_files(raw_dir: Path) -> tuple[list[dict[str, Any]], int]:
    """Load every raw JSON file. Returns (valid_records, n_corrupt)."""
    files = list(_iter_raw_files(raw_dir))
    if not files:
        raise FileNotFoundError(f"No movie_*.json found in {raw_dir.resolve()}")

    records: list[dict[str, Any]] = []
    corrupt = 0
    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"top-level JSON is {type(data).__name__}, expected dict")
            records.append(data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Skipping corrupt file %s: %s", fp.name, e)
            corrupt += 1

    logger.info("Loaded %d raw files (%d corrupt) from %s", len(records), corrupt, raw_dir)
    return records, corrupt


def _validate_schema(records: list[dict[str, Any]]) -> None:
    """Raise if any record is missing a required field."""
    missing = [i for i, r in enumerate(records) if not REQUIRED_FIELDS.issubset(r)]
    if missing:
        raise ValueError(
            f"{len(missing)} records missing required fields {REQUIRED_FIELDS} "
            f"(first offenders: {missing[:5]})"
        )


def _coerce_int_series(series: pd.Series) -> pd.Series:
    """
    Convert a numeric series to nullable Int64 with Python-int values.

    Required for MongoDB: BSON cannot serialize numpy.int64.
    """
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _coerce_str_series(series: pd.Series) -> pd.Series:
    """Normalize a possibly-string series; empty strings → None."""
    s = series.astype("string")
    return s.where(s.str.len() > 0, pd.NA)


def _build_movies_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten scalar fields into the main movies DataFrame."""
    present = set().union(*(r.keys() for r in records))
    fields = [f for f in MOVIE_SCALAR_FIELDS if f in present]
    dropped = set(MOVIE_SCALAR_FIELDS) - present
    if dropped:
        logger.info("Fields absent from raw data (skipped): %s", sorted(dropped))

    df = pd.DataFrame([{k: r.get(k) for k in fields} for r in records])

    # --- id integrity
    if df["id"].isnull().any() or df["id"].duplicated().any():
        n_null = int(df["id"].isnull().sum())
        n_dup  = int(df["id"].duplicated().sum())
        logger.warning("id issues → null: %d, duplicated: %d", n_null, n_dup)
        df = df.dropna(subset=["id"]).drop_duplicates(subset="id", keep="last")

    # --- clean free text (control chars break parquet & BSON)
    for col in FREE_TEXT_FIELDS:
        if col in df.columns:
            df[col] = df[col].map(_clean_text)

    # --- booleans
    for col in BOOL_FIELDS:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    # --- numeric coercion
    numeric_cols = ("budget", "revenue", "runtime", "popularity", "vote_average", "vote_count")
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- TMDB uses 0 to mean "unknown" for budget/revenue
    for col in ZERO_AS_MISSING_FIELDS:
        if col in df.columns:
            df[f"has_{col}"] = df[col].fillna(0) > 0
            df.loc[df[col] == 0, col] = np.nan

    # --- dates
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"]  = df["release_date"].dt.year.astype("Int64")
        df["release_month"] = df["release_date"].dt.month.astype("Int64")

    # --- belongs_to_collection: flatten to two scalar fields
    if "belongs_to_collection" in present:
        collections = [r.get("belongs_to_collection") for r in records if r.get("id") in df["id"].values]
        # Align by re-indexing on id to keep row alignment after dedup
        id_to_collection = {
            r.get("id"): r.get("belongs_to_collection") for r in records
            if isinstance(r.get("belongs_to_collection"), dict)
        }
        df["collection_id"]   = df["id"].map(lambda x: (id_to_collection.get(x) or {}).get("id"))
        df["collection_name"] = df["id"].map(lambda x: (id_to_collection.get(x) or {}).get("name"))
        df["collection_id"]   = _coerce_int_series(df["collection_id"])
        df["collection_name"] = _coerce_str_series(df["collection_name"])

    # --- MongoDB-safe id
    df["id"] = _coerce_int_series(df["id"])

    return df.reset_index(drop=True)


def _extract_nested_json_normalize(
    records: list[dict[str, Any]],
    field: str,
    cfg: dict[str, str],
) -> pd.DataFrame:
    """
    Extract a list-of-dicts field into a long-format DataFrame:
        movie_id | <flat_id> | <flat_name>
    """
    empty_cols = ["movie_id", cfg["flat_id"], cfg["flat_name"]]

    try:
        flat = pd.json_normalize(
            records,
            record_path=field,
            meta=["id"],
            meta_prefix="movie_",
            errors="ignore",
        )
    except (KeyError, TypeError) as e:
        logger.warning("Could not normalize field '%s': %s", field, e)
        return pd.DataFrame(columns=empty_cols)

    if flat.empty:
        logger.warning("Field '%s' → no rows extracted", field)
        return pd.DataFrame(columns=empty_cols)

    expected = {cfg["src_id"], cfg["src_name"], "id"}
    missing = expected - set(flat.columns)
    if missing:
        logger.warning("Field '%s' is missing expected keys: %s", field, sorted(missing))
        flat = flat.drop(columns=[c for c in missing if c in flat.columns])

    flat = flat.rename(columns={
        "movie_id":      "movie_id",
        cfg["src_id"]:   cfg["flat_id"],
        cfg["src_name"]: cfg["flat_name"],
    })

    keep = ["movie_id", cfg["flat_id"], cfg["flat_name"]]
    flat = flat[[c for c in keep if c in flat.columns]]

    flat = flat.drop_duplicates(subset=["movie_id", cfg["flat_id"]]).reset_index(drop=True)

    # --- MongoDB-safe types (respect id_type)
    flat["movie_id"] = _coerce_int_series(flat["movie_id"])
    if cfg["flat_id"] in flat.columns:
        if cfg["id_type"] == "int":
            flat[cfg["flat_id"]] = _coerce_int_series(flat[cfg["flat_id"]])
        else:
            flat[cfg["flat_id"]] = _coerce_str_series(flat[cfg["flat_id"]])
    if cfg["flat_name"] in flat.columns:
        flat[cfg["flat_name"]] = _coerce_str_series(flat[cfg["flat_name"]])

    logger.info("  %-22s → %6d rows", field, len(flat))
    return flat


def _extract_origin_country(records: list[dict[str, Any]]) -> pd.DataFrame:
    """origin_country is a list of ISO codes (strings), not list-of-dicts."""
    rows = []
    for r in records:
        movie_id = r.get("id")
        for code in r.get("origin_country") or []:
            rows.append({"movie_id": movie_id, "country_code": code})

    df = pd.DataFrame(rows, columns=["movie_id", "country_code"])
    if df.empty:
        logger.warning("Field 'origin_country' → no rows extracted")
        return df

    df = df.drop_duplicates().reset_index(drop=True)
    df["movie_id"]     = _coerce_int_series(df["movie_id"])
    df["country_code"] = _coerce_str_series(df["country_code"])
    logger.info("  %-22s → %6d rows", "origin_country", len(df))
    return df


def _extract_keywords(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Extract keywords from nested structure: keywords.keywords[]."""
    rows = []
    for r in records:
        movie_id = r.get("id")
        kw_data = r.get("keywords")
        if isinstance(kw_data, dict):
            for kw in kw_data.get("keywords") or []:
                rows.append({"movie_id": movie_id, "keyword_id": kw.get("id"), "keyword_name": kw.get("name")})

    df = pd.DataFrame(rows, columns=["movie_id", "keyword_id", "keyword_name"])
    if df.empty:
        logger.warning("Field 'keywords' → no rows extracted")
        return df

    df = df.drop_duplicates().reset_index(drop=True)
    df["movie_id"]   = _coerce_int_series(df["movie_id"])
    df["keyword_id"] = _coerce_int_series(df["keyword_id"])
    df["keyword_name"] = _coerce_str_series(df["keyword_name"])
    logger.info("  %-22s → %6d rows", "keywords", len(df))
    return df


# ---------------------------------------------------------------- main
def run_transform(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> dict[str, pd.DataFrame]:
    """Orchestrate the raw → silver transformation."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    records, _ = _load_raw_files(raw_dir)
    _validate_schema(records)

    tables: dict[str, pd.DataFrame] = {}

    logger.info("Building movies table...")
    tables["movies"] = _build_movies_df(records)

    logger.info("Extracting nested tables...")
    for field, cfg in NESTED_FIELDS.items():
        tables[field] = _extract_nested_json_normalize(records, field, cfg)

    # Keywords has nested structure: {"keywords": {"keywords": [...]}}
    tables["keywords"] = _extract_keywords(records)

    tables["origin_country"] = _extract_origin_country(records)

    logger.info("Writing parquet files...")
    for name, df in tables.items():
        if df.empty:
            logger.warning("Skipping empty table: %s", name)
            continue
        out = processed_dir / f"{name}.parquet"
        df.to_parquet(out, index=False)
        logger.info("  %s  (%d rows × %d cols)", out, df.shape[0], df.shape[1])

    logger.info("Transform complete.")
    return tables


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_transform()
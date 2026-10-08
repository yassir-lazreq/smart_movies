"""
Load enriched parquet into MongoDB + run required queries/aggregations.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

from config import DATA_ENRICHED, ensure_directories

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "movie_intelligence"


def get_client() -> MongoClient:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    logger.info("Connected to MongoDB at %s", MONGO_URI)
    return client


def load_collection(db, name: str, df: pd.DataFrame, drop: bool = True) -> int:
    """Insert DataFrame records into a collection."""
    if drop:
        db[name].drop()
    records = df.to_dict("records")
    if records:
        try:
            db[name].insert_many(records, ordered=False)
            logger.info("  Loaded %d docs into %s", len(records), name)
        except BulkWriteError as e:
            logger.warning("Bulk write errors (duplicates?): %s", e.details)
    return len(records)


def build_movies_embedded(movies: pd.DataFrame, nested: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Embed genres, keywords, companies, countries, languages into movies."""
    df = movies.copy()

    for field, cfg in {
        "genres": ("genre_id", "genre_name"),
        "keywords": ("keyword_id", "keyword_name"),
        "production_companies": ("company_id", "company_name"),
        "production_countries": ("country_code", "country_name"),
        "spoken_languages": ("language_code", "language_name"),
    }.items():
        if field not in nested:
            continue
        id_col, name_col = cfg
        agg = nested[field].groupby("movie_id").apply(
            lambda g: g[[id_col, name_col]].to_dict("records")
        ).reset_index(name=field)
        df = df.merge(agg, left_on="id", right_on="movie_id", how="left").drop(columns=["movie_id"])

    # origin_country is just a list of codes
    if "origin_country" in nested:
        oc = nested["origin_country"].groupby("movie_id")["country_code"].apply(list).reset_index()
        df = df.merge(oc, left_on="id", right_on="movie_id", how="left").drop(columns=["movie_id"])
        df.rename(columns={"country_code": "origin_countries"}, inplace=True)

    return df


def run_queries(db) -> None:
    """Execute required MongoDB queries and aggregations."""
    coll = db.movies

    logger.info("\n=== QUERY 1: Top 10 by revenue ===")
    for doc in coll.find({}, {"title": 1, "revenue": 1, "budget": 1, "_id": 0}).sort("revenue", -1).limit(10):
        logger.info("  %s", doc)

    logger.info("\n=== QUERY 2: Movies per genre (unwind + group) ===")
    pipeline = [
        {"$unwind": "$genres"},
        {"$group": {"_id": "$genres.genre_name", "count": {"$sum": 1}, "avg_revenue": {"$avg": "$revenue"}}},
        {"$sort": {"count": -1}},
        {"$project": {"genre": "$_id", "count": 1, "avg_revenue": 1, "_id": 0}},
    ]
    for doc in coll.aggregate(pipeline):
        logger.info("  %s", doc)

    logger.info("\n=== QUERY 3: Average revenue by decade (match + group + sort) ===")
    pipeline = [
        {"$match": {"decade": {"$ne": None}}},
        {"$group": {"_id": "$decade", "avg_revenue": {"$avg": "$revenue"}, "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
        {"$project": {"decade": "$_id", "avg_revenue": 1, "count": 1, "_id": 0}},
    ]
    for doc in coll.aggregate(pipeline):
        logger.info("  %s", doc)

    logger.info("\n=== QUERY 4: Top 5 companies by movie count (unwind + group + sort + limit) ===")
    pipeline = [
        {"$unwind": "$production_companies"},
        {"$group": {"_id": "$production_companies.company_name", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 5},
        {"$project": {"company": "$_id", "count": 1, "_id": 0}},
    ]
    for doc in coll.aggregate(pipeline):
        logger.info("  %s", doc)

    logger.info("\n=== QUERY 5: High ROI movies (roi > 5) with profit projection ===")
    pipeline = [
        {"$match": {"roi": {"$gt": 5}}},
        {"$project": {"title": 1, "budget": 1, "revenue": 1, "profit": 1, "roi": 1, "_id": 0}},
        {"$sort": {"roi": -1}},
        {"$limit": 10},
    ]
    for doc in coll.aggregate(pipeline):
        logger.info("  %s", doc)


def main() -> None:
    ensure_directories()
    client = get_client()
    db = client[DB_NAME]

    # Load enriched parquet
    logger.info("Loading parquet from %s", DATA_ENRICHED)
    movies = pd.read_parquet(DATA_ENRICHED / "movies_enriched.parquet")
    nested = {
        "genres": pd.read_parquet(DATA_ENRICHED / "genres.parquet"),
        "keywords": pd.read_parquet(DATA_ENRICHED / "keywords.parquet"),
        "production_companies": pd.read_parquet(DATA_ENRICHED / "production_companies.parquet"),
        "production_countries": pd.read_parquet(DATA_ENRICHED / "production_countries.parquet"),
        "spoken_languages": pd.read_parquet(DATA_ENRICHED / "spoken_languages.parquet"),
        "origin_country": pd.read_parquet(DATA_ENRICHED / "origin_country.parquet"),
    }

    # Build embedded movies
    logger.info("Building embedded movie documents...")
    movies_embedded = build_movies_embedded(movies, nested)

    # Load into MongoDB
    logger.info("Loading into MongoDB database: %s", DB_NAME)
    load_collection(db, "movies", movies_embedded)
    for name, df in nested.items():
        load_collection(db, name, df)

    # Run queries
    logger.info("\nRunning MongoDB queries & aggregations...")
    run_queries(db)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
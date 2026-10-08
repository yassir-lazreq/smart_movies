# Movie Intelligence ETL Pipeline

Extract → Transform → Enrich → Load movie data from TMDB into MongoDB.

## Architecture

```
TMDB API → data/raw/ (JSON) → data/processed/ (Parquet Silver) → data/enriched/ (Parquet Gold) → MongoDB
```

## Setup

```bash
# 1. Clone & enter
cd Movie_Intelligence

# 2. Create venv & install deps
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure API key
cp .env.example .env
# Edit .env with your TMDB API key

# 4. Start MongoDB (Docker)
docker run -d --name mongodb -p 27017:27017 mongo:latest
```

## Run Pipeline (Step by Step)

```bash
# 1. Extract: fetch ~2000 movies from TMDB → data/raw/
python -m src.etl.extract

# 2. Transform: clean & normalize → data/processed/*.parquet
python -m src.etl.transform

# 3. Enrich: features, outliers, search_text → data/enriched/*.parquet
python -m src.etl.features

# 4. Load: insert into MongoDB + run queries
python -m src.etl.load_mongo
```

## ML Models

```bash
# High Engagement Classification (Étape 6)
python -m src.models.engagement

# Genre Classification
python -m src.models.genre

# Revenue Tier Classification
python -m src.models.revenue_tier
```

## Output

| Stage | Location | Tables |
|-------|----------|--------|
| Raw | `data/raw/movie_*.json` | 1 file per movie |
| Silver | `data/processed/*.parquet` | movies, genres, keywords, production_companies, production_countries, spoken_languages, origin_country |
| Gold | `data/enriched/*.parquet` | movies_enriched (+9 features), + same nested tables |
| MongoDB | `movie_intelligence` DB | movies (embedded), genres, keywords, ... |

## MongoDB Queries Included

1. Top 10 by revenue
2. Movies per genre (`$unwind` + `$group`)
3. Avg revenue by decade (`$match` + `$group` + `$sort`)
4. Top 5 production companies (`$unwind` + `$group` + `$sort` + `$limit`)
5. High ROI movies (ROI > 5) (`$match` + `$project` + `$sort` + `$limit`)

## Key Features (Gold Layer)

- Outlier capping (IQR) on budget, revenue, runtime, popularity, vote_average, vote_count
- Language code → full name mapping
- Profit, ROI, decade, runtime buckets
- IMDB-style weighted rating
- Budget/revenue tiers (quintiles)
- `search_text` = overview + tagline + genres + keywords (for embeddings/full-text)

## Requirements

- Python 3.10+
- MongoDB (local or Atlas)
- TMDB API key (free at themoviedb.org)

## Project Structure

```
├── config.py              # Central config, paths, env
├── requirements.txt
├── .env.example           # Template for environment variables
├── src/
│   ├── etl/               # Data ingestion and tabular transformations
│   │   ├── tmdb_client.py # HTTP client for TMDB
│   │   ├── extract.py     # API → raw JSON
│   │   ├── transform.py   # JSON → normalized Parquet
│   │   ├── features.py    # Silver → Gold enrichment
│   │   └── load_mongo.py  # Parquet → MongoDB + queries
│   └── models/
│       ├── common.py      # Shared utilities
│       ├── engagement.py  # High engagement classification
│       ├── genre.py       # Genre classification
│       └── revenue_tier.py # Revenue tier classification
└── data/                  # Ignored (generated)
    ├── raw/
    ├── processed/
    └── enriched/
```
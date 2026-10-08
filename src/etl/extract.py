"""
Pipeline d'extraction des données brutes TMDB.

1. Discover : récupère une liste d'IDs via /discover/movie
2. Détails : télécharge chaque film via /movie/{id}
3. Sauvegarde : écrit un JSON par film dans data/raw/

Usage :
    python -m src.etl.extract
"""

import json
import logging
import time

from tqdm import tqdm

from config import DATA_RAW, ensure_directories
from src.etl.tmdb_client import TMDBClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Étape 1 — Discover : récupérer une liste d'IDs
# ---------------------------------------------------------------------------
def discover_movie_ids(client, target=2000, sort_by="popularity.desc", min_votes=50):
    """Récupère des IDs via /discover/movie (paginé)."""
    ids = []
    page = 1
    total_pages = 1

    with tqdm(total=target, desc="Discover IDs") as pbar:
        while len(ids) < target and page <= total_pages:
            data = client.get("discover/movie", {
                "sort_by": sort_by,
                "vote_count.gte": min_votes,
                "page": page,
                "include_adult": False,
            })
            if not data:
                logger.warning("Discover échoué page %d, arrêt.", page)
                break

            # TMDB plafonne à 500 pages
            total_pages = min(data.get("total_pages", 1), 500)
            results = data.get("results", [])
            if not results:
                break

            new_ids = [m["id"] for m in results if m.get("id")]
            ids.extend(new_ids)
            pbar.update(len(new_ids))
            page += 1
            time.sleep(0.05)

    return ids[:target]


# ---------------------------------------------------------------------------
# Étape 2 — Détails d'un film
# ---------------------------------------------------------------------------
def fetch_movie_details(client, movie_id):
    """Récupère un film + ses keywords."""
    return client.get(f"movie/{movie_id}", {"append_to_response": "keywords"})


# ---------------------------------------------------------------------------
# Étape 3 — Sauvegarde
# ---------------------------------------------------------------------------
def save_raw(movie, folder=DATA_RAW):
    """Écrit movie_{id}.json dans data/raw/."""
    path = folder / f"movie_{movie['id']}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(movie, f, ensure_ascii=False, indent=2)


def already_downloaded(movie_id, folder=DATA_RAW):
    """True si movie_{id}.json existe déjà (reprise)."""
    return (folder / f"movie_{movie_id}.json").exists()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_extraction(target=2000, force=False):
    """Pipeline complet : discover → fetch → save."""
    ensure_directories()
    client = TMDBClient()

    # --- Discover ---
    logger.info("Récupération des IDs via /discover/movie...")
    ids = discover_movie_ids(client, target=target)
    logger.info("%d IDs récupérés.", len(ids))

    (DATA_RAW / "_movie_ids.json").write_text(json.dumps(ids), encoding="utf-8")

    # --- Détails ---
    logger.info("Téléchargement des détails...")
    skipped = 0
    saved = 0

    for movie_id in tqdm(ids, desc="Fetch details"):
        if not force and already_downloaded(movie_id):
            skipped += 1
            continue

        movie = fetch_movie_details(client, movie_id)
        if not movie or not movie.get("title"):
            continue

        save_raw(movie)
        saved += 1
        time.sleep(0.05)

    logger.info("Terminé : %d nouveaux, %d déjà présents.", saved, skipped)


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_extraction(target=2000)
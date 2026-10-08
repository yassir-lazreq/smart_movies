import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = os.getenv("TMDB_BASE_URL", "https://api.themoviedb.org/3")
TMDB_LANGUAGE = os.getenv("TMDB_LANGUAGE", "fr-FR")

if not TMDB_API_KEY:
    raise ValueError(
        "TMDB_API_KEY manquante. "
        "Crée un fichier .env à la racine avec TMDB_API_KEY=ta_cle."
    )

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_ENRICHED = DATA_DIR / "enriched"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports" / "figures"


def ensure_directories():
    """Create all required directories idempotently."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DATA_ENRICHED.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
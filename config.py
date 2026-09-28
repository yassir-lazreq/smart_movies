import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Chargement du .env dans os.environ
# ---------------------------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------------------------
# Paramètres TMDB
# ---------------------------------------------------------------------------
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = os.getenv("TMDB_BASE_URL", "https://api.themoviedb.org/3")
TMDB_LANGUAGE = os.getenv("TMDB_LANGUAGE", "fr-FR")

# Fail fast : si la clé API manque, on plante immédiatement avec un
# message clair, plutôt que d'échouer silencieusement 2000 fois plus tard.
if not TMDB_API_KEY:
    raise ValueError(
        "TMDB_API_KEY manquante. "
        "Crée un fichier .env à la racine avec TMDB_API_KEY=ta_cle."
    )


# ---------------------------------------------------------------------------
# Chemins du projet (absolus, robustes au cwd)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"

# Création idempotente des dossiers
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
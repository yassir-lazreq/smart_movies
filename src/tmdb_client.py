"""
Client HTTP pour l'API TMDB.

Seul point d'entrée vers l'API TMDB du projet.
Toute communication réseau passe par TMDBClient.
"""

import logging
import time

import requests

from config import TMDB_API_KEY, TMDB_BASE_URL, TMDB_LANGUAGE

logger = logging.getLogger(__name__)


class TMDBClient:
    """Client léger pour l'API TMDB (bearer token v4)."""

    def __init__(
        self,
        api_key=TMDB_API_KEY,
        base_url=TMDB_BASE_URL,
        language=TMDB_LANGUAGE,
        timeout=10,
    ):
        self.base_url = base_url.rstrip("/")
        self.language = language
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
        })
        # La clé v3 va dans les query params (pas dans les headers)
        self.api_key = api_key

    def get(self, endpoint, params=None):
        """GET sur TMDB. Retourne le JSON, ou None si échec."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        params = {**(params or {}), "language": self.language}
        params["api_key"] = self.api_key

        # --- Requête HTTP ---
        try:
            r = self.session.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as e:
            logger.error("Erreur réseau sur %s : %s", url, e)
            return None

        # --- Rate limit : attendre et réessayer ---
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 2))
            logger.warning("Rate limit atteint, attente %ds...", wait)
            time.sleep(wait)
            return self.get(endpoint, params)

        # --- Autres erreurs HTTP ---
        if r.status_code != 200:
            logger.warning("HTTP %d sur %s", r.status_code, url)
            return None

        # --- Parsing JSON ---
        try:
            return r.json()
        except ValueError:
            logger.error("JSON invalide sur %s", url)
            return None
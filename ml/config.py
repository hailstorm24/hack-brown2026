"""Paths and constants for the M&A relevance model."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FLIGHTS_CSV = REPO_ROOT / "flight-radar" / "flight_radar_flights.csv"
MA_CSV = REPO_ROOT / "filing-data" / "Regeneron M&A Filing Dates Analysis - Regeneron M&A Filing Dates Analysis.csv"
LOCATION_RELEVANCE_CSV = REPO_ROOT / "location-relevance" / "location_relevance.csv"
IATA_ICAO_CSV = REPO_ROOT / "iata-icao.csv"

DATA_DIR = Path(__file__).resolve().parent / "data"
AIRPORT_COORDS_CSV = DATA_DIR / "airport_coordinates.csv"
RELEVANCE_EMBEDDINGS_PATH = DATA_DIR / "relevance_embeddings.pt"
WINDOWS_CACHE_PATH = DATA_DIR / "windows_cache.parquet"

WINDOW_SIZE = 14
DAYS_BEFORE_ANNOUNCE = 14
EMBED_DIM = 384  # all-MiniLM-L6-v2
FLIGHT_DIM = 4 + 1 + EMBED_DIM * 2  # coords + flight_date + from_emb + to_emb

COMPANY_TO_ID = {"regeneron": 0, "abbvie": 1, "eli-lilly": 2}
ID_TO_COMPANY = {v: k for k, v in COMPANY_TO_ID.items()}

# M&A CSV company header -> normalized key for flights
MA_COMPANY_TO_KEY = {
    "Regeneron": "regeneron",
    "AbbVie": "abbvie",
    "Eli Lily ": "eli-lilly",
    "Eli Lily": "eli-lilly",
}

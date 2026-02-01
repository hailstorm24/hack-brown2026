"""Airport code -> (lat, lon) from iata-icao.csv or OurAirports. Caches to CSV."""
import re
from pathlib import Path

import pandas as pd

from ..config import AIRPORT_COORDS_CSV, DATA_DIR, IATA_ICAO_CSV

OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"


def extract_code(location_str: str) -> str | None:
    """Extract 3-letter code from 'City (CODE)' or similar. Returns None if missing/invalid."""
    if not location_str or not isinstance(location_str, str):
        return None
    location_str = location_str.strip()
    if location_str in ("—", "-", ""):
        return None
    m = re.search(r"\(([A-Z0-9]{3})\)\s*$", location_str)
    return m.group(1) if m else None


def get_codes_from_flights(flights_df: pd.DataFrame) -> set[str]:
    """Unique airport codes from flight from/to columns."""
    codes = set()
    for col in ("from", "to"):
        if col not in flights_df.columns:
            continue
        for v in flights_df[col].dropna().unique():
            c = extract_code(v)
            if c:
                codes.add(c)
    return codes


def fetch_ourairports(output_path: Path, codes: set[str] | None = None) -> pd.DataFrame:
    """Download OurAirports airports.csv, filter by IATA/code, save to output_path."""
    try:
        df = pd.read_csv(OURAIRPORTS_URL, dtype=str, keep_default_na=False)
    except Exception:
        out = pd.DataFrame(columns=["airport_code", "lat", "lon"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
        return out
    # Normalize column names (OurAirports: latitude_deg, longitude_deg, iata_code, ident)
    df.columns = df.columns.str.strip().str.lower()
    lat_col = "latitude_deg" if "latitude_deg" in df.columns else "lat"
    lon_col = "longitude_deg" if "longitude_deg" in df.columns else "lon"
    if "iata_code" not in df.columns:
        out = pd.DataFrame(columns=["airport_code", "lat", "lon"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
        return out
    # Filter by codes if provided
    if codes:
        codes_upper = {str(c).strip().upper() for c in codes}
        mask = df["iata_code"].fillna("").astype(str).str.strip().str.upper().isin(codes_upper)
        if "ident" in df.columns and df["ident"].notna().any():
            mask = mask | df["ident"].fillna("").astype(str).str[-3:].str.upper().isin(codes_upper)
        df = df.loc[mask]
    df = df.dropna(subset=["iata_code"])
    df = df[df["iata_code"].astype(str).str.strip() != ""]
    df = df.drop_duplicates(subset=["iata_code"], keep="first")
    df = df.rename(columns={lat_col: "lat", lon_col: "lon", "iata_code": "airport_code"})
    if "airport_code" not in df.columns or "lat" not in df.columns or "lon" not in df.columns:
        out = pd.DataFrame(columns=["airport_code", "lat", "lon"])
    else:
        out = df[["airport_code", "lat", "lon"]].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def load_from_iata_icao(codes: set[str] | None = None) -> pd.DataFrame:
    """Load airport_code, lat, lon from repo iata-icao.csv (columns iata, latitude, longitude)."""
    if not IATA_ICAO_CSV.exists():
        return pd.DataFrame(columns=["airport_code", "lat", "lon"])
    df = pd.read_csv(IATA_ICAO_CSV, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip().str.lower()
    if "iata" not in df.columns or "latitude" not in df.columns or "longitude" not in df.columns:
        return pd.DataFrame(columns=["airport_code", "lat", "lon"])
    df = df[df["iata"].str.strip() != ""].copy()
    if codes:
        codes_upper = {str(c).strip().upper() for c in codes}
        df = df[df["iata"].str.strip().str.upper().isin(codes_upper)]
    df = df.drop_duplicates(subset=["iata"], keep="first")
    df = df.rename(columns={"iata": "airport_code", "latitude": "lat", "longitude": "lon"})
    return df[["airport_code", "lat", "lon"]].copy()


def get_airport_coords(flights_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load or build airport_code -> lat, lon. Prefers iata-icao.csv; else OurAirports. Caches to CSV."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if AIRPORT_COORDS_CSV.exists():
        existing = pd.read_csv(AIRPORT_COORDS_CSV)
        if len(existing) > 0:
            return existing
        AIRPORT_COORDS_CSV.unlink(missing_ok=True)
    codes = get_codes_from_flights(flights_df) if flights_df is not None else None
    if IATA_ICAO_CSV.exists():
        out = load_from_iata_icao(codes)
        if not out.empty:
            out.to_csv(AIRPORT_COORDS_CSV, index=False)
            return out
    if flights_df is not None:
        return fetch_ourairports(AIRPORT_COORDS_CSV, codes)
    return fetch_ourairports(AIRPORT_COORDS_CSV)

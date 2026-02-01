"""Build 14-flight windows and labels from flights + announce dates."""
from datetime import timedelta
from pathlib import Path

import pandas as pd

from ..config import (
    FLIGHTS_CSV,
    WINDOW_SIZE,
    DAYS_BEFORE_ANNOUNCE,
    WINDOWS_CACHE_PATH,
)
from .ma_dates import load_announce_dates

DATE_FMT = "%d %b %Y"  # e.g. 30 Jan 2026


def _parse_flight_date(s: str) -> pd.Timestamp | None:
    """Parse flight date string to date."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    try:
        return pd.to_datetime(s, format=DATE_FMT).date()
    except Exception:
        try:
            return pd.to_datetime(s).date()
        except Exception:
            return None


def load_flights(csv_path: Path | None = None) -> pd.DataFrame:
    """Load flights CSV and parse dates. Normalize company to lowercase."""
    path = csv_path or FLIGHTS_CSV
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["source"] = df["source"].astype(str).str.strip().str.lower()
    df["flight_date"] = df["date"].apply(_parse_flight_date)
    df = df.dropna(subset=["flight_date"])
    df = df.sort_values(["source", "flight_date"])
    return df


def _is_positive(last_date: pd.Timestamp, company: str, announce_dates: list[tuple[str, object]]) -> bool:
    """True if last_date is within DAYS_BEFORE_ANNOUNCE days before an announce date for company."""
    for comp, ann in announce_dates:
        if comp != company:
            continue
        if not hasattr(ann, "year"):
            continue
        delta = (ann - last_date) if hasattr(ann, "__sub__") else (pd.Timestamp(ann).date() - last_date)
        if isinstance(delta, timedelta):
            days = delta.days
        else:
            try:
                days = (pd.Timestamp(ann) - pd.Timestamp(last_date)).days
            except Exception:
                continue
        if 0 <= days <= DAYS_BEFORE_ANNOUNCE:
            return True
    return False


def build_windows(
    flights_df: pd.DataFrame | None = None,
    announce_dates: list[tuple[str, object]] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Build one row per 14-flight window: company, list of 14 flight rows, label.
    Returns a DataFrame with columns: company, window_index, flight_indices, label.
    For downstream use we store indices into the sorted flight table per company.
    """
    if flights_df is None:
        flights_df = load_flights()
    if announce_dates is None:
        announce_dates = load_announce_dates()
    if flights_df.empty:
        return pd.DataFrame(columns=["company", "window_index", "flight_indices", "label", "last_date"])

    DATA_DIR = WINDOWS_CACHE_PATH.parent
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if use_cache and WINDOWS_CACHE_PATH.exists():
        try:
            return pd.read_parquet(WINDOWS_CACHE_PATH)
        except Exception:
            pass

    rows: list[dict] = []
    for company, grp in flights_df.groupby("source"):
        grp = grp.reset_index(drop=True)
        if len(grp) < WINDOW_SIZE:
            continue
        for start in range(0, len(grp) - WINDOW_SIZE + 1):
            end = start + WINDOW_SIZE
            window = grp.iloc[start:end]
            last_date = window["flight_date"].iloc[-1]
            if hasattr(last_date, "date"):
                last_date = last_date.date() if callable(getattr(last_date, "date", None)) else last_date
            label = 1 if _is_positive(last_date, company, announce_dates) else 0
            flight_indices = grp.index[start:end].tolist()
            rows.append({
                "company": company,
                "window_index": start,
                "flight_indices": flight_indices,
                "label": label,
                "last_date": last_date,
            })
    out = pd.DataFrame(rows)
    if use_cache and not out.empty:
        try:
            out.to_parquet(WINDOWS_CACHE_PATH, index=False)
        except Exception:
            pass
    return out


def get_flight_rows_for_window(flights_df: pd.DataFrame, company: str, flight_indices: list[int]) -> pd.DataFrame:
    """Return the 14 flight rows for a window (same order as in window)."""
    return flights_df.loc[flight_indices].reset_index(drop=True)

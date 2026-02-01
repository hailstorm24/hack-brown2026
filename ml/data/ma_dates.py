"""Parse M&A CSV and return announce dates only (ignore target airport, critical window)."""
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config import MA_CSV, MA_COMPANY_TO_KEY

ANNOUNCE_PATTERN = re.compile(r"Announce:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})", re.IGNORECASE)
MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Sept": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def _parse_announce_date(s: str) -> datetime | None:
    """Parse 'July 8, 2024' -> date."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    m = ANNOUNCE_PATTERN.search(s)
    if not m:
        return None
    date_str = m.group(1).strip()
    parts = date_str.replace(",", "").split()
    if len(parts) != 3:
        return None
    month_name, day, year = parts
    month = MONTHS.get(month_name[:3])  # Jan, Feb, ...
    if month is None:
        return None
    try:
        return datetime(int(year), month, int(day)).date()
    except (ValueError, TypeError):
        return None


def load_announce_dates(csv_path: Path | None = None) -> list[tuple[str, datetime]]:
    """
    Return list of (company_key, announce_date).
    company_key is normalized for flights (regeneron, abbvie, eli-lilly).
    Skips rows that are not announcements (e.g. Trial Dates, Ongoing Fixes).
    """
    path = csv_path or MA_CSV
    if not path.exists():
        return []
    raw = pd.read_csv(path, header=None)
    out: list[tuple[str, datetime]] = []
    current_company: str | None = None
    for _, row in raw.iterrows():
        row_str = " ".join(str(x) for x in row.values)
        # Company block header (e.g. "Eli Lily ", "AbbVie ", "Regeneron ")
        for ma_name, key in MA_COMPANY_TO_KEY.items():
            if ma_name in row_str and row_str.strip().startswith(ma_name.strip()):
                current_company = key
                break
        if current_company is None:
            continue
        # Last column is "The News (Announcement)"
        last_col = row.iloc[-1] if len(row) > 0 else ""
        if not isinstance(last_col, str):
            continue
        dt = _parse_announce_date(last_col)
        if dt is not None:
            out.append((current_company, dt))
    return out

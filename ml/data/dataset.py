"""PyTorch Dataset: 14-flight windows -> (X, y, company_id)."""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config import (
    FLIGHT_DIM,
    WINDOW_SIZE,
    COMPANY_TO_ID,
    EMBED_DIM,
)
from .airports import get_airport_coords, extract_code
from .windows import load_flights, build_windows
from .embeddings import load_relevance_embeddings, get_embedding


def _date_to_scalar(d: date, ref_date: date) -> float:
    """Days since ref_date (float)."""
    delta = d - ref_date
    return float(getattr(delta, "days", delta))


class FlightWindowDataset(Dataset):
    """
    Returns (X, y, company_id).
    X: (WINDOW_SIZE, FLIGHT_DIM) float32 — coords (4) + flight_date (1) + from_emb (EMBED_DIM) + to_emb (EMBED_DIM) per flight.
    y: 0 or 1.
    company_id: 0, 1, or 2.
    """

    def __init__(
        self,
        flights_df: pd.DataFrame | None = None,
        windows_df: pd.DataFrame | None = None,
        coords_df: pd.DataFrame | None = None,
        emb_cache: dict | None = None,
        ref_date: date | None = None,
    ):
        self.flights_df = load_flights() if flights_df is None else flights_df
        if windows_df is None:
            windows_df = build_windows(self.flights_df)
        self.windows_df = windows_df.reset_index(drop=True)
        if self.windows_df.empty:
            self.coords_df = pd.DataFrame(columns=["airport_code", "lat", "lon"])
            self.emb_cache = {}
            self.ref_date = ref_date or date(2020, 1, 1)
            return
        if coords_df is None:
            coords_df = get_airport_coords(self.flights_df)
        self.coords_df = coords_df
        if emb_cache is None:
            emb_cache = load_relevance_embeddings()
        self.emb_cache = emb_cache
        if ref_date is None and not self.flights_df.empty:
            ref_date = self.flights_df["flight_date"].min()
            if hasattr(ref_date, "date"):
                ref_date = ref_date.date() if callable(getattr(ref_date, "date", None)) else ref_date
        self.ref_date = ref_date or date(2020, 1, 1)
        self._coords_map = {}
        if not self.coords_df.empty:
            key = "airport_code" if "airport_code" in self.coords_df.columns else "iata_code"
            for _, r in self.coords_df.iterrows():
                code = r.get(key)
                if code:
                    try:
                        lat, lon = float(r["lat"]), float(r["lon"])
                    except (ValueError, TypeError, KeyError):
                        lat, lon = 0.0, 0.0
                    self._coords_map[str(code).strip()] = (lat, lon)

    def __len__(self) -> int:
        return len(self.windows_df)

    def _get_coords(self, code: str | None) -> tuple[float, float]:
        if not code:
            return 0.0, 0.0
        return self._coords_map.get(code.strip(), (0.0, 0.0))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        row = self.windows_df.iloc[idx]
        company = row["company"]
        company_id = COMPANY_TO_ID.get(company, 0)
        label = int(row["label"])
        flight_indices = row["flight_indices"]
        if isinstance(flight_indices, str):
            import ast
            flight_indices = ast.literal_eval(flight_indices)
        flight_indices = list(flight_indices)
        flight_rows = self.flights_df.loc[flight_indices]
        vectors = []
        for _, r in flight_rows.iterrows():
            from_loc, to_loc = r.get("from"), r.get("to")
            from_code = extract_code(from_loc) if from_loc else None
            to_code = extract_code(to_loc) if to_loc else None
            lat_from, lon_from = self._get_coords(from_code)
            lat_to, lon_to = self._get_coords(to_code)
            fd = r.get("flight_date")
            if hasattr(fd, "date"):
                fd = fd.date() if callable(getattr(fd, "date", None)) else fd
            days = _date_to_scalar(fd, self.ref_date) if fd else 0.0
            from_emb = get_embedding(from_loc or "", company, self.emb_cache, EMBED_DIM)
            to_emb = get_embedding(to_loc or "", company, self.emb_cache, EMBED_DIM)
            if isinstance(from_emb, torch.Tensor):
                from_emb = from_emb.numpy()
            if isinstance(to_emb, torch.Tensor):
                to_emb = to_emb.numpy()
            vec = np.concatenate([
                [lat_from, lon_from, lat_to, lon_to],
                [float(days)],
                np.asarray(from_emb, dtype=np.float32),
                np.asarray(to_emb, dtype=np.float32),
            ])
            vectors.append(vec)
        X = np.stack(vectors, axis=0).astype(np.float32)
        return torch.from_numpy(X), label, company_id

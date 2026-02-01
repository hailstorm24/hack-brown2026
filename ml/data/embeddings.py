"""Embed location relevance text with sentence-transformers; cache by (location, company)."""
from pathlib import Path

import pandas as pd
import torch

from ..config import LOCATION_RELEVANCE_CSV, RELEVANCE_EMBEDDINGS_PATH, DATA_DIR, EMBED_DIM


def _load_relevance_df(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or LOCATION_RELEVANCE_CSV
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["location"] = df["location"].astype(str).str.strip()
    df["company"] = df["company"].astype(str).str.strip().str.lower()
    df["relevance"] = df["relevance"].fillna("").astype(str)
    return df


def build_relevance_embeddings(csv_path: Path | None = None, save_path: Path | None = None) -> dict[tuple[str, str], torch.Tensor]:
    """Embed all (location, company) relevance texts; save to save_path. Returns dict (loc, company) -> (E,) tensor."""
    df = _load_relevance_df(csv_path)
    if df.empty:
        return {}
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = df["relevance"].tolist()
    embs = model.encode(texts, convert_to_tensor=True)
    out: dict[tuple[str, str], torch.Tensor] = {}
    for i, row in df.iterrows():
        key = (row["location"], row["company"])
        out[key] = embs[i].float().cpu()
    save_path = save_path or RELEVANCE_EMBEDDINGS_PATH
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(out, save_path)
    return out


def load_relevance_embeddings(
    csv_path: Path | None = None,
    cache_path: Path | None = None,
    rebuild: bool = False,
) -> dict[tuple[str, str], torch.Tensor]:
    """Load cached (location, company) -> embedding. Build and cache if missing or rebuild=True."""
    cache_path = cache_path or RELEVANCE_EMBEDDINGS_PATH
    if not rebuild and cache_path.exists():
        return torch.load(cache_path, weights_only=True)
    return build_relevance_embeddings(csv_path, cache_path)


def get_embedding(
    location: str,
    company: str,
    cache: dict[tuple[str, str], torch.Tensor],
    default_dim: int = EMBED_DIM,
) -> torch.Tensor:
    """Return (E,) tensor for (location, company). Zero vector if missing."""
    key = (location.strip(), company.strip().lower())
    if key in cache:
        return cache[key].clone()
    return torch.zeros(default_dim, dtype=torch.float32)

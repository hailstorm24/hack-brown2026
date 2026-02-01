#!/usr/bin/env python3
"""
Generate a six-word description of why a location is relevant to a company
(why a company plane might visit), using the OpenAI API. Results are cached
by (location, company) so repeated queries reuse the stored value.

Loads API key from .env as OPENAI_API_KEY. See https://platform.openai.com/docs/api-reference.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

# Seconds to wait for one API call before giving up (avoids stalling)
_REQUEST_TIMEOUT = 60

# Load .env from this directory
from dotenv import load_dotenv
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

# Cache file in same directory as this module
_CACHE_PATH = Path(__file__).resolve().parent / "relevance_cache.json"
_PROMPT = (
    "In exactly six words, describe the relevance of {location} to {company} "
    "(why would a company plane visit this location), such as headquarters, "
    "vacation, potential partnerships, or no relevance."
)


def _cache_key(location: str, company: str) -> str:
    """Normalize (location, company) for cache key so variants reuse the same entry."""
    loc = (location or "").strip().lower()
    comp = (company or "").strip().lower()
    return f"{loc}|{comp}"


def _load_cache() -> dict[str, str]:
    if not _CACHE_PATH.exists():
        return {}
    try:
        with open(_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict[str, str]) -> None:
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _call_openai(client, prompt: str, cache: dict, key: str) -> str | None:
    """Call OpenAI and return text, or None on failure. Updates cache on success."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        return None
    cache[key] = text
    _save_cache(cache)
    return text


def get_relevance(location: str, company: str, timeout: int = _REQUEST_TIMEOUT) -> str:
    """
    Return a six-word description of the relevance of location to company.
    Uses OpenAI API; results are cached so the same (location, company) is not re-queried.
    """
    key = _cache_key(location, company)
    cache = _load_cache()
    if key in cache:
        return cache[key]

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY not set (check .env)"

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    prompt = _PROMPT.format(location=location or "", company=company or "")
    for attempt in range(3):
        try:
            if attempt > 0:
                time.sleep(2**attempt)  # 2s, 4s backoff
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_call_openai, client, prompt, cache, key)
                text = future.result(timeout=timeout)
            if text:
                time.sleep(1)  # throttle to reduce rate limits
                return text
        except FuturesTimeoutError:
            pass  # retry on next attempt
        except Exception:
            pass
    return "Error fetching relevance"

#!/usr/bin/env python3
"""
Fetch historical flight data for an aircraft by registration (e.g. N310EL) or ICAO24.
Uses the official OpenSky Network API (https://github.com/openskynetwork/opensky-api).
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from opensky_api import OpenSkyApi

# Rate limit: official API uses 10s (anon) / 5s (auth) for get_states; flights may differ
REQUEST_DELAY_SEC = 1.2
# OpenSky API: "You can only query across 2 partitions (days)" per request; use 1-day chunks to be safe
MAX_DAYS_PER_REQUEST = 1
TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"


def _auth():
    """Optional (username, password) for OpenSky; from env or None."""
    u = (os.environ.get("OPENSKY_USERNAME") or "").strip() or None
    p = (os.environ.get("OPENSKY_PASSWORD") or "").strip() or None
    return (u, p) if (u and p) else None


def _client_credentials():
    """Optional (client_id, client_secret) for OAuth2; from env or None."""
    cid = (os.environ.get("CLIENT_ID") or "").strip() or None
    csec = (os.environ.get("CLIENT_SECRET") or "").strip() or None
    return (cid, csec) if (cid and csec) else None


def get_oauth_token(client_id: str, client_secret: str) -> str | None:
    """Get OpenSky OAuth2 access token via client credentials grant."""
    try:
        r = requests.post(
            TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        return (data.get("access_token") or "").strip() or None
    except requests.RequestException as e:
        print(f"OAuth token error: {e}")
        return None


def registration_to_icao24(registration: str) -> str | None:
    """
    Convert US N-number (e.g. N310EL) to ICAO24 hex (lowercase).
    Uses icao_nnumber_converter_us if installed; otherwise supports numeric N1–N99999 only.
    """
    reg = (registration or "").strip().upper()
    if not reg.startswith("N"):
        # Already might be ICAO24 (6 hex chars)
        if re.match(r"^[0-9a-fA-F]{6}$", registration):
            return registration.lower()
        return None

    # Try optional package for full N-number support (e.g. N310EL)
    # pip install icao_nnumber_converter_us
    try:
        from icao_nnumber_converter_us import n_to_icao
        out = n_to_icao(reg)
        return (out or "").lower() if out else None
    except ImportError:
        pass

    # Fallback: numeric only N1 .. N99999 → ICAO24 a00001 .. a1869f
    digits_only = reg[1:].strip()
    if digits_only.isdigit():
        n = int(digits_only)
        if 1 <= n <= 99999:
            icao_int = 0xA00001 + (n - 1)
            return hex(icao_int)[2:].zfill(6).lower()
    return None


def get_icao24(registration_or_icao: str) -> str | None:
    """Resolve registration (e.g. N310EL) or 6-char hex to ICAO24 lowercase."""
    s = (registration_or_icao or "").strip()
    if re.match(r"^[0-9a-fA-F]{6}$", s):
        return s.lower()
    return registration_to_icao24(s)


def flight_to_dict(flight) -> dict:
    """Convert OpenSky FlightData object to JSON-serializable dict."""
    return flight.__dict__.copy()


def track_to_dict(track) -> dict:
    """Convert OpenSky FlightTrack object to JSON-serializable dict."""
    d = track.__dict__.copy()
    if "path" in d and d["path"]:
        d["path"] = [wp.__dict__ if hasattr(wp, "__dict__") else wp for wp in d["path"]]
    return d


def main():
    parser = argparse.ArgumentParser(
        description="Fetch historical flight data for an aircraft by registration (e.g. N310EL) or ICAO24."
    )
    parser.add_argument(
        "registration_or_icao",
        help="Aircraft registration (e.g. N310EL) or 6-char ICAO24 hex (e.g. a0c3f2)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365 * 2,
        help="How many days back to query (default: 730 = ~2 years). API allows max 2 partitions (days) per request; we chunk into 1-day windows.",
    )
    parser.add_argument(
        "--end",
        metavar="DATE",
        default=None,
        help="End of query range as YYYY-MM-DD (default: today UTC). Use if API returns 400 for future end dates.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON file (default: <registration_or_icao>_flights.json)",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Do not fetch track/trajectory for each flight (faster, fewer API calls)",
    )
    parser.add_argument(
        "--icao24",
        action="store_true",
        help="Treat input as ICAO24 hex (skip N-number conversion)",
    )
    parser.add_argument("--username", help="OpenSky username (or set OPENSKY_USERNAME)")
    parser.add_argument("--password", help="OpenSky password (or set OPENSKY_PASSWORD)")
    parser.add_argument("--client-id", help="OpenSky OAuth2 client ID (or set CLIENT_ID)")
    parser.add_argument("--client-secret", help="OpenSky OAuth2 client secret (or set CLIENT_SECRET)")
    args = parser.parse_args()

    # Auth precedence: client credentials (token) > username/password > anonymous
    username, password = None, None
    token = None
    if (args.client_id or os.environ.get("CLIENT_ID")) and (args.client_secret or os.environ.get("CLIENT_SECRET")):
        cid = (args.client_id or os.environ.get("CLIENT_ID", "")).strip()
        csec = (args.client_secret or os.environ.get("CLIENT_SECRET", "")).strip()
        if cid and csec:
            token = get_oauth_token(cid, csec)
            if not token:
                return 1
            print("Using OAuth2 client credentials.")
    elif args.username and args.password:
        username = args.username.strip()
        password = args.password.strip()
    else:
        auth = _auth()
        if auth:
            username, password = auth
        else:
            creds = _client_credentials()
            if creds:
                cid, csec = creds
                token = get_oauth_token(cid, csec)
                if token:
                    print("Using OAuth2 client credentials (from env).")

    if args.icao24:
        icao24 = args.registration_or_icao.strip().lower()
        if not re.match(r"^[0-9a-f]{6}$", icao24):
            print("Error: --icao24 requires a 6-character hex string (e.g. a0c3f2)")
            return 1
    else:
        icao24 = get_icao24(args.registration_or_icao)
        if not icao24:
            print(
                "Error: Could not resolve registration to ICAO24. "
                "For N-numbers like N310EL, install: pip install icao_nnumber_converter_us"
            )
            print("Or pass a 6-char ICAO24 hex with --icao24.")
            return 1

    api = OpenSkyApi(username=username, password=password, token=token)

    print(f"Using ICAO24: {icao24}")
    if args.end:
        try:
            end_dt = datetime.strptime(args.end.strip(), "%Y-%m-%d").replace(tzinfo=None)
        except ValueError:
            print("Error: --end must be YYYY-MM-DD (e.g. 2025-01-31)")
            return 1
    else:
        end_dt = datetime.utcnow()
    begin_dt = end_dt - timedelta(days=args.days)
    begin_ts = int(begin_dt.timestamp())
    end_ts = int(end_dt.timestamp())
    print(f"Querying flights from {begin_dt.date()} to {end_dt.date()} ({args.days} days) ...")

    out_path = Path(args.output or f"{args.registration_or_icao.replace(' ', '_')}_flights.json")
    flights_data = []
    out = {
        "icao24": icao24,
        "registration_or_icao": args.registration_or_icao,
        "query": {"begin_ts": begin_ts, "end_ts": end_ts, "days": args.days},
        "flights": flights_data,
    }

    def write_json():
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

    # Chunk requests; API allows max 2 partitions (days) per request
    chunk_begin = begin_ts
    while chunk_begin < end_ts:
        chunk_end = min(chunk_begin + MAX_DAYS_PER_REQUEST * 86400, end_ts)
        try:
            chunk = api.get_flights_by_aircraft(icao24, chunk_begin, chunk_end)
        except ValueError as e:
            print(f"OpenSky API error: {e}")
            return 1
        if chunk is None:
            code = getattr(api, "_last_status_code", None)
            body = getattr(api, "_last_response_body", None)
            msg = "OpenSky API returned no data."
            if code is not None:
                msg += f" (HTTP {code})"
            if body:
                msg += f"\nResponse: {body[:500]}"
            msg += "\nTry CLIENT_ID/CLIENT_SECRET or OPENSKY_USERNAME/OPENSKY_PASSWORD (see README)."
            print(msg)
            return 1
        for f in chunk:
            flights_data.append(flight_to_dict(f))
        write_json()
        chunk_begin = chunk_end
        if chunk_begin < end_ts:
            time.sleep(REQUEST_DELAY_SEC)

    if not flights_data:
        print("No flights found for this aircraft in the given period.")
        print(f"Wrote {out_path}")
        return 0

    print(f"Found {len(flights_data)} flight(s).")

    # Optionally attach track (trajectory) for each flight using firstSeen
    if not args.no_track:
        for fl in flights_data:
            first = fl.get("firstSeen")
            if first is not None:
                try:
                    track = api.get_track_by_aircraft(icao24, first)
                    if track is not None:
                        fl["track"] = track_to_dict(track)
                except ValueError:
                    pass
                time.sleep(REQUEST_DELAY_SEC)
        write_json()

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

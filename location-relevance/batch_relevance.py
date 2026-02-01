#!/usr/bin/env python3
"""
Batch script: read flight_radar_flights.csv, collect unique (location, company) pairs,
call get_relevance for each (cache is used automatically), and write location_relevance.csv.
"""

import csv
from pathlib import Path

from relevance import get_relevance

# Flight CSV is in sibling directory flight-radar
FLIGHT_CSV = Path(__file__).resolve().parent.parent / "flight-radar" / "flight_radar_flights.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "location_relevance.csv"


def main() -> None:
    if not FLIGHT_CSV.exists():
        print(f"Flight CSV not found: {FLIGHT_CSV}")
        return

    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []

    with open(FLIGHT_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = (row.get("source") or "").strip()
            from_loc = (row.get("from") or "").strip()
            to_loc = (row.get("to") or "").strip()
            if from_loc and source and (from_loc, source) not in seen:
                seen.add((from_loc, source))
                pairs.append((from_loc, source))
            if to_loc and source and (to_loc, source) not in seen:
                seen.add((to_loc, source))
                pairs.append((to_loc, source))

    total = len(pairs)
    print(f"Processing {total} location–company pairs…")
    rows = []
    errors = 0
    for i, (location, company) in enumerate(pairs, 1):
        relevance = get_relevance(location, company)
        if "Error" in relevance:
            errors += 1
        rows.append({"location": location, "company": company, "relevance": relevance})
        # Progress every 10 or on last
        if i % 10 == 0 or i == total:
            print(f"  {i}/{total} – {errors} errors so far")

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["location", "company", "relevance"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Wrote {len(rows)} rows to {OUTPUT_CSV} ({errors} errors)")


if __name__ == "__main__":
    main()

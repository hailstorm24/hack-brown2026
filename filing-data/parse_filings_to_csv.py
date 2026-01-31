#!/usr/bin/env python3
"""
Parse pharma_biotech_filings.json to CSV for Regeneron, Eli Lilly, and AbbVie.
"""

import csv
import json
from pathlib import Path


COMPANIES = {"Regeneron", "Eli Lilly", "AbbVie"}

# Same format as flight-radar CSV: lowercase, hyphen for multi-word
COMPANY_TO_SOURCE = {
    "Regeneron": "regeneron",
    "Eli Lilly": "eli-lilly",
    "AbbVie": "abbvie",
}


def main():
    script_dir = Path(__file__).resolve().parent
    input_path = script_dir / "pharma_biotech_filings.json"
    output_path = script_dir / "pharma_biotech_filings_three_companies.csv"

    with open(input_path, encoding="utf-8") as f:
        filings = json.load(f)

    filtered = [f for f in filings if f.get("company") in COMPANIES]
    # Normalize company name to match flight-radar format
    for f in filtered:
        f["company"] = COMPANY_TO_SOURCE[f["company"]]

    fieldnames = ["company", "form", "filing_date", "accession_number", "description", "url"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(filtered)

    print(f"Wrote {len(filtered)} filings to {output_path}")
    for source in sorted(COMPANY_TO_SOURCE.values()):
        count = sum(1 for f in filtered if f["company"] == source)
        print(f"  {source}: {count}")


if __name__ == "__main__":
    main()

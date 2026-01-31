#!/usr/bin/env python3
"""
Fetch 8-K filings from major pharma/biotech companies using SEC EDGAR API.
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

# SEC requires a User-Agent header with contact info
HEADERS = {
    "User-Agent": "HackBrown2026 Research contact@example.com",
    "Accept": "application/json"
}

# Major pharma/biotech companies with their CIK numbers
PHARMA_BIOTECH_COMPANIES = {
    # Big Pharma
    "Pfizer": "78003",
    "Johnson & Johnson": "200406",
    "Merck": "310158",
    "AbbVie": "1551152",
    "Bristol-Myers Squibb": "14272",
    "Eli Lilly": "59478",
    "Amgen": "318154",
    "Gilead Sciences": "882095",
    "Regeneron": "872589",
    "Vertex Pharmaceuticals": "875320",
    # Large Biotech
    "Moderna": "1682852",
    "BioNTech": "1776985",
    "Biogen": "875045",
    "Illumina": "1110803",
    "Alexion/AstraZeneca": "899866",
    "Seagen": "1157601",
    "BioMarin": "1048477",
    "Incyte": "879169",
    "Exact Sciences": "1124140",
    "Alnylam": "1178670",
    # Mid-cap Biotech
    "Neurocrine Biosciences": "914475",
    "Sarepta Therapeutics": "873303",
    "Ionis Pharmaceuticals": "874015",
    "Blueprint Medicines": "1597264",
    "Argenx": "1697862",
    "Revolution Medicines": "1626280",
    "Karuna Therapeutics": "1727196",
    "Natera": "1604821",
    "10x Genomics": "1770787",
    "CRISPR Therapeutics": "1674101",
}

def get_company_filings(cik: str, company_name: str) -> Dict[str, Any]:
    """Fetch filings for a company from SEC EDGAR API."""
    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.RequestException as e:
        print(f"Error fetching {company_name}: {e}")
        return None

def extract_8k_filings(data: Dict, company_name: str, years_back: int = 5) -> List[Dict]:
    """Extract 8-K filings from the past N years."""
    if not data:
        return []

    cutoff_date = datetime.now() - timedelta(days=years_back * 365)
    filings = []

    recent_filings = data.get("filings", {}).get("recent", {})

    forms = recent_filings.get("form", [])
    filing_dates = recent_filings.get("filingDate", [])
    accession_numbers = recent_filings.get("accessionNumber", [])
    primary_docs = recent_filings.get("primaryDocument", [])
    descriptions = recent_filings.get("primaryDocDescription", [])

    for i, form in enumerate(forms):
        if form in ["8-K", "8-K/A"]:
            filing_date_str = filing_dates[i] if i < len(filing_dates) else ""
            try:
                filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
                if filing_date >= cutoff_date:
                    cik = str(data.get("cik", "")).zfill(10)
                    accession = accession_numbers[i].replace("-", "") if i < len(accession_numbers) else ""
                    primary_doc = primary_docs[i] if i < len(primary_docs) else ""

                    filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"

                    filings.append({
                        "company": company_name,
                        "form": form,
                        "filing_date": filing_date_str,
                        "accession_number": accession_numbers[i] if i < len(accession_numbers) else "",
                        "description": descriptions[i] if i < len(descriptions) else "",
                        "url": filing_url,
                    })
            except ValueError:
                continue

    return filings

def main():
    all_filings = []

    print("Fetching 8-K filings from major pharma/biotech companies...")
    print("=" * 70)

    for company_name, cik in PHARMA_BIOTECH_COMPANIES.items():
        print(f"\nFetching: {company_name} (CIK: {cik})")

        data = get_company_filings(cik, company_name)
        if data:
            filings = extract_8k_filings(data, company_name, years_back=5)
            all_filings.extend(filings)
            print(f"  Found {len(filings)} 8-K filings in the past 5 years")

        # Be polite to SEC servers - rate limit
        time.sleep(0.2)

    # Sort by date descending
    all_filings.sort(key=lambda x: x["filing_date"], reverse=True)

    # Save to JSON
    output_file = "pharma_biotech_filings.json"
    with open(output_file, "w") as f:
        json.dump(all_filings, f, indent=2)
    print(f"\n\nSaved {len(all_filings)} filings to {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY OF PHARMA/BIOTECH FILINGS (Past 5 Years)")
    print("=" * 70)

    # Group by company
    by_company = {}
    for filing in all_filings:
        company = filing["company"]
        if company not in by_company:
            by_company[company] = {"8-K": 0, "8-K/A": 0}
        form = filing["form"]
        by_company[company][form] = by_company[company].get(form, 0) + 1

    print(f"\n{'Company':<30} {'8-K':<8} {'8-K/A':<8}")
    print("-" * 70)
    for company, counts in sorted(by_company.items()):
        print(f"{company:<30} {counts.get('8-K', 0):<8} {counts.get('8-K/A', 0):<8}")

    print(f"\nTotal filings: {len(all_filings)}")

    # Print recent filings
    print("\n" + "=" * 70)
    print("MOST RECENT 20 FILINGS")
    print("=" * 70)
    for filing in all_filings[:20]:
        print(f"\n{filing['filing_date']} | {filing['form']:<6} | {filing['company']}")
        print(f"  URL: {filing['url']}")

if __name__ == "__main__":
    main()

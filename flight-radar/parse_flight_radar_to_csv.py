#!/usr/bin/env python3
"""
Parse flight-radar text exports (regeneron.txt, eli-lilly.txt, AbbVie.txt) into a single CSV.
"""

import csv
import re
from pathlib import Path


def parse_regeneron(filepath: Path) -> list[dict]:
    """Parse regeneron.txt - tab-separated table with status on following line."""
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    # First line: "Flight history for aircraft - N459FX"
    aircraft = "N459FX"
    for line in lines:
        if line.startswith("Flight history for aircraft"):
            m = re.search(r"aircraft\s*-\s*(\S+)", line)
            if m:
                aircraft = m.group(1)
            break

    rows = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Data rows start with space+tab and contain date (e.g. "30 Jan 2026")
        if ("\t" in line and re.search(r"\d{1,2}\s+[A-Za-z]{3}\s+\d{4}\s*\t", line)
                and "DATE" not in line and "FROM" not in line):
            parts = line.strip().split("\t")
            # parts: [date, from, to, flight, flight_time, std, atd, sta, ...]
            if len(parts) >= 8:
                date = parts[0].strip()
                from_apt = parts[1].strip()
                to_apt = parts[2].strip()
                flight = parts[3].strip() if len(parts) > 3 else ""
                flight_time = parts[4].strip() if len(parts) > 4 else ""
                std = parts[5].strip() if len(parts) > 5 else ""
                atd = parts[6].strip() if len(parts) > 6 else ""
                sta = parts[7].strip() if len(parts) > 7 else ""
                status = ""
                if i + 1 < len(lines):
                    status = lines[i + 1].strip().split("\t")[0].strip()
                    # Skip past status line for next iteration
                    i += 1
                rows.append({
                    "source": "regeneron",
                    "aircraft": aircraft,
                    "date": date,
                    "from": from_apt,
                    "to": to_apt,
                    "flight": flight,
                    "flight_time": flight_time,
                    "std": std,
                    "atd": atd,
                    "sta": sta,
                    "status": status,
                })
        i += 1

    return rows


def parse_block_format(filepath: Path, source: str = "eli-lilly", aircraft: str = "") -> list[dict]:
    """Parse block-format files (eli-lilly, AbbVie) - em dash separator and label/value pairs."""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # Separator is space + Unicode em dash (U+2014)
    blocks = content.split(" \u2014")
    rows = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        lines = [ln.strip() for ln in block.split("\n")]
        # Skip leading empty lines
        while lines and not lines[0]:
            lines.pop(0)
        if not lines:
            continue

        # First line is date, second is flight time, then status (may run until STD)
        i = 0
        date = ""
        flight_time = ""
        status_parts = []
        std = atd = sta = from_apt = to_apt = ""

        if i < len(lines):
            date = lines[i]
            i += 1
        if i < len(lines):
            flight_time = lines[i]
            i += 1

        # Status can be multiple lines until we hit STD or FROM (some blocks lack STD/ATD/STA)
        while i < len(lines):
            label = lines[i].upper()
            if label == "STD":
                i += 1
                if i < len(lines):
                    std = lines[i]
                    i += 1
                # Parse ATD, STA, FROM, TO (label on one line, value on next)
                while i < len(lines):
                    lab = lines[i].upper()
                    i += 1
                    val = lines[i] if i < len(lines) else ""
                    if lab == "ATD":
                        atd = val
                        i += 1
                    elif lab == "STA":
                        sta = val
                        i += 1
                    elif lab == "FROM":
                        from_apt = val
                        i += 1
                    elif lab == "TO":
                        to_apt = val
                        i += 1
                    else:
                        i += 1
                break
            if label == "FROM":
                # Block has no STD/ATD/STA; parse FROM and TO only (skip blank lines for values)
                i += 1
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i < len(lines):
                    from_apt = lines[i]
                    i += 1
                while i < len(lines) and lines[i].upper() != "TO":
                    i += 1
                if i < len(lines) and lines[i].upper() == "TO":
                    i += 1
                    while i < len(lines) and not lines[i].strip():
                        i += 1
                    if i < len(lines):
                        to_apt = lines[i]
                        i += 1
                break
            status_parts.append(lines[i])
            i += 1

        status = " ".join(status_parts).strip()
        # Clean HTML/entity fragments in status
        status = re.sub(r'\s*"[^"]*">\s*', " ", status).strip()

        # Only add if we got at least date
        if date and re.match(r"\d{1,2}\s+[A-Za-z]{3}\s+\d{4}", date):
            rows.append({
                "source": source,
                "aircraft": aircraft,
                "date": date,
                "from": from_apt,
                "to": to_apt,
                "flight": "",
                "flight_time": flight_time,
                "std": std,
                "atd": atd,
                "sta": sta,
                "status": status,
            })

    return rows


def main():
    script_dir = Path(__file__).resolve().parent
    # Input files may be in script dir or in txt-files/ subdir
    regeneron_path = script_dir / "regeneron.txt"
    if not regeneron_path.exists():
        regeneron_path = script_dir / "txt-files" / "regeneron.txt"
    eli_lilly_path = script_dir / "eli-lilly.txt"
    if not eli_lilly_path.exists():
        eli_lilly_path = script_dir / "txt-files" / "eli-lilly.txt"
    abbvie_path = script_dir / "AbbVie.txt"
    if not abbvie_path.exists():
        abbvie_path = script_dir / "txt-files" / "AbbVie.txt"
    output_path = script_dir / "flight_radar_flights.csv"

    all_rows = []

    if regeneron_path.exists():
        reg_rows = parse_regeneron(regeneron_path)
        all_rows.extend(reg_rows)
        print(f"Parsed {len(reg_rows)} flights from regeneron.txt")
    else:
        print(f"Not found: {regeneron_path}")

    if eli_lilly_path.exists():
        eli_rows = parse_block_format(eli_lilly_path, source="eli-lilly", aircraft="N310EL")
        all_rows.extend(eli_rows)
        print(f"Parsed {len(eli_rows)} flights from eli-lilly.txt")
    else:
        print(f"Not found: {eli_lilly_path}")

    if abbvie_path.exists():
        abbvie_rows = parse_block_format(abbvie_path, source="abbvie", aircraft="N60AV")
        all_rows.extend(abbvie_rows)
        print(f"Parsed {len(abbvie_rows)} flights from AbbVie.txt")
    else:
        print(f"Not found: {abbvie_path}")

    fieldnames = ["source", "aircraft", "date", "from", "to", "flight", "flight_time", "std", "atd", "sta", "status"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()

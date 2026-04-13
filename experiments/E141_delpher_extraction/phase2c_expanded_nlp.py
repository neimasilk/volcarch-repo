#!/usr/bin/env python3
"""
E141 Phase 2c: NLP on expanded Delpher records (high-relevance subset)
Reuses Phase 2 NLP pipeline on the 117 new high-relevance records.
"""

import requests
import xml.etree.ElementTree as ET
import json
import csv
import re
import sys
import time
import numpy as np
from pathlib import Path
from collections import Counter

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path(__file__).parent / "results"
EXPANDED_CSV = RESULTS_DIR / "delpher_expanded.csv"

DELAY = 1.5

# Location patterns (colonial Dutch)
LOCATIONS = {
    "Soerabaja": (-7.25, 112.75), "Malang": (-7.98, 112.63),
    "Modjokerto": (-7.47, 112.43), "Singosari": (-7.89, 112.66),
    "Trowoelan": (-7.55, 112.38), "Kediri": (-7.82, 112.01),
    "Blitar": (-8.08, 112.17), "Pasoeroean": (-7.64, 112.90),
    "Madioen": (-7.63, 111.52), "Gresik": (-7.16, 112.65),
    "Toeban": (-6.90, 112.05), "Probolinggo": (-7.75, 113.22),
    "Djember": (-8.17, 113.70), "Banjoewangi": (-8.22, 114.35),
    "Djokja": (-7.80, 110.36), "Solo": (-7.57, 110.82),
    "Semarang": (-6.97, 110.42), "Magelang": (-7.47, 110.22),
    "Prambanan": (-7.75, 110.49), "Boroboedoer": (-7.61, 110.20),
    "Diëng": (-7.21, 109.91), "Batavia": (-6.17, 106.83),
    "Bandoeng": (-6.91, 107.61), "Buitenzorg": (-6.60, 106.80),
    "Wonosari": (-7.96, 110.60), "Penataran": (-7.93, 112.21),
    "Panataran": (-7.93, 112.21), "Pakis": (-8.03, 112.62),
    "Bondowoso": (-7.91, 113.82), "Ponorogo": (-7.87, 111.46),
    "Ngandjoek": (-7.60, 111.90), "Nganjuk": (-7.60, 111.90),
    "Toeloengagoeng": (-8.07, 111.90), "Tulungagung": (-8.07, 111.90),
    "Kloet": (-7.93, 112.31), "Kelud": (-7.93, 112.31),
    "Smeroe": (-8.11, 112.92), "Semeru": (-8.11, 112.92),
    "Merapi": (-7.54, 110.45), "Bromo": (-7.94, 112.95),
}

DEPTH_PATTERNS = [
    re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:meter|M\.?|m\.?)\s*(?:diep|diepte|onder)', re.I),
    re.compile(r'diepte\s+van\s+(\d+(?:[.,]\d+)?)\s*(?:meter|M\.?)', re.I),
    re.compile(r'op\s+een?\s+diepte\s+van\s+(\d+(?:[.,]\d+)?)', re.I),
    re.compile(r'(\d+(?:[.,]\d+)?)\s*voet\s*(?:diep|diepte)', re.I),
    re.compile(r'(?:begraven|bedolven|bedekt)\s+(?:onder|door)\s+(\d+(?:[.,]\d+)?)\s*(?:meter|M)', re.I),
]


def fetch_fulltext(url):
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        try:
            root = ET.fromstring(resp.text)
            texts = []
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    texts.append(elem.text.strip())
            return " ".join(texts)
        except ET.ParseError:
            return resp.text
    except Exception:
        return None


def extract_depths(text):
    depths = []
    for pat in DEPTH_PATTERNS:
        for m in pat.finditer(text):
            try:
                val = float(m.group(1).replace(",", "."))
                ctx = text[max(0, m.start()-30):m.end()+30]
                if "voet" in ctx.lower():
                    val *= 0.3
                if val <= 50:  # filter out geological/oil
                    depths.append({"value_m": round(val, 1), "context": ctx.strip()})
            except (ValueError, IndexError):
                continue
    return depths


def extract_locations(text):
    found = []
    text_lower = text.lower()
    for name, (lat, lon) in LOCATIONS.items():
        if name.lower() in text_lower:
            found.append({"name": name, "lat": lat, "lon": lon})
    return found


def main():
    print("=" * 70)
    print("E141 Phase 2c: NLP on Expanded Delpher Records")
    print("=" * 70)

    records = []
    with open(EXPANDED_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append(row)

    # Process high-relevance (>=3) — same threshold as Phase 2
    high_rel = [r for r in records if int(r.get("relevance", 0)) >= 3]
    print(f"Expanded records: {len(records)}")
    print(f"High relevance (>=3): {len(high_rel)}")

    print(f"\nFetching full text + NLP...")
    results = []
    n_depth = 0
    n_loc = 0

    for i, record in enumerate(high_rel):
        url = record.get("identifier", "")
        if not url or not url.startswith("http"):
            continue

        text = fetch_fulltext(url)
        if not text:
            continue

        depths = extract_depths(text)
        locations = extract_locations(text)

        result = {
            "title": record["title"],
            "date": record["date"],
            "identifier": url,
            "relevance": record.get("relevance", 0),
            "tags": record.get("tags", ""),
            "text_length": len(text),
            "n_depths": len(depths),
            "depths_m": "; ".join(f"{d['value_m']}m" for d in depths) if depths else "",
            "depth_contexts": " | ".join(d['context'] for d in depths[:3]) if depths else "",
            "locations": "; ".join(f"{l['name']}" for l in locations),
            "primary_lat": locations[0]["lat"] if locations else None,
            "primary_lon": locations[0]["lon"] if locations else None,
        }
        results.append(result)

        if depths:
            n_depth += 1
        if locations:
            n_loc += 1

        if (i + 1) % 20 == 0 or depths:
            d_str = f" DEPTH: {depths[0]['value_m']}m" if depths else ""
            l_str = f" @ {locations[0]['name']}" if locations else ""
            print(f"  [{i+1}/{len(high_rel)}] {record['title'][:45]}{d_str}{l_str}")

        time.sleep(DELAY)

    # Results
    print(f"\n{'='*70}")
    print(f"EXPANDED NLP RESULTS")
    print(f"{'='*70}")
    print(f"  Fetched: {len(results)}/{len(high_rel)}")
    print(f"  With depth: {n_depth}")
    print(f"  With location: {n_loc}")

    # Archaeological depths
    all_depths = []
    for r in results:
        if r["depths_m"]:
            for d in r["depths_m"].split("; "):
                try:
                    all_depths.append(float(d.replace("m", "")))
                except ValueError:
                    pass

    if all_depths:
        print(f"\n  DEPTH VALUES ({len(all_depths)} total, filtered ≤50m):")
        print(f"    Range: {min(all_depths):.1f}m — {max(all_depths):.1f}m")
        print(f"    Median: {np.median(all_depths):.1f}m")

        for r in results:
            if r["depths_m"]:
                loc = r["locations"].split(";")[0] if r["locations"] else "?"
                print(f"    [{r['date']}] {r['depths_m']:>8s} @ {loc:15s} | {r['title'][:50]}")

    # Location frequency
    all_locs = []
    for r in results:
        if r["locations"]:
            all_locs.extend(r["locations"].split("; "))
    loc_counts = Counter(all_locs).most_common(15)
    print(f"\n  LOCATION FREQUENCIES (expanded):")
    for loc, count in loc_counts:
        print(f"    {loc:20s}: {count}")

    # Save
    if results:
        csv_path = RESULTS_DIR / "delpher_expanded_nlp.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Saved: {csv_path.name} ({len(results)} records)")

    summary = {
        "phase": "2c_expanded_nlp",
        "date": "2026-04-13",
        "records_processed": len(results),
        "with_depth": n_depth,
        "with_location": n_loc,
        "depths": all_depths,
        "location_counts": dict(loc_counts) if loc_counts else {},
    }
    with open(RESULTS_DIR / "delpher_expanded_nlp_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{'='*70}")


if __name__ == "__main__":
    main()

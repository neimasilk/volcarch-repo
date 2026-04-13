#!/usr/bin/env python3
"""
E141 Phase 2: Full-Text NLP Extraction from Delpher Colonial Newspapers
========================================================================
Phase 1 got 529 records with metadata. Phase 2:
1. Fetch full OCR text from KB resolver API
2. Apply NLP patterns for depth, location, material, volcanic context
3. Geocode findspots
4. Cross-reference with VOLCARCH predictions

This breaks DHARMA monoculture (ME#13 Risk 3).
"""

import requests
import xml.etree.ElementTree as ET
import json
import csv
import re
import sys
import time
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path(__file__).parent / "results"
PHASE1_CSV = RESULTS_DIR / "delpher_extraction.csv"

DELAY = 1.5  # seconds between API calls (be nice to KB servers)

# ── NLP patterns (adapted from E091 OV pipeline) ─────────────────────

# Depth patterns (Dutch)
DEPTH_PATTERNS = [
    # "X meter diep" / "op X meter diepte" / "X M. diep"
    re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:meter|M\.?|m\.?)\s*(?:diep|diepte|onder)', re.I),
    # "diepte van X meter"
    re.compile(r'diepte\s+van\s+(\d+(?:[.,]\d+)?)\s*(?:meter|M\.?)', re.I),
    # "op een diepte van X"
    re.compile(r'op\s+een?\s+diepte\s+van\s+(\d+(?:[.,]\d+)?)', re.I),
    # "X voet diep" (Dutch feet ≈ 0.3m)
    re.compile(r'(\d+(?:[.,]\d+)?)\s*voet\s*(?:diep|diepte)', re.I),
    # "begraven onder X meter"
    re.compile(r'(?:begraven|bedolven|bedekt)\s+(?:onder|door)\s+(\d+(?:[.,]\d+)?)\s*(?:meter|M\.?)', re.I),
]

# Location patterns (colonial Dutch spellings)
LOCATION_PATTERNS = {
    # East Java
    "Soerabaja": (-7.25, 112.75), "Surabaya": (-7.25, 112.75),
    "Malang": (-7.98, 112.63),
    "Modjokerto": (-7.47, 112.43), "Mojokerto": (-7.47, 112.43),
    "Singosari": (-7.89, 112.66), "Singhasari": (-7.89, 112.66),
    "Trowoelan": (-7.55, 112.38), "Trowulan": (-7.55, 112.38),
    "Kediri": (-7.82, 112.01),
    "Blitar": (-8.08, 112.17),
    "Pasoeroean": (-7.64, 112.90), "Pasuruan": (-7.64, 112.90),
    "Madioen": (-7.63, 111.52), "Madiun": (-7.63, 111.52),
    "Gresik": (-7.16, 112.65),
    "Toeban": (-6.90, 112.05), "Tuban": (-6.90, 112.05),
    "Probolinggo": (-7.75, 113.22),
    "Djember": (-8.17, 113.70), "Jember": (-8.17, 113.70),
    "Banjoewangi": (-8.22, 114.35), "Banyuwangi": (-8.22, 114.35),
    # Central Java
    "Djokja": (-7.80, 110.36), "Jogjakarta": (-7.80, 110.36), "Yogyakarta": (-7.80, 110.36),
    "Solo": (-7.57, 110.82), "Soerakarta": (-7.57, 110.82),
    "Semarang": (-6.97, 110.42),
    "Magelang": (-7.47, 110.22),
    "Prambanan": (-7.75, 110.49),
    "Boroboedoer": (-7.61, 110.20), "Borobudur": (-7.61, 110.20),
    "Diëng": (-7.21, 109.91), "Dieng": (-7.21, 109.91),
    # West Java
    "Batavia": (-6.17, 106.83), "Djakarta": (-6.17, 106.83),
    "Bandoeng": (-6.91, 107.61), "Bandung": (-6.91, 107.61),
    "Buitenzorg": (-6.60, 106.80), "Bogor": (-6.60, 106.80),
    # Regions
    "Goenoeng Kidoel": (-7.98, 110.61), "Gunung Kidul": (-7.98, 110.61),
    "Wonosari": (-7.96, 110.60),
}

# Material patterns
MATERIAL_KEYWORDS = {
    "statue": ["beeld", "beelden", "standbeeld", "figuur"],
    "temple": ["tempel", "tjandi", "candi", "heiligdom"],
    "stone": ["steen", "steenen", "baksteen", "andesiet"],
    "metal": ["goud", "zilver", "brons", "koper", "ijzer"],
    "pottery": ["potscherven", "aardewerk", "keramiek", "kruik"],
    "bone": ["been", "beenderen", "skelet", "schedel"],
    "inscription": ["inscriptie", "opschrift", "prasasti"],
    "tools": ["bijl", "mes", "messen", "speer", "werktuig"],
}

# Volcanic context
VOLCANIC_KEYWORDS = ["vulkaan", "vulkanisch", "lava", "lahar", "asch", "uitbarsting",
                     "modder", "modderstroom", "kratermeer", "Kloet", "Keloed",
                     "Smeroe", "Semeru", "Merapi", "Bromo"]


def fetch_fulltext(url):
    """Fetch OCR full text from KB resolver."""
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None

        # Parse XML
        try:
            root = ET.fromstring(resp.text)
            # Extract all text content
            texts = []
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    texts.append(elem.text.strip())
            return " ".join(texts)
        except ET.ParseError:
            # Maybe it's plain text
            return resp.text
    except Exception:
        return None


def extract_depths(text):
    """Extract depth mentions from text."""
    depths = []
    for pattern in DEPTH_PATTERNS:
        for match in pattern.finditer(text):
            try:
                val = float(match.group(1).replace(",", "."))
                # Check if it was "voet" (Dutch feet)
                context = text[max(0, match.start()-20):match.end()+20]
                if "voet" in context.lower():
                    val *= 0.3  # convert to meters
                depths.append({
                    "value_m": round(val, 1),
                    "context": context.strip(),
                })
            except (ValueError, IndexError):
                continue
    return depths


def extract_locations(text):
    """Find location mentions and return coordinates."""
    found = []
    text_lower = text.lower()
    for name, (lat, lon) in LOCATION_PATTERNS.items():
        if name.lower() in text_lower:
            found.append({"name": name, "lat": lat, "lon": lon})
    return found


def extract_materials(text):
    """Find material mentions."""
    found = []
    text_lower = text.lower()
    for category, keywords in MATERIAL_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                found.append(category)
                break
    return list(set(found))


def extract_volcanic(text):
    """Find volcanic context mentions."""
    text_lower = text.lower()
    return [kw for kw in VOLCANIC_KEYWORDS if kw.lower() in text_lower]


def main():
    print("=" * 70)
    print("E141 Phase 2: Full-Text NLP from Delpher Colonial Newspapers")
    print("=" * 70)

    # Load Phase 1 results
    records = []
    with open(PHASE1_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append(row)

    print(f"\nPhase 1 records: {len(records)}")

    # Filter to high-relevance first (score >= 2), then fetch rest if time
    high_rel = [r for r in records if int(r.get("relevance", 0)) >= 2]
    print(f"High relevance (>=2): {len(high_rel)}")

    # Fetch full text and apply NLP
    print(f"\nFetching full text + NLP extraction...")
    results = []
    n_fetched = 0
    n_with_depth = 0
    n_with_location = 0
    n_with_volcanic = 0

    for i, record in enumerate(high_rel):
        url = record.get("identifier", "")
        if not url or not url.startswith("http"):
            continue

        # Fetch full text
        text = fetch_fulltext(url)
        if not text:
            if i < 5:
                print(f"  [{i+1}/{len(high_rel)}] SKIP (no text): {record['title'][:50]}")
            continue

        n_fetched += 1

        # Apply NLP
        depths = extract_depths(text)
        locations = extract_locations(text)
        materials = extract_materials(text)
        volcanic = extract_volcanic(text)

        result = {
            "title": record["title"],
            "date": record["date"],
            "source": record["source"],
            "identifier": url,
            "relevance": int(record.get("relevance", 0)),
            "phase1_tags": record.get("tags", ""),
            "text_length": len(text),
            "n_depths": len(depths),
            "depths_m": "; ".join(f"{d['value_m']}m" for d in depths) if depths else "",
            "depth_contexts": " | ".join(d['context'] for d in depths) if depths else "",
            "n_locations": len(locations),
            "locations": "; ".join(f"{l['name']} ({l['lat']:.2f},{l['lon']:.2f})" for l in locations),
            "primary_lat": locations[0]["lat"] if locations else None,
            "primary_lon": locations[0]["lon"] if locations else None,
            "materials": "; ".join(materials),
            "volcanic_context": "; ".join(volcanic),
            "has_depth": len(depths) > 0,
            "has_volcanic": len(volcanic) > 0,
        }
        results.append(result)

        if depths:
            n_with_depth += 1
        if locations:
            n_with_location += 1
        if volcanic:
            n_with_volcanic += 1

        # Progress
        if (i + 1) % 25 == 0 or depths:
            depth_str = f" DEPTH: {depths[0]['value_m']}m" if depths else ""
            loc_str = f" @ {locations[0]['name']}" if locations else ""
            print(f"  [{i+1}/{len(high_rel)}] {record['title'][:45]}{depth_str}{loc_str}")

        time.sleep(DELAY)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PHASE 2 RESULTS")
    print(f"{'='*70}")
    print(f"\n  Records fetched: {n_fetched}/{len(high_rel)}")
    print(f"  With depth values: {n_with_depth}")
    print(f"  With locations: {n_with_location}")
    print(f"  With volcanic context: {n_with_volcanic}")

    # Depth summary
    all_depths = []
    for r in results:
        if r["depths_m"]:
            for d_str in r["depths_m"].split("; "):
                try:
                    all_depths.append(float(d_str.replace("m", "")))
                except ValueError:
                    pass

    if all_depths:
        import numpy as np
        print(f"\n  DEPTH VALUES EXTRACTED:")
        print(f"    Count: {len(all_depths)}")
        print(f"    Range: {min(all_depths):.1f}m — {max(all_depths):.1f}m")
        print(f"    Median: {np.median(all_depths):.1f}m")
        print(f"    Mean: {np.mean(all_depths):.1f}m")

        for r in results:
            if r["depths_m"]:
                print(f"    [{r['date']}] {r['depths_m']} — {r['title'][:60]}")

    # Location summary
    if n_with_location > 0:
        from collections import Counter
        all_locs = []
        for r in results:
            if r["locations"]:
                for loc in r["locations"].split("; "):
                    name = loc.split(" (")[0]
                    all_locs.append(name)
        loc_counts = Counter(all_locs).most_common(15)
        print(f"\n  LOCATION FREQUENCIES:")
        for loc, count in loc_counts:
            print(f"    {loc:20s}: {count}")

    # Material summary
    if results:
        from collections import Counter
        all_mats = []
        for r in results:
            if r["materials"]:
                all_mats.extend(r["materials"].split("; "))
        mat_counts = Counter(all_mats).most_common(10)
        print(f"\n  MATERIAL FREQUENCIES:")
        for mat, count in mat_counts:
            print(f"    {mat:15s}: {count}")

    # ── Save ──────────────────────────────────────────────────────────
    if results:
        csv_path = RESULTS_DIR / "delpher_phase2_fulltext.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Saved: {csv_path.name} ({len(results)} records)")

    summary = {
        "experiment": "E141_phase2",
        "date": "2026-04-13",
        "records_fetched": n_fetched,
        "with_depth": n_with_depth,
        "with_location": n_with_location,
        "with_volcanic": n_with_volcanic,
        "depth_values": all_depths if all_depths else [],
    }
    with open(RESULTS_DIR / "delpher_phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Phase 2 complete. {n_with_depth} depth values, {n_with_location} geocoded.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

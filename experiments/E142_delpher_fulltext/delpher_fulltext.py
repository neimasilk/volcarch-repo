"""
E142: Delpher Full-Text Extraction + NLP Depth Mining
Phase 2 of E141: Fetch actual article text from high-relevance records,
then extract depth values, locations, and materials using NLP.

Uses KB resolver to get OCR text from individual articles.
"""

import requests
import json
import csv
import re
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
REPO = Path(__file__).parent.parent.parent

# Load E141 results
e141_path = REPO / "experiments/E141_delpher_extraction/results/delpher_extraction.csv"
import pandas as pd
df = pd.read_csv(e141_path)
print(f"E141 records loaded: {len(df)}")

# Filter high-relevance
high_rel = df[df["relevance"] >= 3].copy()
print(f"High relevance (>=3): {len(high_rel)}")

# === FETCH FULL TEXT VIA RESOLVER ===

def fetch_article_text(identifier):
    """Fetch OCR text for a Delpher article via resolver."""
    if not identifier or pd.isna(identifier):
        return None

    # Try resolver URL pattern
    # Delpher articles have identifiers like:
    # http://resolver.kb.nl/resolve?urn=MMKB27:021093037:mpeg21:a00152
    resolver_url = identifier
    if not resolver_url.startswith("http"):
        resolver_url = f"http://resolver.kb.nl/resolve?urn={identifier}"

    # Try to get the OCR text version
    text_url = resolver_url.replace("resolve?urn=", "resolve?urn=") + ":ocr"

    try:
        resp = requests.get(text_url, timeout=15, allow_redirects=True)
        if resp.status_code == 200 and len(resp.text) > 50:
            return resp.text
    except Exception:
        pass

    # Fallback: try without :ocr
    try:
        resp = requests.get(resolver_url, timeout=15, allow_redirects=True)
        if resp.status_code == 200:
            # Try to extract text from HTML
            text = re.sub(r'<[^>]+>', ' ', resp.text)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 100:
                return text[:5000]  # limit size
    except Exception:
        pass

    return None


def extract_archaeological_data(text, title=""):
    """Extract depth, location, material from Dutch text."""
    if not text:
        return {}

    combined = f"{title} {text}".lower()
    result = {}

    # Depth extraction (Dutch)
    depth_patterns = [
        (r'(\d+(?:[.,]\d+)?)\s*(?:meter|m\.?)\s*(?:diep|diepte|onder|beneden)', 1.0),
        (r'(?:diepte|diep)\s*(?:van|:)?\s*(\d+(?:[.,]\d+)?)\s*(?:meter|m\.?)', 1.0),
        (r'(\d+(?:[.,]\d+)?)\s*(?:voet|vt\.?)\s*(?:diep|diepte)', 0.3048),  # voet to meter
        (r'(\d+(?:[.,]\d+)?)\s*(?:el)\s*(?:diep|diepte)', 0.69),  # el to meter
        (r'op\s*(?:een\s*)?(?:diepte)\s*(?:van)?\s*(\d+(?:[.,]\d+)?)', 1.0),
    ]

    depths = []
    for pattern, multiplier in depth_patterns:
        for match in re.finditer(pattern, combined):
            try:
                val = float(match.group(1).replace(",", ".")) * multiplier
                if 0.1 < val < 50:  # reasonable range
                    depths.append(round(val, 2))
            except ValueError:
                pass

    if depths:
        result["depths_m"] = depths
        result["max_depth_m"] = max(depths)

    # Location extraction
    locations = []
    loc_patterns = [
        r'(?:te|bij|nabij|in|van)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
    ]
    java_places = ["Soerabaja", "Surabaya", "Malang", "Kediri", "Modjokerto",
                   "Mojokerto", "Djokja", "Jogja", "Semarang", "Batavia",
                   "Singosari", "Trowoelan", "Trowulan", "Blitar", "Madioen",
                   "Madiun", "Probolinggo", "Pasoeroean", "Pasuruan"]
    for place in java_places:
        if place.lower() in combined:
            locations.append(place)
    result["locations"] = locations

    # Material extraction
    materials = []
    material_map = {
        "steen": "stone", "steenen": "stone", "baksteen": "brick",
        "beeld": "statue", "beelden": "statues",
        "goud": "gold", "zilver": "silver", "brons": "bronze",
        "koper": "copper", "ijzer": "iron",
        "aardewerk": "pottery", "potscherven": "pottery sherds",
        "tempel": "temple", "tjandi": "temple", "candi": "temple",
        "fundament": "foundation", "muur": "wall",
        "bijl": "axe", "mes": "knife", "speer": "spear",
        "inscriptie": "inscription", "nagari": "inscription",
    }
    for dutch, english in material_map.items():
        if dutch in combined:
            materials.append(english)
    result["materials"] = list(set(materials))

    # Period indicators
    periods = []
    if any(w in combined for w in ["hindoe", "hindu", "boeddh", "buddh"]):
        periods.append("Hindu-Buddhist")
    if any(w in combined for w in ["majapahit", "madjapahit"]):
        periods.append("Majapahit")
    if any(w in combined for w in ["singosari", "singhasari"]):
        periods.append("Singosari")
    if any(w in combined for w in ["mataram"]):
        periods.append("Mataram")
    if any(w in combined for w in ["oud", "prehistorisch", "steentijd"]):
        periods.append("prehistoric")
    result["periods"] = periods

    return result


# === PROCESS HIGH-RELEVANCE ARTICLES ===

print(f"\n{'=' * 70}")
print(f"FETCHING FULL TEXT FOR {len(high_rel)} HIGH-RELEVANCE ARTICLES")
print("=" * 70)

extracted_finds = []
fetch_count = 0
max_fetch = 50  # limit API calls

for idx, row in high_rel.iterrows():
    if fetch_count >= max_fetch:
        break

    identifier = row.get("identifier", "")
    title = row.get("title", "")
    date = row.get("date", "")

    # Use title + description as fallback text
    text = str(row.get("description", ""))

    # Try to fetch full text
    if identifier and not pd.isna(identifier):
        full_text = fetch_article_text(identifier)
        if full_text:
            text = full_text
            fetch_count += 1

    # Extract data
    data = extract_archaeological_data(text, title)

    if data.get("depths_m") or data.get("materials") or data.get("locations"):
        find = {
            "title": title,
            "date": date,
            "identifier": identifier,
            "depths_m": data.get("depths_m", []),
            "max_depth_m": data.get("max_depth_m"),
            "locations": data.get("locations", []),
            "materials": data.get("materials", []),
            "periods": data.get("periods", []),
            "text_snippet": text[:300] if text else "",
        }
        extracted_finds.append(find)

    time.sleep(0.5)  # be gentle with API

# Also process from title/description for all records (no API needed)
print(f"\nProcessing all {len(df)} records from metadata only...")
for idx, row in df.iterrows():
    title = str(row.get("title", ""))
    desc = str(row.get("description", ""))
    text = f"{title} {desc}"
    date = row.get("date", "")

    data = extract_archaeological_data(text, "")

    if data.get("depths_m") or (data.get("materials") and data.get("locations")):
        find = {
            "title": title,
            "date": date,
            "identifier": row.get("identifier", ""),
            "depths_m": data.get("depths_m", []),
            "max_depth_m": data.get("max_depth_m"),
            "locations": data.get("locations", []),
            "materials": data.get("materials", []),
            "periods": data.get("periods", []),
            "text_snippet": text[:300],
        }
        # Dedup by title
        if not any(f["title"] == find["title"] for f in extracted_finds):
            extracted_finds.append(find)

# === RESULTS ===

print(f"\n{'=' * 70}")
print(f"EXTRACTION RESULTS")
print("=" * 70)

print(f"\n  Total extracted finds: {len(extracted_finds)}")
with_depth = [f for f in extracted_finds if f["max_depth_m"]]
with_location = [f for f in extracted_finds if f["locations"]]
with_material = [f for f in extracted_finds if f["materials"]]

print(f"  With depth values: {len(with_depth)}")
print(f"  With locations: {len(with_location)}")
print(f"  With materials: {len(with_material)}")

if with_depth:
    print(f"\n  FINDS WITH DEPTH VALUES:")
    for f in sorted(with_depth, key=lambda x: x["max_depth_m"] or 0, reverse=True):
        locs = ", ".join(f["locations"]) if f["locations"] else "unknown"
        mats = ", ".join(f["materials"]) if f["materials"] else "-"
        print(f"    {f['max_depth_m']:.1f}m | {f['date'][:10]} | {locs} | {mats} | {f['title'][:60]}")

if with_location and with_material:
    print(f"\n  FINDS WITH LOCATION + MATERIAL (no depth):")
    no_depth = [f for f in extracted_finds if not f["max_depth_m"] and f["locations"] and f["materials"]]
    for f in no_depth[:15]:
        locs = ", ".join(f["locations"])
        mats = ", ".join(f["materials"])
        print(f"    [{f['date'][:10]}] {locs}: {mats} — {f['title'][:60]}")

# === SAVE ===

with open(RESULTS_DIR / "delpher_finds.json", "w", encoding="utf-8") as fp:
    json.dump(extracted_finds, fp, indent=2, ensure_ascii=False)

with open(RESULTS_DIR / "delpher_finds.csv", "w", newline="", encoding="utf-8") as fp:
    writer = csv.writer(fp)
    writer.writerow(["title", "date", "max_depth_m", "locations", "materials", "periods", "identifier"])
    for f in extracted_finds:
        writer.writerow([
            f["title"], f["date"], f.get("max_depth_m", ""),
            "; ".join(f["locations"]), "; ".join(f["materials"]),
            "; ".join(f["periods"]), f["identifier"],
        ])

summary = {
    "experiment": "E142_delpher_fulltext",
    "total_finds": len(extracted_finds),
    "with_depth": len(with_depth),
    "with_location": len(with_location),
    "with_material": len(with_material),
    "api_calls": fetch_count,
}

with open(RESULTS_DIR / "summary.json", "w") as fp:
    json.dump(summary, fp, indent=2)

print(f"\n  Saved to {RESULTS_DIR}/")

"""
E141: Delpher Colonial Newspaper Extraction Pipeline
Automated extraction of archaeological mentions from Dutch colonial newspapers
via KB SRU API. First dataset for P21 ColonialMine.

API: https://jsru.kb.nl/sru/sru (public, no registration needed)
Collection: DDD_artikel (newspaper articles)
"""

import requests
import xml.etree.ElementTree as ET
import json
import csv
import re
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SRU_BASE = "https://jsru.kb.nl/sru/sru"
COLLECTION = "DDD_artikel"
MAX_PER_REQUEST = 20
DELAY = 2  # seconds between requests

# === SEARCH QUERIES ===
# Multiple queries to cover different archaeological contexts

QUERIES = [
    # Direct archaeological finds with depth
    "opgegraven AND diepte AND Java",
    "gevonden AND diepte AND Java AND oud",
    "ontdekt AND meter AND Java AND tempel",
    "oudheidkundige AND vondsten AND Java",
    # Volcanic burial context
    "begraven AND vulkaan AND Java AND steen",
    "bedolven AND Java AND beeld",
    # Construction finds
    "aanleg AND gevonden AND Java AND diep",
    "spoorweg AND gevonden AND Java AND oud",
    # Specific sites
    "Singosari AND opgegraven",
    "Modjokerto AND oudheidkundig",
    "Trowoelan AND opgegraven",
    "Kediri AND oudheidkundig",
]

def search_sru(query, start=1, max_records=20):
    """Query KB SRU API and return parsed results."""
    params = {
        "operation": "searchRetrieve",
        "x-collection": COLLECTION,
        "query": query,
        "startRecord": start,
        "maximumRecords": max_records,
        "recordSchema": "dc",
    }

    try:
        resp = requests.get(SRU_BASE, params=params, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ERROR: {e}")
        return 0, []

    # Parse XML
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        print(f"  ERROR: XML parse failed")
        return 0, []

    ns = {
        "srw": "http://www.loc.gov/zing/srw/",
        "dc": "http://purl.org/dc/elements/1.1/",
    }

    total = int(root.findtext(".//srw:numberOfRecords", "0", ns))
    records = []

    for rec in root.findall(".//srw:record", ns):
        data = rec.find(".//srw:recordData", ns)
        if data is None:
            continue

        record = {
            "title": "",
            "date": "",
            "source": "",
            "identifier": "",
            "description": "",
            "subject": "",
        }

        for elem in data.iter():
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            text = (elem.text or "").strip()
            if not text:
                continue

            if tag == "title" and not record["title"]:
                record["title"] = text
            elif tag == "date" and not record["date"]:
                record["date"] = text[:10]  # YYYY/MM/DD
            elif tag == "source" and not record["source"]:
                record["source"] = text
            elif tag == "identifier" and not record["identifier"]:
                record["identifier"] = text
            elif tag == "description":
                record["description"] += " " + text
            elif tag == "subject":
                record["subject"] += "; " + text if record["subject"] else text

        if record["title"]:
            records.append(record)

    return total, records


def classify_record(record):
    """Classify a record for archaeological relevance."""
    text = f"{record['title']} {record['description']}".lower()

    relevance = 0
    tags = []

    # Depth mentions
    if any(w in text for w in ["diepte", "diep", "meter diep", "voet diep"]):
        relevance += 3
        tags.append("DEPTH")

    # Archaeological terms
    if any(w in text for w in ["opgegraven", "opgraving", "oudheidkundig", "oudheden"]):
        relevance += 3
        tags.append("ARCHAEOLOGY")

    # Materials
    if any(w in text for w in ["beeld", "beelden", "statue"]):
        relevance += 2
        tags.append("STATUE")
    if any(w in text for w in ["tempel", "tjandi", "candi"]):
        relevance += 2
        tags.append("TEMPLE")
    if any(w in text for w in ["steen", "steenen", "baksteen"]):
        relevance += 1
        tags.append("STONE")
    if any(w in text for w in ["goud", "zilver", "brons", "koper"]):
        relevance += 2
        tags.append("METAL")
    if any(w in text for w in ["potscherven", "aardewerk"]):
        relevance += 2
        tags.append("POTTERY")

    # Volcanic context
    if any(w in text for w in ["vulkaan", "vulkanisch", "lava", "asch", "modder"]):
        relevance += 2
        tags.append("VOLCANIC")

    # Location specificity
    if any(w in text for w in ["java", "jawa", "soerabaja", "batavia", "malang",
                                "kediri", "modjokerto", "singosari", "trowoelan"]):
        relevance += 1
        tags.append("JAVA")

    # Extract depth value if present
    depth_m = None
    depth_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:meter|M\.?)\s*(?:diep|diepte)', text)
    if depth_match:
        try:
            depth_m = float(depth_match.group(1).replace(",", "."))
        except ValueError:
            pass

    # Try "voet" (Dutch feet, ~0.3m)
    if depth_m is None:
        voet_match = re.search(r'(\d+(?:[.,]\d+)?)\s*voet\s*(?:diep|diepte)', text)
        if voet_match:
            try:
                depth_m = float(voet_match.group(1).replace(",", ".")) * 0.3
            except ValueError:
                pass

    return relevance, tags, depth_m


# === MAIN EXTRACTION ===

print("=" * 70)
print("E141: DELPHER COLONIAL NEWSPAPER EXTRACTION")
print(f"API: {SRU_BASE}")
print(f"Collection: {COLLECTION}")
print(f"Queries: {len(QUERIES)}")
print("=" * 70)

all_records = []
seen_ids = set()

for qi, query in enumerate(QUERIES):
    print(f"\n[{qi+1}/{len(QUERIES)}] Query: {query}")

    total, records = search_sru(query, max_records=50)
    print(f"  Total available: {total}")

    new = 0
    for r in records:
        rid = r["identifier"]
        if rid not in seen_ids:
            seen_ids.add(rid)
            relevance, tags, depth_m = classify_record(r)
            r["relevance"] = relevance
            r["tags"] = "; ".join(tags)
            r["depth_m"] = depth_m
            all_records.append(r)
            new += 1

    print(f"  New records: {new} (total unique: {len(all_records)})")
    time.sleep(DELAY)

# === ANALYSIS ===

print(f"\n{'=' * 70}")
print(f"EXTRACTION COMPLETE: {len(all_records)} unique records")
print("=" * 70)

# Sort by relevance
all_records.sort(key=lambda r: r["relevance"], reverse=True)

# High relevance
high_rel = [r for r in all_records if r["relevance"] >= 4]
with_depth = [r for r in all_records if r["depth_m"] is not None]

print(f"\n  Total unique records: {len(all_records)}")
print(f"  High relevance (>=4): {len(high_rel)}")
print(f"  With depth value: {len(with_depth)}")

print(f"\n  TOP 20 MOST RELEVANT:")
print(f"  {'Score':>5} {'Date':>10} {'Depth':>6} {'Tags':<30} {'Title'}")
print(f"  {'-'*5} {'-'*10} {'-'*6} {'-'*30} {'-'*50}")

for r in all_records[:20]:
    depth_str = f"{r['depth_m']:.1f}m" if r["depth_m"] else "-"
    print(f"  {r['relevance']:>5} {r['date']:>10} {depth_str:>6} {r['tags']:<30} {r['title'][:50]}")

if with_depth:
    print(f"\n  RECORDS WITH DEPTH VALUES:")
    for r in with_depth:
        print(f"    [{r['date']}] {r['depth_m']:.1f}m — {r['title'][:80]}")

# === SAVE ===

# CSV
csv_path = RESULTS_DIR / "delpher_extraction.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "title", "date", "source", "identifier", "description",
        "subject", "relevance", "tags", "depth_m"
    ])
    writer.writeheader()
    writer.writerows(all_records)

# JSON summary
summary = {
    "experiment": "E141_delpher_extraction",
    "api_endpoint": SRU_BASE,
    "collection": COLLECTION,
    "queries_run": len(QUERIES),
    "total_unique_records": len(all_records),
    "high_relevance": len(high_rel),
    "with_depth_value": len(with_depth),
    "depth_values": [{"date": r["date"], "depth_m": r["depth_m"], "title": r["title"]}
                     for r in with_depth],
}

with open(RESULTS_DIR / "delpher_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n  Saved to {RESULTS_DIR}/")
print(f"  CSV: {csv_path.name} ({len(all_records)} records)")
print(f"  JSON: delpher_summary.json")

#!/usr/bin/env python3
"""
E141 Phase 2b: Expanded Delpher Search — Construction Finds
=============================================================
Phase 1 used 12 queries → 529 records.
Phase 2b adds 25+ new queries targeting:
- Railway construction (spoorweg)
- Canal/irrigation (kanaal, irrigatie)
- Well-digging (put, waterput, bron)
- Foundation work (fundament, grondwerk)
- Road construction (weg, aanleg)
- Specific East Java locations not yet searched
- Depth-specific queries

These contexts are the most valuable because colonial workers
ACCIDENTALLY discovered archaeological material while digging.
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

SRU_BASE = "https://jsru.kb.nl/sru/sru"
COLLECTION = "DDD_artikel"
DELAY = 2

# ── EXPANDED QUERIES ──────────────────────────────────────────────────

QUERIES = [
    # Construction contexts (accidental finds)
    "spoorweg AND gevonden AND Java AND oudheden",
    "spoorweg AND opgegraven AND Java",
    "spoorlijn AND gevonden AND oud AND Java",
    "kanaal AND gevonden AND Java AND oud",
    "irrigatie AND gevonden AND Java AND beeld",
    "fundament AND gevonden AND Java AND oud",
    "grondwerk AND Java AND oudheidkundig",
    "aanleg AND weg AND Java AND gevonden AND oud",
    "put AND gevonden AND Java AND oudheden",
    "waterput AND Java AND oud",
    "bron AND gegraven AND Java AND oud",
    # Depth-specific
    "meter AND diep AND Java AND oud AND gevonden",
    "voet AND diep AND Java AND gevonden AND oud",
    "onder AND grond AND Java AND beeld",
    "begraven AND Java AND tempel",
    "bedolven AND Java AND oudheden",
    # Volcanic burial explicit
    "vulkanische AND asch AND Java AND begraven",
    "lava AND bedekt AND Java AND tempel",
    "lahar AND Java AND oud",
    "modder AND begraven AND Java",
    # Specific East Java locations (not in Phase 1)
    "Penataran AND opgegraven",
    "Kloet AND oudheidkundig",
    "Soerabaja AND oudheidkundig AND vondst",
    "Pasoeroean AND oudheidkundig",
    "Madioen AND oudheidkundig",
    "Djember AND oudheidkundig",
    "Probolinggo AND oudheidkundig",
    "Gresik AND oudheidkundig",
    "Toeban AND oudheidkundig",
    # Material-specific
    "bronstijd AND Java",
    "steenen AND bijlen AND Java",
    "neolithisch AND Java",
    "praehistorisch AND Java",
    "megalithisch AND Java",
]


def search_sru(query, start=1, max_records=50):
    """Query KB SRU API."""
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

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
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

        record = {"title": "", "date": "", "source": "", "identifier": "", "description": ""}
        for elem in data.iter():
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            text = (elem.text or "").strip()
            if not text:
                continue
            if tag == "title" and not record["title"]:
                record["title"] = text
            elif tag == "date" and not record["date"]:
                record["date"] = text[:10]
            elif tag == "source" and not record["source"]:
                record["source"] = text
            elif tag == "identifier" and not record["identifier"]:
                record["identifier"] = text
            elif tag == "description":
                record["description"] += " " + text

        if record["title"]:
            records.append(record)

    return total, records


def classify_record(record):
    """Classify relevance (same as Phase 1 but with construction keywords)."""
    text = f"{record['title']} {record['description']}".lower()
    relevance = 0
    tags = []

    if any(w in text for w in ["diepte", "diep", "meter diep", "voet diep"]):
        relevance += 3; tags.append("DEPTH")
    if any(w in text for w in ["opgegraven", "opgraving", "oudheidkundig", "oudheden"]):
        relevance += 3; tags.append("ARCHAEOLOGY")
    if any(w in text for w in ["beeld", "beelden", "standbeeld"]):
        relevance += 2; tags.append("STATUE")
    if any(w in text for w in ["tempel", "tjandi", "candi"]):
        relevance += 2; tags.append("TEMPLE")
    if any(w in text for w in ["steen", "steenen", "baksteen"]):
        relevance += 1; tags.append("STONE")
    if any(w in text for w in ["goud", "zilver", "brons", "koper"]):
        relevance += 2; tags.append("METAL")
    if any(w in text for w in ["potscherven", "aardewerk"]):
        relevance += 2; tags.append("POTTERY")
    if any(w in text for w in ["vulkaan", "vulkanisch", "lava", "asch", "lahar"]):
        relevance += 2; tags.append("VOLCANIC")
    if any(w in text for w in ["spoorweg", "spoorlijn", "kanaal", "irrigatie",
                                "fundament", "grondwerk", "waterput"]):
        relevance += 2; tags.append("CONSTRUCTION")
    if any(w in text for w in ["praehistorisch", "neolithisch", "bronstijd", "megalithisch"]):
        relevance += 3; tags.append("PREHISTORIC")
    if any(w in text for w in ["java", "jawa", "soerabaja", "batavia", "malang",
                                "kediri", "modjokerto", "singosari", "trowoelan"]):
        relevance += 1; tags.append("JAVA")

    return relevance, "; ".join(tags)


# ── MAIN ──────────────────────────────────────────────────────────────

print("=" * 70)
print("E141 Phase 2b: Expanded Delpher Search")
print(f"Queries: {len(QUERIES)}")
print("=" * 70)

# Load existing IDs to avoid duplicates
existing_ids = set()
if PHASE1_CSV.exists():
    with open(PHASE1_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            existing_ids.add(row.get("identifier", ""))

print(f"Existing records: {len(existing_ids)}")

new_records = []
seen_ids = set(existing_ids)

for qi, query in enumerate(QUERIES):
    print(f"\n[{qi+1}/{len(QUERIES)}] {query}")
    total, records = search_sru(query, max_records=50)
    print(f"  Available: {total}", end="")

    new = 0
    for r in records:
        rid = r["identifier"]
        if rid not in seen_ids:
            seen_ids.add(rid)
            relevance, tags = classify_record(r)
            r["relevance"] = relevance
            r["tags"] = tags
            new_records.append(r)
            new += 1

    print(f"  New: {new} (total new: {len(new_records)})")
    time.sleep(DELAY)

# ── RESULTS ───────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"EXPANDED SEARCH COMPLETE")
print(f"{'='*70}")
print(f"  New unique records: {len(new_records)}")
print(f"  Previous records: {len(existing_ids)}")
print(f"  Combined total: {len(existing_ids) + len(new_records)}")

# Sort by relevance
new_records.sort(key=lambda r: r["relevance"], reverse=True)

high_rel = [r for r in new_records if r["relevance"] >= 3]
construction = [r for r in new_records if "CONSTRUCTION" in r.get("tags", "")]
prehistoric = [r for r in new_records if "PREHISTORIC" in r.get("tags", "")]
volcanic = [r for r in new_records if "VOLCANIC" in r.get("tags", "")]

print(f"  High relevance (>=3): {len(high_rel)}")
print(f"  Construction context: {len(construction)}")
print(f"  Prehistoric mentions: {len(prehistoric)}")
print(f"  Volcanic context: {len(volcanic)}")

print(f"\n  TOP 20 NEW RECORDS:")
for r in new_records[:20]:
    print(f"  [{r['relevance']:>2}] [{r['date']}] {r['tags'][:25]:25s} {r['title'][:55]}")

# Save new records
csv_path = RESULTS_DIR / "delpher_expanded.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "title", "date", "source", "identifier", "description", "relevance", "tags"
    ])
    writer.writeheader()
    writer.writerows(new_records)

# Also save combined (phase 1 + expanded)
combined = []
if PHASE1_CSV.exists():
    with open(PHASE1_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            combined.append(row)
combined.extend(new_records)

combined_path = RESULTS_DIR / "delpher_combined.csv"
with open(combined_path, "w", newline="", encoding="utf-8") as f:
    fields = ["title", "date", "source", "identifier", "description", "relevance", "tags"]
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(combined)

summary = {
    "phase": "2b_expanded",
    "date": "2026-04-13",
    "queries_run": len(QUERIES),
    "new_records": len(new_records),
    "previous_records": len(existing_ids),
    "combined_total": len(combined),
    "high_relevance": len(high_rel),
    "construction_context": len(construction),
    "prehistoric": len(prehistoric),
    "volcanic": len(volcanic),
}

with open(RESULTS_DIR / "delpher_expanded_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Saved: {csv_path.name} ({len(new_records)} new)")
print(f"  Saved: {combined_path.name} ({len(combined)} combined)")
print(f"{'='*70}")

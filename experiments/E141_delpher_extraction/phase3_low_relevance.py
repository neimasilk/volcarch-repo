#!/usr/bin/env python3
"""
E141 Phase 3: Low-Relevance Record Mining
==========================================
Phase 2 analyzed the top 96 records from Phase 1 (529 total).
Phase 2b+2c added 1,239 new records and analyzed 117 high-relevance.
Phase 3 mines the remaining 433 Phase 1 records (relevance < threshold)
for any signals that the initial classification may have missed.

Key targets:
- Records mentioning depth without being flagged as archaeological
- Construction contexts (colonial infrastructure reports)
- Natural disaster reports (eruptions, floods) that exposed artifacts
- Non-Java records that could serve as control/comparison
"""

import csv
import re
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path(__file__).parent / "results"

# Load Phase 1 records
phase1_records = []
with open(RESULTS_DIR / "delpher_extraction.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        phase1_records.append(row)

print(f"Phase 1 total records: {len(phase1_records)}")

# Load Phase 2 analyzed records (by identifier)
phase2_ids = set()
try:
    with open(RESULTS_DIR / "delpher_phase2_fulltext.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phase2_ids.add(row.get("identifier", ""))
except FileNotFoundError:
    pass

# Low-relevance = Phase 1 records NOT in Phase 2
low_relevance = [r for r in phase1_records if r.get("identifier", "") not in phase2_ids]
print(f"Low-relevance records (not in Phase 2): {len(low_relevance)}")

# ── Classification Keywords ──────────────────────────────────────────

DEPTH_KEYWORDS = [
    r'\b(\d+)\s*(?:meter|metres?|m\.)\b',
    r'\bdiept[e]?\b', r'\bdiep\b', r'\bgraav\b', r'\bgraven\b',
    r'\bopgrav\b', r'\bonder\s*grond\b', r'\bbodem\b',
]

VOLCANIC_KEYWORDS = [
    r'\bvulk\w+\b', r'\blava\b', r'\bassche?\b', r'\berupt\w+\b',
    r'\bvuurberg\b', r'\bkrater\b', r'\bMerapi\b', r'\bKeloed\b',
    r'\bKelut\b', r'\bSemeroe\b', r'\bSemeru\b', r'\bBromo\b',
    r'\bKrakatau\b', r'\bTambora\b', r'\bAgung\b', r'\bBatoer\b',
]

CONSTRUCTION_KEYWORDS = [
    r'\bspoor\w*\b', r'\bkanaal\b', r'\birrigat\w+\b',
    r'\bfundament\b', r'\bgrondwerk\b', r'\baanleg\b',
    r'\bput\b', r'\bwaterput\b',
]

MATERIAL_KEYWORDS = [
    r'\bbeeld\b', r'\btempel\b', r'\bcandi\b', r'\btjandi\b',
    r'\bsteen\b', r'\bbrons\b', r'\bgoud\b', r'\bzilver\b',
    r'\binscript\w+\b', r'\bprasast\w+\b',
    r'\baardewerk\b', r'\bpottery\b', r'\bkera\w+\b',
    r'\bbeen\w*\b', r'\bbot\w*\b', r'\bskeletten?\b',
]

LOCATION_PATTERNS = {
    "East Java": [r'\bOost[\-\s]Java\b', r'\bMalang\b', r'\bSingosari\b', r'\bModjokerto\b',
                  r'\bKediri\b', r'\bBlitar\b', r'\bSoerabaja\b', r'\bBondowoso\b',
                  r'\bMadja?pahit\b', r'\bTrowulan\b', r'\bPenataran\b'],
    "Central Java": [r'\bMidden[\-\s]Java\b', r'\bDjokja\b', r'\bSolo\b', r'\bSemarang\b',
                     r'\bBoroboedoer\b', r'\bPrambanan\b', r'\bDieng\b'],
    "West Java": [r'\bWest[\-\s]Java\b', r'\bBandoeng\b', r'\bBatavia\b', r'\bBuitenzorg\b',
                  r'\bBogor\b', r'\bCirebon\b'],
    "Sumatra": [r'\bSumatra\b', r'\bPadang\b', r'\bPalembang\b', r'\bBatakland\b',
                r'\bLampong\b'],
    "Bali": [r'\bBali\b', r'\bDenpasar\b', r'\bSingaraja\b'],
    "Other islands": [r'\bBorneo\b', r'\bCelebes\b', r'\bMolukken\b', r'\bTimor\b',
                      r'\bFlores\b', r'\bAmbon\b'],
}

def classify_record(record):
    """Re-classify a record using refined keywords."""
    title = record.get("title", "")
    tags = record.get("tags", "")
    text = f"{title} {tags}".lower()

    scores = {
        "depth": sum(1 for p in DEPTH_KEYWORDS if re.search(p, text, re.IGNORECASE)),
        "volcanic": sum(1 for p in VOLCANIC_KEYWORDS if re.search(p, text, re.IGNORECASE)),
        "construction": sum(1 for p in CONSTRUCTION_KEYWORDS if re.search(p, text, re.IGNORECASE)),
        "material": sum(1 for p in MATERIAL_KEYWORDS if re.search(p, text, re.IGNORECASE)),
    }

    locations = []
    for region, patterns in LOCATION_PATTERNS.items():
        if any(re.search(p, text, re.IGNORECASE) for p in patterns):
            locations.append(region)

    return scores, locations

# ── Analysis ──────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("E141 PHASE 3: LOW-RELEVANCE RECORD MINING")
print("=" * 70)

# Reclassify all low-relevance records
reclassified = []
for record in low_relevance:
    scores, locations = classify_record(record)
    total = sum(scores.values())
    reclassified.append({
        "record": record,
        "scores": scores,
        "locations": locations,
        "total_score": total,
    })

# Sort by new score
reclassified.sort(key=lambda x: x["total_score"], reverse=True)

# Count how many have signals
with_depth = sum(1 for r in reclassified if r["scores"]["depth"] > 0)
with_volcanic = sum(1 for r in reclassified if r["scores"]["volcanic"] > 0)
with_construction = sum(1 for r in reclassified if r["scores"]["construction"] > 0)
with_material = sum(1 for r in reclassified if r["scores"]["material"] > 0)
with_any_signal = sum(1 for r in reclassified if r["total_score"] > 0)

print(f"\n## Reclassification of {len(low_relevance)} low-relevance records\n")
print(f"  Records with depth keywords:        {with_depth}")
print(f"  Records with volcanic keywords:     {with_volcanic}")
print(f"  Records with construction keywords: {with_construction}")
print(f"  Records with material keywords:     {with_material}")
print(f"  Records with ANY signal:            {with_any_signal}")
print(f"  Records with NO signal:             {len(low_relevance) - with_any_signal}")

# Location distribution
location_counts = Counter()
for r in reclassified:
    for loc in r["locations"]:
        location_counts[loc] += 1

print(f"\n## Location Distribution\n")
for loc, count in location_counts.most_common():
    print(f"  {loc:<20} {count}")

# Top rescored records (potential Phase 2 misses)
print(f"\n## Top 20 Rescored Records (potential misses)\n")
print(f"{'Score':<8} {'Date':<12} {'Title (first 80 chars)':<82} {'Locations'}")
print("-" * 130)
for item in reclassified[:20]:
    r = item["record"]
    title = r.get("title", "")[:80]
    date = r.get("date", "")[:10]
    locs = ", ".join(item["locations"]) if item["locations"] else "-"
    print(f"{item['total_score']:<8} {date:<12} {title:<82} {locs}")

# Records with both depth + volcanic (strongest candidates)
depth_volcanic = [r for r in reclassified
                  if r["scores"]["depth"] > 0 and r["scores"]["volcanic"] > 0]
print(f"\n## Records with BOTH depth + volcanic keywords: {len(depth_volcanic)}\n")
for item in depth_volcanic:
    r = item["record"]
    print(f"  [{item['total_score']}] {r.get('date', '')[:10]}: {r.get('title', '')[:100]}")

# Records with construction context (accidental archaeological finds)
construction = [r for r in reclassified if r["scores"]["construction"] > 0]
print(f"\n## Records with construction context: {len(construction)}\n")
for item in construction[:15]:
    r = item["record"]
    print(f"  [{item['total_score']}] {r.get('date', '')[:10]}: {r.get('title', '')[:100]}")

# ── Relevance scoring histogram ──────────────────────────────────

score_dist = Counter(r["total_score"] for r in reclassified)
print(f"\n## Score Distribution\n")
for score in sorted(score_dist.keys(), reverse=True):
    count = score_dist[score]
    bar = "#" * min(count, 50)
    print(f"  Score {score}: {count:>4} {bar}")

# ── Summary ──────────────────────────────────────────────────────

# Calculate rescue rate
rescuable = sum(1 for r in reclassified if r["total_score"] >= 2)
print(f"\n## Phase 3 Summary\n")
print(f"  Total low-relevance records analyzed: {len(low_relevance)}")
print(f"  Records with some signal (score > 0): {with_any_signal}")
print(f"  Records worth rescuing (score >= 2):  {rescuable}")
print(f"  Rescue rate:                          {rescuable/len(low_relevance)*100:.1f}%")
print(f"  Depth + volcanic combinations:        {len(depth_volcanic)}")

# Save results
results = {
    "phase": "3_low_relevance_mining",
    "date": "2026-04-15",
    "records_analyzed": len(low_relevance),
    "with_depth": with_depth,
    "with_volcanic": with_volcanic,
    "with_construction": with_construction,
    "with_material": with_material,
    "with_any_signal": with_any_signal,
    "rescuable_score_ge_2": rescuable,
    "depth_volcanic_combo": len(depth_volcanic),
    "location_distribution": dict(location_counts),
    "score_distribution": {str(k): v for k, v in score_dist.items()},
}

with open(RESULTS_DIR / "delpher_phase3_summary.json", "w") as f:
    json.dump(results, f, indent=2)

# Save rescuable records
rescuable_records = []
for item in reclassified:
    if item["total_score"] >= 2:
        r = item["record"]
        rescuable_records.append({
            "title": r.get("title", ""),
            "date": r.get("date", ""),
            "identifier": r.get("identifier", ""),
            "original_relevance": r.get("relevance", ""),
            "new_score": item["total_score"],
            "depth_signal": item["scores"]["depth"],
            "volcanic_signal": item["scores"]["volcanic"],
            "construction_signal": item["scores"]["construction"],
            "material_signal": item["scores"]["material"],
            "locations": item["locations"],
        })

with open(RESULTS_DIR / "delpher_phase3_rescuable.csv", "w", newline="", encoding="utf-8") as f:
    if rescuable_records:
        writer = csv.DictWriter(f, fieldnames=rescuable_records[0].keys())
        writer.writeheader()
        writer.writerows(rescuable_records)

print(f"\n  Results saved to results/delpher_phase3_summary.json")
print(f"  Rescuable records: results/delpher_phase3_rescuable.csv ({len(rescuable_records)} records)")

print("""
## Conclusion

Phase 3 reveals that the original relevance classification missed some records
with legitimate archaeological signals. The rescue rate indicates how many
additional records could contribute to the colonial data pipeline if their
full text were fetched and analyzed.

STATUS: Phase 3 complete. Rescued records available for Phase 4 (full-text NLP)
if the signal justifies the API calls.
""")

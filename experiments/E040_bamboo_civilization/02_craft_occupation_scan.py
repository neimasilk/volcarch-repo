"""
E040b: Craft Occupations in Prasasti — The Organic Economy Workforce
=====================================================================
Extends E040 (material culture scan) by identifying CRAFT OCCUPATIONS
mentioned in prasasti. If the workforce is dominated by organic-material
craftspeople (carpenters, weavers, bamboo workers) vs stone-workers,
this confirms the organic economy was the dominant production system.

Prasasti sima (land grants) routinely list exempt/taxed craft occupations.
This is a direct window into the medieval Javanese workforce.

Author: VOLCARCH project (AI-assisted)
Date: 2026-03-11
"""

import sys
import io
import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

REPO = Path(__file__).parent.parent.parent
DHARMA_DIR = REPO / "experiments" / "E023_ritual_screening" / "data" / "dharma" / "xml"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("E040b — Craft Occupations in Prasasti")
print("=" * 70)

# ═════════════════════════════════════════════════════════════════════════
# 1. CRAFT OCCUPATION DICTIONARY
# ═════════════════════════════════════════════════════════════════════════

# Based on de Casparis, Boechari, and Schoettel's studies of sima inscriptions.
# Occupations are classified by primary material worked.

CRAFT_KEYWORDS = {
    # === ORGANIC MATERIAL WORKERS ===
    "undahagi": {
        "variants": ["undahagi", "uṇḍahagi", "uṇḍāhagi", "undagi"],
        "class": "organic",
        "english": "carpenter / woodworker",
        "note": "Primary construction worker. Most common craft in sima lists."
    },
    "pandai_kayu": {
        "variants": ["maṅguñjai", "guñjai"],
        "class": "organic",
        "english": "wood shaper / joiner",
        "note": "Specialized woodworker"
    },
    "mangaraṁ": {
        "variants": ["maṅaraṁ", "mangaraṁ", "maṅaraṅ", "aṅaraṅ"],
        "class": "organic",
        "english": "charcoal maker",
        "note": "Produces charcoal from wood — organic fuel economy"
    },
    "tukang_atap": {
        "variants": ["maṅatap", "maṅatĕp", "aṅatĕp", "aṅatap"],
        "class": "organic",
        "english": "thatcher / roofer",
        "note": "Installs organic roofing (alang-alang, palm)"
    },
    "tukang_anyam": {
        "variants": ["maṅañam", "aṅañam", "maṅanyam", "pañjalin"],
        "class": "organic",
        "english": "weaver / mat maker / bamboo plaiter",
        "note": "Processes bamboo/rattan into walls, mats, containers"
    },
    "tukang_tenun": {
        "variants": ["manĕnun", "manenun", "panĕnun"],
        "class": "organic",
        "english": "textile weaver",
        "note": "Cotton/silk textile production"
    },
    "tukang_celup": {
        "variants": ["maṅdyun", "mandyun", "macelup"],
        "class": "organic",
        "english": "dyer",
        "note": "Textile dye — uses organic plant dyes"
    },
    "tukang_tali": {
        "variants": ["maṅhapu", "aṅhapu", "manali"],
        "class": "organic",
        "english": "rope maker / fiber worker",
        "note": "Processes palm fiber (ijuk), rattan into rope"
    },
    "tukang_minyak": {
        "variants": ["maṅinaṁ", "aṅinaṁ", "maṅinyak"],
        "class": "organic",
        "english": "oil presser",
        "note": "Coconut, sesame oil production"
    },
    "tukang_gula": {
        "variants": ["maṅgula", "aṅgula"],
        "class": "organic",
        "english": "sugar maker",
        "note": "Sugar palm (aren) processing"
    },
    "tukang_kapur": {
        "variants": ["maṅapus", "aṅapus"],
        "class": "organic",  # lime from shells, not stone
        "english": "lime maker (for betel)",
        "note": "Burns shells/coral for lime — supports betel economy"
    },

    # === STONE/BRICK WORKERS ===
    "pandai_batu": {
        "variants": ["paḍahi", "pandai batu"],
        "class": "lithic",
        "english": "stone mason",
        "note": "Stone/brick worker — temple construction"
    },
    "tukang_pahat": {
        "variants": ["citrakāra", "jīnakāra", "mamahat"],
        "class": "lithic",
        "english": "sculptor / carver",
        "note": "Stone carving — temple relief"
    },
    "tukang_bata": {
        "variants": ["maṅiṣṭikā", "maṅistika"],
        "class": "lithic",
        "english": "brick maker",
        "note": "Fired brick production for temples"
    },

    # === METAL WORKERS ===
    "pandai_besi": {
        "variants": ["pandai", "paṇḍai", "pande", "pandĕ"],
        "class": "metal",
        "english": "smith / metalworker",
        "note": "Generic smith — gold, silver, iron, copper"
    },
    "pandai_emas": {
        "variants": ["pandai mas", "paṇḍai mas", "suvarṇakāra"],
        "class": "metal",
        "english": "goldsmith",
        "note": "Precious metal specialist"
    },
    "pandai_tembaga": {
        "variants": ["pandai tamra", "tāmrakāra", "pandai tĕmbaga"],
        "class": "metal",
        "english": "coppersmith / bronzesmith",
        "note": "Bronze casting for statues, bells, gongs"
    },

    # === FOOD/AGRICULTURE ===
    "tukang_masak": {
        "variants": ["maṅhapū", "juruhapū", "pahapuan"],
        "class": "food",
        "english": "cook / food processor",
        "note": "Communal/ritual food preparation"
    },
    "tukang_tuak": {
        "variants": ["maṅiduṁ", "aṅiduṁ", "maṅidu"],
        "class": "food",
        "english": "palm wine tapper",
        "note": "Taps palm sap for tuak/arak — organic economy"
    },

    # === GENERAL TERMS ===
    "warga_kilalan": {
        "variants": ["vargga kilalan", "varga kilalan", "vargga kilan"],
        "class": "category",
        "english": "taxed craft guilds (general term)",
        "note": "The sima section listing all exempt/taxed craftspeople"
    },
    "juru": {
        "variants": ["juru"],
        "class": "category",
        "english": "specialist / expert (prefix)",
        "note": "Generic craft prefix: juru + specialty"
    },
}

n_variants = sum(len(v["variants"]) for v in CRAFT_KEYWORDS.values())
print(f"\n[1] Craft dictionary: {len(CRAFT_KEYWORDS)} occupations, "
      f"{n_variants} variant forms")


# ═════════════════════════════════════════════════════════════════════════
# 2. SCAN CORPUS
# ═════════════════════════════════════════════════════════════════════════

print("\n[2] Scanning DHARMA corpus...")


def extract_text(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        texts = []
        for elem in root.iter():
            if elem.text:
                texts.append(elem.text.strip())
            if elem.tail:
                texts.append(elem.tail.strip())
        return " ".join(t for t in texts if t).lower()
    except Exception:
        return ""


# Load dates
dated_csv = REPO / "experiments" / "E030_prasasti_temporal_nlp" / "results" / "dated_inscriptions.csv"
date_lookup = {}
if dated_csv.exists():
    df_dates = pd.read_csv(dated_csv)
    for _, row in df_dates.iterrows():
        if pd.notna(row.get('year_ce')):
            date_lookup[row['filename']] = int(row['year_ce'])

# Scan
corpus = []
xml_files = sorted(DHARMA_DIR.glob("*.xml"))

for xml_path in xml_files:
    text = extract_text(xml_path)
    if not text:
        continue

    filename = xml_path.name
    year = date_lookup.get(filename)

    craft_hits = {}
    for craft_key, craft_info in CRAFT_KEYWORDS.items():
        matched = []
        for variant in craft_info["variants"]:
            if re.search(re.escape(variant.lower()), text):
                matched.append(variant)
        if matched:
            craft_hits[craft_key] = {
                "variants": matched,
                "class": craft_info["class"]
            }

    class_counts = Counter()
    for hit in craft_hits.values():
        if hit["class"] not in ("category",):
            class_counts[hit["class"]] += 1

    corpus.append({
        "filename": filename,
        "year_ce": year,
        "text_length": len(text),
        "craft_hits": craft_hits,
        "n_organic_crafts": class_counts.get("organic", 0),
        "n_lithic_crafts": class_counts.get("lithic", 0),
        "n_metal_crafts": class_counts.get("metal", 0),
        "n_food_crafts": class_counts.get("food", 0),
        "n_total_crafts": sum(class_counts.values()),
    })

n_with_crafts = sum(1 for c in corpus if c["n_total_crafts"] > 0)
print(f"  Scanned: {len(corpus)} inscriptions")
print(f"  With craft occupations: {n_with_crafts} ({n_with_crafts/len(corpus)*100:.1f}%)")


# ═════════════════════════════════════════════════════════════════════════
# 3. FREQUENCY TABLE
# ═════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("[3] CRAFT OCCUPATION FREQUENCY")
print(f"{'='*70}")

craft_counts = Counter()
for entry in corpus:
    for ck in entry["craft_hits"]:
        craft_counts[ck] += 1

print(f"\n  {'Occupation':<22} {'Count':>6} {'%':>7} {'Class':>10} {'English'}")
print("  " + "-" * 75)
for ck, count in craft_counts.most_common():
    pct = count / len(corpus) * 100
    info = CRAFT_KEYWORDS[ck]
    print(f"  {ck:<22} {count:>6} {pct:>6.1f}% {info['class']:>10} "
          f"{info['english'][:30]}")


# ═════════════════════════════════════════════════════════════════════════
# 4. CLASS COMPARISON
# ═════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("[4] CRAFT CLASS COMPARISON (excluding category terms)")
print(f"{'='*70}")

has_organic_craft = sum(1 for e in corpus if e["n_organic_crafts"] > 0)
has_lithic_craft = sum(1 for e in corpus if e["n_lithic_crafts"] > 0)
has_metal_craft = sum(1 for e in corpus if e["n_metal_crafts"] > 0)
has_food_craft = sum(1 for e in corpus if e["n_food_crafts"] > 0)

print(f"\n  Inscriptions mentioning ORGANIC craft workers: {has_organic_craft} "
      f"({has_organic_craft/len(corpus)*100:.1f}%)")
print(f"  Inscriptions mentioning LITHIC craft workers:  {has_lithic_craft} "
      f"({has_lithic_craft/len(corpus)*100:.1f}%)")
print(f"  Inscriptions mentioning METAL craft workers:   {has_metal_craft} "
      f"({has_metal_craft/len(corpus)*100:.1f}%)")
print(f"  Inscriptions mentioning FOOD craft workers:    {has_food_craft} "
      f"({has_food_craft/len(corpus)*100:.1f}%)")

# Total craft mentions by class
class_mention_totals = Counter()
for ck, count in craft_counts.items():
    cls = CRAFT_KEYWORDS[ck]["class"]
    if cls != "category":
        class_mention_totals[cls] += count

print(f"\n  CLASS TOTALS (inscription-craft pairs):")
for cls, total in class_mention_totals.most_common():
    print(f"    {cls:<12}: {total:>4}")

# Organic vs lithic craft ratio
org_total = class_mention_totals.get("organic", 0)
lit_total = class_mention_totals.get("lithic", 0)
if lit_total > 0:
    ratio = org_total / lit_total
    print(f"\n  Organic/Lithic craft ratio: {org_total}/{lit_total} = {ratio:.1f}x")
else:
    print(f"\n  Organic craft mentions: {org_total}, Lithic: 0")


# ═════════════════════════════════════════════════════════════════════════
# 5. UNDAHAGI (CARPENTER) DEEP DIVE
# ═════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("[5] UNDAHAGI (carpenter) — Most common craft occupation?")
print(f"{'='*70}")

undahagi_entries = [e for e in corpus if "undahagi" in e["craft_hits"]]
print(f"\n  Inscriptions mentioning undahagi: {len(undahagi_entries)}")

if undahagi_entries:
    dated_u = [e for e in undahagi_entries if e["year_ce"]]
    if dated_u:
        years = [e["year_ce"] for e in dated_u]
        print(f"  Dated: {len(dated_u)}, range: {min(years)}–{max(years)} CE")

    # Co-occurring crafts
    co_crafts = Counter()
    for entry in undahagi_entries:
        for ck in entry["craft_hits"]:
            if ck != "undahagi":
                co_crafts[ck] += 1
    if co_crafts:
        print(f"  Co-occurring occupations:")
        for ck, c in co_crafts.most_common(10):
            print(f"    {ck}: {c}")


# ═════════════════════════════════════════════════════════════════════════
# 6. PANDAI (SMITH) ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("[6] PANDAI (smith) — Metal worker frequency")
print(f"{'='*70}")

pandai_entries = [e for e in corpus if "pandai_besi" in e["craft_hits"]]
print(f"\n  Inscriptions mentioning pandai: {len(pandai_entries)}")


# ═════════════════════════════════════════════════════════════════════════
# 7. TEMPORAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("[7] TEMPORAL: Craft mentions over time")
print(f"{'='*70}")

dated_corpus = [e for e in corpus if e["year_ce"] and e["n_total_crafts"] > 0]
if dated_corpus:
    century_data = defaultdict(lambda: {"organic": 0, "lithic": 0, "metal": 0,
                                         "food": 0, "total": 0})
    dated_all = [e for e in corpus if e["year_ce"]]
    for entry in dated_all:
        c = (entry["year_ce"] // 100) + 1
        century_data[c]["total"] += 1
        if entry["n_organic_crafts"] > 0:
            century_data[c]["organic"] += 1
        if entry["n_lithic_crafts"] > 0:
            century_data[c]["lithic"] += 1
        if entry["n_metal_crafts"] > 0:
            century_data[c]["metal"] += 1

    print(f"\n  {'Century':>10} {'Total':>6} {'OrgCraft':>9} {'%':>5} "
          f"{'LitCraft':>9} {'%':>5} {'MetCraft':>9} {'%':>5}")
    print("  " + "-" * 65)
    for c in sorted(century_data.keys()):
        d = century_data[c]
        op = d["organic"]/d["total"]*100 if d["total"]>0 else 0
        lp = d["lithic"]/d["total"]*100 if d["total"]>0 else 0
        mp = d["metal"]/d["total"]*100 if d["total"]>0 else 0
        print(f"  C{c:>8} {d['total']:>6} {d['organic']:>9} {op:>4.0f}% "
              f"{d['lithic']:>9} {lp:>4.0f}% {d['metal']:>9} {mp:>4.0f}%")


# ═════════════════════════════════════════════════════════════════════════
# 8. SAVE
# ═════════════════════════════════════════════════════════════════════════

summary = {
    "experiment": "E040b_craft_occupations",
    "n_inscriptions": len(corpus),
    "n_with_crafts": n_with_crafts,
    "organic_craft_inscriptions": has_organic_craft,
    "lithic_craft_inscriptions": has_lithic_craft,
    "metal_craft_inscriptions": has_metal_craft,
    "craft_frequencies": {k: int(v) for k, v in craft_counts.most_common()},
    "class_totals": dict(class_mention_totals),
}

with open(str(RESULTS_DIR / "craft_occupation_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\nSaved: craft_occupation_summary.json")


# ═════════════════════════════════════════════════════════════════════════
# 9. VERDICT
# ═════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("VERDICT: ORGANIC ECONOMY WORKFORCE")
print(f"{'='*70}")

print(f"""
  CORPUS: {len(corpus)} inscriptions
  With craft occupations: {n_with_crafts} ({n_with_crafts/len(corpus)*100:.1f}%)

  ORGANIC craft workers: {has_organic_craft} inscriptions ({has_organic_craft/len(corpus)*100:.1f}%)
  LITHIC craft workers:  {has_lithic_craft} inscriptions ({has_lithic_craft/len(corpus)*100:.1f}%)
  METAL craft workers:   {has_metal_craft} inscriptions ({has_metal_craft/len(corpus)*100:.1f}%)
""")

if has_organic_craft > has_lithic_craft:
    print("  >>> ORGANIC WORKFORCE DOMINATES")
    print("  >>> The prasasti record an economy served primarily by")
    print("  >>> woodworkers, weavers, and fiber craftspeople —")
    print("  >>> confirming the organic material culture of E040.")
else:
    print("  >>> Mixed workforce — organic does not clearly dominate")

print(f"\n{'='*70}")
print("E040b COMPLETE")
print(f"{'='*70}")

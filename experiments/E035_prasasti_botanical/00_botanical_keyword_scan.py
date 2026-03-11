#!/usr/bin/env python3
"""
E035 — Prasasti Botanical Keyword Expansion
============================================
Question: Do Old Javanese/Malay inscriptions contain botanical terms
          relevant to mortuary, ritual, or economic practices?

Method:
  1. Scan 268 DHARMA XML inscriptions for botanical keywords
     (menyan/benzoin, kamboja/frangipani, cananga/ylang-ylang, etc.)
  2. Co-occurrence with ritual context (hyang, sraddha, sima, etc.)
  3. Temporal distribution of botanical mentions
  4. Economic vs ritual context classification

This extends I-008 (prasasti botanical keyword expansion) and feeds P5/P9.

Author: VOLCARCH project (AI-assisted)
Date: 2026-03-10
"""

import sys
import io
import os
import re
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).parent.parent.parent
DHARMA_DIR = REPO / "experiments" / "E023_ritual_screening" / "data" / "dharma" / "xml"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# TEI namespace
NS = {"tei": "http://www.tei-c.org/ns/1.0"}

print("=" * 70)
print("E035 — Prasasti Botanical Keyword Expansion")
print("=" * 70)

# ═════════════════════════════════════════════════════════════════════════
# 1. BOTANICAL KEYWORD DICTIONARY
# ═════════════════════════════════════════════════════════════════════════

# Keywords organized by plant/product and their cultural significance
BOTANICAL_KEYWORDS = {
    # === Ritual/Mortuary Plants ===
    "menyan": {
        "variants": ["menyan", "mĕnyan", "kĕmĕnyan", "kemenyan",
                      "kamĕnyan", "kammĕnyan"],
        "english": "benzoin/incense resin (Styrax benzoin)",
        "context": "ritual",
        "significance": "Primary ritual incense; Sumatra export; used in mortuary rites"
    },
    "kamboja": {
        "variants": ["kamboja", "kambhoja", "campaka", "campaka",
                      "campak", "cĕmpaka", "cempaka"],
        "english": "frangipani/plumeria (Plumeria spp.) and champak (Michelia champaca)",
        "context": "ritual",
        "significance": "Cemetery tree; mortuary association; flowers for offerings"
    },
    "cananga": {
        "variants": ["cananga", "kananga", "kĕnanga", "kenanga"],
        "english": "ylang-ylang (Cananga odorata)",
        "context": "ritual",
        "significance": "Ritual flower offerings; aromatic"
    },
    "melati": {
        "variants": ["mĕlati", "melati", "malati", "mālatī"],
        "english": "jasmine (Jasminum spp.)",
        "context": "ritual",
        "significance": "Offering flowers; purity symbol"
    },
    "padma": {
        "variants": ["padma", "utpala", "kumuda", "kamala",
                      "nalina", "tunjung", "tunjuṅ"],
        "english": "lotus (Nelumbo/Nymphaea spp.)",
        "context": "ritual",
        "significance": "Buddhist/Hindu iconography; ritual purity"
    },

    # === Economic/Trade Plants ===
    "padi": {
        "variants": ["padi", "pari", "śāli", "śali", "vrīhi",
                      "vrihi", "gabah", "bĕras", "beras"],
        "english": "rice (Oryza sativa)",
        "context": "economic",
        "significance": "Primary crop; tax/tribute medium; sawah agriculture"
    },
    "kelapa": {
        "variants": ["kalapa", "kĕlapa", "kelapa", "nyiur", "ñiur",
                      "nārikela", "narikela"],
        "english": "coconut (Cocos nucifera)",
        "context": "economic",
        "significance": "Multi-use crop; ritual offerings; coastal economy"
    },
    "sirih": {
        "variants": ["sirih", "sĕrĕh", "suruh", "tāmbūla", "tambula",
                      "pān", "nagavallī"],
        "english": "betel (Piper betle)",
        "context": "ritual",
        "significance": "Betel chewing; hospitality ritual; offering component"
    },
    "pinang": {
        "variants": ["pinang", "pinaṅ", "jambe", "jambĕ",
                      "pūga", "kramuka"],
        "english": "areca/betel nut (Areca catechu)",
        "context": "ritual",
        "significance": "Betel chewing companion; ritual offering"
    },
    "tebu": {
        "variants": ["tĕbu", "tebu", "ikṣu", "gula"],
        "english": "sugarcane (Saccharum officinarum)",
        "context": "economic",
        "significance": "Sweet; offering material; agricultural product"
    },
    "kapas": {
        "variants": ["kapas", "karpāsa", "karpasa"],
        "english": "cotton (Gossypium spp.)",
        "context": "economic",
        "significance": "Textile; tax medium; trade good"
    },

    # === Forest/Sacred Trees ===
    "waringin": {
        "variants": ["waringin", "waṅin", "waṅiṅ", "wariṅin",
                      "vata", "nyagrodha"],
        "english": "banyan (Ficus benghalensis/benjamina)",
        "context": "ritual",
        "significance": "Sacred tree; village marker; spirit dwelling"
    },
    "beringin": {
        "variants": ["bĕriṅin", "beringin"],
        "english": "banyan (variant name)",
        "context": "ritual",
        "significance": "Sacred tree; spirit dwelling"
    },
    "cendana": {
        "variants": ["candana", "cĕndana", "cendana"],
        "english": "sandalwood (Santalum album)",
        "context": "ritual",
        "significance": "Sacred wood; incense; Timor/NTT trade; mortuary use"
    },
    "kayu_putih": {
        "variants": ["kayuputi", "kayu puti", "kāyuputi"],
        "english": "cajuput/melaleuca (Melaleuca cajuputi)",
        "context": "economic",
        "significance": "Medicinal oil; eastern Indonesia"
    },
    "bambu": {
        "variants": ["bambu", "bulu", "pĕriṅ", "pering", "veṇu"],
        "english": "bamboo (Bambusoideae)",
        "context": "economic",
        "significance": "Construction; pre-Hindu building material hypothesis (I-040)"
    },
    "jati": {
        "variants": ["jati", "śāka", "sāgwan"],
        "english": "teak (Tectona grandis)",
        "context": "economic",
        "significance": "High-value timber; East Java forests"
    },

    # === Spice/Aromatic Plants ===
    "lada": {
        "variants": ["lada", "marica", "maricā"],
        "english": "pepper (Piper nigrum)",
        "context": "economic",
        "significance": "Major trade spice; Sumatra/Java"
    },
    "kapur_barus": {
        "variants": ["karpūra", "karpura", "kapur", "bhīmasena"],
        "english": "camphor (Dryobalanops aromatica)",
        "context": "ritual",
        "significance": "Ritual fumigant; Sumatra export; mortuary use"
    },
    "kunyit": {
        "variants": ["kunyit", "kuñit", "haridrā", "haridra"],
        "english": "turmeric (Curcuma longa)",
        "context": "ritual",
        "significance": "Ritual coloring; purification; offering ingredient"
    },
    "pala": {
        "variants": ["pala", "jātīphala", "jatiphala"],
        "english": "nutmeg (Myristica fragrans)",
        "context": "economic",
        "significance": "Maluku spice; trade"
    },
    "cengkeh": {
        "variants": ["cĕṅkĕh", "lavanga", "lavaṅga"],
        "english": "clove (Syzygium aromaticum)",
        "context": "economic",
        "significance": "Maluku spice; ritual incense additive"
    },
}

print(f"\n[1] Botanical dictionary: {len(BOTANICAL_KEYWORDS)} plant groups, "
      f"{sum(len(v['variants']) for v in BOTANICAL_KEYWORDS.values())} variant forms")


# ═════════════════════════════════════════════════════════════════════════
# 2. EXTRACT TEXT FROM DHARMA XML
# ═════════════════════════════════════════════════════════════════════════

print("\n[2] Extracting text from DHARMA XML corpus...")

# Ritual keywords for co-occurrence analysis
RITUAL_CONTEXT = {
    "hyang", "hyaṁ",  # sacred/divine
    "sīma", "sapatha",  # boundary oath
    "pūjā", "yajña",  # worship/sacrifice
    "śrāddha", "sraddha",  # death ritual
    "piṇḍa",  # ancestor offering
    "atīta",  # deceased
    "svarga", "svargga",  # heaven
    "pralīna",  # death
    "maṅhuri",  # ancestor return
    "kabuyutan",  # sacred ancestral site
    "wuku", "tithi", "nakṣatra",  # calendar
}


def extract_text(xml_path):
    """Extract all text content from TEI XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get all text from <body> and <div> elements
        texts = []
        for elem in root.iter():
            if elem.text:
                texts.append(elem.text.strip())
            if elem.tail:
                texts.append(elem.tail.strip())

        full_text = " ".join(t for t in texts if t)
        return full_text.lower()
    except Exception as e:
        return ""


def extract_date(xml_path):
    """Extract date from filename or XML metadata."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Try to find date in teiHeader
        for date_elem in root.iter('{http://www.tei-c.org/ns/1.0}date'):
            when = date_elem.get('when')
            if when:
                try:
                    return int(when[:4])
                except (ValueError, IndexError):
                    pass
            notBefore = date_elem.get('notBefore')
            notAfter = date_elem.get('notAfter')
            if notBefore and notAfter:
                try:
                    return (int(notBefore[:4]) + int(notAfter[:4])) // 2
                except (ValueError, IndexError):
                    pass
    except Exception:
        pass
    return None


# Load dates from E030 dated inscriptions if available
dated_csv = REPO / "experiments" / "E030_prasasti_temporal_nlp" / "results" / "dated_inscriptions.csv"
date_lookup = {}
if dated_csv.exists():
    df_dates = pd.read_csv(dated_csv)
    for _, row in df_dates.iterrows():
        if pd.notna(row.get('year_ce')):
            date_lookup[row['filename']] = int(row['year_ce'])
    print(f"  Loaded {len(date_lookup)} dates from E030")

# Scan all XML files
corpus = []
xml_files = sorted(DHARMA_DIR.glob("*.xml"))
print(f"  Scanning {len(xml_files)} XML files...")

for xml_path in xml_files:
    text = extract_text(xml_path)
    if not text:
        continue

    filename = xml_path.name

    # Get date
    year = date_lookup.get(filename) or extract_date(xml_path)

    # Search for botanical keywords
    botanical_hits = {}
    for plant_key, plant_info in BOTANICAL_KEYWORDS.items():
        matched_variants = []
        for variant in plant_info["variants"]:
            # Use word boundary-ish matching (handle Old Javanese morphology)
            pattern = re.escape(variant.lower())
            if re.search(pattern, text):
                matched_variants.append(variant)
        if matched_variants:
            botanical_hits[plant_key] = matched_variants

    # Check ritual context
    ritual_hits = set()
    for kw in RITUAL_CONTEXT:
        if kw.lower() in text:
            ritual_hits.add(kw)

    # Check if Borobudur (for sensitivity analysis)
    is_borobudur = 'borobudur' in filename.lower() or 'borobudur' in text

    corpus.append({
        "filename": filename,
        "year_ce": year,
        "text_length": len(text),
        "botanical_hits": botanical_hits,
        "n_botanical": len(botanical_hits),
        "ritual_hits": ritual_hits,
        "n_ritual": len(ritual_hits),
        "has_ritual_context": len(ritual_hits) > 0,
        "is_borobudur": is_borobudur,
        "text_snippet": text[:200],
    })

print(f"  Processed: {len(corpus)} inscriptions")
n_with_botanical = sum(1 for c in corpus if c['n_botanical'] > 0)
print(f"  With botanical content: {n_with_botanical} ({n_with_botanical/len(corpus)*100:.1f}%)")


# ═════════════════════════════════════════════════════════════════════════
# 3. ANALYSIS A: Plant Frequency
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[3] ANALYSIS A: Botanical Keyword Frequency")
print("=" * 70)

plant_counts = Counter()
plant_inscriptions = defaultdict(list)

for entry in corpus:
    for plant_key in entry['botanical_hits']:
        plant_counts[plant_key] += 1
        plant_inscriptions[plant_key].append(entry['filename'])

print(f"\n  {'Plant':<20} {'Count':>6} {'%':>8} {'Context':>10} {'English'}")
print("  " + "-" * 75)
for plant_key, count in plant_counts.most_common():
    pct = count / len(corpus) * 100
    info = BOTANICAL_KEYWORDS[plant_key]
    print(f"  {plant_key:<20} {count:>6} {pct:>7.1f}% {info['context']:>10} "
          f"{info['english'][:40]}")

# Aggregate by context type
ritual_plants = sum(c for k, c in plant_counts.items()
                    if BOTANICAL_KEYWORDS[k]['context'] == 'ritual')
economic_plants = sum(c for k, c in plant_counts.items()
                      if BOTANICAL_KEYWORDS[k]['context'] == 'economic')
total_mentions = sum(plant_counts.values())
print(f"\n  Total botanical mentions: {total_mentions}")
print(f"    Ritual plants: {ritual_plants} ({ritual_plants/max(total_mentions,1)*100:.1f}%)")
print(f"    Economic plants: {economic_plants} ({economic_plants/max(total_mentions,1)*100:.1f}%)")


# ═════════════════════════════════════════════════════════════════════════
# 4. ANALYSIS B: Co-occurrence with Ritual Context
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[4] ANALYSIS B: Botanical × Ritual Co-occurrence")
print("=" * 70)

# For each plant, what % of inscriptions mentioning it also have ritual keywords?
print(f"\n  {'Plant':<20} {'Total':>6} {'w/Ritual':>10} {'%':>8}")
print("  " + "-" * 50)

for plant_key, count in plant_counts.most_common():
    if count < 1:
        continue
    ritual_co = sum(1 for entry in corpus
                    if plant_key in entry['botanical_hits'] and entry['has_ritual_context'])
    pct = ritual_co / count * 100
    print(f"  {plant_key:<20} {count:>6} {ritual_co:>10} {pct:>7.1f}%")

# Overall: do botanical inscriptions have MORE ritual content?
bot_ritual = sum(1 for e in corpus if e['n_botanical'] > 0 and e['has_ritual_context'])
bot_total = sum(1 for e in corpus if e['n_botanical'] > 0)
no_bot_ritual = sum(1 for e in corpus if e['n_botanical'] == 0 and e['has_ritual_context'])
no_bot_total = sum(1 for e in corpus if e['n_botanical'] == 0)

if bot_total > 0 and no_bot_total > 0:
    print(f"\n  Botanical inscriptions with ritual context: "
          f"{bot_ritual}/{bot_total} ({bot_ritual/bot_total*100:.1f}%)")
    print(f"  Non-botanical inscriptions with ritual context: "
          f"{no_bot_ritual}/{no_bot_total} ({no_bot_ritual/no_bot_total*100:.1f}%)")

    # Fisher's exact or chi-squared
    from scipy import stats
    contingency = [[bot_ritual, bot_total - bot_ritual],
                   [no_bot_ritual, no_bot_total - no_bot_ritual]]
    odds, fisher_p = stats.fisher_exact(contingency)
    print(f"  Fisher's exact: OR={odds:.2f}, p={fisher_p:.4f}")


# ═════════════════════════════════════════════════════════════════════════
# 5. ANALYSIS C: Temporal Distribution
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[5] ANALYSIS C: Temporal Distribution of Botanical Mentions")
print("=" * 70)

# Filter to dated inscriptions
dated_botanical = [e for e in corpus if e['year_ce'] and e['n_botanical'] > 0]
dated_all = [e for e in corpus if e['year_ce']]

print(f"\n  Dated inscriptions: {len(dated_all)}")
print(f"  Dated with botanical: {len(dated_botanical)}")

if dated_botanical:
    # Century-level analysis
    century_bot = Counter()
    century_total = Counter()

    for entry in dated_all:
        century = (entry['year_ce'] // 100) + 1
        century_total[century] += 1
        if entry['n_botanical'] > 0:
            century_bot[century] += 1

    print(f"\n  {'Century':>10} {'Botanical':>10} {'Total':>8} {'%':>8}")
    print("  " + "-" * 40)
    for c in sorted(century_total.keys()):
        bot = century_bot.get(c, 0)
        total = century_total[c]
        pct = bot / total * 100 if total > 0 else 0
        print(f"  C{c:>8} {bot:>10} {total:>8} {pct:>7.1f}%")

    # List specific botanical inscriptions with dates
    print(f"\n  Botanical inscriptions (dated):")
    print(f"  {'Year':>6} {'Filename':<45} {'Plants'}")
    print("  " + "-" * 80)
    for entry in sorted(dated_botanical, key=lambda x: x['year_ce']):
        plants = ", ".join(entry['botanical_hits'].keys())
        print(f"  {entry['year_ce']:>6} {entry['filename']:<45} {plants}")


# ═════════════════════════════════════════════════════════════════════════
# 6. ANALYSIS D: Specific Plant Deep-Dives
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[6] ANALYSIS D: Plant-Specific Deep Dives")
print("=" * 70)

# For P5: menyan (benzoin) — key mortuary incense
print("\n  --- MENYAN (benzoin/incense) ---")
menyan_entries = [e for e in corpus if 'menyan' in e['botanical_hits']]
if menyan_entries:
    for entry in menyan_entries:
        variants = entry['botanical_hits']['menyan']
        print(f"  {entry['filename']}")
        print(f"    Year: {entry['year_ce'] or 'undated'}")
        print(f"    Matched variants: {', '.join(variants)}")
        print(f"    Ritual context: {', '.join(entry['ritual_hits']) if entry['ritual_hits'] else 'none'}")
        print(f"    Snippet: {entry['text_snippet'][:100]}...")
        print()
else:
    print("  No menyan hits in corpus")

# For P5: kamboja/campaka (frangipani/champak)
print("\n  --- KAMBOJA/CAMPAKA (frangipani/champak) ---")
kamboja_entries = [e for e in corpus if 'kamboja' in e['botanical_hits']]
if kamboja_entries:
    for entry in kamboja_entries:
        variants = entry['botanical_hits']['kamboja']
        print(f"  {entry['filename']}")
        print(f"    Year: {entry['year_ce'] or 'undated'}")
        print(f"    Matched variants: {', '.join(variants)}")
        print(f"    Ritual context: {', '.join(entry['ritual_hits']) if entry['ritual_hits'] else 'none'}")
        print(f"    Snippet: {entry['text_snippet'][:100]}...")
        print()
else:
    print("  No kamboja/campaka hits in corpus")

# For P5: cendana (sandalwood) — mortuary wood
print("\n  --- CENDANA (sandalwood) ---")
cendana_entries = [e for e in corpus if 'cendana' in e['botanical_hits']]
if cendana_entries:
    for entry in cendana_entries[:5]:  # limit output
        variants = entry['botanical_hits']['cendana']
        print(f"  {entry['filename']}")
        print(f"    Year: {entry['year_ce'] or 'undated'}")
        print(f"    Matched variants: {', '.join(variants)}")
        print(f"    Ritual context: {', '.join(entry['ritual_hits']) if entry['ritual_hits'] else 'none'}")
        print()
    if len(cendana_entries) > 5:
        print(f"  ... and {len(cendana_entries)-5} more")
else:
    print("  No cendana hits in corpus")

# For P9: sirih/pinang (betel complex) — social ritual
print("\n  --- SIRIH/PINANG (betel complex) ---")
betel_entries = [e for e in corpus if 'sirih' in e['botanical_hits'] or 'pinang' in e['botanical_hits']]
if betel_entries:
    for entry in betel_entries[:5]:
        plants = {k: v for k, v in entry['botanical_hits'].items() if k in ('sirih', 'pinang')}
        print(f"  {entry['filename']}")
        print(f"    Year: {entry['year_ce'] or 'undated'}")
        print(f"    Plants: {plants}")
        print(f"    Ritual context: {', '.join(entry['ritual_hits']) if entry['ritual_hits'] else 'none'}")
        print()
    if len(betel_entries) > 5:
        print(f"  ... and {len(betel_entries)-5} more")
else:
    print("  No sirih/pinang hits in corpus")

# For I-040: bambu (bamboo civilization)
print("\n  --- BAMBU (bamboo) ---")
bambu_entries = [e for e in corpus if 'bambu' in e['botanical_hits']]
if bambu_entries:
    for entry in bambu_entries[:5]:
        variants = entry['botanical_hits']['bambu']
        print(f"  {entry['filename']}")
        print(f"    Year: {entry['year_ce'] or 'undated'}")
        print(f"    Matched variants: {', '.join(variants)}")
        print()
    if len(bambu_entries) > 5:
        print(f"  ... and {len(bambu_entries)-5} more")
else:
    print("  No bambu hits in corpus")


# ═════════════════════════════════════════════════════════════════════════
# 7. VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[7] Generating visualizations...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('E035 — Prasasti Botanical Keywords\n'
             'Plants in Old Javanese/Malay Inscriptions',
             fontsize=14, fontweight='bold', y=0.98)

# Panel A: Plant frequency bar chart
ax1 = axes[0, 0]
if plant_counts:
    plants_sorted = plant_counts.most_common(15)
    names = [p[0] for p in plants_sorted]
    counts = [p[1] for p in plants_sorted]
    colors = ['#e74c3c' if BOTANICAL_KEYWORDS[n]['context'] == 'ritual' else '#3498db'
              for n in names]
    bars = ax1.barh(range(len(names)), counts, color=colors, edgecolor='white')
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel('Number of Inscriptions')
    ax1.set_title('A. Plant Frequency in Prasasti', fontsize=11)
    ax1.invert_yaxis()
    # Legend
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color='#e74c3c', label='Ritual'),
                        Patch(color='#3498db', label='Economic')],
               fontsize=8, loc='lower right')
else:
    ax1.text(0.5, 0.5, 'No botanical keywords found', ha='center', va='center')
    ax1.set_title('A. Plant Frequency', fontsize=11)

# Panel B: Temporal distribution
ax2 = axes[0, 1]
if dated_botanical and century_total:
    centuries = sorted(century_total.keys())
    bot_pcts = [century_bot.get(c, 0) / century_total[c] * 100
                if century_total[c] > 0 else 0 for c in centuries]
    bot_counts_c = [century_bot.get(c, 0) for c in centuries]
    total_counts_c = [century_total[c] for c in centuries]

    x = range(len(centuries))
    width = 0.35
    ax2.bar([i - width/2 for i in x], total_counts_c, width,
            color='#bdc3c7', label='All dated', edgecolor='white')
    ax2.bar([i + width/2 for i in x], bot_counts_c, width,
            color='#27ae60', label='With botanical', edgecolor='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'C{c}' for c in centuries], fontsize=9)
    ax2.set_xlabel('Century CE')
    ax2.set_ylabel('Number of Inscriptions')
    ax2.set_title('B. Botanical Mentions Over Time', fontsize=11)
    ax2.legend(fontsize=8)
else:
    ax2.text(0.5, 0.5, 'Insufficient dated data', ha='center', va='center')
    ax2.set_title('B. Temporal Distribution', fontsize=11)

# Panel C: Ritual vs non-ritual co-occurrence
ax3 = axes[1, 0]
if bot_total > 0:
    categories = ['With Botanical\nKeywords', 'Without Botanical\nKeywords']
    ritual_pcts = [bot_ritual/bot_total*100 if bot_total > 0 else 0,
                   no_bot_ritual/no_bot_total*100 if no_bot_total > 0 else 0]
    bars = ax3.bar(categories, ritual_pcts,
                   color=['#27ae60', '#95a5a6'], edgecolor='white')
    ax3.set_ylabel('% with Ritual Keywords')
    ax3.set_title('C. Ritual Context Co-occurrence', fontsize=11)
    for bar, pct in zip(bars, ritual_pcts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{pct:.1f}%', ha='center', fontsize=10)
    ax3.set_ylim(0, max(ritual_pcts) * 1.2 if max(ritual_pcts) > 0 else 100)
else:
    ax3.text(0.5, 0.5, 'No botanical keywords found', ha='center', va='center')
    ax3.set_title('C. Ritual Co-occurrence', fontsize=11)

# Panel D: Context pie chart
ax4 = axes[1, 1]
if total_mentions > 0:
    context_counts = Counter()
    for plant_key, count in plant_counts.items():
        ctx = BOTANICAL_KEYWORDS[plant_key]['context']
        context_counts[ctx] += count

    labels = list(context_counts.keys())
    sizes = list(context_counts.values())
    colors_pie = ['#e74c3c' if l == 'ritual' else '#3498db' for l in labels]
    ax4.pie(sizes, labels=[f'{l}\n({s})' for l, s in zip(labels, sizes)],
            colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax4.set_title(f'D. Ritual vs Economic Plants\n(n={total_mentions} mentions)',
                  fontsize=11)
else:
    ax4.text(0.5, 0.5, 'No data', ha='center', va='center')
    ax4.set_title('D. Context Distribution', fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(str(RESULTS_DIR / 'botanical_4panel.png'), dpi=150, bbox_inches='tight')
print("  Saved: botanical_4panel.png")
plt.close('all')


# ═════════════════════════════════════════════════════════════════════════
# 8. STRUCTURED OUTPUT
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[8] Saving results...")
print("=" * 70)

# Summary JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, set): return list(obj)
        return super().default(obj)

summary = {
    "experiment": "E035_prasasti_botanical",
    "n_inscriptions_scanned": len(corpus),
    "n_with_botanical": n_with_botanical,
    "pct_with_botanical": round(n_with_botanical / len(corpus) * 100, 1),
    "plant_frequencies": dict(plant_counts.most_common()),
    "total_mentions": total_mentions,
    "ritual_plants_pct": round(ritual_plants / max(total_mentions, 1) * 100, 1),
    "economic_plants_pct": round(economic_plants / max(total_mentions, 1) * 100, 1),
}

if bot_total > 0 and no_bot_total > 0:
    summary["cooccurrence"] = {
        "botanical_with_ritual_pct": round(bot_ritual / bot_total * 100, 1),
        "nonbotanical_with_ritual_pct": round(no_bot_ritual / no_bot_total * 100, 1),
        "fisher_OR": round(odds, 2),
        "fisher_p": round(fisher_p, 4),
    }

with open(str(RESULTS_DIR / 'botanical_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

# Detailed CSV
rows = []
for entry in corpus:
    if entry['n_botanical'] > 0:
        plants = "|".join(entry['botanical_hits'].keys())
        variants = "|".join(
            ",".join(v) for v in entry['botanical_hits'].values()
        )
        ritual = "|".join(entry['ritual_hits'])
        rows.append({
            "filename": entry['filename'],
            "year_ce": entry['year_ce'] or '',
            "text_length": entry['text_length'],
            "plants": plants,
            "variants_matched": variants,
            "n_plants": entry['n_botanical'],
            "ritual_keywords": ritual,
            "has_ritual_context": entry['has_ritual_context'],
        })

df_out = pd.DataFrame(rows)
df_out.to_csv(str(RESULTS_DIR / 'botanical_inscriptions.csv'), index=False)

print("  Saved: botanical_summary.json")
print("  Saved: botanical_inscriptions.csv")


# ═════════════════════════════════════════════════════════════════════════
# 9. HEADLINE
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("HEADLINE FINDING")
print("=" * 70)

print(f"""
  CORPUS: {len(corpus)} inscriptions scanned
  BOTANICAL HITS: {n_with_botanical} inscriptions ({n_with_botanical/len(corpus)*100:.1f}%)
  TOTAL MENTIONS: {total_mentions} across {len(plant_counts)} plant types
  RITUAL PLANTS: {ritual_plants} ({ritual_plants/max(total_mentions,1)*100:.1f}%)
  ECONOMIC PLANTS: {economic_plants} ({economic_plants/max(total_mentions,1)*100:.1f}%)
""")

if plant_counts:
    print("  TOP 5 PLANTS:")
    for plant, count in plant_counts.most_common(5):
        info = BOTANICAL_KEYWORDS[plant]
        print(f"    {plant}: {count} inscriptions — {info['english']}")

print("\n" + "=" * 70)
print("E035 COMPLETE")
print("=" * 70)

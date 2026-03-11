"""
E040: Bamboo Civilization Hypothesis — Material Culture in Prasasti
====================================================================
Hypothesis (I-040): Pre-Hindu Nusantara intentionally built non-lithic
(bambu/kayu). If the epigraphic record mentions organic materials MORE
than lithic ones, the archaeological 'blank' is a preservation bias,
not an absence of civilization.

Method:
  1. Scan 268 DHARMA XML inscriptions for material-culture keywords
  2. Classify: ORGANIC (wood, bamboo, thatch) vs LITHIC (stone, brick) vs METAL
  3. Count mentions, co-occurrence with building/construction context
  4. Temporal analysis: do organic mentions decline as Indianization proceeds?

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
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).parent.parent.parent
DHARMA_DIR = REPO / "experiments" / "E023_ritual_screening" / "data" / "dharma" / "xml"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("E040 — Bamboo Civilization: Material Culture in Prasasti")
print("=" * 70)

# ═════════════════════════════════════════════════════════════════════════
# 1. MATERIAL CULTURE KEYWORD DICTIONARY
# ═════════════════════════════════════════════════════════════════════════

MATERIAL_KEYWORDS = {
    # === ORGANIC (perishable) ===
    "kayu": {
        "variants": ["kayu", "kāyu", "taru", "dāru"],
        "class": "organic",
        "english": "wood/timber",
        "note": "Generic wood; taru = tree (Skt); daru = timber (Skt)"
    },
    "bambu": {
        "variants": ["bambu", "bulu", "pĕriṅ", "pering", "veṇu", "venu",
                      "tinaṅ", "buluh"],
        "class": "organic",
        "english": "bamboo",
        "note": "bulu = bamboo (OJ); venu = bamboo (Skt); pering = bamboo (OJ)"
    },
    "rumah_kayu": {
        "variants": ["rumah", "umah", "grha", "gṛha"],
        "class": "building",
        "english": "house/dwelling",
        "note": "Generic dwelling — may be organic or lithic"
    },
    "atap": {
        "variants": ["atĕp", "atep", "alang", "alaṅ", "alang-alang",
                      "rĕrĕk", "rerek"],
        "class": "organic",
        "english": "roof/thatch (alang-alang grass)",
        "note": "alang-alang = imperata grass roofing"
    },
    "ijuk": {
        "variants": ["ijuk", "arĕn", "aren", "tāla"],
        "class": "organic",
        "english": "palm fiber / sugar palm",
        "note": "ijuk = palm fiber for rope/roofing; aren = sugar palm"
    },
    "rotan": {
        "variants": ["rotan", "huwi", "huvi"],
        "class": "organic",
        "english": "rattan",
        "note": "Binding material; furniture; pre-nail construction"
    },
    "daun": {
        "variants": ["don", "ron", "parṇa", "parna", "patra"],
        "class": "organic",
        "english": "leaf / thatch leaf",
        "note": "don/ron = leaf (OJ); parna = leaf (Skt)"
    },
    "jati": {
        "variants": ["jāti", "jati"],
        "class": "organic",
        "english": "teak (Tectona grandis)",
        "note": "High-value timber; East Java"
    },

    # === LITHIC (durable) ===
    "batu": {
        "variants": ["watu", "batu", "śilā", "sila", "pāṣāṇa", "pasana",
                      "aśman", "asman"],
        "class": "lithic",
        "english": "stone",
        "note": "watu = stone (OJ); sila = stone slab (Skt)"
    },
    "bata": {
        "variants": ["bata", "iṣṭikā", "istika", "iṣṭaka"],
        "class": "lithic",
        "english": "brick",
        "note": "Fired clay brick; major temple material"
    },
    "candi": {
        "variants": ["candi", "caṇḍi", "candī"],
        "class": "lithic",
        "english": "temple/shrine",
        "note": "Stone/brick religious structure"
    },
    "prasada": {
        "variants": ["prāsāda", "prasada", "prasāda"],
        "class": "lithic",
        "english": "tower temple / palace",
        "note": "Major stone structure (Skt)"
    },
    "mandapa": {
        "variants": ["maṇḍapa", "mandapa", "maṇḍapā"],
        "class": "lithic",
        "english": "pavilion / hall",
        "note": "Open-sided hall, usually stone"
    },
    "stambha": {
        "variants": ["stambha", "sthambha", "tugu"],
        "class": "lithic",
        "english": "pillar / column",
        "note": "Stone pillar; tugu = Javanese pillar"
    },

    # === METAL ===
    "emas": {
        "variants": ["mas", "ĕmas", "emas", "suvarṇa", "suvarna",
                      "hiraṇya", "hiranya", "kāñcana", "kancana"],
        "class": "metal",
        "english": "gold",
        "note": "mas = gold (OJ); very common in tax/tribute contexts"
    },
    "perak": {
        "variants": ["pirak", "salaka", "rūpya", "rupya", "rajata"],
        "class": "metal",
        "english": "silver",
        "note": "pirak = silver (OJ); currency unit"
    },
    "tembaga": {
        "variants": ["tĕmbaga", "tembaga", "tāmra", "tamra", "loha"],
        "class": "metal",
        "english": "copper / bronze",
        "note": "tamra = copper (Skt); loha = metal (Skt)"
    },
    "besi": {
        "variants": ["wĕsi", "wesi", "ayas", "lohā"],
        "class": "metal",
        "english": "iron",
        "note": "wesi = iron (OJ); ayas = metal/iron (Skt)"
    },
    "timah": {
        "variants": ["timah", "trapu", "vaṅga"],
        "class": "metal",
        "english": "tin / lead",
        "note": "Important trade metal"
    },

    # === CONSTRUCTION ACTIVITY ===
    "bangun": {
        "variants": ["waṅun", "wangun", "bangun", "pinarĕk", "pinarek"],
        "class": "activity",
        "english": "build / construct / erect",
        "note": "wangun = to build (OJ)"
    },
    "pahat": {
        "variants": ["pahat", "tĕkĕs", "tekes", "ukir"],
        "class": "activity",
        "english": "carve / chisel",
        "note": "Stone-working terminology"
    },
    "sima": {
        "variants": ["sīma", "sima"],
        "class": "institution",
        "english": "tax-free domain / freehold",
        "note": "Land grant — the primary prasasti genre"
    },
}

n_variants = sum(len(v["variants"]) for v in MATERIAL_KEYWORDS.values())
print(f"\n[1] Material dictionary: {len(MATERIAL_KEYWORDS)} categories, "
      f"{n_variants} variant forms")
for cls in ["organic", "lithic", "metal", "building", "activity", "institution"]:
    n = sum(1 for v in MATERIAL_KEYWORDS.values() if v["class"] == cls)
    print(f"    {cls}: {n} categories")


# ═════════════════════════════════════════════════════════════════════════
# 2. EXTRACT TEXT FROM DHARMA XML
# ═════════════════════════════════════════════════════════════════════════

print("\n[2] Extracting text from DHARMA XML corpus...")


def extract_text(xml_path):
    """Extract all text content from TEI XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        texts = []
        for elem in root.iter():
            if elem.text:
                texts.append(elem.text.strip())
            if elem.tail:
                texts.append(elem.tail.strip())
        full_text = " ".join(t for t in texts if t)
        return full_text.lower()
    except Exception:
        return ""


def extract_title(xml_path):
    """Extract inscription title from TEI XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        title_elem = root.find('.//tei:titleStmt/tei:title', ns)
        if title_elem is not None and title_elem.text:
            return title_elem.text.strip()
    except Exception:
        pass
    return xml_path.stem


# Load dates from E030
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
    title = extract_title(xml_path)
    year = date_lookup.get(filename)

    # Count material keyword matches
    material_hits = {}
    for mat_key, mat_info in MATERIAL_KEYWORDS.items():
        matched_variants = []
        for variant in mat_info["variants"]:
            pattern = re.escape(variant.lower())
            if re.search(pattern, text):
                matched_variants.append(variant)
        if matched_variants:
            material_hits[mat_key] = {
                "variants": matched_variants,
                "class": mat_info["class"]
            }

    # Count by class
    class_counts = Counter()
    for hit_info in material_hits.values():
        class_counts[hit_info["class"]] += 1

    corpus.append({
        "filename": filename,
        "title": title[:80],
        "year_ce": year,
        "text_length": len(text),
        "material_hits": material_hits,
        "class_counts": dict(class_counts),
        "n_organic": class_counts.get("organic", 0),
        "n_lithic": class_counts.get("lithic", 0),
        "n_metal": class_counts.get("metal", 0),
        "n_total": len(material_hits),
    })

print(f"  Processed: {len(corpus)} inscriptions")
n_with_material = sum(1 for c in corpus if c['n_total'] > 0)
print(f"  With material keywords: {n_with_material} ({n_with_material/len(corpus)*100:.1f}%)")


# ═════════════════════════════════════════════════════════════════════════
# 3. OVERALL FREQUENCY
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[3] MATERIAL KEYWORD FREQUENCY")
print("=" * 70)

mat_counts = Counter()
for entry in corpus:
    for mat_key in entry['material_hits']:
        mat_counts[mat_key] += 1

print(f"\n  {'Keyword':<18} {'Count':>6} {'%':>7} {'Class':>10} {'English'}")
print("  " + "-" * 75)
for mat_key, count in mat_counts.most_common():
    pct = count / len(corpus) * 100
    info = MATERIAL_KEYWORDS[mat_key]
    print(f"  {mat_key:<18} {count:>6} {pct:>6.1f}% {info['class']:>10} "
          f"{info['english'][:30]}")

# Aggregate by class
print(f"\n  CLASS TOTALS (unique inscription-keyword pairs):")
class_totals = Counter()
for mat_key, count in mat_counts.items():
    class_totals[MATERIAL_KEYWORDS[mat_key]["class"]] += count

for cls, total in class_totals.most_common():
    print(f"    {cls:<12}: {total:>4} mentions")


# ═════════════════════════════════════════════════════════════════════════
# 4. ORGANIC vs LITHIC COMPARISON (THE CORE TEST)
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[4] CORE TEST: Organic vs Lithic Material References")
print("=" * 70)

# Count inscriptions with organic vs lithic mentions
has_organic = sum(1 for e in corpus if e['n_organic'] > 0)
has_lithic = sum(1 for e in corpus if e['n_lithic'] > 0)
has_metal = sum(1 for e in corpus if e['n_metal'] > 0)
has_both = sum(1 for e in corpus if e['n_organic'] > 0 and e['n_lithic'] > 0)
has_organic_only = sum(1 for e in corpus if e['n_organic'] > 0 and e['n_lithic'] == 0)
has_lithic_only = sum(1 for e in corpus if e['n_lithic'] > 0 and e['n_organic'] == 0)

print(f"\n  Inscriptions mentioning ORGANIC materials: {has_organic} "
      f"({has_organic/len(corpus)*100:.1f}%)")
print(f"  Inscriptions mentioning LITHIC materials:  {has_lithic} "
      f"({has_lithic/len(corpus)*100:.1f}%)")
print(f"  Inscriptions mentioning METAL:             {has_metal} "
      f"({has_metal/len(corpus)*100:.1f}%)")
print(f"  Both organic + lithic:                     {has_both}")
print(f"  Organic ONLY (no lithic):                  {has_organic_only}")
print(f"  Lithic ONLY (no organic):                  {has_lithic_only}")

# Organic keyword breakdown
print(f"\n  ORGANIC breakdown:")
organic_keys = [k for k, v in MATERIAL_KEYWORDS.items() if v["class"] == "organic"]
for ok in organic_keys:
    c = mat_counts.get(ok, 0)
    if c > 0:
        print(f"    {ok:<18}: {c:>4} inscriptions ({c/len(corpus)*100:.1f}%)")

print(f"\n  LITHIC breakdown:")
lithic_keys = [k for k, v in MATERIAL_KEYWORDS.items() if v["class"] == "lithic"]
for lk in lithic_keys:
    c = mat_counts.get(lk, 0)
    if c > 0:
        print(f"    {lk:<18}: {c:>4} inscriptions ({c/len(corpus)*100:.1f}%)")

# Sign test: for inscriptions with BOTH, which has more variety?
print(f"\n  PAIRED COMPARISON (inscriptions with both organic + lithic):")
organic_wins = sum(1 for e in corpus
                   if e['n_organic'] > 0 and e['n_lithic'] > 0
                   and e['n_organic'] > e['n_lithic'])
lithic_wins = sum(1 for e in corpus
                  if e['n_organic'] > 0 and e['n_lithic'] > 0
                  and e['n_lithic'] > e['n_organic'])
ties = has_both - organic_wins - lithic_wins
print(f"    Organic > Lithic: {organic_wins}")
print(f"    Lithic > Organic: {lithic_wins}")
print(f"    Ties: {ties}")
if organic_wins + lithic_wins > 0:
    binom_p = stats.binomtest(organic_wins, organic_wins + lithic_wins, 0.5).pvalue
    print(f"    Binomial test (H0: equal): p = {binom_p:.4f}")


# ═════════════════════════════════════════════════════════════════════════
# 5. TEMPORAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[5] TEMPORAL ANALYSIS: Do organic mentions decline over time?")
print("=" * 70)

dated = [e for e in corpus if e['year_ce']]
print(f"\n  Dated inscriptions: {len(dated)}")

if len(dated) > 20:
    # Century-level
    century_data = defaultdict(lambda: {"organic": 0, "lithic": 0, "metal": 0, "total": 0})
    for entry in dated:
        c = (entry['year_ce'] // 100) + 1
        century_data[c]["total"] += 1
        if entry['n_organic'] > 0:
            century_data[c]["organic"] += 1
        if entry['n_lithic'] > 0:
            century_data[c]["lithic"] += 1
        if entry['n_metal'] > 0:
            century_data[c]["metal"] += 1

    print(f"\n  {'Century':>10} {'Total':>6} {'Organic':>8} {'%':>6} "
          f"{'Lithic':>8} {'%':>6} {'Metal':>8} {'%':>6}")
    print("  " + "-" * 70)
    for c in sorted(century_data.keys()):
        d = century_data[c]
        org_pct = d["organic"] / d["total"] * 100 if d["total"] > 0 else 0
        lit_pct = d["lithic"] / d["total"] * 100 if d["total"] > 0 else 0
        met_pct = d["metal"] / d["total"] * 100 if d["total"] > 0 else 0
        print(f"  C{c:>8} {d['total']:>6} {d['organic']:>8} {org_pct:>5.0f}% "
              f"{d['lithic']:>8} {lit_pct:>5.0f}% {d['metal']:>8} {met_pct:>5.0f}%")

    # Organic ratio over time
    org_ratio = []
    for entry in dated:
        if entry['n_organic'] + entry['n_lithic'] > 0:
            ratio = entry['n_organic'] / (entry['n_organic'] + entry['n_lithic'])
            org_ratio.append((entry['year_ce'], ratio))

    if len(org_ratio) > 10:
        years = [x[0] for x in org_ratio]
        ratios = [x[1] for x in org_ratio]
        rho, p = stats.spearmanr(years, ratios)
        print(f"\n  Organic/(Organic+Lithic) ratio vs Year:")
        print(f"    n = {len(org_ratio)} inscriptions with at least one organic or lithic mention")
        print(f"    Spearman rho = {rho:.3f}, p = {p:.4f}")
        print(f"    VCS prediction: NEGATIVE (organic declines as stone-building Indianization proceeds)")
        if rho < 0 and p < 0.05:
            print(f"    >>> SUPPORTS hypothesis: organic mentions decline over time")
        elif rho > 0 and p < 0.05:
            print(f"    >>> OPPOSITE: organic mentions INCREASE over time")
        else:
            print(f"    >>> NOT SIGNIFICANT")


# ═════════════════════════════════════════════════════════════════════════
# 6. SPECIFIC MATERIAL DEEP DIVES
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[6] DEEP DIVES: Specific Materials")
print("=" * 70)

for material_key in ["bambu", "kayu", "batu", "besi", "emas"]:
    entries = [e for e in corpus if material_key in e['material_hits']]
    info = MATERIAL_KEYWORDS[material_key]
    print(f"\n  --- {material_key.upper()} ({info['english']}) — {len(entries)} inscriptions ---")
    for entry in entries[:5]:
        variants = entry['material_hits'][material_key]['variants']
        year = entry['year_ce'] or 'undated'
        other_mats = [k for k in entry['material_hits'] if k != material_key]
        print(f"    {entry['filename'][:45]}")
        print(f"      Year: {year}, Variants: {', '.join(variants)}, "
              f"Co-materials: {', '.join(other_mats) if other_mats else 'none'}")
    if len(entries) > 5:
        print(f"    ... and {len(entries) - 5} more")


# ═════════════════════════════════════════════════════════════════════════
# 7. METAL ECONOMY (bonus)
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[7] METAL ECONOMY: Gold vs Silver vs Copper vs Iron")
print("=" * 70)

metal_keys = ["emas", "perak", "tembaga", "besi", "timah"]
for mk in metal_keys:
    c = mat_counts.get(mk, 0)
    info = MATERIAL_KEYWORDS[mk]
    print(f"  {mk:<12} ({info['english']:<15}): {c:>4} inscriptions ({c/len(corpus)*100:.1f}%)")

# Gold vs Iron ratio — higher gold = more trade/elite context, higher iron = more production
gold_n = mat_counts.get("emas", 0)
iron_n = mat_counts.get("besi", 0)
if gold_n + iron_n > 0:
    print(f"\n  Gold/Iron ratio: {gold_n}/{iron_n} = {gold_n/max(iron_n,1):.1f}x")
    print(f"  Interpretation: {'Elite/trade dominant' if gold_n > iron_n else 'Production/craft dominant'}")


# ═════════════════════════════════════════════════════════════════════════
# 8. VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[8] Generating visualization...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('E040 — Bamboo Civilization: Material Culture in Prasasti\n'
             'What Did Old Javanese Inscriptions Build With?',
             fontsize=13, fontweight='bold', y=0.98)

# A: Material frequency by class
ax1 = axes[0, 0]
class_colors = {"organic": "#27ae60", "lithic": "#7f8c8d", "metal": "#f39c12",
                "building": "#3498db", "activity": "#9b59b6", "institution": "#e74c3c"}
if mat_counts:
    top15 = mat_counts.most_common(15)
    names = [p[0] for p in top15]
    counts = [p[1] for p in top15]
    colors = [class_colors.get(MATERIAL_KEYWORDS[n]["class"], "#bdc3c7") for n in names]
    ax1.barh(range(len(names)), counts, color=colors, edgecolor='white')
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel('Number of Inscriptions')
    ax1.set_title('A. Material Keywords in Prasasti', fontsize=11)
    ax1.invert_yaxis()
    from matplotlib.patches import Patch
    legend_items = [Patch(color=class_colors[c], label=c.capitalize())
                    for c in ["organic", "lithic", "metal"] if c in
                    set(MATERIAL_KEYWORDS[n]["class"] for n in names)]
    ax1.legend(handles=legend_items, fontsize=8, loc='lower right')

# B: Organic vs Lithic comparison
ax2 = axes[0, 1]
categories = ['Organic\n(wood, bamboo,\nthatch)', 'Lithic\n(stone, brick,\ntemple)',
              'Metal\n(gold, silver,\niron)']
vals = [has_organic, has_lithic, has_metal]
colors_bar = [class_colors["organic"], class_colors["lithic"], class_colors["metal"]]
bars = ax2.bar(categories, vals, color=colors_bar, edgecolor='white', width=0.6)
for bar, v in zip(bars, vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{v}\n({v/len(corpus)*100:.0f}%)', ha='center', fontsize=10)
ax2.set_ylabel('Inscriptions Mentioning')
ax2.set_title(f'B. Material Class (n={len(corpus)} inscriptions)', fontsize=11)
ax2.set_ylim(0, max(vals) * 1.3)

# C: Temporal trend
ax3 = axes[1, 0]
if dated and century_data:
    centuries = sorted(century_data.keys())
    org_pcts = [century_data[c]["organic"]/century_data[c]["total"]*100
                if century_data[c]["total"] > 0 else 0 for c in centuries]
    lit_pcts = [century_data[c]["lithic"]/century_data[c]["total"]*100
                if century_data[c]["total"] > 0 else 0 for c in centuries]
    met_pcts = [century_data[c]["metal"]/century_data[c]["total"]*100
                if century_data[c]["total"] > 0 else 0 for c in centuries]

    x = range(len(centuries))
    ax3.plot(x, org_pcts, 'o-', color=class_colors["organic"], label='Organic', linewidth=2)
    ax3.plot(x, lit_pcts, 's-', color=class_colors["lithic"], label='Lithic', linewidth=2)
    ax3.plot(x, met_pcts, '^-', color=class_colors["metal"], label='Metal', linewidth=2)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'C{c}' for c in centuries], fontsize=9)
    ax3.set_xlabel('Century CE')
    ax3.set_ylabel('% of Inscriptions')
    ax3.set_title('C. Material Mentions Over Time', fontsize=11)
    ax3.legend(fontsize=8)

# D: Organic-only vs Lithic-only vs Both
ax4 = axes[1, 1]
venn_data = {
    'Organic only': has_organic_only,
    'Lithic only': has_lithic_only,
    'Both': has_both,
    'Neither': len(corpus) - has_organic - has_lithic_only
}
colors_venn = [class_colors["organic"], class_colors["lithic"], '#8e44ad', '#bdc3c7']
wedges, texts, autotexts = ax4.pie(
    list(venn_data.values()),
    labels=[f'{k}\n(n={v})' for k, v in venn_data.items()],
    colors=colors_venn,
    autopct='%1.0f%%',
    startangle=90
)
ax4.set_title('D. Material Exclusivity', fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(str(RESULTS_DIR / 'material_culture_4panel.png'), dpi=150, bbox_inches='tight')
print("  Saved: material_culture_4panel.png")
plt.close('all')


# ═════════════════════════════════════════════════════════════════════════
# 9. SAVE RESULTS
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("[9] Saving results...")
print("=" * 70)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)

summary = {
    "experiment": "E040_bamboo_civilization",
    "hypothesis": "I-040: Pre-Hindu Nusantara built non-lithic (bambu/kayu)",
    "n_inscriptions": len(corpus),
    "n_with_material_keywords": n_with_material,
    "organic_inscriptions": has_organic,
    "lithic_inscriptions": has_lithic,
    "metal_inscriptions": has_metal,
    "organic_only": has_organic_only,
    "lithic_only": has_lithic_only,
    "both_organic_lithic": has_both,
    "keyword_frequencies": {k: int(v) for k, v in mat_counts.most_common()},
}

with open(str(RESULTS_DIR / "material_culture_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, cls=NpEncoder)

# Detailed CSV
rows = []
for entry in corpus:
    mats = "|".join(entry['material_hits'].keys())
    classes = "|".join(set(h['class'] for h in entry['material_hits'].values()))
    rows.append({
        "filename": entry['filename'],
        "title": entry['title'],
        "year_ce": entry['year_ce'] or '',
        "n_organic": entry['n_organic'],
        "n_lithic": entry['n_lithic'],
        "n_metal": entry['n_metal'],
        "materials": mats,
        "classes": classes,
    })

df = pd.DataFrame(rows)
df.to_csv(str(RESULTS_DIR / "material_culture_inscriptions.csv"), index=False)
print("  Saved: material_culture_summary.json")
print("  Saved: material_culture_inscriptions.csv")


# ═════════════════════════════════════════════════════════════════════════
# 10. VERDICT
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("VERDICT: BAMBOO CIVILIZATION HYPOTHESIS")
print("=" * 70)

print(f"""
  CORPUS: {len(corpus)} inscriptions

  ORGANIC materials mentioned: {has_organic} ({has_organic/len(corpus)*100:.1f}%)
  LITHIC materials mentioned:  {has_lithic} ({has_lithic/len(corpus)*100:.1f}%)
  METAL materials mentioned:   {has_metal} ({has_metal/len(corpus)*100:.1f}%)

  Organic-ONLY inscriptions:   {has_organic_only}
  Lithic-ONLY inscriptions:    {has_lithic_only}
""")

if has_organic > has_lithic:
    print("  >>> ORGANIC > LITHIC — Supports bamboo civilization hypothesis")
    print("  >>> Prasasti record reflects a material culture that was")
    print("  >>> predominantly non-lithic. The archaeological 'blank'")
    print("  >>> is a PRESERVATION BIAS, not an absence of civilization.")
elif has_lithic > has_organic:
    print("  >>> LITHIC > ORGANIC — Does not support hypothesis as stated")
    print("  >>> BUT: prasasti are themselves stone objects, so lithic")
    print("  >>> mentions may reflect the medium, not the culture.")
else:
    print("  >>> EQUAL — Inconclusive")

print("\n" + "=" * 70)
print("E040 COMPLETE")
print("=" * 70)

"""
E165: Ghost Vocabulary — Linguistic Fossils in DHARMA Old Javanese
==================================================================
Directly analyzes the ORIGINAL Old Javanese/Sanskrit edition text
(not English translations) from all 268 DHARMA inscriptions.

'Ghost words' = words that appear in early centuries but vanish later.
These are fossilized remnants of pre-Indic culture overwritten by
Sanskrit administrative vocabulary.

Reverse ghost words = words that appear ONLY in later centuries.
These are Sanskrit imports that replaced indigenous terms.

This analysis goes where no computational study has gone:
into the actual Kawi text at corpus scale.
"""

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict, Counter
import json
import re

print("=" * 70)
print("E165: GHOST VOCABULARY - LINGUISTIC FOSSILS IN OLD JAVANESE")
print("=" * 70)

# ============================================================
# 1. Parse ALL 268 DHARMA XML for edition text
# ============================================================

dharma_dir = Path("D:/documents/volcarch-repo/experiments/E023_ritual_screening/data/dharma/xml")
metadata_path = Path("D:/documents/volcarch-repo/experiments/E030_prasasti_temporal_nlp/results/dated_inscriptions.csv")
geo_path = Path("D:/documents/volcarch-repo/experiments/E082_inscription_georeferencing/results/geocoded_inscriptions.csv")

df_meta = pd.read_csv(metadata_path)
df_geo = pd.read_csv(geo_path)

def extract_edition_text(xml_path):
    """Extract the edition (original language) text from DHARMA TEI-XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

        ed_divs = root.findall('.//tei:div[@type="edition"]', ns)
        text = ""
        for div in ed_divs:
            for elem in div.iter():
                if elem.text:
                    text += elem.text + " "
                if elem.tail:
                    text += elem.tail + " "

        # Clean: remove line numbers, editorial marks, etc.
        text = re.sub(r'\d+[a-z]?\.', '', text)  # line numbers like "1a."
        text = re.sub(r'[\[\]\(\)\{\}]', '', text)  # editorial brackets
        text = re.sub(r'[⌈⌉⟨⟩⎡⎤]', '', text)  # special brackets
        text = re.sub(r'\.\.\.', '', text)  # lacunae
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    except:
        return ""

print("\nParsing DHARMA editions (original Old Javanese/Sanskrit)...")

inscriptions = []
xml_files = sorted(dharma_dir.glob("*.xml"))

for xml_path in xml_files:
    filename = xml_path.name
    edition = extract_edition_text(xml_path)

    if not edition or len(edition) < 10:
        continue

    # Match with metadata
    meta_row = df_meta[df_meta['filename'] == filename]
    geo_row = df_geo[df_geo['filename'] == filename]

    century = int(meta_row['century'].values[0]) if len(meta_row) > 0 and pd.notna(meta_row['century'].values[0]) else None
    year_ce = float(meta_row['year_ce'].values[0]) if len(meta_row) > 0 and pd.notna(meta_row['year_ce'].values[0]) else None
    lang = str(meta_row['lang'].values[0]) if len(meta_row) > 0 else None
    volcano_dist = float(geo_row['volcano_dist_km'].values[0]) if len(geo_row) > 0 and pd.notna(geo_row.iloc[0].get('volcano_dist_km', None)) else None

    # Tokenize: split on whitespace, lowercase, strip punctuation
    tokens = re.findall(r'[a-zA-Z\u0100-\u024F]+', edition.lower())
    tokens = [t for t in tokens if len(t) > 1]  # remove single chars

    inscriptions.append({
        'filename': filename,
        'century': century,
        'year_ce': year_ce,
        'lang': lang,
        'volcano_dist': volcano_dist,
        'tokens': tokens,
        'n_tokens': len(tokens),
        'edition_text': edition[:500],
    })

print(f"  Inscriptions with edition text: {len(inscriptions)}")
dated = [i for i in inscriptions if i['century'] is not None]
print(f"  Dated inscriptions: {len(dated)}")

# Century distribution
century_counts = Counter(i['century'] for i in dated)
print(f"  Century distribution: {dict(sorted(century_counts.items()))}")

total_tokens = sum(i['n_tokens'] for i in inscriptions)
print(f"  Total tokens: {total_tokens:,}")
unique_tokens = len(set(t for i in inscriptions for t in i['tokens']))
print(f"  Unique tokens: {unique_tokens:,}")

# ============================================================
# 2. Build century-vocabulary matrices
# ============================================================
print(f"\n{'='*70}")
print("2. CENTURY-VOCABULARY ANALYSIS")
print(f"{'='*70}")

# Vocabulary by century
century_vocab = defaultdict(Counter)
for i in dated:
    century_vocab[i['century']].update(i['tokens'])

# Find "ghost words" — words that appear in early centuries but vanish
# Definition: appears in C7-C9 (early) but NOT in C10-C14 (later)
early_centuries = [7, 8, 9]
late_centuries = [10, 11, 12, 13, 14]

early_vocab = set()
for c in early_centuries:
    early_vocab.update(century_vocab[c].keys())

late_vocab = set()
for c in late_centuries:
    late_vocab.update(century_vocab[c].keys())

ghost_words = early_vocab - late_vocab  # appear early, vanish later
reverse_ghosts = late_vocab - early_vocab  # appear later, absent early
persistent = early_vocab & late_vocab  # appear in both periods

# Filter: only count words that appear at least 2x in early period
# (to avoid hapax legomena)
early_counts = Counter()
for c in early_centuries:
    early_counts.update(century_vocab[c])

ghost_words_filtered = {w for w in ghost_words if early_counts[w] >= 2}

late_counts = Counter()
for c in late_centuries:
    late_counts.update(century_vocab[c])

reverse_ghosts_filtered = {w for w in reverse_ghosts if late_counts[w] >= 2}

print(f"\n  Early vocabulary (C7-C9): {len(early_vocab)} unique tokens")
print(f"  Late vocabulary (C10-C14): {len(late_vocab)} unique tokens")
print(f"  Persistent (both periods): {len(persistent)} tokens")
print(f"\n  GHOST WORDS (early only, freq >= 2): {len(ghost_words_filtered)}")
print(f"  REVERSE GHOSTS (late only, freq >= 2): {len(reverse_ghosts_filtered)}")
print(f"\n  Ghost ratio: {len(ghost_words_filtered)/(len(ghost_words_filtered)+len(persistent))*100:.1f}% of early vocabulary vanishes")

# ============================================================
# 3. Analyze ghost words — what semantic domains do they represent?
# ============================================================
print(f"\n{'='*70}")
print("3. GHOST WORD ANALYSIS")
print(f"{'='*70}")

# Known Sanskrit markers
sanskrit_markers = {
    'sri', 'maharaja', 'deva', 'dharma', 'karma', 'yoga', 'mantra', 'tantra',
    'puja', 'mandala', 'avatar', 'guru', 'rsi', 'brahmana', 'ksatriya',
    'vaisya', 'sudra', 'cakra', 'vajra', 'padma', 'siva', 'visnu', 'brahma',
    'indra', 'agni', 'varuna', 'yama', 'soma', 'surya', 'candra', 'vayu',
    'prthivi', 'akasa', 'jala', 'teja', 'veda', 'sutra', 'sastra', 'sloka',
    'gatha', 'prasasti', 'sima', 'natha', 'prabhu', 'bhumi', 'nagara',
    'vihara', 'sangha', 'stupa', 'caitya', 'linga', 'yoni', 'garbha',
    'grha', 'pura', 'kuta', 'desa', 'grama', 'vana', 'giri', 'sagara',
    'nadi', 'tirtha', 'ksetra', 'samskara', 'upacara', 'homa', 'bali',
    'dana', 'punya', 'papa', 'svarga', 'naraka', 'moksa', 'nirvana',
    'samsara', 'atman', 'jiva', 'prajna', 'karuna', 'maitri', 'mudita',
}

# Known indigenous (Austronesian/Old Javanese) markers
indigenous_markers = {
    'hyang', 'sang', 'si', 'mapangkah', 'sawah', 'tgal', 'huma',
    'nusa', 'wanua', 'thani', 'karaman', 'rama', 'buyut', 'kabayan',
    'hulun', 'wahuta', 'rakryan', 'samgat', 'pamgat', 'patih',
    'tuhan', 'dang', 'pu', 'dyah', 'rakai', 'mapatih',
    'skar', 'wwah', 'wungkal', 'tampyal', 'suruhan',
    'panumbas', 'panawing', 'pakirakira', 'sapatha',
    'watu', 'kayu', 'pring', 'gadung', 'tawung',
    'sawung', 'hayam', 'kbo', 'sapi', 'wdus', 'celeng',
    'manuk', 'iwak', 'babi', 'anjing', 'kucing',
    'anak', 'bapa', 'ibu', 'kaka', 'adi',
    'rumah', 'imah', 'wisma', 'umah',
    'banua', 'danu', 'tasik', 'laut', 'segara',
    'gunung', 'wukir', 'parwata', 'acala',
}

# Classify ghost words
ghost_sanskrit = ghost_words_filtered & sanskrit_markers
ghost_indigenous = ghost_words_filtered & indigenous_markers
ghost_unknown = ghost_words_filtered - sanskrit_markers - indigenous_markers

reverse_sanskrit = reverse_ghosts_filtered & sanskrit_markers
reverse_indigenous = reverse_ghosts_filtered & indigenous_markers

print(f"\n  Ghost words by origin:")
print(f"    Sanskrit: {len(ghost_sanskrit)} ({', '.join(sorted(ghost_sanskrit)[:20])})")
print(f"    Indigenous: {len(ghost_indigenous)} ({', '.join(sorted(ghost_indigenous)[:20])})")
print(f"    Unknown/unclassified: {len(ghost_unknown)}")

print(f"\n  Reverse ghost words by origin:")
print(f"    Sanskrit (new imports): {len(reverse_sanskrit)} ({', '.join(sorted(reverse_sanskrit)[:20])})")
print(f"    Indigenous (late-emerging): {len(reverse_indigenous)} ({', '.join(sorted(reverse_indigenous)[:20])})")

# ============================================================
# 4. Top ghost words with context
# ============================================================
print(f"\n{'='*70}")
print("4. TOP GHOST WORDS (by frequency in early period)")
print(f"{'='*70}")

ghost_ranked = sorted(ghost_words_filtered, key=lambda w: early_counts[w], reverse=True)

print(f"\n  {'Word':<20} {'Early Freq':>10} {'Centuries':>15} {'Classification':>15}")
print(f"  {'-'*65}")

ghost_details = []
for word in ghost_ranked[:40]:
    centuries_present = [c for c in sorted(century_vocab.keys()) if word in century_vocab[c]]
    centuries_str = ','.join(f'C{c}' for c in centuries_present)

    if word in sanskrit_markers:
        classification = "SANSKRIT"
    elif word in indigenous_markers:
        classification = "INDIGENOUS"
    else:
        classification = "unknown"

    print(f"  {word:<20} {early_counts[word]:>10} {centuries_str:>15} {classification:>15}")

    ghost_details.append({
        'word': word,
        'early_frequency': int(early_counts[word]),
        'centuries': centuries_present,
        'classification': classification,
    })

# ============================================================
# 5. Reverse ghosts: Sanskrit imports over time
# ============================================================
print(f"\n{'='*70}")
print("5. TOP REVERSE GHOSTS (Sanskrit imports appearing later)")
print(f"{'='*70}")

reverse_ranked = sorted(reverse_ghosts_filtered, key=lambda w: late_counts[w], reverse=True)

print(f"\n  {'Word':<20} {'Late Freq':>10} {'Centuries':>15} {'Classification':>15}")
print(f"  {'-'*65}")

for word in reverse_ranked[:30]:
    centuries_present = [c for c in sorted(century_vocab.keys()) if word in century_vocab[c]]
    centuries_str = ','.join(f'C{c}' for c in centuries_present)

    if word in sanskrit_markers:
        classification = "SANSKRIT"
    elif word in indigenous_markers:
        classification = "INDIGENOUS"
    else:
        classification = "unknown"

    print(f"  {word:<20} {late_counts[word]:>10} {centuries_str:>15} {classification:>15}")

# ============================================================
# 6. Volcano zone vs court zone vocabulary
# ============================================================
print(f"\n{'='*70}")
print("6. VOLCANO ZONE vs COURT ZONE VOCABULARY")
print(f"{'='*70}")

volcano_inscriptions = [i for i in inscriptions if i['volcano_dist'] is not None and i['volcano_dist'] <= 20]
court_inscriptions = [i for i in inscriptions if i['volcano_dist'] is not None and 20 < i['volcano_dist'] <= 40]

volcano_vocab = Counter()
for i in volcano_inscriptions:
    volcano_vocab.update(i['tokens'])

court_vocab = Counter()
for i in court_inscriptions:
    court_vocab.update(i['tokens'])

# Words unique to volcano zone (freq >= 3)
volcano_only = {w for w in volcano_vocab if w not in court_vocab and volcano_vocab[w] >= 3}
court_only = {w for w in court_vocab if w not in volcano_vocab and court_vocab[w] >= 3}

print(f"\n  Volcano zone inscriptions: {len(volcano_inscriptions)} (within 20 km)")
print(f"  Court zone inscriptions: {len(court_inscriptions)} (20-40 km)")
print(f"  Volcano-only words (freq >= 3): {len(volcano_only)}")
print(f"  Court-only words (freq >= 3): {len(court_only)}")

if volcano_only:
    print(f"\n  VOLCANO-ONLY vocabulary (top 20):")
    volcano_only_ranked = sorted(volcano_only, key=lambda w: volcano_vocab[w], reverse=True)
    for word in volcano_only_ranked[:20]:
        classification = "SANSKRIT" if word in sanskrit_markers else ("INDIGENOUS" if word in indigenous_markers else "unknown")
        print(f"    {word:<20} freq={volcano_vocab[word]:>4}  {classification}")

if court_only:
    print(f"\n  COURT-ONLY vocabulary (top 20):")
    court_only_ranked = sorted(court_only, key=lambda w: court_vocab[w], reverse=True)
    for word in court_only_ranked[:20]:
        classification = "SANSKRIT" if word in sanskrit_markers else ("INDIGENOUS" if word in indigenous_markers else "unknown")
        print(f"    {word:<20} freq={court_vocab[word]:>4}  {classification}")

# ============================================================
# 7. Vocabulary diversity over time
# ============================================================
print(f"\n{'='*70}")
print("7. VOCABULARY DIVERSITY OVER TIME")
print(f"{'='*70}")

print(f"\n  {'Century':<8} {'N docs':>7} {'Tokens':>8} {'Unique':>8} {'TTR':>8} {'Indigenous%':>12}")
print(f"  {'-'*55}")

for c in sorted(century_counts.keys()):
    if c is None:
        continue
    vocab = century_vocab[c]
    total = sum(vocab.values())
    unique = len(vocab)
    ttr = unique / total if total > 0 else 0

    # Indigenous fraction
    indigenous_count = sum(vocab.get(w, 0) for w in indigenous_markers)
    sanskrit_count = sum(vocab.get(w, 0) for w in sanskrit_markers)
    classified = indigenous_count + sanskrit_count
    indigenous_pct = indigenous_count / classified * 100 if classified > 0 else 0

    print(f"  C{c:<6} {century_counts[c]:>7} {total:>8} {unique:>8} {ttr:>8.3f} {indigenous_pct:>10.1f}%")

# ============================================================
# 8. Save results
# ============================================================

results_dir = Path("D:/documents/volcarch-repo/experiments/E165_ghost_vocabulary/results")

results = {
    "total_inscriptions": len(inscriptions),
    "dated_inscriptions": len(dated),
    "total_tokens": total_tokens,
    "unique_tokens": unique_tokens,
    "ghost_words_count": len(ghost_words_filtered),
    "reverse_ghosts_count": len(reverse_ghosts_filtered),
    "persistent_count": len(persistent),
    "ghost_ratio_pct": len(ghost_words_filtered)/(len(ghost_words_filtered)+len(persistent))*100,
    "ghost_sanskrit": len(ghost_sanskrit),
    "ghost_indigenous": len(ghost_indigenous),
    "ghost_unknown": len(ghost_unknown),
    "top_ghost_words": ghost_details[:20],
    "volcano_only_count": len(volcano_only),
    "court_only_count": len(court_only),
}

with open(results_dir / "ghost_vocabulary.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# Save ghost word list
with open(results_dir / "ghost_words.txt", "w", encoding="utf-8") as f:
    f.write("# Ghost Words: appear in C7-C9, absent from C10-C14 (freq >= 2)\n")
    f.write(f"# Total: {len(ghost_words_filtered)}\n\n")
    for word in ghost_ranked:
        centuries = [c for c in sorted(century_vocab.keys()) if word in century_vocab[c]]
        f.write(f"{word}\t{early_counts[word]}\t{','.join(f'C{c}' for c in centuries)}\n")

print(f"\nResults saved to {results_dir}")
print(f"\nDONE.")

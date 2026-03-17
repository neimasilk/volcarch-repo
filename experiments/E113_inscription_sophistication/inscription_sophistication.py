#!/usr/bin/env python3
"""
E113 — Inscription Sophistication Analysis

HYPOTHESIS: If the earliest Javanese inscriptions (C7-C8) show full literary
sophistication from the start, this implies a pre-existing writing/literary
tradition on organic media. If complexity INCREASES over time (a "learning
curve"), then writing was a new technology being mastered.

METHOD:
  1. Extract edition text from DHARMA XML files (romanized epigraphy)
  2. Merge with E030 dated inscriptions for temporal metadata
  3. Compute sophistication metrics per inscription:
     - Vocabulary diversity (type-token ratio, corrected TTR)
     - Mean word length (characters)
     - Hapax legomena ratio (words appearing only once / total)
     - Unique word count
     - Total word count (inscription length)
     - Sanskrit loanword ratio (heuristic: words with typical Sanskrit
       phonological markers like long vowels, retroflex consonants, etc.)
     - Formulaic density (date/genealogy formulae)
  4. Group by century and compare EARLY (C7-C8) vs MATURE (C10-C12)
  5. Statistical tests: Mann-Whitney U, Spearman correlation over time

DATA SOURCES:
  - DHARMA XML corpus: E023_ritual_screening/data/dharma/xml/ (269 files)
  - Dated inscriptions: E030_prasasti_temporal_nlp/results/dated_inscriptions.csv
  - E023 classification: E023_ritual_screening/results/full_corpus_classification.csv

Author: VOLCARCH project (AI-assisted)
Date: 2026-03-17
"""

import sys
import io
import os
import re
import json
import warnings
import math
import xml.etree.ElementTree as ET
from collections import Counter

# Windows cp1252 console fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore', category=FutureWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XML_DIR = os.path.join(REPO, "experiments", "E023_ritual_screening", "data", "dharma", "xml")
E030_CSV = os.path.join(REPO, "experiments", "E030_prasasti_temporal_nlp", "results", "dated_inscriptions.csv")
E023_CSV = os.path.join(REPO, "experiments", "E023_ritual_screening", "results", "full_corpus_classification.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NS = {'tei': 'http://www.tei-c.org/ns/1.0'}

print("=" * 72)
print("E113 — Inscription Sophistication Analysis")
print("Do early Javanese inscriptions show a 'learning curve' or full maturity?")
print("=" * 72)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[1] Loading data sources...")

# E030 dated inscriptions
df_dated = pd.read_csv(E030_CSV)
print(f"  E030 dated inscriptions: {len(df_dated)}")

# E023 full corpus (for undated inscriptions with known century estimates)
df_e023 = pd.read_csv(E023_CSV)
print(f"  E023 full corpus: {len(df_e023)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. EXTRACT EDITION TEXT FROM XML
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2] Extracting edition text from DHARMA XML files...")

# Sanskrit phonological markers for loanword detection
# These patterns identify words likely borrowed from Sanskrit/Pali
SANSKRIT_MARKERS = re.compile(
    r'(?:'
    r'[āīūṛṝḷḹ]|'           # long vowels, vocalic r/l
    r'[ṭḍṇṣ]|'               # retroflexes
    r'[śṣ]|'                  # sibilants
    r'ñ|'                     # palatal nasal
    r'jñ|'                    # typical Sanskrit cluster
    r'kṣ|'                    # typical Sanskrit cluster
    r'str|'                   # typical Sanskrit cluster
    r'bh|dh|gh|kh|ph|th|'    # aspirated stops
    r'[ḥṃ]'                  # visarga, anusvara (when explicit)
    r')',
    re.IGNORECASE
)

# Words that are clearly Sanskrit/Pali by semantic domain
# (not exhaustive, but captures common legal/religious/calendrical terms)
SANSKRIT_SEMANTIC = {
    'svasti', 'śaka', 'saka', 'varṣa', 'varṣātīta', 'māsa', 'tithi',
    'pakṣa', 'śukla', 'kṛṣṇa', 'nakṣatra', 'yoga', 'muhūrtta', 'muhurtta',
    'karaṇa', 'rāśi', 'rāśī', 'maṇḍala', 'devatā', 'devata', 'grahacāra',
    'vāra', 'vara', 'śrī', 'sri', 'mahārāja', 'maharaja', 'rāja', 'raja',
    'ājñā', 'dharma', 'dharmmā', 'karma', 'pūjā', 'puja', 'homa', 'mantra',
    'svarga', 'piṇḍa', 'pitr', 'pitr̥', 'pralaya', 'pratiṣṭha', 'pratistha',
    'brāhmaṇa', 'brahmana', 'kṣatriya', 'ksatriya', 'vaiśya', 'vaisya',
    'śūdra', 'sudra', 'samudra', 'parvvata', 'parvata', 'nagara', 'nāgara',
    'dāna', 'dana', 'punya', 'kuśala', 'kusala', 'buddha', 'bodhisattva',
    'lokeśvara', 'lokesvara', 'amoghapāśa', 'vajra', 'padma', 'ratna',
    'sīma', 'sima', 'samgat', 'saṅ', 'sang', 'mapatih', 'patih',
    'rakryān', 'rakryan', 'mapañji', 'mapaniji', 'pu', 'dyah', 'dyaḥ',
    'praśasti', 'prasasti', 'jayapattra', 'jayapatra'
}

# Old Javanese/Old Malay indigenous terms (non-Sanskrit)
INDIGENOUS_MARKERS = {
    'hyang', 'hyaṁ', 'haji', 'ratu', 'datu', 'sawah', 'ladang',
    'kabuyutan', 'karaman', 'wanua', 'thani', 'sīma', 'tihaṇḍa',
    'susuk', 'watu', 'gunung', 'laut', 'sungai', 'desa', 'banua',
    'tuha', 'rama', 'buyut', 'ibu', 'anak', 'orang', 'uraṁ',
    'makamatai', 'pahumaan', 'parahu', 'besi', 'emas', 'sawit',
    'kelapa', 'padi', 'beras', 'kapas', 'garam', 'minyak',
    'manghuri', 'maṅhuri', 'wuku', 'panumbas'
}

# Date/formulaic terms (for formulaic density)
DATE_FORMULAE = {
    'svasti', 'śaka', 'saka', 'varṣātīta', 'varsatita', 'māsa', 'masa',
    'tithi', 'pakṣa', 'paksa', 'śukla', 'sukla', 'kṛṣṇa', 'krsna',
    'nakṣatra', 'naksatra', 'yoga', 'muhūrtta', 'muhurtta', 'karaṇa',
    'karana', 'rāśi', 'rasi', 'rāśī', 'maṇḍala', 'mandala', 'devatā',
    'devata', 'grahacāra', 'grahacara', 'vāra', 'vara', 'parvveśa',
    'parvesa', 'pratipada', 'dvitīyā', 'dvitiya', 'trayodaśī', 'trayodasi',
    'caturdaśī', 'caturdasi', 'pañcadaśī', 'pancadasi', 'amāvāsyā',
    'amavasya', 'pūrṇamāsī', 'purnamasi'
}


def extract_edition_text(xml_path):
    """Extract all text from <div type='edition'> in a DHARMA TEI XML file.

    Returns cleaned text string with XML tags, line break markers, and
    milestone markers removed. Preserves the original romanized text.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return ""

    editions = root.findall('.//tei:div[@type="edition"]', NS)
    if not editions:
        return ""

    # Get all text from the edition div(s)
    full_text = ""
    for ed in editions:
        full_text += " ".join(ed.itertext())

    # Clean up: collapse whitespace, remove line number artifacts
    full_text = re.sub(r'\s+', ' ', full_text).strip()

    # Remove standalone numbers (line numbers that leaked through)
    # But keep numbers that are part of dates (4-digit years, etc.)
    full_text = re.sub(r'\b\d{1,2}\b', '', full_text)

    # Remove punctuation-like symbols common in epigraphy notation
    full_text = re.sub(r'[.,:;!?()[\]{}|/\\<>«»"""\'`\-–—=+*#@~^]', ' ', full_text)

    # Collapse whitespace again
    full_text = re.sub(r'\s+', ' ', full_text).strip()

    return full_text


def tokenize(text):
    """Tokenize romanized epigraphy text into words.

    Splits on whitespace and removes very short fragments (single chars
    that are likely notation artifacts, except for known single-char words).
    """
    words = text.lower().split()
    # Remove tokens that are purely numeric or single non-alphabetic chars
    cleaned = []
    for w in words:
        # Strip trailing/leading interpuncts and notation
        w = w.strip('·,.')
        if len(w) < 2:
            continue
        # Skip pure numbers
        if re.match(r'^\d+$', w):
            continue
        cleaned.append(w)
    return cleaned


def compute_sophistication_metrics(text, words):
    """Compute sophistication metrics for an inscription's text.

    Returns dict of metrics. All metrics are computed on the tokenized
    edition text (romanized Old Javanese/Old Malay/Sanskrit).
    """
    if not words or len(words) < 3:
        return None

    total_words = len(words)
    word_freq = Counter(words)
    unique_words = len(word_freq)

    # 1. Type-Token Ratio (TTR)
    ttr = unique_words / total_words

    # 2. Corrected TTR (Guiraud's index): types / sqrt(tokens)
    # More robust for varying text lengths
    guiraud = unique_words / math.sqrt(total_words)

    # 3. Mean word length (characters)
    mean_word_length = np.mean([len(w) for w in words])

    # 4. Hapax legomena ratio (words appearing exactly once)
    hapax = sum(1 for w, c in word_freq.items() if c == 1)
    hapax_ratio = hapax / total_words

    # 5. Sanskrit loanword detection (phonological)
    # A word is classified as "Sanskrit-influenced" if it contains
    # characteristic Sanskrit phonological features
    sanskrit_phon_count = 0
    for w in set(words):
        if SANSKRIT_MARKERS.search(w):
            sanskrit_phon_count += 1
    sanskrit_phon_ratio = sanskrit_phon_count / unique_words if unique_words > 0 else 0

    # 6. Sanskrit semantic detection (known terms)
    sanskrit_sem_count = sum(1 for w in set(words) if w in SANSKRIT_SEMANTIC)
    sanskrit_sem_ratio = sanskrit_sem_count / unique_words if unique_words > 0 else 0

    # 7. Indigenous term count
    indigenous_count = sum(1 for w in set(words) if w in INDIGENOUS_MARKERS)

    # 8. Formulaic density (date/titulary formulae proportion)
    formulaic_count = sum(1 for w in words if w in DATE_FORMULAE)
    formulaic_ratio = formulaic_count / total_words

    # 9. Non-formulaic vocabulary diversity
    # Remove formulaic words and compute TTR on remainder
    non_formulaic = [w for w in words if w not in DATE_FORMULAE]
    nf_total = len(non_formulaic)
    if nf_total >= 3:
        nf_unique = len(set(non_formulaic))
        nf_ttr = nf_unique / nf_total
        nf_guiraud = nf_unique / math.sqrt(nf_total)
    else:
        nf_ttr = np.nan
        nf_guiraud = np.nan

    return {
        'total_words': total_words,
        'unique_words': unique_words,
        'ttr': ttr,
        'guiraud': guiraud,
        'mean_word_length': mean_word_length,
        'hapax_ratio': hapax_ratio,
        'hapax_count': hapax,
        'sanskrit_phon_ratio': sanskrit_phon_ratio,
        'sanskrit_sem_ratio': sanskrit_sem_ratio,
        'sanskrit_sem_count': sanskrit_sem_count,
        'indigenous_count': indigenous_count,
        'formulaic_ratio': formulaic_ratio,
        'nf_ttr': nf_ttr,
        'nf_guiraud': nf_guiraud,
        'nf_word_count': nf_total,
    }


# ── Process all XML files ────────────────────────────────────────────────────

print(f"  XML directory: {XML_DIR}")
xml_files = sorted([f for f in os.listdir(XML_DIR) if f.endswith('.xml')])
print(f"  Found {len(xml_files)} XML files")

records = []
skipped_empty = 0
skipped_short = 0

for fname in xml_files:
    fpath = os.path.join(XML_DIR, fname)
    text = extract_edition_text(fpath)

    if not text:
        skipped_empty += 1
        continue

    words = tokenize(text)

    if len(words) < 3:
        skipped_short += 1
        continue

    metrics = compute_sophistication_metrics(text, words)
    if metrics is None:
        skipped_short += 1
        continue

    # Get language from XML
    try:
        root = ET.parse(fpath).getroot()
        ed = root.findall('.//tei:div[@type="edition"]', NS)
        lang = ed[0].get('{http://www.w3.org/XML/1998/namespace}lang', 'unknown') if ed else 'unknown'
        # Also get title
        title_el = root.find('.//tei:title', NS)
        title = title_el.text if title_el is not None and title_el.text else fname
    except:
        lang = 'unknown'
        title = fname

    record = {'filename': fname, 'title': title, 'lang': lang}
    record.update(metrics)
    records.append(record)

print(f"  Processed: {len(records)} inscriptions with text")
print(f"  Skipped (empty edition): {skipped_empty}")
print(f"  Skipped (too short, <3 words): {skipped_short}")

df_metrics = pd.DataFrame(records)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. MERGE WITH DATES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3] Merging with E030 temporal data...")

# Merge on filename
df = df_metrics.merge(
    df_dated[['filename', 'year_ce', 'century']],
    on='filename',
    how='left'
)

n_dated = df['year_ce'].notna().sum()
n_undated = df['year_ce'].isna().sum()
print(f"  Dated inscriptions: {n_dated}")
print(f"  Undated inscriptions: {n_undated}")

# Filter to dated inscriptions only for temporal analysis
dated = df[df['year_ce'].notna()].copy()

# Filter out extremely short inscriptions (e.g., Borobudur relief labels)
# These are labels, not compositions, and would skew sophistication metrics
MIN_WORDS = 10
dated_full = dated[dated['total_words'] >= MIN_WORDS].copy()
print(f"  Dated inscriptions with >= {MIN_WORDS} words: {len(dated_full)}")

print(f"\n  Century distribution (>= {MIN_WORDS} words):")
for c, count in sorted(dated_full['century'].value_counts().items()):
    year_range = dated_full[dated_full['century'] == c]['year_ce']
    print(f"    C{int(c)}: {count} inscriptions ({int(year_range.min())}-{int(year_range.max())} CE)")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ANALYSIS: SOPHISTICATION BY CENTURY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("[4] ANALYSIS: Sophistication Metrics by Century")
print("=" * 72)

# Key metrics to compare
METRICS = [
    ('guiraud', "Guiraud's Index (vocab richness)"),
    ('mean_word_length', 'Mean Word Length (chars)'),
    ('hapax_ratio', 'Hapax Legomena Ratio'),
    ('sanskrit_phon_ratio', 'Sanskrit Phonology Ratio'),
    ('sanskrit_sem_ratio', 'Sanskrit Semantic Terms Ratio'),
    ('formulaic_ratio', 'Formulaic Density'),
    ('nf_guiraud', 'Non-Formulaic Guiraud Index'),
    ('total_words', 'Total Word Count'),
]

century_stats = []
for century, grp in dated_full.groupby('century'):
    row = {
        'century': int(century),
        'n': len(grp),
        'mean_words': grp['total_words'].mean(),
        'median_words': grp['total_words'].median(),
    }
    for metric, _ in METRICS:
        vals = grp[metric].dropna()
        if len(vals) > 0:
            row[f'{metric}_mean'] = vals.mean()
            row[f'{metric}_median'] = vals.median()
            row[f'{metric}_std'] = vals.std()
        else:
            row[f'{metric}_mean'] = np.nan
            row[f'{metric}_median'] = np.nan
            row[f'{metric}_std'] = np.nan
    century_stats.append(row)

century_df = pd.DataFrame(century_stats)

print("\n  Per-century sophistication summary:")
print("-" * 72)
for _, row in century_df.iterrows():
    c = int(row['century'])
    n = int(row['n'])
    print(f"\n  C{c} (n={n}, mean {row['mean_words']:.0f} words):")
    print(f"    Guiraud index:        {row['guiraud_mean']:.3f} (median: {row['guiraud_median']:.3f})")
    print(f"    Mean word length:     {row['mean_word_length_mean']:.2f}")
    print(f"    Hapax ratio:          {row['hapax_ratio_mean']:.3f}")
    print(f"    Sanskrit phon. ratio: {row['sanskrit_phon_ratio_mean']:.3f}")
    print(f"    Sanskrit sem. ratio:  {row['sanskrit_sem_ratio_mean']:.3f}")
    print(f"    Formulaic density:    {row['formulaic_ratio_mean']:.3f}")
    if not np.isnan(row.get('nf_guiraud_mean', np.nan)):
        print(f"    Non-formulaic Guiraud:{row['nf_guiraud_mean']:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. KEY TEST: EARLY (C7-C8) vs MATURE (C10-C12)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("[5] KEY TEST: Early (C7-C8) vs Mature (C10-C12) Inscriptions")
print("=" * 72)

early = dated_full[dated_full['century'].isin([7, 8])].copy()
mature = dated_full[dated_full['century'].isin([10, 11, 12])].copy()

print(f"\n  Early group (C7-C8):  n={len(early)}")
print(f"  Mature group (C10-C12): n={len(mature)}")

comparison_results = {}

for metric, label in METRICS:
    early_vals = early[metric].dropna()
    mature_vals = mature[metric].dropna()

    if len(early_vals) < 2 or len(mature_vals) < 2:
        print(f"\n  {label}: insufficient data for comparison")
        comparison_results[metric] = {
            'label': label,
            'early_n': len(early_vals),
            'mature_n': len(mature_vals),
            'test': 'INSUFFICIENT DATA'
        }
        continue

    # Mann-Whitney U test
    u_stat, u_p = stats.mannwhitneyu(early_vals, mature_vals, alternative='two-sided')

    # Effect size (rank-biserial correlation)
    n1, n2 = len(early_vals), len(mature_vals)
    r_rb = 1 - (2 * u_stat) / (n1 * n2)

    direction = "EARLY > MATURE" if early_vals.median() > mature_vals.median() else "MATURE > EARLY"
    sig = "***" if u_p < 0.001 else "**" if u_p < 0.01 else "*" if u_p < 0.05 else "n.s."

    print(f"\n  {label}:")
    print(f"    Early (C7-C8):    median={early_vals.median():.4f}, mean={early_vals.mean():.4f} (n={n1})")
    print(f"    Mature (C10-C12): median={mature_vals.median():.4f}, mean={mature_vals.mean():.4f} (n={n2})")
    print(f"    Mann-Whitney U={u_stat:.0f}, p={u_p:.4f} {sig}")
    print(f"    Effect size (rank-biserial r): {r_rb:.3f}")
    print(f"    Direction: {direction}")

    comparison_results[metric] = {
        'label': label,
        'early_n': n1,
        'early_mean': float(early_vals.mean()),
        'early_median': float(early_vals.median()),
        'mature_n': n2,
        'mature_mean': float(mature_vals.mean()),
        'mature_median': float(mature_vals.median()),
        'mann_whitney_U': float(u_stat),
        'p_value': float(u_p),
        'effect_size_r': float(r_rb),
        'direction': direction,
        'significant': u_p < 0.05,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 6. TEMPORAL CORRELATION (Spearman)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("[6] Temporal Correlation: Sophistication vs Time")
print("=" * 72)

temporal_results = {}

for metric, label in METRICS:
    vals = dated_full[['year_ce', metric]].dropna()
    if len(vals) < 5:
        print(f"\n  {label}: insufficient data (n={len(vals)})")
        temporal_results[metric] = {'label': label, 'n': len(vals), 'test': 'INSUFFICIENT DATA'}
        continue

    rho, p = stats.spearmanr(vals['year_ce'], vals[metric])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    trend = "INCREASING" if rho > 0 else "DECREASING"

    print(f"\n  {label}:")
    print(f"    Spearman rho={rho:.4f}, p={p:.4f} {sig} (n={len(vals)})")
    if p < 0.05:
        print(f"    => SIGNIFICANT {trend} trend over time")
    else:
        print(f"    => No significant trend")

    temporal_results[metric] = {
        'label': label,
        'n': int(len(vals)),
        'spearman_rho': float(rho),
        'p_value': float(p),
        'significant': p < 0.05,
        'trend': trend if p < 0.05 else 'NONE',
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 7. LANGUAGE-CONTROLLED ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("[7] Language-Controlled Analysis")
print("=" * 72)

# The DHARMA corpus includes Old Malay (omy-Latn), Old Javanese (kaw-Latn),
# and Sanskrit (san-Latn). Need to control for language shifts.
print("\n  Language distribution in dated inscriptions:")
lang_counts = dated_full['lang'].value_counts()
for lang, count in lang_counts.items():
    century_range = dated_full[dated_full['lang'] == lang]['century']
    print(f"    {lang}: {count} inscriptions (C{int(century_range.min())}-C{int(century_range.max())})")

# Test within Old Javanese (kaw-Latn) only — largest group
kaw_only = dated_full[dated_full['lang'] == 'kaw-Latn'].copy()
print(f"\n  Old Javanese (kaw-Latn) only: n={len(kaw_only)}")

kaw_temporal = {}
if len(kaw_only) >= 10:
    print("  Spearman correlations (kaw-Latn only):")
    for metric, label in METRICS:
        vals = kaw_only[['year_ce', metric]].dropna()
        if len(vals) < 5:
            continue
        rho, p = stats.spearmanr(vals['year_ce'], vals[metric])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"    {label}: rho={rho:.4f}, p={p:.4f} {sig} (n={len(vals)})")
        kaw_temporal[metric] = {
            'spearman_rho': float(rho),
            'p_value': float(p),
            'n': int(len(vals)),
            'significant': p < 0.05,
        }

# ═══════════════════════════════════════════════════════════════════════════════
# 8. LENGTH-CONTROLLED ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("[8] Length-Controlled Analysis (Guiraud index is already length-normalized)")
print("=" * 72)

# Guiraud index is types/sqrt(tokens), which partially controls for length.
# But let's also do a partial correlation controlling for word count.

# Use rank-based partial correlation
from functools import partial

def partial_spearman(x, y, z):
    """Rank-based partial correlation of x and y controlling for z."""
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)

    # Regress ranks of x on ranks of z
    slope_xz = np.polyfit(rz, rx, 1)
    resid_x = rx - np.polyval(slope_xz, rz)

    # Regress ranks of y on ranks of z
    slope_yz = np.polyfit(rz, ry, 1)
    resid_y = ry - np.polyval(slope_yz, rz)

    # Correlate residuals
    rho, p = stats.spearmanr(resid_x, resid_y)
    return rho, p

partial_results = {}
print("\n  Partial Spearman correlations (controlling for word count):")
for metric, label in METRICS:
    if metric in ('total_words',):  # skip word count itself
        continue
    vals = dated_full[['year_ce', metric, 'total_words']].dropna()
    if len(vals) < 10:
        continue

    rho, p = partial_spearman(
        vals['year_ce'].values,
        vals[metric].values,
        vals['total_words'].values
    )
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"    {label}: partial rho={rho:.4f}, p={p:.4f} {sig} (n={len(vals)})")
    partial_results[metric] = {
        'partial_rho': float(rho),
        'p_value': float(p),
        'n': int(len(vals)),
        'significant': p < 0.05,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 9. INTERPRETATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("[9] INTERPRETATION")
print("=" * 72)

# Count significant learning curve signals vs stability signals
n_increasing = sum(1 for m, r in temporal_results.items()
                   if r.get('significant') and r.get('trend') == 'INCREASING')
n_decreasing = sum(1 for m, r in temporal_results.items()
                   if r.get('significant') and r.get('trend') == 'DECREASING')
n_stable = sum(1 for m, r in temporal_results.items()
               if not r.get('significant') and r.get('n', 0) >= 5)

# Early vs mature comparison
n_early_higher = sum(1 for m, r in comparison_results.items()
                     if r.get('significant') and 'EARLY > MATURE' in r.get('direction', ''))
n_mature_higher = sum(1 for m, r in comparison_results.items()
                      if r.get('significant') and 'MATURE > EARLY' in r.get('direction', ''))
n_no_diff = sum(1 for m, r in comparison_results.items()
                if not r.get('significant', True) and r.get('early_n', 0) >= 2)

print(f"""
  TEMPORAL TRENDS (Spearman correlation over time):
    Metrics with INCREASING trend: {n_increasing}
    Metrics with DECREASING trend: {n_decreasing}
    Metrics with NO significant trend (stable): {n_stable}

  EARLY vs MATURE COMPARISON (Mann-Whitney U):
    Metrics where EARLY > MATURE (significantly): {n_early_higher}
    Metrics where MATURE > EARLY (significantly): {n_mature_higher}
    Metrics with NO significant difference: {n_no_diff}
""")

# Determine overall conclusion
if n_increasing >= 3 and n_early_higher == 0:
    conclusion = "LEARNING_CURVE"
    conclusion_text = (
        "Evidence supports a LEARNING CURVE: inscription sophistication increases "
        "significantly over time, consistent with writing being a new technology "
        "that was gradually mastered."
    )
elif n_stable >= 3 and n_increasing <= 1:
    conclusion = "STABLE_SOPHISTICATION"
    conclusion_text = (
        "Evidence supports STABLE SOPHISTICATION: inscription complexity metrics "
        "show no significant temporal trend. The earliest inscriptions are already "
        "as sophisticated as later ones, consistent with a pre-existing literary "
        "tradition on organic media (palm leaf, bark)."
    )
elif n_early_higher >= 2:
    conclusion = "EARLY_PEAK"
    conclusion_text = (
        "Evidence suggests an EARLY PEAK: the earliest inscriptions are MORE "
        "sophisticated than later ones on some metrics. This strongly supports "
        "a mature pre-existing tradition that was selectively transferred to stone."
    )
else:
    conclusion = "MIXED"
    conclusion_text = (
        "Evidence is MIXED: some metrics show temporal trends while others are "
        "stable. The pattern may reflect genre shifts (early Old Malay vs later "
        "Old Javanese) rather than a true learning curve."
    )

print(f"  CONCLUSION: {conclusion}")
print(f"  {conclusion_text}")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("[10] Generating visualizations...")
print("=" * 72)

# ── Figure 1: Key metrics by century (box plots) ─────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

plot_metrics = [
    ('guiraud', "Guiraud's Index\n(vocab richness)"),
    ('mean_word_length', 'Mean Word Length\n(characters)'),
    ('hapax_ratio', 'Hapax Legomena Ratio\n(word uniqueness)'),
    ('sanskrit_phon_ratio', 'Sanskrit Phonology\nRatio'),
    ('nf_guiraud', 'Non-Formulaic\nGuiraud Index'),
    ('total_words', 'Total Word Count'),
]

for idx, (metric, label) in enumerate(plot_metrics):
    ax = axes[idx // 3, idx % 3]

    # Prepare data for box plot by century
    centuries_list = sorted(dated_full['century'].unique())
    box_data = []
    positions = []
    labels_c = []

    for c in centuries_list:
        vals = dated_full[dated_full['century'] == c][metric].dropna()
        if len(vals) >= 1:
            box_data.append(vals.values)
            positions.append(c)
            labels_c.append(f"C{int(c)}\n(n={len(vals)})")

    if box_data:
        bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                        patch_artist=True, showfliers=True)

        # Color early (C7-C8) blue, mature (C10-C12) green, others gray
        for i, (box, pos) in enumerate(zip(bp['boxes'], positions)):
            if pos <= 8:
                box.set_facecolor('#3498db')
                box.set_alpha(0.6)
            elif 10 <= pos <= 12:
                box.set_facecolor('#2ecc71')
                box.set_alpha(0.6)
            else:
                box.set_facecolor('#bdc3c7')
                box.set_alpha(0.4)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels_c, fontsize=8)

    ax.set_ylabel(label, fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Add Spearman result
    tr = temporal_results.get(metric, {})
    if tr.get('spearman_rho') is not None:
        sig_marker = '*' if tr.get('significant') else ''
        ax.set_title(f"rho={tr['spearman_rho']:.3f}, p={tr['p_value']:.3f}{sig_marker}",
                     fontsize=9, color='red' if tr.get('significant') else 'gray')

fig.suptitle('E113: Inscription Sophistication by Century\n'
             'Blue = Early (C7-C8), Green = Mature (C10-C12), Gray = Other',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'sophistication_by_century.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: results/sophistication_by_century.png")

# ── Figure 2: Scatter plots of key metrics vs time ───────────────────────────

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

scatter_metrics = [
    ('guiraud', "Guiraud's Index"),
    ('mean_word_length', 'Mean Word Length (chars)'),
    ('hapax_ratio', 'Hapax Legomena Ratio'),
    ('nf_guiraud', 'Non-Formulaic Guiraud Index'),
]

for idx, (metric, label) in enumerate(scatter_metrics):
    ax = axes2[idx // 2, idx % 2]

    vals = dated_full[['year_ce', metric, 'lang']].dropna()

    # Color by language
    lang_colors = {
        'kaw-Latn': '#3498db',   # Old Javanese = blue
        'omy-Latn': '#e74c3c',   # Old Malay = red
        'san-Latn': '#f39c12',   # Sanskrit = gold
    }

    for lang, color in lang_colors.items():
        subset = vals[vals['lang'] == lang]
        if len(subset) > 0:
            ax.scatter(subset['year_ce'], subset[metric],
                       c=color, alpha=0.5, s=30, label=lang, edgecolors='gray', linewidth=0.3)

    # Other languages
    other = vals[~vals['lang'].isin(lang_colors.keys())]
    if len(other) > 0:
        ax.scatter(other['year_ce'], other[metric],
                   c='gray', alpha=0.3, s=20, label='other')

    # Trend line (all data)
    all_vals = dated_full[['year_ce', metric]].dropna()
    if len(all_vals) >= 5:
        z = np.polyfit(all_vals['year_ce'].values, all_vals[metric].values, 1)
        p_line = np.poly1d(z)
        x_range = np.linspace(all_vals['year_ce'].min(), all_vals['year_ce'].max(), 100)
        ax.plot(x_range, p_line(x_range), '--', color='black', alpha=0.5, linewidth=1)

    ax.set_xlabel('Year CE')
    ax.set_ylabel(label)

    tr = temporal_results.get(metric, {})
    if tr.get('spearman_rho') is not None:
        sig = '*' if tr.get('significant') else ''
        ax.set_title(f'{label}\n(rho={tr["spearman_rho"]:.3f}, p={tr["p_value"]:.3f}{sig})',
                     fontsize=10)
    else:
        ax.set_title(label)

    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)

fig2.suptitle('E113: Inscription Sophistication Over Time\n'
              'Color = script language; dashed line = linear trend',
              fontsize=12, fontweight='bold')
plt.tight_layout()
fig2.savefig(os.path.join(RESULTS_DIR, 'sophistication_vs_time.png'),
             dpi=150, bbox_inches='tight')
plt.close(fig2)
print("  Saved: results/sophistication_vs_time.png")

# ── Figure 3: Early vs Mature comparison ─────────────────────────────────────

fig3, axes3 = plt.subplots(1, 4, figsize=(16, 5))

comp_metrics = [
    ('guiraud', "Guiraud's Index"),
    ('mean_word_length', 'Mean Word Length'),
    ('hapax_ratio', 'Hapax Ratio'),
    ('nf_guiraud', 'Non-Form. Guiraud'),
]

for idx, (metric, label) in enumerate(comp_metrics):
    ax = axes3[idx]

    early_vals = early[metric].dropna()
    mature_vals = mature[metric].dropna()

    data = [early_vals.values, mature_vals.values] if len(early_vals) > 0 and len(mature_vals) > 0 else []

    if data:
        bp = ax.boxplot(data, labels=[f'Early\nC7-C8\n(n={len(early_vals)})',
                                       f'Mature\nC10-C12\n(n={len(mature_vals)})'],
                        patch_artist=True, showfliers=True)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('#2ecc71')
        bp['boxes'][1].set_alpha(0.6)

        cr = comparison_results.get(metric, {})
        if cr.get('p_value') is not None:
            sig = '*' if cr.get('significant') else 'n.s.'
            ax.set_title(f'{label}\np={cr["p_value"]:.4f} ({sig})', fontsize=9)
        else:
            ax.set_title(label, fontsize=9)
    else:
        ax.set_title(f'{label}\n(insufficient data)', fontsize=9)

    ax.grid(axis='y', alpha=0.3)

fig3.suptitle('E113: Early (C7-C8) vs Mature (C10-C12) Inscriptions',
              fontsize=12, fontweight='bold')
plt.tight_layout()
fig3.savefig(os.path.join(RESULTS_DIR, 'early_vs_mature.png'),
             dpi=150, bbox_inches='tight')
plt.close(fig3)
print("  Saved: results/early_vs_mature.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. INDIVIDUAL INSCRIPTION ANALYSIS (EARLIEST)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("[11] Earliest Inscription Profiles")
print("=" * 72)

earliest = dated_full.nsmallest(10, 'year_ce')
print("\n  10 earliest inscriptions:")
for _, row in earliest.iterrows():
    print(f"\n    {row['title'][:60]}")
    print(f"      Year: {int(row['year_ce'])} CE (C{int(row['century'])}), Lang: {row['lang']}")
    print(f"      Words: {int(row['total_words'])}, Unique: {int(row['unique_words'])}")
    print(f"      Guiraud: {row['guiraud']:.3f}, Mean word len: {row['mean_word_length']:.2f}")
    print(f"      Hapax ratio: {row['hapax_ratio']:.3f}")
    print(f"      Sanskrit phon: {row['sanskrit_phon_ratio']:.3f}, sem: {row['sanskrit_sem_ratio']:.3f}")
    print(f"      Formulaic: {row['formulaic_ratio']:.3f}")

# Compare with overall medians
print("\n  Overall medians (all dated, >= 10 words):")
for metric, label in METRICS:
    med = dated_full[metric].median()
    print(f"    {label}: {med:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 12. SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("[12] Saving results...")
print("=" * 72)

# Build summary JSON
summary = {
    "experiment": "E113_inscription_sophistication",
    "date": "2026-03-17",
    "hypothesis": (
        "If the earliest Javanese inscriptions (C7-C8) show full literary "
        "sophistication from the start, this implies a pre-existing writing "
        "tradition on organic media. If complexity increases over time "
        "(a 'learning curve'), writing was a new technology being mastered."
    ),
    "data_sources": [
        "DHARMA XML corpus (269 files, edition text)",
        "E030 dated inscriptions (166 with dates)",
        "E023 full corpus classification (268 inscriptions)"
    ],
    "inscriptions_analyzed": {
        "total_with_text": len(df_metrics),
        "dated": int(n_dated),
        "dated_with_min_words": len(dated_full),
        "min_word_threshold": MIN_WORDS,
        "early_group_C7_C8": len(early),
        "mature_group_C10_C12": len(mature),
    },
    "century_summary": [
        {k: (float(v) if isinstance(v, (np.floating, float)) and not np.isnan(v)
             else int(v) if isinstance(v, (np.integer, int))
             else None if isinstance(v, float) and np.isnan(v)
             else v)
         for k, v in row.items()}
        for _, row in century_df.iterrows()
    ],
    "early_vs_mature_comparison": {
        metric: {k: v for k, v in result.items()}
        for metric, result in comparison_results.items()
    },
    "temporal_correlations": {
        metric: {k: v for k, v in result.items()}
        for metric, result in temporal_results.items()
    },
    "partial_correlations_controlling_word_count": {
        metric: {k: v for k, v in result.items()}
        for metric, result in partial_results.items()
    },
    "old_javanese_only_correlations": kaw_temporal,
    "conclusion": {
        "verdict": conclusion,
        "interpretation": conclusion_text,
        "n_metrics_increasing": n_increasing,
        "n_metrics_decreasing": n_decreasing,
        "n_metrics_stable": n_stable,
        "n_early_higher": n_early_higher,
        "n_mature_higher": n_mature_higher,
        "n_no_difference": n_no_diff,
    },
    "volcarch_implication": (
        "If early inscriptions show mature sophistication, this is evidence "
        "for L1/L3: a literate tradition existed but was INVISIBLE because it "
        "used organic media (palm leaf, bark) that decomposed. The surviving "
        "stone inscriptions are the tip of the iceberg. This strengthens the "
        "argument that the archaeological record of early Indonesia is "
        "fundamentally shaped by taphonomic bias against organic materials."
    ),
    "limitations": [
        "Edition text is romanized transcription, not original script — "
        "some orthographic features are editorial choices",
        f"Borobudur relief labels (C8, 1-2 words) excluded (< {MIN_WORDS} word threshold)",
        "Sanskrit loanword detection is heuristic (phonological patterns), "
        "not based on etymological dictionary",
        "Early corpus dominated by Old Malay (Sriwijaya), mature corpus by "
        "Old Javanese — language shift confounds comparison",
        "Small N for early period (C7 especially)",
        "TTR and related metrics are sensitive to text length despite corrections",
        "DHARMA corpus is not exhaustive — selection bias toward "
        "well-preserved inscriptions",
    ],
    "earliest_inscriptions": [
        {
            'title': str(row['title'])[:80],
            'year_ce': int(row['year_ce']),
            'century': int(row['century']),
            'lang': str(row['lang']),
            'total_words': int(row['total_words']),
            'guiraud': float(row['guiraud']),
            'mean_word_length': float(row['mean_word_length']),
            'hapax_ratio': float(row['hapax_ratio']),
            'sanskrit_phon_ratio': float(row['sanskrit_phon_ratio']),
        }
        for _, row in earliest.iterrows()
    ],
}

# Clean NaN values from JSON
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

summary = clean_for_json(summary)

results_path = os.path.join(RESULTS_DIR, 'e113_results.json')
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"  Saved: results/e113_results.json")

# Save per-inscription metrics
dated_full.to_csv(os.path.join(RESULTS_DIR, 'inscription_metrics.csv'), index=False)
print(f"  Saved: results/inscription_metrics.csv ({len(dated_full)} rows)")

print("\n" + "=" * 72)
print("E113 COMPLETE")
print("=" * 72)

print(f"""
SUMMARY
=======

CONCLUSION: {conclusion}
{conclusion_text}

Key numbers:
  Inscriptions analyzed: {len(dated_full)} (dated, >= {MIN_WORDS} words)
  Early (C7-C8): {len(early)}
  Mature (C10-C12): {len(mature)}

  Temporal trends (Spearman):
    Increasing: {n_increasing} metrics
    Decreasing: {n_decreasing} metrics
    Stable: {n_stable} metrics

  Early vs Mature (Mann-Whitney):
    Early higher: {n_early_higher} metrics
    Mature higher: {n_mature_higher} metrics
    No difference: {n_no_diff} metrics

VOLCARCH implication:
  {summary['volcarch_implication'][:200]}...
""")

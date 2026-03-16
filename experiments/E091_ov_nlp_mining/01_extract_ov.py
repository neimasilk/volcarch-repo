#!/usr/bin/env python3
"""
E091: Automated NLP Extraction from OV Colonial Reports
========================================================
Processes 16 OV volumes (1912-1929, ~259K lines of OCR'd Dutch text)
to extract structured archaeological mentions: depth, sites, volcanoes,
materials, locations.

Builds on E070's regex patterns but adds:
- Paragraph-level co-occurrence analysis (depth + site + volcanic in same context)
- Structured CSV/JSON output ready for downstream analysis
- Cross-validation against DS-1's 52 manual entries
- Material and location extraction

This is the NLP equivalent of what a Dutch-reading archaeologist would do
scanning these volumes for burial depth and volcanic context data.
"""

import os
import re
import sys
import csv
import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
OV_DIR = REPO_ROOT / "data" / "raw" / "colonial_sources" / "OV"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# DS-1 register for validation
DS1_PATH = REPO_ROOT / "experiments" / "E070_colonial_literature_mining" / "results" / "colonial_site_register_v1.0.csv"

# ── Extraction Patterns ───────────────────────────────────────────────
# Each pattern: (compiled_regex, category, subcategory, description)

# DEPTH PATTERNS — numeric depth values
DEPTH_PATTERNS = [
    # Meters
    (re.compile(r'(\d+[\.,]\d+)\s*[Mm]\.?\s*diep', re.I), 'depth', 'meter', 'X.X M diep'),
    (re.compile(r'diepte\s*van\s*(\d+[\.,]\d*)\s*[Mm]', re.I), 'depth', 'meter', 'diepte van X M'),
    (re.compile(r'op\s*(?:een|eene)?\s*diepte\s*van\s*(\d+[\.,]?\d*)\s*[Mm]?', re.I), 'depth', 'meter', 'op een diepte van X'),
    (re.compile(r'(\d+[\.,]\d+)\s*[Mm]\.?\s*(?:onder|beneden)\s*(?:den?|het)\s*(?:grond|maaiveld|oppervlak)', re.I), 'depth', 'meter', 'X M onder maaiveld'),
    (re.compile(r'uitgegraven\s*tot\s*(\d+[\.,]?\d*)\s*[Mm]', re.I), 'depth', 'meter', 'uitgegraven tot X M'),
    (re.compile(r'tot\s*(\d+[\.,]?\d*)\s*[Mm]\.?\s*(?:af)?gegraven', re.I), 'depth', 'meter', 'tot X M gegraven'),
    (re.compile(r'(\d+[\.,]\d+)\s*[Mm]\.?\s*(?:lager|dieper)', re.I), 'depth', 'meter', 'X M lager/dieper'),
    (re.compile(r'grond\s*tot\s*(\d+[\.,]?\d*)\s*[Mm]', re.I), 'depth', 'meter', 'grond tot X M'),
    (re.compile(r'(\d+[\.,]\d+)\s*[Mm]\.?\s*(?:breed|lang|hoog)', re.I), 'dimension', 'meter', 'X M dimension'),
    # Voet (1 voet ≈ 0.3048 m)
    (re.compile(r'(\d+)\s*(?:voet|voeten)\s*(?:diep|onder|beneden)', re.I), 'depth', 'voet', 'X voet diep'),
    (re.compile(r'(\d+)\s*(?:voet|voeten)\s*(?:in|onder)\s*den?\s*grond', re.I), 'depth', 'voet', 'X voet in grond'),
    # El (1 el ≈ 0.69 m)
    (re.compile(r'(\d+[\.,]?\d*)\s*(?:el|ellen)\s*diep', re.I), 'depth', 'el', 'X el diep'),
    # Vadem (1 vadem ≈ 1.7 m)
    (re.compile(r'(\d+)\s*(?:vadem|vademen)\s*diep', re.I), 'depth', 'vadem', 'X vadem diep'),
]

# BURIAL/EXPOSURE PATTERNS — qualitative depth indicators
BURIAL_PATTERNS = [
    (re.compile(r'(?:onder|in)\s*den?\s*grond\s*(?:bedolven|begraven|geraak|gezakt|gevonden|verzonken|weggezonken)', re.I), 'burial', 'underground', 'buried in ground'),
    (re.compile(r'geheel\s*(?:in|onder)\s*den?\s*grond', re.I), 'burial', 'fully_buried', 'completely underground'),
    (re.compile(r'boven\s*den?\s*grond\s*(?:uitstekend|zichtbaar|uitkomend)', re.I), 'burial', 'protruding', 'above ground'),
    (re.compile(r'(?:gedeeltelijk|ten\s*deele|grootendeels)\s*(?:in|onder)\s*den?\s*grond', re.I), 'burial', 'partial', 'partially buried'),
    (re.compile(r'(?:verzakt|verzonken|weggezonken|ingezonken)', re.I), 'burial', 'subsidence', 'sunk/subsided'),
    (re.compile(r'bedekt\s*(?:met|door)\s*(?:aarde|grond|modder|asch|puin|slib|zand)', re.I), 'burial', 'covered', 'covered by deposits'),
    (re.compile(r'(?:blootgelegd|bloot\s*gelegd|te\s*voorschijn\s*gebracht)', re.I), 'burial', 'exposed', 'excavated/exposed'),
    (re.compile(r'(?:opgegraven|uitgegraven)', re.I), 'burial', 'excavated', 'excavated'),
    (re.compile(r'(?:aan\s*het\s*licht|te\s*voorschijn)\s*(?:gekomen|gebracht|geraakt)', re.I), 'burial', 'discovered', 'brought to light'),
    (re.compile(r'onder\s*den?\s*grond\s*(?:bedolven\s*)?liggen', re.I), 'burial', 'underground', 'lie underground'),
]

# VOLCANIC PATTERNS — eruption/deposit references
VOLCANIC_PATTERNS = [
    (re.compile(r'(?:vulkanisch|vulcani)', re.I), 'volcanic', 'general', 'volcanic reference'),
    (re.compile(r'(?:lava|lavastroom|lavabed)', re.I), 'volcanic', 'lava', 'lava'),
    (re.compile(r'(?:lahar|modderstroom|modderlaag|moddervloed)', re.I), 'volcanic', 'lahar', 'lahar/mudflow'),
    (re.compile(r'(?:asch(?:laag|regen)?|puimsteenlaag|puimsteen|tufsteen|tuf(?:laag)?)', re.I), 'volcanic', 'tephra', 'tephra/ash/tuff'),
    (re.compile(r'(?:eruptie|uitbarsting)', re.I), 'volcanic', 'eruption', 'eruption event'),
    (re.compile(r'krater', re.I), 'volcanic', 'crater', 'crater'),
    # Named volcanoes
    (re.compile(r'(?:Kloet|Keloed|Kelut)', re.I), 'volcanic', 'kelud', 'Kelud'),
    (re.compile(r'Merapi', re.I), 'volcanic', 'merapi', 'Merapi'),
    (re.compile(r'(?:Smeroe|Semeru)', re.I), 'volcanic', 'semeru', 'Semeru'),
    (re.compile(r'Bromo', re.I), 'volcanic', 'bromo', 'Bromo'),
    (re.compile(r'(?:Ardjoeno|Arjuno)', re.I), 'volcanic', 'arjuno', 'Arjuno'),
    (re.compile(r'Welirang', re.I), 'volcanic', 'welirang', 'Welirang'),
    (re.compile(r'(?:Raoen|Raung)', re.I), 'volcanic', 'raung', 'Raung'),
    (re.compile(r'(?:Idjen|Ijen)', re.I), 'volcanic', 'ijen', 'Ijen'),
    (re.compile(r'Lamongan', re.I), 'volcanic', 'lamongan', 'Lamongan'),
    (re.compile(r'Penanggungan', re.I), 'volcanic', 'penanggungan', 'Penanggungan'),
    (re.compile(r'Ringgit', re.I), 'volcanic', 'ringgit', 'Ringgit'),
    (re.compile(r'Tengger', re.I), 'volcanic', 'tengger', 'Tengger'),
    (re.compile(r'Wilis', re.I), 'volcanic', 'wilis', 'Wilis'),
    (re.compile(r'(?:Slamet|Slamat)', re.I), 'volcanic', 'slamet', 'Slamet'),
    (re.compile(r'Diëng|Dieng', re.I), 'volcanic', 'dieng', 'Dieng'),
    (re.compile(r'(?:Goenoeng|Gunung)\s+(\w+)', re.I), 'volcanic', 'mountain', 'Mountain reference'),
    (re.compile(r'(?:Sindoro|Soendoro)', re.I), 'volcanic', 'sindoro', 'Sindoro'),
    (re.compile(r'Sumbing', re.I), 'volcanic', 'sumbing', 'Sumbing'),
    (re.compile(r'(?:Oengaran|Ungaran)', re.I), 'volcanic', 'ungaran', 'Ungaran'),
    (re.compile(r'(?:Merbabu|Merbaboe)', re.I), 'volcanic', 'merbabu', 'Merbabu'),
    (re.compile(r'(?:Lawu|Lawoe)', re.I), 'volcanic', 'lawu', 'Lawu'),
]

# SITE PATTERNS — archaeological site identification
SITE_PATTERNS = [
    (re.compile(r'[Tt]jandi\s+([A-Z]\w+(?:\s+[A-Z]\w+)?)', re.I), 'site', 'tjandi', 'Tjandi'),
    (re.compile(r'[Cc]andi\s+([A-Z]\w+(?:\s+[A-Z]\w+)?)', re.I), 'site', 'candi', 'Candi'),
    (re.compile(r'[Tt]empel\w*', re.I), 'site', 'tempel', 'Temple'),
    (re.compile(r'[Hh]eiligdom\w*', re.I), 'site', 'heiligdom', 'Sanctuary'),
    (re.compile(r'[Rr]u[ïi]ne\w*', re.I), 'site', 'ruine', 'Ruin'),
    (re.compile(r'[Mm]onument\w*', re.I), 'site', 'monument', 'Monument'),
    (re.compile(r'[Oo]udheden', re.I), 'site', 'oudheden', 'Antiquities'),
    (re.compile(r'[Bb]aksteenen\s+(?:fundament|gebouw|muur|poort|tempel)', re.I), 'site', 'brick_structure', 'Brick structure'),
    (re.compile(r'[Ff]undament\w*', re.I), 'site', 'fundament', 'Foundation'),
    (re.compile(r'[Oo]pgraving\w*', re.I), 'site', 'opgraving', 'Excavation'),
    (re.compile(r'[Gg]raf(?:steen|tombe|monument|kelder)\w*', re.I), 'site', 'grave', 'Grave/tomb'),
    (re.compile(r'[Ss]tupa\w*', re.I), 'site', 'stupa', 'Stupa'),
    (re.compile(r'[Bb]iaro\w*', re.I), 'site', 'biaro', 'Biaro (Sumatran temple)'),
    (re.compile(r'[Ll]ingga\w*', re.I), 'site', 'lingga', 'Lingga'),
    (re.compile(r'[Yy]oni\w*', re.I), 'site', 'yoni', 'Yoni'),
]

# MATERIAL PATTERNS — archaeological materials found
MATERIAL_PATTERNS = [
    (re.compile(r'(?:beeld|beeldje)\w*', re.I), 'material', 'statue', 'Statue/figurine'),
    (re.compile(r'(?:reliëf|relief)\w*', re.I), 'material', 'relief', 'Relief'),
    (re.compile(r'[Bb]aksteen\w*', re.I), 'material', 'brick', 'Brick'),
    (re.compile(r'[Gg]oud\w*', re.I), 'material', 'gold', 'Gold'),
    (re.compile(r'[Bb]rons\w*', re.I), 'material', 'bronze', 'Bronze'),
    (re.compile(r'[Zz]ilver\w*', re.I), 'material', 'silver', 'Silver'),
    (re.compile(r'[Kk]oper\w*', re.I), 'material', 'copper', 'Copper'),
    (re.compile(r'[Ii]nscriptie\w*', re.I), 'material', 'inscription', 'Inscription'),
    (re.compile(r'(?:aardewerk|potscherf|potscherven|pot(?:ten)?|urn(?:en)?)', re.I), 'material', 'pottery', 'Pottery/ceramics'),
    (re.compile(r'(?:porcelein|porselein)', re.I), 'material', 'porcelain', 'Porcelain'),
    (re.compile(r'(?:Ganesha|Gane[çs]a|Durga|Nandi|Çiwa|Shiva|Wisnu|Vishnu|Brahma|Boeddha|Buddha)', re.I), 'material', 'deity', 'Deity statue'),
    (re.compile(r'(?:koperplaat|inscriptie|oorkonde)', re.I), 'material', 'copper_plate', 'Copper plate/charter'),
    (re.compile(r'(?:steenen\s+bijl|stenen\s+bijl)', re.I), 'material', 'stone_tool', 'Stone tool'),
]

# LOCATION PATTERNS — place references
LOCATION_PATTERNS = [
    (re.compile(r'[Dd]ess?a\s+([A-Z]\w+)', re.I), 'location', 'desa', 'Village'),
    (re.compile(r'[Rr]egentschap\s+(\w+)', re.I), 'location', 'regentschap', 'Regency'),
    (re.compile(r'[Rr]esidentie\s+(\w+)', re.I), 'location', 'residentie', 'Residency'),
    (re.compile(r'[Aa]fdeeling\s+(\w+)', re.I), 'location', 'afdeeling', 'District'),
    (re.compile(r'[Oo]nderafdeeling\s+(\w+)', re.I), 'location', 'onderafdeeling', 'Sub-district'),
    # Major regions
    (re.compile(r'(?:Oost[\s-]?Java|East\s+Java)', re.I), 'location', 'east_java', 'East Java'),
    (re.compile(r'(?:Midden[\s-]?Java|Centraal[\s-]?Java|Central\s+Java)', re.I), 'location', 'central_java', 'Central Java'),
    (re.compile(r'(?:West[\s-]?Java)', re.I), 'location', 'west_java', 'West Java'),
    (re.compile(r'Bali', re.I), 'location', 'bali', 'Bali'),
    (re.compile(r'Sumatra', re.I), 'location', 'sumatra', 'Sumatra'),
    (re.compile(r'Borneo|Kalimantan', re.I), 'location', 'borneo', 'Borneo'),
    # Key archaeological towns
    (re.compile(r'Trowulan|Troewoelan|Trawoelan', re.I), 'location', 'trowulan', 'Trowulan'),
    (re.compile(r'Modjokerto|Mojokerto', re.I), 'location', 'mojokerto', 'Mojokerto'),
    (re.compile(r'Kediri|Kedoeri', re.I), 'location', 'kediri', 'Kediri'),
    (re.compile(r'Blitar', re.I), 'location', 'blitar', 'Blitar'),
    (re.compile(r'Malang', re.I), 'location', 'malang', 'Malang'),
    (re.compile(r'Prambanan', re.I), 'location', 'prambanan', 'Prambanan'),
    (re.compile(r'(?:Djokja|Jogja|Yogyakarta|Jogjakarta)', re.I), 'location', 'yogyakarta', 'Yogyakarta'),
    (re.compile(r'Soerakarta|Surakarta|Solo', re.I), 'location', 'surakarta', 'Surakarta'),
    (re.compile(r'Magelang', re.I), 'location', 'magelang', 'Magelang'),
    (re.compile(r'Pasoeroean|Pasuruan', re.I), 'location', 'pasuruan', 'Pasuruan'),
    (re.compile(r'Soerabaja|Surabaya', re.I), 'location', 'surabaya', 'Surabaya'),
    (re.compile(r'Batavia|Djakarta|Jakarta', re.I), 'location', 'batavia', 'Batavia/Jakarta'),
]

# ── Context window size ────────────────────────────────────────────────
CONTEXT_CHARS = 300  # characters before/after match for context

# ── Paragraph splitting ────────────────────────────────────────────────
def split_paragraphs(text, min_len=50):
    """Split OCR'd text into paragraphs. Handle noisy OCR line breaks."""
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Split on double newlines (paragraph breaks)
    raw_paras = re.split(r'\n\s*\n', text)
    # Merge very short fragments (OCR artifacts)
    paragraphs = []
    buf = ""
    for p in raw_paras:
        p = p.strip()
        if not p:
            continue
        # Join lines within paragraph
        p = re.sub(r'\n\s*', ' ', p)
        p = re.sub(r'\s+', ' ', p)
        if len(p) < min_len and buf:
            buf += " " + p
        else:
            if buf:
                paragraphs.append(buf)
            buf = p
    if buf:
        paragraphs.append(buf)
    return paragraphs


def extract_context(text, match_start, match_end, chars=CONTEXT_CHARS):
    """Extract context around a match."""
    start = max(0, match_start - chars)
    end = min(len(text), match_end + chars)
    before = text[start:match_start]
    after = text[match_end:end]
    matched = text[match_start:match_end]
    return before, matched, after


def parse_depth_value(value_str, unit):
    """Convert depth string + unit to meters."""
    val = float(value_str.replace(',', '.'))
    if unit == 'voet':
        return round(val * 0.3048, 2)
    elif unit == 'el':
        return round(val * 0.69, 2)
    elif unit == 'vadem':
        return round(val * 1.7, 2)
    else:  # meter
        return round(val, 2)


# ── Main extraction ───────────────────────────────────────────────────
def extract_from_volume(filepath):
    """Extract all mentions from a single OV volume."""
    volume_name = filepath.stem  # e.g., "OV_1912_fulltext"
    year_match = re.search(r'(\d{4})', volume_name)
    year = int(year_match.group(1)) if year_match else 0

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    mentions = []
    paragraphs = split_paragraphs(text)

    # Track line position for page estimation
    line_positions = [0]
    for i, ch in enumerate(text):
        if ch == '\n':
            line_positions.append(i)

    def estimate_page(char_pos):
        """Rough page estimate from character position."""
        line_num = 0
        for i, lp in enumerate(line_positions):
            if lp > char_pos:
                break
            line_num = i
        # ~50 lines per page in OV volumes
        return line_num // 50 + 1

    # Process each paragraph
    for para_idx, para in enumerate(paragraphs):
        para_mentions = []

        # Find position of this paragraph in original text (approximate)
        para_pos = text.find(para[:80]) if len(para) >= 80 else text.find(para[:40])
        page_est = estimate_page(para_pos) if para_pos >= 0 else 0

        # Apply all pattern groups
        all_patterns = [
            ('depth', DEPTH_PATTERNS),
            ('burial', BURIAL_PATTERNS),
            ('volcanic', VOLCANIC_PATTERNS),
            ('site', SITE_PATTERNS),
            ('material', MATERIAL_PATTERNS),
            ('location', LOCATION_PATTERNS),
        ]

        categories_found = set()

        for group_name, patterns in all_patterns:
            for pattern_tuple in patterns:
                regex, category, subcategory, description = pattern_tuple
                for m in regex.finditer(para):
                    depth_m = None
                    captured = m.group(1) if m.lastindex and m.lastindex >= 1 else None

                    # Parse depth values
                    if category == 'depth' and captured:
                        try:
                            depth_m = parse_depth_value(captured, subcategory)
                        except (ValueError, TypeError):
                            pass

                    mention = {
                        'volume': volume_name.replace('_fulltext', ''),
                        'year': year,
                        'paragraph_idx': para_idx,
                        'page_est': page_est,
                        'category': category,
                        'subcategory': subcategory,
                        'description': description,
                        'matched_text': m.group(0)[:100],
                        'captured_value': captured,
                        'depth_m': depth_m,
                        'context': para[:500],
                    }
                    para_mentions.append(mention)
                    categories_found.add(category)

        # Tag co-occurrence if paragraph has matches from multiple categories
        if len(categories_found) >= 2:
            for pm in para_mentions:
                pm['cooccurrence'] = '+'.join(sorted(categories_found))
                pm['cooccurrence_count'] = len(categories_found)
        else:
            for pm in para_mentions:
                pm['cooccurrence'] = ''
                pm['cooccurrence_count'] = 1 if categories_found else 0

        mentions.extend(para_mentions)

    return mentions, len(paragraphs), len(text)


def load_ds1():
    """Load DS-1 register for cross-validation."""
    ds1_entries = []
    if not DS1_PATH.exists():
        print(f"  WARNING: DS-1 register not found at {DS1_PATH}")
        return ds1_entries

    with open(DS1_PATH, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds1_entries.append(row)
    return ds1_entries


def validate_against_ds1(all_mentions, ds1_entries):
    """Check how many DS-1 entries are captured by automated extraction."""
    if not ds1_entries:
        return {}

    ds1_sites = []
    for entry in ds1_entries:
        site = entry.get('site_name', '').strip()
        modern = entry.get('modern_name', '').strip()
        volume = entry.get('source', '').strip()
        depth = entry.get('burial_depth_m', '').strip()
        notes = entry.get('notes', '').strip()
        ds1_sites.append({
            'site_name': site,
            'modern_name': modern,
            'volume': volume,
            'depth_m': depth,
            'notes': notes,
            'found': False,
            'match_details': ''
        })

    # Build search index: all extracted contexts grouped by volume
    mention_texts = defaultdict(set)
    for m in all_mentions:
        vol = m['volume']
        mention_texts[vol].add(m['context'].lower())
    # Also build a global combined text per volume for broader search
    volume_combined = {}
    for vol, ctxs in mention_texts.items():
        volume_combined[vol] = ' '.join(ctxs)

    for ds1 in ds1_sites:
        vol = ds1['volume']

        # Build search terms from multiple fields
        search_terms = set()

        # From site_name: extract distinctive words (skip common ones)
        skip_words = {'buried', 'temple', 'the', 'of', 'at', 'near', 'with', 'and',
                       'in', 'from', 'for', 'a', 'an', 'collection', 'generic',
                       'observation', 'identification', 'excavation', 'depth'}
        for word in ds1['site_name'].split():
            w = word.strip('().,').lower()
            if len(w) >= 4 and w not in skip_words:
                search_terms.add(w)

        # From modern_name: split and add distinctive words
        for word in ds1['modern_name'].split():
            w = word.strip('().,/').lower()
            if len(w) >= 4 and w not in skip_words:
                search_terms.add(w)
                # Candi → Tjandi variant
                if w.startswith('candi'):
                    search_terms.add(w.replace('candi', 'tjandi'))

        # Dutch colonial name variants
        for term in list(search_terms):
            # oe→u, dj→j, tj→c mappings (modern→colonial)
            if 'u' in term:
                search_terms.add(term.replace('u', 'oe'))
            if 'j' in term and not term.startswith('dj'):
                search_terms.add('dj' + term[1:] if term[0] == 'j' else term)

        # Search in the correct volume first, then all volumes
        search_vols = [vol] if vol in volume_combined else list(volume_combined.keys())

        for sv in search_vols:
            combined = volume_combined.get(sv, '')
            for term in search_terms:
                if term in combined:
                    ds1['found'] = True
                    ds1['match_details'] = f"found '{term}' in {sv}"
                    break
            if ds1['found']:
                break

    found_count = sum(1 for d in ds1_sites if d['found'])
    total = len(ds1_sites)

    return {
        'total_ds1_entries': total,
        'found_in_extraction': found_count,
        'coverage_pct': round(100 * found_count / total, 1) if total > 0 else 0,
        'missing': [d['site_name'] for d in ds1_sites if not d['found']],
        'details': ds1_sites
    }


def main():
    print("=" * 70)
    print("E091: AUTOMATED NLP EXTRACTION FROM OV COLONIAL REPORTS")
    print("=" * 70)

    # Find all OV volumes
    ov_files = sorted(OV_DIR.glob("OV_*_fulltext.txt"))
    if not ov_files:
        print(f"ERROR: No OV files found in {OV_DIR}")
        return

    print(f"\nFound {len(ov_files)} OV volumes:")
    for f in ov_files:
        print(f"  {f.name}")

    # ── Extract from all volumes ───────────────────────────────────────
    print("\n--- Phase 1: Extraction ---")
    all_mentions = []
    volume_stats = {}

    for ov_file in ov_files:
        print(f"\n  Processing {ov_file.name}...", end=" ")
        mentions, n_paras, n_chars = extract_from_volume(ov_file)
        all_mentions.extend(mentions)

        vol_name = ov_file.stem.replace('_fulltext', '')
        cats = Counter(m['category'] for m in mentions)
        volume_stats[vol_name] = {
            'n_paragraphs': n_paras,
            'n_chars': n_chars,
            'n_mentions': len(mentions),
            'categories': dict(cats)
        }
        print(f"{len(mentions)} mentions ({n_paras} paragraphs, {n_chars:,} chars)")
        for cat, count in sorted(cats.items()):
            print(f"    {cat}: {count}")

    print(f"\n  TOTAL: {len(all_mentions)} mentions across {len(ov_files)} volumes")

    # ── Categorize outputs ─────────────────────────────────────────────
    print("\n--- Phase 2: Categorization ---")

    depth_mentions = [m for m in all_mentions if m['category'] == 'depth' and m['depth_m'] is not None]
    burial_mentions = [m for m in all_mentions if m['category'] == 'burial']
    volcanic_mentions = [m for m in all_mentions if m['category'] == 'volcanic']
    site_mentions = [m for m in all_mentions if m['category'] == 'site']
    material_mentions = [m for m in all_mentions if m['category'] == 'material']
    location_mentions = [m for m in all_mentions if m['category'] == 'location']

    # Co-occurrence: paragraphs with ≥2 categories
    cooccurrence_mentions = [m for m in all_mentions if m['cooccurrence_count'] >= 2]
    high_value = [m for m in all_mentions if m['cooccurrence_count'] >= 3]

    print(f"  Depth (numeric): {len(depth_mentions)}")
    print(f"  Burial (qualitative): {len(burial_mentions)}")
    print(f"  Volcanic: {len(volcanic_mentions)}")
    print(f"  Sites: {len(site_mentions)}")
    print(f"  Materials: {len(material_mentions)}")
    print(f"  Locations: {len(location_mentions)}")
    print(f"  Co-occurrence (≥2 cats): {len(cooccurrence_mentions)}")
    print(f"  High-value (≥3 cats): {len(high_value)}")

    # Depth statistics
    if depth_mentions:
        depths = [m['depth_m'] for m in depth_mentions]
        print(f"\n  Depth statistics:")
        print(f"    Range: {min(depths):.2f} - {max(depths):.2f} m")
        print(f"    Mean: {sum(depths)/len(depths):.2f} m")
        print(f"    Median: {sorted(depths)[len(depths)//2]:.2f} m")

    # Volcanic distribution
    if volcanic_mentions:
        volcano_dist = Counter(m['subcategory'] for m in volcanic_mentions)
        print(f"\n  Volcano mentions:")
        for v, c in volcano_dist.most_common(15):
            print(f"    {v}: {c}")

    # Site name extraction
    if site_mentions:
        site_names = Counter()
        for m in site_mentions:
            if m['captured_value']:
                site_names[m['captured_value']] += 1
            else:
                site_names[m['subcategory']] += 1
        print(f"\n  Top site references:")
        for s, c in site_names.most_common(20):
            print(f"    {s}: {c}")

    # ── Save CSV outputs ───────────────────────────────────────────────
    print("\n--- Phase 3: Saving outputs ---")

    csv_fields = ['volume', 'year', 'paragraph_idx', 'page_est', 'category',
                  'subcategory', 'description', 'matched_text', 'captured_value',
                  'depth_m', 'cooccurrence', 'cooccurrence_count', 'context']

    def save_csv(mentions, filename):
        path = RESULTS_DIR / filename
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
            writer.writeheader()
            for m in mentions:
                writer.writerow(m)
        print(f"  Saved: {path.name} ({len(mentions)} rows)")

    save_csv(all_mentions, 'ov_mentions.csv')
    save_csv(depth_mentions, 'ov_depth_mentions.csv')
    save_csv(volcanic_mentions, 'ov_volcanic_events.csv')
    save_csv(site_mentions, 'ov_site_mentions.csv')
    save_csv(cooccurrence_mentions, 'ov_cooccurrence.csv')

    # ── Cross-validation against DS-1 ──────────────────────────────────
    print("\n--- Phase 4: Cross-validation against DS-1 ---")
    ds1_entries = load_ds1()
    validation = validate_against_ds1(all_mentions, ds1_entries)

    if validation:
        print(f"  DS-1 entries: {validation['total_ds1_entries']}")
        print(f"  Found in extraction: {validation['found_in_extraction']}")
        print(f"  Coverage: {validation['coverage_pct']}%")
        if validation['missing']:
            print(f"  Missing ({len(validation['missing'])}):")
            for site in validation['missing'][:15]:
                print(f"    - {site}")

    # ── Summary statistics ─────────────────────────────────────────────
    print("\n--- Phase 5: Summary ---")

    # Unique paragraphs with archaeological content
    unique_depth_paras = len(set((m['volume'], m['paragraph_idx']) for m in depth_mentions))
    unique_volcanic_paras = len(set((m['volume'], m['paragraph_idx']) for m in volcanic_mentions))
    unique_cooccur_paras = len(set((m['volume'], m['paragraph_idx']) for m in cooccurrence_mentions))

    stats = {
        'experiment': 'E091',
        'title': 'Automated NLP Extraction from OV Colonial Reports',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'input': {
            'n_volumes': len(ov_files),
            'volumes': [f.stem.replace('_fulltext', '') for f in ov_files],
            'total_lines': sum(vs['n_chars'] for vs in volume_stats.values()),
        },
        'extraction_totals': {
            'total_mentions': len(all_mentions),
            'depth_numeric': len(depth_mentions),
            'burial_qualitative': len(burial_mentions),
            'volcanic': len(volcanic_mentions),
            'sites': len(site_mentions),
            'materials': len(material_mentions),
            'locations': len(location_mentions),
            'cooccurrence_2plus': len(cooccurrence_mentions),
            'high_value_3plus': len(high_value),
        },
        'unique_paragraphs': {
            'with_depth': unique_depth_paras,
            'with_volcanic': unique_volcanic_paras,
            'with_cooccurrence': unique_cooccur_paras,
        },
        'depth_stats': {
            'n_values': len(depth_mentions),
            'min_m': round(min(m['depth_m'] for m in depth_mentions), 2) if depth_mentions else None,
            'max_m': round(max(m['depth_m'] for m in depth_mentions), 2) if depth_mentions else None,
            'mean_m': round(sum(m['depth_m'] for m in depth_mentions) / len(depth_mentions), 2) if depth_mentions else None,
        },
        'volcano_distribution': dict(Counter(m['subcategory'] for m in volcanic_mentions).most_common()),
        'volume_stats': volume_stats,
        'ds1_validation': {
            'total_ds1': validation.get('total_ds1_entries', 0),
            'found': validation.get('found_in_extraction', 0),
            'coverage_pct': validation.get('coverage_pct', 0),
            'missing': validation.get('missing', []),
        } if validation else {},
    }

    stats_path = RESULTS_DIR / 'ov_extraction_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {stats_path.name}")

    # ── Final report ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("E091 EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"  Volumes processed: {len(ov_files)}")
    print(f"  Total mentions: {len(all_mentions)}")
    print(f"  Depth values extracted: {len(depth_mentions)}")
    print(f"  Volcanic references: {len(volcanic_mentions)}")
    print(f"  Co-occurrence paragraphs: {unique_cooccur_paras}")
    if validation:
        print(f"  DS-1 coverage: {validation['coverage_pct']}% ({validation['found_in_extraction']}/{validation['total_ds1_entries']})")
    print(f"\n  vs DS-1 manual extraction: 52 entries")
    print(f"  Automated extraction advantage: {len(depth_mentions)}/{52} = {len(depth_mentions)/52:.1f}x")
    print("=" * 70)


if __name__ == '__main__':
    main()

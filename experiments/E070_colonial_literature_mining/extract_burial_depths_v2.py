"""
E070: Enhanced extraction of archaeological site data from colonial OV reports.
V2: Expanded Dutch patterns for burial depth, site identification, volcanic context.
Outputs structured CSV for systematic colonial site register construction.
"""
import os
import re
import csv
from pathlib import Path
from collections import defaultdict

OV_DIR = Path("data/raw/colonial_sources/OV")
RESULTS_DIR = Path("experiments/E070_colonial_literature_mining/results")

# === DEPTH / BURIAL PATTERNS ===
# Each tuple: (regex, type_tag, description)
# Capture group 1 = numeric depth value where applicable
DEPTH_PATTERNS = [
    # Direct depth measurements (meters)
    (r'(\d+[\.,]\d+)\s*[Mm]\.?\s*diep', 'depth_m', 'X.X M diep'),
    (r'diepte\s*van\s*(\d+[\.,]\d+)\s*[Mm]', 'depth_m', 'diepte van X.X M'),
    (r'op\s*(?:een|eene)?\s*diepte\s*van\s*(\d+[\.,]?\d*)\s*[Mm]?', 'depth_m', 'op een diepte van X'),
    (r'(\d+[\.,]\d+)\s*[Mm]\.?\s*onder\s*(?:den?|het)\s*(?:grond|maaiveld|oppervlak)', 'depth_m', 'X M onder den grond'),
    (r'(\d+[\.,]\d+)\s*[Mm]\.?\s*beneden\s*(?:den?|het)\s*(?:grond|maaiveld)', 'depth_m', 'X M beneden den grond'),
    (r'uitgegraven\s*tot\s*(\d+[\.,]?\d*)\s*[Mm]', 'depth_m', 'uitgegraven tot X M'),
    (r'tot\s*(\d+[\.,]?\d*)\s*[Mm]\.?\s*(?:af)?gegraven', 'depth_m', 'tot X M afgegraven'),
    (r'(\d+[\.,]\d+)\s*[Mm]\.?\s*(?:lager|dieper)', 'depth_m', 'X M lager/dieper'),
    (r'grond\s*tot\s*(\d+[\.,]?\d*)\s*[Mm]', 'depth_m', 'grond tot X M'),
    # Depth in voet (1 voet ~ 0.3048 m)
    (r'(\d+)\s*(?:voet|voeten)\s*(?:diep|onder|beneden)', 'depth_voet', 'X voet diep'),
    (r'(\d+)\s*(?:voet|voeten)\s*(?:in|onder)\s*den?\s*grond', 'depth_voet', 'X voet in den grond'),
    # Depth in vadem (1 vadem ~ 1.7 m)
    (r'(\d+)\s*(?:vadem|vademen)\s*diep', 'depth_vadem', 'X vadem diep'),
    # Depth in el (1 el ~ 0.69 m)
    (r'(\d+)\s*(?:el|ellen)\s*diep', 'depth_el', 'X el diep'),
    # Qualitative burial language
    (r'(?:onder|in)\s*den?\s*grond\s*(?:bedolven|begraven|geraak|gezakt|gevonden|verzonken|weggezonken)', 'burial', 'buried/sunk in ground'),
    (r'geheel\s*(?:in|onder)\s*den?\s*grond', 'burial', 'completely underground'),
    (r'boven\s*den?\s*grond\s*(?:uitstekend|zichtbaar|uitkomend)', 'protrusion', 'protruding above ground'),
    (r'(?:gedeeltelijk|ten\s*deele|grootendeels)\s*(?:in|onder)\s*den?\s*grond', 'burial_partial', 'partially underground'),
    (r'(?:verzakt|verzonken|weggezonken|ingezonken)\s', 'subsidence', 'sunk/subsided'),
    (r'den?\s*grond\s*(?:in\s*)?gezakt', 'subsidence', 'sunk into ground'),
    (r'bedekt\s*(?:met|door)\s*(?:aarde|grond|modder|asch|puin|slib|zand)', 'covered', 'covered by deposits'),
    (r'(?:blootgelegd|bloot\s*gelegd|te\s*voorschijn\s*gebracht)', 'exposed', 'exposed/uncovered'),
    (r'diep\s*gelegen', 'deep_situated', 'deeply situated'),
    (r'(?:aan\s*het\s*licht|te\s*voorschijn)\s*(?:gekomen|gebracht|geraakt)', 'discovered', 'brought to light'),
    (r'(?:opgegraven|uitgegraven)\s', 'excavated', 'excavated'),
]

# === VOLCANIC CONTEXT PATTERNS ===
VOLCANIC_PATTERNS = [
    (r'(?:vulkanisch|vulkaan|vulcanisch)\w*', 'volcanic_general', 'volcanic reference'),
    (r'(?:lava|lavastroom|lavabed)\w*', 'lava', 'lava'),
    (r'(?:lahar|modderstroom|modderlaag|moddervloed)\w*', 'lahar', 'lahar/mudflow'),
    (r'(?:asch(?:laag|regen)?|puimsteenlaag|puimsteen|tufsteen|tuf(?:laag)?)\w*', 'tephra', 'tephra/ash/tuff'),
    (r'(?:eruptie|uitbarsting|vulkanische?\s*uitbarsting)\w*', 'eruption', 'eruption'),
    (r'krater\w*', 'crater', 'crater'),
    (r'(?:Kloet|Keloed|Kelut)', 'v_kelud', 'Kelud'),
    (r'Merapi', 'v_merapi', 'Merapi'),
    (r'(?:Smeroe|Semeru)', 'v_semeru', 'Semeru'),
    (r'Bromo', 'v_bromo', 'Bromo'),
    (r'(?:Ardjoeno|Arjuno)', 'v_arjuno', 'Arjuno'),
    (r'Welirang', 'v_welirang', 'Welirang'),
    (r'(?:Raoen|Raung)', 'v_raung', 'Raung'),
    (r'(?:Idjen|Ijen)', 'v_ijen', 'Ijen'),
    (r'Lamongan', 'v_lamongan', 'Lamongan'),
    (r'Penanggungan', 'v_penanggungan', 'Penanggungan'),
    (r'Ringgit', 'v_ringgit', 'Ringgit'),
    (r'Tengger', 'v_tengger', 'Tengger'),
    (r'Wilis', 'v_wilis', 'Wilis'),
]

# === SITE IDENTIFICATION PATTERNS ===
SITE_PATTERNS = [
    (r'[Tt]jandi\s+(\w+(?:\s+\w+)?)', 'tjandi', 'Tjandi (= Candi)'),
    (r'[Cc]andi\s+(\w+(?:\s+\w+)?)', 'candi', 'Candi'),
    (r'[Tt]empel(?:\s+(?:van|te|bij)\s+)?(\w+)?', 'tempel', 'Temple'),
    (r'[Hh]eiligdom(?:\s+(?:van|te|bij)\s+)?(\w+)?', 'heiligdom', 'Sanctuary'),
    (r'[Rr]u[ii]ne(?:\s+(?:van|te|bij)\s+)?(\w+)?', 'ruine', 'Ruin'),
    (r'[Mm]onument\w*', 'monument', 'Monument'),
    (r'[Ff]undament\w*', 'fundament', 'Foundation'),
    (r'[Oo]ntgraving\w*', 'ontgraving', 'Excavation'),
    (r'[Oo]pgraving\w*', 'opgraving', 'Excavation'),
    (r'[Bb]aksteenen?\s*(?:gebouw|muur|fundament|bouwwerk|tempel)?', 'baksteen', 'Brick structure'),
    (r'[Ii]nscriptie\w*', 'inscriptie', 'Inscription'),
    (r'[Bb]eeldhouwwerk\w*', 'beeldhouwwerk', 'Sculpture'),
    (r'[Bb]eeld(?:je)?\s+(?:van\s+)?(?:Gane[sc]a|Nandi|Mahakala|Durga|Wisnu|Siwa|Brahma|\w+)', 'statue', 'Statue'),
    (r'[Yy]oni\w*', 'yoni', 'Yoni'),
    (r'[Ll]ingga\w*', 'lingga', 'Lingga'),
    (r'(?:Gane[sc]a|Nandi|Mahakala|Durga|Wisnu|[CS]iwa|Brahma)\s*(?:beeld)?', 'deity', 'Hindu deity'),
    (r'[Ss]teenen?\s*(?:met|beeld|plaat)', 'stone_obj', 'Stone object'),
    (r'(?:koper|brons|bronzen|gouden|zilveren)\s*(?:beeld|plaat|voorwerp)', 'metal_obj', 'Metal object'),
]

# === LOCATION PATTERNS ===
LOCATION_PATTERNS = [
    (r'[Dd]esa\s+(\w+(?:\s+\w+)?)', 'desa', 'Village'),
    (r'[Dd]essa\s+(\w+(?:\s+\w+)?)', 'dessa', 'Village (old spelling)'),
    (r'[Oo]nderneming\s+(\w+(?:\s+\w+)?)', 'onderneming', 'Plantation'),
    (r'[Rr]esidentie\s+(\w+)', 'residentie', 'Residency'),
    (r'[Aa]fdeeling\s+(\w+)', 'afdeeling', 'Division'),
    (r'[Rr]egentschap\s+(\w+)', 'regentschap', 'Regency'),
    (r'[Dd]istrict\s+(\w+)', 'district', 'District'),
]


def extract_context(text, match_start, match_end, context_chars=500):
    """Extract surrounding context for a match."""
    start = max(0, match_start - context_chars)
    end = min(len(text), match_end + context_chars)
    return text[start:end].replace('\n', ' ').replace('\r', ' ').strip()


def find_nearby(context, patterns):
    """Find all pattern matches within a context string."""
    found = []
    for pattern, ptype, desc in patterns:
        for m in re.finditer(pattern, context, re.IGNORECASE):
            found.append({'type': ptype, 'match': m.group().strip(), 'desc': desc})
    return found


def estimate_page(text, position, chars_per_page=3000):
    """Rough page estimate based on character position."""
    return position // chars_per_page + 1


def parse_depth_value(match_text, depth_type, context=''):
    """Extract numeric depth in meters from match text and type.
    Checks context for 'c.M.' (centimeters) vs 'M.' (meters).
    """
    # Find first number in the match
    num_match = re.search(r'(\d+[\.,]?\d*)', match_text)
    if not num_match:
        return None, ''

    val = float(num_match.group(1).replace(',', '.'))
    depth_str = num_match.group(1)

    # Check if the number is followed by c.M. (centimeters) in the context
    # Search for patterns like "60 c.M." or "60 cM" or "60c.M."
    if context:
        cm_pattern = re.compile(re.escape(depth_str) + r'\s*c[\.\s]*[Mm][\.\s]')
        if cm_pattern.search(context):
            val = val / 100.0

    if 'voet' in depth_type:
        return round(val * 0.3048, 2), 'voet'
    elif 'vadem' in depth_type:
        return round(val * 1.7, 2), 'vadem'
    elif 'el' in depth_type:
        return round(val * 0.69, 2), 'el'
    elif depth_type.startswith('depth_m'):
        return round(val, 2), 'meter'
    return None, ''


def search_ov_volume(filepath):
    """Search a single OV volume for burial/depth/volcanic-archaeological references."""
    year_match = re.search(r'OV_(\d{4})', filepath.name)
    year = year_match.group(1) if year_match else "unknown"

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    results = []
    seen_positions = set()  # Dedup by position bucket

    # --- Pass 1: Depth/burial patterns ---
    for pattern, ptype, desc in DEPTH_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            pos_key = match.start() // 100
            if pos_key in seen_positions:
                continue
            seen_positions.add(pos_key)

            context = extract_context(text, match.start(), match.end())
            nearby_sites = find_nearby(context, SITE_PATTERNS)
            nearby_volcanic = find_nearby(context, VOLCANIC_PATTERNS)
            nearby_locations = find_nearby(context, LOCATION_PATTERNS)

            depth_m, depth_unit = parse_depth_value(match.group(), ptype, context)

            # Best site name
            site_name = ''
            for s in nearby_sites:
                if s['type'] in ('tjandi', 'candi', 'tempel', 'heiligdom'):
                    site_name = s['match']
                    break
            if not site_name and nearby_sites:
                site_name = nearby_sites[0]['match']

            # Best location
            location = ''
            for loc in nearby_locations:
                if loc['type'] in ('desa', 'dessa', 'residentie', 'regentschap'):
                    location = loc['match']
                    break
            if not location and nearby_locations:
                location = nearby_locations[0]['match']

            results.append({
                'volume': f'OV_{year}',
                'year': int(year) if year.isdigit() else 0,
                'match_type': ptype,
                'match_desc': desc,
                'match_text': match.group()[:100],
                'depth_m': depth_m,
                'depth_unit': depth_unit,
                'site_name': site_name,
                'has_site': len(nearby_sites) > 0,
                'has_volcanic': len(nearby_volcanic) > 0,
                'location': location,
                'sites_nearby': '; '.join(s['match'] for s in nearby_sites[:5]),
                'volcanic_nearby': '; '.join(v['match'] for v in nearby_volcanic[:5]),
                'locations_nearby': '; '.join(l['match'] for l in nearby_locations[:5]),
                'page_est': estimate_page(text, match.start()),
                'position': match.start(),
                'context': context[:800],
            })

    # --- Pass 2: Volcanic patterns with archaeological co-occurrence ---
    for pattern, ptype, desc in VOLCANIC_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            pos_key = match.start() // 100
            if pos_key in seen_positions:
                continue

            context = extract_context(text, match.start(), match.end())
            nearby_sites = find_nearby(context, SITE_PATTERNS)

            # Only keep volcanic matches that have archaeological context
            if not nearby_sites:
                continue

            seen_positions.add(pos_key)
            nearby_locations = find_nearby(context, LOCATION_PATTERNS)

            site_name = ''
            for s in nearby_sites:
                if s['type'] in ('tjandi', 'candi', 'tempel'):
                    site_name = s['match']
                    break
            if not site_name:
                site_name = nearby_sites[0]['match']

            location = ''
            for loc in nearby_locations:
                if loc['type'] in ('desa', 'dessa', 'residentie', 'regentschap'):
                    location = loc['match']
                    break
            if not location and nearby_locations:
                location = nearby_locations[0]['match']

            results.append({
                'volume': f'OV_{year}',
                'year': int(year) if year.isdigit() else 0,
                'match_type': f'volcanic_arch_{ptype}',
                'match_desc': f'Volcanic+archaeological: {desc}',
                'match_text': match.group()[:100],
                'depth_m': None,
                'depth_unit': '',
                'site_name': site_name,
                'has_site': True,
                'has_volcanic': True,
                'location': location,
                'sites_nearby': '; '.join(s['match'] for s in nearby_sites[:5]),
                'volcanic_nearby': match.group(),
                'locations_nearby': '; '.join(l['match'] for l in nearby_locations[:5]),
                'page_est': estimate_page(text, match.start()),
                'position': match.start(),
                'context': context[:800],
            })

    return results


def main():
    all_results = []

    print("=" * 70)
    print("E070 Enhanced OV Extraction v2")
    print("Patterns: depth/burial + volcanic-archaeological co-occurrence")
    print("=" * 70)

    for filepath in sorted(OV_DIR.glob("OV_*_fulltext.txt")):
        print(f"\nScanning {filepath.name}...")
        results = search_ov_volume(filepath)
        all_results.extend(results)

        n_depth = sum(1 for r in results if r['depth_m'] is not None)
        n_burial = sum(1 for r in results if r['match_type'] in ('burial', 'burial_partial', 'subsidence', 'covered'))
        n_volcanic = sum(1 for r in results if 'volcanic_arch' in r['match_type'])
        n_site = sum(1 for r in results if r['has_site'])
        n_priority = sum(1 for r in results if r['has_site'] and (r['depth_m'] is not None or r['has_volcanic']))

        print(f"  {len(results):3d} matches | {n_depth} depth | {n_burial} burial | {n_volcanic} volc+arch | {n_site} w/site | {n_priority} PRIORITY")

    # --- Write full CSV ---
    csv_file = RESULTS_DIR / "ov_extraction_v2.csv"
    fieldnames = [
        'volume', 'year', 'match_type', 'match_desc', 'match_text',
        'depth_m', 'depth_unit', 'site_name', 'has_site', 'has_volcanic',
        'location', 'sites_nearby', 'volcanic_nearby', 'locations_nearby',
        'page_est', 'context'
    ]

    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(all_results, key=lambda x: (x['year'], x['position'])):
            row = {k: v for k, v in r.items() if k in fieldnames}
            writer.writerow(row)

    # --- Write PRIORITY matches (site context + depth or volcanic) ---
    priority = [r for r in all_results
                if r['has_site'] and (r['depth_m'] is not None or r['has_volcanic'])]

    priority_file = RESULTS_DIR / "ov_priority_matches_v2.csv"
    with open(priority_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(priority, key=lambda x: (x['year'], x['position'])):
            row = {k: v for k, v in r.items() if k in fieldnames}
            writer.writerow(row)

    # --- Write DEPTH-ONLY matches (have numeric depth value) ---
    depth_matches = [r for r in all_results if r['depth_m'] is not None]
    depth_file = RESULTS_DIR / "ov_depth_values_v2.csv"
    with open(depth_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(depth_matches, key=lambda x: (x['year'], x['position'])):
            row = {k: v for k, v in r.items() if k in fieldnames}
            writer.writerow(row)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Volumes scanned:       {len(list(OV_DIR.glob('OV_*_fulltext.txt')))}")
    print(f"Total matches:         {len(all_results)}")
    print(f"With depth values:     {len(depth_matches)}")
    print(f"Burial language:       {sum(1 for r in all_results if r['match_type'] in ('burial','burial_partial','subsidence','covered'))}")
    print(f"Volcanic+arch:         {sum(1 for r in all_results if 'volcanic_arch' in r['match_type'])}")
    print(f"With site context:     {sum(1 for r in all_results if r['has_site'])}")
    print(f"PRIORITY (site+depth/volc): {len(priority)}")

    print(f"\nOutputs:")
    print(f"  Full CSV:     {csv_file}")
    print(f"  Priority CSV: {priority_file}")
    print(f"  Depth CSV:    {depth_file}")

    print(f"\nPriority matches by volume:")
    for vol in sorted(set(r['volume'] for r in priority)):
        n = sum(1 for r in priority if r['volume'] == vol)
        sites = set(r['site_name'] for r in priority if r['volume'] == vol and r['site_name'])
        print(f"  {vol}: {n} matches — {', '.join(sites) if sites else 'no named sites'}")

    if depth_matches:
        print(f"\nDepth values found:")
        for r in sorted(depth_matches, key=lambda x: (x['depth_m'] or 0), reverse=True):
            print(f"  {r['volume']} p.{r['page_est']}: {r['depth_m']}m ({r['depth_unit']}) — {r['site_name'] or r['match_text'][:50]}")


if __name__ == "__main__":
    main()

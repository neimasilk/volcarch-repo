"""
E093 × E070 Cross-Reference Analysis
=====================================
Programmatically cross-references E093 Indonesian lit database (65 entries)
with E070 colonial site register (52 entries) to identify:
1. Shared sites (same site in both datasets)
2. Shared regions (publications covering E070 volcanic zones)
3. Publications with NEW depth data not captured in E070
4. Geographic proximity between E093 mentioned sites and E070 georeferenced sites
"""

import pandas as pd
import csv
import re
import json
from difflib import SequenceMatcher
from collections import defaultdict

# --- Load E070 colonial register ---
e070_path = "experiments/E070_colonial_literature_mining/results/colonial_site_register_v1.0.csv"
e070 = pd.read_csv(e070_path)

# --- Load E093 literature database ---
e093_path = "experiments/E093_indonesian_lit_mining/results/indonesian_lit_database.csv"
e093 = pd.read_csv(e093_path)

print("=" * 70)
print("E093 × E070 CROSS-REFERENCE ANALYSIS")
print("=" * 70)
print(f"\nE070 colonial register: {len(e070)} entries")
print(f"E093 literature database: {len(e093)} entries")

# --- 1. SITE NAME MATCHING ---
# Extract all site names from E070
e070_sites = set()
for _, row in e070.iterrows():
    if pd.notna(row.get('site_name')):
        e070_sites.add(str(row['site_name']).lower().strip())
    if pd.notna(row.get('modern_name')):
        e070_sites.add(str(row['modern_name']).lower().strip())

# Extract all site names from E093
e093_site_mentions = {}
for _, row in e093.iterrows():
    if pd.notna(row.get('site_names_mentioned')):
        sites = [s.strip().lower() for s in str(row['site_names_mentioned']).split(',')]
        e093_site_mentions[f"{row['author']} ({row['year']})"] = {
            'sites': sites,
            'title': row['title'],
            'has_stratigraphy': row.get('has_stratigraphy', 'no'),
            'burial_depth_mentioned': row.get('burial_depth_mentioned', 'no'),
            'volcanic_zone': row.get('volcanic_zone', 'no'),
            'relevance': row.get('relevance_to_volcarch', 'low'),
        }

# Key site name variants for fuzzy matching
site_aliases = {
    'trowulan': ['trowulan', 'majapahit', 'badjangratoe', 'panggoeng', 'bale kambang'],
    'sambisari': ['sambisari', 'tjandi sambisari'],
    'kedulan': ['kedulan', 'tjandi kedulan'],
    'liyangan': ['liyangan', 'liangan'],
    'prambanan': ['prambanan', 'tjandi prambanan'],
    'borobudur': ['borobudur', 'barabudur'],
    'panataran': ['panataran', 'tjandi panataran'],
    'gedong sanga': ['gedong sanga'],
    'sangiran': ['sangiran'],
    'dieng': ['dieng'],
    'kumitir': ['kumitir'],
    'kimpulan': ['kimpulan'],
}

def normalize_site(name):
    """Normalize site name for matching."""
    name = name.lower().strip()
    name = re.sub(r'candi\s+', '', name)
    name = re.sub(r'tjandi\s+', '', name)
    name = re.sub(r'situs\s+', '', name)
    return name

def find_site_match(site_name, reference_sites, aliases):
    """Check if a site name matches any reference site via aliases or fuzzy."""
    norm = normalize_site(site_name)

    # Direct match
    for ref in reference_sites:
        ref_norm = normalize_site(ref)
        if norm == ref_norm or norm in ref_norm or ref_norm in norm:
            return ref, 'direct'

    # Alias match
    for canonical, alias_list in aliases.items():
        if any(a in norm for a in alias_list):
            for ref in reference_sites:
                ref_norm = normalize_site(ref)
                if any(a in ref_norm for a in alias_list):
                    return ref, 'alias'

    # Fuzzy match (threshold 0.7)
    for ref in reference_sites:
        ratio = SequenceMatcher(None, norm, normalize_site(ref)).ratio()
        if ratio > 0.7:
            return ref, f'fuzzy({ratio:.2f})'

    return None, None

print("\n" + "=" * 70)
print("1. SITE-LEVEL MATCHES (E093 publications mentioning E070 sites)")
print("=" * 70)

matches = []
for pub, info in e093_site_mentions.items():
    for site in info['sites']:
        match, method = find_site_match(site, e070_sites, site_aliases)
        if match:
            matches.append({
                'publication': pub,
                'e093_site': site,
                'e070_match': match,
                'method': method,
                'has_stratigraphy': info['has_stratigraphy'],
                'burial_depth': info['burial_depth_mentioned'],
                'relevance': info['relevance'],
            })

if matches:
    for m in matches:
        print(f"\n  {m['publication']}")
        print(f"    E093 site: {m['e093_site']} -> E070: {m['e070_match']} ({m['method']})")
        print(f"    Stratigraphy: {m['has_stratigraphy']} | Burial depth: {m['burial_depth']} | Relevance: {m['relevance']}")
else:
    print("  No direct site matches found.")

print(f"\n  TOTAL: {len(matches)} site-level matches across {len(set(m['publication'] for m in matches))} publications")

# --- 2. REGIONAL OVERLAP ---
print("\n" + "=" * 70)
print("2. REGIONAL OVERLAP (E093 pubs covering E070 volcanic systems)")
print("=" * 70)

# E070 volcanic systems
e070_volcanics = set()
for v in e070['volcanic_system'].dropna():
    for part in str(v).split('/'):
        e070_volcanics.add(part.strip().lower())

# Volcano keywords in E093
volcano_keywords = {
    'merapi': 'Merapi',
    'kelud': 'Kelud', 'kelut': 'Kelud', 'kloet': 'Kelud',
    'arjuno': 'Arjuno-Welirang', 'welirang': 'Arjuno-Welirang',
    'semeru': 'Semeru', 'smeroe': 'Semeru',
    'sindoro': 'Sindoro/Sumbing', 'sumbing': 'Sindoro/Sumbing', 'sundoro': 'Sindoro/Sumbing',
    'dieng': 'Dieng',
    'bromo': 'Bromo-Tengger', 'tengger': 'Bromo-Tengger',
    'ungaran': 'Ungaran',
    'merbabu': 'Merbabu',
    'wilis': 'Wilis',
}

regional_matches = []
for _, row in e093.iterrows():
    pub = f"{row['author']} ({row['year']})"
    # Check topic keywords and site names for volcano mentions
    text = ' '.join([
        str(row.get('topic_keywords', '')),
        str(row.get('site_names_mentioned', '')),
        str(row.get('region', '')),
        str(row.get('notes', '')),
    ]).lower()

    matched_volcanoes = set()
    for kw, volcano in volcano_keywords.items():
        if kw in text:
            matched_volcanoes.add(volcano)

    if matched_volcanoes:
        regional_matches.append({
            'publication': pub,
            'volcanoes': matched_volcanoes,
            'title': row['title'][:60],
            'has_stratigraphy': row.get('has_stratigraphy', 'no'),
            'burial_depth': row.get('burial_depth_mentioned', 'no'),
        })

for rm in regional_matches:
    volc_str = ', '.join(rm['volcanoes'])
    print(f"\n  {rm['publication']}")
    print(f"    Volcanoes: {volc_str}")
    print(f"    Title: {rm['title']}...")
    print(f"    Stratigraphy: {rm['has_stratigraphy']} | Burial depth: {rm['burial_depth']}")

print(f"\n  TOTAL: {len(regional_matches)} publications cover E070 volcanic systems")

# --- 3. NEW DEPTH DATA CANDIDATES ---
print("\n" + "=" * 70)
print("3. NEW DEPTH DATA CANDIDATES (E093 pubs with burial depth NOT in E070)")
print("=" * 70)

# E070 source years
e070_sources = set()
for _, row in e070.iterrows():
    e070_sources.add(str(row.get('source', '')).lower())

depth_candidates = []
for _, row in e093.iterrows():
    pub = f"{row['author']} ({row['year']})"
    if str(row.get('burial_depth_mentioned', 'no')).lower() == 'yes':
        # Check if this publication's data is likely already in E070
        year = row['year']
        is_ov = 'ov' in str(row.get('journal_or_source', '')).lower() or 'oudheidkundig' in str(row.get('journal_or_source', '')).lower()

        # If it's an OV source, it's likely already in E070
        if is_ov:
            in_e070 = "LIKELY IN E070 (OV source)"
        else:
            in_e070 = "POTENTIALLY NEW"

        depth_candidates.append({
            'publication': pub,
            'title': row['title'][:70],
            'region': row.get('region', ''),
            'sites': row.get('site_names_mentioned', ''),
            'status': in_e070,
            'has_stratigraphy': row.get('has_stratigraphy', 'no'),
            'volcanic_zone': row.get('volcanic_zone', 'no'),
            'relevance': row.get('relevance_to_volcarch', ''),
        })

new_candidates = [d for d in depth_candidates if d['status'] == 'POTENTIALLY NEW']
ov_overlap = [d for d in depth_candidates if 'LIKELY IN E070' in d['status']]

print(f"\n  Publications with burial depth mentioned: {len(depth_candidates)}")
print(f"  Likely already in E070 (OV sources): {len(ov_overlap)}")
print(f"  POTENTIALLY NEW depth data: {len(new_candidates)}")

print("\n  --- POTENTIALLY NEW DEPTH DATA ---")
for d in new_candidates:
    print(f"\n  {d['publication']} [{d['relevance'].upper()}]")
    print(f"    Title: {d['title']}")
    print(f"    Region: {d['region']} | Sites: {d['sites']}")
    print(f"    Stratigraphy: {d['has_stratigraphy']} | Volcanic: {d['volcanic_zone']}")

# --- 4. GEOGRAPHIC PROXIMITY ---
print("\n" + "=" * 70)
print("4. GEOGRAPHIC PROXIMITY ANALYSIS")
print("=" * 70)

# E070 has lat/lon for 43 entries. E093 has region names.
# Map E093 regions to approximate coordinates for proximity check
region_coords = {
    'yogyakarta': (-7.8, 110.4),
    'central java': (-7.5, 110.4),
    'east java': (-7.8, 112.5),
    'bali': (-8.3, 115.3),
    'west java': (-6.9, 107.6),
    'south sulawesi': (-5.1, 119.4),
    'east kalimantan': (0.5, 117.2),
    'maluku': (-3.7, 128.2),
    'lombok': (-8.6, 116.4),
    'sumbawa': (-8.5, 117.4),
}

# E070 sites by volcanic system with coords
e070_zones = defaultdict(list)
for _, row in e070.iterrows():
    if pd.notna(row.get('lat')) and pd.notna(row.get('lon')):
        v = str(row.get('volcanic_system', 'unknown'))
        e070_zones[v].append({
            'site': row['site_name'],
            'lat': row['lat'],
            'lon': row['lon'],
            'depth': row.get('burial_depth_m', ''),
        })

print("\n  E070 sites grouped by volcanic system:")
for zone, sites in sorted(e070_zones.items()):
    print(f"    {zone}: {len(sites)} sites")
    depths = [s['depth'] for s in sites if s['depth'] != '' and pd.notna(s['depth'])]
    if depths:
        print(f"      Depth range: {min(depths):.1f}–{max(depths):.1f} m")

# E093 publications with region in E070 volcanic zones
print("\n  E093 publications in E070 regions with potential for proximity analysis:")
e070_regions = {'east java', 'central java', 'yogyakarta'}
for _, row in e093.iterrows():
    region = str(row.get('region', '')).lower()
    if any(r in region for r in e070_regions):
        if str(row.get('burial_depth_mentioned', '')).lower() == 'yes':
            pub = f"{row['author']} ({row['year']})"
            print(f"    {pub} — {region} — DEPTH DATA")

# --- 5. SYNTHESIS ---
print("\n" + "=" * 70)
print("5. SYNTHESIS: ACTIONABLE CROSS-REFERENCE FINDINGS")
print("=" * 70)

print("""
A. CONFIRMED OVERLAPS (data already partially in E070):
   - OV-sourced publications in E093 overlap with E070's 52 entries
   - These may contain ADDITIONAL depth measurements not yet extracted
   - Priority: Re-read Bosch 1919, Stutterheim 1926, Tichelman 1929

B. NEW DEPTH DATA (NOT in E070):
""")

for i, d in enumerate(new_candidates, 1):
    print(f"   {i}. {d['publication']} — {d['sites']}")

print("""
C. VALIDATION CHAIN:
   E093 literature → E070 colonial register → E083 tephra correlation → E075 model

   New depth measurements from E093 pubs can:
   1. Expand E070 register beyond 52 entries (target: 70+)
   2. Provide independent calibration for E075 sedimentation model
   3. Strengthen P1 revision with additional colonial-era evidence

D. GPR FEASIBILITY:
   Only 1 published GPR study in volcanic Java (Pojoh 2007, Trowulan)
   GPR reached ~2-3m in Arjuno-zone andosols → BELOW E098's meta-analysis mean
   ERT recommended for deeper targets (E098: GPR max 1.5-2.5m in andosols)
""")

# --- Save results ---
results = {
    'site_matches': matches,
    'regional_matches': [{
        'publication': r['publication'],
        'volcanoes': list(r['volcanoes']),
        'title': r['title'],
    } for r in regional_matches],
    'new_depth_candidates': new_candidates,
    'summary': {
        'total_site_matches': len(matches),
        'publications_with_matches': len(set(m['publication'] for m in matches)),
        'regional_overlaps': len(regional_matches),
        'new_depth_candidates': len(new_candidates),
        'e070_entries': len(e070),
        'e093_entries': len(e093),
    }
}

output_path = "experiments/E093_indonesian_lit_mining/results/cross_reference_e070.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n  Results saved to: {output_path}")
print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

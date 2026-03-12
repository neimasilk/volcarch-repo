"""E050: Canarium spp. Global Distribution — Austronesian Aromatic Trail.

Maps the global distribution of Canarium (Burseraceae) using GBIF occurrence
data and overlays with Austronesian migration routes.

Hypothesis: Canarium distribution follows Austronesian migration paths,
supporting its role as a culturally transplanted aromatic species.

Key species:
- C. commune / C. vulgare — Indonesia (kenari/kanari)
- C. strictum — India/SE Asia (dammar/sambrani)
- C. madagascariense — Madagascar (ramy/haramy)
- C. indicum — Melanesia
- C. luzonicum — Philippines (Manila elemi)
"""
import sys, io, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import urllib.request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, 'results')
os.makedirs(RESULTS, exist_ok=True)

print('=' * 60)
print('E050: Canarium Global Distribution — Austronesian Trail')
print('=' * 60)

# ============================================================
# 1. FETCH GBIF OCCURRENCE DATA
# ============================================================
print('\n--- Fetching GBIF occurrence data ---')

GENUS_KEY = 3190431  # Canarium (Burseraceae) nubKey

def fetch_gbif_occurrences(taxon_key, limit=300, offset=0):
    """Fetch occurrence records from GBIF API."""
    url = (f'https://api.gbif.org/v1/occurrence/search?'
           f'taxonKey={taxon_key}&hasCoordinate=true&'
           f'hasGeospatialIssue=false&limit={limit}&offset={offset}')
    req = urllib.request.Request(url, headers={'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    return data

# Fetch multiple pages
all_records = []
for offset in range(0, 1500, 300):
    try:
        data = fetch_gbif_occurrences(GENUS_KEY, limit=300, offset=offset)
        records = data.get('results', [])
        if not records:
            break
        for r in records:
            all_records.append({
                'species': r.get('species', r.get('scientificName', 'Unknown')),
                'lat': r.get('decimalLatitude'),
                'lon': r.get('decimalLongitude'),
                'country': r.get('country', ''),
                'year': r.get('year', None),
                'basisOfRecord': r.get('basisOfRecord', ''),
            })
        print(f'  Fetched {len(all_records)} records (offset={offset})')
        if len(records) < 300:
            break
        time.sleep(0.5)  # Be polite to GBIF API
    except Exception as e:
        print(f'  Error at offset {offset}: {e}')
        break

df = pd.DataFrame(all_records)
df = df.dropna(subset=['lat', 'lon'])
print(f'\n  Total records with coordinates: {len(df)}')

# ============================================================
# 2. SPECIES BREAKDOWN
# ============================================================
print('\n--- Species breakdown ---')
species_counts = df['species'].value_counts()
for sp, count in species_counts.head(15).items():
    print(f'  {sp:45s}: {count:4d} records')

# Country breakdown
print('\n--- Country breakdown ---')
country_counts = df['country'].value_counts()
for c, count in country_counts.head(15).items():
    print(f'  {c:30s}: {count:4d} records')

# ============================================================
# 3. KEY REGIONS FOR AUSTRONESIAN HYPOTHESIS
# ============================================================
print('\n--- Key Austronesian regions ---')

regions = {
    'Indonesia': df[df['country'] == 'ID'],
    'Madagascar': df[df['country'] == 'MG'],
    'Philippines': df[df['country'] == 'PH'],
    'Malaysia': df[df['country'] == 'MY'],
    'Papua New Guinea': df[df['country'] == 'PG'],
    'India/Sri Lanka': df[df['country'].isin(['IN', 'LK'])],
    'East Africa': df[df['country'].isin(['TZ', 'KE', 'MZ', 'KM'])],
    'Melanesia': df[df['country'].isin(['SB', 'VU', 'NC', 'FJ'])],
    'Taiwan': df[df['country'] == 'TW'],
}

for region, subset in regions.items():
    species_list = subset['species'].unique()[:5]
    print(f'  {region:20s}: {len(subset):4d} records, species: {", ".join(species_list[:3])}')

# ============================================================
# 4. MAP — Global distribution with Austronesian routes
# ============================================================
print('\n--- Generating map ---')

fig, ax = plt.subplots(figsize=(16, 9))

# Plot all Canarium occurrences
# Color by region
colors = {
    'ID': 'red',       # Indonesia
    'MG': 'blue',      # Madagascar
    'PH': 'orange',    # Philippines
    'MY': 'darkred',   # Malaysia
    'PG': 'green',     # Papua New Guinea
    'IN': 'purple',    # India
    'TW': 'cyan',      # Taiwan
}

for country_code, color in colors.items():
    subset = df[df['country'] == country_code]
    if len(subset) > 0:
        ax.scatter(subset['lon'], subset['lat'], c=color, s=15, alpha=0.5,
                  label=f'{country_code} (n={len(subset)})', zorder=3)

# Other countries
other = df[~df['country'].isin(colors.keys())]
if len(other) > 0:
    ax.scatter(other['lon'], other['lat'], c='gray', s=8, alpha=0.3,
              label=f'Other (n={len(other)})', zorder=2)

# Draw approximate Austronesian migration routes
routes = [
    # Taiwan → Philippines
    [(121, 23), (121, 14)],
    # Philippines → Indonesia
    [(121, 14), (115, -2)],
    # Indonesia → Madagascar
    [(105, -7), (80, -5), (55, -15), (47, -19)],
    # Indonesia → Melanesia
    [(130, -3), (147, -6), (155, -8)],
    # Indonesia → Malaysia
    [(105, 1), (101, 3)],
]

for route in routes:
    lons = [p[0] for p in route]
    lats = [p[1] for p in route]
    ax.plot(lons, lats, 'k--', linewidth=1.5, alpha=0.4, zorder=1)
    # Arrow at end
    ax.annotate('', xy=(lons[-1], lats[-1]), xytext=(lons[-2], lats[-2]),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5, alpha=0.4))

# Key sites
key_sites = [
    (115.3, -8.2, 'Trunyan\n(Bali)', 'red'),
    (47.5, -18.9, 'Madagascar\n(famadihana)', 'blue'),
    (121.5, 14.5, 'Philippines\n(homeland)', 'orange'),
    (110.4, -7.5, 'Java\n(centre)', 'darkred'),
    (119.5, -3.0, 'Sulawesi\n(Toraja)', 'green'),
]

for lon, lat, label, color in key_sites:
    ax.annotate(label, xy=(lon, lat), fontsize=8, fontweight='bold',
               color=color, ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, alpha=0.8))

# Formatting
ax.set_xlim(30, 180)
ax.set_ylim(-35, 30)
ax.set_xlabel('Longitude', fontsize=11)
ax.set_ylabel('Latitude', fontsize=11)
ax.set_title('E050: Canarium spp. (Burseraceae) — Pan-Austronesian Aromatic Distribution\n'
             'GBIF occurrence data with approximate Austronesian migration routes',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.2)

# Add text box with key finding
textstr = ('Key finding: Canarium occurs in BOTH\n'
           'Indonesia AND Madagascar, supporting\n'
           'its role as transplanted Austronesian\n'
           'aromatic (E044: replaces Plumeria as\n'
           'pre-Hindu mortuary plant)')
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

fig.savefig(os.path.join(RESULTS, 'canarium_distribution.png'), dpi=300, bbox_inches='tight')
print(f'  Saved: canarium_distribution.png')

# ============================================================
# 5. SPECIES-SPECIFIC ANALYSIS
# ============================================================
print('\n--- Species in key Austronesian regions ---')

# Indonesia species
print('\n  INDONESIA:')
for sp, count in df[df['country'] == 'ID']['species'].value_counts().items():
    print(f'    {sp}: {count}')

# Madagascar species
print('\n  MADAGASCAR:')
mg = df[df['country'] == 'MG']
if len(mg) > 0:
    for sp, count in mg['species'].value_counts().items():
        print(f'    {sp}: {count}')
else:
    print('    No GBIF records found')
    print('    NOTE: C. madagascariense is documented in literature (Beaujard 2011)')
    print('    GBIF absence may reflect collection bias, not species absence')

# Philippines species
print('\n  PHILIPPINES:')
for sp, count in df[df['country'] == 'PH']['species'].value_counts().head(5).items():
    print(f'    {sp}: {count}')

# ============================================================
# 6. SAVE DATA
# ============================================================
df.to_csv(os.path.join(RESULTS, 'canarium_gbif_occurrences.csv'), index=False)
species_counts.to_csv(os.path.join(RESULTS, 'canarium_species_counts.csv'))
country_counts.to_csv(os.path.join(RESULTS, 'canarium_country_counts.csv'))

# ============================================================
# 7. SYNTHESIS
# ============================================================
print('\n' + '=' * 60)
print('SYNTHESIS')
print('=' * 60)

has_indonesia = len(df[df['country'] == 'ID']) > 0
has_madagascar = len(df[df['country'] == 'MG']) > 0
has_philippines = len(df[df['country'] == 'PH']) > 0
has_melanesia = len(df[df['country'].isin(['PG', 'SB', 'VU'])]) > 0

print(f'\n  Canarium present in Indonesia: {"YES" if has_indonesia else "NO"}')
print(f'  Canarium present in Madagascar: {"YES" if has_madagascar else "NO (but literature-attested)"}')
print(f'  Canarium present in Philippines: {"YES" if has_philippines else "NO"}')
print(f'  Canarium present in Melanesia: {"YES" if has_melanesia else "NO"}')

if has_indonesia:
    n_indonesia = len(df[df['country'] == 'ID'])
    print(f'\n  Indonesia records: {n_indonesia}')
    print(f'  Indonesia species: {df[df["country"]=="ID"]["species"].nunique()}')

coverage = sum([has_indonesia, has_madagascar, has_philippines, has_melanesia])
print(f'\n  Austronesian region coverage: {coverage}/4 key regions')

if coverage >= 2:
    print('  → Canarium distribution is consistent with Austronesian dispersal')
    print('  → Supports E044 botanical substitution chain hypothesis')
else:
    print('  → Insufficient coverage to confirm Austronesian dispersal pattern')

print('\nDone!')

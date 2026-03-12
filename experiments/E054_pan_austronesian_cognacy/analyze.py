"""E054: Pan-Austronesian Cognacy Gradient — Continental-Scale Peripheral Conservatism.

Hypothesis: Peripheral Austronesian languages (far from the Java/Malay center)
retain higher rates of PMP cognates than central languages — confirming P9's
peripheral conservatism thesis at continental scale with 2,000+ languages.

Method:
1. Calculate PMP cognacy rate for each of 2,000+ Austronesian languages in ABVD
2. Map geographically using lat/lon coordinates
3. Test correlation with distance from Java (the "overwriting center")
4. Test correlation with distance from Taiwan (the "dispersal homeland")
5. Identify the global cognacy gradient
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, 'results')
ABVD = os.path.join(os.path.dirname(BASE), 'E022_linguistic_subtraction', 'data', 'abvd', 'cldf')
os.makedirs(RESULTS, exist_ok=True)

print('=' * 60)
print('E054: Pan-Austronesian Cognacy Gradient')
print('=' * 60)

# ============================================================
# 1. LOAD ABVD DATA
# ============================================================
print('\n--- Loading ABVD CLDF data ---')

langs = pd.read_csv(os.path.join(ABVD, 'languages.csv'))
forms = pd.read_csv(os.path.join(ABVD, 'forms.csv'), low_memory=False)
cognates = pd.read_csv(os.path.join(ABVD, 'cognates.csv'), low_memory=False)
params = pd.read_csv(os.path.join(ABVD, 'parameters.csv'))

print(f'  Languages: {len(langs)}')
print(f'  Forms: {len(forms)}')
print(f'  Cognates: {len(cognates)}')
print(f'  Parameters (concepts): {len(params)}')

# Filter to Austronesian only
an_langs = langs[langs['Family'] == 'Austronesian'].copy()
print(f'  Austronesian languages: {len(an_langs)}')

# Drop languages without coordinates
an_langs = an_langs.dropna(subset=['Latitude', 'Longitude'])
print(f'  With coordinates: {len(an_langs)}')

# ============================================================
# 2. IDENTIFY PMP REFERENCE (Language ID 269)
# ============================================================
print('\n--- Identifying PMP reference forms ---')

PMP_ID = 269  # Proto-Malayo-Polynesian in ABVD

# Get PMP forms and their cognate sets
pmp_forms = forms[forms['Language_ID'] == PMP_ID].copy()
pmp_cognates = cognates[cognates['Form_ID'].isin(pmp_forms['ID'])].copy()

# Build PMP cognate set lookup: concept → set of cognate set IDs
pmp_cogsets = {}
for _, row in pmp_cognates.iterrows():
    form_id = row['Form_ID']
    # Extract concept from form_id (format: "269-CONCEPT-N")
    form_row = pmp_forms[pmp_forms['ID'] == form_id]
    if len(form_row) > 0:
        concept = form_row.iloc[0]['Parameter_ID']
        if concept not in pmp_cogsets:
            pmp_cogsets[concept] = set()
        pmp_cogsets[concept].add(row['Cognateset_ID'])

print(f'  PMP concepts with cognate sets: {len(pmp_cogsets)}')
print(f'  Total PMP cognate sets: {sum(len(v) for v in pmp_cogsets.values())}')

# ============================================================
# 3. CALCULATE COGNACY RATE FOR EACH LANGUAGE
# ============================================================
print('\n--- Calculating cognacy rates (this may take a minute) ---')

# Get all forms for Austronesian languages
an_form_ids = set(an_langs['ID'].values)
an_forms = forms[forms['Language_ID'].isin(an_form_ids)].copy()
print(f'  Austronesian forms: {len(an_forms)}')

# Get cognate assignments for these forms
an_cognates = cognates[cognates['Form_ID'].isin(an_forms['ID'])].copy()
print(f'  Austronesian cognate assignments: {len(an_cognates)}')

# Build lookup: form_id → cognate set IDs
form_to_cogsets = {}
for _, row in an_cognates.iterrows():
    fid = row['Form_ID']
    if fid not in form_to_cogsets:
        form_to_cogsets[fid] = set()
    form_to_cogsets[fid].add(row['Cognateset_ID'])

# For each language, calculate PMP cognacy rate
results = []
lang_ids = an_langs['ID'].values

for i, lang_id in enumerate(lang_ids):
    if (i + 1) % 500 == 0:
        print(f'  Processing language {i+1}/{len(lang_ids)}...')

    lang_forms = an_forms[an_forms['Language_ID'] == lang_id]

    n_concepts = 0
    n_cognate = 0

    # Group by concept
    for concept, concept_forms in lang_forms.groupby('Parameter_ID'):
        if concept not in pmp_cogsets:
            continue  # PMP doesn't have this concept

        n_concepts += 1

        # Check if any form in this concept shares a cognate set with PMP
        is_cognate = False
        for _, form_row in concept_forms.iterrows():
            fid = form_row['ID']
            if fid in form_to_cogsets:
                if form_to_cogsets[fid] & pmp_cogsets[concept]:  # Set intersection
                    is_cognate = True
                    break
        if is_cognate:
            n_cognate += 1

    if n_concepts >= 50:  # Minimum coverage threshold
        results.append({
            'Language_ID': lang_id,
            'n_concepts': n_concepts,
            'n_cognate': n_cognate,
            'cognacy_rate': n_cognate / n_concepts if n_concepts > 0 else 0,
        })

cognacy_df = pd.DataFrame(results)
cognacy_df = cognacy_df.merge(an_langs[['ID', 'Name', 'Glottocode', 'Latitude', 'Longitude',
                                          'Macroarea']], left_on='Language_ID', right_on='ID')

print(f'\n  Languages with ≥50 concepts: {len(cognacy_df)}')
print(f'  Mean cognacy rate: {cognacy_df["cognacy_rate"].mean():.3f}')
print(f'  Std cognacy rate: {cognacy_df["cognacy_rate"].std():.3f}')
print(f'  Range: {cognacy_df["cognacy_rate"].min():.3f} — {cognacy_df["cognacy_rate"].max():.3f}')

# ============================================================
# 4. CALCULATE DISTANCES
# ============================================================
print('\n--- Calculating distances ---')

# Reference points
JAVA_CENTER = (-7.5, 110.4)  # Central Java (court center)
TAIWAN = (23.5, 121.0)       # Taiwan (Austronesian homeland)
BALI = (-8.4, 115.3)         # Bali (peripheral reference)

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

cognacy_df['dist_java_km'] = cognacy_df.apply(
    lambda r: haversine_km(r['Latitude'], r['Longitude'], *JAVA_CENTER), axis=1)
cognacy_df['dist_taiwan_km'] = cognacy_df.apply(
    lambda r: haversine_km(r['Latitude'], r['Longitude'], *TAIWAN), axis=1)

print(f'  Distance from Java: {cognacy_df["dist_java_km"].min():.0f} — {cognacy_df["dist_java_km"].max():.0f} km')
print(f'  Distance from Taiwan: {cognacy_df["dist_taiwan_km"].min():.0f} — {cognacy_df["dist_taiwan_km"].max():.0f} km')

# ============================================================
# 5. CORRELATION TESTS
# ============================================================
print('\n--- Correlation: Cognacy Rate vs Distance ---')

# From Java
rho_java, p_java = spearmanr(cognacy_df['dist_java_km'], cognacy_df['cognacy_rate'])
print(f'  Cognacy vs Distance-from-Java: rho={rho_java:.3f}, p={p_java:.6f}, n={len(cognacy_df)}')

# From Taiwan
rho_taiwan, p_taiwan = spearmanr(cognacy_df['dist_taiwan_km'], cognacy_df['cognacy_rate'])
print(f'  Cognacy vs Distance-from-Taiwan: rho={rho_taiwan:.3f}, p={p_taiwan:.6f}')

# ============================================================
# 6. REGIONAL BREAKDOWN
# ============================================================
print('\n--- Regional Cognacy Rates ---')

# Define regions based on coordinates
def assign_region(row):
    lat, lon = row['Latitude'], row['Longitude']
    if lon < 100 and lat < 0:  # Madagascar
        return 'Madagascar'
    elif lon < 105:  # Mainland SE Asia / Sumatra
        if lat > 5:
            return 'Mainland SEA'
        else:
            return 'Sumatra/Malay'
    elif lon < 110 and lat < -5:  # Java
        return 'Java'
    elif lon < 116 and lat < -5:  # Bali/NTB
        return 'Bali/Lesser Sunda'
    elif lon < 120 and lat > 5:  # Philippines
        return 'Philippines'
    elif lon < 125 and lat < 5:  # Borneo/Sulawesi
        if lat < -1:
            return 'Sulawesi'
        else:
            return 'Borneo'
    elif lon > 140:  # Oceania
        if lat > -10:
            return 'Melanesia'
        else:
            return 'Oceania (Remote)'
    elif 119 < lon < 123 and 22 < lat < 26:  # Taiwan
        return 'Taiwan (Formosan)'
    else:
        return 'Eastern Indonesia'

cognacy_df['region'] = cognacy_df.apply(assign_region, axis=1)

region_stats = cognacy_df.groupby('region').agg(
    n=('cognacy_rate', 'count'),
    mean_cognacy=('cognacy_rate', 'mean'),
    std_cognacy=('cognacy_rate', 'std'),
    mean_dist_java=('dist_java_km', 'mean'),
).reset_index().sort_values('mean_cognacy', ascending=False)

print(f'\n  {"Region":<25} {"N":>4} {"Mean Cog%":>9} {"SD":>6} {"Dist(km)":>9}')
print('-' * 58)
for _, row in region_stats.iterrows():
    print(f'  {row["region"]:<25} {int(row["n"]):>4} {row["mean_cognacy"]*100:>8.1f}% '
          f'{row["std_cognacy"]*100:>5.1f} {row["mean_dist_java"]:>8.0f}')

# ============================================================
# 7. KEY LANGUAGE COMPARISON (E043 validation)
# ============================================================
print('\n--- Key Language Validation (cf. E043) ---')

key_langs = ['Balinese', 'Javanese', 'Malagasy', 'Old Javanese', 'Tenggerese']
# Also try partial matches
for name in key_langs:
    matches = cognacy_df[cognacy_df['Name'].str.contains(name, case=False, na=False)]
    if len(matches) > 0:
        for _, m in matches.head(3).iterrows():
            print(f'  {m["Name"]:<30} cognacy={m["cognacy_rate"]*100:.1f}% '
                  f'(n={m["n_concepts"]}), dist_java={m["dist_java_km"]:.0f}km')
    else:
        print(f'  {name}: NOT FOUND in filtered set')

# Check Merina/Malagasy specifically
malagasy = cognacy_df[cognacy_df['Name'].str.contains('Malagasy|Merina|Betsileo', case=False, na=False)]
if len(malagasy) > 0:
    print(f'\n  MALAGASY varieties found: {len(malagasy)}')
    for _, m in malagasy.iterrows():
        print(f'    {m["Name"]:<30} cognacy={m["cognacy_rate"]*100:.1f}%, '
              f'dist_java={m["dist_java_km"]:.0f}km')

# ============================================================
# 8. MAPS
# ============================================================
print('\n--- Generating maps ---')

# Map 1: Global cognacy gradient
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

ax = axes[0]
sc = ax.scatter(cognacy_df['Longitude'], cognacy_df['Latitude'],
                c=cognacy_df['cognacy_rate'], cmap='RdYlGn', s=15, alpha=0.7,
                vmin=0, vmax=0.6, zorder=3)
plt.colorbar(sc, ax=ax, label='PMP Cognacy Rate', shrink=0.8)

# Mark Java center
ax.plot(*JAVA_CENTER[::-1], 'k*', markersize=15, zorder=5, label='Java (center)')
ax.plot(*TAIWAN[::-1], 'b^', markersize=12, zorder=5, label='Taiwan (homeland)')

ax.set_xlim(30, 200)
ax.set_ylim(-50, 30)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title(f'A. PMP Cognacy Gradient Across {len(cognacy_df)} Austronesian Languages\n'
             f'Red=low retention, Green=high retention', fontsize=11, fontweight='bold')
ax.legend(loc='lower left', fontsize=9)
ax.grid(True, alpha=0.2)

# Map 2: Cognacy vs distance from Java
ax2 = axes[1]
ax2.scatter(cognacy_df['dist_java_km'], cognacy_df['cognacy_rate'] * 100,
            c=cognacy_df['cognacy_rate'], cmap='RdYlGn', s=20, alpha=0.5,
            vmin=0, vmax=0.6)

# Trend line
z = np.polyfit(cognacy_df['dist_java_km'], cognacy_df['cognacy_rate'] * 100, 1)
x_trend = np.linspace(0, cognacy_df['dist_java_km'].max(), 100)
ax2.plot(x_trend, np.polyval(z, x_trend), 'k--', linewidth=2, alpha=0.7,
         label=f'Linear trend')

# Annotate key languages
for name in ['Balinese', 'Javanese', 'Old Javanese']:
    matches = cognacy_df[cognacy_df['Name'].str.contains(name, case=False, na=False)]
    if len(matches) > 0:
        row = matches.iloc[0]
        ax2.annotate(row['Name'], xy=(row['dist_java_km'], row['cognacy_rate'] * 100),
                     fontsize=8, fontweight='bold', ha='left', va='bottom',
                     arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))

ax2.set_xlabel('Distance from Java (km)', fontsize=11)
ax2.set_ylabel('PMP Cognacy Rate (%)', fontsize=11)
ax2.set_title(f'B. Distance from Java vs PMP Retention\n'
              f'Spearman rho={rho_java:.3f}, p={p_java:.2e}', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.2)
ax2.legend(fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(RESULTS, 'pan_austronesian_cognacy_gradient.png'),
            dpi=300, bbox_inches='tight')
print('  Saved: pan_austronesian_cognacy_gradient.png')

# Map 3: Regional boxplot
fig2, ax3 = plt.subplots(figsize=(12, 6))
regions_ordered = region_stats.sort_values('mean_dist_java')['region'].values
box_data = [cognacy_df[cognacy_df['region'] == r]['cognacy_rate'].values * 100
            for r in regions_ordered]
bp = ax3.boxplot(box_data, labels=regions_ordered, patch_artist=True, vert=True)

# Color by distance from Java
cmap = plt.cm.coolwarm
norm = plt.Normalize(0, len(regions_ordered))
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(cmap(norm(i)))
    patch.set_alpha(0.7)

ax3.set_xticklabels(regions_ordered, rotation=45, ha='right', fontsize=9)
ax3.set_ylabel('PMP Cognacy Rate (%)', fontsize=11)
ax3.set_title('E054: Regional Cognacy Rates (ordered by distance from Java)\n'
              'Blue=near Java, Red=far from Java', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
fig2.savefig(os.path.join(RESULTS, 'regional_cognacy_boxplot.png'), dpi=300, bbox_inches='tight')
print('  Saved: regional_cognacy_boxplot.png')

# ============================================================
# 9. SPECIAL ANALYSIS: FORMOSAN vs. MP
# ============================================================
print('\n--- Formosan vs Malayo-Polynesian ---')

formosan = cognacy_df[cognacy_df['region'] == 'Taiwan (Formosan)']
mp = cognacy_df[cognacy_df['region'] != 'Taiwan (Formosan)']

if len(formosan) > 0:
    print(f'  Formosan languages: {len(formosan)}, mean PMP cognacy: {formosan["cognacy_rate"].mean()*100:.1f}%')
    print(f'  Malayo-Polynesian: {len(mp)}, mean PMP cognacy: {mp["cognacy_rate"].mean()*100:.1f}%')

    # Note: Formosan languages should have LOWER PMP cognacy because PMP is
    # Proto-Malayo-Polynesian, not Proto-Austronesian. Formosan retained PAn
    # features that PMP innovated away from.
    print(f'\n  NOTE: Formosan languages are OUTGROUP to PMP.')
    print(f'  Low Formosan PMP-cognacy = expected (Formosan retains PAn, not PMP).')
    print(f'  This is a phylogenetic effect, not "loss."')

# ============================================================
# 10. SYNTHESIS
# ============================================================
print('\n' + '=' * 60)
print('SYNTHESIS')
print('=' * 60)

# Determine if peripheral conservatism holds
if rho_java > 0 and p_java < 0.05:
    direction = 'CONFIRMED: languages farther from Java retain MORE PMP cognates'
elif rho_java < 0 and p_java < 0.05:
    direction = 'REVERSED: languages closer to Java retain MORE PMP cognates (innovation center effect)'
else:
    direction = f'NOT SIGNIFICANT: rho={rho_java:.3f}, p={p_java:.4f}'

print(f'\n  PERIPHERAL CONSERVATISM at continental scale:')
print(f'  {direction}')
print(f'\n  N languages analyzed: {len(cognacy_df)}')
print(f'  Distance-cognacy (Java): rho={rho_java:.3f}, p={p_java:.2e}')
print(f'  Distance-cognacy (Taiwan): rho={rho_taiwan:.3f}, p={p_taiwan:.2e}')

print(f'\n  TOP 10 highest cognacy:')
top10 = cognacy_df.nlargest(10, 'cognacy_rate')
for _, row in top10.iterrows():
    print(f'    {row["Name"]:<30} {row["cognacy_rate"]*100:.1f}% '
          f'(dist_java={row["dist_java_km"]:.0f}km, region={row["region"]})')

print(f'\n  BOTTOM 10 lowest cognacy:')
bot10 = cognacy_df.nsmallest(10, 'cognacy_rate')
for _, row in bot10.iterrows():
    print(f'    {row["Name"]:<30} {row["cognacy_rate"]*100:.1f}% '
          f'(dist_java={row["dist_java_km"]:.0f}km, region={row["region"]})')

# Save
summary = {
    'experiment': 'E054_pan_austronesian_cognacy',
    'title': 'Pan-Austronesian Cognacy Gradient',
    'date': '2026-03-12',
    'n_languages': len(cognacy_df),
    'mean_cognacy_rate': round(float(cognacy_df['cognacy_rate'].mean()), 3),
    'rho_java': round(float(rho_java), 3),
    'p_java': round(float(p_java), 6),
    'rho_taiwan': round(float(rho_taiwan), 3),
    'p_taiwan': round(float(p_taiwan), 6),
    'peripheral_conservatism': bool(rho_java > 0 and p_java < 0.05),
}

with open(os.path.join(RESULTS, 'cognacy_gradient_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

cognacy_df.to_csv(os.path.join(RESULTS, 'pan_austronesian_cognacy.csv'), index=False)

print('\nDone!')

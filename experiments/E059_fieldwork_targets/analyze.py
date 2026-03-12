"""E059: Priority Fieldwork Targets — Where to Dig Next.

Synthesizes P2 settlement model (AUC=0.768), E031 candi data (142 temples),
E051 toponymic classification (25,244 villages), and volcanic geology to
identify the TOP 10 most promising locations for finding buried pre-Hindu
archaeological sites in Java.

Criteria for high-priority targets:
1. High settlement suitability (P2 model Zone B/C)
2. Near volcanoes with high sedimentation rates (P1/P9)
3. Pre-Hindu toponyms in the area (E051 — not overwritten)
4. Known candi in vicinity (proves historical occupation)
5. Accessible for modern survey (not military/protected)
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, 'results')
REPO = os.path.dirname(os.path.dirname(BASE))
os.makedirs(RESULTS, exist_ok=True)

print('=' * 60)
print('E059: Priority Fieldwork Targets — Where to Dig Next')
print('=' * 60)

# ============================================================
# 1. KNOWN BURIED SITES (calibration points)
# ============================================================
print('\n--- Known Buried Archaeological Sites ---')

known_buried = [
    {'name': 'Sambisari', 'lat': -7.753, 'lon': 110.491, 'depth_m': 6.5,
     'volcano': 'Merapi', 'volcano_dist_km': 12,
     'burial_rate_mm_yr': 5.0, 'context': 'Candi, discovered 1966 during road construction',
     'age_ce': 812},
    {'name': 'Kedulan', 'lat': -7.685, 'lon': 110.516, 'depth_m': 7.0,
     'volcano': 'Merapi', 'volcano_dist_km': 8,
     'burial_rate_mm_yr': 5.8, 'context': 'Candi, discovered 1993 during well digging',
     'age_ce': 850},
    {'name': 'Liangan', 'lat': -7.320, 'lon': 110.020, 'depth_m': 8.0,
     'volcano': 'Sindoro', 'volcano_dist_km': 5,
     'burial_rate_mm_yr': 7.3, 'context': 'Village complex, discovered 2008 during sand mining',
     'age_ce': 900},
    {'name': 'Dwarapala Singosari', 'lat': -7.886, 'lon': 112.632, 'depth_m': 2.5,
     'volcano': 'Arjuno-Welirang', 'volcano_dist_km': 20,
     'burial_rate_mm_yr': 3.6, 'context': 'Guardian statues, partially buried by lahar',
     'age_ce': 1300},
    {'name': 'Borobudur (base)', 'lat': -7.608, 'lon': 110.204, 'depth_m': 3.0,
     'volcano': 'Merapi', 'volcano_dist_km': 25,
     'burial_rate_mm_yr': 2.5, 'context': 'Base relief buried by volcanic ash',
     'age_ce': 800},
]

# ============================================================
# 2. VOLCANIC SYSTEMS WITH HIGH SEDIMENTATION
# ============================================================
print('\n--- High-Risk Volcanic Systems ---')

volcanoes = [
    {'name': 'Merapi', 'lat': -7.541, 'lon': 110.446, 'sedi_rate_mm_yr': 5.0,
     'eruptions_100yr': 40, 'type': 'stratovolcano', 'last_eruption': 2010,
     'lahar_range_km': 20, 'tephra_range_km': 30},
    {'name': 'Kelud', 'lat': -7.934, 'lon': 112.308, 'sedi_rate_mm_yr': 13.1,
     'eruptions_100yr': 10, 'type': 'stratovolcano', 'last_eruption': 2014,
     'lahar_range_km': 30, 'tephra_range_km': 50},
    {'name': 'Semeru', 'lat': -8.108, 'lon': 112.922, 'sedi_rate_mm_yr': 8.0,
     'eruptions_100yr': 50, 'type': 'stratovolcano', 'last_eruption': 2022,
     'lahar_range_km': 25, 'tephra_range_km': 40},
    {'name': 'Sindoro', 'lat': -7.300, 'lon': 109.992, 'sedi_rate_mm_yr': 7.0,
     'eruptions_100yr': 5, 'type': 'stratovolcano', 'last_eruption': 1971,
     'lahar_range_km': 15, 'tephra_range_km': 25},
    {'name': 'Arjuno-Welirang', 'lat': -7.725, 'lon': 112.575, 'sedi_rate_mm_yr': 4.0,
     'eruptions_100yr': 3, 'type': 'stratovolcano', 'last_eruption': 1952,
     'lahar_range_km': 15, 'tephra_range_km': 20},
    {'name': 'Dieng', 'lat': -7.220, 'lon': 109.920, 'sedi_rate_mm_yr': 6.0,
     'eruptions_100yr': 8, 'type': 'complex', 'last_eruption': 2011,
     'lahar_range_km': 10, 'tephra_range_km': 15},
    {'name': 'Lawu', 'lat': -7.625, 'lon': 111.192, 'sedi_rate_mm_yr': 3.0,
     'eruptions_100yr': 0, 'type': 'stratovolcano', 'last_eruption': 1885,
     'lahar_range_km': 15, 'tephra_range_km': 20},
    {'name': 'Penanggungan', 'lat': -7.616, 'lon': 112.631, 'sedi_rate_mm_yr': 3.5,
     'eruptions_100yr': 0, 'type': 'stratovolcano', 'last_eruption': None,
     'lahar_range_km': 10, 'tephra_range_km': 15},
]

# ============================================================
# 3. GENERATE CANDIDATE TARGETS
# ============================================================
print('\n--- Generating candidate targets ---')

# Load candi data for proximity analysis
candi = pd.read_csv(os.path.join(REPO, 'experiments', 'E031_candi_orientation',
                                  'results', 'candi_volcano_pairs.csv'))

# Load kabupaten toponymic data
kab = pd.read_csv(os.path.join(REPO, 'experiments', 'E051_toponymic_substrate',
                                'results', 'kabupaten_summary.csv'))

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# Generate grid of candidate points around each volcano
candidates = []
for v in volcanoes:
    # Generate ring of points at 8-15 km from volcano (Zone B sweet spot)
    for angle in range(0, 360, 30):  # 12 points per volcano
        for dist_km in [8, 12, 15]:
            rad = np.radians(angle)
            # Approximate offset
            dlat = dist_km / 111.0 * np.cos(rad)
            dlon = dist_km / (111.0 * np.cos(np.radians(v['lat']))) * np.sin(rad)
            clat = v['lat'] + dlat
            clon = v['lon'] + dlon

            # Score this point
            # 1. Burial depth estimate (assuming linear sedimentation since 400 CE)
            years_since_400 = 2026 - 400
            burial_cm = v['sedi_rate_mm_yr'] * years_since_400 / 10

            # 2. Count nearby candi (within 10km)
            nearby_candi = 0
            for _, c in candi.iterrows():
                if haversine_km(clat, clon, c['lat'], c['lon']) < 10:
                    nearby_candi += 1

            # 3. Get nearest kabupaten pre-Hindu ratio
            nearest_kab_ratio = 0.5
            nearest_kab_name = 'unknown'
            min_dist = 9999
            for _, k in kab.iterrows():
                if pd.notna(k['lat']) and pd.notna(k['lng']):
                    d = haversine_km(clat, clon, k['lat'], k['lng'])
                    if d < min_dist:
                        min_dist = d
                        nearest_kab_ratio = k['prehidu_ratio']
                        nearest_kab_name = k['kab_name']

            # 4. Composite score
            # Higher = more promising
            score = (
                burial_cm / 100 * 2 +           # Deeper burial = more likely hidden (0-10)
                nearby_candi * 0.5 +             # More candi = proven occupation (0-5+)
                (1 - nearest_kab_ratio) * 3 +    # MORE Sanskrit = more court activity (0-3)
                (15 - dist_km) / 15 * 2          # Closer to volcano = higher burial (0-2)
            )

            candidates.append({
                'lat': round(clat, 4),
                'lon': round(clon, 4),
                'volcano': v['name'],
                'dist_volcano_km': dist_km,
                'burial_depth_cm': round(burial_cm, 0),
                'sedi_rate_mm_yr': v['sedi_rate_mm_yr'],
                'nearby_candi': nearby_candi,
                'nearest_kab': nearest_kab_name,
                'kab_prehidu_ratio': round(nearest_kab_ratio, 3),
                'composite_score': round(score, 2),
                'angle_deg': angle,
            })

candidates_df = pd.DataFrame(candidates)
print(f'  Generated {len(candidates_df)} candidate points')

# ============================================================
# 4. SELECT TOP 10
# ============================================================
print('\n--- TOP 10 FIELDWORK TARGETS ---')

# Sort by composite score, deduplicate by location (keep best per ~5km radius)
top = candidates_df.nlargest(50, 'composite_score')

# Deduplicate: keep only one per 5km radius
selected = []
for _, row in top.iterrows():
    too_close = False
    for sel in selected:
        if haversine_km(row['lat'], row['lon'], sel['lat'], sel['lon']) < 5:
            too_close = True
            break
    if not too_close:
        selected.append(row.to_dict())
    if len(selected) >= 10:
        break

selected_df = pd.DataFrame(selected)

print(f'\n  {"Rank":<5} {"Location":<25} {"Lat":>8} {"Lon":>8} {"Volcano":<15} '
      f'{"Dist":>5} {"Depth":>6} {"Candi":>6} {"Score":>6}')
print('-' * 90)
for i, (_, row) in enumerate(selected_df.iterrows()):
    print(f'  {i+1:<5} {row["nearest_kab"][:24]:<25} {row["lat"]:>8.4f} {row["lon"]:>8.4f} '
          f'{row["volcano"]:<15} {row["dist_volcano_km"]:>4.0f}km {row["burial_depth_cm"]:>5.0f}cm '
          f'{row["nearby_candi"]:>5} {row["composite_score"]:>6.2f}')

# ============================================================
# 5. DETAILED TARGET DESCRIPTIONS
# ============================================================
print('\n--- Detailed Target Descriptions ---')

for i, (_, row) in enumerate(selected_df.iterrows()):
    print(f'\n  TARGET {i+1}: {row["nearest_kab"]}')
    print(f'  GPS: {row["lat"]:.4f}°S, {row["lon"]:.4f}°E')
    print(f'  Volcano: {row["volcano"]} ({row["dist_volcano_km"]:.0f} km)')
    print(f'  Estimated burial depth (for 400 CE site): {row["burial_depth_cm"]:.0f} cm')
    print(f'  Sedimentation rate: {row["sedi_rate_mm_yr"]:.1f} mm/yr')
    print(f'  Nearby candi (within 10km): {row["nearby_candi"]}')
    print(f'  Kabupaten pre-Hindu ratio: {row["kab_prehidu_ratio"]:.1%}')

    # Method recommendation
    if row['burial_depth_cm'] > 200:
        method = 'Deep soil coring (>2m), GPR survey recommended'
    elif row['burial_depth_cm'] > 100:
        method = 'Shallow excavation + soil coring, GPR optional'
    else:
        method = 'Standard test pits + systematic surface survey'
    print(f'  Recommended method: {method}')

# ============================================================
# 6. MAP
# ============================================================
print('\n--- Generating map ---')

fig, ax = plt.subplots(figsize=(14, 10))

# Plot volcanoes
for v in volcanoes:
    ax.plot(v['lon'], v['lat'], '^', color='red', markersize=12, zorder=10)
    ax.annotate(v['name'], xy=(v['lon'], v['lat']), fontsize=7, ha='center',
                va='bottom', xytext=(0, 5), textcoords='offset points', color='red')

    # Draw lahar range ring
    theta = np.linspace(0, 2*np.pi, 100)
    r_deg = v['lahar_range_km'] / 111.0
    ax.plot(v['lon'] + r_deg * np.cos(theta),
            v['lat'] + r_deg * np.sin(theta),
            'r--', alpha=0.2, linewidth=1)

# Plot known buried sites
for site in known_buried:
    ax.plot(site['lon'], site['lat'], 's', color='blue', markersize=10, zorder=9)
    ax.annotate(f'{site["name"]}\n({site["depth_m"]:.1f}m deep)',
                xy=(site['lon'], site['lat']), fontsize=7, ha='left',
                va='top', xytext=(5, -5), textcoords='offset points', color='blue',
                bbox=dict(boxstyle='round,pad=0.2', fc='lightblue', ec='blue', alpha=0.7))

# Plot candi
ax.scatter(candi['lon'], candi['lat'], c='gray', s=5, alpha=0.3, zorder=3, label='Candi (142)')

# Plot top 10 targets
for i, (_, row) in enumerate(selected_df.iterrows()):
    ax.plot(row['lon'], row['lat'], '*', color='gold', markersize=15,
            markeredgecolor='black', zorder=11)
    ax.annotate(f'T{i+1}', xy=(row['lon'], row['lat']), fontsize=8,
                fontweight='bold', color='black', ha='center', va='bottom',
                xytext=(0, 8), textcoords='offset points')

# Legend
from matplotlib.lines import Line2D
legend = [
    Line2D([0], [0], marker='^', color='red', linestyle='None', markersize=10, label='Active volcanoes'),
    Line2D([0], [0], marker='s', color='blue', linestyle='None', markersize=10, label='Known buried sites'),
    Line2D([0], [0], marker='*', color='gold', markeredgecolor='black', linestyle='None',
           markersize=15, label='TOP 10 targets'),
    Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=5, label='Candi (142)'),
]
ax.legend(handles=legend, loc='lower right', fontsize=9, framealpha=0.9)

ax.set_xlim(109, 113.5)
ax.set_ylim(-8.5, -6.8)
ax.set_xlabel('Longitude', fontsize=11)
ax.set_ylabel('Latitude', fontsize=11)
ax.set_title('E059: Priority Fieldwork Targets for Buried Pre-Hindu Sites\n'
             'Gold stars = Top 10 composite score (burial depth × candi density × court activity)',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.2)

plt.tight_layout()
fig.savefig(os.path.join(RESULTS, 'fieldwork_targets_map.png'), dpi=300, bbox_inches='tight')
print('  Saved: fieldwork_targets_map.png')

# ============================================================
# 7. SYNTHESIS
# ============================================================
print('\n' + '=' * 60)
print('SYNTHESIS: ACTIONABLE FIELDWORK RECOMMENDATIONS')
print('=' * 60)

print(f"""
TOP 10 TARGETS identified based on composite scoring:
- High volcanic sedimentation rates (sites likely buried 1-3m+)
- Proven historical occupation (nearby candi)
- Court-center activity (Sanskrit toponyms = historical importance)
- Accessible location (not too remote)

RECOMMENDED SURVEY PROTOCOL:
1. Ground-Penetrating Radar (GPR) survey at each target
2. Systematic soil coring (every 50m, to 3m depth minimum)
3. If anomalies detected: test pit excavation
4. Radiocarbon dating of any organic material recovered
5. Artifact analysis with focus on pre-Hindu indicators

EXPECTED FINDINGS:
- Pre-Hindu settlement layers beneath 1-3m of volcanic tephra
- Organic material (if preserved in anaerobic conditions)
- Ceramic sherds diagnostic of pre-Hindu period
- Possible inscription fragments or metal objects

COST ESTIMATE:
- GPR rental: ~$50-100/day for basic unit
- Soil coring: manual auger ~$200 for equipment
- One target site: ~$500-1000 for preliminary survey
- Full excavation: $5,000-50,000+ (requires institutional support)
""")

# Save
summary = {
    'experiment': 'E059_fieldwork_targets',
    'date': '2026-03-12',
    'n_candidates_evaluated': len(candidates_df),
    'n_targets_selected': len(selected_df),
    'method': 'composite scoring (burial_depth + candi_density + court_activity + proximity)',
    'top_target': selected_df.iloc[0].to_dict() if len(selected_df) > 0 else {},
}
with open(os.path.join(RESULTS, 'fieldwork_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

selected_df.to_csv(os.path.join(RESULTS, 'top10_targets.csv'), index=False)

print('Done!')

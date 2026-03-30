"""
E129: Survey Asymmetry Quantification
Mata Elang #10 Blind Spot B1: How biased is the archaeological survey toward temples?

Using the E001 site database (391 sites), classify each site by type
and analyze what proportion of known sites are temples/candi vs
open-air settlements, caves, burials, or other.
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from collections import Counter

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
REPO = Path(__file__).parent.parent.parent

# === LOAD DATA ===

df = pd.read_csv(REPO / "data/processed/east_java_sites_wiki.csv")
print(f"Total sites: {len(df)}")
print(f"Columns: {list(df.columns)}")

# === CLASSIFY SITES BY TYPE ===

def classify_site(row):
    name = str(row.get("name", "")).lower()
    site_type = str(row.get("type", "")).lower()
    notes = str(row.get("notes", "")).lower()

    # Temple/candi
    if any(w in name for w in ["candi", "tjandi", "temple"]):
        return "temple/candi"
    if "candi" in site_type or "temple" in site_type:
        return "temple/candi"

    # Cave
    if any(w in name for w in ["gua", "goa", "cave", "liang", "song"]):
        return "cave"
    if "cave" in site_type or "gua" in site_type:
        return "cave"

    # Megalithic
    if any(w in name for w in ["megalit", "menhir", "dolmen", "sarcoph", "punden"]):
        return "megalithic"

    # Inscription/statue
    if any(w in name for w in ["prasasti", "inscription", "arca", "statue"]):
        return "inscription/statue"

    # Museum/collection
    if any(w in name for w in ["museum", "galeri"]):
        return "museum"

    # Heritage/archaeological site (generic)
    if "situs_arkeologi" in site_type:
        return "archaeological_site"
    if "objek_wisata" in site_type:
        return "tourism_site"
    if "cagar_budaya" in site_type:
        return "heritage_site"

    # Burial/cemetery
    if any(w in name for w in ["makam", "kubur", "burial", "grave"]):
        return "burial"

    # Settlement
    if any(w in name for w in ["kampung", "desa", "settlement", "site"]):
        return "settlement"

    return "other/unclassified"

df["site_class"] = df.apply(classify_site, axis=1)

# === ANALYSIS ===

print(f"\n{'=' * 70}")
print("SITE CLASSIFICATION")
print("=" * 70)

class_counts = df["site_class"].value_counts()
total = len(df)

print(f"\n  {'Class':<25} {'Count':>6} {'Percent':>8}")
print(f"  {'-'*25} {'-'*6} {'-'*8}")
for cls, count in class_counts.items():
    pct = count / total * 100
    print(f"  {cls:<25} {count:>6} {pct:>7.1f}%")

# Temple/monument ratio
temple_count = class_counts.get("temple/candi", 0) + class_counts.get("inscription/statue", 0)
temple_pct = temple_count / total * 100

cave_count = class_counts.get("cave", 0)
cave_pct = cave_count / total * 100

settlement_count = class_counts.get("settlement", 0) + class_counts.get("archaeological_site", 0)
settlement_pct = settlement_count / total * 100

print(f"\n  SUMMARY:")
print(f"  Temple/candi + inscription/statue: {temple_count} ({temple_pct:.1f}%)")
print(f"  Cave: {cave_count} ({cave_pct:.1f}%)")
print(f"  Settlement/archaeological site: {settlement_count} ({settlement_pct:.1f}%)")
print(f"  Other: {total - temple_count - cave_count - settlement_count} ({(total - temple_count - cave_count - settlement_count)/total*100:.1f}%)")

# === PERIOD ANALYSIS ===

print(f"\n{'=' * 70}")
print("PERIOD DISTRIBUTION")
print("=" * 70)

period_counts = df["period"].value_counts()
for period, count in period_counts.items():
    pct = count / total * 100
    print(f"  {str(period):<30} {count:>6} {pct:>7.1f}%")

# === TEMPLE DISTANCE TO VOLCANO ===

print(f"\n{'=' * 70}")
print("TEMPLE VS NON-TEMPLE: Volcanic Proximity")
print("=" * 70)

# Load volcano data
volcanoes = [
    {"name": "Merapi", "lat": -7.54, "lon": 110.44},
    {"name": "Kelud", "lat": -7.93, "lon": 112.31},
    {"name": "Arjuno-Welirang", "lat": -7.73, "lon": 112.58},
    {"name": "Bromo/Tengger", "lat": -7.94, "lon": 112.95},
    {"name": "Semeru", "lat": -8.11, "lon": 112.92},
    {"name": "Raung", "lat": -8.12, "lon": 114.04},
    {"name": "Ijen", "lat": -8.06, "lon": 114.24},
    {"name": "Lawu", "lat": -7.63, "lon": 111.19},
    {"name": "Penanggungan", "lat": -7.62, "lon": 112.63},
]

def min_volcano_dist(lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return np.nan
    dists = []
    for v in volcanoes:
        d = np.sqrt((lat - v["lat"])**2 + (lon - v["lon"])**2) * 111  # rough km
        dists.append(d)
    return min(dists)

df["volcano_dist_km"] = df.apply(lambda r: min_volcano_dist(r["lat"], r["lon"]), axis=1)

temple_mask = df["site_class"].isin(["temple/candi", "inscription/statue"])
non_temple_mask = ~temple_mask & df["volcano_dist_km"].notna()

if temple_mask.sum() > 5 and non_temple_mask.sum() > 5:
    temple_dist = df.loc[temple_mask, "volcano_dist_km"].dropna()
    non_temple_dist = df.loc[non_temple_mask, "volcano_dist_km"].dropna()

    print(f"\n  Temple/candi (n={len(temple_dist)}): mean dist = {temple_dist.mean():.1f} km")
    print(f"  Non-temple (n={len(non_temple_dist)}): mean dist = {non_temple_dist.mean():.1f} km")
    print(f"  Difference: {non_temple_dist.mean() - temple_dist.mean():.1f} km")

    from scipy import stats
    u, p = stats.mannwhitneyu(temple_dist, non_temple_dist, alternative="less")
    print(f"  Mann-Whitney (temples closer?): U={u:.0f}, p={p:.4f}")

# === THE SURVEY BIAS ===

print(f"\n{'=' * 70}")
print("SURVEY BIAS: What This Means for VOLCARCH")
print("=" * 70)

print(f"""
  Of {total} known sites in the database:
  - {temple_pct:.0f}% are temples/candi/monuments (Hindu-Buddhist monumental architecture)
  - These are exactly the site class that SURVIVES volcanic burial (stone, large)
  - These are exactly the site class that archaeologists LOOK FOR

  What's MISSING:
  - Open-air non-monumental settlements (villages, markets, ports)
  - Pre-Hindu sites (organic materials, no stone markers)
  - Sites at depth >2m (below standard survey capability)

  THE ASYMMETRY:
  Archaeological surveys in Java are overwhelmingly temple-focused.
  The database reflects what was LOOKED FOR, not what EXISTS.
  This is blind spot B1 from Mata Elang #10: survey TARGETING bias.

  If even 10% of survey effort had been directed at Zone B (volcanic interior,
  non-temple areas), the archaeological record might look very different.
""")

# === SAVE ===

summary = {
    "experiment": "E129_survey_asymmetry",
    "total_sites": total,
    "classification": {k: int(v) for k, v in dict(class_counts).items()},
    "temple_monument_pct": temple_pct,
    "cave_pct": cave_pct,
    "settlement_pct": settlement_pct,
    "temple_mean_volcano_dist_km": float(temple_dist.mean()) if temple_mask.sum() > 5 else None,
    "non_temple_mean_volcano_dist_km": float(non_temple_dist.mean()) if non_temple_mask.sum() > 5 else None,
    "conclusion": f"{temple_pct:.0f}% of known sites are temples/monuments — massive survey targeting bias toward monumental Hindu-Buddhist architecture",
}

with open(RESULTS_DIR / "survey_asymmetry.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Saved to {RESULTS_DIR}/survey_asymmetry.json")

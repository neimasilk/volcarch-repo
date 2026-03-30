"""
E143: Delpher Finds Spatial Cross-Reference with VOLCARCH Targets
Do colonial newspaper archaeological finds cluster near E080 fieldwork targets?

This tests whether independently-discovered colonial finds validate
VOLCARCH's prediction of WHERE buried sites are most likely.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
REPO = Path(__file__).parent.parent.parent

# === LOAD DATA ===

# E142 Delpher finds with locations
with open(REPO / "experiments/E142_delpher_fulltext/results/delpher_finds.json") as f:
    finds = json.load(f)

# E080 fieldwork targets
import csv
targets = []
with open(REPO / "experiments/E080_fieldwork_targets/results/top20_targets.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        targets.append(row)

print(f"Delpher finds: {len(finds)}")
print(f"E080 targets: {len(targets)}")

# === GEOCODE DELPHER LOCATIONS ===

# Known coordinates for colonial place names
colonial_geocode = {
    "Soerabaja": (-7.25, 112.75),
    "Surabaya": (-7.25, 112.75),
    "Malang": (-7.98, 112.63),
    "Kediri": (-7.82, 112.01),
    "Modjokerto": (-7.47, 112.43),
    "Mojokerto": (-7.47, 112.43),
    "Djokja": (-7.80, 110.36),
    "Jogja": (-7.80, 110.36),
    "Semarang": (-6.97, 110.42),
    "Batavia": (-6.17, 106.83),
    "Singosari": (-7.89, 112.72),
    "Trowoelan": (-7.57, 112.40),
    "Trowulan": (-7.57, 112.40),
    "Blitar": (-8.10, 112.17),
    "Madioen": (-7.63, 111.52),
    "Madiun": (-7.63, 111.52),
    "Probolinggo": (-7.75, 113.22),
    "Pasoeroean": (-7.65, 112.90),
    "Pasuruan": (-7.65, 112.90),
}

# Volcanoes
volcanoes = {
    "Kelud": (-7.93, 112.31),
    "Arjuno-Welirang": (-7.73, 112.58),
    "Merapi": (-7.54, 110.44),
    "Bromo": (-7.94, 112.95),
    "Semeru": (-8.11, 112.92),
    "Penanggungan": (-7.62, 112.63),
}

# Geocode finds
geocoded_finds = []
for find in finds:
    for loc in find.get("locations", []):
        if loc in colonial_geocode:
            lat, lon = colonial_geocode[loc]
            geocoded_finds.append({
                "title": find["title"],
                "date": find["date"],
                "location": loc,
                "lat": lat,
                "lon": lon,
                "materials": find.get("materials", []),
                "depth_m": find.get("max_depth_m"),
            })
            break  # one geocode per find

print(f"\nGeocoded Delpher finds: {len(geocoded_finds)}")

# === COMPUTE DISTANCE TO NEAREST VOLCANO ===

def haversine_km(lat1, lon1, lat2, lon2):
    """Approximate distance in km."""
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111

print(f"\n{'=' * 70}")
print("DELPHER FINDS: DISTANCE TO NEAREST VOLCANO")
print("=" * 70)

for gf in geocoded_finds:
    min_dist = float("inf")
    nearest = ""
    for vname, (vlat, vlon) in volcanoes.items():
        d = haversine_km(gf["lat"], gf["lon"], vlat, vlon)
        if d < min_dist:
            min_dist = d
            nearest = vname
    gf["nearest_volcano"] = nearest
    gf["dist_volcano_km"] = round(min_dist, 1)

# Summarize
dists = [gf["dist_volcano_km"] for gf in geocoded_finds]
print(f"\n  Mean distance to volcano: {np.mean(dists):.1f} km")
print(f"  Median distance: {np.median(dists):.1f} km")
print(f"  Within 30km of volcano: {sum(1 for d in dists if d < 30)}/{len(dists)}")
print(f"  Within 50km: {sum(1 for d in dists if d < 50)}/{len(dists)}")

# By volcano
volc_counts = Counter(gf["nearest_volcano"] for gf in geocoded_finds)
print(f"\n  Finds by nearest volcano:")
for v, n in volc_counts.most_common():
    print(f"    {v}: {n}")

# === COMPARE WITH E080 TARGETS ===

print(f"\n{'=' * 70}")
print("CROSS-REFERENCE: Delpher Finds vs E080 Targets")
print("=" * 70)

# E080 targets are in Zone A (Kelud) and Zone B (Arjuno-Welirang)
target_lats = [float(t["lat"]) for t in targets]
target_lons = [float(t["lon"]) for t in targets]
target_center = (np.mean(target_lats), np.mean(target_lons))

# Delpher finds near targets (within 30km)
near_targets = []
for gf in geocoded_finds:
    for t in targets:
        d = haversine_km(gf["lat"], gf["lon"], float(t["lat"]), float(t["lon"]))
        if d < 30:
            near_targets.append({
                **gf,
                "dist_to_target_km": round(d, 1),
                "target_volcano": t["nearest_volcano"],
            })
            break

print(f"\n  Delpher finds within 30km of E080 targets: {len(near_targets)}/{len(geocoded_finds)}")

if near_targets:
    print(f"\n  {'Location':<15} {'Date':>10} {'Dist to target':>15} {'Materials'}")
    print(f"  {'-'*15} {'-'*10} {'-'*15} {'-'*30}")
    for nt in near_targets:
        mats = ", ".join(nt["materials"][:3]) if nt["materials"] else "-"
        print(f"  {nt['location']:<15} {nt['date'][:10]:>10} {nt['dist_to_target_km']:>14.1f}km {mats}")

# === STATISTICAL TEST ===

print(f"\n{'=' * 70}")
print("TEST: Are Delpher finds closer to volcanoes than random?")
print("=" * 70)

# Compare with random points in East Java
np.random.seed(42)
n_random = 1000
random_dists = []
for _ in range(n_random):
    rlat = np.random.uniform(-8.5, -7.0)
    rlon = np.random.uniform(110.0, 114.5)
    min_d = min(haversine_km(rlat, rlon, vlat, vlon)
                for vlat, vlon in volcanoes.values())
    random_dists.append(min_d)

from scipy import stats
u, p = stats.mannwhitneyu(dists, random_dists, alternative="less")
print(f"\n  Delpher finds mean distance: {np.mean(dists):.1f} km")
print(f"  Random points mean distance: {np.mean(random_dists):.1f} km")
print(f"  Mann-Whitney (finds closer?): U={u:.0f}, p={p:.4f}")
if p < 0.05:
    print(f"  SIGNIFICANT: Colonial finds cluster closer to volcanoes than random")
else:
    print(f"  Not significant")

# === SAVE ===

summary = {
    "experiment": "E143_delpher_spatial",
    "geocoded_finds": len(geocoded_finds),
    "near_e080_targets": len(near_targets),
    "mean_dist_volcano_km": float(np.mean(dists)),
    "random_mean_dist_km": float(np.mean(random_dists)),
    "mann_whitney_p": float(p),
    "finds_within_30km_volcano": sum(1 for d in dists if d < 30),
    "conclusion": "Colonial finds cluster near volcanoes" if p < 0.05 else "No significant clustering",
}

with open(RESULTS_DIR / "delpher_spatial.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Saved to {RESULTS_DIR}/")

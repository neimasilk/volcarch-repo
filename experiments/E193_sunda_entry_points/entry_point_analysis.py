#!/usr/bin/env python3
"""
E193: Do Sunda Shelf Entry Points Predict Coastal Site Distribution?
=====================================================================
E177 identified 5 entry points for displaced Sunda Shelf populations.
If L2 is real, archaeological sites should cluster near these entry points.
Addresses ME#13 Risk 4: "L2 abandoned."
"""

import json, csv, sys, numpy as np
from pathlib import Path
from scipy import stats as sp_stats

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

REPO_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# E177 entry points (paleo-river termini on Java's north coast)
ENTRY_POINTS = [
    {"name": "Surabaya/Madura Strait", "lat": -7.25, "lon": 112.75, "priority": 1,
     "system": "East Sunda", "displaced_k": 100},
    {"name": "Tangerang/Banten", "lat": -6.20, "lon": 106.55, "priority": 2,
     "system": "North+South Sunda", "displaced_k": 75},
    {"name": "Semarang", "lat": -6.95, "lon": 110.40, "priority": 3,
     "system": "North Sunda East", "displaced_k": 40},
    {"name": "Jakarta Bay", "lat": -6.10, "lon": 106.85, "priority": 4,
     "system": "South Sunda", "displaced_k": 20},
    {"name": "Cirebon", "lat": -6.70, "lon": 108.55, "priority": 5,
     "system": "North Sunda West", "displaced_k": 15},
]

# Volcanoes (for L1xL2 interaction analysis)
VOLCANOES = [
    ("Kelud", -7.93, 112.31),
    ("Merapi", -7.54, 110.45),
    ("Arjuno-Welirang", -7.73, 112.58),
    ("Semeru", -8.11, 112.92),
]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    print("=" * 70)
    print("E193: Sunda Shelf Entry Points vs Coastal Site Distribution")
    print("=" * 70)

    # Load sites
    import geopandas as gpd
    sites_path = REPO_ROOT / "data" / "processed" / "east_java_sites.geojson"
    gdf = gpd.read_file(sites_path)
    print(f"\nTotal sites: {len(gdf)}")

    # Extract coordinates
    sites = []
    for _, row in gdf.iterrows():
        if row.geometry is None:
            continue
        sites.append({
            "name": row.get("name", "Unknown"),
            "lat": row.geometry.y,
            "lon": row.geometry.x,
            "type": row.get("type", "unknown"),
            "period": row.get("period", "unknown"),
        })

    # Define coastal band (within 30km of coast, approximated as lat > -7.5 for north coast)
    # Java's north coast runs roughly at lat -6.0 to -7.0
    coastal_sites = [s for s in sites if s["lat"] > -7.5]
    interior_sites = [s for s in sites if s["lat"] <= -7.5]
    print(f"Coastal (lat > -7.5): {len(coastal_sites)}")
    print(f"Interior (lat <= -7.5): {len(interior_sites)}")

    # ── Analysis 1: Distance from entry points ────────────────────────
    print(f"\n--- Analysis 1: Site density near entry points ---")

    for ep in ENTRY_POINTS:
        # Count sites within various radii
        for radius in [25, 50, 100]:
            n = sum(1 for s in coastal_sites
                    if haversine_km(s["lat"], s["lon"], ep["lat"], ep["lon"]) <= radius)
            ep[f"sites_{radius}km"] = n
        print(f"  {ep['name']:25s} | 25km: {ep['sites_25km']:3d} | "
              f"50km: {ep['sites_50km']:3d} | 100km: {ep['sites_100km']:3d}")

    # ── Analysis 2: Compare entry point density vs random coastal points ──
    print(f"\n--- Analysis 2: Entry point density vs random baseline ---")

    # Generate random coastal points for comparison
    np.random.seed(42)
    n_random = 1000
    lat_range = (-7.5, -6.0)
    lon_range = (105.0, 115.0)
    random_counts_50 = []
    for _ in range(n_random):
        rlat = np.random.uniform(*lat_range)
        rlon = np.random.uniform(*lon_range)
        n = sum(1 for s in coastal_sites
                if haversine_km(s["lat"], s["lon"], rlat, rlon) <= 50)
        random_counts_50.append(n)

    entry_counts_50 = [ep["sites_50km"] for ep in ENTRY_POINTS]
    random_mean = np.mean(random_counts_50)
    entry_mean = np.mean(entry_counts_50)

    print(f"  Entry point mean (50km): {entry_mean:.1f} sites")
    print(f"  Random point mean (50km): {random_mean:.1f} sites")
    print(f"  Ratio: {entry_mean / max(random_mean, 0.1):.2f}x")

    # Percentile rank of entry points
    for ep in ENTRY_POINTS:
        pct = np.mean([1 for r in random_counts_50 if r < ep["sites_50km"]]) * 100
        ep["percentile_50km"] = round(pct, 1)
        print(f"  {ep['name']:25s}: {ep['sites_50km']} sites = {pct:.0f}th percentile")

    # ── Analysis 3: North vs South coast ──────────────────────────────
    print(f"\n--- Analysis 3: North vs South coast ---")
    # E177 prediction: north coast > south coast (Sunda entry from north)
    north_coast = [s for s in sites if -7.5 < s["lat"] < -6.0]
    south_coast = [s for s in sites if s["lat"] < -8.0]
    print(f"  North coast (-7.5 to -6.0): {len(north_coast)} sites")
    print(f"  South coast (< -8.0): {len(south_coast)} sites")
    if len(north_coast) > 0 and len(south_coast) > 0:
        ratio = len(north_coast) / len(south_coast)
        print(f"  North/South ratio: {ratio:.2f}")
        print(f"  E177 prediction: north > south ({'CONFIRMED' if ratio > 1 else 'NOT confirmed'})")

    # ── Analysis 4: L1xL2 interaction zones ───────────────────────────
    print(f"\n--- Analysis 4: L1xL2 Double Erasure Zones ---")
    # Sites near BOTH an entry point AND a volcano
    double_erasure = []
    for s in sites:
        near_entry = min(haversine_km(s["lat"], s["lon"], ep["lat"], ep["lon"])
                         for ep in ENTRY_POINTS)
        near_volc = min(haversine_km(s["lat"], s["lon"], vlat, vlon)
                        for _, vlat, vlon in VOLCANOES)
        if near_entry < 75 and near_volc < 30:
            double_erasure.append({
                "site": s["name"], "lat": s["lat"], "lon": s["lon"],
                "dist_entry_km": round(near_entry, 1),
                "dist_volc_km": round(near_volc, 1),
            })

    print(f"  Sites within 75km of entry point AND 30km of volcano: {len(double_erasure)}")
    if double_erasure:
        print(f"  These sites are in the 'double erasure' zone (L1 volcanic + L2 coastal)")
        for de in sorted(double_erasure, key=lambda x: x["dist_volc_km"])[:10]:
            print(f"    {de['site']:30s} entry={de['dist_entry_km']:5.1f}km volc={de['dist_volc_km']:5.1f}km")

    # ── Analysis 5: Statistical test — entry point clustering ─────────
    print(f"\n--- Analysis 5: Are sites significantly clustered near entry points? ---")

    # For each site, compute minimum distance to any entry point
    site_entry_dists = [min(haversine_km(s["lat"], s["lon"], ep["lat"], ep["lon"])
                            for ep in ENTRY_POINTS) for s in coastal_sites]

    # Compare with random expectation
    random_entry_dists = []
    for _ in range(5000):
        rlat = np.random.uniform(-7.5, -6.0)
        rlon = np.random.uniform(105.0, 115.0)
        d = min(haversine_km(rlat, rlon, ep["lat"], ep["lon"]) for ep in ENTRY_POINTS)
        random_entry_dists.append(d)

    if site_entry_dists:
        u, p = sp_stats.mannwhitneyu(site_entry_dists, random_entry_dists, alternative="less")
        print(f"  Site mean dist to nearest entry: {np.mean(site_entry_dists):.1f} km")
        print(f"  Random mean dist to nearest entry: {np.mean(random_entry_dists):.1f} km")
        print(f"  Mann-Whitney U: U={u:.0f}, p={p:.5f}")
        print(f"  Sites {'ARE' if p < 0.05 else 'are NOT'} significantly closer to entry points than random")

        # Kolmogorov-Smirnov test
        ks, ks_p = sp_stats.ks_2samp(site_entry_dists, random_entry_dists)
        print(f"  KS test: D={ks:.3f}, p={ks_p:.5f}")

    # ── Save ──────────────────────────────────────────────────────────
    summary = {
        "experiment": "E193", "date": "2026-04-13",
        "n_coastal_sites": len(coastal_sites),
        "n_interior_sites": len(interior_sites),
        "entry_points": [{k: v for k, v in ep.items()} for ep in ENTRY_POINTS],
        "north_south_ratio": len(north_coast) / max(len(south_coast), 1),
        "double_erasure_sites": len(double_erasure),
        "clustering_test": {
            "site_mean_dist": round(float(np.mean(site_entry_dists)), 1) if site_entry_dists else None,
            "random_mean_dist": round(float(np.mean(random_entry_dists)), 1),
            "MW_p": round(float(p), 5) if site_entry_dists else None,
        },
    }
    with open(RESULTS_DIR / "e193_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(RESULTS_DIR / "double_erasure_sites.csv", "w", newline="", encoding="utf-8") as f:
        if double_erasure:
            writer = csv.DictWriter(f, fieldnames=double_erasure[0].keys())
            writer.writeheader()
            writer.writerows(double_erasure)

    # Verdict
    print(f"\n{'='*70}")
    entry_enrichment = entry_mean / max(random_mean, 0.1)
    if p < 0.05 and entry_enrichment > 1.5:
        verdict = "L2_SUPPORTED"
        print("VERDICT: L2 SUPPORTED — sites cluster near Sunda Shelf entry points")
    elif entry_enrichment > 1.2:
        verdict = "WEAK_SUPPORT"
        print("VERDICT: WEAK SUPPORT — entry points show enrichment but not statistically robust")
    else:
        verdict = "L2_NOT_SUPPORTED"
        print("VERDICT: L2 NOT SUPPORTED — no evidence of entry-point clustering")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

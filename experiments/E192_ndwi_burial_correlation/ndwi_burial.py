#!/usr/bin/env python3
"""
E192: Does NDWI anomaly strength correlate with burial depth?
==============================================================
If buried structures affect surface NDWI (E189 p=0.032), the signal
should weaken with depth — deeper burial = weaker surface expression.

This tests a physical prediction: Spearman rank correlation between
|NDWI center-ring diff| and predicted burial depth at known candi sites.

Negative correlation = deeper burial → weaker signal (validates both models).
Positive correlation = deeper burial → stronger signal (surprising, needs explaining).
No correlation = satellite signal is real but depth-independent.
"""

import json, csv, sys, numpy as np
from pathlib import Path
from scipy import stats as sp_stats

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

REPO_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# E189 core results
E189_CSV = REPO_ROOT / "experiments" / "E189_satellite_spectral_feasibility" / "results" / "spectral_profiles.csv"

# E075 burial predictions
E075_CSV = REPO_ROOT / "experiments" / "E075_sedimentation_model" / "results" / "site_burial_predictions.csv"

# Volcano locations for distance calculation
VOLCANOES = {
    "Kelud": (-7.9300, 112.3080),
    "Arjuno-Welirang": (-7.7290, 112.5750),
    "Semeru": (-8.1080, 112.9220),
    "Bromo": (-7.9420, 112.9500),
    "Penanggungan": (-7.6150, 112.6300),
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def nearest_volcano_dist(lat, lon):
    min_dist = float('inf')
    nearest = None
    for name, (vlat, vlon) in VOLCANOES.items():
        d = haversine_km(lat, lon, vlat, vlon)
        if d < min_dist:
            min_dist = d
            nearest = name
    return nearest, min_dist


def estimate_burial_depth(lat, lon, volc_dist_km):
    """
    Simple burial depth model based on E075 Pyle exponential:
    depth = D0 * exp(-dist/lambda)
    D0 ~ 50m (proximal), lambda ~ 5km (decay constant)
    Scaled for 1600 years of accumulation.
    """
    D0 = 5000  # cm at source (50m)
    lam = 5.0  # km decay constant
    depth_cm = D0 * np.exp(-volc_dist_km / lam)
    return max(depth_cm, 10)  # minimum 10cm


def main():
    print("=" * 70)
    print("E192: NDWI Anomaly vs Burial Depth Correlation")
    print("=" * 70)

    # Load E189 results
    print("\nLoading E189 spectral profiles...")
    e189_data = {}
    try:
        with open(E189_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle both column naming conventions (run_core vs spectral_feasibility)
                cat = row.get("cat") or row.get("category", "")
                if cat != "candi":
                    continue
                name = row["site"]
                # Handle both NDWI_diff and ndwi_diff
                ndwi_diff = float(row.get("ndwi_diff") or row.get("NDWI_diff") or 0)
                ndwi_lvar = row.get("ndwi_lvar") or row.get("NDWI_local_var")
                ndvi_diff = float(row.get("ndvi_diff") or row.get("NDVI_diff") or 0)
                ndvi_lvar = row.get("ndvi_lvar") or row.get("NDVI_local_var")
                # Skip sites with all-zero data (tile coverage artifacts)
                if ndwi_diff == 0 and ndvi_diff == 0:
                    continue
                e189_data[name] = {
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "ndwi_diff": ndwi_diff,
                    "ndwi_lvar": float(ndwi_lvar) if ndwi_lvar and float(ndwi_lvar) != 0 else None,
                    "ndvi_diff": ndvi_diff,
                    "ndvi_lvar": float(ndvi_lvar) if ndvi_lvar and float(ndvi_lvar) != 0 else None,
                }
    except FileNotFoundError:
        print("  E189 results not found!")
        return

    print(f"  Loaded {len(e189_data)} candi sites from E189")

    # Load E075 burial predictions
    print("Loading E075 burial predictions...")
    burial_data = {}
    try:
        with open(E075_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["name"]
                burial_data[name] = {
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "predicted_depth_cm": float(row["predicted_depth_cm"]),
                    "dist_km": float(row["dist_nearest_km"]),
                    "volcano": row["nearest_volcano"],
                }
    except FileNotFoundError:
        print("  E075 not found, using distance-based estimate")

    print(f"  Loaded {len(burial_data)} sites from E075")

    # Match E189 sites with burial depth
    print("\nMatching sites...")
    matched = []
    for name, e189 in e189_data.items():
        # Try exact name match first
        burial = burial_data.get(name)

        # Try fuzzy match by proximity (within 1km)
        if burial is None:
            best_dist = float('inf')
            best_name = None
            for bname, bdata in burial_data.items():
                d = haversine_km(e189["lat"], e189["lon"], bdata["lat"], bdata["lon"])
                if d < best_dist:
                    best_dist = d
                    best_name = bname
            if best_dist < 1.0:
                burial = burial_data[best_name]

        # If still no match, estimate from volcano distance
        volc_name, volc_dist = nearest_volcano_dist(e189["lat"], e189["lon"])
        if burial:
            depth = burial["predicted_depth_cm"]
        else:
            depth = estimate_burial_depth(e189["lat"], e189["lon"], volc_dist)

        matched.append({
            "site": name,
            "lat": e189["lat"],
            "lon": e189["lon"],
            "ndwi_diff": e189["ndwi_diff"],
            "ndwi_abs_diff": abs(e189["ndwi_diff"]),
            "ndwi_lvar": e189["ndwi_lvar"],
            "ndvi_diff": e189["ndvi_diff"],
            "ndvi_abs_diff": abs(e189["ndvi_diff"]),
            "ndvi_lvar": e189["ndvi_lvar"],
            "burial_depth_cm": depth,
            "volc_dist_km": volc_dist,
            "nearest_volcano": volc_name,
            "depth_source": "E075" if burial else "estimated",
        })

    print(f"  Matched: {len(matched)} sites")
    for m in sorted(matched, key=lambda x: x["burial_depth_cm"], reverse=True):
        print(f"    {m['site']:25s} depth={m['burial_depth_cm']:7.0f}cm "
              f"NDWI={m['ndwi_diff']:+.5f} dist={m['volc_dist_km']:.1f}km "
              f"[{m['depth_source']}]")

    # Correlation analysis
    print(f"\n{'='*70}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*70}")

    depths = np.array([m["burial_depth_cm"] for m in matched])
    ndwi_abs = np.array([m["ndwi_abs_diff"] for m in matched])
    ndvi_abs = np.array([m["ndvi_abs_diff"] for m in matched])
    ndwi_lvar = np.array([m["ndwi_lvar"] if m["ndwi_lvar"] is not None else np.nan for m in matched])
    ndvi_lvar = np.array([m["ndvi_lvar"] if m["ndvi_lvar"] is not None else np.nan for m in matched])
    volc_dist = np.array([m["volc_dist_km"] for m in matched])

    results = {}

    # Test 1: NDWI |diff| vs depth
    rho, p = sp_stats.spearmanr(depths, ndwi_abs)
    sig = "***" if p < 0.05 else "   "
    print(f"\n  {sig} |NDWI diff| vs burial depth: rho={rho:+.3f}, p={p:.4f}")
    results["ndwi_diff_vs_depth"] = {"rho": round(float(rho), 4), "p": round(float(p), 4)}

    # Test 2: NDVI |diff| vs depth
    rho, p = sp_stats.spearmanr(depths, ndvi_abs)
    sig = "***" if p < 0.05 else "   "
    print(f"  {sig} |NDVI diff| vs burial depth: rho={rho:+.3f}, p={p:.4f}")
    results["ndvi_diff_vs_depth"] = {"rho": round(float(rho), 4), "p": round(float(p), 4)}

    # Test 3: NDWI lvar vs depth
    valid = ~np.isnan(ndwi_lvar)
    if np.sum(valid) >= 5:
        rho, p = sp_stats.spearmanr(depths[valid], ndwi_lvar[valid])
        sig = "***" if p < 0.05 else "   "
        print(f"  {sig} NDWI local_var vs depth: rho={rho:+.3f}, p={p:.4f}")
        results["ndwi_lvar_vs_depth"] = {"rho": round(float(rho), 4), "p": round(float(p), 4)}

    # Test 4: NDVI lvar vs depth
    valid = ~np.isnan(ndvi_lvar)
    if np.sum(valid) >= 5:
        rho, p = sp_stats.spearmanr(depths[valid], ndvi_lvar[valid])
        sig = "***" if p < 0.05 else "   "
        print(f"  {sig} NDVI local_var vs depth: rho={rho:+.3f}, p={p:.4f}")
        results["ndvi_lvar_vs_depth"] = {"rho": round(float(rho), 4), "p": round(float(p), 4)}

    # Test 5: NDWI vs volcano distance (proxy for depth)
    rho, p = sp_stats.spearmanr(volc_dist, ndwi_abs)
    sig = "***" if p < 0.05 else "   "
    print(f"  {sig} |NDWI diff| vs volcano dist: rho={rho:+.3f}, p={p:.4f}")
    results["ndwi_diff_vs_volcdist"] = {"rho": round(float(rho), 4), "p": round(float(p), 4)}

    # Test 6: Depth vs volcano distance (sanity check — should be negative)
    rho, p = sp_stats.spearmanr(volc_dist, depths)
    print(f"\n  Sanity: depth vs volc_dist: rho={rho:+.3f}, p={p:.4f} (should be negative)")
    results["depth_vs_volcdist"] = {"rho": round(float(rho), 4), "p": round(float(p), 4)}

    # Partial correlation: NDWI vs depth controlling for distance
    print(f"\n--- Partial Correlation (NDWI vs depth | controlling for volcano distance) ---")
    # Residualize both variables against distance
    slope_d, intercept_d, _, _, _ = sp_stats.linregress(volc_dist, depths)
    depth_resid = depths - (slope_d * volc_dist + intercept_d)

    slope_n, intercept_n, _, _, _ = sp_stats.linregress(volc_dist, ndwi_abs)
    ndwi_resid = ndwi_abs - (slope_n * volc_dist + intercept_n)

    rho_partial, p_partial = sp_stats.spearmanr(depth_resid, ndwi_resid)
    sig = "***" if p_partial < 0.05 else "   "
    print(f"  {sig} Partial rho: {rho_partial:+.3f}, p={p_partial:.4f}")
    results["partial_ndwi_depth"] = {"rho": round(float(rho_partial), 4), "p": round(float(p_partial), 4)}

    # Save
    if not matched:
        print("\nERROR: No matched sites. Check E189 CSV format.")
        return

    with open(RESULTS_DIR / "burial_correlation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=matched[0].keys())
        writer.writeheader()
        writer.writerows(matched)

    summary = {
        "experiment": "E192", "date": "2026-04-13",
        "n_sites": len(matched),
        "correlations": results,
    }

    # Interpret
    ndwi_rho = results.get("ndwi_diff_vs_depth", {}).get("rho", 0)
    ndwi_p = results.get("ndwi_diff_vs_depth", {}).get("p", 1)

    if ndwi_rho < -0.3 and ndwi_p < 0.1:
        verdict = "NEGATIVE_CORRELATION"
        interp = "Deeper burial = weaker NDWI signal. Validates both satellite and burial models."
    elif ndwi_rho > 0.3 and ndwi_p < 0.1:
        verdict = "POSITIVE_CORRELATION"
        interp = "Deeper burial = stronger NDWI signal. Surprising — may indicate soil composition effect."
    else:
        verdict = "NO_CORRELATION"
        interp = "NDWI signal is depth-independent. Satellite detects surface drainage patterns, not buried structures directly."

    summary["verdict"] = verdict
    summary["interpretation"] = interp

    with open(RESULTS_DIR / "e192_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"  {interp}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

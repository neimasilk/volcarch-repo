#!/usr/bin/env python3
"""
E195: Is Two Javas Taphonomic? Inscription Age vs Volcano Distance
===================================================================
Core question: are inscriptions near volcanoes systematically YOUNGER
than distant inscriptions? If yes, the Two Javas pattern is (partly)
taphonomic — older volcanic-zone inscriptions are buried.

Predictions:
  TAPHONOMIC: median century INCREASES (younger) near volcanoes
  CULTURAL:   no systematic age-distance relationship

The detection horizon model predicts a SPECIFIC slope:
  At 4mm/yr, inscriptions older than ~776 CE (5m depth) are below
  standard excavation depth. Near volcanoes (higher rate), the
  horizon is even more recent.
"""

import json, csv, sys, numpy as np
from pathlib import Path
from scipy import stats as sp_stats

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

REPO_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

INSCRIPTIONS_CSV = REPO_ROOT / "experiments" / "E082_inscription_georeferencing" / "results" / "geocoded_inscriptions.csv"

# Java-specific volcanoes (filter out Sumatra/Bali)
JAVA_BBOX = {"lat_min": -8.5, "lat_max": -6.0, "lon_min": 105.0, "lon_max": 115.0}


def main():
    print("=" * 70)
    print("E195: Is Two Javas Taphonomic?")
    print("Inscription Age vs Volcano Distance")
    print("=" * 70)

    # Load inscriptions
    inscriptions = []
    with open(INSCRIPTIONS_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                century = int(row["century"]) if row["century"] else None
                lat = float(row["lat"]) if row["lat"] else None
                lon = float(row["lon"]) if row["lon"] else None
                volc_dist = float(row["volcano_dist_km"]) if row["volcano_dist_km"] else None
            except (ValueError, KeyError):
                continue

            if century and lat and lon and volc_dist:
                # Filter to Java only
                if (JAVA_BBOX["lat_min"] <= lat <= JAVA_BBOX["lat_max"] and
                    JAVA_BBOX["lon_min"] <= lon <= JAVA_BBOX["lon_max"]):
                    inscriptions.append({
                        "title": row.get("title", ""),
                        "century": century,
                        "lat": lat, "lon": lon,
                        "volc_dist_km": volc_dist,
                        "volcano": row.get("nearest_volcano", ""),
                    })

    print(f"\nJava inscriptions with century + coordinates: {len(inscriptions)}")
    if len(inscriptions) < 10:
        print("  Too few inscriptions for analysis!")
        return

    # Summary
    centuries = [i["century"] for i in inscriptions]
    dists = [i["volc_dist_km"] for i in inscriptions]
    print(f"  Century range: {min(centuries)} to {max(centuries)}")
    print(f"  Distance range: {min(dists):.1f} to {max(dists):.1f} km")

    # ── Analysis 1: Spearman correlation ──────────────────────────────
    print(f"\n--- Analysis 1: Century vs Volcano Distance ---")
    rho, p = sp_stats.spearmanr(dists, centuries)
    print(f"  Spearman rho = {rho:+.3f}, p = {p:.5f}")
    print(f"  Direction: {'TAPHONOMIC' if rho < 0 else 'CULTURAL/NULL'}")
    print(f"  (Negative = near volcano → younger century → taphonomic truncation)")

    # Pearson for comparison
    r_pear, p_pear = sp_stats.pearsonr(dists, centuries)
    print(f"  Pearson r = {r_pear:+.3f}, p = {p_pear:.5f}")

    # ── Analysis 2: Binned comparison ─────────────────────────────────
    print(f"\n--- Analysis 2: Near vs Far from volcano ---")

    # Split at median distance
    median_dist = np.median(dists)
    near = [i for i in inscriptions if i["volc_dist_km"] <= median_dist]
    far = [i for i in inscriptions if i["volc_dist_km"] > median_dist]

    near_centuries = [i["century"] for i in near]
    far_centuries = [i["century"] for i in far]

    print(f"  Median distance: {median_dist:.1f} km")
    print(f"  Near volcano (n={len(near)}): median century = {np.median(near_centuries):.0f}, "
          f"mean = {np.mean(near_centuries):.1f}")
    print(f"  Far from volcano (n={len(far)}): median century = {np.median(far_centuries):.0f}, "
          f"mean = {np.mean(far_centuries):.1f}")

    u, p_mw = sp_stats.mannwhitneyu(near_centuries, far_centuries, alternative="two-sided")
    print(f"  Mann-Whitney U = {u:.1f}, p = {p_mw:.5f}")

    # ── Analysis 3: Detection horizon prediction ──────────────────────
    print(f"\n--- Analysis 3: Detection Horizon Model ---")

    # At each distance, what's the predicted oldest detectable inscription?
    # Sedimentation rate decreases with distance: rate = D0 * exp(-dist/lambda)
    D0_mm = 6.0  # mm/yr at volcano
    lam = 8.0    # km decay constant
    detection_depth_m = 2.0  # standard excavation depth
    current_year = 2026

    for dist in [5, 10, 15, 20, 30, 50]:
        rate = D0_mm * np.exp(-dist / lam)
        if rate > 0:
            years_detectable = (detection_depth_m * 1000) / rate
            oldest_ce = current_year - years_detectable
        else:
            oldest_ce = -10000
        print(f"  At {dist:3d}km: rate={rate:.2f}mm/yr → detection horizon = {oldest_ce:.0f} CE")

    # For each inscription, compute predicted detection probability
    print(f"\n--- Analysis 4: Inscription by inscription ---")
    print(f"  {'Title':40s} {'C':>3s} {'Dist':>5s} {'Rate':>5s} {'Horizon':>8s} {'Status':>8s}")

    n_predicted_invisible = 0
    n_correctly_near_horizon = 0

    for i in sorted(inscriptions, key=lambda x: x["volc_dist_km"]):
        rate = D0_mm * np.exp(-i["volc_dist_km"] / lam)
        if rate > 0.01:
            years_det = (detection_depth_m * 1000) / rate
            horizon_ce = current_year - years_det
        else:
            horizon_ce = -10000

        status = "ABOVE" if i["century"] * 100 > horizon_ce else "BELOW"
        if status == "BELOW":
            n_predicted_invisible += 1

        # Is it within 200 years of the horizon?
        age_ce = (i["century"] - 0.5) * 100  # mid-century
        if abs(age_ce - horizon_ce) < 300:
            n_correctly_near_horizon += 1

        if i["volc_dist_km"] < 25:  # only print nearby ones
            title = i["title"][:38]
            print(f"  {title:40s} C{i['century']:2d} {i['volc_dist_km']:5.1f} "
                  f"{rate:5.2f} {horizon_ce:8.0f} {status:>8s}")

    print(f"\n  Inscriptions predicted BELOW detection horizon: {n_predicted_invisible}/{len(inscriptions)}")
    print(f"  (These exist because they're STONE — immune to sedimentation)")

    # ── Analysis 5: Earliest inscription per distance bin ─────────────
    print(f"\n--- Analysis 5: Earliest inscription per distance bin ---")
    bins = [(0, 15), (15, 25), (25, 40), (40, 100)]
    earliest_per_bin = []
    for lo, hi in bins:
        in_bin = [i for i in inscriptions if lo <= i["volc_dist_km"] < hi]
        if in_bin:
            earliest = min(i["century"] for i in in_bin)
            earliest_per_bin.append((lo, hi, len(in_bin), earliest))
            # Predicted horizon at midpoint
            mid_dist = (lo + hi) / 2
            rate = D0_mm * np.exp(-mid_dist / lam)
            horizon = current_year - (detection_depth_m * 1000 / max(rate, 0.01))
            horizon_c = int(horizon / 100) + 1
            print(f"  {lo:3d}-{hi:3d}km: n={len(in_bin):3d}, earliest=C{earliest}, "
                  f"predicted horizon~C{horizon_c}")

    # Test: does earliest century correlate with distance?
    if len(earliest_per_bin) >= 3:
        bin_mids = [(lo+hi)/2 for lo, hi, _, _ in earliest_per_bin]
        bin_earliest = [e for _, _, _, e in earliest_per_bin]
        rho_bin, p_bin = sp_stats.spearmanr(bin_mids, bin_earliest)
        print(f"\n  Earliest century vs distance: rho={rho_bin:+.3f}, p={p_bin:.4f}")
        print(f"  {'TAPHONOMIC TRUNCATION CONFIRMED' if rho_bin < -0.5 else 'No clear truncation'}")

    # ── Save ──────────────────────────────────────────────────────────
    results = {
        "experiment": "E195", "date": "2026-04-13",
        "n_inscriptions": len(inscriptions),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(p), 5),
        "near_median_century": float(np.median(near_centuries)),
        "far_median_century": float(np.median(far_centuries)),
        "mannwhitney_p": round(float(p_mw), 5),
        "n_predicted_invisible": n_predicted_invisible,
    }

    with open(RESULTS_DIR / "e195_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(RESULTS_DIR / "inscriptions_analyzed.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=inscriptions[0].keys())
        writer.writeheader()
        writer.writerows(inscriptions)

    # ── Verdict ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    if rho < -0.3 and p < 0.05:
        print("AHA: TAPHONOMIC TRUNCATION CONFIRMED")
        print("Inscriptions near volcanoes are systematically YOUNGER.")
        print("The Two Javas pattern is (partly) a taphonomic artifact.")
        print("Older volcanic-zone inscriptions are buried below detection horizon.")
        verdict = "TAPHONOMIC_CONFIRMED"
    elif rho < -0.2:
        print("WEAK TAPHONOMIC SIGNAL — direction correct but not significant")
        verdict = "WEAK_TAPHONOMIC"
    elif abs(rho) < 0.1:
        print("NO CORRELATION — Two Javas is cultural, not taphonomic")
        verdict = "CULTURAL"
    else:
        print(f"UNEXPECTED DIRECTION (rho={rho:+.3f}) — needs interpretation")
        verdict = "UNEXPECTED"

    results["verdict"] = verdict
    with open(RESULTS_DIR / "e195_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"{'='*70}")


if __name__ == "__main__":
    main()

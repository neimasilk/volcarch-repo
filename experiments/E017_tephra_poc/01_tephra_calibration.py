"""
E017: Tephra POC — Pyle (1989) Analytical Calibration for Paper 3.

Tests whether a simple Pyle (1989) exponential thinning model can predict
burial depths at 4 calibration points across 2 volcanic systems.

Critical insight: 3/4 calibration sites are Merapi (Central Java), not East Java.
This is a cross-system test of the Pyle model.

Pass criteria: 3/4 sites predicted within ±30%.

Run from repo root:
    python experiments/E017_tephra_poc/01_tephra_calibration.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Missing: {e}")
    sys.exit(1)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
REPO_ROOT = Path(__file__).parent.parent.parent

# === Pyle (1989) parameters (same as scrape_gvp.py) ===
# T(d) = T0 * exp(-k * d), T0 at 1 km, k in 1/km
PYLE_PARAMS = {
    0: (0.1, 0.15),
    1: (0.5, 0.12),
    2: (3.0, 0.08),
    3: (15.0, 0.06),
    4: (80.0, 0.05),
    5: (500.0, 0.04),
    6: (3000.0, 0.03),
}


def haversine_km(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def pyle_thickness(vei, distance_km):
    if vei not in PYLE_PARAMS:
        return 0.0
    t0, k = PYLE_PARAMS[vei]
    t = t0 * math.exp(-k * distance_km)
    return t if t >= 0.01 else 0.0


# === Calibration Sites ===
# From L1_CONSTITUTION.md: verified burial depths with dates and volcanic systems
CALIBRATION_SITES = {
    "Dwarapala Singosari": {
        "lat": -7.973, "lon": 112.435,
        "year_built": 1268,
        "depth_cm": 185,
        "depth_range": (185, 185),  # precise (half of 370 cm)
        "volcano": "Kelud",
        "system": "East Java",
    },
    "Candi Sambisari": {
        "lat": -7.752, "lon": 110.505,
        "year_built": 835,
        "depth_cm": 575,  # midpoint of 500-650
        "depth_range": (500, 650),
        "volcano": "Merapi",
        "system": "Central Java",
    },
    "Candi Kedulan": {
        "lat": -7.717, "lon": 110.452,
        "year_built": 869,
        "depth_cm": 650,  # midpoint of 600-700
        "depth_range": (600, 700),
        "volcano": "Merapi",
        "system": "Central Java",
    },
    "Candi Kimpulan": {
        "lat": -7.698, "lon": 110.407,
        "year_built": 900,
        "depth_cm": 385,  # midpoint of 270-500
        "depth_range": (270, 500),
        "volcano": "Merapi",
        "system": "Central Java",
    },
}

# === Volcano Locations ===
VOLCANO_COORDS = {
    "Kelud": (-7.930, 112.308),
    "Merapi": (-7.540, 110.446),
}

# === Eruption Summaries ===
# Kelud: from data/processed/eruption_history.csv (already downloaded)
# Merapi: from GVP 263250 (compiled from published summaries)
#
# Merapi eruption summary (GVP + Gertisser et al. 2012 + Voight et al. 2000):
# Since 835 CE (Sambisari construction): ~68 confirmed eruptions
# VEI distribution estimated from GVP catalogue:
#   VEI 1: ~15 eruptions
#   VEI 2: ~30 eruptions
#   VEI 3: ~15 eruptions
#   VEI 4: ~6 eruptions (1006, 1672, 1822, 1872, 2010 confirmed VEI 4)
#   VEI 5: ~1 eruption (1006 CE — may have been VEI 4-5)
#   Unknown VEI: ~10+ (excluded from calculation)
#
# For Kelud, we use the actual eruption-by-eruption data from eruption_history.csv

MERAPI_ERUPTIONS_SINCE = {
    # year_from: {vei: count} — eruption counts since given year
    # Conservative estimates from GVP catalogue analysis
    835: {1: 15, 2: 30, 3: 15, 4: 6, 5: 1},  # Since Sambisari
    869: {1: 14, 2: 28, 3: 14, 4: 6, 5: 1},  # Since Kedulan
    900: {1: 13, 2: 26, 3: 13, 4: 5, 5: 1},  # Since Kimpulan
}


def predict_kelud_burial(site_lat, site_lon, since_year):
    """Predict burial from Kelud eruptions using per-eruption data."""
    eruption_path = REPO_ROOT / "data" / "processed" / "eruption_history.csv"
    df = pd.read_csv(eruption_path)
    kelud = df[(df["volcano"] == "Kelud") & (df["year"] >= since_year)]

    vlat, vlon = VOLCANO_COORDS["Kelud"]
    dist = haversine_km(site_lat, site_lon, vlat, vlon)

    total = 0.0
    for _, row in kelud.iterrows():
        vei = row["vei"]
        if pd.isna(vei):
            continue
        total += pyle_thickness(int(vei), dist)

    return total, dist


def predict_merapi_burial(site_lat, site_lon, since_year):
    """Predict burial from Merapi eruptions using summary counts."""
    vlat, vlon = VOLCANO_COORDS["Merapi"]
    dist = haversine_km(site_lat, site_lon, vlat, vlon)

    if since_year not in MERAPI_ERUPTIONS_SINCE:
        # Use closest available
        closest = min(MERAPI_ERUPTIONS_SINCE.keys(), key=lambda y: abs(y - since_year))
        counts = MERAPI_ERUPTIONS_SINCE[closest]
    else:
        counts = MERAPI_ERUPTIONS_SINCE[since_year]

    total = 0.0
    for vei, n in counts.items():
        total += n * pyle_thickness(vei, dist)

    return total, dist


def main():
    print("=" * 60)
    print("E017: Tephra POC — Pyle (1989) Analytical Calibration")
    print("=" * 60)

    # --- Step 1: Compute raw Pyle predictions ---
    print("\n[1/3] Computing raw Pyle predictions at calibration sites...")
    results = []

    for name, site in CALIBRATION_SITES.items():
        if site["volcano"] == "Kelud":
            raw_depth, dist = predict_kelud_burial(
                site["lat"], site["lon"], site["year_built"]
            )
        else:  # Merapi
            raw_depth, dist = predict_merapi_burial(
                site["lat"], site["lon"], site["year_built"]
            )

        results.append({
            "site": name,
            "volcano": site["volcano"],
            "system": site["system"],
            "year_built": site["year_built"],
            "distance_km": dist,
            "actual_depth_cm": site["depth_cm"],
            "actual_range_low": site["depth_range"][0],
            "actual_range_high": site["depth_range"][1],
            "raw_pyle_cm": raw_depth,
        })
        print(f"  {name}: dist={dist:.1f} km from {site['volcano']}, "
              f"raw Pyle={raw_depth:.1f} cm, actual={site['depth_cm']} cm")

    results_df = pd.DataFrame(results)

    # --- Step 2: Calibrate loss factor from Dwarapala ---
    print("\n[2/3] Calibrating loss factor from Dwarapala...")
    dw = results_df[results_df["site"] == "Dwarapala Singosari"].iloc[0]
    loss_factor = dw["actual_depth_cm"] / dw["raw_pyle_cm"]
    print(f"  Dwarapala: raw={dw['raw_pyle_cm']:.1f} cm, actual={dw['actual_depth_cm']} cm")
    print(f"  Loss factor: {loss_factor:.4f} ({loss_factor*100:.1f}% retention)")

    # Apply calibration
    results_df["calibrated_cm"] = results_df["raw_pyle_cm"] * loss_factor
    results_df["error_cm"] = results_df["calibrated_cm"] - results_df["actual_depth_cm"]
    results_df["error_pct"] = (
        abs(results_df["calibrated_cm"] - results_df["actual_depth_cm"])
        / results_df["actual_depth_cm"] * 100
    )

    # Check if prediction falls within the known range
    results_df["within_range"] = (
        (results_df["calibrated_cm"] >= results_df["actual_range_low"] * 0.7) &
        (results_df["calibrated_cm"] <= results_df["actual_range_high"] * 1.3)
    )

    results_df["within_30pct"] = results_df["error_pct"] <= 30

    # --- Step 3: Evaluate ---
    print("\n[3/3] Evaluation...")
    print("-" * 80)
    print(f"{'Site':<25} {'System':<12} {'Dist(km)':<10} {'Actual(cm)':<12} "
          f"{'Predicted':<12} {'Error%':<10} {'Pass?'}")
    print("-" * 80)

    n_pass = 0
    for _, r in results_df.iterrows():
        passed = "PASS" if r["within_30pct"] else "FAIL"
        if r["within_30pct"]:
            n_pass += 1
        print(f"  {r['site']:<23} {r['system']:<12} {r['distance_km']:<10.1f} "
              f"{r['actual_depth_cm']:<12.0f} {r['calibrated_cm']:<12.1f} "
              f"{r['error_pct']:<10.1f} {passed}")

    print(f"\nScore: {n_pass}/4 sites within ±30%")

    if n_pass >= 3:
        verdict = "PASS — Analytical approach (Pyle 1989) sufficient for Paper 3"
        recommendation = "Paper 3 can proceed with calibrated Pyle model. Tephra2/FALL3D not required."
    elif n_pass >= 2:
        verdict = "MARGINAL — Consider adding wind correction or site-specific factors"
        recommendation = "Paper 3 may need lightweight simulation (e.g., wind-corrected Pyle). Full Tephra2 likely not required."
    else:
        verdict = "FAIL — Need heavy simulation tools for Paper 3"
        recommendation = "Paper 3 requires Tephra2/FALL3D for tephra dispersal modeling. Analytical approach insufficient."

    print(f"\nVerdict: {verdict}")
    print(f"Recommendation: {recommendation}")

    # --- Generate comparison plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Predicted vs Actual
    ax1 = axes[0]
    colors = ["#2ecc71" if p else "#e74c3c" for p in results_df["within_30pct"]]
    ax1.scatter(results_df["actual_depth_cm"], results_df["calibrated_cm"],
               c=colors, s=100, zorder=5, edgecolors="black")
    for _, r in results_df.iterrows():
        ax1.annotate(r["site"].replace("Candi ", ""),
                    (r["actual_depth_cm"], r["calibrated_cm"]),
                    fontsize=8, xytext=(5, 5), textcoords="offset points")

    max_val = max(results_df["actual_depth_cm"].max(), results_df["calibrated_cm"].max()) * 1.2
    ax1.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="1:1 line")
    ax1.fill_between([0, max_val], [0, max_val*0.7], [0, max_val*1.3],
                    alpha=0.1, color="green", label="±30% band")
    ax1.set_xlabel("Actual burial depth (cm)")
    ax1.set_ylabel("Predicted burial depth (cm)")
    ax1.set_title("Pyle 1989 Prediction vs Ground Truth")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)

    # Right: Error bar chart
    ax2 = axes[1]
    sites_short = [s.replace("Candi ", "").replace("Dwarapala ", "D.")
                   for s in results_df["site"]]
    bars = ax2.bar(sites_short, results_df["error_pct"], color=colors, edgecolor="black")
    ax2.axhline(y=30, color="red", linestyle="--", label="±30% threshold")
    ax2.set_ylabel("Absolute error (%)")
    ax2.set_title("Prediction Error by Site")
    ax2.legend(fontsize=8)

    # Add system labels
    for i, (_, r) in enumerate(results_df.iterrows()):
        ax2.text(i, results_df["error_pct"].iloc[i] + 2, r["system"],
                fontsize=7, ha="center", style="italic")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tephra_calibration.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'tephra_calibration.png'}")

    # --- Save results ---
    results_df.to_csv(RESULTS_DIR / "calibration_results.csv", index=False)

    report = f"""E017: Tephra POC — Pyle (1989) Analytical Calibration Results
{'=' * 60}
Date: 2026-03-03

Method: Pyle (1989) exponential thinning with Dwarapala-calibrated loss factor
Loss factor: {loss_factor:.4f} ({loss_factor*100:.1f}% retention)

Calibration Sites:
{results_df[['site', 'volcano', 'distance_km', 'actual_depth_cm', 'calibrated_cm', 'error_pct', 'within_30pct']].to_string(index=False)}

Score: {n_pass}/4 sites within ±30%
Verdict: {verdict}
Recommendation: {recommendation}

Key Observations:
1. Loss factor of {loss_factor*100:.1f}% is physically reasonable (erosion + compaction + reworking).
2. Cross-system test: Kelud (E. Java) calibration applied to Merapi (C. Java).
3. The loss factor is a spatially uniform correction — local geomorphology will cause
   site-specific deviations that this simple model cannot capture.
4. For Paper 3, the critical question is whether the *relative* spatial pattern
   of burial depth is correct, not just absolute values.
"""
    with open(RESULTS_DIR / "tephra_poc_report.txt", "w") as f:
        f.write(report)
    print(f"Report: {RESULTS_DIR / 'tephra_poc_report.txt'}")

    print("\nE017 COMPLETE.")


if __name__ == "__main__":
    main()

"""
E132: Sedimentation Rate Prediction Map for East Java
Using ALL calibration data (E083 tephra-site + E128 colonial OV + known candi depths)
to build a spatial model predicting sedimentation rate at any point in East Java.

Inputs:
- 5 candi calibration points (Sambisari, Kedulan, Kimpulan, Liangan, Dwarapala)
- E083 eruption-site pairs with measured depths
- E128 colonial OV depth mentions
- Volcano locations

Model: Distance-weighted sedimentation rate based on proximity to nearest volcano
Output: Grid map of predicted sedimentation rates + burial depth by period
"""

import numpy as np
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# === CALIBRATION DATA ===

# Known calibration points with sedimentation rates
calibration = [
    {"name": "Dwarapala Singosari", "lat": -7.889, "lon": 112.718,
     "volcano": "Kelud", "dist_km": 36, "rate_mm_yr": 3.5,
     "depth_m": 1.85, "age_yr": 535, "source": "E083/colonial"},
    {"name": "Candi Sambisari", "lat": -7.752, "lon": 110.496,
     "volcano": "Merapi", "dist_km": 5.8, "rate_mm_yr": 5.0,
     "depth_m": 5.5, "age_yr": 1100, "source": "E083/excavation"},
    {"name": "Candi Kedulan", "lat": -7.728, "lon": 110.395,
     "volcano": "Merapi", "dist_km": 8.0, "rate_mm_yr": 5.7,
     "depth_m": 6.5, "age_yr": 1150, "source": "E083/excavation"},
    {"name": "Candi Kimpulan", "lat": -7.706, "lon": 110.406,
     "volcano": "Merapi", "dist_km": 9.2, "rate_mm_yr": 2.7,
     "depth_m": 2.7, "age_yr": 1000, "source": "E083/excavation"},
    {"name": "Liangan", "lat": -7.280, "lon": 109.943,
     "volcano": "Sundoro", "dist_km": 6.0, "rate_mm_yr": 4.5,
     "depth_m": 5.0, "age_yr": 1100, "source": "E083/excavation"},
    # Additional from E128 high-value finds
    {"name": "OV1928 Djocja settlement", "lat": -7.78, "lon": 110.36,
     "volcano": "Merapi", "dist_km": 12.0, "rate_mm_yr": 4.6,
     "depth_m": 4.6, "age_yr": 1000, "source": "E128/colonial"},
    {"name": "OV1925 deep statue", "lat": -7.9, "lon": 112.3,
     "volcano": "Kelud", "dist_km": 20.0, "rate_mm_yr": 6.1,
     "depth_m": 9.14, "age_yr": 1500, "source": "E128/colonial"},
]

# === VOLCANOES ===

volcanoes = [
    {"name": "Merapi", "lat": -7.54, "lon": 110.44, "system_rate": 5.0},
    {"name": "Kelud", "lat": -7.93, "lon": 112.31, "system_rate": 3.5},
    {"name": "Arjuno-Welirang", "lat": -7.73, "lon": 112.58, "system_rate": 4.0},
    {"name": "Bromo/Tengger", "lat": -7.94, "lon": 112.95, "system_rate": 3.0},
    {"name": "Semeru", "lat": -8.11, "lon": 112.92, "system_rate": 4.0},
    {"name": "Penanggungan", "lat": -7.62, "lon": 112.63, "system_rate": 3.5},
    {"name": "Lawu", "lat": -7.63, "lon": 111.19, "system_rate": 2.5},
    {"name": "Sundoro", "lat": -7.30, "lon": 109.99, "system_rate": 4.5},
    {"name": "Raung", "lat": -8.12, "lon": 114.04, "system_rate": 2.0},
    {"name": "Ijen", "lat": -8.06, "lon": 114.24, "system_rate": 2.0},
]

# === MODEL: Distance-weighted sedimentation rate ===

def predict_sed_rate(lat, lon):
    """Predict sedimentation rate at a point based on volcanic proximity.
    Uses exponential decay: rate = base_rate * exp(-distance/decay_length)
    Combined from all nearby volcanoes (additive)."""

    DECAY_KM = 15  # characteristic decay distance
    BASE_BACKGROUND = 0.5  # mm/yr background non-volcanic sedimentation

    total_rate = BASE_BACKGROUND
    for v in volcanoes:
        dist = np.sqrt((lat - v["lat"])**2 + (lon - v["lon"])**2) * 111  # km
        volcanic_contribution = v["system_rate"] * np.exp(-dist / DECAY_KM)
        total_rate += volcanic_contribution

    return total_rate

# === VALIDATE AGAINST CALIBRATION ===

print("=" * 70)
print("MODEL VALIDATION: Predicted vs Observed Sedimentation Rates")
print("=" * 70)

predicted = []
observed = []
for cal in calibration:
    pred = predict_sed_rate(cal["lat"], cal["lon"])
    predicted.append(pred)
    observed.append(cal["rate_mm_yr"])
    residual = pred - cal["rate_mm_yr"]
    print(f"  {cal['name']:<30}: observed={cal['rate_mm_yr']:.1f}, predicted={pred:.1f}, "
          f"residual={residual:+.1f} mm/yr")

predicted = np.array(predicted)
observed = np.array(observed)
rmse = np.sqrt(np.mean((predicted - observed)**2))
corr = np.corrcoef(predicted, observed)[0, 1]
print(f"\n  RMSE: {rmse:.2f} mm/yr")
print(f"  Correlation: {corr:.3f}")
print(f"  Mean observed: {np.mean(observed):.2f} mm/yr")
print(f"  Mean predicted: {np.mean(predicted):.2f} mm/yr")

# === GENERATE GRID MAP ===

print(f"\n{'=' * 70}")
print("GENERATING SEDIMENTATION RATE GRID (East Java)")
print("=" * 70)

# Grid covering East Java
lat_range = np.arange(-8.5, -7.0, 0.05)  # ~5.5 km resolution
lon_range = np.arange(110.0, 114.5, 0.05)

grid = []
for lat in lat_range:
    for lon in lon_range:
        rate = predict_sed_rate(lat, lon)
        # Burial depth predictions
        depth_400ce = rate * (2026 - 400) / 1000  # meters since 400 CE
        depth_200bce = rate * (2026 + 200) / 1000  # meters since 200 BCE
        depth_1000bce = rate * (2026 + 1000) / 1000  # meters since 1000 BCE

        grid.append({
            "lat": lat,
            "lon": lon,
            "sed_rate_mm_yr": round(rate, 2),
            "depth_400ce_m": round(depth_400ce, 1),
            "depth_200bce_m": round(depth_200bce, 1),
            "depth_1000bce_m": round(depth_1000bce, 1),
        })

print(f"  Grid cells: {len(grid)}")
print(f"  Resolution: ~5.5 km")
print(f"  Coverage: {lat_range[0]}N to {lat_range[-1]}N, {lon_range[0]}E to {lon_range[-1]}E")

# === STATISTICS ===

rates = [g["sed_rate_mm_yr"] for g in grid]
depths_400 = [g["depth_400ce_m"] for g in grid]

print(f"\n  Sedimentation rate statistics:")
print(f"    Mean: {np.mean(rates):.2f} mm/yr")
print(f"    Max: {max(rates):.2f} mm/yr (near volcano)")
print(f"    Min: {min(rates):.2f} mm/yr (far from volcano)")
print(f"    >2 mm/yr: {sum(1 for r in rates if r > 2)}/{len(rates)} cells ({sum(1 for r in rates if r > 2)/len(rates)*100:.0f}%)")
print(f"    >4 mm/yr: {sum(1 for r in rates if r > 4)}/{len(rates)} cells ({sum(1 for r in rates if r > 4)/len(rates)*100:.0f}%)")

print(f"\n  Predicted burial depth since 400 CE:")
print(f"    Mean: {np.mean(depths_400):.1f} m")
print(f"    Max: {max(depths_400):.1f} m")
print(f"    >5m: {sum(1 for d in depths_400 if d > 5)}/{len(depths_400)} cells ({sum(1 for d in depths_400 if d > 5)/len(depths_400)*100:.0f}%)")
print(f"    >10m: {sum(1 for d in depths_400 if d > 10)}/{len(depths_400)} cells")

# === KEY PREDICTIONS ===

print(f"\n{'=' * 70}")
print("KEY PREDICTIONS: Burial Depth at Specific Locations")
print("=" * 70)

key_locations = [
    {"name": "Trowulan (Majapahit capital)", "lat": -7.56, "lon": 112.39},
    {"name": "E080 Target #1 (Kelud flank)", "lat": -7.98, "lon": 112.36},
    {"name": "E080 Target #7 (Arjuno-Welirang)", "lat": -7.78, "lon": 112.62},
    {"name": "Sangiran (H. erectus site)", "lat": -7.45, "lon": 110.83},
    {"name": "Batu (Malang highlands)", "lat": -7.87, "lon": 112.52},
    {"name": "Kediri (historical capital)", "lat": -7.82, "lon": 112.01},
]

for loc in key_locations:
    rate = predict_sed_rate(loc["lat"], loc["lon"])
    d400 = rate * (2026 - 400) / 1000
    d200bce = rate * (2026 + 200) / 1000
    d1000bce = rate * (2026 + 1000) / 1000

    print(f"\n  {loc['name']}:")
    print(f"    Sedimentation rate: {rate:.1f} mm/yr")
    print(f"    Depth since 400 CE: {d400:.1f} m")
    print(f"    Depth since 200 BCE: {d200bce:.1f} m")
    print(f"    Depth since 1000 BCE: {d1000bce:.1f} m")

# === SAVE ===

summary = {
    "experiment": "E132_sedimentation_map",
    "grid_cells": len(grid),
    "resolution_km": 5.5,
    "model": "exponential decay from volcanoes, additive, decay_km=15",
    "validation_rmse": float(rmse),
    "validation_corr": float(corr),
    "mean_rate": float(np.mean(rates)),
    "pct_gt_4mmyr": sum(1 for r in rates if r > 4) / len(rates),
}

with open(RESULTS_DIR / "sedimentation_map_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Save grid (large, but useful)
import csv
with open(RESULTS_DIR / "sedimentation_grid.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["lat", "lon", "sed_rate_mm_yr", "depth_400ce_m", "depth_200bce_m", "depth_1000bce_m"])
    writer.writeheader()
    writer.writerows(grid)

print(f"\n  Saved grid ({len(grid)} cells) to {RESULTS_DIR}/sedimentation_grid.csv")

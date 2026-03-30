"""
E138: Detection Probability by Archaeological Method
For each survey method, what is the probability of detecting a site
at various burial depths? Creates the "method selection guide" for
VOLCARCH fieldwork planning.

Feeds directly into E116 (testable predictions) and NatGeo proposal.
"""

import numpy as np
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# === SURVEY METHODS ===

methods = {
    "surface_survey": {
        "max_depth_m": 0.5,
        "cost_per_km2_usd": 500,
        "time_per_km2_days": 1,
        "detection_probability": lambda d: 1.0 if d < 0.5 else 0.0,
        "material_sensitivity": ["stone", "pottery", "metal"],
        "limitations": "Cannot detect anything below 50cm. Standard method in Indonesia.",
        "notes": "E117: surface survey reaches ~1900 CE at 4mm/yr",
    },
    "shovel_test_pit": {
        "max_depth_m": 1.5,
        "cost_per_km2_usd": 5000,
        "time_per_km2_days": 5,
        "detection_probability": lambda d: 1.0 if d < 1.0 else 0.5 if d < 1.5 else 0.0,
        "material_sensitivity": ["stone", "pottery", "metal", "bone", "charcoal"],
        "limitations": "Labor-intensive. Only samples small areas. Depth limited.",
    },
    "GPR_ground_penetrating_radar": {
        "max_depth_m": 5.0,
        "cost_per_km2_usd": 20000,
        "time_per_km2_days": 3,
        "detection_probability": lambda d: 0.9 if d < 2 else 0.7 if d < 3 else 0.4 if d < 5 else 0.0,
        "material_sensitivity": ["stone", "brick", "metal", "void", "density_contrast"],
        "limitations": "Signal attenuates in clay/wet soil. Best in dry sandy soil (Java volcanic = good). Cannot detect organic-only sites well.",
        "notes": "E117: GPR reaches ~776 CE. Key method for VOLCARCH targets.",
    },
    "ERT_electrical_resistivity": {
        "max_depth_m": 20.0,
        "cost_per_km2_usd": 15000,
        "time_per_km2_days": 5,
        "detection_probability": lambda d: 0.6 if d < 5 else 0.4 if d < 10 else 0.2 if d < 20 else 0.0,
        "material_sensitivity": ["stone", "brick", "void", "wet/dry_contrast"],
        "limitations": "Lower resolution than GPR. Better for deeper targets. Good for mapping buried walls/foundations.",
    },
    "magnetic_survey": {
        "max_depth_m": 3.0,
        "cost_per_km2_usd": 8000,
        "time_per_km2_days": 2,
        "detection_probability": lambda d: 0.8 if d < 1 else 0.5 if d < 2 else 0.2 if d < 3 else 0.0,
        "material_sensitivity": ["fired_clay", "iron", "hearth", "kiln"],
        "limitations": "Good for detecting fired features (kilns, hearths). Java's volcanic substrate may create noise.",
    },
    "LiDAR_airborne": {
        "max_depth_m": 0,  # surface only, but detects micro-topography
        "cost_per_km2_usd": 100,
        "time_per_km2_days": 0.01,  # fast aerial coverage
        "detection_probability": lambda d: 0.3 if d < 0.5 else 0.1 if d < 1 else 0.0,
        "material_sensitivity": ["topographic_anomaly", "buried_wall_outline", "platform"],
        "limitations": "Surface-only. Detects subtle topographic features that may indicate buried structures. Amazon 2024 success. No Java deployment for archaeology.",
        "notes": "LiDAR pitch ready (docs/dissemination/lidar_pitch.md). Cheapest per km2.",
    },
    "mechanical_coring": {
        "max_depth_m": 30.0,
        "cost_per_km2_usd": 50000,  # per borehole, not per km2
        "time_per_km2_days": 10,
        "detection_probability": lambda d: 0.3 if d < 5 else 0.3 if d < 15 else 0.2 if d < 30 else 0.0,
        "material_sensitivity": ["all_materials_in_core", "stratigraphy", "tephra_layers"],
        "limitations": "Very small diameter (5-10cm). Low probability of intersecting features. But provides DEPTH information and stratigraphy.",
        "notes": "E117: deep coring reaches ~474 BCE. Most expensive per target.",
    },
    "satellite_NDVI": {
        "max_depth_m": 0,  # surface vegetation anomaly
        "cost_per_km2_usd": 0,  # free Sentinel-2 data
        "time_per_km2_days": 0.1,
        "detection_probability": lambda d: 0.1 if d < 2 else 0.05 if d < 5 else 0.0,
        "material_sensitivity": ["vegetation_anomaly", "crop_mark"],
        "limitations": "Indirect detection via vegetation stress. E076: 2.5x NDVI variance at candi sites. Requires ground truth.",
        "notes": "FREE data. E076 v2 script ready.",
    },
}

# === DETECTION PROBABILITY MATRIX ===

print("=" * 70)
print("E138: DETECTION PROBABILITY BY METHOD AND DEPTH")
print("=" * 70)

depths = [0.5, 1, 2, 3, 5, 7, 10, 15, 20]

print(f"\n  {'Method':<30}", end="")
for d in depths:
    print(f"  {d}m", end="")
print(f"  {'Cost/km2':>10}")

print(f"  {'-'*30}", end="")
for d in depths:
    print(f"  {'---':>4}", end="")
print(f"  {'-'*10}")

for method_name, method_data in methods.items():
    print(f"  {method_name:<30}", end="")
    for d in depths:
        p = method_data["detection_probability"](d)
        if p == 0:
            print(f"  {'  -':>4}", end="")
        else:
            print(f"  {p:>.1f}", end="")
    print(f"  ${method_data['cost_per_km2_usd']:>8,}")

# === OPTIMAL METHOD FOR EACH DEPTH ===

print(f"\n{'=' * 70}")
print("OPTIMAL METHOD BY TARGET DEPTH")
print("=" * 70)

for d in [0.5, 1, 2, 3, 5, 7, 10]:
    best_method = None
    best_p = 0
    best_cost = float("inf")

    for method_name, method_data in methods.items():
        p = method_data["detection_probability"](d)
        if p > best_p or (p == best_p and method_data["cost_per_km2_usd"] < best_cost):
            best_p = p
            best_method = method_name
            best_cost = method_data["cost_per_km2_usd"]

    period = ""
    if d <= 0.5:
        period = "(colonial, ~1900 CE)"
    elif d <= 2:
        period = "(late Hindu, ~1300 CE)"
    elif d <= 5:
        period = "(early Hindu, ~700 CE)"
    elif d <= 7:
        period = "(pre-Hindu, ~400 CE)"
    elif d <= 10:
        period = "(Bronze Age, ~200 BCE)"

    print(f"  {d:>5.1f}m {period:<30}: {best_method} (P={best_p:.1f}, ${best_cost:,}/km2)")

# === VOLCARCH FIELDWORK RECOMMENDATION ===

print(f"\n{'=' * 70}")
print("VOLCARCH FIELDWORK RECOMMENDATION")
print("=" * 70)

print(f"""
  Phase 1: LiDAR + Satellite ($5,000-10,000)
    - LiDAR: 50-100 km2 of Kelud/Arjuno-Welirang flanks
    - Satellite NDVI: free Sentinel-2 analysis (E076 v2)
    - Purpose: identify surface anomalies, narrow target areas
    - Timeline: 1-2 months

  Phase 2: GPR + ERT ($20,000-40,000)
    - GPR at 10-20 target locations identified by Phase 1
    - ERT for deeper profiling at best GPR hits
    - Purpose: detect subsurface anomalies at 2-10m depth
    - Timeline: 2-3 months

  Phase 3: Targeted Coring ($10,000-20,000)
    - 10-20 cores at GPR/ERT anomaly locations
    - 15-30m depth, with phytolith and geochemical sampling
    - Purpose: confirm buried cultural layers, date them
    - Timeline: 1-2 months

  Total budget: $35,000-70,000 for decisive test
  Expected outcome: 0-6 confirmed buried sites (E116 prediction)
  P(zero finds) = 7% (E116)
""")

# === SAVE ===

# Flatten detection probabilities for JSON
detection_matrix = {}
for method_name, method_data in methods.items():
    detection_matrix[method_name] = {
        f"{d}m": method_data["detection_probability"](d) for d in depths
    }
    detection_matrix[method_name]["cost_per_km2_usd"] = method_data["cost_per_km2_usd"]
    detection_matrix[method_name]["max_depth_m"] = method_data["max_depth_m"]

summary = {
    "experiment": "E138_detection_methods",
    "methods_analyzed": len(methods),
    "detection_matrix": detection_matrix,
    "optimal_budget_usd": "35,000-70,000",
    "recommendation": "LiDAR/satellite -> GPR/ERT -> targeted coring (3-phase approach)",
}

with open(RESULTS_DIR / "detection_methods.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"  Saved to {RESULTS_DIR}/")

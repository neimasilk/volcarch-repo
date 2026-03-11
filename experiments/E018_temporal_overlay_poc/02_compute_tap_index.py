"""
E018: Compute Taphonomic Pressure Index (TAP).

Loads volcano and coastal exposure data, normalizes to [0,1],
computes composite TAP_index, and merges into TOM table.

TAP_index = alpha * V_score + (1 - alpha) * C_score
  where:
    V_score = normalized volcanic density (n_volcanoes / area * eruption intensity)
    C_score = normalized shelf exposure fraction
    alpha = 0.6 (default; tested in sensitivity sweep)

Output:
  - results/tom_table.csv  (updated with TAP columns)

Run from repo root:
    python experiments/E018_temporal_overlay_poc/02_compute_tap_index.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# === Paths ===
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"

# === Parameters ===
ALPHA = 0.6  # weight for volcanic component


def normalize_01(x):
    """Min-max normalize to [0, 1]."""
    x_min = x.min()
    x_max = x.max()
    if x_max == x_min:
        return np.zeros_like(x, dtype=float)
    return (x - x_min) / (x_max - x_min)


def main():
    print("=" * 60)
    print("E018 Step 2: Compute Taphonomic Pressure Index")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/4] Loading TAP components...")
    tap_path = DATA_DIR / "tap_components.csv"
    tom_path = RESULTS_DIR / "tom_table.csv"

    for p in [tap_path, tom_path]:
        if not p.exists():
            print(f"  ERROR: {p} not found")
            if p == tom_path:
                print("  Run 01_compile_tom_data.py first.")
            sys.exit(1)

    tap = pd.read_csv(tap_path)
    tom = pd.read_csv(tom_path)

    print(f"  TAP components: {len(tap)} regions")
    print(f"  TOM table: {len(tom)} regions")

    # --- Compute volcanic score ---
    print("\n[2/4] Computing volcanic score (V_score)...")
    # Volcano density: volcanoes per 10,000 km2
    tap["volcano_density"] = tap["n_volcanoes_holocene"] / (tap["area_km2"] / 10000)
    # Eruption intensity: VEI3+ eruptions per volcano (0 if no volcanoes)
    tap["eruption_intensity"] = np.where(
        tap["n_volcanoes_holocene"] > 0,
        tap["n_eruptions_vei3plus"] / tap["n_volcanoes_holocene"],
        0.0
    )
    # Combined volcanic raw score: density * (1 + intensity)
    tap["V_raw"] = tap["volcano_density"] * (1 + tap["eruption_intensity"])
    tap["V_score"] = normalize_01(tap["V_raw"])

    print("  Volcanic scores:")
    for _, row in tap.iterrows():
        print(f"    {row['region']:16s}  density={row['volcano_density']:.2f}  "
              f"intensity={row['eruption_intensity']:.2f}  V_score={row['V_score']:.3f}")

    # --- Compute coastal score ---
    print("\n[3/4] Computing coastal score (C_score)...")
    tap["C_score"] = normalize_01(tap["shelf_exposure_frac"])

    print("  Coastal scores:")
    for _, row in tap.iterrows():
        print(f"    {row['region']:16s}  shelf_frac={row['shelf_exposure_frac']:.2f}  "
              f"C_score={row['C_score']:.3f}")

    # --- Compute composite TAP_index ---
    print(f"\n[4/4] Computing TAP_index (alpha={ALPHA})...")
    tap["TAP_index"] = ALPHA * tap["V_score"] + (1 - ALPHA) * tap["C_score"]

    print("\n  TAP Index (composite):")
    for _, row in tap.sort_values("TAP_index", ascending=False).iterrows():
        print(f"    {row['region']:16s}  V={row['V_score']:.3f}  C={row['C_score']:.3f}  "
              f"TAP={row['TAP_index']:.3f}")

    # --- Merge into TOM table ---
    tom = tom.merge(
        tap[["region", "n_volcanoes_holocene", "volcano_density",
             "V_score", "C_score", "TAP_index"]],
        on="region", how="left"
    )

    # Save updated TOM
    tom.to_csv(tom_path, index=False)
    print(f"\n  Updated: {tom_path}")

    # Also save TAP details
    tap_out = RESULTS_DIR / "tap_details.csv"
    tap.to_csv(tap_out, index=False)
    print(f"  Saved: {tap_out}")

    nan_tap = tom["TAP_index"].isna().sum()
    if nan_tap > 0:
        print(f"\n  WARNING: {nan_tap} regions missing TAP_index!")
    else:
        print(f"\n  All {len(tom)} regions have TAP_index values.")

    print("\nStep 2 COMPLETE.")


if __name__ == "__main__":
    main()

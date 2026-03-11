"""
E018: Compile Temporal Overlay Matrix (TOM) data.

Loads linguistic, genetic, and archaeological age estimates for 8 regions,
merges them, and computes temporal gaps (L_gap, G_gap, max_gap).

Output:
  - results/tom_table.csv  (merged TOM with gap computations)

Run from repo root:
    python experiments/E018_temporal_overlay_poc/01_compile_tom_data.py
"""

import sys
from pathlib import Path

import pandas as pd

# === Paths ===
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("E018 Step 1: Compile Temporal Overlay Matrix")
    print("=" * 60)

    # --- Load input CSVs ---
    print("\n[1/3] Loading input data...")
    ling_path = DATA_DIR / "tom_input_linguistic.csv"
    gen_path = DATA_DIR / "tom_input_genetic.csv"
    arch_path = DATA_DIR / "tom_input_archaeological.csv"

    for p in [ling_path, gen_path, arch_path]:
        if not p.exists():
            print(f"  ERROR: {p} not found")
            sys.exit(1)

    ling = pd.read_csv(ling_path)
    gen = pd.read_csv(gen_path)
    arch = pd.read_csv(arch_path)

    print(f"  Linguistic:     {len(ling)} regions")
    print(f"  Genetic:        {len(gen)} regions")
    print(f"  Archaeological: {len(arch)} regions")

    # --- Merge on region ---
    print("\n[2/3] Merging on region...")
    tom = ling[["region", "L_age_bp", "L_age_ci_lo", "L_age_ci_hi"]].copy()
    tom = tom.merge(
        gen[["region", "G_age_bp", "G_age_ci_lo", "G_age_ci_hi"]],
        on="region", how="outer"
    )
    tom = tom.merge(
        arch[["region", "A_age_bp", "A_age_ci_lo", "A_age_ci_hi", "site_name"]],
        on="region", how="outer"
    )

    print(f"  Merged regions: {len(tom)}")
    missing = tom[tom[["L_age_bp", "G_age_bp", "A_age_bp"]].isna().any(axis=1)]
    if len(missing) > 0:
        print(f"  WARNING: {len(missing)} regions have missing data:")
        for _, row in missing.iterrows():
            print(f"    {row['region']}")

    # --- Compute gaps ---
    print("\n[3/3] Computing temporal gaps...")
    # L_gap: how much older is linguistic estimate than archaeological evidence
    tom["L_gap"] = tom["L_age_bp"] - tom["A_age_bp"]
    # G_gap: how much older is genetic estimate than archaeological evidence
    tom["G_gap"] = tom["G_age_bp"] - tom["A_age_bp"]
    # max_gap: the larger of the two gaps (primary test variable)
    tom["max_gap"] = tom[["L_gap", "G_gap"]].max(axis=1)

    # Reorder columns
    tom = tom[["region", "L_age_bp", "G_age_bp", "A_age_bp", "site_name",
               "L_gap", "G_gap", "max_gap",
               "L_age_ci_lo", "L_age_ci_hi",
               "G_age_ci_lo", "G_age_ci_hi",
               "A_age_ci_lo", "A_age_ci_hi"]]

    # --- Save ---
    out_path = RESULTS_DIR / "tom_table.csv"
    tom.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("TOM Summary:")
    print("=" * 60)
    for _, row in tom.iterrows():
        print(f"  {row['region']:16s}  L={row['L_age_bp']:6.0f}  G={row['G_age_bp']:6.0f}  "
              f"A={row['A_age_bp']:6.0f}  max_gap={row['max_gap']:7.0f}")

    nan_count = tom[["L_age_bp", "G_age_bp", "A_age_bp", "max_gap"]].isna().sum().sum()
    print(f"\n  Total NaN values: {nan_count}")
    print(f"  Rows: {len(tom)}")
    print("\nStep 1 COMPLETE.")


if __name__ == "__main__":
    main()

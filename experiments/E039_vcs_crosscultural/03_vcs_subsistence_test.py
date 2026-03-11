"""
E039c: VCS Subsistence/Cooperation Test
=========================================
Following E039 (informative negative on ritual complexity), test whether
volcanic proximity correlates with GROUP-BASED subsistence strategies
and resource management institutions.

Hypothesis (I-044): Volcanic hazard selects for communal coordination
mechanisms. Not ritual complexity per se, but cooperative subsistence
and resource management.

Variables tested:
  Q16 — Resource management tapu (communal resource control)
  Q58 — Group hunting (requires coordination)
  Q59 — Agriculture/horticulture (surplus + redistribution)
  Q61 — Group fishing (requires coordination)
  Q50 — Largest political community size
  Q44 — Population estimate

Author: VOLCARCH project (AI-assisted)
Date: 2026-03-11
"""

import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

PULOTU = Path("D:/documents/volcarch-repo/experiments/E023_ritual_screening/data/pulotu/cldf")
E039_RESULTS = Path("D:/documents/volcarch-repo/experiments/E039_vcs_crosscultural/results")
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

print("=" * 70)
print("E039c: VCS Subsistence/Cooperation Test")
print("=" * 70)

# Load Pulotu
cultures = pd.read_csv(PULOTU / "cultures.csv", encoding="utf-8")
responses = pd.read_csv(PULOTU / "responses.csv", encoding="utf-8")

# Merge
merged = responses.merge(
    cultures[["ID", "Name", "Latitude", "Longitude"]],
    left_on="Language_ID", right_on="ID", how="left",
    suffixes=("_resp", "_cult")
)

# Extract numeric values
merged["code_value"] = merged["Code_ID"].apply(
    lambda x: int(str(x).split("-")[-1]) if pd.notna(x) and "-" in str(x) else np.nan
)


def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


merged["numeric_value"] = merged["code_value"].where(
    merged["code_value"].notna(), merged["Value"].apply(safe_float)
)

# Pivot
pivot = merged.pivot_table(
    index=["Language_ID", "Name", "Latitude", "Longitude"],
    columns="Parameter_ID", values="numeric_value", aggfunc="first"
).reset_index()
pivot.columns = [str(c) for c in pivot.columns]

print(f"Cultures: {len(pivot)}")

# Load pre-computed volcano distances
dist_file = E039_RESULTS / "culture_volcano_distances.csv"
if dist_file.exists():
    distances = pd.read_csv(dist_file)
    pivot = pivot.merge(distances[["Name", "distance_km", "nearest_volcano", "nearest_eruptions"]],
                        on="Name", how="left")
    print(f"With volcano distances: {pivot['distance_km'].notna().sum()}")
else:
    print("ERROR: culture_volcano_distances.csv not found")
    sys.exit(1)

# Variables of interest
VARIABLES = {
    "16": "Resource management tapu",
    "58": "Group hunting",
    "59": "Agriculture / horticulture",
    "61": "Group fishing",
    "50": "Largest political community",
    "44": "Population estimate",
    "37": "Political-religious differentiation",
    "84": "Religious authority",
    "86": "Political authority",
}

# Also compute composite indices
# Group subsistence = mean of Q58 + Q61 (group-based food acquisition)
group_vars = ["58", "61"]
available_group = [v for v in group_vars if v in pivot.columns]
if available_group:
    pivot["group_subsistence"] = pivot[available_group].mean(axis=1)

# Cooperation index = Q16 + Q58 + Q61 (all require group coordination)
coop_vars = ["16", "58", "61"]
available_coop = [v for v in coop_vars if v in pivot.columns]
if available_coop:
    pivot["cooperation_index"] = pivot[available_coop].mean(axis=1)

# ═══════════════════════════════════════════════════════
# CORRELATION TESTS
# ═══════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("CORRELATION TESTS: Distance to Nearest Volcano vs Subsistence/Cooperation")
print(f"{'='*70}")

results = {}

for var_id, var_name in VARIABLES.items():
    if var_id not in pivot.columns:
        print(f"\n  {var_id} ({var_name}): NOT IN DATASET")
        continue

    valid = pivot.dropna(subset=["distance_km", var_id])
    if len(valid) < 10:
        print(f"\n  {var_id} ({var_name}): n={len(valid)} — TOO FEW")
        continue

    rho, p = stats.spearmanr(valid["distance_km"], valid[var_id])
    print(f"\n  --- Q{var_id}: {var_name} ---")
    print(f"    n = {len(valid)}")
    print(f"    Spearman rho = {rho:.3f}, p = {p:.4f}")
    print(f"    VCS predicts: NEGATIVE (closer to volcano = more of this)")

    if rho < 0 and p < 0.05:
        verdict = "SUPPORTS VCS"
    elif rho > 0 and p < 0.05:
        verdict = "OPPOSITE VCS"
    else:
        verdict = "NOT SIGNIFICANT"
    print(f"    >>> {verdict}")

    results[f"Q{var_id}_{var_name.replace(' ', '_')[:20]}"] = {
        "rho": round(rho, 4), "p": round(p, 4), "n": len(valid),
        "verdict": verdict
    }

# Composite indices
for idx_name in ["group_subsistence", "cooperation_index"]:
    if idx_name in pivot.columns:
        valid = pivot.dropna(subset=["distance_km", idx_name])
        if len(valid) > 10:
            rho, p = stats.spearmanr(valid["distance_km"], valid[idx_name])
            print(f"\n  --- {idx_name.upper()} (composite) ---")
            print(f"    n = {len(valid)}")
            print(f"    Spearman rho = {rho:.3f}, p = {p:.4f}")
            if rho < 0 and p < 0.05:
                print(f"    >>> SUPPORTS VCS")
            elif rho > 0 and p < 0.05:
                print(f"    >>> OPPOSITE VCS")
            else:
                print(f"    >>> NOT SIGNIFICANT")

            results[idx_name] = {
                "rho": round(rho, 4), "p": round(p, 4), "n": len(valid)
            }

# ═══════════════════════════════════════════════════════
# ERUPTION COUNT CORRELATIONS
# ═══════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("ERUPTION COUNT CORRELATIONS")
print(f"{'='*70}")

for var_id, var_name in list(VARIABLES.items())[:5]:
    if var_id not in pivot.columns:
        continue
    valid = pivot.dropna(subset=["nearest_eruptions", var_id])
    if len(valid) < 10:
        continue
    rho, p = stats.spearmanr(valid["nearest_eruptions"], valid[var_id])
    print(f"\n  Q{var_id} ({var_name}) vs eruption count:")
    print(f"    rho = {rho:.3f}, p = {p:.4f}, n = {len(valid)}")

# ═══════════════════════════════════════════════════════
# QUARTILE ANALYSIS
# ═══════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("DISTANCE QUARTILE ANALYSIS")
print(f"{'='*70}")

valid_all = pivot.dropna(subset=["distance_km"])
valid_all["dist_q"] = pd.qcut(valid_all["distance_km"], 4,
                                labels=["Q1_closest", "Q2", "Q3", "Q4_farthest"])

for var_id, var_name in [("16", "Resource tapu"), ("59", "Agriculture"),
                          ("58", "Group hunting"), ("61", "Group fishing")]:
    if var_id not in pivot.columns:
        continue
    print(f"\n  --- Q{var_id}: {var_name} ---")
    for q, grp in valid_all.groupby("dist_q"):
        vals = grp[var_id].dropna()
        if len(vals) > 0:
            print(f"    {q}: n={len(vals)}, mean={vals.mean():.3f} ± {vals.std():.3f}")

    # Kruskal-Wallis
    groups = [grp[var_id].dropna().values for _, grp in valid_all.groupby("dist_q")]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2:
        h, p_kw = stats.kruskal(*groups)
        print(f"    Kruskal-Wallis: H={h:.2f}, p={p_kw:.4f}")

# ═══════════════════════════════════════════════════════
# INDONESIAN SUBSET
# ═══════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("INDONESIAN/MELANESIAN SUBSET")
print(f"{'='*70}")

indo = valid_all[(valid_all["Latitude"].between(-15, 10)) &
                  (valid_all["Longitude"].between(95, 180))]
print(f"  n = {len(indo)}")

if len(indo) > 10:
    for var_id, var_name in [("16", "Resource tapu"), ("59", "Agriculture"),
                              ("58", "Group hunting")]:
        if var_id not in indo.columns:
            continue
        v = indo.dropna(subset=[var_id])
        if len(v) > 5:
            rho, p = stats.spearmanr(v["distance_km"], v[var_id])
            print(f"  Q{var_id} ({var_name}): rho={rho:.3f}, p={p:.4f}, n={len(v)}")

# ═══════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("VERDICT")
print(f"{'='*70}")

n_sig = sum(1 for r in results.values() if r.get("p", 1) < 0.05
            and r.get("rho", 0) < 0)
n_tests = len(results)

if n_sig >= 2:
    print(f"\n  >>> {n_sig}/{n_tests} tests support VCS subsistence hypothesis")
elif n_sig == 1:
    print(f"\n  >>> {n_sig}/{n_tests} tests marginally support VCS")
else:
    print(f"\n  >>> {n_sig}/{n_tests} tests support VCS — NOT SIGNIFICANT")
    print(f"  >>> VCS does not predict subsistence cooperation at global Austronesian scale")

# Save
with open(OUT / "vcs_subsistence_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUT / 'vcs_subsistence_results.json'}")

print("\n" + "=" * 70)
print("E039c COMPLETE")
print("=" * 70)

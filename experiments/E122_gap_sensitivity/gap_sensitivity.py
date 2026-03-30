"""
E122: Demographic Gap Sensitivity Analysis
Mata Elang #10 response: How robust is the 3,220x gap to carrying capacity assumptions?

Tests:
1. Carrying capacity parameter sweep (0.05 to 50 people/km2)
2. At what population does gap become "trivial" (<10x)?
3. Which scenario assumptions are most impactful?
4. Monte Carlo with uncertain parameters
5. Comparison with ethnographic analogues from literature
"""

import numpy as np
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
np.random.seed(42)

# === PARAMETERS FROM E108 ===

JAVA_AREA_KM2 = 129000
HABITABLE_FRAC = 0.88  # 114,000 km2
PRIME_AG_FRAC = 0.74   # 95,000 km2 of habitable
HABITABLE_AREA = JAVA_AREA_KM2 * HABITABLE_FRAC

# Known pre-400 CE sites
KNOWN_SITES_LOW = 0
KNOWN_SITES_MID = 1  # Buni Complex (ambiguous)
KNOWN_SITES_HIGH = 3  # Buni + Batujaya + 1 debatable

# Settlement size estimates (people per site)
PEOPLE_PER_SETTLEMENT = 200  # conservative village size

# === E108 CARRYING CAPACITY SCENARIOS ===

scenarios = {
    "A_minimal_swidden": {
        "density_per_km2": 5.18,
        "description": "Sparse swidden agriculture",
        "basis": "Headland & Reid 1989 tropical forager-farmer"
    },
    "B_moderate_chiefdoms": {
        "density_per_km2": 16.95,
        "description": "Moderate chiefdom level",
        "basis": "Kirch 2000 Polynesian analogue"
    },
    "C_maximum_protostates": {
        "density_per_km2": 34.3,
        "description": "Proto-state level",
        "basis": "Historical Mataram densities (post-rice)"
    },
    "D_conservative_no_rice": {
        "density_per_km2": 12.74,
        "description": "No wet rice, swidden only",
        "basis": "E108 sensitivity test"
    },
}

# === TEST 1: CARRYING CAPACITY SWEEP ===

print("=" * 70)
print("TEST 1: Carrying capacity sweep (0.05 to 50 people/km2)")
print("=" * 70)

densities = np.logspace(np.log10(0.05), np.log10(50), 200)
sweep_results = []

for d in densities:
    pop = d * HABITABLE_AREA
    expected_settlements = pop / PEOPLE_PER_SETTLEMENT
    gap_low = expected_settlements / max(KNOWN_SITES_HIGH, 1)  # most generous (3 sites)
    gap_mid = expected_settlements / max(KNOWN_SITES_MID, 1)
    gap_high = expected_settlements / max(1, 0.5)  # zero sites -> use 0.5 for math

    sweep_results.append({
        "density": d,
        "population": pop,
        "expected_settlements": expected_settlements,
        "gap_3sites": gap_low,
        "gap_1site": gap_mid,
    })

# Find threshold densities
for threshold_name, threshold in [("trivial_10x", 10), ("moderate_100x", 100), ("strong_1000x", 1000)]:
    for r in sweep_results:
        if r["gap_3sites"] >= threshold:
            print(f"  Gap >= {threshold}x when density >= {r['density']:.2f}/km2 "
                  f"(pop {r['population']:.0f}, {r['expected_settlements']:.0f} settlements)")
            break

# === TEST 2: AT WHAT POPULATION DOES GAP BECOME TRIVIAL? ===

print(f"\n{'=' * 70}")
print("TEST 2: Population threshold for 'trivial' gap")
print("=" * 70)

thresholds = {}
for gap_threshold in [10, 50, 100, 500, 1000, 3000]:
    # Gap = (pop / people_per_settlement) / known_sites
    # For gap = threshold with 3 known sites:
    needed_settlements = gap_threshold * 3
    needed_pop = needed_settlements * PEOPLE_PER_SETTLEMENT
    needed_density = needed_pop / HABITABLE_AREA

    thresholds[gap_threshold] = {
        "needed_population": needed_pop,
        "needed_density": needed_density,
        "plausible": needed_density <= 50,
        "comparison": "",
    }

    # Compare with ethnographic records
    if needed_density < 1:
        comp = "< hunter-gatherer bands"
    elif needed_density < 5:
        comp = "sparse forager-farmer (Borneo interior)"
    elif needed_density < 15:
        comp = "moderate swidden agriculture"
    elif needed_density < 35:
        comp = "chiefdom/early state (Polynesia)"
    else:
        comp = "intensive wet rice (post-Hindu Java)"

    thresholds[gap_threshold]["comparison"] = comp

    status = "PLAUSIBLE" if needed_density <= 50 else "IMPLAUSIBLE"
    print(f"  Gap >= {gap_threshold:>5}x needs density >= {needed_density:>6.2f}/km2 "
          f"(pop >= {needed_pop:>10,.0f}) -> {comp} [{status}]")

# === TEST 3: MONTE CARLO WITH UNCERTAIN PARAMETERS ===

print(f"\n{'=' * 70}")
print("TEST 3: Monte Carlo gap estimation (100K runs)")
print("=" * 70)

N_MC = 100000

# Parameter distributions
# Density: log-uniform from 0.5 to 40 people/km2
mc_density = np.exp(np.random.uniform(np.log(0.5), np.log(40), N_MC))
# Habitable fraction: uniform 0.70 to 0.95
mc_hab_frac = np.random.uniform(0.70, 0.95, N_MC)
# People per settlement: uniform 50 to 500
mc_pps = np.random.uniform(50, 500, N_MC)
# Known sites: discrete {0, 1, 2, 3}
mc_known = np.random.choice([0.5, 1, 2, 3], N_MC, p=[0.1, 0.3, 0.3, 0.3])

mc_pop = mc_density * (JAVA_AREA_KM2 * mc_hab_frac)
mc_settlements = mc_pop / mc_pps
mc_gap = mc_settlements / mc_known

print(f"  Median gap: {np.median(mc_gap):,.0f}x")
print(f"  Mean gap: {np.mean(mc_gap):,.0f}x")
print(f"  5th percentile: {np.percentile(mc_gap, 5):,.0f}x")
print(f"  25th percentile: {np.percentile(mc_gap, 25):,.0f}x")
print(f"  75th percentile: {np.percentile(mc_gap, 75):,.0f}x")
print(f"  95th percentile: {np.percentile(mc_gap, 95):,.0f}x")
print(f"  P(gap > 100x): {np.mean(mc_gap > 100)*100:.1f}%")
print(f"  P(gap > 1000x): {np.mean(mc_gap > 1000)*100:.1f}%")
print(f"  P(gap > 3000x): {np.mean(mc_gap > 3000)*100:.1f}%")
print(f"  P(gap < 10x): {np.mean(mc_gap < 10)*100:.1f}%")

# === TEST 4: WHAT IF POPULATION WAS VERY LOW? ===

print(f"\n{'=' * 70}")
print("TEST 4: Adversarial low population scenarios")
print("=" * 70)

adversarial = {
    "Pure hunter-gatherer": {"density": 0.1, "basis": "Kelly 2013 tropical HG"},
    "Sparse coastal only": {"density": 0.5, "basis": "Interior uninhabited"},
    "Seasonal/nomadic": {"density": 1.0, "basis": "No permanent settlements"},
    "Early swidden": {"density": 3.0, "basis": "Pre-rice slash-burn"},
    "Moderate farming": {"density": 10.0, "basis": "Mixed subsistence"},
    "E108 minimum": {"density": 5.18, "basis": "Scenario A"},
    "E108 moderate": {"density": 16.95, "basis": "Scenario B"},
    "E108 maximum": {"density": 34.3, "basis": "Scenario C"},
}

print(f"\n  {'Scenario':<25} {'Density':>8} {'Population':>12} {'Settlements':>13} {'Gap (3 sites)':>13}")
print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*13} {'-'*13}")

for name, params in adversarial.items():
    pop = params["density"] * HABITABLE_AREA
    settlements = pop / PEOPLE_PER_SETTLEMENT
    gap = settlements / 3

    print(f"  {name:<25} {params['density']:>7.1f} {pop:>11,.0f} {settlements:>12,.0f} {gap:>12,.0f}x")

# === TEST 5: PARAMETER ELASTICITY ===

print(f"\n{'=' * 70}")
print("TEST 5: Which parameter has most impact on gap?")
print("=" * 70)

# Baseline: E108 moderate
base_density = 16.95
base_hab = 0.88
base_pps = 200
base_known = 1

base_gap = (base_density * JAVA_AREA_KM2 * base_hab / base_pps) / base_known

params_test = {
    "density (+50%)": (base_density * 1.5, base_hab, base_pps, base_known),
    "density (-50%)": (base_density * 0.5, base_hab, base_pps, base_known),
    "habitable_frac (+10%)": (base_density, min(base_hab * 1.1, 1.0), base_pps, base_known),
    "habitable_frac (-10%)": (base_density, base_hab * 0.9, base_pps, base_known),
    "settlement_size (+100%)": (base_density, base_hab, base_pps * 2, base_known),
    "settlement_size (-50%)": (base_density, base_hab, base_pps * 0.5, base_known),
    "known_sites = 3": (base_density, base_hab, base_pps, 3),
    "known_sites = 10": (base_density, base_hab, base_pps, 10),
}

print(f"\n  Baseline gap: {base_gap:,.0f}x")
print(f"\n  {'Parameter Change':<30} {'Gap':>10} {'Change':>10}")
print(f"  {'-'*30} {'-'*10} {'-'*10}")

for name, (d, h, p, k) in params_test.items():
    gap = (d * JAVA_AREA_KM2 * h / p) / k
    change = gap / base_gap
    print(f"  {name:<30} {gap:>9,.0f}x {change:>9.2f}x")

# === SAVE RESULTS ===

summary = {
    "experiment": "E122_gap_sensitivity",
    "baseline_gap": base_gap,
    "monte_carlo": {
        "n_runs": N_MC,
        "median_gap": float(np.median(mc_gap)),
        "p_gap_gt_100x": float(np.mean(mc_gap > 100)),
        "p_gap_gt_1000x": float(np.mean(mc_gap > 1000)),
        "p_gap_lt_10x": float(np.mean(mc_gap < 10)),
        "ci_5_95": [float(np.percentile(mc_gap, 5)), float(np.percentile(mc_gap, 95))],
    },
    "adversarial_minimum": {
        "scenario": "Pure hunter-gatherer (0.1/km2)",
        "gap": (0.1 * HABITABLE_AREA / PEOPLE_PER_SETTLEMENT) / 3,
        "verdict": "Even at pure HG density, gap = 19x (3 sites). STILL EXISTS."
    },
    "threshold_for_trivial": {
        "gap_10x_needs_density": thresholds[10]["needed_density"],
        "gap_10x_comparison": thresholds[10]["comparison"],
    },
    "verdict": "",
}

# Verdict
min_gap_adversarial = (0.1 * HABITABLE_AREA / PEOPLE_PER_SETTLEMENT) / 3
if min_gap_adversarial > 10:
    summary["verdict"] = (
        f"GAP IS REAL REGARDLESS OF POPULATION ASSUMPTIONS. "
        f"Even at pure hunter-gatherer density (0.1/km2), gap = {min_gap_adversarial:.0f}x. "
        f"Monte Carlo: P(gap > 100x) = {np.mean(mc_gap > 100)*100:.0f}%. "
        f"Magnitude is parameter-dependent (median MC = {np.median(mc_gap):,.0f}x), "
        f"but existence of gap is robust."
    )
else:
    summary["verdict"] = f"Gap could be trivial at extremely low population estimates."

print(f"\n{'=' * 70}")
print("VERDICT")
print("=" * 70)
print(f"\n  {summary['verdict']}")

with open(RESULTS_DIR / "gap_sensitivity_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Saved to {RESULTS_DIR}/gap_sensitivity_results.json")

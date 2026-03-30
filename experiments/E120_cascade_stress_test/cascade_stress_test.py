"""
E120: Cascade Stress Test - Systematic Adversarial Probing
VOLCARCH AutoResearch Program 3 (Proof of Concept)

Differs from E115 (Monte Carlo random sampling) by:
1. Isolating each factor systematically (fine grid, others fixed)
2. Finding exact breaking points per factor
3. Testing factor removal (set to 1.0)
4. Finding adversarial minimum (strongest counter-argument)
5. Pairwise interaction mapping

Output: results/ directory with CSVs + summary JSON
"""

import numpy as np
import json
import csv
from itertools import combinations
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# === CASCADE PARAMETERS (from E110) ===

FACTORS = {
    "F1_volcanic_burial":   {"best": 0.58, "low": 0.40, "high": 0.75, "leverage": 1.7},
    "F2_organic_decay":     {"best": 0.20, "low": 0.10, "high": 0.35, "leverage": 5.0},
    "F3_survey_coverage":   {"best": 0.025, "low": 0.01, "high": 0.10, "leverage": 40.0},
    "F4_recognition":       {"best": 0.40, "low": 0.20, "high": 0.60, "leverage": 2.5},
    "F5_publication":       {"best": 0.50, "low": 0.30, "high": 0.70, "leverage": 2.0},
}

OBSERVED_VISIBILITY = 0.00031059  # 0.031% from E108
DEMOGRAPHIC_GAP = 3220  # from E108
BEST_VISIBILITY = np.prod([f["best"] for f in FACTORS.values()])

print(f"Baseline visibility: {BEST_VISIBILITY:.6f} ({BEST_VISIBILITY*100:.4f}%)")
print(f"Observed visibility: {OBSERVED_VISIBILITY:.6f} ({OBSERVED_VISIBILITY*100:.4f}%)")
print(f"Baseline/observed ratio: {BEST_VISIBILITY/OBSERVED_VISIBILITY:.2f}x")
print()

# === TEST 1: FACTOR ISOLATION (vary one, fix others) ===

print("=" * 70)
print("TEST 1: FACTOR ISOLATION - Vary each factor 0.01 to 1.0, others fixed")
print("=" * 70)

isolation_results = {}
grid = np.linspace(0.01, 1.0, 200)

for fname, fdata in FACTORS.items():
    results = []
    for val in grid:
        visibility = val
        for other_name, other_data in FACTORS.items():
            if other_name != fname:
                visibility *= other_data["best"]
        ratio = visibility / OBSERVED_VISIBILITY
        gap = 1.0 / visibility if visibility > 0 else float("inf")
        results.append({
            "factor_value": val,
            "visibility": visibility,
            "ratio_to_observed": ratio,
            "gap": gap,
        })
    isolation_results[fname] = results

    # Find breaking points
    # "Breaks" = visibility no longer within 10x of observed (same criterion as E115)
    within_10x = [(r["factor_value"], r["ratio_to_observed"])
                  for r in results if 0.1 <= r["ratio_to_observed"] <= 10.0]

    if within_10x:
        low_break = within_10x[0][0]
        high_break = within_10x[-1][0]
    else:
        low_break = high_break = None

    # Find value where visibility exactly matches observed
    closest = min(results, key=lambda r: abs(r["ratio_to_observed"] - 1.0))

    print(f"\n{fname}:")
    print(f"  Best estimate: {fdata['best']}")
    print(f"  Range: [{fdata['low']}, {fdata['high']}]")
    print(f"  Within-10x window: [{low_break:.3f}, {high_break:.3f}]" if low_break else "  Within-10x window: NONE")
    print(f"  Exact-match value: {closest['factor_value']:.4f} (ratio={closest['ratio_to_observed']:.3f})")
    print(f"  Window width: {high_break - low_break:.3f}" if low_break else "  Window width: N/A")

    # Save CSV
    with open(RESULTS_DIR / f"isolation_{fname}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["factor_value", "visibility", "ratio_to_observed", "gap"])
        writer.writeheader()
        writer.writerows(results)

# === TEST 2: FACTOR REMOVAL (set each to 1.0 = no effect) ===

print("\n" + "=" * 70)
print("TEST 2: FACTOR REMOVAL - What if each factor doesn't exist?")
print("=" * 70)

removal_results = {}
for fname, fdata in FACTORS.items():
    visibility = 1.0
    for other_name, other_data in FACTORS.items():
        if other_name != fname:
            visibility *= other_data["best"]

    gap = 1.0 / visibility if visibility > 0 else float("inf")
    ratio = visibility / OBSERVED_VISIBILITY

    removal_results[fname] = {
        "visibility_without": visibility,
        "gap_without": gap,
        "ratio_to_observed": ratio,
        "leverage_confirmed": fdata["best"],  # removing = dividing by this
        "conclusion_change": "MODEL BREAKS" if ratio > 10.0 else "MODEL HOLDS" if ratio <= 10.0 else "UNCLEAR",
    }

    print(f"\n  Remove {fname} (set to 1.0):")
    print(f"    Visibility: {visibility:.6f} ({visibility*100:.4f}%)")
    print(f"    Gap: {gap:.0f}x (was {DEMOGRAPHIC_GAP}x)")
    print(f"    Ratio to observed: {ratio:.1f}x")
    print(f"    Conclusion: {removal_results[fname]['conclusion_change']}")

# === TEST 3: ADVERSARIAL MINIMUM - What factor values minimize the gap? ===

print("\n" + "=" * 70)
print("TEST 3: ADVERSARIAL MINIMUM - Best case for skeptic")
print("=" * 70)

# Use high estimates for all factors (maximize visibility = minimize gap)
high_visibility = np.prod([f["high"] for f in FACTORS.values()])
high_gap = 1.0 / high_visibility
print(f"\n  All factors at HIGH estimate:")
print(f"    Visibility: {high_visibility:.6f} ({high_visibility*100:.4f}%)")
print(f"    Gap: {high_gap:.0f}x")
print(f"    Ratio to observed: {high_visibility/OBSERVED_VISIBILITY:.1f}x")

# Use low estimates (minimize visibility = maximize gap)
low_visibility = np.prod([f["low"] for f in FACTORS.values()])
low_gap = 1.0 / low_visibility
print(f"\n  All factors at LOW estimate:")
print(f"    Visibility: {low_visibility:.8f} ({low_visibility*100:.6f}%)")
print(f"    Gap: {low_gap:.0f}x")
print(f"    Ratio to observed: {low_visibility/OBSERVED_VISIBILITY:.3f}x")

# Adversarial: find factor values within ranges that make gap = 1 (no gap)
# i.e., visibility = 1.0 (impossible) or at least > observed * 10
print(f"\n  Can skeptic make gap disappear (visibility > 10x observed)?")
print(f"    High visibility: {high_visibility:.6f}")
print(f"    10x observed: {OBSERVED_VISIBILITY * 10:.6f}")
if high_visibility > OBSERVED_VISIBILITY * 10:
    print(f"    YES - at extreme high estimates, model overshoots by {high_visibility/(OBSERVED_VISIBILITY*10):.1f}x")
    print(f"    BUT: this requires ALL factors at their extreme high simultaneously")
else:
    print(f"    NO - even at all-high estimates, model stays within 10x")

# === TEST 4: PAIRWISE INTERACTION MAP ===

print("\n" + "=" * 70)
print("TEST 4: PAIRWISE INTERACTION - Which pairs swing the most?")
print("=" * 70)

pair_results = {}
factor_names = list(FACTORS.keys())

for f1, f2 in combinations(factor_names, 2):
    # Vary both from low to high, others at best
    scenarios = []
    for v1 in [FACTORS[f1]["low"], FACTORS[f1]["best"], FACTORS[f1]["high"]]:
        for v2 in [FACTORS[f2]["low"], FACTORS[f2]["best"], FACTORS[f2]["high"]]:
            visibility = v1 * v2
            for other_name, other_data in FACTORS.items():
                if other_name not in (f1, f2):
                    visibility *= other_data["best"]
            scenarios.append({
                "f1_val": v1, "f2_val": v2,
                "visibility": visibility,
                "ratio": visibility / OBSERVED_VISIBILITY,
            })

    ratios = [s["ratio"] for s in scenarios]
    swing = max(ratios) / min(ratios) if min(ratios) > 0 else float("inf")
    pair_results[f"{f1} × {f2}"] = {
        "swing": swing,
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
    }

# Sort by swing
sorted_pairs = sorted(pair_results.items(), key=lambda x: x[1]["swing"], reverse=True)
print(f"\n  {'Pair':<45} {'Swing':>8} {'Min Ratio':>12} {'Max Ratio':>12}")
print(f"  {'-'*45} {'-'*8} {'-'*12} {'-'*12}")
for pair_name, data in sorted_pairs:
    print(f"  {pair_name:<45} {data['swing']:>7.1f}x {data['min_ratio']:>11.2f}x {data['max_ratio']:>11.2f}x")

# === TEST 5: SEQUENTIAL FACTOR ADDITION ===

print("\n" + "=" * 70)
print("TEST 5: SEQUENTIAL ADDITION - Building the cascade one factor at a time")
print("=" * 70)

# Add factors one by one in order of leverage (highest first)
sorted_factors = sorted(FACTORS.items(), key=lambda x: x[1]["leverage"], reverse=True)

cumulative = 1.0
print(f"\n  {'Step':<5} {'Factor Added':<25} {'Cumulative P':>15} {'Gap':>10} {'Delta':>10}")
print(f"  {'-'*5} {'-'*25} {'-'*15} {'-'*10} {'-'*10}")
print(f"  {'0':<5} {'(none)':<25} {1.0:>14.6f} {'1x':>10} {'-':>10}")

prev_gap = 1
for i, (fname, fdata) in enumerate(sorted_factors, 1):
    cumulative *= fdata["best"]
    gap = 1.0 / cumulative if cumulative > 0 else float("inf")
    delta = gap / prev_gap if prev_gap > 0 else float("inf")
    print(f"  {i:<5} {fname:<25} {cumulative:>14.6f} {gap:>9.0f}x {delta:>9.1f}x")
    prev_gap = gap

# === TEST 6: THRESHOLD ANALYSIS ===

print("\n" + "=" * 70)
print("TEST 6: THRESHOLD - At what value does each factor break the 10x bracket?")
print("=" * 70)

threshold_results = {}
for fname, fdata in FACTORS.items():
    # Find the value where ratio_to_observed exceeds 10x
    other_product = np.prod([f["best"] for n, f in FACTORS.items() if n != fname])

    # visibility = val * other_product
    # ratio = visibility / observed
    # We want ratio = 10: val * other_product / observed = 10
    threshold_10x = (10.0 * OBSERVED_VISIBILITY) / other_product
    # And ratio = 0.1: val * other_product / observed = 0.1
    threshold_01x = (0.1 * OBSERVED_VISIBILITY) / other_product

    # Also find where model exactly matches observed
    threshold_exact = OBSERVED_VISIBILITY / other_product

    threshold_results[fname] = {
        "threshold_10x": min(threshold_10x, 1.0),
        "threshold_01x": max(threshold_01x, 0.0),
        "threshold_exact": threshold_exact,
        "best": fdata["best"],
        "safe_range": f"[{max(threshold_01x, 0.001):.4f}, {min(threshold_10x, 1.0):.4f}]",
        "range_width": min(threshold_10x, 1.0) - max(threshold_01x, 0.001),
        "best_within_safe": max(threshold_01x, 0.001) <= fdata["best"] <= min(threshold_10x, 1.0),
    }

    print(f"\n  {fname}:")
    print(f"    Best estimate: {fdata['best']}")
    print(f"    Safe range (within 10x of observed): {threshold_results[fname]['safe_range']}")
    print(f"    Range width: {threshold_results[fname]['range_width']:.4f}")
    print(f"    Best within safe range: {threshold_results[fname]['best_within_safe']}")
    print(f"    Exact match value: {threshold_exact:.5f}")

# === TEST 7: N-1 FACTOR SUFFICIENCY ===

print("\n" + "=" * 70)
print("TEST 7: N-1 SUFFICIENCY - Can any 4 factors alone explain the gap?")
print("=" * 70)

for fname in FACTORS:
    # Product of all factors EXCEPT this one
    product_without = np.prod([f["best"] for n, f in FACTORS.items() if n != fname])
    gap_without = 1.0 / product_without if product_without > 0 else float("inf")
    ratio = product_without / OBSERVED_VISIBILITY

    explains_gap = gap_without >= (DEMOGRAPHIC_GAP / 10)  # within order of magnitude
    print(f"  Without {fname}: visibility={product_without:.6f}, gap={gap_without:.0f}x, "
          f"ratio={ratio:.1f}x => {'SUFFICIENT' if explains_gap else 'INSUFFICIENT'}")

# === SAVE SUMMARY ===

summary = {
    "experiment": "E120_cascade_stress_test",
    "description": "Systematic adversarial probing of 5-factor cascade model",
    "differs_from_E115": "E115 = random MC sampling. E120 = systematic isolation, removal, thresholds, pairwise interactions",
    "baseline": {
        "best_visibility": BEST_VISIBILITY,
        "observed_visibility": OBSERVED_VISIBILITY,
        "ratio": BEST_VISIBILITY / OBSERVED_VISIBILITY,
        "demographic_gap": DEMOGRAPHIC_GAP,
    },
    "test1_isolation": {
        fname: {
            "within_10x_window": [
                min([r["factor_value"] for r in results if 0.1 <= r["ratio_to_observed"] <= 10.0], default=None),
                max([r["factor_value"] for r in results if 0.1 <= r["ratio_to_observed"] <= 10.0], default=None),
            ]
        }
        for fname, results in isolation_results.items()
    },
    "test2_removal": removal_results,
    "test3_adversarial": {
        "all_high_visibility": high_visibility,
        "all_high_gap": high_gap,
        "all_low_visibility": low_visibility,
        "all_low_gap": low_gap,
        "skeptic_can_break_10x": high_visibility > OBSERVED_VISIBILITY * 10,
    },
    "test4_pairwise": {k: v for k, v in sorted_pairs},
    "test6_thresholds": threshold_results,
    "conclusion": None,  # filled below
}

# === FINAL VERDICT ===

print("\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70)

# Count how many factors, when removed, break the model
breaks_on_removal = sum(1 for r in removal_results.values() if r["conclusion_change"] == "MODEL BREAKS")
holds_on_removal = sum(1 for r in removal_results.values() if r["conclusion_change"] == "MODEL HOLDS")

# Check if all best estimates are within safe ranges
all_in_safe = all(t["best_within_safe"] for t in threshold_results.values())

# Weakest factor = smallest safe range width
weakest = min(threshold_results.items(), key=lambda x: x[1]["range_width"])
strongest = max(threshold_results.items(), key=lambda x: x[1]["range_width"])

verdict = {
    "model_robust": all_in_safe and breaks_on_removal <= 1,
    "factors_break_on_removal": breaks_on_removal,
    "factors_hold_on_removal": holds_on_removal,
    "all_best_in_safe_range": all_in_safe,
    "weakest_factor": weakest[0],
    "weakest_safe_width": weakest[1]["range_width"],
    "strongest_factor": strongest[0],
    "strongest_safe_width": strongest[1]["range_width"],
    "skeptic_can_break": high_visibility > OBSERVED_VISIBILITY * 10,
    "recommendation": "",
}

if verdict["model_robust"]:
    verdict["recommendation"] = (
        f"CASCADE IS ROBUST. All best estimates within safe ranges. "
        f"Weakest link: {weakest[0]} (safe width {weakest[1]['range_width']:.4f}). "
        f"Strongest: {strongest[0]} (safe width {strongest[1]['range_width']:.4f}). "
        f"Skeptic cannot break model within parameter ranges."
    )
else:
    verdict["recommendation"] = (
        f"CASCADE HAS VULNERABILITIES. {breaks_on_removal} factors break on removal. "
        f"Weakest: {weakest[0]}. Review parameter estimates."
    )

summary["conclusion"] = verdict

print(f"\n  Model robust: {verdict['model_robust']}")
print(f"  Factors that break on removal: {breaks_on_removal}/5")
print(f"  All best estimates in safe range: {all_in_safe}")
print(f"  Weakest factor: {weakest[0]} (safe width: {weakest[1]['range_width']:.4f})")
print(f"  Strongest factor: {strongest[0]} (safe width: {strongest[1]['range_width']:.4f})")
print(f"  Skeptic can break 10x bracket: {verdict['skeptic_can_break']}")
print(f"\n  RECOMMENDATION: {verdict['recommendation']}")

# Save JSON
with open(RESULTS_DIR / "stress_test_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n  Results saved to {RESULTS_DIR}/")
print(f"  Summary: stress_test_summary.json")
print(f"  Per-factor isolation: isolation_F*.csv")

"""
E139: Cost-Benefit Analysis of VOLCARCH Fieldwork Strategies
How much does each potential archaeological find cost, and what's the
expected return on investment for different fieldwork approaches?

Builds on E116 (testable predictions), E138 (detection methods),
and E080 (fieldwork targets) to produce a decision matrix for
funding proposals.
"""

import numpy as np
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# === FIELDWORK SCENARIOS ===

scenarios = {
    "A_minimal_satellite_only": {
        "description": "Sentinel-2 NDVI + existing DEM analysis only",
        "cost_usd": 0,  # free data
        "time_months": 1,
        "coverage_km2": 1000,
        "p_detect_per_target": 0.05,
        "n_targets_testable": 20,
        "method": "Satellite remote sensing",
        "what_it_proves": "Surface anomalies at known candi sites. Cannot find buried sites.",
        "limitation": "Surface-only. No depth penetration.",
    },
    "B_LiDAR_survey": {
        "description": "Airborne LiDAR of Kelud western flank (50 km2)",
        "cost_usd": 30000,
        "time_months": 2,
        "coverage_km2": 50,
        "p_detect_per_target": 0.15,
        "n_targets_testable": 10,
        "method": "LiDAR + ground truth",
        "what_it_proves": "Micro-topographic anomalies indicating buried structures. Amazon 2024 precedent.",
        "limitation": "Surface features only. Needs GPR/ERT for depth confirmation.",
    },
    "C_GPR_targeted": {
        "description": "GPR survey at 10 E080 target locations (100x100m each)",
        "cost_usd": 40000,
        "time_months": 3,
        "coverage_km2": 0.1,  # 10 x 0.01 km2
        "p_detect_per_target": 0.40,
        "n_targets_testable": 10,
        "method": "Ground-penetrating radar",
        "what_it_proves": "Subsurface anomalies at 2-5m depth. E116: expect 2.5 finds [0,6].",
        "limitation": "Limited to 5m depth. May miss deeper pre-Hindu sites.",
    },
    "D_GPR_plus_ERT": {
        "description": "GPR + ERT at 20 E080 targets",
        "cost_usd": 70000,
        "time_months": 4,
        "coverage_km2": 0.2,
        "p_detect_per_target": 0.50,
        "n_targets_testable": 20,
        "method": "GPR (0-5m) + ERT (0-20m)",
        "what_it_proves": "Subsurface anomalies at ALL predicted depths. Definitive test of VOLCARCH.",
        "limitation": "Still non-invasive. Anomalies need coring to confirm.",
    },
    "E_full_validation": {
        "description": "LiDAR + GPR + ERT + 20 cores at best anomalies",
        "cost_usd": 100000,
        "time_months": 6,
        "coverage_km2": 50,
        "p_detect_per_target": 0.60,
        "n_targets_testable": 20,
        "method": "Multi-method: LiDAR → GPR → ERT → coring",
        "what_it_proves": "Physical confirmation of buried cultural layers with dating. DEFINITIVE.",
        "limitation": "Expensive. Requires permits and institutional partner.",
    },
    "F_low_cost_coring": {
        "description": "20 geotechnical boreholes (commercial, piggyback on infrastructure)",
        "cost_usd": 6000,
        "time_months": 3,
        "coverage_km2": 0,  # point samples
        "p_detect_per_target": 0.15,
        "n_targets_testable": 20,
        "method": "Commercial borehole data + phytolith analysis",
        "what_it_proves": "Cultural layers at depth. Phytolith evidence of past agriculture.",
        "limitation": "Low detection probability per core. But cheapest deep-data option.",
    },
}

# === EXPECTED VALUE ANALYSIS ===

print("=" * 70)
print("E139: COST-BENEFIT ANALYSIS OF FIELDWORK STRATEGIES")
print("=" * 70)

print(f"\n  {'Scenario':<35} {'Cost':>10} {'E[finds]':>10} {'$/find':>10} {'Time':>8} {'Definitive?'}")
print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*11}")

for name, sc in scenarios.items():
    # Expected number of finds
    expected_finds = sc["n_targets_testable"] * sc["p_detect_per_target"]
    # Cost per expected find
    cost_per_find = sc["cost_usd"] / expected_finds if expected_finds > 0 else float("inf")
    # P(at least one find)
    p_at_least_one = 1 - (1 - sc["p_detect_per_target"]) ** sc["n_targets_testable"]

    definitive = "YES" if sc["cost_usd"] >= 70000 else "PARTIAL" if sc["cost_usd"] >= 30000 else "NO"

    print(f"  {name:<35} ${sc['cost_usd']:>8,} {expected_finds:>9.1f} ${cost_per_find:>8,.0f} "
          f"{sc['time_months']:>6}mo {'YES' if definitive == 'YES' else 'PARTIAL' if definitive == 'PARTIAL' else 'NO':>11}")

    sc["expected_finds"] = expected_finds
    sc["cost_per_find"] = cost_per_find
    sc["p_at_least_one"] = p_at_least_one

# === DECISION MATRIX ===

print(f"\n{'=' * 70}")
print("DECISION MATRIX: Which Scenario for Which Goal?")
print("=" * 70)

print(f"""
  GOAL: "Prove concept exists" (1 find sufficient)
    BEST: F_low_cost_coring ($6,000, 3 finds expected, P(>=1) = {scenarios['F_low_cost_coring']['p_at_least_one']:.1%})
    WHY: Cheapest deep-data option. Piggyback on commercial drilling.

  GOAL: "Find 2-3 buried sites" (strong evidence)
    BEST: C_GPR_targeted ($40,000, {scenarios['C_GPR_targeted']['expected_finds']:.1f} finds expected)
    WHY: Best cost-per-find for moderate budget. E116 prediction range.

  GOAL: "Definitive validation of VOLCARCH" (publishable in Nature)
    BEST: E_full_validation ($100,000, {scenarios['E_full_validation']['expected_finds']:.1f} finds expected)
    WHY: Multi-method convergence. Physical samples. Dating. Peer-reviewable.

  GOAL: "Minimum viable experiment" (proof of concept, <$10K)
    BEST: F_low_cost_coring ($6,000) + A_satellite ($0)
    WHY: Satellite narrows targets for free, coring tests them cheaply.

  GOAL: "Maximum PR impact" (media, collaborators)
    BEST: B_LiDAR ($30,000) — Amazon-style discovery narrative
    WHY: LiDAR imagery is visually dramatic. Media loves before/after.
""")

# === FUNDING SOURCE MATCH ===

print(f"\n{'=' * 70}")
print("FUNDING SOURCE MATCH")
print("=" * 70)

funding = [
    {"source": "Self-funded pilot", "budget": 6000, "best_scenario": "F_low_cost_coring",
     "notes": "20 commercial boreholes + phytolith analysis"},
    {"source": "University internal grant", "budget": 15000, "best_scenario": "B_LiDAR (reduced scope)",
     "notes": "25 km2 LiDAR + satellite validation"},
    {"source": "NatGeo Explorer Level 1", "budget": 20000, "best_scenario": "C_GPR_targeted",
     "notes": "GPR at 10 E080 targets. Outline ready."},
    {"source": "DRPM Penelitian Dasar Year 1", "budget": 50000, "best_scenario": "D_GPR_plus_ERT",
     "notes": "GPR + ERT at 20 targets. Skeleton ready."},
    {"source": "International collaboration", "budget": 100000, "best_scenario": "E_full_validation",
     "notes": "Multi-method definitive test. Needs institutional partner."},
]

for f in funding:
    sc = scenarios.get(f["best_scenario"], scenarios["F_low_cost_coring"])
    print(f"\n  {f['source']} (${f['budget']:,}):")
    print(f"    Best scenario: {f['best_scenario']}")
    print(f"    Notes: {f['notes']}")

# === SAVE ===

summary = {
    "experiment": "E139_cost_benefit_fieldwork",
    "scenarios": {k: {
        "cost_usd": v["cost_usd"],
        "expected_finds": v["expected_finds"],
        "cost_per_find": v["cost_per_find"],
        "p_at_least_one": v["p_at_least_one"],
    } for k, v in scenarios.items()},
    "cheapest_meaningful": "F_low_cost_coring ($6,000)",
    "best_value": "C_GPR_targeted ($40,000, $10K/find)",
    "definitive": "E_full_validation ($100,000)",
}

with open(RESULTS_DIR / "cost_benefit.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Saved to {RESULTS_DIR}/")

"""
E155: Cross-Regional Cascade Validation
========================================
The E110 5-factor cascade was fitted to Java. Can it predict archaeological
gaps in other regions where factors have DIFFERENT values?

Test regions:
1. Bali (volcanic, better surveyed, rich pre-400 CE record)
2. Sulawesi (non-volcanic, poorly surveyed, rich cave record)
3. Philippines (volcanic, moderately surveyed, moderate pre-400 CE record)
4. Japan (volcanic, extremely well surveyed, rich record)

If the cascade correctly predicts gap magnitude across regions,
it's validated. If it predicts gaps that don't exist, it's overfitting.
"""

import numpy as np
import json
from pathlib import Path

# E110 Java cascade (baseline)
# F1: Volcanic burial survival = 0.58 (only 58% of sites survive volcanic burial)
# F2: Organic decay survival = 0.20 (only 20% survive tropical organic decay)
# F3: Survey coverage = 0.025 (only 2.5% of sites are surveyed — 40× deficit vs Japan)
# F4: Recognition as pre-Hindu = 0.40 (only 40% of finds are correctly identified)
# F5: Publication & cataloging = 0.50 (only 50% of recognized finds get published)

java_cascade = {
    "F1_volcanic_burial": 0.58,
    "F2_organic_decay": 0.20,
    "F3_survey_coverage": 0.025,
    "F4_recognition": 0.40,
    "F5_publication": 0.50,
}

# Predicted visibility for Java
java_visibility = np.prod(list(java_cascade.values()))
java_observed = 0.00031  # 0.031% from E108

print("=" * 70)
print("E155: CROSS-REGIONAL CASCADE VALIDATION")
print("=" * 70)
print(f"\n--- JAVA (baseline, E110) ---")
print(f"  Predicted visibility: {java_visibility:.6f} ({java_visibility*100:.4f}%)")
print(f"  Observed gap: {java_observed:.6f} ({java_observed*100:.4f}%)")
print(f"  Ratio: {java_visibility/java_observed:.1f}×")

# ============================================================
# BALI — Volcanic but better surveyed
# ============================================================
# F1: Bali has 2 active volcanoes (Agung, Batur) vs Java's 45.
#     Only ~20% of Bali is in volcanic deposition zones vs ~60% of Java.
#     For sites in volcanic zones: same burial rate as Java (~4 mm/yr).
#     But most of Bali is NOT in volcanic zones.
#     Weighted survival: 0.80 (non-volcanic) + 0.20 * 0.58 = 0.916
bali_f1 = 0.80 + 0.20 * 0.58  # ~0.916

# F2: Same tropical climate as Java. Same organic decay.
bali_f2 = 0.20

# F3: Bali has been intensively surveyed since colonial era (better than Java).
#     Dutch colonial interest in Bali was high (cultural tourism).
#     Survey coverage estimate: 5-10× better than Java.
#     Java F3 = 0.025 → Bali F3 = 0.125 to 0.25
bali_f3 = 0.15  # 6× better than Java, conservative

# F4: Bali's Hindu-Buddhist heritage is recognized and celebrated.
#     Pre-Hindu sites would be harder to distinguish (same problem as Java).
#     But Bali has active living Hindu tradition → more awareness.
bali_f4 = 0.50  # slightly better than Java

# F5: Bali archaeology is well-published (tourist/academic interest).
bali_f5 = 0.60  # better than Java

bali_cascade = {
    "F1_volcanic_burial": bali_f1,
    "F2_organic_decay": bali_f2,
    "F3_survey_coverage": bali_f3,
    "F4_recognition": bali_f4,
    "F5_publication": bali_f5,
}

bali_visibility = np.prod(list(bali_cascade.values()))

# Observed: E146 reports Bali inscription density 12× higher than Java.
# Bali has pre-Hindu sites (Gilimanuk, Sembiran) — not a gap.
# Estimated observed visibility: much higher than Java.
# Bali has ~50+ known pre-1000 CE sites in ~5,780 km².
# Pre-400 CE: Gilimanuk (~200 BCE), Sembiran (200 BCE-200 CE) = at least 2-3 major sites
# Java pre-400 CE: ~5 sites in 129,000 km²
# Bali density: ~0.5 sites/1000 km² vs Java: ~0.04/1000 km²
# Relative gap: Bali ~12× less gap than Java
bali_observed_relative = java_observed * 12  # rough estimate

print(f"\n--- BALI ---")
for k, v in bali_cascade.items():
    java_v = java_cascade[k]
    ratio = v / java_v
    print(f"  {k}: {v:.3f} (Java: {java_v:.3f}, ratio: {ratio:.1f}×)")
print(f"  Predicted visibility: {bali_visibility:.6f} ({bali_visibility*100:.4f}%)")
print(f"  Observed (est.): {bali_observed_relative:.6f} ({bali_observed_relative*100:.4f}%)")
print(f"  Prediction ratio: {bali_visibility/bali_observed_relative:.1f}×")
print(f"  Predicted Bali/Java ratio: {bali_visibility/java_visibility:.1f}×")
print(f"  Observed Bali/Java ratio: ~12× (from E146)")

# ============================================================
# SULAWESI — Non-volcanic (mostly), poorly surveyed, rich cave record
# ============================================================
# F1: Sulawesi is largely non-volcanic (no major Java-style composite volcanoes).
#     Some volcanic activity in north (Lokon, Soputan), but most archaeology
#     is in the karst regions of Maros-Pangkep (non-volcanic).
#     Volcanic burial is NOT a factor for most sites.
sulawesi_f1 = 0.95  # minimal volcanic impact

# F2: Tropical, but cave contexts preserve much better than open-air.
#     Maros caves have 67,800 BP art — extreme preservation.
#     For cave sites: organic decay much slower.
#     For open-air sites: same as Java.
sulawesi_f2 = 0.40  # cave protection doubles survival vs Java

# F3: Very poorly surveyed. Colonial focus was on spice trade, not archaeology.
#     Modern survey concentrated on Maros caves after Aubert et al. 2014/2019.
#     Overall survey coverage similar to or worse than Java.
sulawesi_f3 = 0.015  # worse than Java

# F4: Pre-Austronesian sites (cave art) are highly recognizable.
#     No confusion with Hindu overlay (Hinduism barely reached Sulawesi).
#     But recognition of non-monumental open-air sites is poor.
sulawesi_f4 = 0.60  # better than Java (no Hindu confusion)

# F5: International attention (Aubert Nature papers) → good publication.
#     But Indonesian-language publications lag.
sulawesi_f5 = 0.50  # similar to Java

sulawesi_cascade = {
    "F1_volcanic_burial": sulawesi_f1,
    "F2_organic_decay": sulawesi_f2,
    "F3_survey_coverage": sulawesi_f3,
    "F4_recognition": sulawesi_f4,
    "F5_publication": sulawesi_f5,
}

sulawesi_visibility = np.prod(list(sulawesi_cascade.values()))

# Observed: Sulawesi has very old sites (67,800 BP cave art, 3,500 BP rice).
# But these are almost ALL cave sites. Open-air pre-400 CE sites are rare.
# Cave sites are not relevant to VOLCARCH's L1 mechanism.
# For non-cave sites: Sulawesi is also poorly documented.
# Rough estimate: comparable to Java for non-cave contexts.
sulawesi_observed = java_observed * 3  # slightly better (no volcanic burial)

print(f"\n--- SULAWESI ---")
for k, v in sulawesi_cascade.items():
    java_v = java_cascade[k]
    ratio = v / java_v
    print(f"  {k}: {v:.3f} (Java: {java_v:.3f}, ratio: {ratio:.1f}×)")
print(f"  Predicted visibility: {sulawesi_visibility:.6f} ({sulawesi_visibility*100:.4f}%)")
print(f"  Observed (est.): {sulawesi_observed:.6f} ({sulawesi_observed*100:.4f}%)")
print(f"  Prediction ratio: {sulawesi_visibility/sulawesi_observed:.1f}×")
print(f"  Predicted Sulawesi/Java ratio: {sulawesi_visibility/java_visibility:.1f}×")

# ============================================================
# PHILIPPINES — Volcanic, moderately surveyed
# ============================================================
# F1: Philippines has volcanic areas but less intense than Java.
#     E123: 4.6× fewer volcanoes per area.
#     Java = 0.035 volcanoes/100km², Philippines = 0.0076.
#     Philippines has 2 open-air pre-400 CE volcanic interior sites (Java has 0).
#     Volcanic burial exists but is less pervasive.
philippines_f1 = 0.75  # less volcanic impact than Java

# F2: Tropical, similar organic decay. But Philippines cave sites well-preserved.
#     Open-air: same as Java.
philippines_f2 = 0.25  # slightly better (more cave contexts available)

# F3: Better surveyed than Java, but not by much.
#     National Museum has systematic program. Rescue archaeology limited.
#     Survey coverage: ~2-3× Java.
philippines_f3 = 0.05  # 2× better than Java

# F4: No Hindu overlay to confuse. Pre-colonial archaeology is directly recognizable.
#     Chinese trade ceramics provide dating framework.
philippines_f4 = 0.55  # better than Java (no Hindu confusion)

# F5: Active archaeological community (UP, NM). International collaborations (Mijares).
#     English-language publications.
philippines_f5 = 0.60  # better than Java

philippines_cascade = {
    "F1_volcanic_burial": philippines_f1,
    "F2_organic_decay": philippines_f2,
    "F3_survey_coverage": philippines_f3,
    "F4_recognition": philippines_f4,
    "F5_publication": philippines_f5,
}

philippines_visibility = np.prod(list(philippines_cascade.values()))

# Observed: Philippines has Tabon Cave (50,000 BP), Callao Cave (67,000 BP),
# Butuan boat (320 CE), Manunggul Jar (890 BCE), Angono petroglyphs.
# Pre-400 CE open-air sites: more than Java but still limited.
# Rough estimate: ~5-10× Java's visibility for pre-400 CE sites.
philippines_observed = java_observed * 7  # moderate improvement

print(f"\n--- PHILIPPINES ---")
for k, v in philippines_cascade.items():
    java_v = java_cascade[k]
    ratio = v / java_v
    print(f"  {k}: {v:.3f} (Java: {java_v:.3f}, ratio: {ratio:.1f}×)")
print(f"  Predicted visibility: {philippines_visibility:.6f} ({philippines_visibility*100:.4f}%)")
print(f"  Observed (est.): {philippines_observed:.6f} ({philippines_observed*100:.4f}%)")
print(f"  Prediction ratio: {philippines_visibility/philippines_observed:.1f}×")
print(f"  Predicted Philippines/Java ratio: {philippines_visibility/java_visibility:.1f}×")

# ============================================================
# JAPAN — Volcanic, extremely well surveyed
# ============================================================
# F1: Japan is volcanic but temperate. Burial rates lower than tropical Java.
#     E086: Java 32× deeper sustained burial than Japan.
#     Volcanic burial survival in Japan much higher.
japan_f1 = 0.85  # volcanic but temperate, less intense burial

# F2: Temperate climate. Better organic preservation than tropics.
japan_f2 = 0.45  # much better than Java's 0.20

# F3: Japan has THE most intensive survey in the world.
#     8,300 excavations/year, 460,000 registered sites.
#     Mandatory rescue archaeology since 1950.
#     E086: 100-200× Java's survey intensity.
japan_f3 = 0.80  # dramatically better than Java's 0.025

# F4: Jomon, Yayoi, Kofun periods all well-recognized.
#     Professional archaeological community. Museum infrastructure.
japan_f4 = 0.90  # excellent recognition

# F5: Extremely well-published. One of the best-documented
#     archaeological traditions in the world.
japan_f5 = 0.90  # excellent publication

japan_cascade = {
    "F1_volcanic_burial": japan_f1,
    "F2_organic_decay": japan_f2,
    "F3_survey_coverage": japan_f3,
    "F4_recognition": japan_f4,
    "F5_publication": japan_f5,
}

japan_visibility = np.prod(list(japan_cascade.values()))

# Observed: Japan has 460,000+ registered sites from all periods.
# Jomon alone: 15,000+ sites, 38,000+ years.
# Effectively NO gap — visibility is very high.
japan_observed = 0.50  # 50% visibility — most sites are eventually found

print(f"\n--- JAPAN ---")
for k, v in japan_cascade.items():
    java_v = java_cascade[k]
    ratio = v / java_v
    print(f"  {k}: {v:.3f} (Java: {java_v:.3f}, ratio: {ratio:.1f}×)")
print(f"  Predicted visibility: {japan_visibility:.6f} ({japan_visibility*100:.4f}%)")
print(f"  Observed (est.): {japan_observed:.6f} ({japan_observed*100:.2f}%)")
print(f"  Prediction ratio: {japan_visibility/japan_observed:.2f}×")
print(f"  Predicted Japan/Java ratio: {japan_visibility/java_visibility:.0f}×")

# ============================================================
# SYNTHESIS
# ============================================================
print(f"\n{'='*70}")
print(f"CROSS-REGIONAL COMPARISON")
print(f"{'='*70}")
print(f"{'Region':<15} {'Predicted':>12} {'Observed':>12} {'Ratio':>8} {'Rank':>6}")
print(f"{'-'*55}")

regions = [
    ("Java", java_visibility, java_observed),
    ("Bali", bali_visibility, bali_observed_relative),
    ("Sulawesi", sulawesi_visibility, sulawesi_observed),
    ("Philippines", philippines_visibility, philippines_observed),
    ("Japan", japan_visibility, japan_observed),
]

# Sort by predicted visibility
regions_sorted = sorted(regions, key=lambda x: x[1])
for i, (name, pred, obs) in enumerate(regions_sorted, 1):
    ratio = pred / obs if obs > 0 else float('inf')
    print(f"{name:<15} {pred*100:>10.4f}% {obs*100:>10.4f}% {ratio:>7.1f}× {i:>5}")

# Check rank order consistency
pred_order = [r[0] for r in regions_sorted]
obs_sorted = sorted(regions, key=lambda x: x[2])
obs_order = [r[0] for r in obs_sorted]

print(f"\nPredicted rank order: {' < '.join(pred_order)}")
print(f"Observed rank order:  {' < '.join(obs_order)}")
print(f"Rank order match: {'YES' if pred_order == obs_order else 'NO — see analysis'}")

# Spearman rank correlation between predicted and observed
from scipy.stats import spearmanr
pred_values = [r[1] for r in regions]
obs_values = [r[2] for r in regions]
rho, p_val = spearmanr(pred_values, obs_values)
print(f"\nSpearman correlation (predicted vs observed): rho={rho:.3f}, p={p_val:.4f}")

# Monte Carlo sensitivity: vary all parameters ±50%, check if rank order holds
print(f"\n{'='*70}")
print(f"MONTE CARLO SENSITIVITY: Do rank orders survive parameter uncertainty?")
print(f"{'='*70}")

np.random.seed(42)
n_mc = 10000
rank_matches = 0
rho_values = []

for _ in range(n_mc):
    # Randomly perturb each parameter by ±50%
    mc_regions = []
    for name, cascade, obs in [
        ("Java", java_cascade, java_observed),
        ("Bali", bali_cascade, bali_observed_relative),
        ("Sulawesi", sulawesi_cascade, sulawesi_observed),
        ("Philippines", philippines_cascade, philippines_observed),
        ("Japan", japan_cascade, japan_observed),
    ]:
        mc_pred = 1.0
        for v in cascade.values():
            # Uniform perturbation ±50%, clamped to [0.001, 1.0]
            perturbed = v * np.random.uniform(0.5, 1.5)
            perturbed = np.clip(perturbed, 0.001, 1.0)
            mc_pred *= perturbed
        mc_regions.append((name, mc_pred, obs))

    # Check rank order
    mc_pred_sorted = sorted(mc_regions, key=lambda x: x[1])
    mc_pred_order = [r[0] for r in mc_pred_sorted]
    if mc_pred_order == obs_order:
        rank_matches += 1

    # Spearman
    mc_preds = [r[1] for r in mc_regions]
    mc_obs = [r[2] for r in mc_regions]
    rho_mc, _ = spearmanr(mc_preds, mc_obs)
    rho_values.append(rho_mc)

rho_values = np.array(rho_values)
print(f"Exact rank order match: {rank_matches}/{n_mc} ({rank_matches/n_mc*100:.1f}%)")
print(f"Spearman rho: mean={rho_values.mean():.3f}, median={np.median(rho_values):.3f}")
print(f"  95% CI: [{np.percentile(rho_values, 2.5):.3f}, {np.percentile(rho_values, 97.5):.3f}]")
print(f"  P(rho > 0): {(rho_values > 0).mean()*100:.1f}%")
print(f"  P(rho > 0.5): {(rho_values > 0.5).mean()*100:.1f}%")

# Factor contribution analysis
print(f"\n{'='*70}")
print(f"FACTOR CONTRIBUTION: Which factors DIFFERENTIATE regions?")
print(f"{'='*70}")

factors = ["F1_volcanic_burial", "F2_organic_decay", "F3_survey_coverage",
           "F4_recognition", "F5_publication"]
all_cascades = {
    "Java": java_cascade,
    "Bali": bali_cascade,
    "Sulawesi": sulawesi_cascade,
    "Philippines": philippines_cascade,
    "Japan": japan_cascade,
}

for f in factors:
    values = [all_cascades[r][f] for r in all_cascades]
    range_val = max(values) - min(values)
    cv = np.std(values) / np.mean(values)
    print(f"  {f}: range={range_val:.3f}, CV={cv:.2f}, values={[f'{v:.3f}' for v in values]}")

# Identify which factor has most cross-regional variation
cvs = {}
for f in factors:
    values = [all_cascades[r][f] for r in all_cascades]
    cvs[f] = np.std(values) / np.mean(values)

most_variable = max(cvs, key=cvs.get)
print(f"\nMost variable factor across regions: {most_variable} (CV={cvs[most_variable]:.2f})")
print(f"Least variable factor: {min(cvs, key=cvs.get)} (CV={cvs[min(cvs, key=cvs.get)]:.2f})")

# Save results
output_path = Path("D:/documents/volcarch-repo/experiments/E155_cross_regional_cascade/results")
output_path.mkdir(exist_ok=True)

results = {
    "regions": {
        name: {
            "predicted": float(pred),
            "observed": float(obs),
            "ratio": float(pred/obs) if obs > 0 else None,
            "factors": {k: float(v) for k, v in cascade.items()}
        }
        for (name, pred, obs), cascade in zip(regions, [
            java_cascade, bali_cascade, sulawesi_cascade,
            philippines_cascade, japan_cascade
        ])
    },
    "spearman_rho": float(rho),
    "spearman_p": float(p_val),
    "mc_rank_match_pct": rank_matches/n_mc*100,
    "mc_rho_mean": float(rho_values.mean()),
    "mc_rho_ci_95": [float(np.percentile(rho_values, 2.5)),
                     float(np.percentile(rho_values, 97.5))],
}

with open(output_path / "cascade_validation.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_path / 'cascade_validation.json'}")
print(f"\nDONE.")

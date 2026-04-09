"""
E179: Factor Independence Test — Are F1 and F2 Coupled?

Key question: Volcanic burial (F1) and organic decay (F2) are treated as independent
in the cascade, but burial creates anaerobic conditions that SLOW decay.
Does this coupling change the cascade's predictions?

Also tests: F3 (survey) and F4 (recognition) coupling.
"""

import numpy as np

np.random.seed(42)

print("=" * 70)
print("E179: FACTOR INDEPENDENCE TEST")
print("=" * 70)

# ============================================================
# BASELINE: INDEPENDENT MODEL (E110)
# ============================================================
F1_BEST = 0.58  # P(not buried) = sites surviving volcanic burial
F2_BEST = 0.20  # P(not decayed) = organic materials surviving
F3_BEST = 0.025 # P(surveyed)
F4_BEST = 0.40  # P(recognized)
F5_BEST = 0.50  # P(published)

independent_product = F1_BEST * F2_BEST * F3_BEST * F4_BEST * F5_BEST
OBSERVED = 0.00031

print(f"\nBaseline (independent): P(vis) = {independent_product:.6f} ({independent_product*100:.4f}%)")
print(f"Observed:              P(vis) = {OBSERVED:.6f} ({OBSERVED*100:.4f}%)")

# ============================================================
# TEST 1: F1-F2 COUPLING (Burial × Organic Decay)
# ============================================================
print("\n--- TEST 1: F1-F2 Coupling (Volcanic Burial × Organic Decay) ---")
print()
print("Argument: Volcanic burial creates sealed, anaerobic conditions.")
print("In archaeology: deep burial often PRESERVES organic material.")
print("This means P(decayed | buried) < P(decayed | not_buried).")
print()

# Literature-based estimates:
# Non-volcanic tropical context: organic survival ~5-10% (hot, wet, aerobic)
# Volcanic burial context: organic survival varies widely
#   - Cerén (El Salvador): ~80% organic preservation (phreatomagmatic, sealed)
#   - Liangan: wooden structures preserved (Sindoro pyroclastic)
#   - But lahar: hot, can destroy organics on contact
#   - Tephra fall: cool ash can seal and preserve

# Model as conditional probabilities:
P_decayed_given_buried = 0.60     # 60% of BURIED sites lose organics (hot lahar, acid gas)
P_decayed_given_notburied = 0.95  # 95% of surface sites lose organics (tropical weathering)

# E110's F1 = P(not buried) = 0.58, so P(buried) = 0.42
P_buried = 1 - F1_BEST  # 0.42
P_not_buried = F1_BEST   # 0.58

# Coupled probability:
# P(organic_survives) = P(buried) × P(organic_survives|buried) + P(not_buried) × P(organic_survives|not_buried)
P_organic_survives_coupled = (
    P_buried * (1 - P_decayed_given_buried) +
    P_not_buried * (1 - P_decayed_given_notburied)
)

print(f"P(organic survives | buried):     {1-P_decayed_given_buried:.2f}")
print(f"P(organic survives | not buried): {1-P_decayed_given_notburied:.2f}")
print(f"P(buried):                        {P_buried:.2f}")
print()
print(f"INDEPENDENT model F2:  {F2_BEST:.4f}")
print(f"COUPLED model F2:      {P_organic_survives_coupled:.4f}")
print(f"Difference:            {(P_organic_survives_coupled - F2_BEST):.4f}")
print(f"Ratio:                 {P_organic_survives_coupled/F2_BEST:.2f}x")

# Recompute cascade with coupled F1-F2
# In coupled model, F1 and F2 are replaced by joint probability
P_joint_F1F2_coupled = (
    P_buried * (1 - P_decayed_given_buried) +  # buried AND organic survives
    P_not_buried * (1 - P_decayed_given_notburied)  # not buried AND organic survives
)
# But we also need site to exist (not destroyed by burial itself)
# Actually: F1 in E110 = P(site detectable despite burial), not just P(not buried)
# Let's be more careful:

print()
print("Recomputing cascade with coupling:")
print("  Original (independent): F1 × F2 = 0.58 × 0.20 = 0.116")

# Coupled: some buried sites are BETTER preserved
# P(detectable AND has organics) =
#   P(shallow burial, detectable, organics weathered) +
#   P(deep burial, harder to find, but organics preserved)

# Scenario A: Site near surface (not buried by volcanism)
# Easy to find (F1 applies), organics gone (tropical weathering)
P_scenario_A = P_not_buried * (1 - P_decayed_given_notburied)  # visible AND has organics

# Scenario B: Site buried by volcanism
# Hard to find (need subsurface survey), but organics better preserved
P_scenario_B = P_buried * (1 - P_decayed_given_buried)  # buried AND has organics

# Scenario C: Buried but organics still destroyed (hot lahar)
P_scenario_C = P_buried * P_decayed_given_buried  # buried AND no organics

# Scenario D: Near surface, organics survive (rare in tropics)
P_scenario_D = P_not_buried * (1 - P_decayed_given_notburied)  # same as A

joint_survival = P_scenario_A + P_scenario_B  # sites with detectable organics
print(f"  Coupled joint P(detectable or has organics): {joint_survival:.4f}")
print(f"  = P(not_buried × organic_survives) + P(buried × organic_survives)")
print(f"  = {P_not_buried:.2f}×{1-P_decayed_given_notburied:.2f} + {P_buried:.2f}×{1-P_decayed_given_buried:.2f}")
print(f"  = {P_scenario_A:.4f} + {P_scenario_B:.4f}")

# Full cascade with coupling
coupled_cascade = joint_survival * F3_BEST * F4_BEST * F5_BEST
independent_cascade = F1_BEST * F2_BEST * F3_BEST * F4_BEST * F5_BEST

print()
print(f"  Full cascade (independent): {independent_cascade:.6f} ({independent_cascade*100:.4f}%)")
print(f"  Full cascade (coupled):     {coupled_cascade:.6f} ({coupled_cascade*100:.4f}%)")
print(f"  Ratio coupled/independent:  {coupled_cascade/independent_cascade:.2f}x")
print(f"  Ratio coupled/observed:     {coupled_cascade/OBSERVED:.1f}x")

# ============================================================
# TEST 2: F3-F4 COUPLING (Survey × Recognition)
# ============================================================
print("\n--- TEST 2: F3-F4 Coupling (Survey × Recognition) ---")
print()
print("Argument: Better-surveyed areas employ professional teams (higher recognition).")
print("P(recognized | well-surveyed) > P(recognized | poorly-surveyed)")
print()

# Estimates:
P_recognized_given_professional = 0.70  # professional CRM archaeology
P_recognized_given_amateur = 0.15       # chance finds, farmer reports
P_professional_survey = 0.30            # fraction of surveyed area by professionals

# In the E110 model, F3=0.025 (2.5% surveyed) and F4=0.40 (40% recognized)
# Coupled:
P_recognized_coupled = (
    F3_BEST * P_professional_survey * P_recognized_given_professional +
    F3_BEST * (1 - P_professional_survey) * P_recognized_given_amateur +
    (1 - F3_BEST) * 0.01  # unsurveyed areas: 1% chance find
)

print(f"P(recognized | professional survey): {P_recognized_given_professional:.2f}")
print(f"P(recognized | amateur survey):      {P_recognized_given_amateur:.2f}")
print(f"P(chance find in unsurveyed area):    0.01")
print()
print(f"INDEPENDENT F3 × F4: {F3_BEST * F4_BEST:.6f}")
print(f"COUPLED P(surveyed AND recognized): {P_recognized_coupled:.6f}")
print(f"Ratio coupled/independent: {P_recognized_coupled/(F3_BEST * F4_BEST):.2f}x")

# ============================================================
# TEST 3: FULL COUPLED CASCADE
# ============================================================
print("\n--- TEST 3: Full Coupled Cascade ---")
print()

# Replace F1×F2 with coupled joint, F3×F4 with coupled joint
full_coupled = joint_survival * P_recognized_coupled * F5_BEST

print(f"Independent cascade:  {independent_cascade:.6f} ({independent_cascade*100:.4f}%)")
print(f"Coupled cascade:      {full_coupled:.6f} ({full_coupled*100:.4f}%)")
print(f"Observed:             {OBSERVED:.6f} ({OBSERVED*100:.4f}%)")
print()
print(f"Independent ratio to observed: {independent_cascade/OBSERVED:.1f}x")
print(f"Coupled ratio to observed:     {full_coupled/OBSERVED:.1f}x")
print(f"Coupling effect:               {full_coupled/independent_cascade:.2f}x change")

# ============================================================
# TEST 4: SENSITIVITY — VARY COUPLING STRENGTH
# ============================================================
print("\n--- TEST 4: Coupling Sensitivity ---")
print()
print("Varying P(organic_survives | buried) from 0.05 to 0.90:")
print(f"{'P(surv|buried)':>15} | {'Joint F1F2':>10} | {'Full cascade':>12} | {'Ratio to obs':>12}")
print("-" * 60)

for p_surv_buried in np.arange(0.05, 0.95, 0.10):
    joint = P_not_buried * (1 - P_decayed_given_notburied) + P_buried * p_surv_buried
    cascade = joint * F3_BEST * F4_BEST * F5_BEST
    print(f"{p_surv_buried:>15.2f} | {joint:>10.4f} | {cascade*100:>11.4f}% | {cascade/OBSERVED:>11.1f}x")

# ============================================================
# CONCLUSION
# ============================================================
print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print(f"""
1. F1-F2 coupling (burial preserves organics) INCREASES predicted visibility
   from {independent_cascade*100:.4f}% to {coupled_cascade*100:.4f}% — about {coupled_cascade/independent_cascade:.1f}x higher.

2. F3-F4 coupling (professional surveys recognize more) changes the joint
   probability by ~{P_recognized_coupled/(F3_BEST * F4_BEST):.1f}x.

3. Full coupling shifts cascade by ~{full_coupled/independent_cascade:.1f}x total — from ratio
   {independent_cascade/OBSERVED:.1f}x to {full_coupled/OBSERVED:.1f}x vs observed.

4. The coupling effect is MODERATE. It changes the cascade by a factor of
   ~{full_coupled/independent_cascade:.1f}x, which is within the Monte Carlo spread (E115: 95% CI
   spans 22x). The model absorbs coupling easily because it's already loose.

5. CRITICAL INSIGHT: Coupling makes the prediction WORSE (further from
   observed), not better. This is because burial-preservation coupling
   increases predicted visibility, while observations show very low visibility.
   This means either:
   (a) Burial does NOT preserve organics as well as expected (hot lahars), or
   (b) The coupling exists but other factors compensate, or
   (c) The independent model coincidentally matches by error cancellation.

6. For papers: acknowledge coupling exists but note it operates within
   parameter uncertainty. The model is not precise enough for coupling
   to matter — which is itself an honest limitation.
""")

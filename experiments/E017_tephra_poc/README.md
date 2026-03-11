# E017: Tephra POC (Pyle 1989 Calibration for Paper 3)

**Status:** COMPLETE (FAILED — analytical approach insufficient)
**Date:** 2026-03-03
**Paper:** P3 feasibility test

## Hypothesis
The Pyle (1989) exponential thinning model, calibrated with a single-point loss factor, can predict burial depths at 3 of 4 calibration sites within +/-30%.

## Critical Insight
3 of 4 calibration sites are Merapi system (Central Java):
- Sambisari (~835 CE, 500-650 cm, Merapi ~10 km)
- Kedulan (~869 CE, 600-700 cm, Merapi ~12 km)
- Kimpulan (~900 CE, 270-500 cm, Merapi ~14 km)

Only Dwarapala is East Java (Kelud system).

## Method
1. Compile Merapi eruption summary (GVP 263250)
2. Apply Pyle (1989) with loss factor calibrated at Dwarapala
3. Predict burial at each calibration site
4. Score: +/-30% for 3/4 sites = Paper 3 GO (analytical approach sufficient)

## Pass Criteria
- 3 of 4 calibration sites predicted within +/-30%
- If FAIL: Paper 3 needs heavy simulation tools (Tephra2/FALL3D)

## Result
**FAILED (1/4 sites within +/-30%)**

Dwarapala (Kelud) calibration factor (29.1% retention) fails on all 3 Merapi sites — predicts ~115-142 cm vs actual 385-650 cm. The Kelud retention factor under-predicts Merapi burial by 3-5x.

**Key insight:** Merapi burial is dominated by pyroclastic density currents and lahars, not distal tephra fallout. The Pyle (1989) model only captures the tephra component. Cross-system calibration with a single loss factor is insufficient.

**Implication for Paper 3:** Needs per-volcano calibration (separate loss factors for Kelud vs Merapi systems) OR simulation tools (Tephra2/FALL3D + lahar routing). The analytical Pyle approach is a useful first-order bound but not sufficient for the +/-30% accuracy target.

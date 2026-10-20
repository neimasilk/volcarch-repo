# E235 — Elevation or volcanic setting? A band-by-band control for P2

**Status:** INCONCLUSIVE on suitability (M2); decisive on overlap (M1: **E\* = 1,000 m**); NOT SPECIFIC on design
disagreement (M3). Run 2026-10-20, as pre-registered.
**Pre-registration:** `DESIGN.md`, frozen with commit `3ac41cc` (pushed 2026-10-20 before the script existed).
**Line:** 01_spatial · **Serves:** P2 (JCAA #280 round-2 revision), Reviewer 2 R2-1 · **Ledger:** C077 ·
**Revision file:** `papers/P2_settlement_model/REVISION_JCAA_R2_20261019.md`

**Hypothesis:** where volcanic and non-volcanic terrain occur at the same elevation in East Java, the model's
suitability and the background designs' disagreement do not differ by volcanic setting. The reason to expect this:
no model feature carries volcanic information.

**Why:** the reviewer asked whether the model picks out volcanic ground or simply high ground. Our round-1 control
(E219 part C) had almost no non-volcanic land in its top elevation quintile (868 vs 41,878 matched cells), and the
manuscript still read it as "directionally consistent with a volcanic-specific effect".

**Method** (`DESIGN.md`):
- Data: the stored E221 ensemble maps (3 designs × 3 algorithms, 588,535 cells), joined exactly to the E217 frame.
- Setting: volcanic V = ≤ 20 km from any of the 30 canonical centres; non-volcanic N = ≥ 40 km.
- 200 m elevation bands. A band is usable when each arm has ≥ 25 km² and ≥ 2 spatial blocks.
- Within each band: coarsened exact matching on slope × TRI × TWI terciles.
- Uncertainty: spatial block bootstrap, 1,000 replicates, seed 235.
- Verdicts by fixed rules, δ = 0.05 suitability or 5 percentage points.

**Data:** `experiments/E221_seed_ensemble_stability/results/e221_priority_sets_*.npz` ·
`data/processed/dem/jatim_*.tif` (through `E217/01_maxent_benchmark.py::build_frame`) ·
`data/processed/dashboard/volcanoes_java_full.csv`.

## Result

**M1: overlap (the decisive fact).** In the primary run the usable bands are 0–200 … 800–1,000 m, so
**E\* = 1,000 m**. Above 1,000 m, East Java has 3,435 km² of land:
- **97.5 %** of it is within 20 km of a volcanic centre;
- **0.4 %** is 40 km or more away.

Non-volcanic land per band is 19,742, 2,701, 778, 435 and 185 km² from 0 to 1,000 m. It is **15 km²** (4 blocks) at
1,000–1,200 m and **zero above 1,200 m**. With the alternative definitions (V ≤ 10 or 15 km, N ≥ 30 km),
E\* = 1,200 m and the non-volcanic share above it is 0.0 %. **Within the study area, the comparison with
high-elevation non-volcanic terrain cannot be made: such terrain does not exist there.**

**M2: suitability, volcanic − non-volcanic, matched within bands up to 1,000 m.** XGBoost (primary), 53,927 matched
cells:

| Design | Δ̄ | 95 % CI | Verdict (δ 0.05) | at δ 0.03 | at δ 0.10 |
|---|---|---|---|---|---|
| random | +0.049 | +0.024 to +0.077 | INCONCLUSIVE | SPECIFIC | NOT SPECIFIC |
| TGB | +0.049 | +0.024 to +0.076 | INCONCLUSIVE | SPECIFIC | NOT SPECIFIC |
| hybrid | +0.060 | +0.030 to +0.091 | SPECIFIC | SPECIFIC | NOT SPECIFIC |

**Summary: INCONCLUSIVE** (2 of 3 designs). The difference is positive in every band, with no trend with elevation
(random: +0.047, +0.050, +0.055, +0.040, +0.064 from 0 to 1,000 m). Its size sits on the pre-declared threshold.
Sensitivity:
- **It vanishes when the volcanic zone is drawn tighter:** V ≤ 10 / N ≥ 30 km gives −0.002 (−0.047 to +0.046),
  NOT SPECIFIC.
- V ≤ 15 / N ≥ 30 km gives +0.016 (−0.016 to +0.051), INCONCLUSIVE.
- RandomForest: INCONCLUSIVE (primary), NOT SPECIFIC under both alternatives.
- MaxEnt: INCONCLUSIVE under all three definitions.

**M3: design disagreement.** In the same matched cells, the share of cells in the contingent fringe (top decile
under exactly one design) differs by +0.56 pp (+0.26 to +0.85). |hybrid − random| differs by +0.0074 (+0.0033 to
+0.0110). **Both are NOT SPECIFIC**, for all three algorithms and all three definitions (largest |Δ| 0.56 pp against
a 5 pp threshold).

**M4 (descriptive).** Within the usable bands, an elevation spline alone explains about 6 % (5.7–6.9 %) of XGBoost suitability
variance. Volcanic setting adds a partial R² of 0.032–0.038 for suitability and 0.015 for |hybrid − random|.

**M5 (descriptive only; M2 is not SPECIFIC overall).** Matched V cells face south more than N cells (mean cos(aspect)
−0.11, CI −0.19 to −0.03). River distance does not differ (−0.35 km, CI −1.84 to +0.81).

**Figure:** `results/e235_response_by_band.png`.
- A: land per band.
- B: unmatched band means of suitability for the three designs.
- C: contingent-fringe share.

**What the figure shows outside the overlap.** Above 1,000 m, where all land is volcanic, the designs part company:
- the hybrid design keeps suitability near 0.5;
- random and TGB fall to about 0.1;
- the fringe rises from 1–5 % below 1,000 m to about 25 %.

This is the "remote upland" fringe of manuscript Figure 5. In this study area it cannot be attributed to elevation
rather than volcanic setting, or the other way round.

**Verification (G1):** an independent pandas re-derivation (`groupby` matching; different code path, kept in the
2026-10-20 session scratchpad) reproduced every primary number exactly:
- 3,435 km², 97.5 %, 0.4 %;
- 53,927 matched cells;
- Δ̄ random +0.0491; fringe +0.563 pp.

## Conclusion

The reviewer's question has a clear answer, and it is mostly about the study area, not the model:
- **Above 1,000 m, elevation and volcanic setting are the same thing in East Java.** Nothing the model does up there
  can be called volcanic rather than elevational, or the reverse. The paper must say so and make no claim either way.
- **Below 1,000 m, the designs' disagreement does not depend on volcanic setting.**
- **Suitability is slightly higher near volcanoes than in terrain-matched ground far from them** (+0.05, at the
  threshold). It disappears under a tighter volcanic zone, so no volcanic effect is claimed (pre-registered branch:
  INCONCLUSIVE → report with numbers, no direction).

**What this changes in P2** (fixed by DESIGN §6; the PI writes the prose):
- "directionally consistent with a volcanic-specific effect" is removed;
- the n = 2 site-density comparison stops being evidence;
- E\* = 1,000 m and the 97.5 % / 0.4 % split are stated;
- the M2 numbers are reported as inconclusive;
- the M3 result is stated for the range where it holds;
- Figure 1 gets topography.

`results/e235_response_by_band.png` is a candidate manuscript figure; it shows the reviewer exactly where the two can
and cannot be separated.

**Outputs:** `01_elevation_vs_volcanic.py`, `results/e235_outcome.json`, `results/e235_bands.csv`,
`results/e235_response_by_band.png`.

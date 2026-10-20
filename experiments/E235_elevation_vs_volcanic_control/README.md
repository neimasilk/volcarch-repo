# E235 — Elevation or volcanic setting? A band-by-band control for P2

**Status:** INCONCLUSIVE. Run 2026-10-20, as pre-registered.
- **M1 (overlap) is decisive:** E\* = 1,000 m.
- **M2 (suitability) is INCONCLUSIVE**, and post hoc checks show it is not robust either way.
- **M3 (design disagreement):** the pre-registered verdict is "NOT SPECIFIC", but **the test had no power**, so that
  verdict carries no information.

See §Audit, added the same day after an independent audit. It corrects three sentences of the first write-up.

**Terminology (after the audit):** "volcanic" here means within 20 km of a **Quaternary** volcanic centre, and
"non-volcanic" means 40 km or more from one. The far-zone uplands are mostly the Southern Mountains, which are an
**Eocene–Miocene volcanic arc** (Smyth, Hall & Nichols 2008, GSA Special Paper 436, doi 10.1130/2008.2436(10);
IPA 2018, doi 10.29118/ipa.629.05.g.045). Read "non-volcanic" as "far from Quaternary centres".
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
E\* = 1,200 m and the non-volcanic share above it is 0.0 %. **Within the study area, high-elevation land far from
Quaternary centres hardly exists:** 15 km² at 1,000–1,200 m (below the 25 km² minimum) and none above 1,200 m. The
ceiling depends on that minimum: 800 m at about 190 km², 1,200 m below 15 km². *(Corrected after the audit: the
first version said such terrain "does not exist".)*

**M2: suitability, volcanic − non-volcanic, matched within bands up to 1,000 m.** XGBoost (primary), 53,927 matched
cells:

| Design | Δ̄ | 95 % CI | Verdict (δ 0.05) | at δ 0.03 | at δ 0.10 |
|---|---|---|---|---|---|
| random | +0.049 | +0.024 to +0.077 | INCONCLUSIVE | SPECIFIC | NOT SPECIFIC |
| TGB | +0.049 | +0.024 to +0.076 | INCONCLUSIVE | SPECIFIC | NOT SPECIFIC |
| hybrid | +0.060 | +0.030 to +0.091 | SPECIFIC | SPECIFIC | NOT SPECIFIC |

**Summary: INCONCLUSIVE** (2 of 3 designs). The difference is positive in every band, with no trend with elevation
(random: +0.047, +0.050, +0.055, +0.040, +0.064 from 0 to 1,000 m). Its size sits on the pre-declared threshold.
Pre-registered sensitivity:
- V ≤ 10 / N ≥ 30 km gives −0.002 (−0.047 to +0.046), NOT SPECIFIC.
- V ≤ 15 / N ≥ 30 km gives +0.016 (−0.016 to +0.051), INCONCLUSIVE.

*(Corrected after the audit: the first version said the difference "vanishes when the volcanic zone is drawn
tighter". These runs also loosen the far zone, from 40 to 30 km. Tightening the volcanic zone alone leaves +0.029.
See §Audit.)*
- RandomForest: INCONCLUSIVE (primary), NOT SPECIFIC under both alternatives.
- MaxEnt: INCONCLUSIVE under all three definitions.

**M3: design disagreement.** In the same matched cells, the share of cells in the contingent fringe (top decile
under exactly one design) differs by +0.56 pp (+0.26 to +0.85). |hybrid − random| differs by +0.0074 (+0.0033 to
+0.0110). Both get the pre-registered verdict NOT SPECIFIC, for all three algorithms and all three definitions
(largest |Δ| 0.56 pp against a 5 pp threshold). **But the test could not have come out SPECIFIC.** Below 1,000 m the
fringe covers only 2.5 % of cells near volcanoes and 1.4 % far from them, so a 5-point threshold exceeds the rate
itself. Mean |hybrid − random| is 0.056 near and 0.039 far, against δ = 0.05. In relative terms the fringe is about
1.5–1.8 times as frequent near volcanoes, and the CI excludes zero. **No statement that the disagreement "does not
depend on volcanic setting" can be made.** *(Corrected after the audit; it was a design flaw: the thresholds were set
without looking at the base rates.)*

**M4 (descriptive).** Within the usable bands, an elevation spline alone explains about 6 % (5.7–6.9 %) of XGBoost suitability
variance. Volcanic setting adds a partial R² of 0.032–0.038 for suitability and 0.015 for |hybrid − random|.

**M5 (descriptive only; M2 is not SPECIFIC overall).** Matched V cells face south more than N cells (mean cos(aspect)
−0.11, CI −0.19 to −0.03). River distance does not differ (−0.35 km, CI −1.84 to +0.81).

**Figure:** `results/e235_response_by_band.png`.
- A: land per band.
- B: unmatched band means of suitability for the three designs.
- C: contingent-fringe share.

**What the figure shows outside the overlap.** Above 1,000 m, where nearly all land is within 20 km of a Quaternary
centre, the designs part company with height:
- the hybrid design stays at about 0.5–0.6 suitability;
- random and TGB fall from about 0.5 (1,000–1,200 m) to about 0.1 above 1,600 m;
- the contingent fringe rises from 10 % (1,000–1,200 m) to about 25 % above 1,600 m.

*(Corrected after the audit: "about 0.1" and "about 25 %" hold only above about 1,600 m.)*

This is the "remote upland" fringe of manuscript Figure 5. In this study area it cannot be attributed to elevation
rather than volcanic setting, or the other way round.

**Verification (G1):** an independent pandas re-derivation (`groupby` matching; different code path, kept in the
2026-10-20 session scratchpad) reproduced every primary number exactly:
- 3,435 km², 97.5 %, 0.4 %;
- 53,927 matched cells;
- Δ̄ random +0.0491; fringe +0.563 pp.

## Audit (2026-10-20, same day; the checks below are POST HOC, not pre-registered)

An independent adversarial audit (`AUDIT_20261020.md`) found **no implementation error**.
- The code matches DESIGN on every rule.
- Its own code reproduced every primary number; its bootstrap CIs differ in the third decimal.
- UTM against geodesic distance changes Δ̄ by 0.0002.

It found three overreaching sentences, a design flaw, and a fragile verdict. The orchestrator re-ran the main checks
(`02_posthoc_audit_checks.py`; XGBoost, random design unless stated):

| Post hoc variant | Δ̄ (V − N) | Reading by the DESIGN rule |
|---|---|---|
| primary (≤ 20 / ≥ 40 km), as run | +0.049 | INCONCLUSIVE |
| volcanic zone tighter only (≤ 10 / ≥ 40) | +0.029 | — |
| volcanic zone ≤ 15 / ≥ 40 | +0.041 | — |
| far zone stricter (≤ 20 / ≥ 50) | +0.056 / +0.056 / +0.067 (random / TGB / hybrid), CIs exclude 0 | SPECIFIC in all three designs |
| round-1 set of 13 in-bounds centres (≤ 20 / ≥ 40) | +0.051 / +0.050 / +0.062, CIs exclude 0 | SPECIFIC in all three designs |
| river distance added to the matching | +0.028 | — |
| river distance added, 5 bins per variable | +0.010 | — |

- **Fragile suitability verdict.** The +0.05 sits on the threshold. Depending on choices the pre-registration did not
  fix, it reads as SPECIFIC or shrinks to +0.01. **Matching on river distance as well** (a model feature that was not
  matched) cuts it from +0.049 to +0.028, and to +0.010 with finer bins. The pre-registered verdict stays INCONCLUSIVE, and no direction is claimed.
- **The far-zone uplands are one region.** 95 % of far-zone cells at 400–1,000 m lie south of 7.9°S, and 87 % west of
  112.2°E: the western Southern Mountains, an Eocene–Miocene volcanic arc with karst (sources in the Status block).
  The audit also reports that dropping the north-coast far-zone cells halves Δ̄, while dropping the Southern Mountains
  raises it. This is the audit's figure; the orchestrator did not re-run it.
- **Frame:** the 111–115°E frame also takes in small parts of Central Java and the western tip of Bali. This is the
  audit's figure; the orchestrator did not check it.

## Conclusion (revised after the audit)

The reviewer's question has one firm answer, and it is about the study area, not the model:
- **Above about 1,000 m, elevation and nearness to a Quaternary volcanic centre cannot be separated in this study
  area.** 97.5 % of that land is within 20 km of a centre and 0.4 % is 40 km or more away. Nothing the model does up
  there can be called volcanic rather than elevational, or the reverse. The paper says so and claims neither.
- **Below 1,000 m, nothing firm either way.**
  - The model's suitability is somewhat higher near Quaternary centres than in terrain-matched ground far from them
    (+0.01 to +0.07, depending on choices of zone and matching; pre-registered verdict INCONCLUSIVE).
  - Matching on river distance too cuts it to +0.028 (+0.010 with finer bins).
  - The far-zone comparison ground is mostly old (Tertiary) volcanic and karst terrain in one region.
  - The test of design disagreement was too weak to say anything.

**What this changes in P2** (DESIGN §6, unchanged by the audit; the PI writes the prose):
- "directionally consistent with a volcanic-specific effect" is removed;
- the n = 2 site-density comparison stops being evidence;
- the overlap ceiling (about 1,000 m) and the 97.5 % / 0.4 % split are stated;
- the suitability difference is reported as inconclusive, with its range;
- **no statement about the design disagreement and volcanic setting** (the M3 test had no power);
- wording: "far from Quaternary volcanic centres", not "non-volcanic";
- Figure 1 gets topography.

`results/e235_response_by_band.png` is a candidate manuscript figure; it shows the reviewer exactly where the two can
and cannot be separated.

**Outputs:** `01_elevation_vs_volcanic.py`, `results/e235_outcome.json`, `results/e235_bands.csv`,
`results/e235_response_by_band.png`.

# ADV-3: Survey Intensity Sufficiency Test

**Status:** INFO NEG — verdict inverted; corrected 2026-10-01 (originally reported as SUCCESS / VOLCARCH SUPPORTED)
**Date:** 2026-03-13 (correction 2026-10-01)

> ## ⚠ CORRECTION 2026-10-01 — the verdict below is INVERTED
>
> `volcano_dist` is a **distance** (km to the nearest volcano, z-scored). In a Poisson model a
> **negative** coefficient means expected site counts **fall** as distance grows, i.e. **more
> recorded sites closer to volcanoes**. This README (the "Fewer sites near volcanoes" table row and
> the interpretation paragraph) and the script's verdict logic (`volc_coef < 0 → "VOLCARCH
> SUPPORTED"`) read it the other way round. The same table reads the survey rows correctly
> ("road_dist −7.150 → fewer sites far from roads"), which is how the inversion was spotted (WF
> evidence reader, 2026-10-01).
>
> **Independent check** (`direction_check_20261001.py` → `results/canonical30/direction_check_20261001.txt`):
> the refit reproduces β exactly (−0.476 on the 7-volcano inventory, −0.831 on the canonical 30).
> Partial effect with the survey proxies held at their means: more expected sites per cell at the
> 10th-percentile distance than at the 90th, on both inventories. Raw: **3.18 sites/cell within 10 km**
> of a volcano vs 0.10 at 50–100 km and 0 beyond 100 km (375 in-grid sites, 703 cells). The canonical
> script, re-run with corrected verdict logic, reproduces every number and now reports
> `SURPLUS NEAR VOLCANOES (not a burial deficit)`.
>
> **What the result actually says:** after controlling for three survey-intensity proxies, recorded
> sites (≈71% temples, E129) are **more** numerous near volcanoes. That is **not** evidence that a
> burial deficit survives survey control. At the 11-km cell scale, on an all-period inventory, the
> data show no deficit at all. The surplus is consistent with temple siting on volcanic slopes (P11's
> sacred landscape), fertile soils, and survey focus on monuments; it cannot separate burial from
> survey effort for **settlements** (5 of 391 known East Java sites, E129).
>
> **Design flaw behind it:** the pre-registered success criterion in `../README.md` ("residual
> volcanic signal after controlling for survey effort") was **direction-agnostic**, so a surplus near
> volcanoes also counted as support. The burial hypothesis predicts a *deficit* (a positive
> coefficient on distance); that is the criterion it should have been held to.
>
> **Propagation:** ledger **C023** (`docs/CRITIQUE_LEDGER.md`) lists every document that carried the
> inverted reading, including P11 v0.7 (corrected before submission) and P1 JASREP v4.0.
**Type:** ADVERSARIAL / FALSIFICATION
**Papers:** P1, P2 (thesis-level validation)

## Hypothesis

**H0:** Site distribution is FULLY explained by survey intensity proxies. No residual volcanic signal exists.

**H1 (VOLCARCH):** After controlling for survey intensity, volcanic proximity still explains additional variance in site distribution.

## Method

1. Grid East Java into ~11km cells (0.1 degree, 722 cells)
2. Count archaeological sites per cell (E001 database, 666 sites)
3. Compute survey intensity proxies per cell:
   - Road distance (from E013 raster `jatim_road_dist_expanded.tif`)
   - Min distance to BPCB heritage offices (Trowulan, Yogyakarta, Prambanan)
   - Min distance to archaeology departments (UGM, UI, Unibraw, Unair, Udayana)
4. Compute min volcanic proximity per cell (7 East Java volcanoes)
5. Nested Poisson regression with quasi-Poisson correction:
   - Model 1 (survey only): `site_count ~ road_dist + bpcb_dist + uni_dist`
   - Model 2 (survey + volcanic): `site_count ~ road_dist + bpcb_dist + uni_dist + volcano_dist`
6. Likelihood ratio test (chi-squared)

## Results

### Grid Statistics
- 703/722 valid cells (on land, with road distance data)
- 110 cells have >= 1 site
- Max 49 sites in one cell
- 375/666 sites within grid bounds

### Model 1: Survey Only
| Predictor | Beta (std) | Direction |
|-----------|-----------|-----------|
| road_dist | -7.150 | Fewer sites far from roads |
| bpcb_dist | -1.331 | Fewer sites far from BPCB |
| uni_dist  | -0.526 | Fewer sites far from universities |

- Log-likelihood: -682.63
- AIC: 1373.25
- Pseudo-R2: 0.382

### Model 2: Survey + Volcanic
| Predictor | Beta (std) | Direction |
|-----------|-----------|-----------|
| road_dist | -7.158 | Fewer sites far from roads |
| bpcb_dist | -1.231 | Fewer sites far from BPCB |
| uni_dist  | -0.258 | Fewer sites far from universities |
| **volcano_dist** | **-0.477** | **Fewer sites near volcanoes** |

- Log-likelihood: -664.81
- AIC: 1339.61
- Pseudo-R2: 0.398

### Likelihood Ratio Test
- LR statistic: 35.64
- df: 1
- Raw p-value: 2.4e-9

### Overdispersion Correction
- Dispersion ratio: 14.16 (severely overdispersed)
- Quasi-Poisson phi: 3.55
- **Adjusted LR statistic: 10.03**
- **Adjusted p-value: 0.0015**

## Interpretation

**VERDICT: VOLCARCH SUPPORTED**

Adding volcanic proximity **significantly improves** the model (adjusted p=0.0015), even after:
1. Controlling for road accessibility (strongest predictor)
2. Controlling for BPCB office proximity
3. Controlling for university proximity
4. Correcting for severe overdispersion (quasi-Poisson)

The volcanic coefficient is **negative** (beta=-0.477), confirming: fewer archaeological sites are found closer to volcanoes, even after accounting for survey intensity.

## Caveats

1. **Overdispersion is severe** (14.16x) — zero-inflated model would be ideal but result survives quasi-Poisson correction
2. **Delta pseudo-R2 is modest** (0.016) — volcanic effect is real but small relative to survey intensity
3. **375/666 sites** fell within grid bounds (rest outside East Java core area)
4. **Grid resolution** (11km) may miss local patterns
5. **Survey proxies are crude** — road distance is best available, not actual survey records
6. **Endogeneity risk** — sites near universities might be discovered because universities are near old civilizations, not the reverse

## Conclusion

This adversarial experiment **fails to falsify** the VOLCARCH thesis. The volcanic signal survives multi-proxy survey intensity control. Survey intensity (especially road access) is the dominant predictor, but volcanic proximity adds a statistically significant independent contribution.

**For revision defense:** "We conducted an adversarial regression controlling for three survey intensity proxies (road accessibility, heritage office proximity, university proximity). Volcanic proximity remained a significant negative predictor (quasi-Poisson LR p=0.0015), confirming that the volcanic site deficit is not solely attributable to differential survey effort."

## Files
- `adv3_survey_intensity.py` — Main analysis script
- `results/adv3_results.json` — Full results in JSON
- `results/adv3_cell_data.csv` — Per-cell data for visualization

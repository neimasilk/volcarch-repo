# E213 — Aggradation–Exposure Geomorphic Asymmetry

**Status:** INCONCLUSIVE (first operationalization) — spine for P7 NOT yet established
**Date:** 2026-06-08
**Trigger:** P7 rejection (Antiquity AQY-2026-0104). After the "distance from volcano" variable was shown to be an artifact (ME#17), test the geomorphologically correct variable: deep archaeology is visible only where erosion/karst EXPOSES it.
**Layer:** L1 (taphonomy), detection-horizon framing

## Hypothesis (falsifiable)
H1: Settlement suitability favors low-relief plains while archaeological visibility favors relief/incision/karst → the two anti-correlate, so the land people would settle is exactly where the deep record is buried. (If true, surface absence on the plains is uninformative — a *detection-horizon* statement, not a claim that buried sites exist. This would answer Antiquity Reviewer-2 critique C without the equifinality trap.)

## Method
- Copernicus 30 m DEM derivatives for East Java (UTM 49S): slope, TRI, river-distance.
- Sampled slope/TRI/river-dist at the 4 known deep-time sites + 21,811 settlement-suitability grid cells.
- Tested Spearman(suitability, slope) and compared mean slope of high- vs low-suitability cells; computed the low-relief ("buryable") fraction of high-suitability land.

## Result — H1 NOT SUPPORTED (honest negative)
| Test | Expected (H1) | Observed | Verdict |
|---|---|---|---|
| Spearman(suitability, slope) | strongly negative | **−0.039** (p=8e-9) | ~zero; **fails** |
| mean slope, high vs low suitability | high < low | **7.34° vs 7.39°** | no difference; **fails** |
| % high-suit cells that are flat plains (<2°) | majority | 40.0% | weak/minority |
| Deep-time site terrain | all relief/incision | Trinil ✓ (6.5°, 130 m from Solo); Wajak ✓ (karst hill); **Song Terus flat (0.6°)**; Sangiran off-DEM | mixed |

## Why it failed (diagnosis)
1. **Slope is the wrong proxy in volcanic terrain.** Volcano flanks are high-relief AND heavily buried (lahars/tephra). Only *non-volcanic* relief (Kendeng/S.-Mountains uplift, karst, deep river incision) exposes deep strata. Slope cannot distinguish "buried volcanic slope" from "exposed uplift/karst."
2. **Suitability ≠ flat.** Java's XGBoost suitability rates fertile volcanic slopes as suitable, so the assumed "people settle on flat plains" premise is false here.
3. **Partial circularity.** The suitability model uses slope as an input feature, so suitability-vs-slope is not independent.

## What a valid test needs (next step)
A **geology / volcanic-cover layer** (volcanic vs limestone-karst vs alluvium vs uplifted Tertiary). Then classify exposure = {karst ∪ Tertiary uplift ∪ incised terrace} vs burial = {volcanic cover ∪ aggradational alluvium}, and test whether deep-time sites fall in exposure regimes and whether suitable land is dominated by burial regimes. Source candidates: Pusat Survei Geologi 1:100k/250k geology, or a global lithology raster (GLiM). Coordinates of the expanded Java Pleistocene site set (Ngandong, Sambungmacan, Perning, Kedung Brubus, Kali Baksoka, Patiayam…) must be VERIFIED before use (the very error E213 exists to correct).

## Implication for P7
The non-circular spatial spine for a P7 overhaul is **not established by available data**. Rewriting P7 now would repeat the original sin (writing before the evidence holds). Decision deferred to Pak Amien: (a) acquire geology layer + redo properly, or (b) shelve P7, redirect to the palynology channel (independent + falsifiable). See ME#17 §8–§9.

## Output
- `results/deep_time_terrain_signature.csv`, `results/e213_summary.json`
- `analyze.py`

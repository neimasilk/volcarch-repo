# E109: Forward Simulation — Archaeological Record Under Burial Hypothesis

**Date:** 2026-03-17
**Paper:** P1, P2, L1
**Status:** MIXED — Reveals survey-burial confound; estimates 824 hidden sites

## Hypothesis

The observed distribution of archaeological sites in East Java can be modeled as:
sites_observed(cell) ~ N_total × P_suitability(cell) × P_detection(depth, survey)

If burial reduces detection: deeper zones should have fewer sites (after controlling survey access).

## Method

1. Merged E075 burial grid (2,838 cells) with E069 survey data (703 cells) → 592 overlapping cells
2. Analyzed site density by burial depth quartile
3. Fitted MLE detection model: P(detect) = exp(-depth/τ) × 1/(1 + road_dist/ρ)
4. Estimated total hidden sites
5. Simulated Japan-level survey scenario

## Results

### SURPRISE: Site Density INCREASES with Burial Depth

| Quartile | Cells | Mean Depth (cm) | Sites | Density | Mean Road Dist |
|----------|:---:|:---:|:---:|:---:|:---:|
| Q1 (shallow) | 148 | 7.6 | 13 | 0.088 | 9,826 m |
| Q2 | 148 | 35.5 | 36 | 0.243 | 11,209 m |
| Q3 | 148 | 109.7 | 49 | 0.331 | 11,259 m |
| **Q4 (deep)** | 148 | 459.9 | **259** | **1.750** | **1,578 m** |

Chi-square p < 0.000001: sites are NOT uniformly distributed.
Trend: density INCREASES with depth (rho = 1.0).

### Explanation: Spatial Confound

The deep-burial zones are ALSO the zones closest to volcanoes, which are:
- Most fertile (volcanic soil)
- Most populated (modern and historical)
- Best road access (shortest road distance: 1,578 m vs 9,826 m)
- Most intensively surveyed

**The survey intensity effect OVERWHELMS the burial effect in raw data.**

### MLE Detection Model

| Parameter | Value | Interpretation |
|-----------|:---:|---|
| λ₀ (base rate) | 1.99 sites/cell | High suitability overall |
| τ (depth scale) | ∞ cm | Burial depth NOT a predictor in this model |
| ρ (road scale) | 181 m | Road access is the dominant predictor |

**The model converges on τ=∞ because burial depth is confounded with road access.** E069 (ADV-3) resolved this confound using nested model comparison and found p=0.0015 for volcanic proximity AFTER controlling survey. E109 confirms the confound exists but cannot separate the effects without the nested approach.

### Hidden Site Estimation

- Observed: 357 sites
- Estimated total: 1,181 sites
- **Estimated hidden: 824 sites (detection rate 30.2%)**
- 22.5% of cells require subsurface methods (>200 cm burial)

### Japan-Level Survey Scenario

- Current survey: ~357 sites detected
- Japan-level survey (50× better access): ~846 sites detectable (2.4×)
- Even Japan-level survey can't find everything at >5m depth

## Conclusion

E109 reveals the fundamental methodological challenge: in East Java, volcanic zones are simultaneously the most buried AND the most surveyed. The raw data shows an inverted pattern (more sites near volcanoes) because survey intensity dominates.

**Key insight:** The 824 estimated hidden sites (detection rate 30.2%) are hidden by SURVEY ACCESS, not burial depth, in this model. E069's nested approach is needed to isolate the burial effect.

**Reinforces E086 (Japan):** Survey deficit is the primary constraint. Burial is secondary but contributes to the ~22.5% of cells that require subsurface methods regardless of survey intensity.

## Caveats

1. Grid merge produced only 592 cells (different extents between E075 and E069)
2. MLE model is simple (2 parameters) — a joint model with more features would improve separation
3. τ=∞ doesn't mean burial doesn't matter — it means the model can't disentangle it from road access
4. The 824 hidden sites estimate is driven by road access variation, not burial depth

## Files

| File | Description |
|---|---|
| `forward_simulation.py` | Main simulation script |
| `results/e109_results.json` | Full results with MLE parameters |

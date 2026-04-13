# E191: Multi-temporal NDWI — Dry vs Wet Season Contrast

**Date:** 2026-04-13
**Status:** SUCCESS (INFORMATIVE — borderline signal, new metric discovered)
**Paper:** P23 (future), P1/P17 revision ammo
**Layer:** L1

---

## Hypothesis

The NDWI (water index) difference between candi sites and controls is LARGER in the wet season than the dry season. Rationale: buried stone structures impede water infiltration differently than surrounding andosol. When the water table rises (wet season), the moisture anomaly above buried structures is amplified — creating a stronger spectral contrast.

E189 showed NDWI is significant in dry season (p=0.032). If wet-season NDWI is even stronger, this validates the moisture-based detection mechanism and may push the borderline metrics to full significance.

## Method

1. Extract NDWI at same 15 candi + 5 controls for two seasons:
   - **Dry:** July-September 2024 (E189 baseline)
   - **Wet:** December 2023 - February 2024
2. Compute NDWI delta (wet minus dry) at each site
3. Test: is the seasonal NDWI change different at candi vs controls?

## Results

**20/20 sites analyzed for both seasons.**

### Season Comparison

| Metric | Candi (n=15) | Control (n=5) | p-value |
|--------|:---:|:---:|:---:|
| Wet NDWI \|diff\| vs ctrl | larger | — | 0.071 |
| Candi wet>dry \|NDWI diff\| | — | — | 0.084 |
| **Delta local variance** | **+0.00021** | **-0.00027** | **0.066** |
| Delta diff candi vs ctrl | — | — | 0.933 (NS) |

### Key Finding: Delta Local Variance

The most discriminating multi-temporal metric: **candi sites show INCREASED NDWI heterogeneity from dry→wet season (+0.00021), while controls show DECREASED heterogeneity (-0.00027).**

Physical interpretation: when the water table rises (wet season), moisture pools differently above buried stone structures vs surrounding andosol → candi become MORE spectrally heterogeneous. Natural sites become MORE homogeneous as the landscape saturates.

### Individual Notable Sites

- **Candi Songgoriti:** delta=-0.178 (largest seasonal shift) — thermal spring + volcanic substrate
- **Candi Tegowangi:** delta=+0.102 (largest positive shift) — NDWI switches direction wet vs dry
- **Candi Kidal:** delta=-0.076 (strong negative) — volcanic slope, drainage amplifies

### Comparison with E189/E190

| Experiment | Best metric | p-value | Verdict |
|:---:|------|:---:|---------|
| E189 Optical | NDWI dry | **0.032** | **Best single metric** |
| E190 SAR | — | 0.72 | Ruled out (C-band) |
| E191 Multi-temporal | Delta lvar | 0.066 | New metric, borderline |

## Conclusion

Multi-temporal analysis does NOT push the signal to clear significance, but discovers a new metric (delta local variance) that captures the differential moisture response of buried structures across seasons. With more control sites (n>15), this could become significant.

**The dry-season optical NDWI from E189 remains the strongest single metric.**

## Scripts

- `multitemporal.py` — Seasonal comparison analysis

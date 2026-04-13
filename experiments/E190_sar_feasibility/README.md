# E190: SAR Feasibility — Can Sentinel-1 Radar See Buried Candi?

**Date:** 2026-04-13
**Status:** SUCCESS (INFORMATIVE NEGATIVE)
**Paper:** P23 (future), P1/P17 revision ammo
**Layer:** L1

---

## Hypothesis

Sentinel-1 C-band SAR backscatter (VV/VH polarization) shows stronger archaeological signal at known candi sites than Sentinel-2 optical. Rationale: E189 showed NDWI (moisture proxy) is the strongest optical signal (p=0.032). SAR directly measures soil moisture and penetrates vegetation canopy — it should amplify the moisture-based detection mechanism.

Specific predictions:
- **H1:** VH/VV ratio (cross-pol ratio) differs between candi and control sites
- **H2:** SAR local variance is higher at candi (buried structures create backscatter heterogeneity)
- **H3:** SAR center-ring difference is larger at candi than control

## Method

1. **Data:** Sentinel-1 GRD IW mode (VV + VH polarization, ~10m resolution) via Planetary Computer
2. **Season:** Dry season (Jul-Sep 2024) for comparison with E189
3. **Sites:** Same 15 candi + 5 controls as E189 core
4. **Analysis:** Center vs ring backscatter difference, local variance, VH/VV ratio, Mann-Whitney U tests

## Relation to Other Experiments

- Follows: E189 (optical feasibility — NDWI p=0.032)
- Builds on: E076 (Planetary Computer pipeline), E080/E097 (target zones)
- Tests: whether SAR confirms/strengthens E189's moisture-based signal

## Results

**STATUS: INFORMATIVE NEGATIVE — C-band SAR cannot detect buried candi in tropical Java**

### Backscatter Comparison

| Metric | Candi (n=15) | Control (n=5) | p-value | Direction |
|--------|:---:|:---:|:---:|:---:|
| VV center-ring diff | +0.308 dB | -0.400 dB | 0.867 | NS |
| VH center-ring diff | +0.113 dB | -0.141 dB | 0.751 | NS |
| Cross-pol ratio diff | -0.026 | +0.028 | 0.723 | NS |
| VV local variance | 0.909 | **1.030** | 0.967 | **Control higher!** |
| VH local variance | 0.839 | **0.868** | 0.916 | **Control higher!** |

### Key Findings

1. **C-band SAR reflects off canopy, not ground.** Tropical vegetation dominates the 5.6 cm wavelength return. Controls (more diverse land cover) actually have HIGHER SAR variability.
2. **Exposed stone structures DO produce strong returns:** Candi Singosari (+1.56 dB), Sumberawan (+1.23 dB) — but this is surface scattering, not subsurface detection.
3. **Cohen's d = -0.92 for VV local variance** — large, wrong-direction effect. SAR heterogeneity is a proxy for land-cover diversity, not archaeological complexity.
4. **Comparison with E189:** Optical NDWI (p=0.032) is MUCH stronger than any SAR metric. The moisture signal in E189 comes from vegetation response to subsurface structures, not from direct subsurface sensing.

### Implication

- **C-band SAR is ruled out** for buried-candi prospection in tropical Java.
- **L-band SAR (ALOS PALSAR, 24 cm wavelength)** could penetrate deeper — this is the next sensor to test.
- **Multi-temporal optical (E191)** is a more promising near-term path: amplify E189's marginal NDWI signal.

## Conclusion

**SUCCESS (informative negative).** C-band SAR is NOT useful for buried archaeological prospection in tropical andosol. The signal is dominated by canopy structure, not subsurface features. This rules out the most obvious SAR approach and redirects effort toward L-band SAR or multi-temporal optical analysis.

## Scripts

- `sar_feasibility.py` — SAR analysis with GCP-based georeferencing

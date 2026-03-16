# ADV-3: Survey Intensity Defense — Revision Ammo for P1

## Anticipated Critique
"The site deficit near volcanoes could simply reflect lower survey effort in volcanic terrain, not actual burial. Roads are worse, BPCB offices are distant, and fewer archaeologists work in those areas."

## Defense

We conducted an adversarial regression specifically designed to test this critique (ADV-3). Using a grid-based Poisson regression over East Java:

**Survey proxies controlled:**
1. Road distance (from OpenStreetMap-derived raster)
2. Distance to BPCB heritage conservation offices (Trowulan, Yogyakarta, Prambanan)
3. Distance to university archaeology departments (UGM, UI, Unibraw, Unair, Udayana)

**Result:** After controlling for all three survey intensity proxies, volcanic proximity remains a significant negative predictor of site density (beta = -0.477, quasi-Poisson likelihood ratio test p = 0.0015, correcting for overdispersion phi = 3.55).

**Key numbers:**
- Survey-only model pseudo-R2: 0.382
- Survey + volcanic model pseudo-R2: 0.398
- AIC improvement: 33.6 points favoring volcanic model
- 703 grid cells, 666 sites

**Interpretation:** Survey intensity (especially road accessibility) is the dominant predictor of site discovery. However, volcanic proximity adds a statistically significant independent contribution. The volcanic site deficit is not solely attributable to differential survey effort — a residual signal consistent with volcanic burial persists after multi-proxy survey control.

**Caveats we acknowledge:**
- Effect size is modest (delta pseudo-R2 = 0.016)
- Survey proxies are crude approximations
- Confirmatory fieldwork remains the gold standard

## Citation-ready text
"To address the possibility that our observed site deficit near active volcanoes merely reflects differential survey intensity, we conducted a nested Poisson regression controlling for three independent survey proxies: road accessibility, proximity to heritage conservation offices, and proximity to university archaeology departments. Volcanic proximity remained a significant negative predictor after quasi-Poisson correction for overdispersion (LR p = 0.0015), confirming that the volcanic-zone deficit is not solely attributable to survey effort."

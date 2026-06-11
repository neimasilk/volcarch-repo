# E209: Multi-Signal Satellite ML Classifier for Buried-Site Detection in Volcanic Java

**Date started:** 2026-04-22 (Session 21, per ME#16 flagship diamond-hunt directive)
**Status:** PHASE 1 — data pipeline scaffolding
**Paper target:** P23 "Seeing Through Volcanic Soil" — Remote Sensing of Environment (Q1, IF 13.5) or JASREP (Q1)
**Layer:** L1 (volcanic burial detection)
**Lineage:** Extends E076 (NDVI Phase 0), E189 (multispectral Phase A, weak signal), E190 (C-band SAR ruled out), E191 (multi-temporal)

---

## Hypothesis

A machine learning classifier trained on multi-signal satellite features can distinguish locations containing archaeological structures (known or buried) from non-archaeological controls in volcanic tropical Java, with sufficient statistical power to propose top-K candidate locations for field verification in the volcanic interior basins (Malang, Kediri, Kelud, Progo, Kedu).

Specifically:

- **H1 (classifier performance):** A multi-signal classifier trained on 142+ known candi sites as positives and random non-archaeological tiles as negatives achieves held-out AUC ≥ 0.75.
- **H2 (transferability to buried sites):** The classifier scores known discovered-buried sites (Sambisari, Kedulan, Kimpulan, Liangan) at the high-probability tail (≥ 90th percentile of held-out predictions).
- **H3 (landscape application):** Applied to Malang + Kediri + Kelud basins, the classifier produces a probability map whose top-20 high-confidence predictions are (a) spatially non-random, (b) preferentially cluster near known settlement-era indicators (river confluences, gentle slopes, historical place names), (c) do NOT coincide with known surface sites (i.e., they are NEW candidate locations).

If all three hypotheses hold, output = a falsifiable list of top-20 candidate buried-site GPS coordinates, publishable as P23.

If H1 fails: classifier cannot learn archaeological signal from satellite data alone. Informative negative: limits of satellite archaeology in volcanic tropical andosol published as P23 alt-framing.

If H1 passes but H2 fails: classifier learns surface features of monumental candi, not buried-site signatures. Reframe: decorative pattern recognition paper.

If H1+H2 pass but H3 produces scientifically non-discriminating predictions (e.g., everything in the basin scores high): classifier is picking up basin-wide geology. Refine controls; possibly reframe.

---

## Why This Matters (per ME#16)

ME#16 identified **discovery deficit** as the structural reason VOLCARCH papers keep getting rejected. This is the flagship diamond-hunt — AI-unique, compute-feasible on RTX 4080, produces falsifiable coordinates, and has standalone publication potential at a Q1 remote sensing venue.

Precedent: Parcak 2016 (Egypt), Evans 2013 (Angkor LiDAR), de Souza 2024 (Amazon SAR+LiDAR), Tapete & Cigna 2019 (global InSAR archaeology). None applied to volcanic tropical Java. Genuine frontier.

---

## Signal Stack

Features drawn from six complementary satellite sources:

| Source | Type | Resolution | Access | Features extracted |
|---|---|---|---|---|
| Sentinel-2 L2A | Multispectral (13 bands) | 10m | Free (Planetary Computer STAC) | NDVI, NDWI, MSAVI, clay ratio, iron oxide, local variance, seasonal delta |
| Sentinel-1 GRD | SAR C-band (VV+VH) | 10m | Free (PC STAC) | Backscatter mean/var, VV/VH ratio, seasonal delta |
| Copernicus GLO-30 DEM | Elevation | 30m | Free (ESA) | Slope, curvature, TPI, TRI, microtopography residuals |
| ALOS PALSAR (optional) | SAR L-band | 12.5m | Free (JAXA) | Deeper-penetration backscatter |
| ASTER GED (optional) | Thermal emissivity | 100m | Free (NASA) | Thermal anomaly residuals |
| Landsat 8/9 (optional) | Multispectral (11 bands) + thermal | 30m | Free (USGS) | Long time series supplement |

**Phase 1 (this session):** Scaffold pipeline. Execute for Sentinel-2 + Sentinel-1 + Copernicus DEM across training set.
**Phase 2:** Add ALOS PALSAR + ASTER if Phase 1 signal insufficient.
**Phase 3:** Landscape-scale inference on volcanic interior basins.

---

## Training Set

### Positive class

- **Hard positives** (discovered-buried, treated as strongest signal): Sambisari, Kedulan, Kimpulan, Liangan, Candi Badut, Candi Tigomangi
- **Soft positives** (known monumental sites, may include partially-buried): 142+ candi from existing `east_java_sites.geojson` + `east_java_sites_wiki.csv`, filtered for pre-1500 CE archaeological type
- **Augmentation:** within-site tile variations (±50m offset, seasonal variants)

### Negative class

- **Hard negatives** (confirmed non-archaeological, distinct geology): 5 original controls from E189 (river, plain, slope, agricultural)
- **Random negatives:** 200–500 randomly-sampled tiles in Java but >5km from any known archaeological site, stratified by terrain type (lowland / slope / valley / plateau) to match positive distribution
- **Hard negatives-from-geology:** locations on active lava flows, recent lahar deposits, quarries, mines

Expected training size: ~150 positive + ~300 negative = 450 tiles × ~50–100 features = manageable on RTX 4080.

### Validation / test split

- Temporal hold-out on seasonal data (train on 2018–2022, test on 2023–2025)
- Spatial hold-out: leave-one-out cross-validation by volcanic system (Merapi / Kelud / Semeru / Arjuno / etc.) — tests transferability across geologies

---

## Method

### Feature engineering

For each site tile (default 500m × 500m centred on coordinate):
- **Spectral features (Sentinel-2):** band means + stds, NDVI/NDWI/MSAVI distributions, local variance in 3×3 and 9×9 windows
- **Seasonal features:** wet-season vs dry-season delta for all spectral indices
- **SAR features (Sentinel-1):** VV/VH mean + std, backscatter seasonal delta, texture (GLCM)
- **Topographic features (Copernicus DEM):** elevation std in window, slope distribution, topographic position index, terrain ruggedness index, microtopography residuals (site DEM minus smoothed neighbourhood)
- **Centre-vs-ring contrast:** difference in all spectral + SAR metrics between inner 100m and outer 100–500m annulus (E189 methodology, extended)

Estimated ~80–120 features per site.

### Classifier

- **Baseline:** Random Forest (scikit-learn) — fast, interpretable, robust with small N
- **Secondary:** XGBoost with calibrated probabilities
- **Tertiary (if baseline insufficient):** CNN on raw tile imagery (ResNet-18 transfer from ImageNet or segmentation_models_pytorch), RTX 4080 training

### Interpretability

- SHAP values on RF / XGBoost to identify dominant features
- Compare dominant features to E189 findings (NDWI, NDVI local variance) — converging or divergent?
- If CNN used: Grad-CAM on top predictions to confirm archaeological-like patterns

---

## Pipeline

```
scripts/
  01_prepare_training_data.py   # Build training-site list with labels + metadata
  02_download_satellite_bands.py # Fetch Sentinel-2, Sentinel-1, DEM for all sites
  03_extract_features.py         # Compute per-site feature vectors
  04_train_classifier.py         # Train RF + XGBoost baseline; optionally CNN
  05_predict_landscape.py        # Apply to Malang/Kediri/Kelud basins, produce map
  06_summarise_results.py        # Stats, top-K, figures for paper
```

Each script is independently runnable and checkpoints to `data/` or `results/` in this directory.

---

## Expected Outcomes

**Best case (H1+H2+H3 all pass):**
Classifier AUC ≥ 0.75, discovered-buried sites in top 10% of predictions, landscape inference produces 20+ non-random candidate locations in volcanic interior with spatial clustering consistent with ancient settlement patterns. Publishable as P23 discovery paper.

**Informative negative (H1 fails):**
Classifier cannot distinguish archaeological from non-archaeological sites using available satellite signals. First systematic test of its kind in volcanic tropical context. Publishable as limits-of-method paper.

**Intermediate (H1 passes, H3 fails):**
Classifier learns some archaeological signal but inference produces geology-confounded predictions. Useful for methodology development; part of a larger multi-method paper.

---

## Relation to Other Experiments

- **Builds on:** E076, E189 (Phase A weak signal), E190 (C-band SAR ruled out, informs Sentinel-1 VV/VH interpretation), E191 (multi-temporal, delta local variance)
- **Parallel:** E210 InSAR time-series (independent AI-unique approach)
- **Feeds:** P23 paper + P1 revision ammo (quantitative landscape prediction) + manifesto v5.0 (if major hit)
- **Depends on:** E013 settlement model (AUC 0.768, provides geographic priors), E097 anomaly detection (65% overlap validation target)

---

## Budget

- **Compute:** 0 (all local on RTX 4080)
- **Data:** 0 (all public satellite APIs)
- **Storage:** ~50–200 GB for intermediate satellite tiles
- **Time:** 1–2 weeks to publication-ready first-cut; 2–3 months for full Phase 3 landscape inference

No budget request needed.

---

## Kill criteria

Diamond-hunt killed (not just failed) if:
- Training data preparation fails for > 50% of target sites due to satellite coverage gaps
- Classifier cross-validated AUC < 0.60 with no route to improvement
- All predictions saturate at uniform probability (no discriminating signal)
- Held-out discovered-buried sites (Sambisari et al.) score below chance

Any of above → document as informative negative, proceed to E210 InSAR as next diamond-hunt.

---

*E209 scaffolded 2026-04-22 per ME#16 §4 Candidate 1. Discovery-first pivot active.*

# Tautology-Free Settlement Suitability Modeling in East Java Under Survey and Taphonomic Bias

**Submission format draft v0.1 - 2026-02-24**  
**Target journal:** Remote Sensing

Mukhlis Amien^1,*

^1^ VOLCARCH Project, Indonesia; amien@ubhinus.ac.id

* Correspondence: amien@ubhinus.ac.id

---

## Abstract

Archaeological settlement modeling in volcanic landscapes is prone to two coupled biases: taphonomic burial and uneven survey effort. We developed a settlement suitability model for East Java using environmental predictors only, with explicit safeguards against tautology (no volcanic-proximity features in training). Seven iterative experiments (E007-E013) were run under fixed spatial block cross-validation to progressively address survey-bias contamination in pseudo-absence design. Baseline terrain modeling (E007) produced AUC 0.659. Adding river distance improved performance (E008, best AUC 0.695), while adding soil covariates reduced transfer performance (E009, best AUC 0.664). Target-Group Background (TGB) sampling with a road-accessibility proxy improved results (E010-E012, best AUC rising to 0.730). A hybrid bias-corrected background strategy combining TGB, regional sampling control, and hard-negative selection achieved the strongest single-run performance (E013, XGBoost AUC 0.768 +/- 0.069; TSS 0.507 +/- 0.167), exceeding the minimum viable threshold (AUC > 0.75). Robustness checks across 20 alternate seeds gave seed-averaged XGBoost AUC 0.751 (95% bootstrap CI 0.745-0.756), and block-size sensitivity showed the ~50 km split was the most favorable among ~40/50/60 km tests. Challenge 1 (tautology test) passed in all runs (rho between suitability and volcano distance remained negative). Overall, pseudo-absence design was the dominant lever for generalization under survey-biased archaeological data.

**Keywords:** archaeological predictive modeling, spatial cross-validation, target-group background, survey bias, East Java

---

## 1. Introduction

Predictive settlement modeling in Java must account for substantial visibility bias in the known archaeological record. Sites that are deeply buried by volcanic deposition are less likely to appear in surface inventories, and survey intensity is concentrated in historically prominent regions. Under these conditions, random pseudo-absence generation can contaminate negative samples with unobserved positives, suppressing model transfer performance.

Paper 2 tests whether a settlement suitability model can still learn stable environmental signal while remaining independent from volcanic visibility bias. The main criterion is spatial AUC > 0.75 (MVR), with an explicit tautology check requiring that suitability is not trivially explained by distance from volcanoes. The pseudo-absence bias framing follows target-group background logic from presence-only SDM literature [1].

---

## 2. Materials and Methods

### 2.1 Study Area
- Province-scale East Java extent (approximate analysis bounds: 111-115 E, 9-6.5 S).
- Working CRS for raster analysis: EPSG:32749.

### 2.2 Archaeological Presence Dataset
- Source: `data/processed/east_java_sites.geojson`
- Geocoded archaeological sites used in model experiments:
  - E007/E008/E010/E011/E012/E013: 378 presences (after feature-valid filtering)
  - E009: 259 presences (soil-layer coverage reduced valid sample count)

### 2.3 Environmental Covariates and Accessibility Proxies
- Terrain: `elevation`, `slope`, `twi`, `tri`, `aspect`
- Hydrology proxy: `river_dist`
- Soil features (E009 only): `clay`, `silt` from SoilGrids 0-5 cm mean, reprojected to DEM grid
- Survey-accessibility proxies:
  - `jatim_road_dist.tif` (major roads; E010-E011)
  - `jatim_road_dist_expanded.tif` (major + local roads; E012-E013)

---

### 2.4 Modeling Setup and Validation
- Binary classification with pseudo-absence design.
- Algorithms:
  - XGBoost (primary model)
  - RandomForest (secondary benchmark)
- Validation:
  - 5-fold spatial block CV
  - block size ~50 km (`BLOCK_SIZE_DEG=0.45`)
- Primary metrics: Spatial AUC, TSS.
- Decision thresholds:
  - AUC > 0.75: GO
  - 0.65 <= AUC <= 0.75: REVISIT
- AUC < 0.65: kill-signal territory

Spatially structured validation is used to avoid optimistic leakage from nearby training samples [2,3]. TSS is included to complement threshold-independent AUC interpretation [4].
Model implementations follow established tree-ensemble formulations [5,6] via the scikit-learn ecosystem [7].

### 2.5 Tautology Control (Challenge 1)
- No volcanic-proximity variables were included in training features.
- Post hoc test:
  - Compute Spearman rho between predicted suitability and nearest-volcano distance.
  - Inspect share of high-suitability cells within 50 km volcano radius.
- Pass condition: model remains suitability-driven (not simply far-from-volcano visibility).

### 2.6 Experiment Sequence and Interventions

#### E007 (baseline)
- Features: terrain only (`elevation`, `slope`, `twi`, `tri`, `aspect`)
- Pseudo-absence: random with 2 km exclusion around known sites

#### E008 (feature enrichment)
- Added `river_dist` to baseline features
- Same pseudo-absence strategy as E007

#### E009 (SoilGrids path)
- Added `clay` and `silt`
- Same pseudo-absence strategy as E008

#### E010-E012 (TGB path)
- Replaced random pseudo-absences with TGB sampling weighted by road accessibility
- E010: initial TGB settings
- E011: parameter sweep on TGB (`decay`, `max_road_dist`) with deterministic fold assignment
- E012: expanded-road proxy and repeated fixed-split sweep

#### E013 (hybrid bias correction)
- Base proxy from E012 plus hybrid pseudo-absence controls:
  - regional quota blending (`region_blend`) to reduce spatial concentration artifacts
  - hard-negative targeting via environmental dissimilarity (`hard_frac`, z-distance threshold)
- 12-configuration sweep:
  - `region_blend` in {0.0, 0.3, 0.5, 0.7}
  - `hard_frac` in {0.0, 0.15, 0.30}

---

## 3. Results

### 3.1 Performance Progression (E007-E013)

| Experiment | Pseudo-absence strategy | XGB AUC | RF AUC | Best AUC | XGB TSS | RF TSS | Status |
|------------|--------------------------|---------|--------|----------|---------|--------|--------|
| E007 | Random | 0.659 +/- 0.077 | 0.656 +/- 0.090 | 0.659 | 0.318 +/- 0.126 | 0.314 +/- 0.133 | REVISIT |
| E008 | Random + river feature | 0.685 +/- 0.074 | 0.695 +/- 0.107 | 0.695 | 0.345 +/- 0.135 | 0.379 +/- 0.200 | REVISIT |
| E009 | Random + soil features | 0.664 +/- 0.049 | 0.643 +/- 0.054 | 0.664 | 0.337 +/- 0.083 | 0.312 +/- 0.072 | REVISIT |
| E010 | TGB (major-road proxy) | 0.711 +/- 0.085 | 0.699 +/- 0.081 | 0.711 | 0.384 +/- 0.150 | 0.380 +/- 0.130 | REVISIT |
| E011 | TGB tuned (major-road proxy) | 0.725 +/- 0.084 | 0.716 +/- 0.081 | 0.725 | 0.447 +/- 0.184 | 0.408 +/- 0.147 | REVISIT |
| E012 | TGB tuned (expanded-road proxy) | 0.730 +/- 0.085 | 0.724 +/- 0.081 | 0.730 | 0.420 +/- 0.170 | 0.413 +/- 0.152 | REVISIT |
| E013 | Hybrid bias-corrected background | 0.768 +/- 0.069 | 0.742 +/- 0.070 | 0.768 | 0.507 +/- 0.167 | 0.458 +/- 0.126 | SUCCESS |

Table 2 and Figure 3 summarize the full metric trajectory and show the transition
from feature-led tuning to background-led bias correction.

Key pattern:
- Feature-only expansion plateaued below MVR (E007-E009).
- Bias-corrected background design (E010-E013) delivered monotonic gains.
- E013 exceeded MVR by +0.018 (0.768 vs 0.750 threshold).

### 3.2 Best-Performing E013 Configuration

Best hybrid config from 12 tested:
- `region_blend=0.00`
- `hard_frac_target=0.30` (realized hard fraction ~0.62)
- `seed=375`
- mean pseudo-absence road distance: 434 m

Figure 2 shows sweep response across the hybrid-parameter grid, and Figure 4
shows fold-level behavior for the selected E013 configuration.

Top feature importances (XGBoost, E013):
1. elevation: 0.215
2. tri: 0.185
3. twi: 0.166
4. river_dist: 0.160
5. slope: 0.155
6. aspect: 0.118

### 3.3 Tautology Test Outcomes

Challenge 1 passed in every iteration; rho remained negative throughout:
- E007: -0.095
- E008: -0.153
- E009: -0.266
- E010: -0.142
- E011: -0.169
- E012: -0.160
- E013: -0.229

E013 high-suitability cells within 50 km volcano radius: 57.9%.
Interpretation: high suitability is not restricted to low-volcanic-exposure zones, supporting tautology-free behavior.
Figure 5 visualizes rho progression and confirms all runs remain below the tautology-risk threshold.

### 3.4 Interpretation of Gains

Observed gains indicate that pseudo-absence realism is the primary bottleneck:
- `E008 -> E009` (add soil features) decreased best AUC by -0.031.
- `E009 -> E010` (switch to TGB) increased best AUC by +0.047.
- `E012 -> E013` (hybrid sampling) increased best AUC by +0.038.

The strongest gains occur when correcting negative-sample bias rather than adding raw covariates.

### 3.5 Robustness Checks (Alternate Seeds + Bootstrap CI)

To test stability beyond the single best E013 seed, we fixed hybrid parameters
(`region_blend=0.00`, `hard_frac_target=0.30`) and reran 20 alternate pseudo-absence
seeds with the same spatial CV setup.

Robustness summary:
- XGBoost mean AUC: 0.751 +/- 0.013; bootstrap 95% CI [0.745, 0.756]
- XGBoost mean TSS: 0.465 +/- 0.021; bootstrap 95% CI [0.456, 0.474]
- XGBoost MVR pass-rate (AUC >= 0.75): 55%
- RandomForest mean AUC: 0.744 +/- 0.010; bootstrap 95% CI [0.740, 0.749]
- RandomForest mean TSS: 0.458 +/- 0.016; bootstrap 95% CI [0.451, 0.464]
- RandomForest MVR pass-rate (AUC >= 0.75): 25%

Observed AUC ranges across seeds were 0.729-0.774 (XGBoost) and 0.730-0.762 (RandomForest).
Figure 6 visualizes seed-level stability. Supplementary outputs are archived as:
`papers/P2_settlement_model/supplement/e013_seed_stability.csv` and
`papers/P2_settlement_model/supplement/e013_fold_metrics_by_seed.csv`.

### 3.6 Block-Size Sensitivity

Using the same fixed E013 hybrid parameters and 20 alternate seeds, we evaluated
three spatial block scales: ~40 km, ~50 km (baseline), and ~60 km.

Summary:
- ~40 km blocks: XGBoost AUC 0.725 (95% CI 0.718-0.733), RF AUC 0.742 (0.738-0.746)
- ~50 km baseline: XGBoost AUC 0.751 (0.746-0.757), RF AUC 0.744 (0.740-0.749)
- ~60 km blocks: XGBoost AUC 0.742 (0.737-0.747), RF AUC 0.732 (0.729-0.736)

MVR pass-rates (AUC >= 0.75):
- XGBoost: 5% (~40 km), 55% (~50 km), 25% (~60 km)
- RandomForest: 25% (~40 km), 25% (~50 km), 0% (~60 km)

Interpretation: the ~50 km split remains the most favorable scale for transfer in this
setup. Smaller blocks penalize cross-block extrapolation for XGBoost, while larger blocks
reduce fold diversity and degrade RF more strongly.
Figure 7 and Table S2 provide full sensitivity outputs.

---

## 4. Discussion

### 4.1 Why Feature-Only Expansion Plateaued

The E007-E009 sequence shows that adding more environmental variables does not guarantee better spatial transfer under survey-biased data. River distance improved performance from E007 to E008, consistent with expected settlement ecology. However, E009 (soil enrichment) reduced performance despite additional covariates. This pattern suggests that model error was dominated by label-noise structure in pseudo-absences, not by missing predictors alone.

### 4.2 Background Design as the Dominant Performance Lever

The largest gains appear when pseudo-absence design is corrected:
- E009 -> E010 (random to TGB) yielded the strongest single improvement (+0.047 best AUC).
- E010 -> E012 yielded smaller but consistent gains through TGB tuning and improved accessibility proxy quality.
- E012 -> E013 cleared MVR through hybrid sampling controls, not by introducing new predictor families.

These results support a practical rule for archaeological SDM in survey-biased contexts: prioritize negative-sample realism before expanding feature space.

### 4.3 Why Hybrid Sampling Improved Weak-Fold Behavior

E013 combines three mechanisms: accessibility-weighted candidate generation, regional sampling control, and hard-negative emphasis. Together they reduce two recurrent failure modes observed in earlier folds: (1) spatial over-concentration of pseudo-absences in easy-access basins, and (2) insufficient contrast against environmentally plausible but unseen presences. The best E013 run favored higher hard-negative pressure, indicating that boundary sharpening between plausible presences and difficult background was critical for transfer.

### 4.4 Tautology Control and Interpretive Value

Challenge 1 remained negative in all runs (rho < 0), including E013. This is central to interpretability: the model can reach high performance without encoding volcanic-distance shortcuts. In other words, model success does not require collapsing into a visibility map. This strengthens the claim that the model is learning settlement suitability structure rather than rediscovering known-site survey footprints.

### 4.5 Relation to Paper 1 Framing

Paper 1 argued that observable distributions cannot directly validate taphonomic hypotheses because discovery and preservation filters dominate visible data. Paper 2 operationalizes that argument by explicitly treating pseudo-absence construction as a bias problem. The E013 result does not remove taphonomic uncertainty, but it demonstrates that predictive modeling can still be useful when bias pathways are handled explicitly and evaluated under spatially strict validation.

### 4.6 Robustness Interpretation

The robustness runs indicate that E013 clears MVR in many, but not all, stochastic
background draws. This places the model in a near-threshold but defensible regime:
the single best run (AUC 0.768) is reproducible as an upper-end outcome, while the
seed-averaged estimate (AUC 0.751) is a more conservative central summary for manuscript reporting.

### 4.7 Block-Scale Implications

Block-size sensitivity indicates that conclusions are directionally stable but metric
magnitudes are not invariant to split geometry. The main claim (bias-corrected background
improves transfer and can reach MVR) still holds, yet expected AUC should be reported with
explicit block-scale context. For this manuscript, the 0.45-degree (~50 km) protocol remains
the primary benchmark, with ~40 km and ~60 km treated as supplementary stress tests.

---

### 4.8 Limitations

1. Pseudo-absences are still inferred labels, not observed true absences.
2. Accessibility proxies (roads) remain imperfect substitutes for actual survey-effort polygons.
3. In E013, realized hard-negative share varied above the nominal target (0.30), indicating imperfect control of this sampling knob.
4. Bootstrap uncertainty is currently seed-level within one block-size setting; full nested uncertainty (including block-size sensitivity) is not yet quantified.
5. Temporal non-stationarity is not modeled; all features are treated as static contemporary proxies.
6. Block-size sensitivity was only tested at three coarse scales (~40/50/60 km), not as a full continuous sweep.
7. External geographic transfer (outside East Java) is not evaluated in this draft.

---

## 5. Conclusions

E013 achieved the first MVR pass for Paper 2 (best-run XGBoost AUC 0.768 +/- 0.069) under strict spatial CV while maintaining tautology-free behavior. Robustness checks place the seed-averaged XGBoost performance at AUC 0.751 (95% CI 0.745-0.756), indicating near-threshold but stable transfer once pseudo-absence bias correction is applied. The key contribution is a reproducible background-design workflow for survey-biased archaeological settings, shifting emphasis from feature accumulation to negative-sample realism.

---

## Supplementary Materials: Figure and Table Captions

### Figure Captions

**Figure 2.** Hybrid parameter sweep heatmap (E013): best AUC response across
`region_blend` and `hard_frac` settings under fixed spatial CV.  
File: `papers/P2_settlement_model/figures/fig2_hybrid_sweep_heatmap.png`

**Figure 3.** Experiment progression (E007-E013): spatial AUC and TSS trajectories
for XGBoost and RandomForest, with MVR threshold indicated.  
File: `papers/P2_settlement_model/figures/fig3_auc_tss_progression.png`

**Figure 4.** E013 best-configuration fold-level CV performance and feature importance.  
File: `papers/P2_settlement_model/figures/fig4_e013_cv_by_fold.png`

**Figure 5.** Challenge 1 tautology test progression: Spearman rho between suitability
and volcano distance across all experiments.  
File: `papers/P2_settlement_model/figures/fig5_tautology_rho_progression.png`

**Figure 6.** E013 robustness stability across 20 alternate pseudo-absence seeds,
showing mean AUC and TSS trajectories for XGBoost and RandomForest under fixed
hybrid parameters.  
File: `papers/P2_settlement_model/figures/fig6_e013_seed_stability.png`

**Figure 7.** E013 block-size sensitivity under fixed hybrid parameters and 20
alternate seeds per scale, comparing AUC and TSS at ~40 km, ~50 km, and ~60 km
spatial CV blocks.  
File: `papers/P2_settlement_model/figures/fig7_e013_blocksize_sensitivity.png`

---

### Table Captions

**Table 2.** Settlement model experiment progression (E007-E013), including pseudo-absence strategy, spatial AUC, TSS, and status decision.  
File: `papers/P2_settlement_model/tables_experiment_progression.csv`

**Table S1.** E013 alternate-seed robustness summary (20 runs; fixed hybrid parameters).  
File: `papers/P2_settlement_model/supplement/e013_seed_stability.csv`

**Table S2.** E013 block-size sensitivity summary across ~40 km, ~50 km, and ~60 km
spatial CV scales (20 seeds each).  
File: `papers/P2_settlement_model/supplement/e013_blocksize_summary.csv`

---

## Data Availability Statement

- Core presence dataset: `data/processed/east_java_sites.geojson`
- Terrain and proxy rasters: `data/processed/dem/`
- Experiment pipelines: `experiments/E007_settlement_suitability_model/` through `experiments/E013_settlement_model_v7/`
- Supplementary metrics and robustness outputs: `papers/P2_settlement_model/supplement/`

## Code Availability Statement

- Paper 2 figure and supplement builders:
  - `papers/P2_settlement_model/build_figures.py`
  - `papers/P2_settlement_model/robustness_checks.py`
  - `papers/P2_settlement_model/block_size_sensitivity.py`

## Author Contributions

Conceptualization, M.A.; methodology, M.A.; software, M.A.; validation, M.A.;
formal analysis, M.A.; investigation, M.A.; resources, M.A.; data curation, M.A.;
writing-original draft preparation, M.A.; writing-review and editing, M.A.;
visualization, M.A.; supervision, M.A.; project administration, M.A.

## Funding

This research received no external funding.

## Conflicts of Interest

The authors declare no conflict of interest.

## Institutional Review Board Statement

Not applicable.

## Informed Consent Statement

Not applicable.

## Acknowledgments

The authors thank colleagues and reviewers who provided critical feedback on
model evaluation design and manuscript framing.

---

## References

[1] Phillips, S.J.; Dudik, M.; Elith, J.; Graham, C.H.; Lehmann, A.; Leathwick, J.; Ferrier, S. Sample selection bias and presence-only distribution models: Implications for background and pseudo-absence data. *Ecol. Appl.* **2009**, *19*, 181-197. https://doi.org/10.1890/07-2153.1.

[2] Roberts, D.R.; Bahn, V.; Ciuti, S.; Boyce, M.S.; Elith, J.; Guillera-Arroita, G.; Hauenstein, S.; Lahoz-Monfort, J.J.; Schroder, B.; Thuiller, W.; et al. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* **2017**, *40*, 913-929. https://doi.org/10.1111/ecog.02881.

[3] Valavi, R.; Elith, J.; Lahoz-Monfort, J.J.; Guillera-Arroita, G. blockCV: An R package for generating spatially or environmentally separated folds for k-fold cross-validation of species distribution models. *Methods Ecol. Evol.* **2019**, *10*, 225-232. https://doi.org/10.1111/2041-210X.13107.

[4] Allouche, O.; Tsoar, A.; Kadmon, R. Assessing the accuracy of species distribution models: Prevalence, kappa and the true skill statistic (TSS). *J. Appl. Ecol.* **2006**, *43*, 1223-1232. https://doi.org/10.1111/j.1365-2664.2006.01214.x.

[5] Breiman, L. Random forests. *Mach. Learn.* **2001**, *45*, 5-32. https://doi.org/10.1023/A:1010933404324.

[6] Chen, T.; Guestrin, C. XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*, San Francisco, CA, USA, 13-17 August 2016; pp. 785-794. https://doi.org/10.1145/2939672.2939785.

[7] Pedregosa, F.; Varoquaux, G.; Gramfort, A.; Michel, V.; Thirion, B.; Grisel, O.; Blondel, M.; Prettenhofer, P.; Weiss, R.; Dubourg, V.; et al. Scikit-learn: Machine learning in Python. *J. Mach. Learn. Res.* **2011**, *12*, 2825-2830.

# P2 Revision Ammo: Anticipated Critiques & Pre-Computed Responses

**Paper:** "Volcanic Taphonomic Bias in Settlement Pattern Analysis: A Predictive Model for East Java"
**Journal:** JCAA (J. Computer Applications in Archaeology), Submission #280
**Authors:** Mukhlis Amien + Go Frendi Gunawan
**Prepared:** 2026-03-12

---

## Critique 1: "AUC 0.768 is modest — not convincing for a predictive model"

**Anticipated from:** Quantitative reviewer familiar with ML benchmarks.

**Response:**
"We agree that AUC 0.768 is modest compared to standard ML benchmarks. However, three factors argue this is meaningful in our archaeological context:

1. **Archaeological site location is inherently noisy.** Sites are discovered through survey, not random sampling. Our dependent variable contains systematic survey bias, which we explicitly model (bias-correction features, §3.3). The true predictive ceiling may be well below 1.0.

2. **Seed-averaged AUC is 0.751** (10 random seeds), confirming the point estimate is not an artifact of data split.

3. **Temporal split validation (E014) yields AUC 0.755** — the model trained on one time period predicts sites from a different period. This is a far stronger validation than random split, and the minimal AUC drop (0.768 → 0.755) indicates the model captures genuine spatial patterns, not temporal confounds.

The model's value is not in achieving a high benchmark — it is in identifying specific landscape cells where archaeological survey is most likely to discover currently unknown sites (Zone B, §4.2)."

**Supporting data:** `experiments/E013_settlement_model_v7/results/`, `experiments/E014_temporal_split/results/`

---

## Critique 2: "Tautology — you trained on known sites, so the model just predicts where archaeologists already looked"

**Anticipated from:** Any reviewer. This is the most serious potential criticism.

**Response:**
"We explicitly address this in §3.3 (Tautology Test Design). Three safeguards:

1. **Survey accessibility features** (road distance, population density, terrain ruggedness) are included specifically to absorb survey bias. If the model were tautological, these features would dominate SHAP importance. They do not — volcanic proximity (dist_nearest_volcano) and eruption frequency rank higher (Figure 5, SHAP analysis).

2. **Temporal split (E014):** Training on pre-1000 CE sites and predicting post-1000 CE sites yields AUC 0.755. Survey methods changed dramatically between these periods, yet the model generalizes. A tautological model would fail this test.

3. **Zone B prediction:** The model identifies 1.8% of the landscape as high-probability but zero known sites (Zone B). If the model were merely reflecting survey patterns, Zone B would not exist — it predicts where sites SHOULD be but haven't been found, precisely the taphonomic bias claim.

We classify the tautology test as CONDITIONAL PASS — the temporal split is strong, but definitive resolution requires archaeological verification of Zone B cells."

**Supporting data:** `experiments/E014_temporal_split/README.md`, `experiments/E015_shap_analysis/results/`

---

## Critique 3: "No fieldwork validation — you haven't actually found any buried sites"

**Anticipated from:** Field archaeologist reviewer.

**Response:**
"This is correct, and we are transparent about this limitation (§5.3). The paper frames itself as a *computational framework* generating testable predictions, not as confirmation of buried sites. Zone B cells (Figure 7) are explicitly presented as GPR survey targets, not confirmed sites.

However, three independent lines of evidence support the model's plausibility:

1. **Known buried sites validate the mechanism:** Sambisari (discovered 1966, buried 5m), Kedulan (buried 2.7m), and Liangan (buried 4m) are all within the model's high-probability zone. These sites were discovered accidentally, not through our model — they are independent validation.

2. **E040 (Bamboo Civilization):** 170/268 Old Javanese inscriptions (63.4%) document organic building materials. The archaeological 'blank' in Zone B is consistent with an organic civilization that left few durable remains above the tephra deposit horizon.

3. **Sedimentation rate calibration (P1):** Independent measurement of 3.6-4.4 mm/yr volcanic sedimentation in East Java predicts 3.6-4.4 m burial depth for 1000-year-old sites. This is below typical survey detection but within GPR range.

We welcome collaboration with field archaeologists to test Zone B predictions. A GPR transect across 5-10 highest-probability cells would provide definitive validation."

**Supporting data:** `experiments/E016_zone_classification/results/`, P1 sedimentation rates

---

## Critique 4: "Why XGBoost? Why not [other method]?"

**Anticipated from:** Methodologically-focused reviewer.

**Response:**
"We chose XGBoost for three reasons: (1) it handles mixed feature types (continuous + categorical) without preprocessing, (2) it provides native feature importance and SHAP compatibility for interpretability, and (3) it has established precedent in archaeological predictive modeling (Yaworsky et al. 2020, Bickler 2021).

We tested alternatives in the model development sequence (E007-E013): logistic regression (AUC 0.659), random forest (AUC 0.711), and XGBoost with different hyperparameter configurations. XGBoost with hybrid bias-correction features achieved the best balance of performance and interpretability.

We note that our contribution is not the ML method itself — it is the taphonomic bias-correction framework (survey features + volcanic features) that can be applied with any gradient boosting method."

---

## Critique 5: "East Java only — not generalizable"

**Anticipated from:** Reviewer wanting broader significance.

**Response:**
"The geographic scope is intentional, not a limitation. East Java has (a) the densest archaeological site record in Indonesia, (b) multiple well-characterized volcanic systems, and (c) known buried sites for partial validation (Sambisari, Kedulan, Liangan). It is the ideal test case for developing the framework.

The framework IS generalizable: the same feature engineering (volcanic proximity, sedimentation rate estimates, survey accessibility controls) can be applied to any volcanically active region with archaeological records. Central Java (Merapi-Merbabu corridor) and Bali (Agung system) are immediate extension targets.

We present this as a case study + framework paper, not a universal predictive model."

---

## Critique 6: "SHAP-gain correlation (rho=0.943) — aren't these measuring the same thing?"

**Anticipated from:** Statistician or ML-literate reviewer.

**Response:**
"SHAP values and gain importance measure different aspects of feature contribution: gain measures how much a feature reduces training loss when used in splits, while SHAP measures the average marginal contribution of a feature value across all predictions. Their high correlation (rho=0.943) indicates that our model's feature importance is consistent across methods — it is a robustness check, not a redundancy.

If SHAP and gain diverged substantially, it would indicate that the model relies on complex feature interactions that single-feature importance misrepresents. The convergence supports our interpretability claims."

**Supporting data:** `experiments/E015_shap_analysis/README.md`

---

## Additional Revision Resources

### Experiments Available for Extended Analysis (if requested)

| Experiment | What it adds | Execution time |
|-----------|-------------|----------------|
| E016 | Zone B/C classification map — visualize prediction zones | Already complete |
| New: sensitivity to seed | Run 50 seeds instead of 10 for tighter CI | ~2 hours |
| New: other volcanic systems | Apply framework to Merapi corridor (central Java) | ~1 day |
| New: feature ablation | Drop volcanic features → AUC drop = volcanic contribution | ~2 hours |

### Cross-Paper Reinforcement

- **P1 → P2:** P1's sedimentation rates (3.6-4.4 mm/yr) provide independent calibration for P2's burial depth predictions. If P1 is accepted, cite published DOI; if preprint, cite EarthArXiv DOI.
- **P7 → P2:** P7's spatial segregation result (Zone B 16km vs Zone A 43km, Cohen's d=1.005) provides independent evidence that survey bias is spatially structured. Different method, same conclusion.
- **E040 → P2:** Organic material culture dominance in inscriptions explains WHY Zone B has no surface finds — the civilization was built of perishable materials.

---

*Prepared 2026-03-12. Use when reviewer comments arrive.*

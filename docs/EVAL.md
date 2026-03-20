# EVAL.md — Evaluation Criteria & Validation Protocol

**Rule: Define how you measure success BEFORE running experiments.**
**Last updated:** 2026-03-16

---

## 1. Settlement Suitability Model (Paper 2)

### Validation Method
**Spatial block cross-validation** — NOT random CV. Archaeological sites have spatial autocorrelation; random CV would inflate metrics.

Implementation: Divide study area into spatial blocks (e.g., 5km × 5km grid). Each fold holds out entire blocks, not individual points.

### Primary Metrics

| Metric | MVR (Minimum) | Good | Excellent |
|--------|--------------|------|-----------|
| Spatial AUC-ROC | > 0.75 | > 0.80 | > 0.85 |
| True Skill Statistic (TSS) | > 0.40 | > 0.50 | > 0.60 |

### Secondary Metrics (report but don't gate on)
- **Precision@5%**: Of the top 5% highest-probability area, what fraction contains known sites?
- **Feature importance** (SHAP or permutation): Which environmental features drive predictions?

### Kill Signal
- Spatial AUC consistently < 0.65 after reasonable feature engineering → model is not learning useful patterns → pivot or abandon.

---

## 2. Volcanic Burial Depth Model (Paper 3) — KILLED

**Status:** KILLED (2026-03-10, Mata Elang #2). E017 POC FAILED — 1/4 calibration sites passed. Generic Pyle 1989 model insufficient; requires per-volcano calibration with Tephra2/FALL3D and geologist co-author.

**Resurrection condition:** Geologist collaborator + Tephra2/FALL3D access.

~~Calibration Points, Validation Method, and Kill Signal below are archived for reference.~~

### Calibration Points (archived)

| Site | Known Depth | Acceptable Prediction Range (±30%) |
|------|------------|--------------------------------------|
| Dwarapala Singosari | ~185 cm | 130 – 240 cm |
| Candi Sambisari | ~650 cm | 455 – 845 cm |
| Candi Kedulan | ~700 cm | 490 – 910 cm |
| Candi Kimpulan | ~270 cm | 189 – 351 cm |

### Kill Signal (triggered)
- Cannot predict Dwarapala within ±50% → fundamental model problem. **E017 confirmed: only 1/4 sites passed ±30%.**

---

## 3. Tautology Test (Challenge 1)

**Purpose:** Verify the settlement model learns *suitability* not *visibility*.

### Design
1. Train model using ONLY environmental features (slope, river distance, soil, TWI, TRI, aspect). NO features related to volcanic proximity, burial depth, or modern accessibility.
2. Generate probability map.
3. Test: Does the model predict high suitability in areas with HIGH volcanic deposition (where few sites are currently known)?

### Success Criteria
- Model achieves spatial AUC > 0.70 using only environmental features.
- AND model predicts some high-suitability zones in high-burial-depth areas → evidence that suitability and burial are independent → H1 supported.

### Failure Mode
- Model only predicts high suitability where sites are already found → tautology → need to redesign features or approach.

---

## 3b. Temporal Split Validation (Enhanced Tautology Test)

**Purpose:** Provide stronger evidence against tautology by testing model on sites that were discovered *later* (post-2000) when trained only on sites discovered *earlier* (pre-2000).

### Design (E014)
1. Split positive samples by discovery year (or accessibility as proxy).
2. Train on pre-2000 / easy-access sites (likely discovered earlier).
3. Test on post-2000 / hard-access sites (likely discovered later).

### Success Criteria
- Temporal AUC > 0.65 → Model predicts "undiscovered" sites (tautology-resistant)
- Temporal AUC within 0.05 of spatial CV AUC → Good generalization

### Results (E014)
| Metric | Value |
|--------|-------|
| Temporal Test AUC | **0.755** |
| Spatial CV AUC | 0.785 ± 0.058 |
| Difference | -0.030 |
| Verdict | **PASS** |

### Integrated Tautology Verdict (E013 + E014)

| Test | Verdict | Key Metric |
|------|---------|-----------|
| T1: Multi-Proxy Correlation | GREY_ZONE | max \|rho\| = 0.307 (road_dist) |
| T2: Spatial Prediction Gap | GREY_ZONE | D = 0.322, far-zone 13% high-suit |
| T3: Stratified CV | **PASS** | Delta AUC = +0.057, Q4 > Q1 |
| T4: Temporal Split | **PASS** | AUC = 0.755 vs 0.785 spatial |
| **Overall** | **CONDITIONAL PASS** | T3-T4 robust; T1-T2 near threshold |

**Rationale:** T3 and T4 provide strong anti-tautology evidence (model performs *better* in least-surveyed areas and generalizes to held-out "undiscovered" sites). T1-T2 are in the grey zone but not failing. Overall verdict is CONDITIONAL rather than unconditional because definitive tautology absence cannot be proven from observational data alone.

---

## 4. Integrated Map (Paper 4 — Phase 2)

### Zone Classification Validation

| Zone | Expected Content | Validation Method |
|------|-----------------|-------------------|
| A (High suit., shallow) | Correlates with known sites | Check overlap with site database |
| B (High suit., moderate burial) | **Priority GPR targets** | Fieldwork (Phase 2) |
| C (High suit., deep) | Likely present, hard to reach | Literature check for any deep finds |
| E (Low suit., any) | Few or no sites expected | Should have few known sites |

### Minimum Fieldwork Validation (Phase 2)
- GPR survey at 5–10 Zone B locations.
- Success: At least 1 location shows subsurface anomaly consistent with anthropogenic material.

---

## 5. Paper-Specific Evaluation Criteria

### Paper 1: Taphonomic Bias Framework (Asian Perspectives)
- **Core claim:** Volcanic sedimentation rates 2.4–6.2 mm/yr across 2 volcanic systems
- **MVR:** Multi-site calibration consistent within order of magnitude — **MET**
- **Key test:** ADV-3 (E069) — volcanic signal survives survey intensity control (p=0.0015)
- **Known weakness:** ADV-1 (E086) requires framing as volcanism × survey deficit, not volcanism alone
- **Revision readiness:** Japan paragraph (ADV1_japan_comparanda.md), depth argument (ADV2_depth_vs_sitetype.md)

### Paper 5: Volcanic Ritual Clock (BKI)
- **Core claim:** Slametan and ritual practices preserve pre-Indic cosmology beneath Sanskrit overlay
- **MVR:** Quantitative evidence for pre-Indic persistence across centuries — **MET** (E030: rho=+0.502, p<0.001)
- **Supporting:** E023 (43% hyang), E025 (Monte Carlo p<0.001), E035 (mortuary plants absent from epigraphy)
- **Known weakness:** E032 seasonality correlation is FDR casualty (p=0.042, fails BH)
- **Revision readiness:** E058 (agriculture 91% native), E048 (genre taphonomy)

### Paper 7: Temporal Overlay Matrix (Antiquity Project Gallery)
- **Core claim:** Deep-time spatial segregation of archaeological sites reveals taphonomic bias
- **MVR:** Statistically significant spatial pattern — **MET** (E019: Cohen's d=1.005)
- **Supporting:** E020 (cave bias universal), E065 (Zone A 17.9× overrepresented), E066 (equinox p=4.9e-14)
- **Format:** Short project gallery format (~2000 words + 4 images)

### Paper 8: Linguistic Fossils (Oceanic Linguistics)
- **Core claim:** Computational detection of pre-Indic phonological substrate in western Indonesian languages
- **MVR:** ML substrate detection AUC > 0.70 — **MET** (E027: AUC=0.762, LOLO 5/6 ≥ 0.65)
- **Supporting:** E028 (kappa=0.61), E036 (33→20 consonants), E029 (parallel innovation, not shared substrate)
- **Known weakness:** ADV-5 (E087) grey zone — C5 AUC=0.713 from ABVD documentation gaps
- **Revision readiness:** Must reframe as "phonological non-conformity" not "substrate detection" (ADV5_negative_control.md)

### Paper 9: Peripheral Conservatism (JSEAS)
- **Core claim:** Peripheral communities (Bali, Tengger, Trunyan) conserve pre-Indic features lost in court centers
- **MVR:** Quantitative cognacy difference between peripheral and central languages — **MET** (E043: Bal 40.3% > Jav 33.0%)
- **Supporting:** E044 (Canarium pan-AN), E050 (GBIF confirmation), E054 (1,309 languages, local gradient confirmed)
- **MS#:** JSEAS-202603-051

### Paper 11: Temple Siting as Archaeological Proxy (Indonesia/Cornell)
- **Core claim:** Candi siting patterns reveal volcanic awareness and can serve as archaeological survey proxy
- **MVR:** Statistically significant spatial relationship between candi and volcanic zones
- **Supporting:** E065 (Zone A 17.9×), E066 (equinox p=4.9e-14), E082 (182 inscriptions geocoded), E084 (MW p=5.2e-08)
- **Current status:** v0.3 drafted (18pp), all self-citations removed, 10 references. Next: manual review → submit to Indonesia (Cornell)
- **Key constraint:** VCS rejected globally (E039), valid only locally (Java/Bali)

---

## 6. Multi-Test Correction Strategy

### Approach: Benjamini-Hochberg FDR Control

**Documented in E068 (FDR Meta-Audit, 2026-03-13).**

Across 117 experiments (as of 2026-03-20, originally audited at 90), 41 distinct statistical hypothesis tests were identified. Applied Benjamini-Hochberg procedure at α=0.05.

| Category | Count |
|----------|-------|
| Tests submitted | 41 |
| Survive BH (q<0.05) | 30 (73%) |
| FDR casualties | 3 |
| Top 10 findings | All p < 10⁻⁴ |

### FDR Casualties (report as "suggestive," not "significant")
- E032 Pranata Mangsa seasonality: p=0.042 uncorrected
- E048 partial correlation (organic × pre-Indic): p=0.038 uncorrected
- E053 aDNA taphonomic gap: Fisher p=0.047 uncorrected

### Reporting Rule
- **p < 0.01 after BH:** Report as "significant"
- **0.01 < p < 0.05 after BH:** Report as "marginally significant"
- **p > 0.05 after BH but < 0.05 uncorrected:** Report as "suggestive" with explicit FDR note
- **All papers must reference E068** when claiming statistical significance across multiple tests

---

## 7. General Reporting Rules

- Always report **spatial** metrics, never random-split metrics.
- Always report **uncertainty**: bootstrap 95% CI for AUC/TSS (minimum 100 iterations).
- Always report **sample size**: number of positive sites, number of pseudo-absences, study area extent.
- Failed experiments: report metrics honestly. Do not cherry-pick runs.

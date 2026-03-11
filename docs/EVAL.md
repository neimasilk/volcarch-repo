# EVAL.md — Evaluation Criteria & Validation Protocol

**Rule: Define how you measure success BEFORE running experiments.**
**Last updated:** 2026-03-10

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

## 5. General Reporting Rules

- Always report **spatial** metrics, never random-split metrics.
- Always report **uncertainty**: bootstrap 95% CI for AUC/TSS (minimum 100 iterations).
- Always report **sample size**: number of positive sites, number of pseudo-absences, study area extent.
- Failed experiments: report metrics honestly. Do not cherry-pick runs.

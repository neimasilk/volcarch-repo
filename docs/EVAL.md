# EVAL.md — Evaluation Criteria & Validation Protocol

**Rule: Define how you measure success BEFORE running experiments.**
**Last updated:** 2026-02-24

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

## 2. Volcanic Burial Depth Model (Paper 3)

### Calibration Points

| Site | Known Depth | Acceptable Prediction Range (±30%) |
|------|------------|--------------------------------------|
| Dwarapala Singosari | ~185 cm | 130 – 240 cm |
| Candi Sambisari | ~650 cm | 455 – 845 cm |
| Candi Kedulan | ~700 cm | 490 – 910 cm |
| Candi Kimpulan | ~270 cm | 189 – 351 cm |

### Validation Method
Leave-one-out among calibration points (given we have very few). Predict each point using model calibrated on the others.

### Primary Metric
- Predictions within ±30% for at least 3 of 4 calibration points.

### Kill Signal
- Cannot predict Dwarapala within ±50% → fundamental model problem.

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

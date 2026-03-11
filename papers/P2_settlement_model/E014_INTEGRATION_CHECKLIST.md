# E014 Integration Verification — Paper 2

**Date:** 2026-02-26  
**Paper:** submission_remote_sensing_v0.3.tex / submission_remote_sensing_FINAL.pdf

---

## ✅ E014 Content in Paper 2

### 1. Abstract (Page 1)
**Status:** ✅ Included
```latex
Challenge 1 (tautology test) passed across all runs (negative Spearman $\rho$ 
between suitability and volcano distance), with temporal split validation 
confirming generalization to ``undiscovered'' sites (AUC = 0.755).
```

### 2. Table 2 — Enhanced Tautology Test Suite (Page 8)
**Status:** ✅ Included

| Test | Verdict | Key Metric |
|------|---------|------------|
| T1: Multi-Proxy Correlation | GREY_ZONE | max $|\rho|$ = 0.307 |
| T2: Spatial Prediction Gap | GREY_ZONE | $D$ = 0.322 |
| T3: Stratified CV | **PASS** | $\Delta$AUC = +0.057 |
| **T4: Temporal Split** | **PASS** | **AUC = 0.755** |

### 3. Section 3.2 — Test 4 Description (Page 8)
**Status:** ✅ Included

> "We conducted a temporal stress test by training on easy-access sites (road distance ≤1 km, proxy for early discovery) and testing on hard-access sites (road distance >1 km, proxy for later discovery). The model achieved AUC = 0.755 on the held-out hard-access sites, only 0.030 below the spatial CV baseline (AUC = 0.785)."

### 4. Verdict Update (Page 8)
**Status:** ✅ Included — Updated to **PASS**

> "Overall, the enhanced tautology test suite returns a PASS verdict: the temporal validation (T4) and stratified CV (T3) provide robust evidence against tautology..."

### 5. Data Availability Statement (Page 15)
**Status:** ✅ Included
- Mentions temporal validation (E014)
- Links to GitHub repository

### 6. Code Availability Statement (Page 15)
**Status:** ✅ Included
```latex
\item \texttt{experiments/E014_temporal_validation/01_temporal_split_test.py} 
      — Temporal validation (E014)
```

---

## 📄 File Locations

| File | Path |
|------|------|
| Source (LaTeX) | `papers/P2_settlement_model/submission_remote_sensing_v0.3.tex` |
| PDF (Latest) | `papers/P2_settlement_model/submission_remote_sensing_v0.3.pdf` |
| PDF (Final) | `papers/P2_settlement_model/submission_remote_sensing_FINAL.pdf` |

---

## 🔍 How to Verify in PDF

**Option 1: Search PDF for "E014"**
- Should appear in Code Availability Statement

**Option 2: Search PDF for "0.755"**
- Should appear in Abstract, Table 2, and Section 3.2

**Option 3: Go to specific pages**
- Page 1: Abstract mention
- Page 8: Table 2 and Section 3.2
- Page 15: Code Availability Statement

---

## ✅ Integration Complete

All E014 results have been properly integrated into Paper 2.

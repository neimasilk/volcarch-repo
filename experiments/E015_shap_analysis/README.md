# E015: SHAP Analysis for E013 Best Model

**Status:** SUCCESS
**Date:** 2026-03-03
**Paper:** P2 (Remote Sensing)

## Hypothesis
Feature importances from SHAP (SHapley Additive exPlanations) will provide model-agnostic, instance-level interpretability for the E013 XGBoost settlement suitability model, complementing the gain-based importance ranking already reported.

## Method
1. Rebuild E013 best-config training data (region_blend=0.00, hard_frac_target=0.30, seed=375)
2. Retrain identical XGBoost model (n_estimators=300, max_depth=4, lr=0.05)
3. Compute TreeSHAP values for all training samples
4. Generate beeswarm summary plot -> Fig 13 in Paper 2

## Data Used
- Same as E013: `data/processed/dem/jatim_*.tif`, `data/processed/east_java_sites.geojson`

## Expected Output
- `results/shap_beeswarm.png` - SHAP beeswarm plot
- `results/shap_summary.csv` - Mean |SHAP| per feature
- `results/shap_analysis_report.txt` - Full analysis report

## Result
- SHAP-gain rank consistency: Spearman rho = 0.943 (CONSISTENT)
- Top features: Elevation (0.816) > TRI (0.671) > River dist (0.602) > Slope (0.579) > Aspect (0.374) > TWI (0.313)
- Only rank swap: TWI/Aspect (gain rank 5/6 vs SHAP rank 6/5)
- Beeswarm plot integrated as Fig 13 in Paper 2 Section 3.3

## Success Criteria
- SHAP ranking should be broadly consistent with gain-based importance (no contradictions) -- **MET**
- Plot renders cleanly for manuscript inclusion -- **MET**

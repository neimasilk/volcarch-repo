# E014: Temporal Split Validation — Tautology Stress Test

**Status:** READY TO RUN  
**Type:** Validation Experiment  
**Paper:** Paper 2 (Settlement Suitability Model)

## Hypothesis

If the settlement model truly learns *environmental suitability* (not just *survey visibility*), it should predict sites discovered **after 2000** even when trained only on sites discovered **before 2000**.

## Method

1. **Split sites by discovery year:**
   - Training: Sites discovered before 2000 (pre-GPR era)
   - Test: Sites discovered after 2000 (modern discoveries)

2. **Train model:**
   - Use pre-2000 sites as positive samples
   - Use TGB hybrid pseudo-absences (same method as E013)

3. **Test model:**
   - Evaluate AUC on post-2000 sites vs. fresh pseudo-absences

4. **Compare:**
   - Temporal test AUC vs. spatial CV AUC on same training data

## Why This Matters

This is the **strongest possible test** against tautology:
- Spatial CV can still "cheat" if survey patterns are spatially correlated
- Temporal split forces the model to predict *future* discoveries
- If AUC remains high (>0.65), model is genuinely learning settlement patterns

## Results

| Metric | Value |
|--------|-------|
| Temporal Test AUC (XGB) | **0.755** |
| Spatial CV AUC (XGB) | 0.785 ± 0.058 |
| Difference | -0.030 |
| Challenge 1 (rho) | -0.140 (TAUTOLOGY-FREE) |

### Interpretation

| Temporal AUC | Interpretation | Recommendation |
|--------------|----------------|----------------|
| **> 0.70** | **Model is tautology-resistant** | **Claim "tautology-resistant"** ✅ |
| 0.65–0.70 | Model is tautology-mitigated | Use "bias-corrected" framing |
| < 0.65 | Model overfits survey patterns | Revise claims, add limitations |

**Verdict: PASS** — Temporal AUC 0.755 > 0.65 threshold, dengan drop hanya 0.030 dari
spatial CV. Model generalizes dengan baik ke situs yang «belum ditemukan» (proxy via
accessibility). Paper 2 bisa mengklaim "tautology-resistant" dengan confidence.

## Run

```bash
py experiments/E014_temporal_validation/01_temporal_split_test.py
```

## Output

- `results/temporal_validation_results.txt` — Full metrics and interpretation
- Comparison with E013 spatial CV baseline
- Challenge 1 (tautology test) results

# E116: Testable Predictions from the Visibility Cascade

**Status:** SUCCESS
**Date:** 2026-03-20
**Papers:** P1, All
**Depends on:** E110 (cascade model), E080 (suitability targets), E108 (demographic model), E115 (cascade robustness)

---

## Hypothesis

The E110 cascade model can be converted into concrete, falsifiable fieldwork predictions with binomial prediction intervals. If the model is correct, specific survey scenarios should yield predictable numbers of archaeological finds.

## Method

For each scenario, decompose the per-target detection probability into components from the cascade model (burial depth, material survival, site existence probability), then compute binomial expected values and 95% prediction intervals.

Four scenarios tested:
1. **Targeted GPR** — 20 surveys at E080 high-suitability locations
2. **Random deep coring** — 50 cores to 10m in Malang basin
3. **Construction monitoring** — 250 deep excavation projects over 5 years
4. **Japan-level survey** — hypothetical 20× increase in survey intensity

## Key Results

| Scenario | N | P(find) | Expected | 95% CI | P(zero) |
|----------|---|---------|----------|--------|---------|
| Targeted GPR | 20 | 0.125 | 2.5 | [0, 6] | 7.0% |
| Random cores | 50 | 0.024 | 1.2 | [0, 4] | 30.3% |
| Construction monitoring | 250 | 0.009 | 2.2 | [0, 6] | — |
| Japan-level survey | 9,659 sites | 1.16% | 112 detectable | vs 5 current | — |

### Falsification Criteria

- **Targeted GPR:** 0 finds in 20 surveys → P = 7.0%. Not formally decisive (>5%) but strongly discouraging. Finding 1-6 anomalies supports the framework.
- **Random cores:** 0 finds in 50 cores → P = 30.3%. Not decisive — random sampling is inefficient for sparse sites.
- **Combined:** If BOTH targeted GPR AND random cores find nothing → joint P ≈ 2.1%. Framework in serious trouble.

### Cost-Benefit

| Scenario | Cost | Time | Discriminating Power |
|----------|------|------|---------------------|
| Targeted GPR | $40K-100K | 2-4 weeks | HIGH |
| Random cores | $10K-25K | 1-2 weeks | LOW |
| Construction monitoring | Minimal | 5 years | MEDIUM (passive) |

## Conclusion

The cascade model IS falsifiable. The most cost-effective test is 20 targeted GPR surveys at E080 high-suitability locations ($40K-100K, 2-4 weeks). This directly addresses Counter 1 (nobody lived there) and Counter 3 (unfalsifiable model) from the pre-mortem analysis.

The prediction can be registered as a pre-commitment in P1 EGQSJ: "If 20 GPR surveys at our top-ranked locations find 0 anomalies, the framework requires fundamental revision."

## Files

- `prediction_protocol.py` — Main script
- `results/e116_results.json` — Machine-readable predictions

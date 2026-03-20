# E118: Information Gain from Volcanic Context

**Status:** SUCCESS
**Date:** 2026-03-20
**Papers:** P1, P2, All
**Depends on:** E110 (cascade model), E108 (demographics), E080 (targets), E075 (burial model)

---

## Hypothesis

The strongest counter-argument to VOLCARCH (pre-mortem Counter 4): "Survey deficit has 40× leverage; volcanic burial only 1.7×. Indonesia just needs more archaeology, period."

We test whether volcanic context provides practical value beyond absolute leverage — specifically through spatial targeting and depth prediction.

## Method

Information-theoretic analysis:
1. Shannon entropy reduction from knowing volcanic context
2. Search efficiency comparison: random vs. volcanic-informed vs. oracle
3. Cost-effectiveness of targeted vs. random GPR deployment
4. Depth prediction advantage: which method to use at which distance

## Key Results

### Information Gain

| Metric | Value |
|--------|-------|
| H(find \| no context) | 0.610 bits |
| H(find \| volcanic context) | 0.433 bits |
| Information gain | 0.177 bits (29.0% entropy reduction) |

### Search Efficiency

Volcanic-informed search is **3.5× more efficient** than random across all budget levels. At a budget of 20 GPR surveys:
- Random: expect 3.0 finds
- Volcanic-targeted: expect 10.5 finds

### Cost-Effectiveness

| Goal | Random Cost | Volcanic Cost | Savings |
|------|------------|---------------|---------|
| First find | $23,333 | $6,667 | $16,667 (3.5×) |
| 5 finds | $116,667 | $33,333 | $83,333 (3.5×) |

### Depth Prediction (Unique Advantage)

Without VOLCARCH, you don't know HOW DEEP to survey. A shallow excavation at 5km from Kelud finds nothing (sites at 8m). VOLCARCH's burial model (r=0.951) tells you which method is needed at each location.

## Conclusion

**Survey deficit is the bigger PROBLEM. Volcanic context is the better SOLUTION.**

Counter 4 correctly identifies that Indonesia needs more archaeology regardless. But "do more archaeology" is not a prediction — it's a plea. VOLCARCH provides three things a generic "more archaeology" argument cannot:

1. **WHERE to dig** (3.5× efficiency gain)
2. **HOW DEEP to dig** (r=0.951 depth prediction)
3. **WHAT to expect** (E116: [0,6] anomalies at 20 locations)

This is science, not advocacy.

## Files

- `information_gain.py` — Main analysis
- `results/e118_results.json` — Machine-readable results

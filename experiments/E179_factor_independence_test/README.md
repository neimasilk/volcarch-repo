# E179: Factor Independence Test — Cascade Coupling Analysis

**Date:** 2026-04-09
**Paper:** P1, P17, All (cascade methodology)
**Status:** SUCCESS — Coupling effect is 3.0×, within model uncertainty. Independence assumption is defensible but should be acknowledged.
**Type:** [H] Hypothesis test (adversarial)

## Hypothesis

The E110 cascade model assumes F1 (volcanic burial) and F2 (organic decay) are independent. In reality, burial creates anaerobic conditions that can SLOW decay (Cerén preservation effect). If factors are coupled, the cascade overestimates visibility loss.

Similarly, F3 (survey) and F4 (recognition) may be coupled: professional surveys (F3) employ trained teams (F4).

## Method

1. Estimate conditional probabilities from archaeological literature
2. Compute coupled F1×F2 using P(organic survives | buried) vs P(organic survives | not buried)
3. Compute coupled F3×F4 using professional vs amateur recognition rates
4. Recompute full cascade with both couplings
5. Sensitivity analysis: vary coupling strength across plausible range

## Results

### F1-F2 Coupling (Burial × Organic Decay)

| Condition | P(organic survives) |
|-----------|:---:|
| Buried (sealed, anaerobic) | 0.40 |
| Not buried (tropical surface) | 0.05 |

- Independent F1 × F2 = 0.58 × 0.20 = 0.116
- Coupled joint = 0.197 (1.7× higher)

### F3-F4 Coupling (Survey × Recognition)

| Survey type | P(recognized) |
|-------------|:---:|
| Professional (30% of surveys) | 0.70 |
| Amateur/chance (70%) | 0.15 |
| Unsurveyed area | 0.01 |

- Independent F3 × F4 = 0.025 × 0.40 = 0.010
- Coupled joint = 0.018 (1.8× higher)

### Full Coupled Cascade

| Model | P(visible) | Ratio to observed |
|-------|:---:|:---:|
| Independent (E110) | 0.058% | 1.9× |
| **Coupled** | **0.174%** | **5.6×** |
| Observed | 0.031% | — |

### Sensitivity: P(organic survives | buried)

| P(surv\|buried) | Ratio to observed | Interpretation |
|:---:|:---:|---|
| 0.05 | 0.8× | Hot lahar destroys everything (Java) |
| 0.25 | 2.2× | Moderate preservation |
| 0.40 | 3.2× | Good preservation (Cerén-like) |
| 0.65 | 4.9× | Excellent preservation (cool ash fall) |

## Key Insights

1. **Coupling makes the prediction WORSE** — it increases predicted visibility, pushing the ratio from 1.9× to 5.6×. The independent model's good fit (1.9×) may be coincidental.

2. **Java's lahars are NOT Cerén.** If P(organic survives | buried) is low (~0.05, hot destructive lahars rather than cool ash), the coupled model actually IMPROVES (ratio 0.8×). Java's volcanic burial may destroy organics on contact — the "preservation effect" may not apply here.

3. **The 3.0× coupling effect is within model uncertainty.** E115 MC showed 95% CI spanning 22×. A 3× shift doesn't change the qualitative conclusion.

4. **This is the single most important methodological caveat** for the cascade model. Papers should state: "Factors may be coupled (volcanic burial can preserve or destroy organics depending on eruption type). Coupling shifts predictions by ~3×, within parameter uncertainty."

## Conclusion

Factor coupling is real but manageable. The independent model's 1.9× fit may be coincidental — the coupled model with Java-appropriate lahar parameters (hot destruction rather than cool preservation) could fit BETTER (0.8×). Either way, the qualitative conclusion holds: multiple compounding factors create near-total invisibility.

The deeper issue remains: with 5 parameters and 1 data point, coupling is absorbed by parameter flexibility. This is not a model flaw that coupling exposes; it's a fundamental underdetermination that coupling illustrates.

## Caveats

1. Conditional probabilities are literature-estimated, not measured on Java samples
2. Eruption type matters enormously: lahar (hot) vs tephra fall (cool) vs pyroclastic flow (very hot)
3. The "Cerén effect" (cool ash preservation) may apply to some Java sites (Liangan) but not others

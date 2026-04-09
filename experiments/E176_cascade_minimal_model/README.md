# E176: Cascade Minimal Model Comparison

**Date:** 2026-04-09
**Paper:** P1, P17, All (cascade reframing)
**Status:** SUCCESS — Exposes 5-factor model as over-parameterized. 3 factors sufficient.
**Type:** [H] Hypothesis test (adversarial, self-critique)

## Hypothesis

The 5-factor cascade (E110) is over-parameterized. A simpler 3-factor model brackets the same observations, making the 5-factor decomposition pedagogically useful but scientifically unnecessary.

## Method

1. Enumerate all 31 possible subset models (1-5 factors from the E110 cascade)
2. For each, compute P(visible) at best/low/high parameter estimates
3. Test which models bracket the observed 0.031% visibility
4. Compute AIC-like parsimony metric: k + 2×|log(ratio)|
5. Monte Carlo: draw 100K random parameter values from established ranges; what fraction bracket within 10×?

## Results

### Key Findings

| N factors | Models tested | Bracket observed? | Best ratio |
|:---------:|:------------:|:-----------------:|:----------:|
| 1 | 5 | 0/5 (0%) | 80.6× (Survey) |
| 2 | 10 | 0/10 (0%) | 16.1× (Organic+Survey) |
| **3** | **10** | **5/10 (50%)** | **6.5× (Organic+Survey+Recognition)** |
| 4 | 5 | 4/5 (80%) | 3.2× |
| 5 | 1 | 1/1 (100%) | 1.9× |

### Monte Carlo: Random Draws Within 10× of Observed

| N factors | % within 10× |
|:---------:|:------------:|
| 1 | 0.0% |
| 2 | 1.6% |
| 3 | 11.1% |
| 4 | 39.6% |
| **5** | **83.8%** |

### Parsimony (AIC analog)

| N | Best Model | AIC |
|---|-----------|:---:|
| 3 | Organic+Survey+Recognition | 6.73 |
| 4 | Organic+Survey+Recognition+Publication | 6.34 |
| 5 | All five | 6.25 |

AIC barely improves from 3→5 factors (6.73→6.25). The additional 2 parameters are not justified by improved fit.

### Minimal Bracketing Models (3 factors)

1. **Organic+Survey+Recognition** — ratio 6.5×, range [0.01%, 2.8%] — BEST
2. Volcanic+Organic+Survey — ratio 9.4×, range [0.015%, 3.4%]
3. Organic+Survey+Publication — ratio 8.1×, range [0.0125%, 3.2%]
4. Volcanic+Survey+Recognition — ratio 18.7×, range [0.03%, 5.95%]
5. Survey+Recognition+Publication — ratio 16.1×, range [0.025%, 5.6%]

Note: F3 (Survey) appears in ALL 5 minimal models. It is structurally necessary.

### Factor Hierarchy

1. **F3 Survey Coverage (40×)** — structurally necessary, appears in all bracketing models
2. **F2 Organic Decay (5×)** — appears in 3/5 minimal models
3. **F4 Recognition (2.5×)** — appears in 3/5 minimal models
4. F5 Publication (2×) — appears in 2/5 minimal models
5. **F1 Volcanic Burial (1.7×)** — appears in 2/5 minimal models, the LEAST necessary factor

## Conclusion

**The 5-factor cascade is OVER-PARAMETERIZED.** Three factors (typically Survey + Organic + Recognition) bracket the observed gap. The full 5-factor model achieves a tighter best-estimate ratio (1.9× vs 6.5×) but this precision is cosmetic: 83.8% of random 5-factor draws also bracket within 10×.

**Critical implication for VOLCARCH framing:** Volcanic burial (F1) is the LEAST necessary factor in the cascade. It appears in only 2/5 minimal bracketing models. The project's unique contribution is NOT that volcanic burial is the dominant cause of invisibility — it's that volcanic burial is the **only spatially predictable** factor, enabling targeted fieldwork.

**Honest reframing for papers:**
> "The observed demographic gap is primarily driven by survey coverage deficit (40× leverage) and organic material decay (5× leverage). Volcanic burial adds a computationally predictable spatial component (1.7× leverage) that enables targeted recovery. The 5-factor decomposition is pedagogically useful but empirically underdetermined."

## Implications

1. Papers should stop claiming the cascade "matches data" — any 3+ factor model with plausible ranges does
2. The West Java smoking gun (E110) remains strong — it's an OBSERVATION, not a model
3. F3 (survey) being structurally necessary reinforces the ADV-1 Japan finding (E086)
4. VOLCARCH's unique value proposition = spatial prediction of burial zones, NOT the cascade arithmetic

## Caveats

1. "Bracketing" = observed falls within [low, high] range. This is a weak test.
2. Factor ranges were estimated by literature review, not measured directly.
3. Independence of factors not tested here (see E179).

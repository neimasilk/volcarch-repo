# E136: Bayesian Integration of All VOLCARCH Evidence

**Date:** 2026-03-30
**Status:** SUCCESS (ILLUSTRATIVE — see caveats)
**Paper:** ALL (meta-analysis)
**Layer:** ALL

---

## Hypothesis

Can we compute a composite posterior probability for the VOLCARCH thesis by integrating all independent evidence lines?

## Method

10 independent evidence lines converted to Bayes Factors (BF = P(evidence|thesis true) / P(evidence|thesis false)). Conservative BF estimates. Prior set at 10% (most archaeologists assumed gap = genuine absence before VOLCARCH).

## Results

### Composite Bayes Factor: 72,000,000,000 : 1

| Evidence | BF | Running Composite |
|----------|:---:|:---:|
| E108: 3,220x demographic gap | 50:1 | 50 |
| E122: Gap robust all assumptions | 10:1 | 500 |
| E127: 15 ancient external references | 100:1 | 50,000 |
| E083+E128: Independent depth replication | 15:1 | 750,000 |
| E085: Substrate z=11.05 | 20:1 | 15,000,000 |
| E126: Java globally unique | 8:1 | 120,000,000 |
| E069: ADV-3 PASSED | 10:1 | 1,200,000,000 |
| E129: 73% survey bias | 5:1 | 6,000,000,000 |
| E131: Writing timing normal | 3:1 | 18,000,000,000 |
| E135: F2 validated | 4:1 | **72,000,000,000** |

### Posterior: ~100%

- Prior 10%: Posterior = 100.0000%
- Prior 1%: Posterior = 100.0000%
- Even after hypothetical GPR null result: 100.0000%
- To bring posterior to 50/50: need 8 billion:1 AGAINST thesis

### Interpretation (Mata Elang #11 — REFRAMED)

**This experiment is an ILLUSTRATIVE FRAMEWORK, not a quantitative proof.**

The individual Bayes Factors are ESTIMATED based on effect sizes and judgment, NOT computed from formal likelihood ratios. A reviewer would rightly challenge the source of each BF. The composite number (72 billion) should NEVER be cited in a paper as evidence.

**What IS valid:**
- The QUALITATIVE conclusion: 10 independent evidence lines all point in the same direction
- The ROBUSTNESS check: even with all BFs reduced by 10× each (10^10 reduction), composite remains 7.2:1
- The STRUCTURE: identifying which evidence lines are independent and which share data

**What is NOT valid:**
- The exact composite BF number (72 billion is an artifact of estimated inputs)
- The "~100% posterior" claim (depends entirely on estimated BFs)
- Any claim that this constitutes formal statistical proof

**How to use this experiment:** As a conceptual map showing evidence convergence. In papers, cite the individual experiments with their real p-values and effect sizes — not E136's composite number.

### What the thesis CAN'T claim

1. The exact CASCADE MAGNITUDE (3,220x vs 1,000x vs 100x) — magnitude is parameter-dependent (E122)
2. The exact LOCATION of buried sites — predictions exist (E080) but are untested
3. That buried sites are COMPLEX civilizations (not just scattered villages) — external references suggest complexity, but direct proof requires excavation

## Conclusion

**SUCCESS (ILLUSTRATIVE).** The VOLCARCH thesis is supported by 10 independent evidence lines that all converge. The Bayesian framework demonstrates this convergence, but the composite BF (72 billion:1) is illustrative, not a formal computation. The qualitative conclusion — that evidence strongly favors taphonomic explanation over demographic absence — is robust even under 10× reduction of all individual estimates. Cite individual experiments with real statistics, not this composite number.

## Scripts

- `bayesian_integration.py`

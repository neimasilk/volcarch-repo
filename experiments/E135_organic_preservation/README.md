# E135: Organic Material Preservation Model

**Date:** 2026-03-30
**Status:** SUCCESS
**Paper:** P1, P19
**Layer:** L1 (cascade factor F2)

---

## Hypothesis

Material-specific decomposition rates can independently estimate cascade factor F2 (organic decay), validating the E110 estimate of P=0.20.

## Method

Modeled 7 material types (bamboo, lontar, wood, thatch, cloth, bone, stone) with half-lives in volcanic soil. Weighted by E040 inscription mention frequency to compute aggregate survival probability.

## Results

### F2 Independently Validated

| Period | Weighted Survival | F2 estimate |
|--------|:---:|:---:|
| 100 yr (colonial) | 42.4% | 0.42 |
| 500 yr (Majapahit) | 24.9% | 0.25 |
| 1000 yr (Mataram) | 23.3% | 0.23 |
| **1600 yr (pre-400 CE)** | **22.9%** | **0.229** |

**E110 estimate: 0.20. Model prediction: 0.229. CONSISTENT.**

### Material Survival at 1600 Years

| Material | Volcanic soil | Normal soil | Ratio |
|----------|:---:|:---:|:---:|
| Stone | 99.8% | 99.8% | 1.0x |
| Bone | 33.0% | 0.4% | 84x better in volcanic |
| Hardwood | 0.06% | ~0% | Much better in volcanic |
| Bamboo/lontar | ~0% | ~0% | Both destroyed |

**Volcanic soil preserves BETTER than normal aerobic soil** (sealing under ash reduces oxygen). But at 1600 years, only stone and bone survive significantly.

## Conclusion

**SUCCESS.** Independent material science derivation confirms E110's F2 parameter within 15%. Key insight: volcanic burial is NOT purely destructive — it preserves through anaerobic sealing. But organic materials (bamboo, lontar, thatch) are destroyed regardless of burial condition at multi-century timescales. This explains: (1) why candi survive, (2) why Liangan is exceptional (quick ash burial), (3) why pre-Hindu organic settlements are invisible.

## Scripts

- `organic_preservation.py`

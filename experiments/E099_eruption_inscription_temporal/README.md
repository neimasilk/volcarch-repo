# E099 — Eruption Frequency x Inscription Visibility Gradient

**Status:** INCONCLUSIVE (data-limited, suggestive signal at decade resolution)
**Date:** 2026-03-17
**Layer:** L6 (Periodicity) + L1 (Volcanic Burial)
**Papers:** P5, P11 revision ammo
**Experiment #100** in the VOLCARCH series

---

## Hypothesis

If volcanic eruptions suppress inscription production (by destroying infrastructure, disrupting courts, displacing populations), eruption frequency should anti-correlate with inscription density at temporal resolutions from decades to centuries.

## Method

Cross-tabulated 13 GVP-dated eruptions (0-1500 CE, East Java volcanoes) against 165 DHARMA dated inscriptions (600-1500 CE). Tested at three resolutions: century, 50-year bins, and decade bins with lag analysis.

## Results

### Century-level: WEAK (data-limited)
- Spearman rho = **-0.289**, p = 0.451 (NOT significant)
- Only 13 eruptions across 9 centuries — too sparse for century resolution
- Direction is negative (as predicted) but not statistically reliable

### 50-year bins: WEAK
- Spearman rho = **-0.269**, p = 0.280 (NOT significant)
- 18 bins, same direction but insufficient power

### Decade-level: SUGGESTIVE SIGNAL
- **Lag 0: rho = -0.260, p = 0.013** (significant at alpha=0.05)
- **Lag 1: rho = -0.264, p = 0.012** (significant)
- Eruption decades and the following decade show suppressed inscription production
- Lag 2-5: not significant (effect dissipates after ~20 years)

### Quiet vs Active periods
- Quiet periods (>20yr without eruptions): **7.7 inscriptions/century**
- Active periods: **32.1 inscriptions/century**
- Ratio: **0.24x** — quiet periods produce 4x fewer inscriptions

**Interpretation:** This is COUNTERINTUITIVE and likely confounded. The "quiet periods" are concentrated in C12-C13 (post-Mataram collapse), when inscription production dropped for POLITICAL reasons (E096: 929 CE discursive shift), not volcanic reasons. The eruption clustering in C14-C15 (Kelud) coincides with a period of already-low inscription production (Majapahit era, different epigraphic conventions).

## Honest Assessment

**The GVP dataset is too sparse for this analysis.** Only 13 dated eruptions for 4 volcanoes over 1500 years — a known limitation of the GVP for Indonesian volcanoes before the colonial period. The decade-level signal (p=0.013) is suggestive but could be driven by the C14-C15 Kelud eruption cluster coinciding with the end of classical Javanese epigraphy for independent reasons.

**To make this conclusive:** Need either (1) better eruption chronology (tephra core dating, not GVP-only), or (2) a Merapi-specific analysis with higher eruption resolution (Newhall et al. 2000 lists many more Merapi events than GVP).

## Status

**INCONCLUSIVE** — Direction is as predicted (negative correlation), decade-level signal is statistically significant (p=0.013), but confounded by data sparsity and political factors. Honest result: the hypothesis is plausible but not demonstrated with current data.

This is experiment **#100** — the 100th VOLCARCH experiment (including E095 activation).

## Output

- `results/e099_results.json`

# P1 Revision Support Material: E110 Visibility Cascade — Why Survey Deficit Is #1

**Paper:** Asian Perspectives MS# 019A-0326
**Date:** 2026-03-17
**Severity:** HIGH — reframes the entire argument; use proactively
**New since submission:** YES

---

## The Argument

P1 as submitted frames volcanic burial as the primary explanation. Post-submission analysis (E110) reveals a more nuanced picture: volcanic burial is ONE of FIVE multiplicative factors, and NOT the most impactful one.

## The Cascade Model

```
P(visible) = P(not_buried) × P(not_decayed) × P(surveyed) × P(recognized) × P(published)
           = 0.58 × 0.20 × 0.025 × 0.40 × 0.50
           = 0.058%
```

Observed rate (E108): 3/9,659 = 0.031%. **Model matches data within 2×.**

## Sensitivity Ranking

| # | Factor | Leverage | If fixed alone |
|---|--------|:---:|---|
| **1** | **Survey coverage (0.025)** | **40×** | **Most impactful intervention** |
| 2 | Organic decay (0.20) | 5× | Irreversible |
| 3 | Recognition bias (0.40) | 2.5× | Training + dating methods |
| 4 | Publication barrier (0.50) | 2× | Institutional |
| **5** | **Volcanic burial (0.58)** | **1.7×** | **Only spatially predictable** |

## Key Reframe for Reviewers

**OLD framing:** "Volcanic burial hides ancient sites"
**NEW framing:** "Five compounding factors create near-total invisibility. Survey deficit is the primary constraint (40× leverage). Volcanic burial is the computationally predictable factor that enables prioritized recovery."

## The West Java Decisive Case

- **Buni Complex** (Tangerang coast, 200 BCE–500 CE): extensive pottery, beads, metalwork — NON-VOLCANIC
- **Batujaya** (Karawang, 2nd–5th century CE): Buddhist brick complex — NON-VOLCANIC
- **East Java volcanic interior**: ZERO pre-400 CE sites

Same island. Same culture. Same timeframe. Different geology.

## Suggested Paragraph

> "Post-submission modeling reveals that volcanic burial operates within a multiplicative cascade of five independent factors. A five-factor visibility model — combining volcanic burial (P=0.58), organic material decay (P=0.20), archaeological survey coverage (P=0.025), recognition probability (P=0.40), and publication probability (P=0.50) — predicts 0.058% archaeological visibility, matching the observed 0.031% gap between modeled and actual site counts. Sensitivity analysis identifies survey coverage as the highest-leverage intervention (40×), with volcanic burial contributing 1.7× as an individual factor but serving as the only factor that generates spatially predictable fieldwork candidates. Within-island comparison supports this model: non-volcanic coastal West Java (Buni Complex, Batujaya) preserves rich pre-400 CE archaeology, while the volcanic interior preserves none."

## Supporting Data

- `experiments/E110_visibility_cascade/results/e110_results.json`
- `experiments/E108_demographic_null_model/results/e108_results.json`

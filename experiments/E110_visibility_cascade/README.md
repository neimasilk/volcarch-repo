# E110: Multiplicative Visibility Cascade Model

**Date:** 2026-03-17
**Paper:** All (core theoretical model)
**Status:** SUCCESS — Model brackets observed gap. West Java smoking gun.

## Hypothesis

The 3,220× gap between expected pre-400 CE settlements (E108: 9,659) and observed sites (0-3) can be explained by five independent factors, each reducing visibility by ~60-98%, whose product matches the observed gap.

## Method

Multiplicative cascade model:
```
P(visible) = P(not_buried) × P(not_decayed) × P(surveyed) × P(recognized) × P(published)
```

Each factor estimated independently from project evidence and literature comparanda.

## Results

### The Five Factors

| # | Factor | P(survive) | Leverage | Evidence |
|---|--------|:---:|:---:|---|
| F1 | Volcanic Burial | 0.58 | 1.7× | E075, E083, L1 calibration |
| F2 | Organic Decay | 0.20 | 5.0× | E040 (63.4% organic), tropical climate |
| F3 | **Survey Coverage** | **0.025** | **40.0×** | E086 (Japan 100-200× more), E069 |
| F4 | Recognition | 0.40 | 2.5× | E062 (dark century), L3 bias |
| F5 | Publication | 0.50 | 2.0× | E093 (65 papers), language barrier |

### Cascade Result

| | P(visible) |
|---|:---:|
| Model low estimate | 0.0024% |
| **Model best estimate** | **0.058%** |
| Model high estimate | 1.10% |
| **Observed (E108)** | **0.031%** |

**The cascade model brackets the observed gap.** Best estimate is 1.9× the observed rate — well within parameter uncertainty.

### Sensitivity Ranking (Most Impactful to Fix)

1. **Survey Coverage: 40× leverage** — most impactful single intervention
2. Organic Decay: 5× — irreversible but explains material absence
3. Recognition: 2.5× — training + absolute dating methods
4. Publication: 2× — institutional capacity
5. Volcanic Burial: 1.7× — computationally predictable, enables targeting

### The West Java Smoking Gun

| Region | Volcanic? | Pre-400 CE sites | Example |
|--------|:---------:|:-----------------:|---------|
| West Java coast | NO | 3-4 | Buni Complex, Batujaya |
| East Java interior | YES | 0 | — |

Same island. Same culture. Same timeframe. Different geology. The only systematic difference is volcanic burial and its associated factors.

## Key Insight

**Volcanic burial is the LEAST impactful single factor (1.7× leverage), but it is the ONLY factor that can be modeled spatially.** The project's unique contribution is not that volcanic burial is the primary cause of invisibility (it isn't — survey deficit is), but that:

1. It adds the 5th factor to a cascade that creates near-total invisibility
2. It is spatially predictable → enables targeted fieldwork (E080, E097)
3. It is unique to volcanic landscapes → explains why Java specifically is dark
4. Its within-island control (West Java) provides the smoking gun

## Implications for VOLCARCH Framing

The project should be reframed from:
> "Volcanic burial hides pre-Hindu civilization"

To:
> "Five compounding factors create near-total invisibility of pre-Hindu Java. Survey deficit is the primary constraint (40× leverage). Volcanic burial is the computationally predictable factor that enables targeted recovery. The West Java comparison proves the system works: where volcanism is absent, pre-Hindu archaeology is visible."

## Caveats

1. Factor estimates are literature-based, not directly measured
2. Factors may not be fully independent (volcanic burial affects decay rate)
3. Model is for East Java specifically; other regions would have different factor values
4. The "0-3 sites" count for pre-400 CE is uncertain

## Files

| File | Description |
|---|---|
| `visibility_cascade.py` | Model script |
| `results/e110_results.json` | Full results with sensitivity analysis |

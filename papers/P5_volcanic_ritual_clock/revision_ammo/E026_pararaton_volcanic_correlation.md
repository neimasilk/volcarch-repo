# E026 Pararaton Volcanic Correlation — Revision Ammo for P5

**Source:** P14 (killed 2026-03-11, Mata Elang #4). E026 results folded here.
**IDEA_REGISTRY:** I-097

## Key Finding

Pararaton narrative events cluster near volcanic eruptions:
- **Proximity test:** 6/6 political events within 5 years of GVP-confirmed eruption (p=0.037, Fisher's exact)
- **Rate ratio:** 2.18× higher political event rate within ±5yr eruption windows vs. outside
- **GVP match:** 3/3 named eruptions in Pararaton confirmed in GVP database

## Caveats

- **Bonferroni correction kills significance:** adj. p=0.222 (6 tests). Does NOT survive multiple comparison correction.
- **Poisson rate test:** p=0.255, not significant. Clustering is suggestive, not conclusive.
- Small sample (6 events, ~350 year period).
- Pararaton is a literary text, not a contemporary chronicle — events may be telescoped.

## How to Use in P5 Revision

If BKI reviewers ask "is there independent evidence that volcanic events shaped Javanese political narratives?":
- Cite E026 as **exploratory supporting evidence** (NOT as proof)
- Frame: "a suggestive pattern consistent with volcanic disruption driving political instability, though sample size precludes definitive statistical claims"
- Complements P5's main argument (ritual timing encodes taphonomic knowledge) by showing volcanic influence extends to political chronology
- Do NOT overclaim — the Bonferroni failure is real

## Data

- Experiment: `experiments/E026_pararaton_volcanic_correlation/`
- Key file: `experiments/E026_pararaton_volcanic_correlation/results/` (if exists)
- GVP eruption data: `data/raw/` (Kelud, Semeru records)

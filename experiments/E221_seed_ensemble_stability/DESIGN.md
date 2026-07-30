# E221 — Design (pre-registration)

**Written 2026-07-27, before running.** Requested by the co-author review
(`papers/P2_settlement_model/review_package_20260727/05_REVIEW_COAUTHOR_GO_FRENDI.md` §8). Two purposes:
turn E219's seed-instability finding into an actionable protocol (relevance for heritage management —
R1-F/R2-B), and produce the map products the revised figures need (handoff block F).

E219 compared maps but did not store them, and used 5 seeds. E221 re-produces the surfaces at **10 seeds ×
3 designs × 3 algorithms** (90 surfaces, 588,535 cells each) and stores them, enabling ensemble analyses
E219 could not do.

## Part A — Stabilisation curve: how many seeds before the priority tier stops moving?

Per (algorithm, design): ensemble-of-k top-decile Jaccard against ensemble-of-10, averaged over 50 random
subsets per k = 1…9. Report k* = smallest k with mean J ≥ 0.9, plus the single-run baseline (one seed vs
the 10-seed ensemble).

**Pre-registered readings:**

| Outcome | Consequence |
|---|---|
| k* ≤ 6 for all 9 combos | Protocol recommendation: "ensemble ≥ k* seeds" enters the corrected protocol; headline instability finding stands AND has a cheap fix |
| k* > 6 for some combo | Recommendation becomes algorithm-specific; the instability finding gets sharper, not weaker — report which model family is least stable |
| Between-design ensemble divergence disappears under ensembling | Part of E219's design effect was seed noise amplified; the manuscript's map-divergence claim is downgraded to match |

The third row is the falsification branch: if ensembling erases the between-design gap, we say so.

## Part B — Robust vs contingent priority sets (the archaeological deliverable)

Per algorithm, from the three design ensembles' top-decile sets:

- **robust priority** — top-decile under ALL three background designs;
- **design-contingent** — top-decile under exactly one design;
- the rest.

Characterise each set on: road distance, elevation, distance to nearest canonical volcano (13-inventory),
distance to nearest known site, area, and count of known sites falling inside (nearest frame cell). Export
per-cell products (npz): x, y, suitability per design ensemble, and the 0–3 divergence count — the raw
material for the divergence-map figure.

**Pre-registered readings:**

| Outcome | Consequence |
|---|---|
| Robust set holds higher known-site density than contingent sets | Consistency (not validation — sites trained the models): the stable core is where the record lives; the contingent fringe is where analytic choices send people differently. Heritage paragraph writes itself |
| Robust set does NOT differ | The design divergence is geographically unstructured; the heritage framing narrows to "pick a design and say why" |
| Contingent sets cluster by road/elevation/volcano profile | Descriptive, no committed direction — report the profiles |

This is the artefact-era version of Reviewer 2's two-stage ask (R2-C): not "where are buried sites" but
"which survey priorities survive arbitrary analytic choices, and which are artefacts of them".

## Part C — Seed turnover, both definitions, precisely

From the 45 within-design seed pairs per (algorithm, design): Jaccard J; report **1 − J** (share of the
combined footprint flagged by only one run) and **(1 − J)/(1 + J)** (share of one run's top decile
replaced). The v0.2 text must pick one definition and say which (co-author review §7).

## Sequencing

One script, ~50 min (90 fits + full-frame predictions, E219 machinery). Maps persist to
`results/maps/*.npy`; summary products to `results/`; verdicts to `results/e221_outcome.json`.

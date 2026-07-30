# E221 — Seed-Ensemble Stabilisation + Robust/Contingent Priority Sets

**Status:** SUCCESS | **Date:** 2026-07-27 | **Pre-registration:** `DESIGN.md` (written before running)
**Commissioned by:** co-author review `papers/P2_settlement_model/review_package_20260727/05_REVIEW_COAUTHOR_GO_FRENDI.md` §8
**Scripts:** `01_seed_stability.py` (~50 min; produces and stores 90 full-frame surfaces),
`02_split_half_control.py` (post-hoc matched control from stored surfaces, no new fits)

## Hypothesis

E219 found the survey-priority map moves between background designs and between seeds, but stored no maps
and used 5 seeds. Two questions decide whether that finding is actionable: (A) how many seeds until the
priority tier stops moving — i.e. is there a cheap fix; (B) which priorities survive arbitrary analytic
choices at all — the artefact-era version of Reviewer 2's two-stage ask.

## Method

10 seeds × 3 designs × 3 algorithms; models fit on all presences (map production, E219 machinery),
surfaces stored to `results/maps/`. Part A: ensemble-of-k vs ensemble-of-10 top-decile Jaccard, 50
subsets per k. Part B: per algorithm, partition the frame into **robust** (top-decile under all three
designs), **design-contingent** (top-decile under exactly one), and the rest; characterise on road
distance, elevation, canonical volcano distance (13-inventory), distance to known sites, and known-site
enrichment. Part C: within-design seed turnover under both definitions (1−J and (1−J)/(1+J)).

## Results

**Part A — the instability has a cheap fix, and it is algorithm-specific.**
k* (smallest k with ensemble-vs-ensemble J ≥ 0.9): **4 for RandomForest and MaxEnt, 7 for XGBoost**, all
designs. Protocol floor = **7 seeds**. A single run agrees with the 10-seed ensemble at only J = 0.65–0.68
(XGB), 0.77–0.80 (RF), 0.78–0.79 (MaxEnt): one-seed maps are not publication-grade artefacts; a 7-seed
ensemble stabilises all three families.

**Design divergence survives ensembling (matched control, `02`).** The split-half (5+5 seeds) same-design
noise floor is 0.75–0.87; hybrid-vs-anything ensembles sit far below it (MaxEnt 0.41–0.43, XGB 0.61–0.62,
RF 0.72–0.73), while random↔tgb sits at the floor. **Hybrid is what moves the map, and it is not seed
noise.** (Script 01's first-pass verdict used a mismatched single-run reference and said otherwise for
XGB/RF; `02_split_half_control.py` supersedes it — see that file's header.)

**Part B — the stable core is where the record lives; the contingent fringe is remote upland.**
Known-site density, robust vs contingent: XGB **40.8 vs 9.4**, RF **30.7 vs 15.9**, MaxEnt **31.7 vs 5.7**
sites/1000 km² (consistency check, not validation — sites trained the models). The MaxEnt contingent
fringe is distinctive: median 1.0 km from roads (vs 0.12 km robust), 1,107 m elevation (vs 533 m),
11.7 km from a volcano. Reading for the heritage paragraph: designs agree on the accessible lowlands that
hold the known record; they disagree precisely over the remote uplands that different bias assumptions
promote or demote — which is where a survey budget would be sent differently.

**Part C — turnover, both definitions (10 seeds, 45 pairs per combo).**
Share of combined footprint flagged by only one run (1−J): XGB 43–47%, RF 28–33%, MaxEnt 29–31%.
Share of one run's top decile replaced ((1−J)/(1+J)): XGB 28–31%, RF 16–20%, MaxEnt 17–19%.
**The v0.2 text must state which definition it uses** (co-author review §7); the submitted-paper-era
"31–45%" was the first definition.

## Conclusion

E219's two map findings both sharpen. (1) Seed instability is real but fixable: ensemble ≥ 7 seeds (≥ 4 if
the model is RF/MaxEnt) — a concrete, citable protocol recommendation. (2) The design effect is not
sampling noise: hybrid backgrounds move the priority tier beyond the matched ensemble floor in all three
algorithms, and the cells at stake are exactly the remote uplands a heritage authority would survey
differently. Together with E220, the revision now has its decision-level message: *the evaluation choice
changes where people are sent; the fix is a declared evaluation availability, a fixed common background,
and a seed ensemble.*

## Files

- `results/maps/{algo}_{design}_seed{0..9}.npy` — 90 stored surfaces (588,535 cells, float32)
- `results/e221_stabilisation_curve.csv` — ensemble-of-k curves + single-run baseline
- `results/e221_ensemble_between_design.csv` / `e221_split_half_control.csv` + `e221_outcome_split_half.json`
- `results/e221_priority_sets.csv` + `e221_priority_sets_{algo}.npz` — robust/contingent products
  (per-cell suitability per design ensemble + 0–3 divergence count; raw material for the divergence-map figure)
- `results/e221_turnover_pairs.csv` — 45 pairs per combo, both turnover definitions
- `results/e221_outcome.json`

# E217 — MaxEnt Benchmark Across the Pseudo-Absence Ladder

**Status:** PRE-REGISTERED (decision rules below locked before the first run, 2026-07-27)
**Driver:** JCAA #280 revise-and-resubmit, **Reviewer 1** (decision 2026-07-23)
**Paper:** P2 — `papers/P2_settlement_model/`
**Plan:** `papers/P2_settlement_model/revision_ammo/JCAA_R1_RESPONSE_PLAN_20260727.md` (item R1-D)

---

## The reviewer's ask

> "as the main topic is around the presence-only modeling, the reason for not using, comparing to, or at
> least relating to the similar approach through the Maximum Entropy functions is problematic. Maxent is
> indirectly referred to in pointing to (Phillips et al., 2006) for the experimental design. But why not use
> Maxent to evaluate the results? Or at the very minimum explain the reasoning for not using Maxent, and
> what this method does that Maxent does not."

Reviewer 1 raised this twice (§4 Research design, §9 Additional feedback) and called relating to MaxEnt
"essential". It is the single hard scientific ask in that review.

## Hypothesis

P2's central claim is that **pseudo-absence realism dominates feature accumulation** as the lever on spatial
transfer under survey-biased archaeological data. That claim was established using boosted trees only.

**H1:** The monotonic gain across the background ladder (random → target-group → hybrid) is
**algorithm-independent** — it reproduces under MaxEnt, which is the canonical presence-background method
and the one the bias-correction literature (Phillips et al. 2009) was developed around.

**H0:** The gain is specific to tree ensembles, i.e. an artefact of the learner rather than a property of
the background design.

If H1 holds, the paper's claim becomes **stronger** than in the submitted version, not weaker: an effect
that survives a change of model family is a property of the data design, which is exactly what the paper
argues.

## Method

Factorial benchmark under **one** set of deterministic spatial-block CV folds
(5 folds, 0.45° ≈ 50 km blocks — identical to E013):

| Factor | Levels |
|---|---|
| Background design | `random` (E007/E008 design), `tgb` (E010–E012 design), `hybrid` (E013 design) |
| Feature set | `terrain` (elevation, slope, TWI, TRI, aspect), `terrain_river` (+ river distance) |
| Algorithm | `maxent` (elapid/maxnet, linear+hinge+product, β=1.5, cloglog), `xgboost`, `randomforest` |
| Seeds | 5 background-sampling seeds |

XGBoost and RandomForest hyperparameters are copied unchanged from E013. Pseudo-absence ratio 5:1.
The `terrain` vs `terrain_river` contrast measures the **feature** effect on the same folds that the
background contrast measures the **background** effect — so the two levers are compared like for like.

**Documented deviation from E013:** backgrounds are drawn from a 10×-decimated raster lattice (~300 m
spacing) shared by all three designs, rather than by continuous-coordinate rejection sampling. This is what
makes the designs directly comparable. Absolute AUCs may therefore differ slightly from the published
E007–E013 values; the quantity of interest is the **within-E217 contrast**, which is what Reviewer 1 asked
about. This deviation is disclosed in the paper, not silently absorbed.

## Pre-registered decision rules (locked before running)

| Outcome | Reading | What goes in the revision |
|---|---|---|
| MaxEnt ladder is monotonic and gain ≈ tree gain | **H1 supported** | Report as algorithm-independence; strengthens the central claim; MaxEnt becomes a supporting benchmark, not a rival. |
| MaxEnt ladder is monotonic but gain much smaller | Partial H1 | Claim holds directionally; state the magnitude is learner-dependent. |
| MaxEnt ladder is **not** monotonic | **H0 — claim is learner-specific** | Downgrade the central claim in the abstract and conclusions to "under tree ensembles"; report the MaxEnt failure explicitly. |
| **MaxEnt matches or beats XGBoost at the hybrid design** | Algorithm choice is not a performance win | Report plainly. Reframe the XGBoost choice as interpretability/SHAP convenience, **not** performance. This outcome does not damage the paper's thesis; concealing it would. |

The last row is recorded here in advance precisely because it is the outcome least convenient to report.

## Data

- Presences: `data/processed/east_java_sites.geojson`, clipped to 111–115°E / 9–6.5°S (as E013).
- Covariates: `data/processed/dem/` — Copernicus GLO-30 DEM and derivatives, OSM-derived river/road distance.
- No volcanic predictors are used, in either training or background construction (unchanged from the paper).

## Result

**Status: SUCCESS as an experiment — and it REFUTES P2's central claim.**

### Step 0 — the reimplementation reproduces the published pipeline

Before anything is concluded, the pipeline is validated against the submitted manuscript:

| Quantity | Published P2 | E217 reimplementation |
|---|---|---|
| E013 hybrid, seed-averaged XGBoost AUC (own background) | **0.751** | **0.750** |
| E007 terrain-only random background, XGBoost AUC | 0.659 | 0.670 |
| Realised hard-negative fraction (zdist ≥ 2) in hybrid design | **0.62** | **0.623** |

The third row matters most: the manuscript flagged the 0.62 realised hard fraction as an unexplained
anomaly ("This pool composition effect should be considered when interpreting the absolute AUC values",
Methods §2.4). An independent reimplementation lands on the same idiosyncratic value. Whatever follows is
therefore a property of the paper's design, not of a divergent implementation.

### Step 1 — run 01 (each design scored on its own background)

| Background | terrain | terrain+river |
|---|---|---|
| | maxent / RF / XGB | maxent / RF / XGB |
| random | 0.621 / 0.667 / 0.669 | 0.692 / 0.718 / 0.719 |
| tgb | 0.624 / 0.671 / 0.666 | 0.688 / 0.710 / 0.705 |
| hybrid | 0.675 / 0.712 / 0.714 | 0.715 / 0.739 / 0.742 |

No algorithm — MaxEnt included — produced a monotonic random → tgb → hybrid ladder, and the background
gain (+0.022) was already smaller than the gain from adding a single feature (+0.045).

### Step 2 — run 02 (all designs scored on ONE common evaluation background)

Two confounds were then ruled out (see `02_matched_evaluation.py` header): non-comparable AUCs across
designs, and the site-buffer exclusion. Decomposition on the full feature set, 5 seeds:

| Component | MaxEnt | XGBoost | RandomForest |
|---|---|---|---|
| Site-buffer exclusion alone | −0.006 | −0.000 | +0.006 |
| TGB over buffered random | +0.002 | −0.002 | −0.002 |
| Hybrid over TGB | −0.027 | −0.005 | −0.007 |
| **Total, common evaluation** | **−0.032** | **−0.007** | **−0.003** |
| Total, own background (what the paper reports) | +0.015 | +0.043 | +0.037 |

Paired across seeds:

- **Background redesign, common evaluation:** MaxEnt 0/5 seeds positive, XGBoost 1/5, RandomForest 2/5.
  There is no reliable positive effect in any algorithm.
- **Adding river distance (feature effect):** **+0.042 AUC, positive in 60/60 paired comparisons.**
- **Inflation caused by scoring each design on its own background: +0.041 to +0.051 AUC,
  positive in 15/15 paired comparisons.**

That inflation is the same magnitude as the entire reported E007 → E013 improvement.

### Mechanism

The hybrid design's background sits systematically further from the presences in environmental space
than a random background does (realised zdist ≥ 2 fraction: hybrid 0.623 vs random 0.503). Discriminating
presences from *more dissimilar* negatives is an easier problem, so AUC rises without the model
transferring any better. The manuscript observed this pool-composition effect and filed it as a caveat
instead of testing it; tested, it accounts for the headline result.

This is the standard presence-background caution (Lobo et al. 2008 — AUC is not comparable across
different background samples). The manuscript **already cites Lobo** and then does not apply it to its
own ladder. Reviewer 1's demand to engage the MaxEnt/ENM literature led directly to the critique that
literature would have supplied.

## Conclusion

**H1 is not supported, and H0 is not the interesting outcome either.** The pre-registered rules anticipated
"the ladder is learner-specific". The actual result is stronger and different: **the ladder is
evaluation-specific.** It is not a property of the learner or of the background design — it is an artefact
of scoring each design against negatives it selected itself.

**Consequence for P2 (SIG territory, PI decision required):** the manuscript's stated main finding —
*"pseudo-absence realism, not feature count alone, is the dominant lever for spatial transfer under
survey-biased archaeological data"* (Abstract) — **does not survive a matched-evaluation test.** Under a
common evaluation background the ranking reverses: the single hydrological feature is the dominant lever
(+0.042, 60/60) and background redesign contributes approximately nothing (−0.014 mean).

Per `docs/SUBMISSION_INTEGRITY_GATE.md`, this must be fixed or the claim downgraded — it cannot be
reworded. Options are set out in the revision plan
(`papers/P2_settlement_model/revision_ammo/JCAA_R1_RESPONSE_PLAN_20260727.md`).

**What is salvageable, and it is substantial:** "the apparent benefit of background redesign in
archaeological presence-background models is an evaluation artefact, and background designs must be
compared on a held-fixed evaluation set" is a genuine, transferable methodological contribution, it is
novel in the archaeological predictive-modelling literature, and it answers both reviewers at once —
R1's MaxEnt/ENM engagement and R2's demand for a sharper research question.

**Status: SUCCESS (negative result on the paper's own claim). Not a failed experiment.**

## Files

- `01_maxent_benchmark.py` → `results/e217_raw_results.csv`, `e217_summary.csv`, `e217_auc_matrix.csv`, `e217_outcome.json`
- `02_matched_evaluation.py` → `results/e217b_raw_results.csv`, `e217b_summary.csv`, `e217b_auc_common.csv`, `e217b_auc_own.csv`, `e217b_outcome.json`

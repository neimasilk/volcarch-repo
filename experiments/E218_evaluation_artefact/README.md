# E218 — Is the Evaluation Artefact Real, and What Drives It?

**Status: SUCCESS — E217's refutation CONFIRMED decisively. Proposed mechanism NOT established
(and the test built to check it was itself flawed — see Stage C).**
**Pre-registration:** `DESIGN.md`, written before running. Run 2026-07-27.
**Driver:** before E217's refutation of P2's central claim goes into a manuscript or an email to the
editor, it has to survive the same adversarial treatment the paper received.

---

## Stage A (decisive) — 3 training designs × 4 fixed evaluation backgrounds × 3 algorithms × 20 seeds

**Pre-registered prediction:** if the artefact is real, the hybrid design wins *only* when evaluated
against hybrid-like negatives. If the paper was right, hybrid wins under all four.

Mean AUC:

| algorithm | training design | uniform | tgb | **hybrid** | stratified |
|---|---|---|---|---|---|
| maxent | hybrid | 0.661 | 0.663 | **0.705** | 0.675 |
| | random | 0.694 | 0.689 | 0.699 | 0.707 |
| | tgb | 0.691 | 0.687 | 0.696 | 0.706 |
| randomforest | hybrid | 0.702 | 0.699 | **0.733** | 0.709 |
| | random | 0.712 | 0.706 | 0.724 | 0.715 |
| | tgb | 0.714 | 0.709 | 0.725 | 0.718 |
| xgboost | hybrid | 0.706 | 0.705 | **0.741** | 0.715 |
| | random | 0.714 | 0.709 | 0.726 | 0.719 |
| | tgb | 0.719 | 0.712 | 0.728 | 0.721 |

**Hybrid ranked best in 0/3 algorithms under uniform, 0/3 under tgb, 0/3 under stratified —
and 3/3 under the hybrid evaluation background.**

Paired per seed (hybrid − random AUC, 20 seeds):

| Evaluation background | MaxEnt | XGBoost | RandomForest |
|---|---|---|---|
| uniform | −0.033 (0/20) | −0.009 (4/20) | −0.009 (4/20) |
| tgb | −0.027 (0/20) | −0.004 (8/20) | −0.007 (6/20) |
| **hybrid** | **+0.007 (14/20)** | **+0.015 (19/20)** | **+0.010 (18/20)** |
| stratified | −0.032 (0/20) | −0.004 (7/20) | −0.006 (4/20) |

The sign flips — and flips only — when the evaluation background matches the training design.
XGBoost goes from 4/20 seeds favouring hybrid under uniform evaluation to 19/20 under hybrid evaluation.
A design that wins solely on its own turf is not a better model.

**TSS** reproduces the same signature (hybrid best 0/3, 0/3, 2/3, 0/3).

### The artefact-immune metric

The **continuous Boyce index** (Hirzel et al. 2006) is presence-only and is computed here against a fixed
uniform availability sample, so it cannot be inflated by the choice of training background. Under it,
hybrid − random across 20 seeds:

| Algorithm | mean | seeds favouring hybrid |
|---|---|---|
| MaxEnt | +0.017 | 11/20 — indistinguishable from chance |
| XGBoost | +0.041 | 13/20 — weak |
| RandomForest | **−0.095** | **2/20 — reliably worse** |

**Reading:** under a metric that avoids the problem, the hybrid design shows **no reliable advantage**,
and for RandomForest it is reliably worse. This is stated carefully: the finding is "no reliable benefit
under an honest metric", not "background design does nothing" — the latter would be its own overclaim.

## Stage B — block size

Hybrid − random on a common evaluation background, at the paper's own three scales:

| Block | MaxEnt | RandomForest | XGBoost |
|---|---|---|---|
| ~40 km | −0.016 | −0.006 | +0.001 |
| ~50 km | −0.020 | −0.006 | +0.002 |
| ~60 km | −0.015 | −0.002 | +0.004 |

Flat or negative at every scale. Block size does not rescue the ladder.

## Stage D — lattice resolution

At a ~150 m lattice (vs the ~300 m used in E217), XGBoost:

| Design | AUC on own background | AUC on common background |
|---|---|---|
| random | 0.711 | 0.705 |
| tgb | 0.717 | 0.710 |
| hybrid | **0.757** | **0.704** |

The ladder persists on own-background scoring (+0.047) and vanishes on common-background scoring
(−0.001). The sampling frame is not doing the work.

## Stage C — MECHANISM TEST FAILED, AND THE TEST WAS BADLY DESIGNED

Pre-registered hypothesis: AUC inflation rises monotonically with the background's environmental
dissimilarity from the presences. **Not supported** — Spearman(dissimilarity, inflation) = **−0.077,
p = 0.41.**

But the test cannot be trusted either, and the diagnosis matters more than the null result. Backgrounds
were built by sampling a *narrow zdist band* around each target. That produces a background concentrated
in a thin shell of environmental space, which is trivially separable from the presence cloud regardless of
how far away the shell sits — `auc_own` reached **0.98** at the *nearest* band (zdist ≈ 0.86), the opposite
of what the dissimilarity hypothesis predicts. Meanwhile `auc_common` collapsed to **0.55–0.59**, barely
above chance, because a thin-shell background trains a model that does not generalise to the landscape at
all. The construction confounded *distance from the presences* with *concentration in a shell*, so it
never tested the intended quantity.

**Consequence:** the artefact is established; its mechanism is not. The manuscript must **not** claim
"inflation is proportional to background dissimilarity" on this evidence.

**Redesign (E218b, specified before running):** sweep the paper's own `hard_frac` knob from 0.0 to 1.0,
drawing from the natural candidate pool instead of a band. This varies mean dissimilarity while keeping
the background a plausible draw from the landscape, and it is interpretable in the paper's own terms
because `hard_frac` is the parameter E013 already tunes.

---

## E218b — mechanism, redesigned: MECHANISM ESTABLISHED, and it is worse than "inflation"

Stage C's band construction was replaced by a sweep of the paper's own `hard_frac` knob (0.0 → 1.0),
drawing from the natural candidate pool so every background stays a plausible landscape draw.
Script: `02_mechanism_hardfrac.py`. 5 seeds × 11 settings × 3 algorithms.

| hard_frac | mean zdist | frac zdist≥2 | **AUC on own background** | **AUC on common background** | inflation |
|---|---|---|---|---|---|
| 0.0 | 2.10 | 0.467 | 0.721 | 0.699 | 0.022 |
| 0.1 | 2.18 | 0.516 | 0.714 | 0.693 | 0.021 |
| 0.2 | 2.26 | 0.574 | 0.725 | 0.695 | 0.031 |
| **0.3** | **2.31** | **0.622** | **0.738** | **0.695** | **0.044** |
| 0.4 | 2.39 | 0.678 | 0.746 | 0.686 | 0.060 |
| 0.5 | 2.48 | 0.740 | 0.760 | 0.681 | 0.079 |
| 0.6 | 2.52 | 0.780 | 0.763 | 0.677 | 0.086 |
| 0.7 | 2.61 | 0.836 | 0.783 | 0.662 | 0.121 |
| 0.8 | 2.68 | 0.892 | 0.806 | 0.654 | 0.152 |
| 0.9 | 2.74 | 0.946 | 0.827 | 0.637 | 0.190 |
| 1.0 | 2.81 | 1.000 | **0.844** | **0.602** | 0.242 |

| Relationship | Spearman | p |
|---|---|---|
| dissimilarity → inflation | **+0.961** | 1.1e-92 |
| dissimilarity → AUC on own background | +0.886 | 2.3e-56 |
| dissimilarity → AUC on common background | **−0.708** | 2.0e-26 |

**The two curves run in opposite directions.** As the knob is turned up, the number a paper would report
climbs from 0.721 to 0.844, while the model's actual ability to generalise falls from 0.699 to 0.602.

**This is materially stronger than the E217 result and stronger than the caution it descends from.**
Lobo et al. (2008) says AUC is not comparable across background samples. This says something sharper and
quantified: **within this design space, optimising the reported metric systematically selects worse
models**, with a monotonic dose–response across the full range of a parameter practitioners actually tune.

**It applies to the manuscript's own tuning.** E013 swept `hard_frac` ∈ {0.0, 0.15, 0.30} and selected
0.30 — the maximum available, and the row bolded above (realised frac zdist≥2 = 0.622, matching the 0.62
the manuscript flags as unexplained). Across that swept range the reported AUC rose **+0.018** while
generalisation fell **−0.004**. The tuning procedure gained nothing real; had the sweep extended to 1.0 it
would have reported 0.844 for a model that generalises at 0.602.

**Pre-registered branch 3 did not occur, and its opposite did.** The sweep was pre-registered to report a
partial rehabilitation if `auc_common` rose with `hard_frac` (i.e. hard negatives genuinely help
generalisation). It falls instead: hard negatives **actively degrade** generalisation here
(Spearman −0.708). Recorded because it was pre-committed, and it points the other way.

---

## Conclusion

**E217 is confirmed under every robustness check that was designed to break it:** four evaluation
backgrounds, 20 seeds, three algorithms, three metrics, three block sizes, two lattice resolutions.
The reported E007 → E013 improvement in P2 is an artefact of scoring each background design against the
negatives it selected for itself.

**The mechanism is now established** (E218b, above), on the second attempt and after the first attempt's
instrument was diagnosed as broken: reported AUC and true generalisation move in **opposite** directions as
background dissimilarity rises (+0.961 vs −0.708).

**One thing remains NOT established and must not be written as though it were:** that background design is
worthless. Boyce says "no reliable discrimination benefit"; E219 shows it changes the map substantially.
Those are different claims and both are weaker than "worthless".

**Status of the follow-ups:** E218b — done (above). E219 (does background design change the predicted map
even when it does not change the score?) — done, see `../E219_map_divergence/README.md`; it supplies the
constructive replacement claim and answers Reviewer 2's objection that the work is not specifically
archaeological.

## Files

`01_artefact_robustness.py` → `results/e218_stageA_raw.csv`, `e218_stageA_{auc,tss,boyce}_matrix.csv`,
`e218_stageB_blocksize.csv`, `e218_stageC_dissimilarity.csv`, `e218_stageC_summary.csv`,
`e218_stageD_decimation.csv`, `e218_outcome.json`

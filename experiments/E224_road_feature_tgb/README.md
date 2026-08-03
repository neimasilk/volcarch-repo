# E224 — Does target-group background work once the bias variable is a feature?

**Status: FAILED (hypothesis not supported) — and that is the result, not a problem with the run.**
**Date:** 2026-08-03 · **Line:** 01 spatial (serves P2/JCAA #280 v0.2) · **Model:** Opus 5
**Pre-registration:** `DESIGN.md`, committed **before** execution (`d4f44af`).

---

## Hypothesis

Correction **K4** (`review_package_20260727/09_REVIEW_ATAS_BABAK2.md` §5) proposed an explanation for
E222's null result: target-group background (TGB) did not beat a random background because the model's
feature set — `elevation, slope, twi, tri, aspect, river_dist` — **does not contain `road_dist`**. The
survey-bias factor *s(x)* is therefore not representable in feature space, so it enters as label noise;
TGB is designed to cancel *s(x)* **in feature space**, and if *s(x)* is not there, there is nothing to
cancel.

> **H:** target-group background can only help when the bias variable is correlated with the model's
> feature space.
> **Prediction:** adding `road_dist` to the feature set makes TGB beat a random background in the same
> synthetic worlds where it did not.

## Method

E222 world A, reproduced exactly — same lattice (555,609 cells), same intensity coefficients, same
world seeds, same config seeds, same background draws, same 5-fold spatial-block CV, same three
algorithms — with **one** difference: whether `road_dist` is in the feature set.
10 worlds × {random, TGB} × {`no_road`, `with_road`} × 3 algorithms = **120 runs**.

Primary metric (pre-registered): **`map_jaccard`, TGB − random, paired within (world, algorithm)**,
30 pairs per arm. Decision threshold: mean > 0 **and** ≥ 60% of pairs positive.

## Result

| Arm | TGB − random, map Jaccard | positive | TGB − random, true AUC | positive |
|---|---|---|---|---|
| `no_road` (control) | **−0.0254** | 30% | −0.0006 | 36.7% |
| `with_road` (test) | **−0.0217** | 30% | −0.0000 | 60.0% |

**H is not supported.** On the pre-registered primary metric the test arm is indistinguishable from the
control: TGB remains slightly *worse* than a random background, in the same 30% of pairs, whether or
not the model can see the bias variable.

**Control-arm replication (DESIGN §3):** the `no_road` arm reproduces E222 world A —
max |Δ| = 0.0004 (`auc_true`), 0.0003 (`auc_own`), 0.017 (`map_jaccard`, mean |Δ| 0.0013). Below the
0.02 void threshold. **The run is valid.**

**Levels** (means across worlds and algorithms), for context — the pre-registered note said absolute
levels are not evidence either way:

| Arm | map Jaccard random / TGB | true AUC random / TGB |
|---|---|---|
| `no_road` | 0.7145 / 0.6890 | 0.8298 / 0.8292 |
| `with_road` | 0.7075 / 0.6858 | 0.8292 / 0.8292 |

Adding the bias variable did **not** degrade truth-anchored performance either, which was the expected
side effect. It changed almost nothing at all.

## What this does to K4

**K4 is downgraded from a diagnosis to a refuted conjecture.** The manuscript may no longer say that
the TGB null is "predicted" or "explained by feature-space representability". The honest statement is:

> In our synthetic worlds, target-group background did not improve truth-anchored recovery over a
> random background. We tested one explanation — that the model could not represent the bias variable —
> by adding road distance to the feature set. It made no difference (−0.022 vs −0.025 map Jaccard,
> 30% of pairs positive in both). The null is therefore **unexplained** by that account, and we report
> it as such.

This is a loss of a rhetorically convenient claim and a gain in accuracy. The E222 null itself is
unchanged and still reportable; what is gone is our explanation of it.

## Caveat that weakens the manipulation (pre-registered check, DESIGN §5)

`road_dist` is **not** independent of the existing feature set: `river_dist` **+0.49**,
`elevation` +0.31, `tri` +0.29, `slope` +0.28, `twi` −0.22, `aspect` +0.00.

So the premise of K4 was already only partly true: about half the road-distance signal was reachable
through river distance before the manipulation. Two readings, and the honest thing is to give both:

1. **The manipulation was weaker than intended** — the model could already partly express *s(x)*, so
   "adding" it changed less than the design assumed. On this reading H is under-tested, not refuted.
2. **The manipulation was sufficient and H is wrong** — the model went from partial to full access to
   the bias variable and TGB still did not help, in either condition.

Reading 1 is the reason this experiment is filed as **FAILED, not REFUTED**. A cleaner test needs a
bias variable that is orthogonal to the feature set by construction — a synthetic survey-effort surface
uncorrelated with terrain. That is a design change, not a re-run, and it is future work.

## Files

- `DESIGN.md` — pre-registration (both branches written before the run)
- `01_road_feature_tgb.py` — the experiment
- `results/e224_runs.csv` — 120 runs
- `results/e224_outcome.json` — pre-registered readout, collinearity check, replication check

## Consequences

- `papers/P2_settlement_model/review_package_20260727/10_SET_KLAIM_TERKOREKSI.md` §K-E: E1/E2 marked
  **refuted**; the manuscript reports the null as unexplained.
- `lines/01_spatial/CLAUDE.md`: the K4 row is corrected.
- Response to reviewers: this becomes part of the disclosure — we proposed an explanation, tested it,
  and it failed.

# E224 — DESIGN (pre-registration)

**Written 2026-08-03, before any code was run.** Line 01 (spatial), serves P2/JCAA #280 v0.2.
**Predecessor:** `E222_synthetic_ground_truth` (worlds A/B). **Origin:** correction **K4** in
`papers/P2_settlement_model/review_package_20260727/09_REVIEW_ATAS_BABAK2.md` §5, carried into the
claim set as **K-E** in `10_SET_KLAIM_TERKOREKSI.md`.

---

## 1. The question

E222 found that target-group background (TGB) does **not** improve truth-anchored recovery over a
random background, even though its simulated survey bias
(`clip(exp(-road/12000), 0.03, 1)`) is exactly the condition TGB theory (Phillips et al. 2009) asks
for. Reported as a null, it reads as a bug.

The proposed explanation (K4): the model's feature set is
`elevation, slope, twi, tri, aspect, river_dist`. **`road_dist` is not in it.** The survey-bias factor
*s(x)* therefore cannot be expressed in feature space at all; it enters as label noise (some
high-intensity cells simply go unobserved). TGB is designed to cancel *s(x)* **in feature space**. If
*s(x)* is not representable there, there is nothing for TGB to cancel.

**If that explanation is right, making *s(x)* representable should make TGB work.**

## 2. Hypothesis and the falsifiable prediction

> **H:** target-group background can only help when the bias variable is correlated with the model's
> feature space.
>
> **Prediction:** adding `road_dist` to the feature set makes TGB beat a random background on
> truth-anchored recovery, in the same synthetic worlds where it did not.

This is a **conditional** claim about when TGB works, not a claim that `road_dist` should be a
modelling feature. It must not be read as a recommendation — see §6.

## 3. Design

Identical to E222 world **A_observed** in every respect (same lattice, same intensity coefficients,
same world seeds `10_000 + 997·w`, same config seeds, same spatial-block CV, same three algorithms,
same E217 draw functions) except for **one** thing: the feature set.

| Arm | Feature set | Role |
|---|---|---|
| `no_road` | `elevation, slope, twi, tri, aspect, river_dist` | reproduces E222's condition; internal control |
| `with_road` | the same **+ `road_dist`** | the test condition |

- **Configs:** `random` and `tgb` only. The hybrid dial is not part of this question.
- **Worlds:** 10 (w = 0…9), surface A_observed.
- **Algorithms:** maxent, xgboost, randomforest → **10 × 2 × 2 × 3 = 120 runs**.
- **Metrics (identical definitions to E222):** `auc_true` (block-CV AUC against an unbiased held-out
  presence sample), `map_jaccard` (top-decile overlap with the true intensity surface),
  `spearman_remote` (rank agreement restricted to the least road-accessible quintile), `auc_own`.

The `no_road` arm should reproduce E222's world-A numbers for random and tgb. **If it does not, the
run is void** and the discrepancy is investigated before anything is interpreted — that check is the
reason the control arm is re-run rather than copied from `e222_runs.csv`.

## 4. Pre-registered decision rule

Primary metric: **`map_jaccard`, TGB − random, paired within (world, algorithm)** — 30 pairs per arm.
This is the same primary metric E222's P3 used. Secondary: `auc_true`, same pairing.

| Outcome in the `with_road` arm | Reading | Consequence |
|---|---|---|
| TGB − random mean **> 0** and positive in **≥ 60%** of the 30 pairs, while the `no_road` arm stays ≈0 | **H supported.** The E222 null was a *predicted* null, not a failure of TGB | K-E2 becomes a tested condition in the manuscript; the TGB null converts from weakness to contribution |
| TGB − random ≈ 0 in **both** arms | **H not supported.** The feature-space account does not explain the null | K-E1/E2 are downgraded to a conjecture in the discussion, explicitly labelled untested. The manuscript keeps the null as unexplained |
| TGB − random **< 0** in `with_road` (TGB actively worse) | The added feature changes the problem in a way the account did not predict | Report as-is; do not reinterpret post hoc. Report the mechanism as open |

**Also pre-registered as an expected side effect, not an outcome:** adding `road_dist` will probably
*lower* `auc_true` for **both** configs, because the model can now learn the survey-bias pattern
instead of the intensity surface. That is not evidence for or against H. H is about the **difference**
between TGB and random within the same arm, never about absolute level.

**Power.** 30 paired comparisons per arm; E222's P3 gave −0.010 (46.7% positive). The rule is a sign
test with a 60% threshold. With n=30, 60% positive is not statistically decisive on its own; the
decision rule is therefore about **direction and magnitude relative to the control arm**, and any
result will be reported with its uncertainty, not as a significance verdict.

## 5. What would make this experiment worthless

- If the `no_road` arm fails to reproduce E222 (§3) — void.
- If `road_dist` turns out to be strongly collinear with an existing feature, the manipulation is not
  clean. To be checked and reported: correlation of `road_dist` with each of the six features across
  the frame.

## 6. Interpretation limit — state this in the manuscript

`road_dist` is the manuscript's **highest tautology-correlated proxy** (|ρ| = 0.307, Test 1) and the
basis of the TGB draw itself. Including it as a predictor is a **diagnostic manipulation, not a
recommended design**: it deliberately lets the model see the bias variable in order to test whether
representability is what TGB needs. Any wording that suggests "add road distance to your features"
would contradict the manuscript's own tautology argument.

## 7. Cost and stop rule

Same pipeline as E222 with one third of its configs. If a single world exceeds ~15 minutes wall clock,
drop `spearman_full`/Boyce extras before dropping worlds — the world count is what carries the
inference.

---

*Pre-registered by Claude Opus 5 on 2026-08-03 under the E217–E223 protocol: design fixed and committed
before execution, both branches written down, no post-hoc metric substitution.*

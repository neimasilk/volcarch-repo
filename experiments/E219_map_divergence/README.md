# E219 — Does Background Design Change the Map Rather Than the Score?

**Status: SUCCESS — mixed and genuinely informative. Closes INT-1. Answers Reviewer 2's R2-F with evidence.**
**Pre-registration:** `../E218_evaluation_artefact/DESIGN.md`. Run 2026-07-27.
**Driver:** E217/E218 showed P2's AUC ladder is an evaluation artefact. That is a demolition, and it makes
Reviewer 2's objection *worse* — going more methodological amplifies "what makes this archaeological?".
This experiment tests whether anything constructive survives.

Scale: 378 presences, 588,535 frame cells, 5 seeds × 3 background designs × 3 algorithms = 45 full-landscape
prediction surfaces.

---

## Part A — map divergence, against a noise floor

The control that makes this falsifiable: maps are compared **between designs** (same seed) *and*
**within a design** (different seeds). If two draws of the same design disagree as much as two different
designs, there is no design effect — only sampling noise.

Top-10% ("survey priority tier") Jaccard overlap:

| Algorithm | within design (noise floor) | between designs | design effect > noise? |
|---|---|---|---|
| MaxEnt | 0.684 | **0.466** | **yes** |
| XGBoost | 0.549 | **0.488** | yes (marginal) |
| RandomForest | 0.690 | 0.651 | no |

By design pair (top-10% Jaccard):

| Algorithm | random ↔ tgb | random ↔ hybrid | tgb ↔ hybrid |
|---|---|---|---|
| MaxEnt | 0.664 | **0.345** | **0.389** |
| XGBoost | 0.550 | **0.459** | **0.454** |
| RandomForest | 0.731 | 0.625 | 0.598 |

**Two findings, and the second may matter more than the first.**

1. **The hybrid design moves the map; TGB alone barely does.** Under MaxEnt, only 35% of priority cells
   survive the switch from a random to a hybrid background — half the recommended survey targets change,
   while every discrimination metric says the models are equivalent. Confirmed beyond the noise floor in
   2 of 3 learners.

2. **The noise floor is itself alarming.** Re-running the *same* design with only a different random seed
   changes 31–45% of the top-decile priority cells (Jaccard 0.549–0.690). A survey-prioritisation map from
   this pipeline is not stable to the seed. That is a practical, archaeologically consequential result,
   and it is not in the submitted manuscript.

## Part B — is the disagreement organised the way bias correction predicts?

Target-group background is supposed to pull predictions *away* from over-surveyed, road-accessible ground.
Mean percentile-rank shift relative to a random background, by road-distance quintile:

| Algorithm | design | Q1 (nearest roads) | Q2 | Q3 | Q4 | Q5 (most remote) |
|---|---|---|---|---|---|---|
| MaxEnt | hybrid | −0.025 | −0.024 | −0.033 | −0.018 | **+0.101** |
| XGBoost | hybrid | −0.019 | −0.018 | −0.023 | −0.015 | **+0.076** |
| RandomForest | hybrid | −0.021 | −0.017 | −0.019 | −0.009 | **+0.067** |
| (tgb rows) | tgb | ≈ −0.003 | ≈ −0.003 | ≈ −0.003 | ≈ −0.002 | +0.009…+0.014 |

**Partial support, honestly stated.** The hybrid background does shift predicted suitability toward the
least accessible ground — the direction bias correction predicts. But the effect is confined to the most
remote quintile, the overall monotonic association is weak (Spearman +0.065 to +0.124), and **elevation is
a competing explanation of comparable or greater strength** (up to +0.305 for MaxEnt). This should be
reported as a directional signal with a live confound, not as confirmation.

## Part C — terrain-matched volcanic vs non-volcanic uplands (Reviewer 2, R2-F)

### INT-1 closed

The submitted code hardcodes **7** volcanoes. The canonical inventory
(`data/processed/dashboard/volcanoes_java_full.csv`) has **13** inside the paper's own stated bounds
(111–115°E): the 7 plus **Lawu, Wilis, Kawi-Butak, Penanggungan, Iyang-Argapura, Baluran**.

Test 1 tautology correlation (predicted suitability vs nearest-volcano distance), recomputed:

| Inventory | Spearman ρ |
|---|---|
| 7 legacy volcanoes | −0.243 |
| **13 canonical volcanoes** | **−0.281** |

(The manuscript reports −0.163 for the legacy set; the difference from our −0.243 reflects an independently
reimplemented model and sampling frame, so this is a *directional* correction, not a claim to have
reproduced their exact figure.) **The correction strengthens the correlation but leaves it far below the
0.5 FAIL threshold — so the paper's Test 1 GREY_ZONE verdict survives the inventory fix.** The defect is
real and must be disclosed; it does not overturn the tautology conclusion.

### The matched comparison

Uplands = elevation ≥ 200 m. Volcanic = ≤20 km from a canonical centre (112,093 cells); non-volcanic =
≥40 km (44,495 cells). Coarsened exact matching on elevation × slope × TRI × TWI (5 bins each);
90 of 100 strata occupied in both arms.

| | volcanic uplands | non-volcanic uplands |
|---|---|---|
| Matched-weighted mean predicted suitability | 0.2249 | 0.1702 (**+0.055**) |
| Observed site density | **0.01377 / km²** (145 sites / 10,528 km²) | **0.00048 / km²** (2 sites / 4,183 km²) |

**The answer to R2-F is not the one either side expected.** Reviewer 2 worried the model might be
detecting a volcanic effect indirectly through elevation and slope. It is not — after matching on terrain,
the model predicts only a +0.055 suitability difference (~32% relative), while observed site density
differs by roughly **29-fold**. The terrain model does not recover the volcanic concentration; it
massively under-predicts it.

That cuts both ways and both must be stated:
- **For the model's honesty:** it is not a disguised volcano-proximity detector. R2's specific concern is
  answered in the negative, with a matched design rather than an assertion.
- **Against the paper's ambition:** whatever actually structures this site distribution — volcanic soil
  fertility, or survey history concentrating on the temple landscapes — the terrain covariates barely see
  it. The suitability surface is only weakly related to what predicts site presence at this scale.

**Caveat that must travel with this number:** the non-volcanic matched arm contains **2 sites**. The
direction (145 vs 2) is unambiguous; the ratio is fragile and must not be quoted as a precise multiple.
Whether that near-absence is real or is itself survey bias is exactly the question the paper exists to
ask, and this experiment cannot settle it.

## Conclusion

Something constructive does survive the demolition, and it is archaeological rather than statistical:

1. **Background design changes which cells a fieldworker is sent to (up to 65% of the priority tier
   turns over) while leaving every discrimination metric unchanged.** Discrimination metrics therefore
   cannot be used to choose a background design — the choice has to be made on bias-correction grounds and
   defended as such. This is the positive claim that replaces the refuted one.
2. **The priority map is unstable to the random seed alone** (31–45% turnover), which is a reproducibility
   finding of direct practical consequence and is absent from the submitted manuscript.
3. **R2-F answered with evidence**, and INT-1 fixed, with the honest note that fixing it does not change
   the tautology verdict.

**Not established:** that the map differences are *better* — no ground truth exists to adjudicate. The
paper must say "different, and consequential", not "improved".

## Files

`01_map_divergence.py` → `results/e219_map_divergence.csv`, `e219_agreement_summary.csv`,
`e219_disagreement_by_road.csv`, `e219_terrain_matched.csv`, `e219_outcome.json`

*Note: the first run crashed at the final console print (a Unicode arrow on a cp1252 Windows console)
after all results were written; the character has been replaced with ASCII.*

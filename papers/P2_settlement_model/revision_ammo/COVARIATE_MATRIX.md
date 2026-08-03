# Covariate matrix and analytical roles — answers Reviewer 2, items D and E

**Built 2026-08-03 for P2/JCAA #280 v0.2.** Source of truth: the experiment scripts themselves
(`FEAT_COLS` / background parameters read out of each `experiments/E0NN_*/01_*.py`), not the
manuscript text. Machine-readable version: `covariate_matrix.csv`.

Reviewer 2 wrote: *"is not clear which variables are included and excluded"* and *"The variables also
need to be separated according to their analytical role."* This document answers both, and is meant to
become **two tables in v0.2** (§2.1 and §2.4).

---

## 1. Table A — every variable, and the role it plays

Roles are exclusive by construction: a variable that enters as a predictor is **never** also used to
build the background, with one deliberate exception (E224, flagged).

| Variable | Raster / source | Analytical role | How it enters the analysis | In the E013 feature set? |
|---|---|---|---|---|
| `elevation` | `jatim_dem.tif` — Copernicus GLO-30 (SRTM lineage), 30 m, decimated ×10 → ~300 m lattice | **Settlement suitability** | training feature | ✅ |
| `slope` | derived from DEM | **Settlement suitability** | training feature | ✅ |
| `twi` | topographic wetness index, derived from DEM | **Settlement suitability** (water availability) | training feature | ✅ |
| `tri` | terrain ruggedness index, derived from DEM | **Settlement suitability** (terrain cost) | training feature | ✅ |
| `aspect` | derived from DEM | **Settlement suitability** (insolation/orientation) | training feature | ✅ |
| `river_dist` | OpenStreetMap waterways / HydroSHEDS, Euclidean distance | **Settlement suitability** (water access) | training feature, added at E008 | ✅ |
| `clay`, `silt` | SoilGrids topsoil composition | **Preservation / substrate** | training features **in E009 only**; dropped afterwards — they *lowered* spatial AUC (0.695 → 0.664) | ❌ |
| `road_dist` | `jatim_road_dist_expanded.tif` — OSM roads; from E012 the expanded class set (`unclassified`, `residential`, `service`) | **Modern accessibility / survey effort** | **never a training feature.** It defines (a) the target-group background acceptance probability, (b) the hybrid background pool, (c) the E014 holdout split, (d) tautology Test 1 and Test 3 | ❌ **by design** |
| `volcano_dist` | `data/processed/dashboard/volcanoes_java_full.csv` (canonical, 30 centres; **13** inside the paper's 111–115°E bounds) | **Taphonomy / burial — diagnostic only** | **never a training feature.** Used post hoc in tautology Test 1 and in Figure 2 | ❌ **by design (this is the tautology control)** |
| `zdist` | Mahalanobis-type distance of a cell from the presence centroid in feature space | **Background design parameter** | selects "hard negatives" (2.0 ≤ z ≤ 5.0) inside the hybrid pool | n/a |
| `region_id` | spatial quadrant of the study frame | **Background design parameter** | regional quota blending in the hybrid design | n/a |

**Why the roles must not be mixed** (R2-E, in the manuscript's own voice):

- A **suitability** variable answers *would people have settled here?*
- An **accessibility** variable answers *would we have found it if they had?*
- A **preservation** variable answers *would it still be there to find?*

Putting all three in one feature set makes the output uninterpretable: a low score could mean any of
the three, and the model cannot tell you which. This paper therefore keeps only the suitability block
as predictors, expresses accessibility **through the background design**, and treats preservation and
volcanic proximity as **diagnostics applied to the output**. That separation is not a technicality —
after E217 it is the paper's actual subject, because it is precisely the background (the accessibility
channel) that turned out to drive the reported number.

> **This is also the honest answer to R2-B** (*"what makes the approach specifically archaeological?
> sites function mainly as spatial observations"*): the archaeological content is not in the label
> set, it is in the **background design** — the claim about where a survey could plausibly have found
> a site. Say so plainly.

---

## 2. Table B — per-experiment inclusion matrix

`✔` = used as a training feature · `–` = not used · `bg` = used to construct the background, not as a
feature · `diag` = used only as a post-hoc diagnostic.

| Exp | elev | slope | twi | tri | aspect | river | clay/silt | road | volc | Background design | Reported spatial AUC |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|---|
| **E007** v1 | ✔ | ✔ | ✔ | ✔ | ✔ | – | – | – | – | uniform random, 1:5, 2 km site buffer | 0.659 |
| **E008** v2 | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | – | – | – | uniform random, 1:5 | 0.695 |
| **E009** v3 | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | – | – | uniform random, 1:5 | 0.664 ⬇ |
| **E010** v4 | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | – | bg | – | **TGB**: decay 12 km, max road 40 km, min p 0.03 | 0.711 |
| **E011** v5 | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | – | bg | – | TGB, parameters swept | 0.725 |
| **E012** v6 | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | – | bg | – | TGB on **expanded** road classes, max road 20 km | 0.730 |
| **E013** v7 | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | – | bg | – | **hybrid** = TGB pool + regional quota blend ∈ {0, 0.3, 0.5, 0.7} + hard-negative fraction ∈ **{0, 0.15, 0.30}**, z ∈ [2, 5] | **0.768** best run / **0.751** seed-averaged |
| **E014** holdout | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | – | **split** | diag | E013 background; presences split by road distance ≤ / > 1 km | 0.755 holdout vs 0.785 spatial CV |
| **E015** SHAP | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | – | bg | – | E013 | — (interpretation only) |
| **E016** zones | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | – | bg | – | E013 | — (zoning only) |
| **E217** MaxEnt benchmark | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ ⁽ᵃ⁾ | – | bg | – | random / TGB / hybrid × **common** evaluation background | see §3 |
| **E218** artefact | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | – | bg | – | 3 train designs × 4 evaluation backgrounds | — |
| **E219** divergence | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | – | bg | diag | E013; + terrain-matched volcanic/non-volcanic control | — |
| **E220–E223** | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | – | bg | – | dial sweep, seeds, synthetic worlds, robustness | — |
| **E224** ⚠ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | – | **✔ (diagnostic)** | – | random / TGB, synthetic ground truth | pre-registered, see DESIGN |

⁽ᵃ⁾ E217 runs two feature sets on purpose (`terrain` = 5 vars, `terrain_river` = 6 vars) in order to
compare the size of a **feature** effect against the size of a **background-design** effect.

> ⚠ **E224 is the only experiment in the entire series in which `road_dist` is a predictor**, and it
> is there to test whether target-group background can work at all when the bias variable is invisible
> to the model (correction K4). It is a **diagnostic manipulation, not a recommended design** — see
> `experiments/E224_road_feature_tgb/DESIGN.md` §6. Do not let this row read as an endorsement.

**Reproducibility statement for R2-D.** Every cell above is read from the script that produced the
result. The rasters are in `data/processed/dem/`, the presences in
`data/processed/east_java_sites.geojson` (378 with valid features inside 111–115°E), the CV is
deterministic 5-fold spatial blocking at 0.45°, and the pseudo-absence ratio is 1:5 throughout.

---

## 3. Three things this table exposes that the submitted manuscript does not state

**(a) `road_dist` carries four jobs at once.** It builds the background, it defines the E014 holdout,
and it is a tautology proxy in Tests 1 and 3 — while being excluded from the features. The submitted
manuscript acknowledges the dual role in one limitations sentence (§ limitations, item on road
accessibility). After E217 that acknowledgement is no longer adequate: **the variable that defines the
background is the variable that drives the reported number.** v0.2 must state the four roles in §2.1,
not in the limitations.

**(b) INT-4 — E014's stored result file was mislabelled.** The script has two branches: a real
discovery-year split, and an accessibility fallback used when too few sites carry known discovery
dates. The fallback ran, but the old output template printed the temporal labels regardless, so
`results/temporal_validation_results.txt` read *"Split year: 2000 / Pre-2000: 333 sites / Post-2000: 45
sites"* for a split that was actually road distance ≤ 1 km vs > 1 km.
**Verified 2026-08-03:** sampling `jatim_road_dist_expanded.tif` at the 378 valid site locations gives
exactly **333 ≤ 1 km and 45 > 1 km** — the same counts. The manuscript's Test 4 text describes the
accessibility split **correctly**; only the result artefact was wrong. AUC = 0.755 is unaffected. The
output template now records the branch that ran, and the stored file carries a correction notice.

**(c) Calling it a "temporal split" is a misnomer we should retire.** The abstract says *"temporal
split validation"* with no qualifier; the split is an accessibility proxy for discovery order. It
already misled one internal document (INT-2 in `JCAA_R1_RESPONSE_PLAN`, which described the split as
chronological) and it is exactly the kind of undefined term Reviewer 1 objected to. **Rename it in
v0.2 to "accessibility-proxy holdout"** and state the proxy assumption explicitly: sites far from
roads stand in for sites found late. That is a weaker and more honest claim, and it costs the paper
nothing — the result is unchanged.

---

## 4. What to build from this

1. **v0.2 Table 1** = Table A above, minus the last two rows (`zdist`, `region_id` go to §2.4 prose).
2. **v0.2 Table 2** = Table B, rows E007–E014 only; E217–E224 belong in the new results section.
3. One paragraph in §2.1 on role separation (§1 above), one in §2.4 on the four jobs of `road_dist`.
4. Response-to-reviewers: R2-D and R2-E are answered by these two tables; say that building them
   surfaced INT-4, and that we corrected it ourselves.

---

*Generated 2026-08-03 by reading the experiment scripts. Regenerate the CSV with
`revision_ammo/covariate_matrix.csv` as the canonical machine-readable copy.*

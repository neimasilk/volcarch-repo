# E236 — Design (pre-registration): do P2's real-data results hold on a cleaned presence set?

**Written:** 2026-10-26, before any analysis below was run on the cleaned set. **Frozen** when this file is pushed (the
PI's "ok, perbaiki" of 26 Oct, on the recommendation D1 in `papers/P2_settlement_model/AGENDA_PI_20261026.md`).
Any change after the freeze is logged in §8 with date and reason.
**Line:** 01_spatial · **Serves:** P2 (JCAA #280, round-2 revision); ledger **C081** (presence data) and **C083**
(seed-stability curve).
**Not a thesis test (K9):** it asks whether the evaluation results hold on better presence data. No branch says
anything about volcanic burial.

---

## 1. Question

P2's real-data analyses use 378 "presences" from `data/processed/east_java_sites.geojson`. The audit of 26 Oct
(C081, `papers/P2_settlement_model/revision_ammo/presence_audit_20261026.py`) found three problems:
- about 124 of them are modern monuments, statues or memorials;
- 13 are misplaced or are not sites;
- all 85 rows that come from Wikipedia lists were geocoded from their names by Nominatim. Several fall on one
  fallback point in Malang (Candi Sewu, Sari, Mendut, Kalasan, Bima, and Sawentar from Blitar).

**Do the paper's real-data results hold when the models are re-fitted on presences that are archaeological features
with their own coordinates?**

A second question rides along (C083). E221's seed-stability curve compares each k-seed ensemble with a 10-seed
ensemble that contains those seeds. **How well do two ensembles that share no seed agree?**

## 2. What is known before freezing (declared)

1. **The cleaned set has 161 presences.** The canonical file has 666 rows; the rule of §3 keeps 162; E217's
   study-area clip and raster filter keep 161 (`clean_sites.py`, run 26 Oct). Of these, 155 come from OSM and 6 from
   Wikidata; no two share a coordinate. This is a count of inputs, not of results.
2. **The full-set values that the paper reports** were re-derived on 26 Oct by `03_summarise.py full`, with the G1
   script's definitions:
   - E217b: redesign on the common background −0.0142; river +0.0424 (12/12).
   - E218A: hybrid own-background inflation 60/60, mean +0.037; hybrid beats random only on its own background
     (3/3 there, 0/3 elsewhere); TGB inflation −0.0054 (22/60).
   - E218b: own-background criterion 0.7208 → 0.8435, maximum at 1.0; common background 0.699 → 0.602.
   - E221: densities, robust core vs contingent fringe — 1.93× (RF), 4.34× (XGBoost), 5.62× (MaxEnt).
3. **The disjoint-ensemble agreement is known only at k = 5.** E221's split-half control gives 0.751 (XGBoost),
   0.866 (RF) and 0.858 (MaxEnt) as means over designs. No other k has been computed.
4. **The cleaned set is less than half the size**, so fold-level noise will be larger. The thresholds of §5 were set
   with that in mind: they ask for the same direction and a material size, not the same values.

## 3. The cleaning rule (frozen; `clean_sites.py`)

Applied to the canonical file before E217's own clip and raster filter:
- **R1** — drop rows whose coordinates were geocoded from the name (`accuracy_level == "nominatim"`; all are from the
  Wikipedia lists).
- **R2** — drop rows typed `monument`, or whose name matches the modern-feature list of C029 (tugu, monumen, patung,
  museum, …), unless the name matches an ancient-feature list (candi, situs, arca, prasasti, petirtaan, gapura, punden,
  yoni, lingga, …).
- **R3** — drop:
  - named temples of other regions (Borobudur, Prambanan, Besakih, …);
  - `objek_wisata` rows;
  - names starting "Pura" (modern Balinese temples in East Java);
  - "Ponpes"; "Pabriek".
- **R4** — keep one row per coordinate (rounded to 10⁻⁵ degree).

It is a conservative subset, **not a validated inventory** (C029's "OPEN-fix" stays open). Some genuine sites are
lost with R1, for example Candi Singosari and Kidal, whose only rows are name-geocoded. Some uncertain rows stay: OSM
features without a name but tagged `archaeological_site`, and graves (makam) of unknown age.

## 4. Analyses (each runs the original script unchanged, through `01_run_part.py`)

| Part | Original | What is re-run on the 161 presences | Output |
|---|---|---|---|
| A | E217b `02_matched_evaluation.py` | 4 designs × 2 feature sets × 3 algorithms × 5 seeds, own and common background | `results/A_e217b/` |
| B | E218 `01_artefact_robustness.py`, stage A only | 3 training designs × 4 evaluation backgrounds × 3 algorithms × 20 seeds | `results/B_e218a/` |
| C | E218b `02_mechanism_hardfrac.py` | hard-negative share 0.0–1.0 (11 values) × 5 seeds × 3 algorithms | `results/C_e218b/` |
| D | E221 `01_seed_stability.py` | 10 seeds × 3 designs × 3 algorithms, full-frame maps, priority sets, densities, turnover | `results/D_e221/` (maps ignored by git; they regenerate) |
| E | E235 `01_elevation_vs_volcanic.py` | the near − far comparison on part D's maps, all three zone definitions | `results/E_e235/` |
| F | new, `02_disjoint_ensembles.py` | two disjoint k-seed ensembles, k = 1–5, 50 random splits, on E221's stored maps (full) and on part D's maps (clean) | `results/F_disjoint_{full,clean}.csv` |

Not re-run, and the paper will say so:
- **E007–E013:** the historical sequence the paper examines; it was built on the 378 records.
- **E219:** Test 1 correlation, Section 3.8.
- **E223:** confidence intervals and regularisation.
- **E014:** the accessibility holdout.
- **E222 and E224:** synthetic, so not affected.

## 5. Decision rules (frozen; quantities as in `03_summarise.py`)

| Claim in P2 | Quantity (cleaned set) | HOLDS | FAILS | otherwise |
|---|---|---|---|---|
| **A** On a common background, redesign adds nothing, while the river feature adds | redesign = mean over algorithms of AUC(hybrid) − AUC(random, no buffer), common background, full features; river = mean over 12 design × algorithm pairs of AUC(with river) − AUC(terrain only), common background | redesign ≤ +0.010, **and** river > redesign, **and** river positive in ≥ 9/12 | redesign ≥ +0.020 | MIXED |
| **B** The hybrid design is inflated by its own background | inflation = AUC(hybrid train, hybrid eval) − AUC(hybrid train, uniform eval), per seed × algorithm (60 cells); wins of hybrid over random per evaluation background | ≥ 90 % of cells positive, **and** mean ≥ +0.015, **and** hybrid beats random in ≤ 1/3 algorithms on every non-hybrid background | < 2/3 of cells positive, **or** mean < +0.005 | MIXED |
| **C** The selection criterion has no interior optimum | mean over algorithms and seeds, by hard-negative share | argmax of own-background AUC at 0.9 or 1.0, **and** own(1.0) > own(0.0), **and** common(1.0) < common(0.0) | argmax ≤ 0.7 | MIXED |
| **D** Known sites are denser in the robust core than in the contingent fringe | sites per 1,000 km², robust / contingent, per algorithm | ratio > 1.5 for all three algorithms | ratio ≤ 1.0 for any algorithm | MIXED |
| **E** Above about 1,000 m, elevation and volcanic proximity cannot be separated; below, suitability near − far | E235's own pre-registered rules and verdicts (M1; M2 at δ = 0.05) | — reported as E235 reports, for all three definitions | — | — |
| **F** Averaging seeds stabilises the map | disjoint-ensemble Jaccard by k, mean over designs | "averaging helps": J(5) − J(1) ≥ 0.10 for all three algorithms (full and clean, separately) | — | not helps |

Descriptive only (no verdict), reported because the text depends on them:
- TGB − random on its own and on the common background (C084);
- XGBoost hybrid on the uniform background;
- seed turnover;
- median elevation of the fringe;
- J(5) values.

## 6. How the paper uses the results (fixed now)

1. **The data description changes whatever the result.** The 378 rows are presence records compiled from OSM,
   Wikipedia and Wikidata. About a third are modern monuments or misplaced records, and the name-geocoded rows are
   identified. "Archaeological sites" is no longer used for the 378.
2. **The full-set values stay the reported values** for E217–E224 and E235. They examine the sequence as it was
   built. The cleaned-set values go into a supplementary table and one paragraph.
3. **Per claim A–D:**
   - **HOLDS** — the paper says the result holds on the cleaned set, and gives its values.
   - **MIXED** — the paper gives both sets of values and narrows the claim to what holds on both.
   - **FAILS** — the claim is withdrawn, or stated as specific to the uncleaned records, and C081 stays open.
4. **E:** the cleaned-set verdicts are reported next to the full-set ones. Section 3.9 makes no claim stronger than
   the weaker of the two.
5. **F:**
   - The protocol sentence and the abstract quote the **disjoint** agreement actually measured, J(5), and not
     "J ≥ 0.90 from seven seeds".
   - If "averaging helps" HOLDS, "average at least seven seeds" stays, as a floor justified by the disjoint curve
     rising with k, not as a stability guarantee.
   - If it does not hold, the protocol says "report the agreement of disjoint ensembles" instead of prescribing k.

## 7. Run order

Push this design → run A, B, C and D (in parallel) → E (needs D) → F (needs D) → `03_summarise.py clean` →
README with verdicts → independent check of the summary against the raw outputs → manuscript text.

## 8. Changes after the freeze

(none yet)

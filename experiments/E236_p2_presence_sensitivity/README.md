# E236 — P2's real-data results on a cleaned presence set (C081) + disjoint seed ensembles (C083)

**Status:** SUCCESS. The experiment ran as pre-registered and answered its question. **For P2 the answer is a
reversal: claim A FAILS on the cleaned set and claim B is MIXED**; C, D and F hold.
**Line:** 01_spatial · **Serves:** P2 (JCAA #280, round 2) · **Ledger:** C081, C083
**Design:** `DESIGN.md`, pushed as `f9d2eb9` (2026-10-26) before any run on the cleaned set.
**Run:** 26 Oct 2026, 08:58–09:11 WIB. Parts A–D ran in parallel, then E and F.
**Independent check:** a second code path written from DESIGN §5 alone — see §Check.

## Hypothesis

P2's real-data results (E217b, E218, E218b, E221, E235) hold when the models are re-fitted on presences that are
archaeological features with their own coordinates, instead of the 378 records. About a third of those records are
modern monuments or misplaced (C081).

## Method

- **Cleaning rule** R1–R4 (`clean_sites.py`). It drops:
  - rows geocoded from their names (R1);
  - modern monuments (R2);
  - temples of other regions, tourist objects and modern institutions (R3).

  It then keeps one row per coordinate (R4). Result: **161 presences**, 155 from OSM and 6 from Wikidata.
- **Re-runs.** Each original script ran unchanged through `01_run_part.py`. It intercepts the one call that reads
  the site file and redirects only the output folders, so no original result was touched.
- **Disjoint ensembles.** `02_disjoint_ensembles.py` compares two ensembles of k seeds that share no seed
  (k = 1–5, 50 random splits).
- **Decision quantities.** `03_summarise.py` uses the G1 script's definitions; on the full set it reproduces the
  manuscript's values exactly.

## Data

- `data/processed/east_java_sites.geojson` (canonical);
- E217's frame and rasters;
- E221's stored maps, for F full.

## Results (`results/summary_full.json`, `results/summary_clean.json`)

| Claim (DESIGN §5) | Full set, 378 | Cleaned set, 161 | Verdict (cleaned) |
|---|---|---|---|
| **A** redesign hybrid − random, common background (mean of 3 algorithms) | **−0.0142** | **+0.0507** (14/15 seed × algorithm cells positive) | **FAILS** (≥ +0.020) |
| A, same on its own background | +0.0316 | +0.1033 | — |
| A, river feature, common background | +0.0424 (12/12) | +0.0692 (12/12) | (part of A) |
| A, TGB − random: own / common | +0.0032 / −0.0009 | −0.0045 / +0.0012 | descriptive |
| **B** hybrid inflation, own (hybrid) − uniform background | 60/60, mean +0.0370 (+0.005 to +0.084) | 54/60, mean +0.0415 (−0.035 to +0.125) | **MIXED** |
| B, hybrid beats random, per evaluation background (uniform / TGB / hybrid / stratified) | 0 / 0 / 3 / 0 | **3 / 3 / 3 / 3** | — |
| B, TGB inflation | −0.0054 (22/60) | −0.0043 (24/60) | descriptive |
| B, XGBoost hybrid on uniform background | 0.706 | 0.723 | descriptive |
| **C** own-background criterion 0.0 → 0.3 → 1.0 | 0.721 → 0.738 → 0.844 | 0.697 → 0.739 → 0.866 | **HOLDS** |
| C, common background 0.0 → 0.3 → 1.0 | 0.699 → 0.695 → 0.602 | 0.680 → **0.713** → 0.656 | — |
| **D** site density robust / contingent, ratio (RF, XGB, MaxEnt) | 1.93, 4.34, 5.62 | 2.54, 8.51, 5.30 | **HOLDS** |
| D, seed turnover of the top decile | 28.1–47.4 % | 41.6–54.8 % | descriptive |
| **F** disjoint ensembles, Jaccard at k = 1 → k = 5 (XGB, RF, MaxEnt) | 0.548 → 0.759, 0.702 → 0.861, 0.699 → 0.849 | 0.467 → 0.701, 0.517 → 0.725, 0.579 → 0.768 | **HOLDS** (averaging helps, both sets) |
| **E** E235 near − far, XGBoost, below 1,000 m | primary +0.049 / +0.049 / +0.060 (random / TGB / hybrid); 10/30 km −0.002; 15/30 km +0.016 | primary +0.046 / +0.046 / +0.060; 10/30 km −0.000 / −0.001 / +0.018; 15/30 km +0.019 / +0.018 / +0.037 | as E235: primary INCONCLUSIVE; 10/30 INCONCLUSIVE; 15/30 NOT SPECIFIC. **No volcano-specific effect in either set** |

E235's M1 is unchanged, because it depends on land only: above 1,000 m, 97.5 % of the land is near a volcanic centre
and 0.4 % is far.

## Conclusion

1. **The paper's real-data headline depends on the presence data.**
   - On the 378 records as compiled, redesigning the background adds nothing on a common evaluation background
     (−0.014).
   - On the 161 cleaned sites it adds **+0.051**, and the hybrid design beats random on every evaluation background.
   - Own-background evaluation still **doubles** the apparent gain in both sets (+0.103 against +0.051 cleaned;
     +0.032 against −0.014 full), and hybrid's own-background inflation is +0.04 in both.
   - So "the design gain is entirely an evaluation artefact" holds only for the uncleaned records. What holds for
     both: **own-background evaluation overstates the design gain by about +0.04–0.05, and only a common evaluation
     background shows how much real gain is left.**
2. **The TGB steps add nothing in either set**, on either background. The original TGB gain is not reproduced.
3. **The selection criterion still has no interior optimum** (C).
   - On the cleaned set the common-background AUC peaks at a hard-negative share of 0.3 (0.713), the value E013 used.
   - The own-background criterion keeps rising to 1.0, where the common-background AUC is lowest (0.656).
   - The criterion would therefore move a modeller away from the best setting.
4. **Seed averaging helps, but maps are less stable than the paper said** (F). Two disjoint five-seed ensembles
   agree at J 0.76–0.86 on the full set and 0.70–0.77 on the cleaned set. That is not "J ≥ 0.90 from seven seeds".
5. **Under DESIGN §6:**
   - A (FAILS): the claim is withdrawn, or stated as specific to the uncleaned records.
   - B (MIXED): both sets are reported and the claim is narrowed to what holds on both (inflation +0.04).
   - C, D, F: reported as holding.
   - E: reported next to the full set.
   - **How P2's central claim is reframed is the PI's decision.** See `papers/P2_settlement_model/AGENDA_PI_20261026.md`.

## Check

An independent re-derivation was made on 26 Oct by a separate agent. It read only DESIGN §4–5 and the raw files;
it did not see `03_summarise.py` or the summaries. Its script is in the session scratchpad.
- **Every quantity and every verdict matches:** clean A FAILS (+0.0507), B MIXED, C HOLDS, D HOLDS, F helps; on
  the full set all five hold.
- **Its per-algorithm redesign on the cleaned set:** MaxEnt +0.0650, RandomForest +0.0475, XGBoost +0.0397. All
  three are positive.
- **Alternative readings of the design text that it tested changed no verdict:**
  - terrain-only features for A (clean +0.0771);
  - the buffered random comparator (+0.0481);
  - seed-level counts.
- **Two cautions it raised:**
  - **D:** the cleaned set's contingent fringe holds only 5, 15 and 5 sites, so its ratios are fragile, although all
    exceed 1.5 by a wide margin.
  - **C:** the rule tests the own-background criterion, which peaks at 1.0 for every algorithm. The common-background
    curve, however, has an interior peak at 0.3 on the cleaned set.
- **F:** its full-set k = 5 values differ by 0.005–0.009 from E221's split-half figures. The procedures differ (50
  random disjoint splits vs one fixed 5 + 5 split).

## Files

- `clean_sites.py` · `01_run_part.py` · `02_disjoint_ensembles.py` · `03_summarise.py`;
- `results/{A_e217b,B_e218a,C_e218b,D_e221,E_e235}/` — the original scripts' outputs for the cleaned set. Part D's
  90 maps are not in git (`results/.gitignore`); they regenerate with `01_run_part.py D` in about 4 min;
- `results/F_disjoint_{full,clean}.csv`; `results/summary_{full,clean}.json`. Run logs are in `results/logs/` locally (`*.log` is ignored by git).

# E235 — Design (pre-registration): elevation or volcanic setting?

**Written:** 2026-10-20, before any of the comparisons below were computed. **Frozen** when the PI pushes it (D3 in
`papers/P2_settlement_model/REVISION_JCAA_R2_20261019.md`). Any change after the freeze is logged in §8, with date
and reason.
**Line:** 01_spatial · **Serves:** P2 (JCAA #280, round-2 revision), Reviewer 2 point R2-1 (methods), and R2-2/R1-7
(Figure 1).
**Not a thesis test (K9):** no branch of this design supports or weakens the volcanic-burial thesis, and no
statement about burial is drawn from it. It asks what the **model** responds to.

---

## 1. Question

Reviewer 2's question, in our words: when the model gives low or high suitability in volcanic terrain, is that
specific to volcanic settings, or would non-volcanic terrain at the same elevation get the same response?

## 2. What is known before freezing (declared, because it shaped the design)

1. **No model feature encodes volcanoes.** The features are elevation, slope, TWI, TRI, aspect and river distance
   (`E217/01_maxent_benchmark.py`, `FULL_COLS`). Volcano distance is a post-hoc diagnostic only (manuscript Table 1).
   Any volcanic pattern in the output must therefore come through terrain features.
2. **E219 part C** (the round-1 answer) matched upland cells (≥ 200 m) on elevation × slope × TRI × TWI, using
   quintile bins. Volcanic = ≤ 20 km from a canonical centre; non-volcanic = ≥ 40 km. Hybrid map only. Matched mean
   suitability was 0.2249 against 0.1702 (+0.055). **In the top elevation quintile the common strata hold 41,878
   volcanic cells and only 868 non-volcanic ones** (read from `E219/results/e219_terrain_matched.csv` on 2026-10-20).
   The site arm has n = 2.
3. **E221 stores per-cell ensemble suitability** for the three background designs (random, TGB, hybrid) and three
   algorithms, plus the number of designs that put each cell in their top decile. 588,535 cells, x/y in EPSG:32749
   (`E221/results/e221_priority_sets_{xgboost,randomforest,maxent}.npz`). These are the maps behind manuscript
   Figure 5.
4. The threshold δ in §5 was fixed **knowing** E219's +0.055 (point 2). It is the size at which that earlier
   difference would count as material. The sensitivity values in §5 are there because of this.

No elevation-band comparison of the kind below has been computed.

## 3. Data and definitions

- **Cells:** the E221 lattice. Elevation, slope, TRI and TWI come from `base.build_frame()` (E217), joined on
  (x, y). The join must be one-to-one and cover all 588,535 cells; otherwise stop and report.
- **Volcanic setting (primary):** distance to the nearest centre in `data/processed/dashboard/volcanoes_java_full.csv`
  (all 30 centres count for distance, including those outside 111–115°E).
  - **V** (volcanic) = ≤ 20 km; **N** (non-volcanic) = ≥ 40 km. Cells at 20–40 km are excluded (as in E219).
  - **Sensitivity** pairs: (V ≤ 10, N ≥ 30) and (V ≤ 15, N ≥ 30) km.
  - Limitation, stated up front: this definition is by distance to Quaternary centres, not by mapped lithology.
    Parts of the Southern Mountains are Tertiary volcanic rock. No geological map is in the repo, and none is
    fetched for this run.
- **Elevation bands:** 200 m wide, from 0 m up to the maximum. Band 0–200 m is kept as the lowland reference.
- **Usable band:** both V and N have **≥ 25 km²** in the band (cell area = (10 × 30.66 m)² ≈ 0.094 km², so about
  266 cells), and each arm touches **≥ 2 spatial blocks** (`base.assign_blocks`, ~50 km).
- **Overlap ceiling E\*:** the upper edge of the highest usable band.
- **Algorithms:** XGBoost is primary (the algorithm of manuscript Figure 5). RandomForest and MaxEnt are reported
  alongside and do not enter the decision rules.

## 4. Measures (M2 per design: random, TGB, hybrid; M3 once per algorithm, because it compares designs)

- **M1 — overlap:** area of V and N in each band; E\*; the share of all land above E\* that is V. Descriptive; no
  decision rule beyond fixing E\*.
- **M2 — suitability response:** in each usable band, Δ_b = mean suitability(V) − mean suitability(N), after
  coarsened exact matching **within the band** on slope × TRI × TWI (tercile bins, computed within the band;
  strata present in both arms only, weight = smaller arm). The summary Δ̄ is the mean of Δ_b over usable bands,
  weighted by each band's matched weight (the sum of the smaller-arm cell counts over its strata). CI: spatial block
  bootstrap (resample blocks with replacement within each arm, then recompute every band and Δ̄; 1,000 replicates,
  seed 235; percentile interval).
- **M3 — design disagreement:** the same construction, for two outcomes:
  - (a) the share of cells in the contingent fringe (`n_designs_top10 == 1`), V − N, in percentage points;
  - (b) the mean |suit_hybrid − suit_random|.
- **M4 — elevation alone vs elevation + setting** (descriptive, no rule): within usable bands, the partial R² of
  the V/N indicator after a natural cubic spline in elevation (5 knots), for suitability and for |hybrid − random|.
- **M5 — where a difference comes from** (only if M2 is SPECIFIC): within-band means of aspect and river distance
  (the two features not matched), V against N.

## 5. Decision rules (XGBoost; for M2 each design is judged separately; δ = 0.05 suitability for M2 and M3b, 5 percentage points for M3a)

| Verdict for a design | Rule (on Δ̄ and its 95 % block-bootstrap CI) |
|---|---|
| **SPECIFIC** | CI excludes 0 **and** \|Δ̄\| ≥ δ |
| **NOT SPECIFIC** | CI lies entirely within (−δ, +δ) |
| **INCONCLUSIVE** | anything else |

- **Summary for the paper:** for M2, the verdict shared by ≥ 2 of the 3 designs; otherwise "mixed", reported design
  by design. M3a and M3b each get one verdict by the same table.
- **Sensitivity:** the same rules at δ = 0.03 and δ = 0.10, and at the two alternative V/N distance pairs. They are
  reported in full. They do not change the primary verdict but must be mentioned if they disagree with it.
- **If fewer than 2 bands are usable:** M2 and M3 are not judged. The finding is M1 alone: in East Java, elevation
  and volcanic setting cannot be separated.

## 6. What each outcome does to the manuscript (fixed now)

**In every branch:**
- The phrase "directionally consistent with a volcanic-specific effect" (manuscript line 493) is removed.
- The n = 2 site-density comparison stops being evidence: it moves to the supplement as a description, or is cut.
- The text states E\*, and that above E\* elevation and volcanic setting cannot be separated in the study area,
  so the paper makes no statement about volcanic specificity there.
- New Figure 1 shows topography (R2-2).

**Branches (by the M2 summary verdict):**
- **NOT SPECIFIC:** "Where both settings occur (up to E\*), the model's suitability depends on terrain, not on
  volcanic setting, as expected from features that contain no volcanic information." If M3 is also NOT SPECIFIC,
  the same sentence covers the designs' disagreement: the contingent fringe is an upland, remote-terrain
  phenomenon, not a volcanic one.
- **SPECIFIC:** report Δ̄ with its CI and the M5 feature contrast. State that the model reaches volcanic terrain
  through those features. No taphonomic reading.
- **INCONCLUSIVE / mixed:** report as such, with the numbers. No direction is claimed.

Whatever the verdict, the result goes into the manuscript. A result that does not suit the paper is reported
like any other.

## 7. Outputs

- `01_elevation_vs_volcanic.py` (UTF-8 wrapper; reads only the files in §3; writes only to `results/`).
- `results/e235_bands.csv` (M1–M3 per band × design × algorithm × definition), `results/e235_outcome.json`
  (verdicts, E\*, CIs), and `results/e235_response_by_band.png` (suitability and fringe share against elevation
  band, V and N, with CIs and area histograms; a candidate manuscript or supplement figure).
- The Figure 1 redraw is **not** part of E235. It lives in `papers/P2_settlement_model/build_v02_figures.py` and
  uses the same V/N zones.

## 8. Deviations log

*(empty at freezing)*

**2026-10-20, after the run. No deviation from the analysis.** Implementation choices the design left open:
- Tercile edges are computed over each band's V ∪ N cells.
- Aspect is circular, so M5 compares cos(aspect) and sin(aspect) rather than a mean in degrees.
- M5 is computed for every run but reported only as description, because the M2 summary is not SPECIFIC (only the
  hybrid design is).
- The figure draws band means only where an arm has the pre-registered 25 km². A first draft drew empty bands as
  0.1 km² bars and showed the unusable 1,000–1,200 m non-volcanic band; both were corrected before any reading. The
  numbers are unchanged.
- E219 measured distance to the 13 in-bounds centres; E235 uses all 30, as §3 fixed.

**2026-10-20, after an independent audit (`AUDIT_20261020.md`). Still no deviation, but a design weakness is
recorded.** The M3 thresholds (5 percentage points; δ = 0.05 for |hybrid − random|) were set without looking at the
base rates. Below 1,000 m the fringe covers 1–3 % of cells, so M3 could not have returned SPECIFIC; its verdict is
uninformative. The audit's further checks are post hoc and are reported as such in the README (§Audit). They do not
replace the pre-registered verdicts.

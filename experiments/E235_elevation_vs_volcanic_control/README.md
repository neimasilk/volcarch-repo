# E235 — Elevation or volcanic setting? A band-by-band control for P2

**Status:** PRE-REGISTERED (`DESIGN.md`, 2026-10-20) — not run. It runs after the PI's word (D3), once the design
is pushed.
**Line:** 01_spatial · **Serves:** P2 (JCAA #280 round-2 revision), Reviewer 2 R2-1 ·
**Ledger:** C077 · **Revision file:** `papers/P2_settlement_model/REVISION_JCAA_R2_20261019.md`

**Hypothesis:** where volcanic and non-volcanic terrain occur at the same elevation in East Java, the model's
suitability and the background designs' disagreement do not differ by volcanic setting. The reason to expect this:
no model feature carries volcanic information.

**Why:** the reviewer asked whether the model picks out volcanic ground or simply high ground. Our round-1 control
(E219 part C) has almost no non-volcanic land in its top elevation quintile (868 vs 41,878 matched cells), so it
could not answer that question at high elevation. The manuscript still read it as "directionally consistent with
a volcanic-specific effect".

**Method:** see `DESIGN.md`.
- 200 m elevation bands; volcanic = ≤ 20 km from a canonical centre, non-volcanic = ≥ 40 km.
- Within-band matching on slope × TRI × TWI; block bootstrap.
- Run on the stored E221 ensemble maps (three designs × three algorithms); no retraining.
- Decision rules with δ = 0.05, and an overlap ceiling E\* above which the two settings cannot be separated.

**Data:** `experiments/E221_seed_ensemble_stability/results/e221_priority_sets_*.npz`, the E217 frame builder
(`data/processed/dem/jatim_*.tif`), and `data/processed/dashboard/volcanoes_java_full.csv`.

**Result:** —
**Conclusion:** —

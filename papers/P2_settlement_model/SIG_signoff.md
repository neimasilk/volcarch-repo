# SIG sign-off — P2 / JCAA #280 revision v0.2 — 2026-08-05 — run by Claude

**DECISION: 🟢 GO on the manuscript's integrity.** All 9 gates GREEN — G9 (adversarial cross-model
review, completed 2026-08-05) refuted **no** claim; every headline number reproduced from raw CSVs;
its 3 phrasing risks were tightened and re-verified (A8/A9 added). **B1 (co-author sign-off) was
resolved by the PI on 2026-08-05** — the submission is no longer held by authorship approval.

This sign-off **replaces** `SIG_signoff.md` (3 Aug NO-GO). The 3 Aug NO-GO's dominant blocker was
"the manuscript does not exist"; that blocker is gone. The 5 Aug knowledge-base review's blockers
(figures, ENM citations, Table 4 hole, tautology definitions, abstract) were all closed in this
session (see JOURNAL 2026-08-05).

---

## Gate readout

| Gate | 3 Aug | 5 Aug (this run) | Basis |
|---|---|---|---|
| **G1** re-derivation | 🟡 unproven | 🟢 **GREEN** | `verify_headline_numbers.py` re-run 2026-08-05: **64 checks, 60 OK, 4 mismatch** — all four are the *withdrawn* claims the manuscript explicitly corrects (K5, K6, K7, G1c). A7/A8/A9 added (S1 patch + G9 tightening) and passing. |
| **G2** domain-sanity | 🔴 not run | 🟢 **GREEN** | 5 new domain questions for the reframed methods paper, answered against the final prose: `revision_ammo/SIG_G2_DOMAIN_20260805.md`. All pass on the manuscript's own disclosures (Q1 fixed-background robustness via the E218 matrix; Q2 "always" scoped to both sweeps; Q3 selection-criterion framing disclosed; Q4 synthetic-transfer limits in Limitations 1/3/4; Q5 K4 discipline intact). |
| **G3** canonical data | 🟡 unproven | 🟢 **GREEN** | Text carries the 13-centre list and ρ = −0.281 (§3.8); the study-area map (Figure 1) now labels all 13 canonical centres inside 111–115°E (INT-1 visible, not just asserted). |
| **G4** circularity | 🟢 | 🟢 **GREEN** | Unchanged — the paper's subject *is* a circularity; it is named, measured (60/60 own-background inflation) and placed in the title position. |
| **G5** equifinality | 🟡 prose-conditional | 🟢 **GREEN** | The TGB null is reported as **unexplained** (§3.5); E224 (the failed manipulation) is disclosed; the road_dist–river_dist (+0.49) test limitation is disclosed; the secondary-metric reinterpretation is explicitly refused. The one gate that could flip RED on editing stayed intact. |
| **G6** counter-evidence foregrounded | 🟢 | 🟢 **GREEN** | The paper is the disconfirmation of its own published claim; E224 disconfirms our own replacement explanation; INT-1, INT-4, the non-reproducing ρ and K1–K7 are all disclosed in the body or response letter. |
| **G7** reproducibility | 🔴 | 🟢 **GREEN** | `build_v02_figures.py` regenerates all 6 figures from raw per-run files (`e218_stageA_raw.csv`, `e218b_hardfrac_sweep.csv`, `e222_runs.csv`, `e221_stabilisation_curve.csv`, `e221_priority_sets_xgboost.npz`, `east_java_sites.geojson`, `volcanoes_java_full.csv`); `verify_headline_numbers.py` regenerates every headline number. Compiles clean: 27 pages, 0 error, 0 undefined, 0 overfull. Known caveat (G9): the E219 volcano-correlation ρ has no per-run file — it is recorded in `e219_outcome.json` (flagged in the verify script, and the paper's claim that the published −0.163 does not reproduce holds either way). |
| **G8** overstatement scan | 🔴 unrunnable | 🟢 **GREEN** | Formal grep over the final tex for the banned-phrase list (K5/K6/K7/G1c family): **CLEAN**. Remaining "always"/"never" uses all carry a fraction or explicit scope (e.g. "never the truth-worst (0/60)", "always sits at the edge, in both the synthetic and real sweeps"). |
| **G9** cross-model review | 🔴 not done | 🟢 **GREEN** | Adversarial subagent (prompted to REFUTE) re-derived every headline number from the raw CSVs: **no claim refuted, no numerical mismatch, no reviewer-gap blocking**. 3 phrasing risks tightened (see §G9 follow-up) and re-verified via A8/A9. Verdict: *ready for the hostile reviewer.* |
| **G10** human review | ⚪ N/A | ⚪ N/A | Required for P0/masterpiece only. |

**Score: 9 GREEN · 1 N/A.**

---

## What the 5 Aug review required, and where it landed

| 5 Aug blocker / finding | Status this run |
|---|---|
| B1 — Authors' Contributions asserts co-author approval that does not exist | ✅ **RESOLVED 2026-08-05.** PI confirmed Go Frendi is OK with the reversed claim set; `submission_jcaa_v0.2.tex:573` is now factually true. |
| B2 — hinge sentence read backwards (`It is not.`) | ✅ Fixed: "The rise is not real." |
| B3 — 2 real figures of 9; G7 red | ✅ 6 figures, all from raw data. See G7. |
| B4 — `[NEEDS CITATION]` rendered in body; R1-A half-answered | ✅ 5 ENM citations verified against publisher records; marker removed. |
| S1 — Table 4 level claim vs own-background inflation | ✅ Patched: level claim survives (0.706 vs DKNS 0.646, margin ~+0.06), Table 4 reframed as upper bound. Doc 10 A7. |
| S2 — "tautology" undefined | ✅ Test 1/Test 3 defined at first use (§2.4); §3.8 renamed. |
| S3 — study area two sentences | ✅ East Java archaeological background expanded (§1.4). |
| S4 — abstract 328 words, dense | ✅ 216 words, one headline number (+0.042), per-iteration AUC dropped. |

---

## G9 follow-up — adversarial review findings and what changed

The G9 cross-model review (a fresh model prompted to **refute** the manuscript, given the raw CSVs)
found **no refuted claim and no numerical mismatch**: all 12 claim groups reproduced exactly from the
per-run files, including the new A7 (0.706 common-background AUC). It flagged three phrasing risks,
all tightened and re-verified:

| G9 finding | Action |
|---|---|
| Limitation 3 "no design beat uniform on truth" reads absolute; per-run TGB beats random on truth in 27/60 runs (45%), so it holds only as an aggregate (Δ ≈ −0.0004 AUC) | Rephrased: "no background design exceeded uniform on truth recovery by a meaningful margin (aggregate difference ≈ −0.0004 AUC; TGB scored above random in 27/60 runs)". New doc-10 row **A9**, verify check passing. |
| "Every value in this revision is seed-ensembled" overreaches (the E007–E012 ladder is the original single-seed pipeline under examination) | Scoped: "The corrected values reported here are seed-ensembled; the E007–E013 ladder… is the original published pipeline, reproduced as the object under examination." |
| AI disclosure "every headline number re-derived blind from raw per-run outputs" overstated (E219 ρ has no per-run file; ladder from stored results) | Scoped: "re-derived from the raw per-run outputs… or from the stored experiment results where no per-run file exists." |
| The abstract's mechanism ("each design was scored against the negatives it had selected for itself") read as if the whole ladder inflated; actually the 60/60 own-background inflation is **hybrid-specific** (TGB scored *higher* on uniform than own: −0.0054, 22/60) | Added an explicit sentence in §3.2: home-court inflation is specific to the hybrid design; the TGB rungs (E010–E012) are invalidated by the common-background comparison (≈0), not by inflation. New doc-10 row **A8**, verify check passing. |

Non-blocking G9 notes recorded: (a) E219 volcano-correlation ρ lives in `e219_outcome.json` with no
per-run file (flagged in verify script; paper's "published −0.163 does not reproduce" holds regardless);
(b) E224 could not re-run in the original environment — the numexpr/bottleneck incompatibility was
repaired 2026-08-05 (pip upgrade), so it is re-runnable now; (c) the E013 20-seed stability source is
Table S1 (`supplement/e013_seed_stability.csv`), which the manuscript cites.

---

## Non-SIG blockers that still gate submission

1. ~~**Go Frendi's sign-off on the reversed conclusion (B1).**~~ ✅ **RESOLVED 2026-08-05** — PI
   confirmed he is OK with the reversed claim set; the Authors' Contributions sentence is now true.
2. **Title.** Candidate 3 in use; PI confirm or swap to candidate 1 (both vetted safe by claim-set §4).
3. **APC waiver** (£593) still unresolved since 2026-04-07.
4. **Push** the local commits to GitHub (PI).

---

*Replaces the 3 Aug sign-off. Re-run G1 and this sign-off once more immediately before upload, and
again if the manuscript changes in response to G9.*

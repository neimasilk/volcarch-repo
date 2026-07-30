# P2 / JCAA #280 — Revision Plan (R1 decision received 2026-07-23)

**Decision:** *Revisions requested* — "These revisions may then undergo further peer review prior to acceptance."
**Editor:** Dr Philip Verhagen. **Deadline stated:** 4 weeks from 2026-07-23 → **2026-08-20**.
**Reviewer recommendations:** Reviewer 1 = *Resubmit for Review*; Reviewer 2 = *Resubmit Elsewhere* (editor overrode toward revision).
**Manuscript under review:** `submission_jcaa_v0.1.tex` (compiled `submission_jcaa_v0.1.pdf`, submitted 2026-03-11).
**Next canonical version:** `submission_jcaa_v0.2.tex`.

This is the **first revise-and-resubmit** the project has received in 14 months and the first content-level
review that did **not** end in rejection. It is also the closest live shot at a first acceptance.

---

## ⚠ PLAN v2 (2026-07-27, later same day) — THE DIRECTION CHANGED

**E217 refuted the manuscript's central claim.** The MaxEnt benchmark built to answer Reviewer 1 (item
R1-D below) showed that the reported E007 → E013 AUC ladder is an **evaluation artefact**: each design was
scored against negatives it selected for itself. Held to a common evaluation background, background
redesign contributes ≈0 (mean −0.014 AUC, 3/15 paired comparisons positive) while adding one hydrological
feature gives +0.042 AUC (60/60). Full evidence: `experiments/E217_maxent_benchmark/README.md`.

**PI decision taken 2026-07-27: reframe the paper around the corrected finding (Path A).** The paper
becomes a methods contribution — *the apparent benefit of pseudo-absence redesign in archaeological
presence-background modelling can be an artefact of evaluation design; background designs must be compared
on a held-fixed evaluation set.* Same empirical material, reversed conclusion.

**What this changes in the triage below:**

| Item | v1 status | v2 status |
|---|---|---|
| R1-D MaxEnt | new run required | **DONE** (E217) — now the paper's centrepiece, not a supplement |
| R1-A novelty vs ENM | moderate the claim | **Strengthened** — the paper now *contributes* to the ENM/AUC-comparability literature |
| R1-B "tautology-free" overclaim | claim downgrade | **Superseded** — the tautology framing is no longer the paper's spine (see §3 v2) |
| R2-A research question "Poor" | add taxonomy | **Largely dissolved** — the reframed question is single and sharp |
| R2-C two-stage design (E219) | new run required | **Out of scope** — no site-prediction claim remains to decompose |
| R2-F non-volcanic control (E218) | new run required | **Demoted to optional** — the taphonomic claim is withdrawn, which dissolves the ask; run only if time permits, to *show* rather than assert |
| R2-D/E variable tables | new tables | **Unchanged, still required** — and now easier, since the covariate roles are the point |
| R1-C/E/F/G/H/I, R2-B/G/H | rewrite | **Unchanged, still required** |
| INT-1 volcano inventory | must fix | **Unchanged, still must fix** |

Sections 1 (reviewer triage), 4 (integrity items) stand. Sections 2, 3, 5, 6 are superseded where marked.

**Sent to the editor:** `docs/correspondence/EMAIL_VERHAGEN_JCAA_DISCLOSURE_DRAFT_20260727.md` — self-discloses
the error, proposes the reframe, asks whether he wants it as a revision or a fresh submission, and requests
an extension to 2026-09-30. Draft awaiting Pak Amien's send.

---

## 0. Read this first — how to weight the two reviewers

Reviewer 2 rated *Framing of the research question* = **Poor** and *Relevance for the journal* = **Fair**,
and recommended resubmission elsewhere. Reviewer 1 is winnable with additions; **Reviewer 2 is the gate.**
Since the editor states revisions "may then undergo further peer review", assume both reviewers see v0.2.

Therefore: **R2's structural asks (research-question taxonomy, variable roles, non-volcanic control,
two-stage design) get first call on effort — not R1's stylistic asks.** R1's one hard scientific ask
(MaxEnt) is treated at the same priority as R2's structural items.

**Both reviewers independently flagged the same overclaim: "tautology-free".** That is the single
cheapest, highest-yield fix in the whole revision, and the manuscript's own results already
contradict the title (Table 4 verdict is *CONDITIONAL PASS*, T1–T2 in grey zone).

---

## 1. Reviewer items → response type

Legend — **[TEXT]** rewrite only · **[TABLE]** new table/figure from existing results ·
**[RUN]** new analysis required · **[CLAIM]** claim downgrade · **[LIT]** literature work.

### Reviewer 1

| ID | Ask | Type | Effort | Notes |
|---|---|---|---|---|
| R1-A | Finding not novel vs ecological niche modelling; no archaeological ENM examples cited | [LIT][CLAIM] | M | Add Yaworsky et al. 2020, Banks (ecocultural niche modelling), Franklin, Howey, Verhagen & Whitley 2012, Noviello et al. Moderate the novelty claim to: *the transfer-gain ranking under archaeological survey bias*, not the bias problem itself. |
| R1-B | "Tautology-free" claim exceeds evidence; needs definition + support | [CLAIM] | S | **Retitle** (see §3). Add an operational definition of what the tautology suite can and cannot establish. Results already say CONDITIONAL PASS — align title and abstract with them. |
| R1-C | Structure obscures the research question behind methodological complexity | [TEXT] | M | Collapse E007–E013 narrative into one progression table + 2 paragraphs. Move iteration detail to supplement. |
| R1-D | **MaxEnt: benchmark against it, or justify not using it.** "Essential." | **[RUN]** | **L** | → **E217** (§2.1). This is R1's decisive ask, stated twice (§4 and §9). |
| R1-E | Archaeology underdeveloped; roads/accessibility motive revealed only at the end | [TEXT] | M | Move the survey-bias rationale for road rasters into §2.1 where the covariates are first listed. Expand East Java archaeological background. |
| R1-F | Put results in context of current/future heritage management in East Java | [TEXT] | S | Cagar Budaya framework (UU 11/2010), BPCB Jawa Timur survey practice, how a suitability surface would be used for permitting/rescue priorities. |
| R1-G | Jargon undefined: "tautology suite", "conditional pass", "null-model ceiling"; abstract too technical | [TEXT] | M | Glossary box + first-use definitions for TGB, DKNS, MVR, TRI, TWI. **Rewrite abstract**: drop per-iteration AUC values, keep one headline number. |
| R1-H | Citations used descriptively, not to support specific claims | [TEXT] | M | Pass over §1: every citation must attach to a specific proposition. |
| R1-I | Define TRI and TWI in the text, not only in figures | [TEXT] | XS | Trivial. |

### Reviewer 2

| ID | Ask | Type | Effort | Notes |
|---|---|---|---|---|
| R2-A | Research question ambiguous — modelling suitability vs predicting sites vs detecting burial vs correcting survey bias are different questions | [TEXT] | M | Add an explicit §1 taxonomy of the four absence mechanisms (environmentally unsuitable / not surveyed / buried / destroyed) and state that **this paper models suitability only**; the other three are the interpretive frame, not the output. |
| R2-B | What makes the approach specifically *archaeological*? Sites act only as spatial observations | [TEXT] | M | Answer honestly: the archaeological content is in the **background design** (where survey could plausibly have detected a site), not in the label set. That is exactly the paper's own thesis — surface it. |
| R2-C | Proposes a two-stage design: model known-settlement environment → find suitable-but-absent areas → test which mechanism explains absence | **[RUN]**[TEXT] | **L** | → **E219** (§2.3). Partially reframing, partially new analysis. |
| R2-D | Cannot tell which variables are in/out per experiment; model not clearly reproducible | [TABLE] | S | **Per-experiment covariate inclusion matrix.** Highest value-per-hour item in the revision. |
| R2-E | Separate variables by analytical role (suitability / accessibility-survey / preservation) | [TABLE][TEXT] | S | Role column in the covariate table + one paragraph on why roles must not be mixed in interpretation. |
| R2-F | **Elevation/slope may drive low suitability in rugged terrain regardless of volcanism — compare volcanic to environmentally similar NON-volcanic uplands** | **[RUN]** | **L** | → **E218** (§2.2). This is R2's decisive ask and a genuine falsification test. |
| R2-G | Low suitability ≠ buried site; high suitability ≠ site exists | [TEXT] | S | Discussion. Directly aligned with the project's own SIG discipline. |
| R2-H | Fig 1 too simple / doesn't show data integration; Fig 1 & 4 label overflow; Fig 5 must explain how importance is computed and interpreted | [TABLE] | M | Confirmed against `submission_jcaa_v0.1.aux`: **Fig 1** = interdisciplinary framework, **Fig 4** = AUC/TSS progression, **Fig 5** = feature importance across E007–E013. Redraw Fig 1 as a data-flow diagram; fix label boxes; expand Fig 5 caption (gain vs SHAP, and what elevation/slope dominance does *not* mean). |

---

## 2. New analyses required

### 2.1 E217 — MaxEnt benchmark across the background ladder *(answers R1-D)*

**Design.** Run maxnet-style MaxEnt under the *identical* spatial block CV folds and the *identical*
pseudo-absence/background designs used in E007–E013, and report a head-to-head table
(MaxEnt vs XGBoost vs RandomForest × background design).

**Why this framing is the strong one.** The paper's central claim is that *pseudo-absence realism
dominates feature accumulation*. If the same monotonic gain appears in MaxEnt, the claim becomes
**algorithm-independent** — which is a stronger result than the current single-family evidence. It also
answers R1's implicit question directly: target-group background *is* the bias correction developed in
the MaxEnt literature (Phillips et al. 2009, already cited); we are applying it to tree ensembles and can
now show the effect is not an artefact of boosting.

**Honest failure branch (pre-registered).** If MaxEnt matches or beats XGBoost at E013, we report that
plainly and reframe the algorithm choice as a convenience/interpretability decision rather than a
performance claim. That outcome does **not** damage the paper's thesis; concealing it would.

**Dependency:** `pip install elapid` (currently **not installed**; xgboost/sklearn/rasterio/shap all present).
Fallback if elapid is unworkable: maxnet-equivalent via regularised logistic regression on
hinge/linear/quadratic feature transforms.

### 2.2 E218 — Terrain-matched volcanic vs non-volcanic upland control *(answers R2-F)*

**Design.** Coarsened exact matching (or propensity matching) of raster cells on
elevation × slope × TRI × TWI, comparing:
- **volcanic uplands** — matched cells within *n* km of a canonical volcanic centre;
- **non-volcanic uplands** — matched cells far from any volcanic centre (East Java's Southern Mountains
  karst and the Kendeng/northern limestone hills).

Then compare, between the two matched sets: (a) mean predicted suitability, (b) observed site density per km².

**Pre-registered decision rule — write this before running:**

| Outcome | Reading | Consequence for the paper |
|---|---|---|
| Suitability similar, site density **similar** | Model captures general terrain constraint; no volcanic-specific signal | **R2 is right.** Drop the taphonomic interpretation; paper becomes purely methodological (which R1 already values). |
| Suitability similar, site density **lower in volcanic uplands** | Terrain is matched, so the density gap is *not* suitability | Supports the taphonomic/survey reading — and does so under R2's own test. |
| Suitability **differs** between matched sets | Matching failed, or a non-terrain covariate drives it | Diagnose before interpreting. |

All rasters needed are on disk (`data/processed/dem/`). Related prior work: **E178** (karst as hidden factor)
is directly relevant — the Southern Mountains are the karst comparandum.

### 2.3 E219 — Two-stage "suitable but absent" decomposition *(answers R2-C)*

Stage 1 is already done (suitability from known sites). Stage 2: take high-suitability cells with no known
site and decompose them descriptively across candidate explanations — volcanic deposit/burial depth
(E166 burial-depth raster exists), road distance / survey access, terrain, hydrology. Output is a
contingency table, **not** a causal claim. Explicitly report how much of the "suitable but empty" area
remains unexplained.

---

## 3. Title — SUPERSEDED BY v2

**v1 process note (kept for audit trail):** the original title *Tautology-Free Settlement Suitability
Modeling…* was flagged as an overclaim by both reviewers, and on 2026-07-27 Pak Amien selected
*Tautology-Controlled Settlement Suitability Modeling in East Java Under Survey and Taphonomic Bias*.

**That choice is now moot.** E217 changed what the paper is about: the spine is no longer tautology
control but the evaluation-background artefact. New candidates, to be settled when the v0.2 framing is
drafted:

1. *The Evaluation Background Is the Result: Why Pseudo-Absence Redesign Appears to Improve Archaeological Predictive Models*
2. *Comparing Pseudo-Absence Designs Requires a Fixed Evaluation Background: A MaxEnt-Benchmarked Reanalysis from East Java*
3. *An Evaluation Artefact in Presence-Background Archaeological Modelling: Evidence from East Java and a Corrected Protocol*

Note the honest shape of the retitling: v1 was a claim **downgrade** to match evidence; v2 is a claim
**replacement** because the evidence turned out to support a different proposition. Both are SIG-legitimate.
Rewording the original claim to survive the critique would not have been.

---

## 4. Integrity items found while preparing this plan (not raised by reviewers)

**INT-1 — Volcano inventory defect, same class as the one that sank P7. MUST FIX.**
`enhanced_tautology_tests.py` (and `experiments/E013_settlement_model_v7/01_settlement_model_v7.py`) hardcode
**7** volcanoes: Kelud, Semeru, Arjuno-Welirang, Bromo, Lamongan, Raung, Ijen.
The canonical file `data/processed/dashboard/volcanoes_java_full.csv` contains **13** centres inside the
paper's own stated bounds (111–115°E): the 7 above **plus Lawu, Wilis, Kawi-Butak, Penanggungan,
Iyang-Argapura, Baluran.** Kawi-Butak and Penanggungan sit in the Malang–Mojokerto site concentration,
so the omission distorts the distance field precisely where the sites are.

*Impact is contained but real:* volcano distance is **not** a training feature, so model AUCs are unaffected.
It affects the **Test 1 tautology diagnostic** (reported ρ = −0.163) and Figure 2's "major volcanic centres".
Recompute both with the canonical 13 before resubmission, and state the correction in the response letter.
Fixing this ourselves, unprompted, is also a credibility asset with a reviewer who is already sceptical.

**INT-2 — `revision_ammo/anticipated_critiques.md` is STALE. Do not use verbatim.**
It was written 2026-03-12 against an earlier version and contradicts the submitted manuscript in two
material ways: (a) it claims the temporal split is **chronological** (pre-1000 CE vs post-1000 CE) — the
submitted paper's E014 split is an **accessibility proxy** (road distance ≤1 km vs >1 km); (b) it claims
volcanic proximity ranks high in SHAP — the submitted model **excludes volcanic predictors entirely**.
Using those responses in a reviewer letter would misdescribe the analysis. File is now flagged; supersede
it with this document.

**INT-3 — SIG G1 applies at resubmission:** re-derive every headline number (seed-averaged AUC 0.751,
best run 0.768, TSS, null-model gaps, all four tautology-test metrics) blind from the code before upload.

---

## 5. Timeline to 2026-08-20

| Window | Work | Owner |
|---|---|---|
| Jul 28 – Aug 3 | INT-1 volcano fix + recompute T1/Fig 2; E217 MaxEnt; R2-D/R2-E covariate tables | Claude |
| Aug 4 – Aug 10 | E218 matched control; E219 absence decomposition; **go/no-go on extension request** | Claude |
| Aug 11 – Aug 17 | v0.2 rewrite: title, abstract, §1 taxonomy, ENM literature, heritage-management framing, jargon pass, figures | Claude → Pak Amien review |
| Aug 18 – Aug 20 | SIG gate (G1 blind re-derivation) + G9 cross-model review + response-to-reviewers letter + upload | Both |

**Extension:** the editor invites early notice if 4 weeks is not enough. Recommendation — **work to 2026-08-20,
and decide by Aug 10** whether E218/E219 outcomes require a 2-week extension request. Requesting one is
routine and costless; resubmitting a rushed package to a reviewer who already said "resubmit elsewhere" is not.

---

## 6. Decisions needed from Pak Amien

1. **GO on the three new experiments** (E217 MaxEnt, E218 matched control, E219 absence decomposition)?
   Without E217 and E218 the revision does not answer either reviewer's decisive ask.
2. **Title choice** (§3, options 1–3) — or veto the downgrade.
3. **`pip install elapid`** for the MaxEnt benchmark — OK?
4. **APC waiver:** requested 2026-04-06, acknowledged by Verhagen 2026-04-07, **never resolved**.
   JCAA APC is £593 against an absolute zero-APC constraint. Recommendation: re-raise it in one line
   when submitting the revision — not as a blocker on doing the work now.
5. **Accept the pre-registered failure branch of E218** (if matched non-volcanic uplands behave identically,
   the taphonomic interpretation is dropped and the paper becomes purely methodological)?

---

*Prepared 2026-07-27. Supersedes `anticipated_critiques.md` (see INT-2).*

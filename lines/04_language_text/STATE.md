# STATE — Line 04 LANGUAGE & TEXT

**Updated:** 2026-10-05 · **Temperature:** 🟢 P8 IN REVISION (conditional acceptance), plus one overdue rewrite

> **2026-10-05 — P8 decision landed.** *Oceanic Linguistics* (OL-03-2026-11): **accepted pending revisions** (editor Sander Adelaar, 14:17 WIB).
> Not a final acceptance; no deadline stated. Reviewer 1 (specialist in Sulawesi languages): 14 points, mostly wording/classification,
> two needing analysis (Makasar vs a published 62 % figure; glottal-stop orthography) and a request to release the residual lists.
> Reviewer 2: 24 highlights in the PDF, all "explain this for non-technical readers" (AUC, SHAP, CV, `E0xx` codes, Figure 2, Table 5).
> The editors ask for extra attention to readability, Section 3 above all. **Full ledger, gates and order of work:
> `papers/P8_linguistic_fossils/REVISION_OL_20261005.md`.** Journal = Scimago **Q2** (not Q1). The reviewed text = `draft_v0.1_anonymous.*` (verified).
> Gates that bind the revision: G1 (no rewording of a central critique), G14 (14-day freeze), G15 (fee check, URL + date), G16 (**the PI's own prose**, response letter included).

> **2026-10-05 evening — P8 verified and re-tested (E227, E228).** Work plan with the point-by-point facts:
> `papers/P8_linguistic_fossils/REVISION_WORKPLAN.md`. Reviewer answers that came out clean: Makasar (39.3 % retained from PMP,
> 25.9 % coded otherwise, 34.8 % uncoded → the published 62 % is reproduced as 60.7 %); Tolaki (70.9 % of uncoded forms have a
> look-alike within Bungku–Tolaki, chance 10.5 %); Muna *dh* (two forms, both converted); glottal conventions (partial dependence,
> up to −0.016 AUC). **Against the submitted text:** two label sets (356 / 438); the "PAn cross-check" was a 15-concept filter;
> κ 0.61 was in-sample (0.31 out of fold); "fewer prefixes" is reversed (37.0 % vs 25.2 %); headline 0.763 includes the language's
> identity (form only 0.672, form + meaning 0.727); the cross-language test p = 0.569 is replaced by a permutation test that finds a
> small shared component (p = 0.0001); the 16-language geographic sentence is not supported. `revision_ammo/anticipated_critiques.md`
> describes a different paper — flagged, do not use. References: 2 invented `.bib` entries (uncited), `ross2005` cited with a false
> venue, details in `REFERENCE_CHECK_20261005.md`. **Submitted files untouched; `revision_v0.2/` holds a working copy with only the
> reviewer-dictated term changes. Seven PI decisions (D1–D7) come first.**

> **2026-10-01 evening — downstream audit (`docs/research_notes/DOWNSTREAM_AUDIT_C023_C032_20261001.md`; ledger C039–C043):** this line's E085, E105 carry correction notes; headline statuses changed in the index. Do not cite their old headline numbers.

> **2026-10-01 inbox (orbit re-entry; [BRIDGE → 04]):** the objective-answer synthesis
> (`docs/research_notes/OBJECTIVE_ANSWER_20261001.md`) puts two text tests on this line. **T1** (regional onset
> of Sanskrit inscriptions, with DHARMA as the dating authority; documentation of H0, not a test) — draft
> `docs/drafts/T1_REGIONAL_ONSET_DESIGN_DRAFT_20261001.md`. **T7** (detection calibration: villages named in
> 8th–10th c. inscriptions vs recorded settlements; shared with line 05) — feasibility scan run 2026-10-01.
> Also: the reviewers rate the **P8 negative result** (no coherent shared substrate) and **E090 v7** as
> legitimate negative findings; E131's hand-typed onset table is not to be cited until corrected.
> P5 (C018) still needs PARKED.md or a reframe.

---

## Current position

- **P8** — **conditionally accepted at *Oceanic Linguistics* (2026-10-05)**; revision plan in
  `papers/P8_linguistic_fossils/REVISION_OL_20261005.md`. First positive decision of the series; recorded in
  `docs/WORKSTATE.md`, memory and JOURNAL. The next step is the PI's: where the revision sits in the queue relative to
  the other live work, and the order in the plan (wording → terms → analyses → Section 3 → freeze → response letter).
- **P5 is the only writable paper in this line, and its rewrite is overdue** (planned ~June 2026).
  Target *Asian Ethnology*: zero APC, Scopus Q2, humanities reframe as *indigenous knowledge
  resilience*. The strategy document is ready; the rewrite is not started.
- **P16 is parked** and stays parked. **P9** is on HOLD behind P2/P8 by design. **P14** is closed.

---

## Next actions for Claude

- [x] **P8 revision analyses and G1 audit** — done 2026-10-05: `E227` (85 statements checked) and `E228` (S1–S7, pre-registered;
      amendment A2 is post hoc). Release files for reviewer 1's request: `experiments/E228_p8_revision_analyses/release/`.
- [ ] **P8 — queue for the next session: `docs/HANDOFF_20261005.md` §4.** PI on 2026-10-05 ±16:00: carry out the recommendation, no hurry.
      D1–D6 proceed as recommended; sending the editor note (D7) and the prose (G16) stay with the PI.
- [ ] **P8, working under D1–D6** (`REVISION_WORKPLAN.md` §2, §7 step 4): tables and Figure 1 for the chosen model, the
      examples table, the Makasar table, a `VENUE.md` look at recent OL articles; later the language check and G1/G8/G11 on the PI's new
      text and on the PDF to be uploaded. **Not** the manuscript prose (G16).
- [ ] **P8 domain check** — the forms marked ⚠ in `E228/README.md` and `E227/results/pan_rescued_75.csv` need a historical linguist
      of Sulawesi before any single form is discussed in print (G10).
- [ ] **P5 rewrite** — the highest-value unblocked item in this line, and the cheapest route to
      *exposure* (a submitted manuscript) that does not depend on any external human. Start from the
      existing strategy doc; the reframe is already decided, so this is execution.
      ⚠ Before writing: check whether P5 quotes any volcano-distance number — if so it needs
      [02_taphonomy](../02_taphonomy/)'s WS-E sweep first.
- [ ] Confirm the `E027` upgrade (from `E107`/ADV-5) is actually reflected wherever E027 is cited —
      the upgrade was recorded in the scorecard but may not have propagated into paper text.
- [ ] For any P5/P9 claim, add the explicit corpus-dependency sentence (DHARMA 268, closed
      2026-04-09) rather than leaving the monoculture implicit.

## Do NOT do

- ❌ Edit `arxiv_submission.tex` or post an arXiv v2 for P8 (the preprint is public) before the PI decides; revision edits go in a separate revision copy and pass the gates.
- ❌ Quote from the submitted P8 text any of: "438 (26.5 %)", κ = 0.61 / 266 consensus forms, "fewer prefixes", p = 0.569, AUC 0.763
  as "phonological only", the per-language rates of the 16 extra lists (Acehnese 63 % etc.). E227/E228 supersede them.
- ❌ Use `papers/P8_linguistic_fossils/revision_ammo/anticipated_critiques.md` (describes a different paper).
- ❌ Answer R1's Makasar point or the glottal-stop point by rewording — they need analysis (SIG: never reword a central, valid critique). If the analysis weakens the "fingerprint" claim, downgrade the claim.
- ❌ Cite any reference the reviewers supplied before the PI has read it (citation-integrity rule).
- ❌ Unpark P16 by reframing. `PARKED.md` names two specific unpark conditions; anything else is the
  rewording-instead-of-fixing move the SIG forbids.
- ❌ Open new DHARMA-only experiments. The corpus is closed and the monoculture is a known weakness.
- ❌ Start P9 ahead of its queue position.

## Blocked / external

| Item | Blocker |
|---|---|
| P8 formal acceptance | **first the PI's decisions D1–D7**, then his revised text; then the editors' sign-off (no deadline stated) and the formal acceptance letter |
| P9 | its own trigger ("after P2 or P8 decision") has now been met by P8's conditional decision, but G13/G14 still apply → PI decision |
| P16 unpark | needs an unsupervised, non-circular convergence design that passes a shuffle control |
| I-025 Krama comparison, I-026 Osing substrate | need Tegal/Banyumas wordlists or fieldwork contact (see `docs/TRIGGER_MAP.md`) |

## Inbox

- **Jatim glass beads lead** (`docs/research_notes/JATIM_BEADS_LEAD_2026_06_08.md`): East Java glass
  beads 5th–8th c CE, npj Heritage Science 2024, Datong Northern Wei tomb → Java. It is
  durable-trace/selective-survival evidence of indigenous sophistication but **is not pre-400 CE** —
  the pre-400 angle would need Sembiran/Bali verified. Good material for P5's resilience reframe.
- `E204` (bronze drums) reframes selective survival and is not yet used in any live manuscript.

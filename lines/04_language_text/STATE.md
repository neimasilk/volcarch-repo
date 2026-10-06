# STATE — Line 04 LANGUAGE & TEXT

**Updated:** 2026-10-06 (11:30) · **Temperature:** 🟢 P8 IN REVISION (conditional acceptance) — readings corrected in two rounds today; the next steps are the PI's

> **2026-10-06, 11:30 — P8: three adversarial reads, three rounds of correction. START HERE: `REVISION_WORKPLAN.md` §10.7.**
> The block below this one (written 08:31) says "everything Claude can prepare is prepared". That was written before the
> first adversarial read returned; what follows supersedes it wherever the two differ.
> - **What happened.** The session was cut off at 08:54 (API error, then a power cut; no file damaged) and resumed at 10:05.
>   Read 1 (Opus; 17 findings, 7 high): every number sound, several readings not. Read 2 (Opus; 14 findings, 3 high): three
>   of the *replacement* readings were wrong as well. The orchestrator re-derived the second reader's counts with its own
>   script before changing anything. Number trace by four Sonnet readers: 902 statements, 890 matching, 12 small
>   differences, all corrected. Read 3 (Opus; closure + review of the post hoc script): the script computes what it says
>   (coarser strata and a logistic model agree); 9 new findings, 2 high — round 2 had **over-corrected Makasar** and left
>   the E228 README untouched at S2. Its counts were re-derived (tables P8–P12) and a third round of corrections made.
>   Round 3 was traced number by number (about 470 statements; all figures match; one reading — the "Konawe pair" — was
>   found to be a synonym effect and withdrawn) but **not** read adversarially again.
> - **E231** `experiments/E231_p8_what_coded_means/` — tables A–H (descriptive; four unadjusted paired tests) and post hoc
>   tables P1–P12 (`02_posthoc_checks.py`). What it shows, in the order that matters for the paper:
>   1. **"Coded" is not one kind of label.** 44 of Muna's 185 coded forms are coded only because they recur in ABVD's second
>      Muna list (Wuna); without them Muna is 35.6 % uncoded, not 15.5 %. Tolaki's five dialect lists were not cross-coded.
>      5 of the 11 forms ABVD flags as loans carry a set number, one of them in the PMP entry's own set.
>   2. **Makasar has more uncoded meanings than Bugis and Sa'dan Toraja** (34.8 % against 23.4 % and 19.0 %); "lower
>      retention" and "fewer shared sets" are that one fact counted again. Whether the uncoded forms are uncoded cognates
>      or replaced words is **not decided**; a mechanical screen does not favour "merely uncoded" (2 of 75 resemble the PMP
>      form; 10 of 80 a Bugis or Sa'dan Toraja form). Compatible with the divergence reported in the literature; no proof
>      of it, nothing on its cause. ⚠ **Nothing on Makasar is to be written** before the PI has read the source of the
>      published 38 % (a specialist's look at `results/P9_makasar_uncoded_meanings_for_specialist.csv` is optional).
>   3. **"Fewer prefixes" is withdrawn and "more prefixes" is not established:** the onset-string input is not separable
>      from length (1.09 [0.80, 1.48] with all letters held fixed; 2.51 with the letters after the string held fixed); nor
>      is the nasal input. What holds at equal length: written glottal mark (2.60) and action meaning (2.22); consonant-
>      letter clusters are weaker (1.38 [0.98, 1.93] once the glottal mark is held fixed).
>   4. **Tolaki:** 70.9 % of its uncoded forms have a look-alike within Bungku–Tolaki (coded forms 95.9 %). By meaning (first
>      form of each meaning) list 674 is coded for 38.3 % and the five Tolaki dialect lists for 36–41 %, against a median
>      of 46.7 % in 42 comparison lists: the Tolaki lists sit on the lower side of a thinly coded subgroup, by a moderate
>      margin that depends on the unit (a first version of this item read a synonym effect in the Konawe list as a
>      coding difference; caught by the last number trace). Twelve forms look like the PMP form (about two
>      expected by chance). Nothing on origin.
>   5. **Cross-list test:** among uncoded forms a small excess of similarity; the coded class is not examined, so neither
>      "a shared layer" nor "no large shared layer" follows. Inside and outside South Sulawesi are not distinguishable.
> - **Files corrected:** the outline (every item separates *measured* from *reading*; §0 and §6 are a proposed argument);
>   the work plan (§10.7 rewritten; cells in §1, §3, §4, §5, §6, §8, §10.3 edited in place); the editor note (fact list in
>   Indonesian; the Makasar sentence held; "fewer prefixes" withdrawn without announcing "more"); READMEs of E227–E231; an
>   amendment in the E230 design; ledger C046/C048/C050/C053 corrected and **C055–C065** added; the reviewer ledger rows 2 and 10.
> - **New:** `experiments/E228_p8_revision_analyses/release/README.md` — column description of the two release files
>   (Sonnet, checked line by line; it reproduced all 1,357 scores and all distances from the released strings).
> - **What the PI can write from now** (third reader's assessment, after its corrections were made): outline §1, §3, §4,
>   5.1, 5.4, 5.6, 5.8, §8; in 5.5 size, glottal mark, action meaning; in 5.3 the look-alike result. **Not yet:** anything
>   on Makasar (the PI reads the source of the 38 % first); 5.7 (waits on D4). Etymologies of single forms, hyphens, final
>   *k* and the glottal stop in Muna and Wolio are simply **not written as claims** — no specialist is waited for.
>   The release description is a **draft** (ledger C065).
> - **For the PI (new today, besides D1–D10):** (1) ⚠ the arXiv *paper password* of the P8 preprint has been in the public
>   git history since April (C061; redacted in the working tree) → ask arXiv for a new one; (2) **D4 has three options**
>   (§10.7; (a) recommended); (3) read the source of the Makasar 38 % before any comparison is written; (4) a note on the
>   second Muna list under Table 1 (recommended).
> - **A Sulawesi specialist is not a blocker** (recommendation revised 12:00 after the PI asked): forms are printed as ABVD
>   data, "not assessed etymologically"; reviewer 1 is the specialist and will see the revision; G10 = one linguist reads
>   the finished draft during the G14 freeze. Points a specialist could settle if one is at hand (C060): whether Muna and Wolio lack a glottal stop or merely do not write it; final *k*
>   in Sa'dan Toraja; what the hyphen marks in the Bugis and Sa'dan Toraja lists; the numeral compounds; *ma-* in the
>   look-alike pairs; every single lexical example.
> - **Not done, by design:** no manuscript prose, no response letter, nothing sent or uploaded, no Zenodo, no arXiv. Committed on main at the PI's word (`450aef6` and a small follow-up); not pushed.

> **2026-10-06 — P8: the handoff queue is done; one more defect found on the way.** Start from
> `papers/P8_linguistic_fossils/REVISION_OUTLINE_20261006.md` (what each section has to contain, facts only, in Indonesian so
> that the English is the PI's) and `REVISION_WORKPLAN.md` §10 (what is new since 10-05).
> - **E229** — tables T1–T7 and Figure 1 for the 25-input model with the form-only model beside it; 75 anchors against
>   E227/E228; T1 and the XGBoost rows re-derived by a second script (163 cells, no difference). Two things the old tables hid:
>   as a yes/no classifier the model is close to "always coded" (0.719 vs 0.677) and **below the majority answer for a held-out
>   list in 5 of 6 lists** — only the ranking carries over; and the SHAP arrows describe the classifier, not the vocabulary
>   (all 285 consonant-final forms are in the three South Sulawesi lists, so a form input also identifies the source list).
> - **E230** (pre-registered; revisits E041/E042) — the "IPA / syllable / no-length" numbers of §3.4 reproduce but belong to a
>   26-input model with the language-identity code, against a hybrid baseline. For the revision models: digraph conversion
>   leaves discrimination unchanged (75 forms); **"does not depend on form length at all" is withdrawn** (−0.014 / −0.023 when
>   both size inputs go); the Muna sentence is withdrawn. The abstract's "not orthographic artifacts" has no test behind it.
> - **Checked against the original email and annotated PDF:** nothing missing from the ledger; R1-12 holds a direct question
>   (false *negative*?) whose answer is yes; a conditional letter of acceptance is available already now (the ledger had said "after").
> - **Venue (`VENUE.md`):** the journal requires a **Word** file for the final version (no LaTeX); ≤ 30 pages; abstracts of
>   120–160 words (ours about 260); figures ≤ 312 pt; fees stated nowhere → G15 open until the editor answers.
> - Also ready: corrected `.bib` in `revision_v0.2/` (8 corrections, 2 deletions); draft note to the editor
>   (`docs/correspondence/EMAIL_OL_EDITOR_P8_NOTE_DRAFT_20261006.md`, **not sent**); Word tables
>   (`experiments/E229_p8_revision_tables/results/tables_docx/`).
> - **Not done, by design:** no manuscript prose, no response letter, nothing sent or uploaded, no Zenodo, no arXiv.

> **2026-10-05 — P8 decision landed.** *Oceanic Linguistics* (OL-03-2026-11): **accepted pending revisions** (editor Sander Adelaar, 14:17 WIB).
> Not a final acceptance; no deadline stated. Reviewer 1 (specialist in Sulawesi languages): 14 points, mostly wording/classification,
> two needing analysis (Makasar vs a published 62 % figure; glottal-stop orthography) and a request to release the residual lists.
> Reviewer 2: 24 highlights in the PDF, all "explain this for non-technical readers" (AUC, SHAP, CV, `E0xx` codes, Figure 2, Table 5).
> The editors ask for extra attention to readability, Section 3 above all. **Full ledger, gates and order of work:
> `papers/P8_linguistic_fossils/REVISION_OL_20261005.md`.** Journal = Scimago **Q2** (not Q1). The reviewed text = `draft_v0.1_anonymous.*` (verified).
> Gates that bind the revision: G1 (no rewording of a central critique), G14 (14-day freeze), G15 (fee check, URL + date), G16 (**the PI's own prose**, response letter included).

> *(Corrected 2026-10-06 — in the block below read: the Makasar figure is a lower bound and is not to be called
> "reproduced" or "consistent" before the source is read; p ≤ 0.0001; "fewer prefixes" is withdrawn, "more" is not
> established; nothing follows about a shared layer. `REVISION_WORKPLAN.md` §10.7.)*
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
- [x] **P8 — queue of `HANDOFF_20261005.md` §4, items 1–5** — done 2026-10-06 (E229, E230, corrected `.bib`, outline, editor-note
      draft, `VENUE.md`). Item 6 waits for the PI's text; item 7 (downstream audit C050) is an orbit task.
- [ ] **P8 — after the PI has written:** language check; G1 on the new text (every number against E227–E231); G8/G11 scans; check of
      the file to be uploaded; then the 14-day rest (G14). If the PI chooses to write in the journal's Word template (D8), the
      checks run on the `.docx`.
- [ ] **P8 — if the PI wants it:** the working copy converted into the journal's Word template as a starting file (mechanical; not
      done because it depends on D8).
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
  as "phonological only", the per-language rates of the 16 extra lists (Acehnese 63 % etc.), the robustness numbers of §3.4
  (0.772 → 0.774, "no length" 0.769), "does not depend on form length", F1 = 0.82 as a detection score. E227–E230 supersede them.
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
| P8 formal acceptance | **first the PI's decisions D1–D10** (`REVISION_WORKPLAN.md` §2, §10.5) and the note to the editor, then his revised text; then the editors' sign-off (no deadline stated) and the formal acceptance letter |
| P9 | its own trigger ("after P2 or P8 decision") has now been met by P8's conditional decision, but G13/G14 still apply → PI decision |
| P16 unpark | needs an unsupervised, non-circular convergence design that passes a shuffle control |
| I-025 Krama comparison, I-026 Osing substrate | need Tegal/Banyumas wordlists or fieldwork contact (see `docs/TRIGGER_MAP.md`) |

## Inbox

- **Jatim glass beads lead** (`docs/research_notes/JATIM_BEADS_LEAD_2026_06_08.md`): East Java glass
  beads 5th–8th c CE, npj Heritage Science 2024, Datong Northern Wei tomb → Java. It is
  durable-trace/selective-survival evidence of indigenous sophistication but **is not pre-400 CE** —
  the pre-400 angle would need Sembiran/Bali verified. Good material for P5's resilience reframe.
- `E204` (bronze drums) reframes selective survival and is not yet used in any live manuscript.

# P2 round-2 revision — worksheets for the PI's writing sessions 2–4

**Prepared:** 2026-10-21 by Claude, after writing session 1, on the PI's "kalau ada yg perlu kamu kerjakan kerjakan".
**Purpose:** each session starts from a list of what to change, where it is now in `submission_jcaa_v0.3.tex` and the
fact to keep. The PI then only has to write.
- The facts come from `REVISION_R2_WRITING_PACK.md`; this sheet re-maps them to **v0.3 line numbers** (as of commit
  `106f821`).
- Line numbers shift after each insertion, so every row also quotes the opening words.

**How we work (same as session 1):**
1. The PI writes in Indonesian or rough English, in the chat or in any file.
2. Claude translates faithfully, checks every number against its source and inserts the text.
3. Every change beyond translation is told to the PI first and logged in `PI_PROSE_LOG_R2.md`.

Under JCAA's AI policy this is "language assistance" (`VENUE.md` A5), and it is declared.

**Done in session 1:** Section 3.7 (C078), new Section 3.9 + Figure 7 (E235), the verdict labels, one Limitations item.

**Session 2 (2026-10-22):** **all three items done** — the research question in the first paragraph (log passage 4),
**Table 1, Key terms** (passage 5) and "no interior optimum" at line 136 (passage 6). Also done: the 3B row for lines
221–222 (C080, passage 7). **Line numbers below are those of 21 Oct.** In the file after session 2, add **+1** to lines 70–84, **+28** to
lines 85–108, **+30** to lines 109–221 (e.g. "strong gains" 115 → 145) and **+32** after line 222 (e.g. 231 → 263).
The quoted openings still find each place.

**Session 3 (2026-10-22): done** — 41 language edits (log "Session 3, part 1") and the PI's blocks A–E (passage 8).
**Session 4, part 1 (2026-10-22): done** — keywords 7 → 6, Authors' Contributions, Data Availability + E235, S7–S8
captions kept (log "Session 4, part 1"). **Session 4, part 2 (2026-10-22):** abstract, conclusions, the "strong
gains" sentence and acknowledgements **drafted by Claude at the PI's request** (log "Session 4, part 2"). **Open:** the AI
statement and the style swaps → `AGENDA_PI_20261026.md` (26 Oct), which also starts the "After session 4" steps below.

---

## Session 2 — Introduction and key terms (R1-2, R1-3)

**What R1 asked (paraphrase):**
- state the research question earlier, more clearly and in a more teaching way;
- explain the statistics for readers who are not statisticians.

**Current structure (v0.3):**

| Lines | Part | Note |
|---|---|---|
| 65–69 | Opening paragraph | The question is already there (line 68: "how do you know that redesigning the background points improved a presence–background model?"), but it is wrapped in "a narrower question than it first appears to" and followed by "uncomfortable … about our own model" |
| 71–83 | 1.1 What presence–background models are | Technical from the first sentence (TGB, availability, spatial blocking). The natural home for a plain explanation |
| 85–101 | 1.2 Four reasons a site may be absent | Has two history sentences (lines 89, 101) → session 3 |
| 103–124 | 1.3 Contribution and relation to prior work | Has five history sentences (105–107, 110, 122–124) → session 3. Line 115 holds the unsupported "strong gains" sentence → session 4 |
| 126– | 1.4 Study area | Fine |

**To write:**
1. **The research question in plain words, in the first paragraph.** The fact behind it: when a modeller changes how
   background points are drawn, the cross-validated AUC can rise because the *test* became easier, not because the
   model became better. The paper asks how to tell the two apart, and answers it with its own East Java model.
2. **A short "Key terms" box or table** (JCAA has no rule on boxes; `VENUE.md` A13). The definitions are ready in
   writing pack §3, for these terms:
   - presence–background model; background design (random / TGB / hybrid);
   - evaluation background (own / common); availability domain;
   - AUC, TSS and the Boyce index;
   - spatial block cross-validation; seed ensemble; Jaccard overlap; coarsened exact matching.

   Placement: at the end of 1.1, or as Table 1 (then the later tables renumber automatically).
3. **"No interior optimum" explained once, at first use** (abstract line 55 and introduction line 108). The fact: the
   criterion keeps rising as the hard-negative share grows, so its maximum always sits at the end of the range tried,
   and it never says where to stop.

---

## Session 3 — Text that talks about the review or earlier versions (R1-1), in four chunks

**Rule for every row:** keep the fact; drop the history of the review.
- The development of the study ("we built this sequence of designs and read it, at the time, as a series of
  improvements") is history of the research, which R1 does not object to.
- No P2 preprint was ever posted, so the earlier claim exists only in the review record and the public code
  repository.

### 3A — Introduction (6 places)

| v0.3 line | Opens with | Fact to keep |
|---|---|---|
| 89 | "Following a reviewer's taxonomy, a site may be absent…" | Four reasons a site can be absent; present the list as the paper's own framework |
| 101 | "We state this restraint up front because the earlier version…" | A suitability score cannot tell "never settled" from "buried"; keep the restraint, drop the reason |
| 105 | "A reviewer of the previous version asked us to benchmark against Maximum Entropy…" | Every design is benchmarked against MaxEnt, the standard presence–background method, under the same folds |
| 106–107 | "The benchmark refuted our own central claim… We have therefore not revised the paper's argument; we have replaced it." | The benchmark shows that the AUC ladder is an evaluation artefact; say it as a finding |
| 110 | "…because a reviewer of the previous version observed — correctly — that…" | This family of problems is well established in ecology (Lobo et al. 2008; Barve et al. 2011; Warren & Seifert 2011) |
| 122–124 | "The honest shape of this revision is worth stating…" | Delete the paragraph |

### 3B — Methods (5 places)

| v0.3 line | Opens with | Fact to keep |
|---|---|---|
| 182 | "…and a Maximum Entropy model … added at a reviewer's request." | Drop the clause |
| 231–232 | "The submitted manuscript acknowledged this coupling once, in the limitations. Given the benchmark…" | `road_dist` has four roles (line 228); state them as a property of the design |
| 233 | "This is also the honest answer to the question of what makes the approach specifically archaeological…" | The archaeological content lies in the background design, i.e. where a survey could plausibly have found a site. Say it without "answer to the question" |
| 239–240 | "We previously called this a 'temporal split'…"; "(The stored result file had itself been mislabelled… see the Further Disclosures…)" | E014 is an accessibility-proxy holdout: train ≤ 1 km from a road, test > 1 km; AUC 0.755. Name it once, correctly. The mislabelled-file detail goes to the supplement or is cut |
| 254 | Paragraph title "E217 – common-background benchmark (answers the Maximum Entropy request)" | Drop the parenthesis |
| 221–222 *(added 22 Oct, C080; ✅ done 22 Oct, passage 7)* | "The hybrid design of E013 blends … ``hard negatives'' (cells environmentally dissimilar from presences, 2.0 ≤ z ≤ 5.0)…"; "Note that the realised proportion of environmentally dissimilar pseudo-absences … 0.62 … the TGB pool, constrained to road-accessible cells, is intrinsically more dissimilar…" | **A factual error, not history.** (1) "Dissimilar" is too strong: 40.7 % of the sites themselves lie in the 2–5 band; say what the rule is (standardised distance 2–5 from the mean site environment; Table 1). (2) The TGB pool is **not** more dissimilar than random sampling (z ≥ 2: TGB 0.485, random 0.503). The 0.62 comes from the rule: 30 % forced from the band + 70 % from a z ≤ 5 pool in which ~46 % already have z ≥ 2 → ≈ 0.62. Keep the conclusion that follows (absolute AUC levels depend on pool composition, hence a common evaluation background). Source: `experiments/E217_maxent_benchmark/results/e217_zdist_sites_check.txt` |

### 3C — Results (6 places)

| v0.3 line | Opens with | Fact to keep |
|---|---|---|
| 328 (Table caption) | "…Gaps are computed against the E013 seed-average (0.751), not the best run… the submitted version…" | State the rule (seed average, because a single run is not reportable); drop "the submitted version…" |
| 345 (also 310, Figure 2 caption, and 351: "the published pipeline") | "We reject the published ladder gain (+0.092)…" | **"published" is inaccurate**: P2 was never published. "reported" (or "original") is accurate |
| 373 | "We note what the criterion does *not* do, correcting an earlier internal claim:…" | The selected design hybrid(1.0) is never the truth-worst (0/60); drop "correcting an earlier internal claim" |
| 441 | "Two corrections to our own diagnostics, none prompted by the reviewers." (TODO-PI comment) | Two corrections to the diagnostics; drop the reviewer clause. The subsection title can become e.g. "Volcanic diagnostics" |
| 448 | "The submitted manuscript reported ρ = −0.163…" | A single model instance gives −0.163; a five-seed re-run on the same 7-centre inventory gives −0.243. That is seed instability inside the diagnostic. Whether to keep the −0.163 at all is the PI's call |
| 488–492 | Subsection title "The taphonomic claim is withdrawn, and a quota design that failed"; "…in the reviewer's own terms…" | A suitability surface cannot separate the mechanisms of absence. Keep the quota result with its numbers (line 493). Title e.g. "What a suitability surface cannot show" |

### 3D — Discussion and back matter (5 places)

| v0.3 line | Opens with | Fact to keep |
|---|---|---|
| 516 | "With the site-prediction claim withdrawn, the output is a suitability surface…" | The output is a suitability surface with declared limits, not a map of where sites are |
| 523 | "…only under two conditions this revision makes explicit." | Say "this study" |
| 529 | "…which is the constructive contribution of the revision:" | Say it of the study |
| 549 (Limitations) | "…which requires the site-prediction claim we have withdrawn…" | Site-level prediction is outside what this design supports |
| 596 (Authors' contributions) | "Both authors approved the withdrawal of the central claim reported in this revision." | "Both authors read and approved the final manuscript." |

### Idioms (R1-4): a term table instead of 40 separate edits

R1 named "dial", "judging at home" and "Jaccard said no". They occur at these v0.3 lines (repeats count separately):
- **dial** (12): 119, 262 ×2, 366 ×2, 367, 372, 378 ×2 (caption), 379, 385, 535.
- **ladder** (21): 52, 56, 106, 245, 252, 271, 280, 281, 303, 310, 332, 341, 343, 345, 358 (caption), 365, 451, 491, 557,
  559, 560. **rung** (2): 341, 343.
- **"at home" / "home-court"** (6): 259, 338, 340, 342, 343, 358 (caption). Line 493 "home regime" is a different sense
  (the regime the quota design was built for). Rephrase it too, to avoid echoing the idiom.
- **"map Jaccard said no"**: line 401.

**Proposal (needs the PI's yes):** the PI approves one substitution per idiom. Claude then applies the substitutions
mechanically, as language help, and lists every changed sentence in `PI_PROSE_LOG_R2.md`. A sentence that needs more
than a word swap goes back to the PI. Candidates (writing pack §2):

| Idiom | Candidate |
|---|---|
| dial | "the share of hard negatives in the background (0 to 1)"; after first use, "the hard-negative share" |
| ladder / rung | keep "ladder", defined plainly once (as done in passage 8A: seven experiments, E007–E013, reported AUC 0.659 → 0.751 with one dip at E009 — *not* "each better than the last", which was wrong); "rung" → "step" |
| judged at home / home-court inflation | "scored on its own background" / "own-background inflation" |
| map Jaccard said no | "the pre-registered primary measure, overlap between top-decile maps (Jaccard), did not improve" |

---

## Session 4 — Abstract, keywords, conclusion and back matter

| Item | Where (v0.3) | Facts and limits |
|---|---|---|
| **Abstract** (A11 (b): the PI writes a new one) | 50–57 | ≤ 300 words; heading "Abstract". Facts below. **Paste the same text into the portal's metadata at upload** (the promise of 2 Oct to the editor) |
| **Keywords** | 60 | **There are 7; JCAA allows up to 6** (`VENUE.md` A3). One must go. Current: presence-background modelling; pseudo-absence design; evaluation artefact; target-group background; spatial cross-validation; survey bias; East Java |
| **Line-157 sentence** (A2) | 115, the part after "— and recent machine-learning work in archaeology reports strong gains…" | None of the three papers supports "strong gains". Facts and two shapes in `REFERENCE_FIX_20261020.md` §2 (Castiello & Tonini: AUC 0.71–0.72, "fair"; Wang et al.: about +0.03 over MaxEnt; Comer et al.: validation design changes the measured AUC) |
| **Conclusion** | 556–564 | Rewrite around the protocol. Drop "the more useful finding the reviewer's request produced" (563) and "internal revision drafts" (564). "We applied it to our own earlier analyses" is enough history |
| **Acknowledgements** (A5) | 585 | One neutral sentence, e.g. thanks to the editors and the reviewers of both rounds, whose comments led to a re-analysis that changed the paper's conclusion |
| **Supplement** | `supplementary_tables_v0.3.tex` (generated) and `supplement/e235_tables_v03.tex` | Write the captions of Tables S7 and S8; the stubs are marked "[S7 – caption for the PI]" and hold the facts. Edit the wording in `build_e235_tables.py` and re-run, never the .tex. Also look at the four-sentence intro of the supplement (TODO-PI in `build_supplement.py`, `main_v03`) |
| **AI statement** (A6) | 608–615 | Must describe *this* version truthfully. It now says "drafting and editing of this manuscript" and "corrections listed in the Response to Reviewers" |

**Facts for the abstract** (all in the current manuscript; numbers re-derived 2026-08-03 under G1):
- 378 presences; XGBoost, RandomForest and MaxEnt under fixed spatial-block cross-validation.
- Seven experiments E007–E013, reported AUC 0.659 (E007, one run) → 0.751 (E013, seed mean; 0.768 best run), with one
  dip at E009 (0.695 → 0.664); read at the time as the effect of more realistic background points. *(Corrected
  2026-10-22, log passage 8 A1: the earlier "seven designs, each better than the last" was wrong.)*
- On a common evaluation background: redesigning the background −0.014 AUC; adding one hydrological feature +0.042.
- Synthetic worlds: a design scored on its own background is reported above its truth in 343/360 runs (95.3 %). The
  selection criterion has no interior optimum.
- Seed ensembles: ≥ 7 seeds stabilise the top-decile map.
- E235: above about 1,000 m, elevation and nearness to Quaternary volcanoes cannot be separated in East Java.
- The protocol: hold the evaluation background fixed; declare the availability domain; seed-ensemble the maps.

**Facts for the AI statement** (what AI did in this revision; the PI chooses the wording):
- analysis code and runs (E235), figures, reference checks against Crossref and the papers;
- verification: blind re-derivation of numbers and an adversarial read;
- **translation of the PI's Indonesian prose into English, and language checking** (`PI_PROSE_LOG_R2.md`).

The research questions, the decisions and the prose content are the authors'. The JCAA example wording is in `VENUE.md`
A5; the project template is `docs/AI_DISCLOSURE_TEMPLATE.md`.

---

## After session 4 (Claude)

1. Re-run `build_jcaa_bibliography.py`.
2. G1 on every number in the final text.
3. An independent adversarial read.
4. Trace the response letter (`RESPONSE_R2_SKELETON.md`) against the final file.
5. Update the alt-text list: Figure 6 = seed stabilisation; Figure 7 = E235.
6. Template check (`VENUE.md`).
7. Freeze ≤ 9 Nov → upload ≤ 16 Nov (PI login), with one sentence on the waiver in the cover letter.

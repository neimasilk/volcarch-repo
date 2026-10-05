# P8 — revision work plan and point-by-point matrix (*Oceanic Linguistics*, OL-03-2026-11)

**Written:** 2026-10-05 · **Line:** 04 · **Companion to** `REVISION_OL_20261005.md` (the ledger of reviewer points, paraphrased).
**Evidence:** `experiments/E227_p8_g1_blind_rederivation/` (audit of the submitted numbers, 85 statements) and
`experiments/E228_p8_revision_analyses/` (pre-registered analyses for the reviewers' questions).
**Division of labour (gate G16):** this file gives facts, numbers, locations and what a reply has to contain.
It contains **no manuscript text and no letter text** — those are the PI's.
Public repo: reviewer comments are paraphrased; the portal link is not recorded.
Line numbers = `draft_v0.1_anonymous.tex` (the text the reviewers read).

---

## 1. Where things stand

1. **Every reviewer point now has a verified factual answer** (§3, §4). Two needed real analysis and gave clean
   results: the published Makasar figure is reproduced from ABVD (60.7 % vs 62 %) and decomposed; the reviewer's
   low-level-reconstruction scenario is confirmed for Tolaki (71 % of its uncoded forms have a look-alike elsewhere in
   Bungku–Tolaki; chance 10 %).
2. **The stored numbers reproduce exactly** from raw ABVD (G7 passes).
3. **Several statements of the submitted text do not describe what was computed**, and three of them sit in the
   abstract (§5). They surfaced because four reviewer questions lead straight to them (R1-12, R1-13, R2 on the
   "E022 label", R2 on the five concepts), and because the data release R1 asks for would expose them.
4. Re-computed without those defects, the paper's quantitative claims are **weaker but coherent** (§6), and the
   result that is new and of interest to readers of the journal — Makasar and Tolaki — is **stronger**.
5. The submitted files, the preprint and the journal record are untouched. A working copy exists —
   `revision_v0.2/p8_revision_v0.2.tex` — with **only** the fifteen term substitutions that reviewer 1 dictated
   (points 1, 4, 5, 6; list in `revision_v0.2/CHANGES.md`); it compiles; it is **not** a corrected version.
   **Seven decisions are the PI's** (§2).

## 2. Decisions for the PI (in order; each has a recommendation)

| # | Decision | Recommendation | Why | If declined |
|---|---|---|---|---|
| **D1** | One definition of the label everywhere | *candidate* = no cognate-set assignment in ABVD (the label every published model already used); drop the 15-meaning step and the string-matched loan lists | E227 A10/A13/A15: the 15-meaning step compared no form with any reconstruction; the loan lists matched five unrelated words | Table 1 stays at 356 forms while every other table uses 438; R1-12 and R1-13 cannot be answered consistently |
| **D2** | Which model carries the headline | the 25-input model (no language-level input): CV 0.727, held-out language 0.701 — and report the form-only model beside it (0.672 / 0.641) | E228 S7: the submitted 0.763 model's strongest input is the language's identity code; identity alone gives 0.680 | "trained exclusively on phonological features" stays in the abstract next to a number it does not describe (R1-10) |
| **D3** | The "two-method consensus" | drop the framing; report label × out-of-fold profile (κ 0.31; 172 forms) | E227 F03: the 0.61 is the model's fit to its own training label | the 266-form set and κ 0.61 stay; a reader with the released lists can show in minutes that they are in-sample |
| **D4** | The central negative result | report the permutation test and say "no large shared layer across the four subgroups", not "no shared forms"; replace "each language innovated independently" by what S3/S5 show | E228 S5: p = 0.0001 on the pre-registered primary set (4 look-alike pairs vs 0.5 expected; coded forms 472 vs 8.5) | p = 0.569 stays although the test behind it compared a mean of 20 with single draws of mostly cognate vocabulary (E227 G07) |
| **D5** | The 16 additional languages | remove the geographic sentence from the abstract; delete the Acehnese and Bolaang Mongondow interpretations; keep at most a short honest paragraph (mean AUC 0.64; ≥ 0.60 in 10 of 16) | E227 H06, E228 S6: the published per-language rates were produced by two language-level inputs; without them Acehnese drops from 63 % to 8 % | Table 5 / Figure 4 stay with numbers that cannot be reproduced without those inputs |
| **D6** | §4.5 (Javanese script) | remove it, or reduce it to one cautious sentence | not asked for by any reviewer; outside Sulawesi; leans on properties that changed (D2, prefix direction); quotes 10.5 % for the wrong group (F11); F9 (correlated channels). ⚠ The statement about a Javanese phonemic glottal stop absent from the script needs a specialist — I am not confident it is right | stays; the editor is a specialist in exactly this area |
| **D7** | Tell the editor before rewriting | yes — two or three plain sentences: in preparing the data release asked for by reviewer 1 the authors re-derived every number, found errors of their own, and will correct them in the revision; some abstract figures will change | the decision letter asks for revisions "along these lines"; a revision that changes abstract numbers should not arrive unannounced | the editor learns of it from the response letter; risk of a second review round either way |

Also the PI's: whether to adopt the spelling *Makasar* (reviewer's preference, Glottolog's name; ABVD writes
*Makassar*); whether the title and the word "fingerprint" survive D2 (R1-10 asks to separate phonological from
semantic); when to publish the release files; whether G13 counts a manuscript in revision; co-author sign-off
(Go Frendi Gunawan) on every change of substance; an arXiv v2 **after** the journal accepts the revision.

**Status 2026-10-05 ±16:00.** The PI asked for a considered recommendation and for it to be carried out; no deadline, no hurry.
The recommendations of the table stand. Reading recorded in `docs/HANDOFF_20261005.md` §2: D1–D6 proceed as recommended
(D6 with a confirmation wanted); **sending** the note of D7 and **the prose** (G16) stay with the PI until he says otherwise.

## 3. Reviewer 1 — fourteen points

Types as in the ledger: **T** wording/classification · **C** clarification · **A** analysis · **D** data.
"Reply must contain" lists facts, not sentences.

| # | Point (paraphrase) | What was found | Change in the manuscript | Reply must contain | Evidence |
|---|---|---|---|---|---|
| 1 | "any proto-form" is too strong; low-level reconstruction is possible | Confirmed with data: for Tolaki 70.9 % of uncoded forms have a same-meaning look-alike in Proto-Bungku-Tolaki or in one of 42 other Bungku–Tolaki lists in ABVD (chance 10.5 %; strict threshold 63.4 % vs 3.0 %) | tex 38, 56 as the reviewer words it; tex 60 and 540 use the phrase in another sense — PI decides. Add the Tolaki result where residuals are introduced | agreement; the number; that it is a mechanical screen, not etymologies | E228 S3 |
| 2 | Makasar: published 38 % retention / 62 % "open" against our figure; anything to add? | (a) The reviewer writes 30.1 %; Table 1 prints **30.9 %** — a slip in the report, mention gently. (b) From ABVD, per concept (201 comparable meanings): retained from PMP **39.3 %**, coded but not the PMP etymon **25.9 %**, no cognate set **34.8 %**. Not retained = **60.7 %** ≈ the published 62 %. (c) Bugis 48.3 / 28.4 / 23.4, Tae' 48.0 / 33.0 / 19.0: Makasar retains about 9 points less and has 11–16 points more uncoded meanings than its two relatives. (d) With doubtful assignments counted, retention is 42.8 % | new short subsection or paragraph + one small table; cite the Makasar literature **after reading it** | the three-way split; that the two published numbers and ours are consistent; that "uncoded" is an upper bound on anything non-Austronesian (see #1, #13); what the study cannot say (origin of the uncoded part) | E228 S2 |
| 3 | "at least ten primary subgroups (Blust 2009)" — the source lists eleven; Sneddon 1993 | Confirmed: the 2013 edition names **eleven "microgroups"** (§2.4.6, p. 82) and reports Sneddon's ten (§4.1.6, p. 193). The book does not call them primary subgroups | tex 57: number, term and edition | corrected count and source, as read by the PI | §8c |
| 4 | Celebic is a supergroup, not a primary subgroup | accepted | tex 65: South Sulawesi, Bungku–Tolaki, Muna–Buton | — | — |
| 5 | "parallel innovation" is a technical term | accepted; but see D4: "independent innovation by each language" is itself no longer what the data show | tex 73, 509 (heading), 517, 598, 614 | that the term is dropped **and** that the claim was re-examined | E228 S3, S5 |
| 6 | "three subgroups" but four listed; Sa'dan Toraja is South Sulawesi; Wolio is Wotu–Wolio; spelling; no abbreviation for Bolaang Mongondow | accepted. The abbreviation is at tex 468 (Table 5), 486 and inside Figure 4; "Toraja-Sadan" also inside Figures 2 and 4. ABVD's own name for list 226 is *Tae' (S. Toraja)*, source van der Veen 1940 — say which list was used | tex 86–87, 237, 298, 405, 468, 486; figures regenerated | four subgroups as the reviewer gives them; the consequence: tex 534 ("the South Sulawesi mainstream represented by Bugis and Makassar") must follow the corrected grouping | E227 A02 |
| 7 | "under-documentation" is the wrong word | The fact meant: 36 % of Tolaki forms have a cognate-set assignment (75 of 209). The cause is not sparse primary data (ABVD holds five further Tolaki dialect lists; of the 86 uncoded forms whose meaning those lists cover, 88 % recur there) | tex 114, 528–534: replace by the number and the S3 result | the distinction: the *list* is well documented, the *cognate coding* is sparse | E228 S3 |
| 8 | Muna *dh* missing from the digraph list | Two Muna forms contain *dh*: *akaradhaa* 'to work', *idho* 'green'; both are cognate-coded; both **were** converted in the test — the sentence omitted *dh* (Muna) and *gh* (Wolio). The script used a fricative symbol as a one-character placeholder; the reviewer's description (dental stop) is the phonetic fact | tex 348: complete the list; do not call *dh* a fricative | the two forms; that the count (75 forms changed, 54 in Muna) is unaffected | E227 I01–I03 |
| 9 | Glottal stop is written many ways; same result under each convention? | **No, not quite.** Marker rate by source: Makasar 36 %, Tae' 22 %, Bugis 21 % (ʔ); Tolaki 14 % (apostrophe); Wolio 2 %; Muna 0 %. Re-coding all markers as *q* or *k* costs 0.016 of AUC (25 inputs; 0.020–0.024 across held-out languages), leaving them unwritten 0.006, as geminates 0.007; ≥ 0.710 under every convention. Pre-registered verdict: *partial dependence*. Inside the sources that mark it, marked forms are more often uncoded (OR 2.67 [1.88, 3.79]). ʔ + consonant also counts as a "cluster" in 35 forms | tex 42, 135, 355, 382, 577: the sentence "robust … rather than orthographic artifacts" cannot stand for this property; describe glottal marking as a property of four sources | the table of conventions; the plain statement that the property is visible only where the source writes it | E228 S1; E227 D02, D06 |
| 10 | A semantic feature cannot belong to a *phonological* fingerprint | Sharper than the reviewer assumed: of the 26 inputs of the headline model 17 come from the written form, 8 from the meaning, 1 is the language's identity. AUC by group: form only **0.672** (0.641 held-out language), meaning only 0.646, identity only 0.680, form + meaning 0.727 | abstract (tex 39–41), 71, 127, 132–141, 385, 497, 594–595: say what the inputs are; D2 | the breakdown table; which model the quoted AUC belongs to | E228 S7; E227 B03–B04 |
| 11 | Example lexemes for each quadrant | Lists ready (five per cell, three per language and cell). Examples that the PI should check against ABVD and the ACD before use: *candidate with profile* — Tolaki *umi'ia* 'to cry', Makasar *ammikkiriʔ* 'to think' (ABVD marks it a loan), Tae' *limaŋpulo* 'fifty'; *candidate without profile* — Tae' *annan* 'six', Tolaki *omba* 'four', Makasar *jukuʔ* 'fish', Muna *riwu* 'thousand'; *coded with profile* — Bugis *mar-eŋkaliŋa* 'to hear', Tolaki *lumangu* 'to swim'; *coded without profile* — *ama* 'father', *ana* 'child' | table in §3.5 (cells from D3) | the table; the honest observation it supports: the profile tracks affixed, compounded and glottal-marked citation forms | E228 S4 (`S4_cell_examples.csv`) |
| 12 | "false positive" / "unlabeled positive" used in two senses | Root cause: two label sets (356 and 438) and a "rescue" step that was a concept filter (§5 items 1–2). With D1 the term is not needed: *coded* / *uncoded (candidate)*; "false alarm" only for the decimal compounds | tex 94, 107, 183, 247, 412, 440, 445, 520, 546, 581 | one definition per term; that the confusion came from the authors' own inconsistency, now removed | E227 A10–A14 |
| 13a | What does "documentation gaps" mean for Tolaki? | The second reading (cognates exist, not yet assigned): S3. Of the 34 uncoded forms that match a Proto-Bungku-Tolaki entry, 7 match an entry that ABVD does assign to a cognate set | tex 534 | both numbers | E228 S3 |
| 13b | Release the residual lists | Files built: `experiments/E228_p8_revision_analyses/release/p8_candidates.csv` (438 rows; **Tolaki 134**, not 114 — say why) and `p8_forms_all.csv` (1,357 rows) with ABVD ids, flags, out-of-fold probability, nearest look-alike. ABVD is CC BY 4.0. Deposit: Zenodo (no cost; D1/D2 precedent) → DOI in the data-availability statement | tex 619 | DOI; column description | `release/` |
| — | Opening remark: the SHAP figure is unreadable for a linguist | Figure 1 carries a software title, raw variable names and (in the caption) the wrong sign | regenerate as a plain bar chart for the D2 model | — | E227 E04 |

## 4. Reviewer 2 — twenty-four highlights, by group

| Group | What was found | Change | Evidence |
|---|---|---|---|
| Internal codes (E022, E027, E028, E029, E041) | 29 occurrences on 23 lines (tex 97, 117, 174, 178, 181–184, 187, 191, 229, 370, 392, 396–397, 404, 411–412, 414, 435, 477, 512, 613) and inside the titles of Figures 2 and 4. "E022 binary label" cannot be found in the rule-based section because the label is **not** what that section describes (§5 item 1) | names instead of codes; D1 | E227 A10 |
| AUC — meaning and scale | Plain meaning available: the chance that a randomly chosen uncoded form is ranked above a randomly chosen coded one; 0.5 = coin, 1 = perfect. There is no agreed verbal scale; the "0.65 threshold" at tex 313 is the project's own go/no-go line (E227 C18) — say so or drop it. ± 0.007 is the SD of ten seed means; across the 50 folds it is 0.028 | definition at first use; one honest sentence on size | E227 C01, C18 |
| "near-perfect", Table 2 walk-through, "both models" | Model A: 31 inputs, AUC 0.9997–1.000 because four of them encode the label. F1 in Table 2 is the F1 of the **coded** class; for the candidate class it is 0.53 (27-input model). Accuracy 0.741 vs 0.677 for always answering "coded". XGBoost was **not** class-weighted (tex 153 says it was) | Table 2 rebuilt for the D2 model with the candidate-class score and the baseline; tex 153 corrected | E227 C05–C09 |
| SHAP, feature ablation, typewriter font, Δ, CV, κ | SHAP top five reproduce, but are those of the 27-input model; for the 25-input model see `E227/results/shap_pure25.csv`. `language_cognacy_coverage` does not hold the coverage (hard-coded stale values; E227 B02) — with D2 it disappears | glossary box or first-use definitions (PI); Figure 1 regenerated | E227 B02, E01–E05 |
| "classifier … robustness" (three classifiers) | RF 0.762, LR 0.747, XGBoost 0.760 (27 inputs) reproduce | keep, for the D2 model | E227 C08 |
| "language identity" feature | an arbitrary integer 0–5 in alphabetical order of the list names; the strongest input of the submitted headline model | D2 removes it; if kept, describe it as what it is | E228 S7 |
| Source of the semantic domains (Concepticon? WOLD?) | Neither. The seven domains and the "core" flag are hand-made concept sets in the feature script. The "Swadesh-100" flag covers **174 of the 210** meanings (83 % of forms; the set lists 176 names, two match no ABVD concept). ABVD's 210-item list is itself not a Swadesh list in the strict sense ⚠ (PI: check how ABVD describes it) | tex 89–90, 139: describe as the authors' grouping; correct or drop the "Swadesh-100" flag; mapping to Concepticon is possible future work, do not claim it | E227 E06 |
| "scikit-learn / XGBoost — Python?" | Python 3.11; xgboost 3.0.3, scikit-learn 1.8.0, shap 0.50.0 reproduce the stored results | one clause | E227 summary |
| Why k = 5 … 30; DBSCAN parameters | No reason: 30 is the edge of the range and the silhouette keeps rising beyond it (0.126 at 40, 0.205 at 100); the value also shifts with row order (0.102–0.114). Ward linkage presupposes Euclidean distances | with D4 the clustering paragraph can shrink to two sentences or go; the permutation test replaces it | E227 G01–G03 |
| What "false positive" stands for | as R1-12 | D1 | — |
| Figure 2 panels not referred to; Table 5 cut off | Figure 2 has four panels (A–D) with internal codes in titles and axes; panel D itself shows prefix-like onsets **higher** in the "consensus" group, against the text. Table 5 has seven columns under `\small` and runs off the page | regenerate (D3) or drop Figure 2; Table 5 rebuilt or dropped (D5) | figures; E227 D01 |
| Why 'One Hundred', 'Fifty', 'Twenty', 'to stand', 'to hit' in ≥ 4 languages | The first three are decimal compounds of inherited numerals, uncoded in four lists and coded in Bugis. The other two were on the submitted version's own list of inherited meanings; the "consensus" was computed on the label that ignored that list. Out of fold, **no** meaning is "candidate with profile" in four languages | D1 + D3 remove the paragraph's premise; keep the numeral observation | E228 S4; E227 F12–F13 |
| "generalises across Sulawesi languages" concretely | = a model fitted on five lists ranks the forms of the sixth: AUC 0.61–0.81 (25 inputs: mean 0.701, five of six ≥ 0.65). Accuracy for Tolaki and Muna in Table 3 (0.36, 0.40) is far below the majority baseline — a threshold effect that the text passes over | say it in those words; comment on the two accuracies or drop the column | E227 C10–C12 |
| Define "fingerprint" early | see R1-10; the profile that the data support (uncoded vs coded forms, all six lists): longer (6.10 vs 5.20 characters; 2.57 vs 2.29 vowel groups), more consonant-letter clusters (0.48 vs 0.32), glottal marking where written (23.5 % vs 11.6 %), action meanings (40.4 % vs 23.4 %), and **more** prefix-like onsets (37.0 % vs 25.2 %) | Introduction | E227 D01–D05 |
| Citation for "resists reconstruction" (p. 2) | open — a source has to be found and read by the PI; none is proposed here | tex 56 | — |

## 5. Found while verifying — not raised by the reviewers (E227 row ids)

Each is a statement of the submitted text that the data or the code contradict. None can be mended by wording.

1. **Two residual sets presented as one** (A07, A10, A14, F06, H02): 356 forms in Table 1; 438 in every model, in
   Table 3, in the agreement table and in the clustering. Abstract: "438 … (26.5 %)" — 438 is 32.3 %.
2. **The Proto-Austronesian cross-check compared no forms** (A13): all uncoded forms for 15 meanings were removed.
   `E227/results/pan_rescued_75.csv` — e.g. 'rope': *tulu, otɛrɛʔ, kakoo, oloo, puloli* against *\*talih*.
3. **The loanword lists matched five unrelated words** (A15), among them Tolaki *beli* 'blood' (Malay *beli* 'buy'),
   which has an exact counterpart in Kodeoha.
4. **"In five or more of six languages" counted forms** (A16): three of the eight meanings qualify.
5. **"Fewer canonical Austronesian prefixes" is the reverse of the data** (D01), in the abstract and at tex 385, 448,
   497. The argument at tex 447–448 that rests on it fails; the data lean the other way, toward the
   morphological-complexity reading that the paragraph was meant to answer.
6. **Figure 1 caption states the sign the wrong way round** (E04); figure from the 27-input model (E03).
7. **In-sample agreement presented as independent confirmation** (F03) — abstract, tex 404, 546, 597.
8. **p = 0.569 does not test the stated hypothesis** (G07); "optimal k = 30" is the search limit (G02).
9. **Language-level inputs in the 16-language application** (H06); "significantly" without a test (H05); the 0.890
   for the six original lists is in-sample (H03); four of the six "original" rates in Table 5 are stale constants (H02).
10. Smaller: class weighting (C07), F1 class (C05), "Swadesh-100" (E06), coverage variable (B02), "syllable" = vowel
    group (D04), wrong group for 10.5 % in §4.5 (F11), SD of seeds vs folds (C01).
11. **`revision_ammo/anticipated_critiques.md` must not be used**: it describes a different paper (other title, a
    six-language set of Bare'e, Muna, Tolaki, Toba Batak, Ngaju, Manggarai, experiment folders that do not exist).
    A warning has been put at its top.

## 6. The numbers after correction (for the PI's tables; all from E227/E228)

| Quantity | Submitted | Corrected | Note |
|---|---|---|---|
| Candidate forms | 438 "(26.5 %)" | 438 = **32.3 %** of 1,357 | D1 |
| Table 1, per list (Muna, Bugis, Tae', Wolio, Makasar, Tolaki) | 26, 49, 32, 68, 67, 114 | **34, 62, 45, 83, 80, 134** = 15.5, 25.6, 20.8, 32.7, 36.9, 64.1 % | D1 |
| Share of forms with a cognate set | 84, 74, 79, 67, 63, 36 % | same (84.5, 74.4, 79.2, 67.3, 63.1, 35.9) | unchanged |
| Headline discrimination | 0.763 ± 0.007, "phonological only" | form only **0.672** (held-out language 0.641); form + meaning **0.727** (0.701) | D2 |
| Lists ≥ 0.65 when held out | 6 of 6 | 5 of 6 (form + meaning); 3 of 6 (form only) | D2 |
| Without Tolaki | 0.698 (Δ −0.062) | 25 inputs: 0.701 (Δ −0.025) | E227 C17 |
| Agreement label × classifier | κ 0.611; 266 / 878 / 172 / 41 | κ **0.31**; 172 / 814 / 266 / 105 (out of fold) | D3 |
| Meanings "consensus" in ≥ 4 languages | five | none | D3 |
| Cross-language test | p = 0.569 | permutation p = 0.0001; look-alike pairs 4 vs 0.5 expected (coded forms 472 vs 8.5) | D4 |
| Prefix-like onset | "fewer" | **more**: 37.0 % vs 25.2 % (OR 1.57 [1.20, 2.04]) | fact |
| Glottal convention | "robust" | −0.006 to −0.016 AUC depending on convention | R1-9 |
| 16 lists: mean AUC / mean probability Sulawesi vs west | 0.663 / 0.606 vs 0.393 "significantly" | 0.638 (≥ 0.60 in 10 of 16) / 0.308 vs 0.229 (p = 0.023; among coded forms p = 0.37) | D5 |
| Makasar | 30.9 % residual | 39.3 % retained from PMP · 25.9 % coded otherwise · 34.8 % uncoded (of 201 meanings) | new |
| Tolaki | "artifact of under-documentation" | 70.9 % of uncoded forms shared within Bungku–Tolaki (chance 10.5 %) | new |

## 7. Order of work and gates

| Step | Who | What | Gate |
|---|---|---|---|
| 1 | PI | D1–D7; read §3–§5; open `release/p8_candidates.csv` | — |
| 2 | PI | read the references a reviewer supplied (§8) — none is cited before that | citation integrity |
| 3 | PI (+ co-author informed) | short note to the editor (D7), with the one-line question about author-side charges (§8e) | G15 |
| 4 | Claude | for the model chosen in D2: Tables 1–3 as data files, Figure 1 as a plain chart, the examples table (R1-11), the Makasar table (R1-2); `VENUE.md` check of two or three recent OL articles for table and example conventions | G15 |
| 5 | PI | writes abstract, §2–§4, conclusion; first-use definitions for the ML terms (R2); Section 3 shortened — with D3–D5 it loses a table, a figure and two subsections, which is the readability the editors ask for | G16 |
| 6 | Claude | language check; every number in the new text against E227/E228 (G1 on the new text); overstatement and causal-connector scans (G8, G11); check of the PDF that will be uploaded (G1-bis item 3) | G1, G8, G11 |
| 7 | — | freeze ≥ 14 days; PI re-reads in full | G14 |
| 8 | PI | response letter (R1 14 points, R2 24 highlights, plus a section "corrections by the authors"); upload; download the file back and compare | G16, G12 |
| 9 | PI | Zenodo deposit of the release files (DOI into the data-availability statement before step 8); arXiv v2 only after the journal's decision | zero cost |

No deadline was stated. With steps 1–3 in the week of 6 Oct and writing by about 26 Oct, the freeze ends about
9 Nov and resubmission falls in mid-November. These are suggestions, not commitments.
Human reader (G10): the forms marked ⚠ in E228 and the look-alike lists need a historical linguist who knows
Sulawesi before any single form is discussed in print.
G15: the journal's fee policy is **not stated anywhere** — ask the editor (§8e).

## 8. References (full table with evidence URLs: `REFERENCE_CHECK_20261005.md`)

Checked on 2026-10-05 against fetched pages only (publisher and catalogue records, Crossref, the open-access PDFs);
existence and bibliographic data, **not** whether the work supports the sentence it is cited for.

**(a) The 35 entries of `references.bib`:** 25 correct · 8 need correction · **2 do not exist**.

| Key | Finding | Cited in the text? |
|---|---|---|
| `mead2005` | **No such work.** *Papuan Pasts* has 28 chapters, none by Mead; the pages given straddle two real chapters | no — delete |
| `vandenBerg1996` | **No such work** (the atlas pages given lie inside another author's chapter) | no — delete |
| `ross2005` | Real title, **false venue**: it is chapter 2 of *Papuan Pasts* (Pacific Linguistics, 2005), pp. 15–65 — not *Oceanic Linguistics* 44(2): 343–380 | **yes (tex 585)** — must be corrected; the journal named is the one the paper is going to |
| `bellwood1995` | Book chapter in *The Austronesians* (eds Bellwood, Fox, Tryon), entered as an article; the entry mixes the 1995 pages (96–111, not directly confirmed) with the 2006 ANU E Press imprint (there pp. 103–118) | yes (tex 66) — choose one edition |
| `casparis1975` | Brill, Leiden only; no support for the "LIPI / Jakarta" co-publisher | yes (tex 554; D6) |
| `swadesh1955`, `thurgood1999`, `mcelhanon1970`, `list2012`, `anderson2018` | data right, entry type wrong (article / book / proceedings) — affects how the reference prints | 4 of 5 cited |
| `list2018` | data right, **wrong use**: CLICS² is a database of colexifications, not a tool for phylogenetic analysis (tex 62) | yes — replace or reword |
| `levenshtein1966` | 1966 is the English translation; Russian original 1965 | yes — fine as is |
| `pawley2005`, `blust1993` | correct, but cited nowhere | no |

**(b) The references reviewer 1 supplied** — all exist. Three details differ from the short forms in the report:
van den Berg's "demise of focus" paper is **1996** (Pacific Linguistics A-84, pp. 89–114), not 1989; Bulbeck, Pasqua & Di Lello
is *Asian Perspectives* 39(1–2): 71–108, issue dated **2000** (copyright line 2001), p. 103 inside the range; Blust 2012 has a
longer title ("… A Reply to Reid", *OL* 51(2): 538–566). Mills 1975 is a two-volume Michigan dissertation; Sirk 1989 is in
NUSA; Bulbeck 1992 an ANU thesis; Sneddon 1993 *OL* 32(1); Mead 2003 pp. 115–141. Not seen: the pinpoint pages in Mills
and Bulbeck 1992, and Bellwood 1997: 115 itself. **None may be cited before the PI has read it.**
The sentence with 38 % / 62 % is in the Bulbeck et al. article as the reviewer reports it. For R1-2 note what Blust (2013,
§5) says a retention percentage is: reconstructed basic vocabulary compared with its reflexes in the modern language — the
quantity that E228 S2 computes from ABVD's PMP list.

**(c) R1-3, the number of subgroups.** Blust (2013 revised edition, §2.4.6, p. 82) recognises **eleven "microgroups"** in
Sulawesi and names them; §4.1.6 (p. 193) reports Sneddon's (1993) ten and Mead's addition of Wotu–Wolio. The reviewer is
right. Two cautions for tex 57: the book's word is *microgroups* — it treats two of the eleven as branches of the Philippine
subgroup, so "primary subgroups" is not its claim; and only the 2013 edition was read, while the manuscript cites the 2009 one.

**(d) Citations whose fit to the sentence the PI should check while reading** (not checked here): `bellwood1995` for the
sentence about volcanism and substrate retention (tex 66); `himmelmann2005` (tex 65); `donohue2010` and `adelaar2005`
(tex 57); `lefebvre2004` and `thomason1988` for verbs resisting replacement (tex 506); `blust2010` (tex 540);
`casparis1975` for "33 to 20 aksara" (tex 554); `thurgood1999` (tex 487; falls with D5).

**(e) Fees (G15).** No page of the journal or the press states a fee, and none states that there is none (journal page,
author guidelines, open-access page, instructions for contributors, portal — read 2026-10-05; the journal is not listed
as open access). That is silence, not a zero. **One line to the editor settles it** — it can go in the same note as D7:
whether any charge falls on authors (page, colour-figure or other), and that the authors do not take a paid open-access option.

## 9. Not done

The submitted manuscript files, the figures and `references.bib` were not edited (the `.bib` corrections of §8 are
listed, not applied). The only manuscript-side change is the working copy `revision_v0.2/` described in §1.5.
Nothing was sent, uploaded or published; nothing was committed.
The release files are in the repository working tree only (the repository is public once pushed — the PI decides
whether these lists go out before the Zenodo deposit).

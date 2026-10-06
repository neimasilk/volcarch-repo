# P8 — revision work plan and point-by-point matrix (*Oceanic Linguistics*, OL-03-2026-11)

**Written:** 2026-10-05 · **Updated:** 2026-10-06 (§10: check against the original reports, E229 tables, E230, venue facts) · **Line:** 04 · **Companion to** `REVISION_OL_20261005.md` (the ledger of reviewer points, paraphrased).
**Evidence:** `experiments/E227_p8_g1_blind_rederivation/` (audit of the submitted numbers, 85 statements) and
`experiments/E228_p8_revision_analyses/` (analyses for the reviewers' questions; decision rules written down the same
day, before the runs — not registered externally).
**Division of labour (gate G16):** this file gives facts, numbers, locations and what a reply has to contain.
It contains **no manuscript text and no letter text** — those are the PI's.
Public repo: reviewer comments are paraphrased; the portal link is not recorded.
Line numbers = `draft_v0.1_anonymous.tex` (the text the reviewers read).

---

## 1. Where things stand

1. **Every reviewer point now has a verified factual answer** (§3, §4). Two needed real analysis: the ABVD figure for
   Makasar is close to the published one (39.3 % against a quoted 38 %; ⚠ the source is unread and may be the same
   data — no "consistent with" before the PI has read it) and is decomposed; 71 % of
   Tolaki's uncoded forms have a look-alike elsewhere in Bungku–Tolaki (chance 10.5 %), which shows that the list is not
   short of data and says nothing about origin. *(Wording corrected 2026-10-06: the earlier "reproduced" and
   "scenario confirmed" claimed more than was shown — §10.7.)*
2. **The stored numbers reproduce exactly** from raw ABVD (G7 passes).
3. **Several statements of the submitted text do not describe what was computed**, and three of them sit in the
   abstract (§5). They surfaced because four reviewer questions lead straight to them (R1-12, R1-13, R2 on the
   "E022 label", R2 on the five concepts), and because the data release R1 asks for would expose them.
4. Re-computed without those defects, the paper's quantitative claims are **weaker but coherent** (§6), and the
   result that is new and of interest to readers of the journal — Makasar and Tolaki — is **better founded** (read
   with §10.7: both are descriptions of ABVD's coding, not statements about origin).
5. The submitted files, the preprint and the journal record are untouched. A working copy exists —
   `revision_v0.2/p8_revision_v0.2.tex` — with **only** the fifteen term substitutions that reviewer 1 dictated
   (points 1, 4, 5, 6; list in `revision_v0.2/CHANGES.md`); it compiles; it is **not** a corrected version.
   **Ten decisions are the PI's** (§2, §10.5; D4 reframed in §10.7).

## 2. Decisions for the PI (in order; each has a recommendation)

| # | Decision | Recommendation | Why | If declined |
|---|---|---|---|---|
| **D1** | One definition of the label everywhere | *candidate* = no cognate-set assignment in ABVD (the label every published model already used); drop the 15-meaning step and the string-matched loan lists | E227 A10/A13/A15: the 15-meaning step compared no form with any reconstruction; the loan lists matched five unrelated words | Table 1 stays at 356 forms while every other table uses 438; R1-12 and R1-13 cannot be answered consistently |
| **D2** | Which model carries the headline | the 25-input model (no language-level input): CV 0.727, held-out language 0.701 — and report the form-only model beside it (0.672 / 0.641) | E228 S7: the submitted 0.763 model's strongest input is the language's identity code; identity alone gives 0.680 | "trained exclusively on phonological features" stays in the abstract next to a number it does not describe (R1-10) |
| **D3** | The "two-method consensus" | drop the framing; report label × out-of-fold profile (κ 0.31; 172 forms) | E227 F03: the 0.61 is the model's fit to its own training label | the 266-form set and κ 0.61 stay; a reader with the released lists can show in minutes that they are in-sample |
| **D4** | The central negative result | report the permutation test as what it measures — among uncoded forms, same-meaning forms are slightly more alike than different-meaning ones — and say neither "no shared forms" nor "no large shared layer" (the test does not examine the coded class; **§10.7**, corrected 2026-10-06); replace "each language innovated independently" by what S3/S5 show | E228 S5: p ≤ 0.0001 (none of 10,000 permutations) on the primary set, whose rule was written before the run (4 look-alike pairs vs 0.5 expected; coded forms 472 vs 8.5) | p = 0.569 stays although the test behind it compared a mean of 20 with single draws of mostly cognate vocabulary (E227 G07) |
| **D5** | The 16 additional languages | remove the geographic sentence from the abstract; delete the Acehnese and Bolaang Mongondow interpretations; keep at most a short honest paragraph (mean AUC 0.64; ≥ 0.60 in 10 of 16) | E227 H06, E228 S6: the published per-language rates were produced by two language-level inputs; without them Acehnese drops from 63 % to 8 % | Table 5 / Figure 4 stay with numbers that cannot be reproduced without those inputs |
| **D6** | §4.5 (Javanese script) | remove it, or reduce it to one cautious sentence | not asked for by any reviewer; outside Sulawesi; leans on properties that changed (D2, prefix direction); quotes 10.5 % for the wrong group (F11); F9 (correlated channels). ⚠ The statement about a Javanese phonemic glottal stop absent from the script needs a specialist — I am not confident it is right | stays; the editor is a specialist in exactly this area |
| **D7** | Tell the editor before rewriting | yes — along the fact list in `docs/correspondence/EMAIL_OL_EDITOR_P8_NOTE_DRAFT_20261006.md` (rewritten 2026-10-06): before revising, the authors re-checked the manuscript's numbers from the raw files (an audit of 85 statements, then the robustness section), found errors of their own — some prompted by the reviewers' questions, some beyond them — and will correct them; the paper's claim and several abstract figures change | the decision letter asks for revisions "along these lines"; a revision that changes abstract numbers should not arrive unannounced | the editor learns of it from the response letter; risk of a second review round either way |

Also the PI's: whether to adopt the spelling *Makasar* (reviewer's preference, Glottolog's name; ABVD writes
*Makassar*); whether the title and the word "fingerprint" survive D2 (R1-10 asks to separate phonological from
semantic); when to publish the release files; whether G13 counts a manuscript in revision; co-author sign-off
(Go Frendi Gunawan) on every change of substance; an arXiv v2 **after** the journal accepts the revision.

**Status 2026-10-05 ±16:00.** The PI asked for a considered recommendation and for it to be carried out; no deadline, no hurry.
The recommendations of the table stand. Reading recorded in `docs/HANDOFF_20261005.md` §2: D1–D6 proceed as recommended
(D6 with a confirmation wanted); **sending** the note of D7 and **the prose** (G16) stay with the PI until he says otherwise.

## 3. Reviewer 1 — thirteen comments plus an opening remark (fourteen rows below: comment 13 is split into 13a / 13b)

Types as in the ledger: **T** wording/classification · **C** clarification · **A** analysis · **D** data.
"Reply must contain" lists facts, not sentences.

| # | Point (paraphrase) | What was found | Change in the manuscript | Reply must contain | Evidence |
|---|---|---|---|---|---|
| 1 | "any proto-form" is too strong; low-level reconstruction is possible | Measured: for Tolaki 70.9 % of uncoded forms have a same-meaning look-alike in Proto-Bungku-Tolaki or in one of 42 other Bungku–Tolaki lists in ABVD (chance 10.5 %; strict threshold 63.4 % vs 3.0 %; coded forms 95.9 %). This shows recurrence inside the subgroup, **not origin**: the reviewer's own example is a word borrowed by speakers of a low-level proto-language (§10.7) | tex 38, 56 as the reviewer words it; tex 60 and 540 use the phrase in another sense — PI decides. Add the Tolaki result where residuals are introduced | agreement; the number with its control; that it is a mechanical screen across 42 lists of 13 languages (41 by one compiler), not etymologies; nothing on origin | E228 S3; E231 D |
| 2 | Makasar: published 38 % retention / 62 % "open" against our figure; anything to add? | (a) The reviewer writes 30.1 %; Table 1 prints **30.9 %** — a slip in the report, mention gently. (b) From ABVD, per concept (201 comparable meanings): retained from PMP **39.3 %**, coded but not the PMP etymon **25.9 %**, no cognate set **34.8 %**. **Corrected twice on 2026-10-06 (§10.7):** 39.3 % [32.8, 46.2] is the share of meanings *in the PMP entry's set* — a lower bound on retention (39–74 %), since an uncoded meaning cannot be in the set; do not write "not retained 60.7 %"; ⚠ no "consistent with the published 38 %" before the PI has read where that figure comes from (possibly the same data). (c) Bugis 48.3 / 28.4 / 23.4, Tae' 48.0 / 33.0 / 19.0: Makasar has 11–16 points more uncoded meanings than its two relatives; its lower raw figure (9 points) is that same fact counted again. ⚠ Whether the uncoded forms are uncoded cognates or replaced words is **not decided** by these data; a mechanical screen does not favour "merely uncoded" (2 of 75 resemble the PMP form; 10 of 80 a Bugis or Sa'dan Toraja form; E231 P8, P9). (d) With doubtful assignments counted, retention is 42.8 % | new short subsection or paragraph + one small table; cite the Makasar literature **after reading it** | the three-way split; ⚠ **not to be written before the PI has read the source of the 38 %** (a specialist's look at the meanings of E231 P9 is optional): the three classes named by what is counted; the uncoded share (11–16 points more; p = 0.006, 0.0001); the screen result; that the data are compatible with the divergence reported in the literature and settle neither it nor its cause (§10.7) | E228 S2; E231 A–C, P1, P8, P9 |
| 3 | "at least ten primary subgroups (Blust 2009)" — the source lists eleven; Sneddon 1993 | Confirmed: the 2013 edition names **eleven "microgroups"** (§2.4.6, p. 82) and reports Sneddon's ten (§4.1.6, p. 193). The book does not call them primary subgroups | tex 57: number, term and edition | corrected count and source, as read by the PI | §8c |
| 4 | Celebic is a supergroup, not a primary subgroup | accepted | tex 65: South Sulawesi, Bungku–Tolaki, Muna–Buton | — | — |
| 5 | "parallel innovation" is a technical term | accepted; but see D4: "independent innovation by each language" is itself no longer what the data show | tex 73, 509 (heading), 517, 598, 614 | that the term is dropped **and** that the claim was re-examined | E228 S3, S5 |
| 6 | "three subgroups" but four listed; Sa'dan Toraja is South Sulawesi; Wolio is Wotu–Wolio; spelling; no abbreviation for Bolaang Mongondow | accepted. The abbreviation is at tex 468 (Table 5), 486 and inside Figure 4; "Toraja-Sadan" also inside Figures 2 and 4. ABVD's own name for list 226 is *Tae' (S. Toraja)*, source van der Veen 1940 — say which list was used | tex 86–87, 237, 298, 405, 468, 486; figures regenerated | four subgroups as the reviewer gives them; the consequence: tex 534 ("the South Sulawesi mainstream represented by Bugis and Makassar") must follow the corrected grouping | E227 A02 |
| 7 | "under-documentation" is the wrong word | The fact meant: 36 % of Tolaki forms have a cognate-set assignment (75 of 209). The cause is not sparse primary data (ABVD holds five further Tolaki dialect lists; of the 86 uncoded forms whose meaning those lists cover, 88 % recur there) | tex 114, 528–534: replace by the number and the S3 result | the distinction: the *list* is well documented, the *cognate coding* is sparse | E228 S3 |
| 8 | Muna *dh* missing from the digraph list | Two Muna forms contain *dh*: *akaradhaa* 'to work', *idho* 'green'; both are cognate-coded; both **were** converted in the test — the sentence omitted *dh* (Muna) and *gh* (Wolio). The script used a fricative symbol as a one-character placeholder; the reviewer describes *dh* as a voiced interdental stop — use the reviewer's word. On the reviewer's guess: no loanword was removed from the classifier data, and one of the two forms (*akaradhaa*) carries ABVD's loan flag (E231 A) | tex 348: complete the list; do not call *dh* a fricative | the two forms; that the count (75 forms changed, 54 in Muna) is unaffected | E227 I01–I03 |
| 9 | Glottal stop is written many ways; same result under each convention? | **No, not quite.** Marker rate by source: Makasar 36 %, Tae' 22 %, Bugis 21 % (ʔ); Tolaki 14 % (apostrophe); Wolio 2 %; Muna 0 %. Re-coding all markers as *q* or *k* costs 0.016 of AUC (25 inputs; 0.020–0.024 across held-out languages), leaving them unwritten 0.006, as geminates 0.007; ≥ 0.710 under every convention (cross-validated; held-out lists down to 0.677). The six conventions are three distinct manipulations and "apostrophe → ʔ" changes no input (§10.7). Verdict by the rule written before the run: *partial dependence*. Eleven Sa'dan Toraja forms end in *k* (⚠ whether that writes a glottal stop: specialist; E231 F). Inside the sources that mark it, marked forms are more often uncoded (OR 2.67 [1.88, 3.79]). ʔ + consonant also counts as a "cluster" in 35 forms | tex 42, 135, 355, 382, 577: the sentence "robust … rather than orthographic artifacts" cannot stand for this property; give the counts per list (four lists mark it routinely, Wolio four forms, Muna none) and leave open whether that is spelling or phonology (⚠ specialist) | the table of conventions; that the method sees only a written mark; the counts per list | E228 S1; E227 D02, D06 |
| 10 | A semantic feature cannot belong to a *phonological* fingerprint | Sharper than the reviewer assumed: of the 26 inputs of the headline model 17 come from the written form, 8 from the meaning, 1 is the language's identity. AUC by group: form only **0.672** (0.641 held-out language), meaning only 0.646, identity only 0.680, form + meaning 0.727 | abstract (tex 39–41), 71, 127, 132–141, 385, 497, 594–595: say what the inputs are; D2 | the breakdown table; which model the quoted AUC belongs to | E228 S7; E227 B03–B04 |
| 11 | Example lexemes for each quadrant | Lists ready (five per cell, three per language and cell). Examples that the PI should check against ABVD and the ACD before use: *candidate with profile* — Tolaki *umi'ia* 'to cry', Makasar *ammikkiriʔ* 'to think' (ABVD marks it a loan), Tae' *limaŋpulo* 'fifty'; *candidate without profile* — Tae' *annan* 'six', Tolaki *omba* 'four', Makasar *jukuʔ* 'fish', Muna *riwu* 'thousand'; *coded with profile* — Bugis *mar-eŋkaliŋa* 'to hear', Tolaki *lumangu* 'to swim'; *coded without profile* — *ama* 'father', *ana* 'child' | table in §3.5 (cells from D3) | the table; that the examples are the forms the classifier scores most extreme and so show its inputs by construction — affixation or compounding is a reading, not tested (§10.7) | E228 S4 (`S4_cell_examples.csv`) |
| 12 | "false positive" / "unlabeled positive" used in two senses; the reviewer also asks directly whether a form that should be positive but is tagged negative is not a false *negative* | Root cause: two label sets (356 and 438) and a "rescue" step that was a concept filter (§5 items 1–2). **The reviewer's question has the answer yes:** tex 94 defines *positive* = Austronesian, while tex 107, 247 and 445 say "false positive" for an inherited form left in the residual — the classes swapped in mid-paper. With D1 the term is not needed: *coded* / *uncoded (candidate)*; "false alarm" only for the decimal compounds | tex 94, 107, 183, 247, 412, 440, 445, 520, 546, 581 | the plain yes to the question; one definition per term; that the confusion came from the authors' own inconsistency, now removed; the reviewer's remark that adapted loans are *expected* to look regular (forms "worth a closer look", not errors) | E227 A10–A14 |
| 13a | What does "documentation gaps" mean for Tolaki? | Not the first reading (the list is not short of words). The second (cognates exist, not yet assigned) is visible for part of the forms: of the 34 uncoded forms that match a Proto-Bungku-Tolaki entry, 7 match an entry that ABVD does assign to a cognate set, — only these 7 are gaps in the plain sense. 12 of 128 are look-alikes of ABVD's PMP form for the same meaning by a mechanical screen (about two expected by chance; coding gap, chance or loan not distinguished; E231 D, P5; ⚠ specialist). For the rest the study cannot decide | tex 534 | both numbers | E228 S3 |
| 13b | Release the residual lists | Files built: `experiments/E228_p8_revision_analyses/release/p8_candidates.csv` (438 rows; **Tolaki 134**, not 114 — say why) and `p8_forms_all.csv` (1,357 rows) with ABVD ids, flags, out-of-fold probability, nearest look-alike. ABVD is CC BY 4.0. Deposit: Zenodo (no cost; D1/D2 precedent) → DOI in the data-availability statement | tex 619 | DOI; column description | `release/` |
| — | Opening remark: the SHAP figure is unreadable for a linguist | Figure 1 carries a software title, raw variable names and (in the caption) the wrong sign | regenerate as a plain bar chart for the D2 model | — | E227 E04 |

## 4. Reviewer 2 — twenty-four highlights, by group

| Group | What was found | Change | Evidence |
|---|---|---|---|
| Internal codes (E022, E027, E028, E029, E041) | 29 occurrences on 23 lines (tex 97, 117, 174, 178, 181–184, 187, 191, 229, 370, 392, 396–397, 404, 411–412, 414, 435, 477, 512, 613) and inside the titles of Figures 2 and 4. "E022 binary label" cannot be found in the rule-based section because the label is **not** what that section describes (§5 item 1) | names instead of codes; D1 | E227 A10 |
| AUC — meaning and scale | Plain meaning available: the chance that a randomly chosen uncoded form is ranked above a randomly chosen coded one; 0.5 = coin, 1 = perfect. Conventions for a verbal scale differ; a rule of thumb is widely quoted (⚠ the PI reads the source before citing; §10.7); the "0.65 threshold" at tex 313 is the project's own go/no-go line (E227 C18) — say so or drop it. ± 0.007 is the SD of ten seed means; across the 50 folds it is 0.028 (27-input model; for the 25-input model 0.007 and 0.027, E229 T2) | definition at first use; one honest sentence on size | E227 C01, C18 |
| "near-perfect", Table 2 walk-through, "both models" | Model A: 31 inputs, AUC 0.9997–1.000 because four of them encode the label. F1 in Table 2 is the F1 of the **coded** class; for the candidate class it is 0.53 (27-input model). Accuracy 0.741 vs 0.677 for always answering "coded". XGBoost was **not** class-weighted (tex 153 says it was) | Table 2 rebuilt for the D2 model with the candidate-class score and the baseline; tex 153 corrected | E227 C05–C09 |
| SHAP, feature ablation, typewriter font, Δ, CV, κ | SHAP top five reproduce, but are those of the 27-input model; for the 25-input model see `E227/results/shap_pure25.csv`. `language_cognacy_coverage` does not hold the coverage (hard-coded stale values; E227 B02) — with D2 it disappears | glossary box or first-use definitions (PI); Figure 1 regenerated | E227 B02, E01–E05 |
| "classifier … robustness" (three classifiers) | RF 0.762, LR 0.747, XGBoost 0.760 (27 inputs) reproduce | keep, for the D2 model | E227 C08 |
| "language identity" feature | an arbitrary integer 0–5 in alphabetical order of the list names; the strongest input of the submitted headline model | D2 removes it; if kept, describe it as what it is | E228 S7 |
| Source of the semantic domains (Concepticon? WOLD?) | Neither. The seven domains and the "core" flag are hand-made concept sets in the feature script. The "Swadesh-100" flag covers **174 of the 210** meanings (83 % of forms; the set lists 176 names, two match no ABVD concept). ABVD's 210-item list is itself not a Swadesh list in the strict sense ⚠ (PI: check how ABVD describes it) | tex 89–90, 139: describe as the authors' grouping; correct or drop the "Swadesh-100" flag; mapping to Concepticon is possible future work, do not claim it | E227 E06 |
| "scikit-learn / XGBoost — Python?" | Python 3.11; xgboost 3.0.3, scikit-learn 1.8.0, shap 0.50.0 reproduce the stored results | one clause | E227 summary |
| Why k = 5 … 30; DBSCAN parameters | No reason: 30 is the edge of the range and the silhouette keeps rising beyond it (0.126 at 40, 0.205 at 100); the value also shifts with row order (0.102–0.114). Ward linkage presupposes Euclidean distances | with D4 the clustering paragraph can shrink to two sentences or go; the permutation test replaces it | E227 G01–G03 |
| What "false positive" stands for | as R1-12 | D1 | — |
| Figure 2 panels not referred to; Table 5 cut off | Figure 2 has four panels (A–D) with internal codes in titles and axes; panel D itself shows prefix-like onsets **higher** in the "consensus" group, against the text. Table 5 has seven columns under `\small` and runs off the page | regenerate (D3) or drop Figure 2; Table 5 rebuilt or dropped (D5) | figures; E227 D01 |
| Why 'One Hundred', 'Fifty', 'Twenty', 'to stand', 'to hit' in ≥ 4 languages | ⚠ 'Fifty' and 'Twenty' look like decimal compounds (for a specialist); 'One Hundred' is not uniform (the Makasar form looks like a different word); all three are uncoded in four lists and coded in Bugis. The other two were on the submitted version's own list of inherited meanings; the "consensus" was computed on the label that ignored that list. Out of fold, **no** meaning is "candidate with profile" in four languages | D1 + D3 remove the paragraph's premise; keep the numeral observation | E228 S4; E227 F12–F13 |
| "generalises across Sulawesi languages" concretely | = a model fitted on five lists ranks the forms of the sixth: AUC 0.61–0.81 (25 inputs: mean 0.701, five of six ≥ 0.65). Accuracy for Tolaki and Muna in Table 3 (0.36, 0.40) is far below the majority baseline — a threshold effect that the text passes over | say it in those words; comment on the two accuracies or drop the column | E227 C10–C12 |
| Define "fingerprint" early | see R1-10; the measured differences (uncoded vs coded forms): longer (6.10 vs 5.20 characters; 2.57 vs 2.29 vowel groups), more consonant-letter clusters (0.48 vs 0.32), a written glottal mark where the source writes one (23.5 % vs 11.6 %; four lists routinely), action meanings (40.4 % vs 23.4 %). The onset-string input (37.0 % vs 25.2 %) is **not separable from length** — "fewer prefixes" is withdrawn, "more" is not established (§10.7) | Introduction | E227 D01–D05 |
| Citation for "resists reconstruction" (p. 2) | open — a source has to be found and read by the PI; none is proposed here | tex 56 | — |

## 5. Found while verifying — some prompted by the reviewers' questions, some beyond them (E227 row ids)

Each is a statement of the submitted text that the data or the code contradict. None can be mended by wording.

1. **Two residual sets presented as one** (A07, A10, A14, F06, H02): 356 forms in Table 1; 438 in every model, in
   Table 3, in the agreement table and in the clustering. Abstract: "438 … (26.5 %)" — 438 is 32.3 %.
2. **The Proto-Austronesian cross-check compared no forms** (A13): all uncoded forms for 15 meanings were removed.
   `E227/results/pan_rescued_75.csv` — e.g. 'rope': *tulu, otɛrɛʔ, kakoo, oloo, puloli* against *\*talih*.
3. **The loanword lists matched five unrelated words** (A15), among them Tolaki *beli* 'blood' (Malay *beli* 'buy'),
   which has an exact counterpart in Kodeoha.
4. **"In five or more of six languages" counted forms** (A16): three of the eight meanings qualify.
5. **"Fewer canonical Austronesian prefixes" is not what the data show** (D01), in the abstract and at tex 385, 448,
   497: the onset-string input is more frequent among uncoded forms (37.0 % vs 25.2 %), but at equal length it does
   not separate them (1.09 [0.80, 1.48]; E231 P3). The argument at tex 447–448 that rests on it fails; whether
   morphological complexity explains the profile was **not tested** *(corrected 2026-10-06; §10.7)*.
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
12. **The robustness numbers of §3.4 belong to another model** (E230 Part A; added 2026-10-06): AUC 0.772 → 0.774, 0.768 → 0.769 etc.
    reproduce, but for a 26-input model that includes the language-identity code, under other fold seeds, and E041's
    "orthographic" baseline took seven of its inputs from the converted strings. "Does not depend on form length at all"
    (tex 359, 449) is **false for the models of the revision**: removing both size inputs costs 0.014–0.023 of AUC.
    "Muna shows the largest improvement" (tex 354) does not hold in the form-only model. The abstract's "phonological
    patterns rather than orthographic artifacts" (tex 42) has no test behind it.
13. **The arrows of the SHAP figure describe the classifier, not the vocabulary** (E229; added 2026-10-06): for *ends in a
    vowel* the model's direction is the reverse of the within-list association. All 285 consonant-final forms are in
    the three South Sulawesi lists, so this form input also identifies the source list; it also overlaps with the
    glottal input, and which of the two accounts for the sign was not tested (E231 F).

## 6. The numbers after correction (for the PI's tables; all from E227/E228)

| Quantity | Submitted | Corrected | Note |
|---|---|---|---|
| Candidate forms | 438 "(26.5 %)" | 438 = **32.3 %** of 1,357 | D1 |
| Table 1, per list (Muna, Bugis, Tae', Wolio, Makasar, Tolaki) | 26, 49, 32, 68, 67, 114 | **34, 62, 45, 83, 80, 134** = 15.5, 25.6, 20.8, 32.7, 36.9, 64.1 % | D1 |
| Share of forms with a cognate set | 84, 74, 79, 67, 63, 36 % | same (84.5, 74.4, 79.2, 67.3, 63.1, 35.9) | unchanged |
| Headline discrimination | 0.763 ± 0.007, "phonological only" | form only **0.672** (held-out language 0.641); form + meaning **0.727** (0.701) | D2 |
| Lists ≥ 0.65 when held out | 6 of 6 | 5 of 6 (form + meaning); 3 of 6 (form only) | D2 |
| Without Tolaki | 0.698 (Δ −0.062) | 25 inputs: 0.701 (Δ −0.025) | E227 C16–C17; E229 T4 |
| Agreement label × classifier | κ 0.611; 266 / 878 / 172 / 41 | κ **0.31**; 172 / 814 / 266 / 105 (out of fold) | D3 |
| Meanings "consensus" in ≥ 4 languages | five | none | D3 |
| Cross-language test | p = 0.569 | permutation p ≤ 0.0001 (none of 10,000); look-alike pairs 4 vs 0.5 expected (coded forms 472 vs 8.5); the test covers uncoded forms only (§10.7) | D4 |
| Forms beginning with one of nine letter strings (the code's "prefix-like") | "fewer" | not fewer: 37.0 % vs 25.2 % (OR 1.57 [1.20, 2.04]); at equal length 1.09 [0.80, 1.48] — not separable from length (E231 P3) | fact |
| Glottal convention | "robust" | −0.006 to −0.016 AUC depending on convention | R1-9 |
| 16 lists: mean AUC / mean probability Sulawesi vs west | 0.663 / 0.606 vs 0.393 "significantly" | 0.638 (≥ 0.60 in 10 of 16) / 0.308 vs 0.229 (p = 0.023; among coded forms p = 0.37) | D5 |
| Makasar | 30.9 % residual | 39.3 % in the PMP entry's set · 25.9 % coded otherwise · 34.8 % uncoded (of 201 meanings); among coded meanings 60.3 % in the PMP set, as in Bugis and Sa'dan Toraja (E231 P1) | new |
| Table 2: F1 / accuracy | 0.822 / 0.741 (27 inputs; F1 of the *coded* class) | FM25: accuracy 0.719 against 0.677 for always answering "coded"; F1 of the **candidate** class 0.480 (precision 0.597, recall 0.404); form only 0.696 / 0.391 | E229 T2 |
| Three classifiers | 0.747–0.762 | FM25 0.700–0.727; form only 0.671–0.683 | E229 T2 |
| Held-out list, accuracy | 0.36–0.71, no baseline | below the majority answer in 5 of 6 lists (both models); only the ranking carries over | E229 T3 |
| Digraph ("IPA") conversion | 0.772 → 0.774 (+0.002) | FM25 −0.0003, form only +0.003; 75 forms changed (5.5 %) — "unchanged", nothing more | E230 D1 |
| Length "can be removed" | CV 0.769 ("equivalent") | `form_length` alone: no change; with the vowel count also removed: −0.014 (FM25), −0.023 (form only) | E230 L2, L3 |
| Profile, with intervals | five properties, one reversed | a correlated bundle: size, written glottal mark, action meaning (the last two hold at equal length: 2.60, 2.22); onset string and nasal input not separable from length; "reduplication" and "final vowel" do not stand as separate properties; table with 95 % CI | E229 T6; E231 E, F, P3, P11 |
| Tolaki | "artifact of under-documentation" | 70.9 % of uncoded forms have a look-alike within Bungku–Tolaki (chance 10.5 %; coded forms 95.9 %); by meaning (first form), on shared meanings, list 674 is coded for 38.3 % and the five Tolaki dialect lists for 36–41 %, against a median of 46.7 % in 42 comparison lists (E231 D, P10b) | new |

## 7. Order of work and gates

| Step | Who | What | Gate |
|---|---|---|---|
| 1 | PI | D1–D7; read §3–§5; open `release/p8_candidates.csv` | — |
| 2 | PI | read the references a reviewer supplied (§8) — none is cited before that | citation integrity |
| 3 | PI (+ co-author informed) | short note to the editor (D7), with the one-line question about author-side charges (§8e) | G15 |
| 4 | Claude | for the model chosen in D2: Tables 1–3 as data files, Figure 1 as a plain chart, the examples table (R1-11), the Makasar table (R1-2); `VENUE.md` check of two or three recent OL articles for table and example conventions | G15 |
| 5 | PI | writes abstract, §2–§4, conclusion; first-use definitions for the ML terms (R2); Section 3 rewritten for readability — with D3–D5 three of the four figures and two tables go (count in the outline §9); whether it ends up shorter is known only once it is written | G16 |
| 6 | Claude | language check; every number in the new text against E227/E228 (G1 on the new text); overstatement and causal-connector scans (G8, G11); check of the PDF that will be uploaded (G1-bis item 3) | G1, G8, G11 |
| 7 | — | freeze ≥ 14 days; PI re-reads in full | G14 |
| 8 | PI | response letter (R1 13 comments, R2 24 highlights, plus a section "corrections by the authors"); upload; download the file back and compare | G16, G12 |
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
§5) says a retention percentage is: reconstructed basic vocabulary compared with its reflexes in the modern language.
⚠ E228 S2 does **not** compute that quantity: it counts set membership in ABVD and treats every uncoded meaning as
not retained (§10.7). And if the published figure and ABVD's Makasar coding go back to the same lists, the two
numbers are one dataset read twice — to be checked by the PI when reading the source.

**(c) R1-3, the number of subgroups.** Blust (2013 revised edition, §2.4.6, p. 82) recognises **eleven "microgroups"** in
Sulawesi and names them; §4.1.6 (p. 193) reports Sneddon's (1993) ten and Mead's addition of Wotu–Wolio. The reviewer is
right. Two cautions for tex 57: the book's word is *microgroups* — according to the reading note it treats two of the eleven as
primary branches of the Philippine subgroup and a third (Gorontalic) as part of Greater Central Philippines (⚠ the PI
reads the sentence), so "primary subgroups" is not its claim; and only the 2013 edition was read, while the manuscript cites the 2009 one.

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

---

## 10. Additions of 2026-10-06

### 10.1 The ledger checked against the original reports

The decision email and the annotated PDF of reviewer 2 were opened again on 2026-10-06 (the PI's mailbox; copies kept
outside the repository). **Every point is in the ledger: reviewer 1 = 13 separated comments plus an opening remark;
reviewer 2 = 24 highlights, all 24 found in `REVISION_OL_20261005.md` §5.** Five details that the ledger and this
plan had not carried:

1. **R1-12 contains a direct question** (is a form that should be positive but is tagged negative not a false
   *negative*?). The answer is yes — row 12 of §3 now says so.
2. **The formal letter.** The decision letter says that a formal letter of acceptance — conditional on the revisions
   being made to the editor's satisfaction — is available already now, before the revision is in. The ledger had "sent after that". Whether to
   ask for it is the PI's call; the honest order is the note of D7 first (draft: optional lines).
3. **Spelling.** *Makasar* is mentioned as the preference of some linguists, in a side remark; the report itself uses
   both spellings. *Sa'dan Toraja* is recommended for English. The abbreviation of Bolaang Mongondow is called unnecessary.
4. **R1-3** opens by granting that the sentence is true as stated. The reviewer then observes that the cited source
   lists eleven and that Sneddon 1993 could be cited as well — an observation and a suggestion, not a demanded change.
   Giving "eleven microgroups" with the edition actually read is the authors' choice.
5. **What the reviewers say about the computations** *(heading corrected 2026-10-06: "neither checked" was an inference)*. Reviewer 1 states that the computational parts are outside their
   field and hopes another reviewer covers them; reviewer 2's method-related highlights ask for explanations and for the
   reason behind choices (the range of k, which models, what "near-perfect" means), not for a re-analysis. The defects
   of §5 were not raised as such by this review, although several reviewer questions lead to them — a consideration for
   D7, **not** something to write to the editor.

For the response letter: number the answers by the reviewer's own thirteen comments (13a/13b of §3 are one comment).

### 10.2 Tables and Figure 1 for the model of D2 — done (`experiments/E229_p8_revision_tables/`)

| For | File | What it gives |
|---|---|---|
| Table 1 (D1) | `results/T1_label_by_list.csv` | one label: 34 / 62 / 80 / 83 / 45 / 134 candidates (Muna, Bugis, Makasar, Wolio, Sa'dan Toraja, Tolaki), 438 = 32.3 % |
| Table 2 (R2: walk-through, "both models", "near-perfect", classifiers) | `T2_cv_performance.csv` | two input sets × three classifiers; the score of the *candidate* class; the "always coded" baseline 0.677; SD of seeds and SD of folds |
| Table 3 (R2: "generalises") | `T3_lolo.csv` | AUC per held-out list (mean 0.701 / 0.641) and the accuracy against the majority answer |
| Tolaki sensitivity | `T4_without_tolaki.csv` | −0.025 (FM25), −0.026 (form only) |
| cells (D3; R1-11, R1-12) | `T5_cells_by_list.csv` | 172 / 266 / 105 / 814 per list; κ 0.31 |
| the profile (R1-10; R2 "define the fingerprint") | `T6_profile.csv` | ten properties, candidates vs coded, within-list effect with 95 % CI |
| examples (R1-11) | `T7_examples_R1-11.csv` | 92 rows (73 different forms) in four cells with ABVD's PMP / PAn entry, loan flag, nearest Sulawesi look-alike |
| Figure 1 (R1 opening remark; R2 SHAP) | `F1_input_importance.{tif,png,pdf}` | plain bars, 312 pt wide, wording in `labels.csv` |

Verified: 75 anchors against E227/E228; T1 and the XGBoost rows of T2/T3 re-derived by a second script from the stored
March feature matrix (163 cells, no difference). Three things the PI should know before writing from these tables:

- **As a yes/no classifier the model is close to the trivial answer** (accuracy 0.719 against 0.677; it finds 40 % of
  the candidates) and **below the majority answer for a list it has not seen** (5 of 6 lists). What carries over is the
  *ranking* (AUC 0.70). R2's question what "generalises across Sulawesi languages" means has this as its honest answer.
- **Figure 1 is about the classifier.** For "ends in a vowel" its direction is the reverse of what the lists show.
  The input carries list membership (consonant-final forms exist only in the three South Sulawesi lists) and overlaps
  with the glottal input (E231 F); which of the two accounts for the sign was not tested. Statements about the
  vocabulary rest on T6.
- The profile (T6) is a correlated bundle of measured differences: candidates are about one character longer and more
  often carry a written glottal mark (four lists routinely; four forms in Wolio, none in Muna) and an action meaning.
  The onset-string input is not separable from length (§10.7). That these are morphologically complex citation forms
  is **a reading, not tested** — no form was parsed — and reviewer 1's report contains no such reading *(corrected
  2026-10-06; §10.7)*.

### 10.3 The robustness section (tex 347–359) — re-derived and re-run (`experiments/E230_p8_revisit_E041_E042_digraph_length/`)

Not covered on 2026-10-05. Reviewer 1's points 8 and 9 are about this section, and the abstract draws its
"not orthographic artifacts" from it. Decision rules written before the run (same day, not registered externally);
result in §5 item 12 and §6. What can be said after it:

| Sentence of the submitted text | Status | What the data allow |
|---|---|---|
| tex 42, 355: robust to IPA conversion, "phonological patterns rather than orthographic artifacts" | cannot stand | converting five digraphs, four of which occur, to single symbols (75 forms) leaves discrimination unchanged; the glottal convention changes it by up to 0.016 (E228 S1) |
| tex 348: digraph list | incomplete | `ng`, `ny` (all lists; `ny` converts nothing); Muna `gh` 29 forms, `bh` 14, `dh` 2; Wolio `gh` 0 |
| tex 354: Muna improves most, digraphs were noise | withdrawn (rule fixed in advance) | Muna +0.017 with form + meaning, −0.010 with the written form alone |
| tex 358: syllable count instead of characters | holds, as "no change" | Δ −0.0003 / +0.003 |
| tex 359, 449: "does not depend on form length at all" | **withdrawn** | −0.014 / −0.023 when both size inputs are removed; size is one of the things the classifier uses |
| tex 447–449: the two observations offered against the morphological-complexity concern | **both fail** (the first by E227 D01 and E231 P3, the second by E230) | the two observations do not hold; the concern itself was not tested (no form was parsed) |

⚠ Domain flag: the symbols used for Muna *bh*, *dh*, *gh* in the conversion are placeholders; nothing about Muna
phonology may be printed from E230 without a specialist.

### 10.4 Venue facts (`VENUE.md`)

The journal requires a **Word** file for the final version and does not support LaTeX; asks for at most 30 pages;
abstracts are one paragraph of typically 120–160 words (ours: about 260); figures ≤ 312 pt as separate `.jpg`/`.tiff`;
reconstructions not italic, glosses in single quotes, "percent" in running text. Fees are stated nowhere (G15 open).

### 10.5 Decisions added for the PI

| # | Decision | Recommendation | Why |
|---|---|---|---|
| **D8** | Where the revision is written | in the journal's Word template from the start | a Word file is required anyway; converting a finished LaTeX text costs a second proof-reading |
| **D9** | The robustness subsection (§3.4) | reduce to one short paragraph with the three numbers of §10.3 and the glottal table of E228 S1; drop the abstract sentence | it answers R1-8 and R1-9 directly and removes two false sentences |
| **D10** | Ask for the formal (conditional) acceptance letter | only after the editor has answered the D7 note | asking before disclosing would look like securing the letter first |

### 10.6 Order of work — status

Step 4 of §7 is done (tables, Figure 1, examples, Makasar and glottal tables, `VENUE.md`). Also done: the corrected
`.bib` in `revision_v0.2/` (8 corrections, 2 deletions; compiles, 0 BibTeX warnings); the note to the editor as a fact
list (`docs/correspondence/EMAIL_OL_EDITOR_P8_NOTE_DRAFT_20261006.md`, **not sent**; no English draft); the content
outline per section (`REVISION_OUTLINE_20261006.md`, facts in order, no sentences; its interpretive items were
rewritten after the adversarial read — §10.7). Next is step 1–3 and 5 — all the PI's: decisions,
reading, the note, the prose.

### 10.7 Readings corrected after three adversarial reads (E231) — 2026-10-06

Two adversarial reads on 2026-10-06 (Opus). The first (17 findings, 7 high) found **every number sound** and **several
readings unsupported**; `experiments/E231_p8_what_coded_means/` replaced those readings by counts. The second (14 new
findings, 3 high) found that three of the *replacement* readings were themselves wrong; the orchestrator re-derived its
counts with an own script (`02_posthoc_checks.py`, tables P1–P7; three anchors reproduce) before changing anything.
Number trace of the rewritten documents by four Sonnet readers: 902 statements, 890 matching, 12 small differences
(rounding, counts of entries against spellings), all corrected. A third read (9 new findings, 2 high) found the
second round's Makasar wording over-corrected and the E228 README untouched at S2; its counts were re-derived as
tables P8–P12 (trace of round 2: 340 statements, 337 matching). The third round was traced but not read adversarially again.
**Where a cell of §1–§6 or §10.2–10.3 still reads otherwise, this section governs; the cells named by the second read
were edited in place.** The middle column is working English for the PI's orientation, not text for the paper (G16).

| Reading that had been written | What the data show | Source; where it now stands |
|---|---|---|
| "Coded in ABVD" = inherited Austronesian; "uncoded" = an upper bound on non-Austronesian vocabulary; the Makasar middle class "has Austronesian cognates" | A set number = grouped with other forms by ABVD's editors. ABVD flags loans only occasionally (11 forms; none in the Sa'dan Toraja list); 5 of the 11 carry a set number, one of them (Tolaki *kila* 'lightning', ABVD: from Malay) in the PMP entry's own set. Of Makasar's 52 middle-class meanings 33 % are in sets of ≤ 5 lists and 17 % in sets confined to the Sulawesi box | E231 A; outline §0 item 1, §1, §5.2 |
| *(round 1 wrote:)* "a third of Muna's coded forms sit in local, Sulawesi-only sets — coding of the neighbourhood" | **Withdrawn.** 44 of Muna's 185 coded forms — 44 of the 62 "Sulawesi-only" ones — are coded only because they recur in ABVD's second Muna list (Wuna). Counted as uncoded they put Muna at 35.6 %, not 15.5 % (a counterfactual: ABVD does number some singleton sets). Tolaki's five dialect lists were not cross-coded in that way (0 forms). Label sensitivity (E231 P12): with the 44 relabelled, CV AUC 0.727 → 0.741 (form + meaning), 0.672 → 0.699 (form only) — the headline does not rest on them. "Sulawesi-only" is a coordinate box of 96 lists (49 Bungku–Tolaki, 17 Bajo, 3 South Sulawesi target lists), and breadth counts lists, not languages | E231 P2, P6; outline §0 item 1, §5.1, §5.2, §5.7 |
| The published Makasar figure is "reproduced" *(round 1: "consistent with")*; "retained 39.3 %", "not retained 60.7 %" | 39.3 % [32.8, 46.2] = meanings whose form is in the PMP entry's set: a **lower bound** on retention (39–74 %). **Source read 2026-10-06:** the 38 % is Blust's (1981a) lexicostatistical count against a 200-word PMP list, quoted by Bellwood 1997: 115 beside a Western Malayo-Polynesian mean of 41 % (Sundanese 35, Javanese 30). ABVD's PMP list is Blust's too, so the agreement is **not independent confirmation**, and in the source the figure is ordinary, not low. A more independent comparator: Sirk 1989, Table 1 — Makasar–Bugis 45, Makasar–Sa'dan 42, Bugis–Sa'dan 60 against 41.4 / 39.2 / 53.1 % shared sets in ABVD (same order, similar gaps). With doubtful assignments 42.8 % | E228 S2; E231 B, C, P1; outline §5.2; `REFERENCE_CHECK` Part D |
| *(round 1 wrote:)* "Makasar retains less than its relatives and shares fewer sets with them — two measures that agree"; *(round 2 wrote:)* "only not yet coded", "the study does not show that Makasar is more divergent" | **Both withdrawn.** One fact, not three: more **uncoded** meanings (34.8 % against 23.4 % and 19.0 %; p = 0.006, 0.0001, exact McNemar, unadjusted). On meanings coded in both lists: in the PMP set 70 against 69 of 110 (Makasar, Bugis) and 70 against 71 of 115 (Makasar, Sa'dan Toraja); shared sets 76.3 / 70.1 / 81.0 % — figures that are the same whether the uncoded forms are uncoded cognates or replaced words, so they decide nothing. Screen: 2 of 75 uncoded Makasar forms resemble the PMP form (about one by chance; sensitivity 38 % on forms ABVD puts in the PMP set); 10 of 80 resemble a Bugis or Sa'dan Toraja form of the same meaning (coded forms: 90 of 136). A small excess over chance within South Sulawesi (12.5 % against 3.8 %), far below the coded forms (66 %): not in favour of "merely uncoded" for most of them; compatible with the lexical divergence reported in the literature; no proof of it and nothing on its cause. For a specialist: the 28 (Bugis) and 25 (Sa'dan Toraja) meanings where the relative is in the PMP set and Makasar is uncoded (`P9_makasar_uncoded_meanings_for_specialist.csv`) | E231 B, C, P1, P8, P9; outline §0 item 4, §5.2, §6 item 2 |
| Tolaki's uncoded forms are "ordinary" subgroup vocabulary, "exactly the reviewer's scenario"; *(round 1:)* "36 % is the subgroup's rate", "top-level coding gaps", "a floor"; *(round 3, first version:)* "the Konawe pair shows coding status belongs to a list, not a language" | 70.9 % have a look-alike within Bungku–Tolaki (chance 10.5 %; coded forms 95.9 %); the list is not short of data (200 of 210 meanings; 88.4 % of the covered candidates recur in the dialect lists). Coded share — **the unit decides the answer.** By form: 4 of 42 comparison lists at or below Tolaki, but the share by form falls with list length (ρ = −0.55). By meaning on shared meanings, first form of each meaning: list 674 38.3 %, the five Tolaki dialect lists 36.1–40.6 %, median of the 42 comparison lists 46.7 % (5 at or below). With "any form coded" as the unit, lists rich in synonyms rise (Konawe, 384 forms: 50.4 %; median 48.4 %, 3 at or below) — an effect of synonyms, not of coding. So the six Tolaki lists sit on the lower side of a thinly coded subgroup, by a moderate margin, among lists almost all by one compiler. 12 of 128 candidates are look-alikes of the PMP form (about two by chance; one loan-flagged; nine at the strict threshold; screen sensitivity 48 %). Nothing on origin | E228 S3; E231 D, P5, P10, P10b; outline §5.3 |
| "No large shared layer across the four subgroups" | Measured: among **uncoded** forms, same-meaning forms are slightly more alike than different-meaning ones (0.836 against 0.869; p ≤ 0.0001, none of 10,000 permutations; 4 look-alike pairs against 0.5 expected). Not shown: the absence of a shared layer (the coded class is not examined; three of the four subgroups fall inside the proposed Celebic supergroup — ⚠ PI reads Blust 2013: 82; all four pairs involve Wolio). Size: about one eighth of the effect in coded vocabulary on mean distance, about one twenty-ninth on look-alike pairs (two different bases). Inside South Sulawesi 0.046, elsewhere 0.030 — **not distinguishable** (about one combined SD). The meanings that pull the statistic down most include prefixed quality terms ('heavy', 'dull', 'near'), and two of the four pairs begin with *ma-* on both sides (⚠ prefix or root: specialist). 68 profile-only pairs: too few to tell. Leave-one-meaning-out (E231 H): largest change 0.003 | E228 S5; E231 A, H, P7; outline §5.7 |
| The profile "is" that of morphologically complex citation forms; *(round 1:)* the bundle includes the onset string; "more prefixes, not fewer"; contrasts "survive stricter definitions and get larger"; *(round 2:)* "clusters hold at equal length" | Holding at equal length (E231 P3; the third reader's own re-runs with coarser strata and with a logistic model gave the same picture — those runs are not stored): written glottal mark (2.60 [1.79, 3.76]) and action meaning (2.22 [1.67, 2.95]); size itself (+1.09 characters; letters only +0.93). Consonant-letter clusters (1.51 [1.09, 2.09]) are **not** on the same footing: they overlap with nasal + stop pairs and with ʔ + consonant; with the glottal mark also held fixed 1.38 [0.98, 1.93]. **Not separable from length:** onset string — 1.09 [0.80, 1.48] with all letters held fixed, 2.51 [1.85, 3.40] with the letters after the matched string held fixed (two ways of equalising length, two answers); nasal input 1.24 [0.84, 1.81] strict, 0.95 as coded. "Fewer prefixes" is withdrawn; "more prefixes" is not established; "no difference" is not established either. Morphology: a reading, not tested; the one compiler-marked segmentation (hyphens) is mixed (Bugis 34.4 % against 22.5 %; Sa'dan Toraja 20.0 % against 20.9 %; not tested per list) | E229 T6; E231 E, P3, P4, P11; outline §0 item 3, §5.5, §6 item 1 |
| "Reduplication" *(round 1: "does not survive")* | The estimate is the same under each definition (1.68; repeated block 1.61 [0.87, 2.98]; hyphen 1.57 [0.94, 2.63]); with half the forms the interval includes 1 — too little to stand alone, not evidence of absence | E231 E, P4; outline §5.5 |
| "Ends in a vowel" as a property; the SHAP sign explained by list membership | The final glottal mark counted twice: with glottal-final forms set aside 0.91 [0.58, 1.44] (no difference detected; wide interval). The input carries list membership **and** overlaps with the glottal input; which accounts for the sign was not tested | E231 F; outline §5.5 |
| Glottal mark "written in four sources" as a spelling habit; the reviewer's *k* "is in the data" | The Muna list has no glottal mark and the Wolio list four (two candidates, two coded) — ⚠ spelling or phonology is for a specialist, and it changes the answer to R1-9. Sa'dan Toraja: 45 forms end in a glottal mark and 11 in *k*, side by side in one source, and ABVD's PMP entries for 'bird' and 'sea' end in *k* — do **not** present that *k* as a glottal stop. Six conventions = three manipulations; no noise estimate; "partial dependence" is a threshold rule, not a test | E227; E228 S1; E231 F, P7; outline §5.6 |
| "Pre-registered" | Decision rules written before the analysis — the same day, by the same analyst, not registered externally. The E228 design carries two later amendments (hash differs from the freeze hash); design and results entered git in one commit | outline §4 (2.7); E228 README |
| "No agreed verbal scale" for AUC; pooled AUC 0.737 as headline | A rule of thumb is widely quoted (⚠ named from memory — the PI reads the source). The pooled figure averages probabilities over ten repetitions (a small ensemble) and its interval leaves out training variation; headline stays 0.727 / 0.672, "about ± 0.03" | E231 G; outline §4 (2.5) |

**D4 — what the PI now decides (three options).**
**(a)** Keep the cross-list test in the Results as one statement on the small excess among uncoded forms, with its limits
(the coded class is not examined; part of the similarity may be a shared prefix). *Recommended; nothing further to compute.*
**(b)** In addition, a test on the coded class (are sets confined to Sulawesi shared across the six lists more often than
expected?). ⚠ What it cannot deliver: such sharing is expected from subgroup inheritance alone (three of the four
subgroups are placed in Celebic), "Sulawesi-only" is a coordinate box with 17 Bajo lists and three South Sulawesi lists,
and for Muna it is mostly the second Muna list — so (b) would describe ABVD's coding once more and could support neither
"a shared layer" nor "no shared layer". It needs a new design and a specialist. *Not recommended.*
**(c)** Take the cross-list test out of the Results and keep one sentence in the limitations. No reviewer asked for the
test; under (a) it yields a small effect of unclear meaning.
Under every option the wording reviewer 1 offered in place of "parallel" (*independent* innovations) is **not** adopted,
because the data do not show independence either; the response letter says so and gives the reason. The paper then has no
cross-language claim, positive or negative. The note to the editor announces nothing stronger than (a).

**Decisions that the second and third reads add for the PI:**
1. Table 1 with a note on the second Muna list (recommended: yes), and whether the label-sensitivity row (E231 P12) is
   printed (recommended: one sentence).
2. Makasar: the sources of the published figure were fetched and the pages located on 2026-10-06 (`REFERENCE_CHECK`
   Part D); what is left is the PI's own reading of three pages before citing them — Bellwood 1997: 115, Sirk 1989:
   71 (Table 1), Bulbeck et al. 2000: 103 — and of Mills 1975: 491–492 if Mills is cited (the reviewer's pages
   341–342 do not carry the claim). The section itself needs no specialist: three classes, the uncoded share, the
   screen result, the comparison with Sirk's table, and "undecided".
3. **Whether a Sulawesi specialist is needed** *(recommendation revised 2026-10-06, 12:00, after the PI asked; the first
   version said "before drafting §3" and made an outside reader a precondition, which it is not)*. **Not a blocker**, as
   long as the paper prints no claim that needs one: forms are printed as ABVD records them (form, gloss, set number)
   with one sentence that they were not assessed etymologically; counts are given without interpretation (glottal
   marks per list; nothing on final *k*, on hyphens, on numeral etymologies, or on why Makasar forms are uncoded).
   Reviewer 1 is a specialist in these languages, asked for the examples and the lists in order to inspect them, and
   will see the revision. What still binds is the project's own gate **G10**: one linguist (a co-author or colleague,
   not necessarily a Sulawesi specialist) reads the finished draft before resubmission — inside the 14-day freeze
   (G14), so it adds no delay. Optional and not blocking: sending `P9_makasar_uncoded_meanings_for_specialist.csv` to
   a Sulawesi linguist — the one place where a specialist could add something the paper cannot otherwise say.
4. Whether the twelve Tolaki / PMP pairings stay listed by form in a public file (the E231 README now gives the glosses
   and points to the CSV; the CSV itself is in the repository).
5. The note to the editor: AI assistance and the preprint (its section C); if D4 = (c), its item A4 bullet 5 falls away.
6. The release description (`experiments/E228_p8_revision_analyses/release/README.md`) is a **draft**: before deposit
   the PI supplies authors, date, version, licence and citation, the reason why Bajo lists were left out and on what
   basis the 42 list ids were taken to be Bungku–Tolaki, and rewrites its section 3 in his own words (G16).
**Other items of the two reads, closed in the files named:** the editor note (fact list in Indonesian; no English draft;
says that the claim changes, that some printed figures and two tests are wrong and that some numbers were constants; no
longer says that the reviewers did not raise the issues, nor that neither reviewer checked the computations; "fewer
prefixes" withdrawn without announcing "more"; the Makasar sentence held; the reply address flagged); the E230 README;
domain statements flagged ⚠ in the outline (the Philippine placement of three of Blust's eleven groups; the numeral
compounds, with Makasar 'one hundred' set apart; *bokoti* / *bukoti* as similar, not identical; "42 lists of 13
languages"; hyphens; final *k*); the explanations of §2.4–2.6 of the outline reduced to what has to be explained (G16);
a column description for the release files (`experiments/E228_p8_revision_analyses/release/README.md`), which says that a
distance of 0.0 does not mean identical forms.

### 10.8 What could be done through a browser — 2026-10-06, afternoon

At the PI's word ("do whatever can be done with Playwright; I log in where needed"). The Playwright MCP server did not
connect in this session; public pages were fetched directly, two Sonnet agents did the retrieval, and the orchestrator
checked the key pages itself.

| Task | Result | Where |
|---|---|---|
| Source of the Makasar 38 % | found and read: Blust (1981a) via Bellwood 1997: 115; not an independent figure; ordinary for its group | §10.7; `REFERENCE_CHECK` Part D |
| The reviewer's eight references | six fetched, exact pages located (Mills's claim is on pp. 491–492, not 341–342; Bulbeck 1992 pp. 512–513 are in Appendix A); **Sneddon 1993 and Blust 2012 have no open copy** | `REFERENCE_CHECK` Part D1; copies in the git-ignored cache |
| A verbal scale for AUC (R2) | Hosmer & Lemeshow located by page through secondary sources (book page not seen); two open-access statements of a scale; one open-access paper saying such labels are arbitrary | Part D3; outline §4 (2.5) |
| A reference for "resists reconstruction" (R2) | no source states the sentence as written; three candidates for "no cognate found" (Reid 1994 the closest) | Part D4; outline §3 item 1 |
| Journal template and instructions | template downloaded (34 "OL …" paragraph styles; page 432 × 648 pt); **use of the template is optional**; the instructions are **silent on revised manuscripts** (tracked changes, where the response letter goes, whether the revision stays anonymous) → three practical questions for the editor; no statement on fees found, again | `VENUE.md` (additions of 2026-10-06); editor note, item A10 |
| Figure 1 against the press's image guidelines | was DejaVu Sans at 6.5–8 pt, 600 dpi, with an alpha channel → **redrawn** (E229 amendment A9): Arial, 1,000 dpi greyscale TIFF with LZW, 312 pt wide; tick labels and legend 8 pt instead of the recommended 9 pt (stated deviation); values unchanged, 75/75 anchors | `VENUE.md`; `experiments/E229_p8_revision_tables/` |
| arXiv paper password (ledger C061) | arXiv's help pages describe no self-service reset; the password lets anyone claim ownership at once, and owners can replace or withdraw; the route is arXiv's user-support portal or help@arxiv.org | below |
| Gmail (read-only) | no mail from the journal since the decision letter of 5 October | — |

**Needs the PI's login (asked for on 2026-10-06):** arXiv (user page: is there a way to change the paper password; else a
support request); the journal's portal (what the "revise" page asks for: files, anonymity, deadline); JSTOR for Sneddon
1993; Project MUSE for Blust 2012 and for two recent articles of the journal as models (Barlow 2025; Schapper & Kamholz
2026). Zenodo waits until the release description is completed by the PI.

**Login round (same afternoon; the PI logged in to arXiv, the journal's portal and JSTOR; no Project MUSE access).**
Playwright could not attach to the logged-in Chrome (it hangs on this build's browser-UI targets), so each tab was read
through its own DevTools socket; nothing was typed, submitted or uploaded.

| Where | What was read | Consequence |
|---|---|---|
| arXiv, user page | The actions for an owned article are Replace, Withdraw, Cross list, Journal ref, Link code & data and **Get Paper Password** (which e-mails the existing password). There is **no way to change a paper password** from the account | A request to arXiv support is the only route. **Sent 2026-10-06, 14:14** from the PI's university address to help@arxiv.org at the PI's word (text shown to him first): reset the paper password of 2604.00023 and say which accounts are registered as its owners. Waiting for arXiv's reply |
| Journal portal, manuscript page | Decision recorded as "Accept for Publication, Pending Revision / 2026-10-05"; current revision 0; **no due date shown**. The record carries the submitted abstract (438 "candidate substrate forms", AUC 0.763, "fewer canonical prefixes", 266 "high-confidence" forms, the geographic sentence), the running title, seven keywords and three subject areas | At resubmission the **metadata** have to be rewritten too (abstract, possibly title and running title, keywords), not only the files |
| Journal portal, "Revise" page | A four-step upload (Files → Manuscript Information → Validate → Submit); entered information is saved at each step; file types offered: Author Cover Letter, Article File, Figure, Table, Supplemental Material; figures as separate files; a PDF article file cannot be used by the publisher if the paper is accepted. **"Continue" was not pressed** (it opens a revision draft) | The response letter has no file type of its own → ask the editor whether it goes in as "Author Cover Letter" or "Supplemental Material" |
| Journal portal, author instructions | The same text as the journal's web page; the word "revision" does not occur; nothing on fees. Contact about a submission in progress is through **"Send Manuscript Correspondence"** on the manuscript page (to the Editorial Assistant; the form has a CAPTCHA) | The note to the editor can go through that form (the PI sends it himself) or as a reply to the decision e-mail |
| JSTOR | Sneddon 1993, p. 2: nine microgroups plus Banggai as a single-member microgroup | `REFERENCE_CHECK` Part D1; outline §3 item 2 |
| Project MUSE | not accessible; Blust 2012 unread. Barlow 2025 is listed as open access on MUSE (article 960927) but sits behind a human-verification page | Blust 2012 is not cited; the PI can open Barlow 2025 himself |

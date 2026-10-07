# Response to the reviewers — SCAFFOLD (facts per point; the sentences are the authors' own, gate G16)

Manuscript OL-03-2026-11 · scaffold generated 2026-10-07 from `REVISION_WORKPLAN.md` §3–§6, §10.7–§10.9 · numbering follows the reviewers' own comments (R1: thirteen comments plus an opening remark; R2: 24 highlights in page order). Items marked HOLD wait for the PI's reading. Every number below is traceable to E227–E231; nothing is to be added that is not in those files.

## 0. To the editors

- [PI writes the reply here]
- Facts: thanks; both reports followed; the editors' readability request (Section 3 above all) answered by rewriting Sections 2–3 for non-technical readers, defining every term at first use, replacing Figure 1, and removing Model A, the consensus analysis, the clustering, the ablation table and the 16-language application.
- Facts: a section 'Corrections by the authors' follows the replies (Part C); if the note of D7 was sent, refer to it and to the editor's answer.

## A. Reviewer 1

### R1, opening remark
*Point (paraphrased):* The SHAP beeswarm plot for Model B cannot be followed by a linguist.

Reply must contain:
- Figure 1 is now a plain bar chart of the 25-input model (mean absolute contribution per input), with the inputs named by what the code computes and the direction convention in the legend.
- Software names, raw variable names and the sign error of the old caption are gone (E227 E04).
- The figure describes the classifier; statements about the vocabulary rest on the profile table (E229 T6).

Change in the manuscript: Figure 1 replaced (E229 F1, 312 pt wide TIFF, greyscale).
Evidence: E229 F1; E227 E03-E04

[PI writes the reply here]

### R1-1
*Point (paraphrased):* 'Resist reconstruction to any proto-form' is too strong: forms may reconstruct to a lower-level proto-language.

Reply must contain:
- Accepted: 'higher-level proto-forms' at the two places the reviewer marks (tex 38, 56); tex 60 and 540 use the phrase in another sense (PI decides).
- Added where residuals are introduced: 70.9 % of Tolaki's uncoded forms have a same-meaning look-alike in Proto-Bungku-Tolaki or in one of 42 other Bungku-Tolaki lists (chance 10.5 %; strict threshold 63.4 % vs 3.0 %; coded forms 95.9 %).
- That is a mechanical screen across 42 lists of 13 languages (41 lists by one compiler), not etymologies, and it says nothing about origin: the reviewer's own example is a word borrowed by speakers of a low-level proto-language.

Change in the manuscript: Wording at tex 38, 56; new sentence(s) in Section 3 (Tolaki).
Evidence: E228 S3; E231 D

[PI writes the reply here]

### R1-2
*Point (paraphrased):* Makasar: the literature (Mills 1975; Sirk 1989; Bulbeck 1992; Bulbeck et al. 2000) reports 38 % retention / 62 % 'open to investigation'; how does our Table 1 figure fit, and what does the study add?

Reply must contain:
- HOLD until the PI has read the source pages (REFERENCE_CHECK Part D).
- Slip in the report: it quotes 30.1 %; Table 1 printed 30.9 % (mention gently).
- From ABVD, per meaning (201 meanings with a PMP entry): in the PMP entry's cognate set 39.3 % [32.8, 46.2]; coded in another set 25.9 % [20.3, 32.3]; no cognate set 34.8 % [28.6, 41.6] (Wilson 95 %). Bugis 48.3 / 28.4 / 23.4; Sa'dan Toraja 48.0 / 33.0 / 19.0.
- 'In the PMP entry's set' is set membership in ABVD, a lower bound on retention (39.3-74.1 %), not retention; with doubtful assignments counted 42.8 %. Do not write 'not retained 60.7 %'.
- One fact, not three: Makasar has 11-16 points more uncoded meanings than its two relatives (exact McNemar on shared meanings: p = 0.006 vs Bugis, p = 0.0001 vs Sa'dan Toraja, unadjusted). On meanings coded in both lists the shares in the PMP set are equal (70 vs 69 of 110; 70 vs 71 of 115) - these conditional figures decide nothing.
- What the uncoded forms are: a mechanical look-alike screen finds a PMP look-alike for 2 of 75 (about 1 expected by chance) and a Bugis/Sa'dan Toraja look-alike for 10 of 80 (chance 3.8 %; coded forms 90 of 136). The screen's sensitivity, 38 % on forms ABVD coded, is an UPPER bound for missed cognates (coders and screen both use surface resemblance; C066). So: the screen does not favour 'merely uncoded'; the data are compatible with the divergence reported in the literature, do not establish it, and say nothing about its cause.
- The published 38 % is Blust's (1981a) lexicostatistical count against a 200-word PMP list, quoted by Bellwood 1997: 115 beside a WMP mean of 41 % (Sundanese 35, Javanese 30); ABVD's PMP list is Blust's too, so the agreement with 39.3 % is a count of the same kind in the same tradition, not independent confirmation, and in the source the figure is not low. The step to '62 % open to investigation' is Bulbeck et al. 2000: 103.
- A more independent comparator: Sirk 1989, Table 1 (p. 71): Makasar-Bugis 45, Makasar-Sa'dan 42, Bugis-Sa'dan 60; ABVD shared sets for the same pairs 41.4 / 39.2 / 53.1 % (same order, similar gaps). Mills's divergence claim is on pp. 491-492 (not 341-342); the paper must say which Mills 1975 (dissertation or Archipel article).
- What the study adds: the three-way split with intervals; the uncoded share as the one measurable difference; the screen result; a list of the 70 uncoded Makasar meanings with the relatives' forms, released for inspection (E231 P9).

Change in the manuscript: New subsection on Makasar in Section 3 with one small table (E228 TABLE_R1-2 + E231 C intervals); literature added after reading; the 'substrate' word is not used for these forms.
Evidence: E228 S2; E231 B, C, P1, P8, P9; REFERENCE_CHECK Part D

[PI writes the reply here]

### R1-3
*Point (paraphrased):* 'At least ten primary subgroups (Blust 2009)': the source lists eleven; Sneddon 1993 could be cited.

Reply must contain:
- Blust (2013 revised edition, section 2.4.6, p. 82) names eleven 'microgroups'; section 4.1.6 (p. 193) reports Sneddon's ten.
- Sneddon 1993: 2 (read): nine microgroups plus Banggai as a single-member microgroup; reasonable doubt only about Muna-Buton.
- Neither source calls them 'primary subgroups' - the word is 'microgroups'; the book treats three of the eleven as belonging under Philippine groups (PI checks the sentence). Edition actually read: 2013.

Change in the manuscript: tex 57: number, term, edition and the Sneddon citation.
Evidence: REFERENCE_CHECK C1; Sneddon 1993: 2

[PI writes the reply here]

### R1-4
*Point (paraphrased):* Celebic is a supergroup, not a primary subgroup.

Reply must contain:
- Accepted: South Sulawesi, Bungku-Tolaki, Muna-Buton named instead; Celebic described as the supergroup proposed by van den Berg (1996, not 1989) and Mead (2003), with their own qualifiers (after the PI has read the pages).
- Consequence noted in Section 3: three of the paper's four subgroups fall inside the proposed Celebic supergroup, so 'across four subgroups' overstates independence.

Change in the manuscript: tex 65; a sentence in the cross-list limits.
Evidence: REFERENCE_CHECK D1

[PI writes the reply here]

### R1-5
*Point (paraphrased):* 'Parallel innovation' is a technical term meaning something else.

Reply must contain:
- The term is dropped at every occurrence (tex 73, 509 heading, 517, 598, 614).
- The replacement the reviewer offers ('independent innovations') is NOT adopted as a claim either: the test behind the submitted negative result (p = 0.569) compared a mean of 20 concepts with single draws of mostly cognate vocabulary (E227 G07) and was replaced by a permutation test; among uncoded forms, same-meaning forms are slightly more alike than different-meaning ones (mean distance 0.836 vs 0.869; none of 10,000 permutations; 4 look-alike pairs vs 0.5 expected; coded forms 472 vs 8.5), the coded class is not examined, and all four pairs involve Wolio.
- The revised paper therefore makes no cross-language claim, positive or negative (decision D4 (a)).

Change in the manuscript: Subsection title and sentences removed; the cross-list result reported as one statement with its limits.
Evidence: E227 G07; E228 S5; E231 H, P7

[PI writes the reply here]

### R1-6
*Point (paraphrased):* 'Three subgroups' but four are listed; the correct grouping; spelling of Makasar / Sa'dan Toraja; the abbreviation 'Bol. Mongondow'.

Reply must contain:
- Four subgroups as the reviewer gives them: Muna-Buton (Muna), Wotu-Wolio (Wolio), South Sulawesi (Bugis, Makasar, Sa'dan Toraja), Bungku-Tolaki (Tolaki); Sa'dan Toraja is never Celebic.
- The list used is ABVD 226, 'Tae' (S. Toraja)', source van der Veen 1940 - said in the data section.
- The abbreviation (tex 468, 486, Figure 4) disappears with Table 5 / Figure 4 (decision D5); 'Toraja-Sadan' in Figures 2 and 4 disappears with them.
- Spelling: [PI's decision - Makasar (reviewer's preference, Glottolog) or Makassar (ABVD)]; 'Sa'dan Toraja' adopted.
- tex 534 ('the South Sulawesi mainstream represented by Bugis and Makassar') follows the corrected grouping.

Change in the manuscript: tex 86-87, 237, 298, 405; figures regenerated or removed.
Evidence: E227 A02

[PI writes the reply here]

### R1-7
*Point (paraphrased):* 'Under-documentation' is unclear.

Reply must contain:
- What was meant: 36 % of the Tolaki forms have a cognate-set assignment (75 of 209). The word is replaced by the number.
- The list is not short of primary data: 200 of 210 meanings filled; of the 86 uncoded forms whose meaning the five Tolaki dialect lists cover, 88.4 % recur there.
- The subgroup is thinly coded in ABVD: by meaning (first form per meaning, shared meanings) list 674 is coded for 38.3 %, the five Tolaki dialect lists for 36.1-40.6 %, the median of 42 comparison lists is 46.7 % (5 at or below). The unit matters: by form the share falls with list length (Spearman -0.55); 'any form coded' rewards synonym-rich lists (Konawe 50.4 %).

Change in the manuscript: tex 114, 528-534 rewritten with the numbers.
Evidence: E228 S3; E231 D, P5, P10, P10b

[PI writes the reply here]

### R1-8
*Point (paraphrased):* Muna dh is also a digraph but is missing from the digraph list.

Reply must contain:
- Two Muna forms contain dh: akaradhaa 'to work' and idho 'green'; both are cognate-coded; both WERE converted in the test - the sentence omitted dh (and Wolio gh, which occurs in no form).
- No loanword was removed from the classifier data; akaradhaa carries ABVD's loan flag.
- Count unaffected: 75 forms changed (Muna 54, Tolaki 20, Sa'dan Toraja 1); the reviewer's term (voiced interdental stop) is used; the script's one-character placeholders say nothing about Muna phonology.

Change in the manuscript: tex 348: complete list; the robustness subsection reduced to one paragraph (D9).
Evidence: E227 I01-I03; E230 D1

[PI writes the reply here]

### R1-9
*Point (paraphrased):* The glottal stop is written in many ways; does the method give the same result under every convention?

Reply must contain:
- Not quite. The method sees only a written mark (ʔ or apostrophe). Marker rate by list: Makasar 36 %, Sa'dan Toraja 22 %, Bugis 21 % (ʔ); Tolaki 14 % (apostrophe); Wolio 2 % (4 forms); Muna 0.
- Six conventions = three manipulations: mark re-coded as a consonant letter (q or k): -0.016 AUC (held-out lists -0.020 to -0.024); mark left unwritten: -0.006; pre-glottalised consonant written as a geminate: -0.007; 'apostrophe to ʔ' changes no input. AUC stays >= 0.710 under every convention (held-out minimum 0.677). Verdict by the rule written before the run: partial dependence (a threshold rule, not a test; no noise estimate).
- Within the lists that mark it, marked forms are more often uncoded: pooled OR 2.67 [1.88, 3.79]; per list Bugis 2.04, Makasar 2.01, Sa'dan Toraja 2.40; in Tolaki all 29 marked forms are uncoded; without Tolaki 2.11 (E231 P13).
- ʔ + consonant occurs in 35 forms and also counts as a 'cluster'; eleven Sa'dan Toraja forms end in k beside 45 ending in a glottal mark - that k is NOT presented as a glottal stop (ABVD's PMP entries for 'bird' and 'sea' end in k).
- The abstract sentence 'robust ... rather than orthographic artifacts' is withdrawn; whether the per-list marker rates are spelling or phonology is left open.

Change in the manuscript: tex 42, 135, 355, 382, 577; table of conventions (E228 TABLE_R1-9); counts per list.
Evidence: E228 S1; E227 D02, D06; E231 F, P7, P13

[PI writes the reply here]

### R1-10
*Point (paraphrased):* A semantic feature cannot belong to a 'phonological' fingerprint.

Reply must contain:
- Sharper than assumed: of the submitted model's 26 inputs 17 come from the written form, 8 from the meaning, 1 is the list's identity code.
- AUC by input group: form only 0.672 (held-out list 0.641), meaning only 0.646, identity only 0.680, form + meaning 0.727 (0.701). The submitted 0.763 included the identity code (strongest input).
- The headline model is now the 25-input form + meaning model with the form-only model beside it (D2); 'trained exclusively on phonological features' is withdrawn; 'fingerprint' is replaced by 'profile', defined in the Introduction as part written form, part meaning.

Change in the manuscript: Abstract, tex 71, 127, 132-141, 385, 497, 594-595; Table 2 rebuilt.
Evidence: E228 S7; E227 B03-B04; E229 T2

[PI writes the reply here]

### R1-11
*Point (paraphrased):* Give example lexemes for each quadrant.

Reply must contain:
- Table of examples for the four cells (label x out-of-fold profile score; 172 / 266 / 105 / 814 forms), 2-3 per cell, printed as ABVD records them (form, gloss, set number, loan flag, nearest look-alike) with the sentence that they were not assessed etymologically.
- The examples are the forms the classifier scores most extremely, so they show its inputs by construction; affixation or compounding is a reading, not tested.
- Candidate pool: E229 T7 (92 rows); every printed example checked by the PI against ABVD and the ACD.

Change in the manuscript: New table in Section 3; the old CS / CA / RO / MO labels replaced by plain cell names.
Evidence: E228 S4; E229 T5, T7

[PI writes the reply here]

### R1-12
*Point (paraphrased):* 'Unlabeled positive' / 'false positive' are used in two senses; is a form tagged negative that should be positive not a false negative?

Reply must contain:
- Yes. With positive = Austronesian (tex 94), an inherited form left in the residual is a false negative; tex 107, 247 and 445 swapped the classes.
- Root of the confusion: two label sets (356 forms in Table 1, 438 everywhere else) and a 'rescue' step that was a concept filter (it compared no form with any reconstruction; E227 A13); the loan lists matched five unrelated words (A15).
- Now one definition: coded (has a cognate-set number in ABVD) / uncoded = candidate (has none); 'positive', 'false positive' and 'unlabeled positive' are no longer used; 'false alarm' only for the decimal numeral compounds.
- The reviewer's remark that adapted loans look regular is acknowledged (Blust 2012: 556, after the PI has read it): such forms are worth a closer look, not errors.

Change in the manuscript: tex 94, 107, 183, 247, 412, 440, 445, 520, 546, 581.
Evidence: E227 A10-A15

[PI writes the reply here]

### R1-13
*Point (paraphrased):* What does 'documentation gaps' mean for Tolaki? Release the 114 Tolaki residual items (ideally every list).

Reply must contain:
- (a) Not the first reading (the list is not short of words). The second (cognates exist but are unassigned) is visible for part: of 34 uncoded forms matching a Proto-Bungku-Tolaki entry, 7 match an entry ABVD does assign to a set (plain gaps); 12 of 128 are look-alikes of the PMP form for the same meaning by the screen (about 2 by chance; one loan-flagged; sensitivity of the screen 48 % on coded forms, an upper bound). For the rest the study cannot decide.
- (b) Released: p8_candidates.csv (438 rows; Tolaki 134, not 114 - one label definition, see R1-12) and p8_forms_all.csv (1,357 rows) with ABVD ids, cognate numbers, loan flags, out-of-fold scores, nearest look-alikes; column description included; ABVD is CC BY 4.0. Zenodo DOI: [insert before upload].

Change in the manuscript: tex 534; data-availability statement (tex 619).
Evidence: E228 S3; E228 release/

[PI writes the reply here]

## B. Reviewer 2 (24 highlights, page numbers of the reviewer's PDF)

### R2-1 (p. 2)
*Point (paraphrased):* A reference for the claim that vocabulary 'resists reconstruction'.

Reply must contain:
- No source states the sentence as written (three properties together, for Sulawesi). Located: Reid 1994 (OL 33, section 2.1: 'unique' forms without cognates, 17-29 % in three Philippine Negrito languages), Blust 2013: 8, Bulbeck et al. 2000: 103 (Makasar) - all for 'no cognate found' only. [PI reads before citing; or the sentence is reduced to ABVD coding status].

Change in the manuscript: tex 56.
Evidence: REFERENCE_CHECK D4

[PI writes the reply here]

### R2-2 (p. 5)
*Point (paraphrased):* What are E027 / E028 / E022?

Reply must contain:
- Internal experiment codes; 29 occurrences on 23 lines and in the titles of Figures 2 and 4. All replaced by descriptive names (rule-based label, classifier, ...).

Change in the manuscript: Whole text; Figures 2 and 4 removed.
Evidence: E227 A10

[PI writes the reply here]

### R2-3 (p. 6)
*Point (paraphrased):* Source of the semantic domains - Concepticon? WOLD?

Reply must contain:
- Neither: the seven domains and the 'core' flag are the authors' own concept sets in the feature script; the 'core' flag covers 174 of 210 meanings (83 % of forms), so it is not 'Swadesh-100'. Said as such; mapping to Concepticon named as future work, not claimed. ABVD's 210-item list is not a Swadesh list in the strict sense (PI checks ABVD's own wording).

Change in the manuscript: tex 89-90, 139.
Evidence: E227 E06

[PI writes the reply here]

### R2-4 (p. 6)
*Point (paraphrased):* What does the 'language identity' feature mean?

Reply must contain:
- An integer 0-5 in alphabetical order of the list names; the strongest input of the submitted headline model (alone AUC 0.680). Removed from the headline model (D2) and said so.

Change in the manuscript: Methods.
Evidence: E228 S7

[PI writes the reply here]

### R2-5 (p. 7)
*Point (paraphrased):* What is the classifier and what does it show about 'robustness'?

Reply must contain:
- Plain-words definition at first use [PI]; three classifiers (XGBoost 0.727, random forest 0.727, logistic regression 0.700 with form + meaning; 0.672 / 0.683 / 0.671 form only) give the same picture - that is the only sense of 'robust' kept.

Change in the manuscript: Methods; Table 2.
Evidence: E229 T2

[PI writes the reply here]

### R2-6 (p. 7)
*Point (paraphrased):* 'Implemented in scikit-learn / XGBoost' - Python?

Reply must contain:
- Python 3.11; scikit-learn 1.8.0, xgboost 3.0.3, shap 0.50.0 (the stored results reproduce with these versions).

Change in the manuscript: One clause in Methods.
Evidence: E227 summary

[PI writes the reply here]

### R2-7 (p. 8)
*Point (paraphrased):* What do the false positives stand for?

Reply must contain:
- Same as R1-12: the term is no longer used; one definition per term.

Change in the manuscript: See R1-12.
Evidence: -

[PI writes the reply here]

### R2-8 (p. 8)
*Point (paraphrased):* 'E022 binary label' cannot be found in the rule-based section.

Reply must contain:
- Because the label used by the models (438 forms) is not what that section described (356 forms); now one label, defined once: no cognate-set number in ABVD.

Change in the manuscript: Methods 2.2; Table 1.
Evidence: E227 A10

[PI writes the reply here]

### R2-9 (p. 8)
*Point (paraphrased):* Kappa = agreement?

Reply must contain:
- Defined at first use if kept [PI]; the submitted 0.61 was in-sample (the classifier's fit to its own training label); out of fold the label x profile agreement is kappa 0.31 (172 / 266 / 105 / 814). The 'two-method consensus' framing is dropped (D3).

Change in the manuscript: Section 3 (cells table E229 T5).
Evidence: E227 F03; E229 T5

[PI writes the reply here]

### R2-10 (p. 9)
*Point (paraphrased):* Why k = 5 ... 30?

Reply must contain:
- No reason existed: 30 was the edge of the search (silhouette keeps rising: 0.126 at 40, 0.205 at 100; value shifts with row order). The clustering paragraph is removed (D4); the permutation test replaces it.

Change in the manuscript: Removed.
Evidence: E227 G01-G03

[PI writes the reply here]

### R2-11 (p. 10)
*Point (paraphrased):* 'Both models' - which?

Reply must contain:
- Model A (31 inputs, four of which encode the label, AUC 0.9997-1.000) is removed entirely; one model family remains (25 inputs; form-only variant beside it).

Change in the manuscript: Removed.
Evidence: E227 C05-C09

[PI writes the reply here]

### R2-12 (p. 11)
*Point (paraphrased):* 'Near-perfect' = what number; how to read Table 2.

Reply must contain:
- 0.9997 (Model A, because four inputs encode the label) - gone with Model A. Table 2 rebuilt: AUC, accuracy beside the 'always coded' baseline (0.719 vs 0.677), and the candidate-class precision / recall / F1 (0.597 / 0.404 / 0.480; the submitted F1 0.82 was the coded class). Walk-through sentence for a non-technical reader [PI].

Change in the manuscript: Table 2 (E229 T2).
Evidence: E229 T2

[PI writes the reply here]

### R2-13 (p. 11)
*Point (paraphrased):* AUC - what does it stand for and mean?

Reply must contain:
- Area under the ROC curve: the chance that a randomly chosen uncoded form is ranked above a randomly chosen coded one; 0.5 = coin, 1 = perfect. Defined at first use [PI].

Change in the manuscript: Methods 2.5.
Evidence: E227 C01

[PI writes the reply here]

### R2-14 (p. 11)
*Point (paraphrased):* What range of AUC counts as 'moderate' / 'reliable'?

Reply must contain:
- Verbal scales exist and differ (Hosmer & Lemeshow 2013: 177 as quoted by others: 0.7-0.8 'acceptable'; Nahm 2022 Table 4: 0.7-0.8 'fair', 0.6-0.7 'poor'); White et al. 2023 (BMC Medicine 21: 339, p. 2): such labels have no scientific basis. Honest answer: name a scale, place 0.73 and 0.67 on it, give the interval (about +/- 0.03), and say the labels are arbitrary. The old '0.65 threshold' was the project's own go/no-go line - dropped or named as such. [PI reads White et al. and one scale source before citing.]

Change in the manuscript: Methods 2.5.
Evidence: E227 C18; E231 G; REFERENCE_CHECK D3

[PI writes the reply here]

### R2-15 (p. 12)
*Point (paraphrased):* What does 'generalises across Sulawesi languages' mean concretely?

Reply must contain:
- A model fitted on five lists ranks the forms of the sixth: AUC 0.614-0.809, mean 0.701 (form only 0.641); 5 of 6 lists >= 0.65 (form only 3 of 6). As a yes/no classifier it is below the majority answer in 5 of 6 held-out lists - only the ranking carries over. Said in those words.

Change in the manuscript: Section 3 (E229 T3).
Evidence: E227 C10-C12; E229 T3

[PI writes the reply here]

### R2-16 (p. 12)
*Point (paraphrased):* 'Feature ablation'; why is language_cognacy_coverage in a different font?

Reply must contain:
- Ablation in plain words: the classifier is run again without a group of inputs and the AUCs compared - that is now the form + meaning / form only / meaning only / identity only comparison. The variable in typewriter font no longer exists: its values in the code were stale hand-written constants, not the coverage (E227 B02). No variable names are printed.

Change in the manuscript: Methods 2.3.
Evidence: E227 B02; E228 S7

[PI writes the reply here]

### R2-17 (p. 12)
*Point (paraphrased):* SHAP?

Reply must contain:
- Plain-words explanation of what the bars of Figure 1 show (average size of each input's contribution to the prediction) [PI]; the figure describes the classifier, not the vocabulary.

Change in the manuscript: Methods 2.6; Figure 1.
Evidence: E229 F1, N1

[PI writes the reply here]

### R2-18 (p. 13)
*Point (paraphrased):* CV?

Reply must contain:
- Cross-validation in plain words [PI]: five folds x ten repetitions = 50 tests; plus one list held out at a time.

Change in the manuscript: Methods 2.5.
Evidence: E229 T2, T3

[PI writes the reply here]

### R2-19 (p. 13)
*Point (paraphrased):* The symbol delta?

Reply must contain:
- Difference in AUC between two runs; written out or defined at first use. The +/- 0.007 was the SD of ten seed means; over the 50 folds it is 0.027 - both printed with their names if printed at all.

Change in the manuscript: Methods 2.5.
Evidence: E227 C01; E229 T2

[PI writes the reply here]

### R2-20 (p. 18)
*Point (paraphrased):* The panels of Figure 2 are neither referred to nor explained.

Reply must contain:
- Figure 2 (four panels with internal codes; panel D contradicted the text) is removed; the cells table (E229 T5) replaces it.

Change in the manuscript: Removed.
Evidence: E227 D01

[PI writes the reply here]

### R2-21 (p. 18)
*Point (paraphrased):* Why do 'One Hundred', 'Fifty', 'Twenty', 'to stand', 'to hit' appear as consensus substrate in >= 4 languages?

Reply must contain:
- 'Fifty' and 'Twenty' look like decimal compounds (uncoded in four lists, coded in Bugis) - the one place 'false alarm' fits (PI checks the reconstructions in the ACD); 'One Hundred' is not uniform (the Makasar form looks like a different word) and is not grouped with them; 'to hit' and 'to stand' were on the submitted version's own list of inherited meanings - the consensus was computed on the label that ignored that list. Out of fold no meaning is 'candidate with profile' in four or more lists. The paragraph's premise is gone (D1 + D3).

Change in the manuscript: Section 3.
Evidence: E228 S4; E227 F12-F13

[PI writes the reply here]

### R2-22 (p. 18)
*Point (paraphrased):* DBSCAN parameters look cryptic.

Reply must contain:
- Removed with the clustering (D4).

Change in the manuscript: Removed.
Evidence: E227 G01-G03

[PI writes the reply here]

### R2-23 (p. 20)
*Point (paraphrased):* Table 5 is cut off.

Reply must contain:
- Table 5 (and Figure 4) are removed: the 16-language rates rested on two language-level inputs (E227 H06; Acehnese 63 % -> 8 % without them); at most one short paragraph remains (mean AUC 0.638, >= 0.60 in 10 of 16; no geographic claim).

Change in the manuscript: Removed (D5).
Evidence: E227 H02-H06; E228 S6

[PI writes the reply here]

### R2-24 (p. 21)
*Point (paraphrased):* Define 'fingerprint' as a probabilistic phonological profile at the start, not in the Discussion.

Reply must contain:
- Defined in the Introduction as 'profile' (part written form, part meaning - so not 'phonological'; R1-10). Measured content: uncoded forms are about one character longer (6.10 vs 5.20), more often carry a written glottal mark (23.5 % vs 11.6 %; where the source writes one) and an action meaning (40.4 % vs 23.4 %); the latter two hold at equal length (OR 2.60, 2.22); the onset-string input (37.0 % vs 25.2 %) is not separable from length - 'fewer prefixes' withdrawn, 'more' not established.

Change in the manuscript: Introduction; Section 3 profile table (E229 T6).
Evidence: E227 D01-D05; E229 T6; E231 E, P3, P11

[PI writes the reply here]

## C. Corrections by the authors (found while re-deriving the numbers; not raised as such by the reviewers, though several of their questions lead to them)

| What | Submitted text | Corrected | Evidence |
|---|---|---|---|
| Two residual sets presented as one | 356 forms in Table 1; 438 in every model, Table 3, the agreement table and the clustering; abstract '438 (26.5 %)' | one label: 438 = 32.3 % of 1,357; Table 1 per list 34 / 62 / 80 / 83 / 45 / 134 | E227 A07, A10, A14 |
| The 'Proto-Austronesian cross-check' compared no forms | described as a cross-check of residual forms against 15 reconstructions | all uncoded forms for 15 meanings were removed; the step is deleted and said to be | E227 A13 |
| The loanword lists matched five unrelated words | Sanskrit / Arabic / Malay trade-word filter | deleted (e.g. Tolaki beli 'blood' matched Malay beli 'buy') | E227 A15 |
| 'Fewer canonical Austronesian prefixes' | abstract, tex 385, 448, 497 | the onset-string input is MORE frequent among uncoded forms (37.0 % vs 25.2 %) but not separable from length (OR 1.09 at equal letters); 'fewer' withdrawn, 'more' not established; the argument at tex 447-448 fails | E227 D01; E231 P3, P11 |
| Headline AUC 0.763 'phonological only' | 26 inputs including the list identity code | form + meaning 0.727 (held-out 0.701); form only 0.672 (0.641); interval about +/- 0.03 | E228 S7; E229 T2 |
| In-sample agreement presented as independent confirmation | kappa 0.61; 266 'high-confidence' forms | out of fold kappa 0.31; 172 / 266 / 105 / 814; the consensus framing dropped | E227 F03; E229 T5 |
| p = 0.569 did not test the stated hypothesis; 'optimal k = 30' was the search limit | clustering and cross-linguistic cognate test | permutation test: among uncoded forms a small excess of same-meaning similarity (none of 10,000 permutations; 4 vs 0.5 pairs); no statement about a shared layer; clustering removed | E227 G02, G07; E228 S5 |
| 16-language application | mean P 0.606 vs 0.393 'significantly'; Table 5 with four stale constants | language-level inputs drove it; without them mean AUC 0.638, no geographic claim (p = 0.023 pooled, 0.37 among coded forms); removed | E227 H02-H06; E228 S6 |
| Robustness section (tex 347-359) | 0.772 -> 0.774 etc.; 'does not depend on form length at all' | numbers belonged to a 26-input model with the identity code; digraph conversion: unchanged (75 forms); removing both size inputs costs 0.014 / 0.023 -> sentence withdrawn; 'not orthographic artifacts' had no test behind it | E230; C052 |
| Figure 1 caption sign; Table 2 F1; class weighting | sign reversed; F1 0.82 = coded class; 'class-weighted XGBoost' | Figure 1 regenerated; candidate-class F1 0.480 beside the 0.677 baseline; XGBoost was not class-weighted - corrected | E227 E04, C05, C07 |
| Smaller: 'Swadesh-100' flag; coverage variable; 'syllable' | 174 of 210 meanings; stale constants; vowel groups | described as what they are | E227 E06, B02, D04 |
| References | ross2005 cited with a false venue; two non-existent entries (uncited) | ross2005 = Papuan Pasts ch. 2, pp. 15-65; mead2005 and vandenBerg1996 deleted; eight further entries corrected | REFERENCE_CHECK Part A |
| Section 4.5 (Javanese script) | convergent evidence from script adaptation | removed (D6): not asked for, outside Sulawesi, rests on properties that changed, quoted 10.5 % for the wrong group (E227 F11) | E227 F11 |

[PI writes the reply here]

## D. Questions to the editors (if not already settled by the note of D7)

- Are there any author-side charges (page, colour figure or other)? The authors do not take a paid open-access option (G15).
- Should the revised manuscript remain anonymised? Is a tracked-changes version wanted beside the clean copy? Under which file type does the response letter go in the portal (Author Cover Letter / Supplemental Material)?
- If the note of D7 was sent: refer to it here; if the editor asked for another procedure (e.g. a second review round), say that the authors follow it.

## E. Before sending (checklist)

- Every number in the letter re-checked against E227–E231 (G1 on the letter).
- No verbatim reviewer text quoted beyond what is needed to identify the point; the repo copy of this file stays paraphrased.
- Zenodo DOI inserted (R1-13b) before upload.
- Decide: Makasar / Makassar spelling (R1-6); whether tex 60 and 540 change (R1-1); which Mills 1975 is cited (R1-2).


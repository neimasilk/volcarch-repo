# E229 — data files behind the revised tables and Figure 1 of P8 (*Oceanic Linguistics*)

**Status:** SUCCESS — all files produced; 75 of 75 anchors reproduce; the independent re-derivation agrees in 163 of 163 cells.
**Lines:** 04_language_text (P8 revision). **Date:** 2026-10-06.
**Design:** `DESIGN.md`, frozen before the script was written; eight dated amendments in its §8 (none changes a number; A6 and A7 add post hoc descriptive counts; A8 renames four labels).
**Follows:** E227 (audit), E228 (reviewer analyses). Nothing in E022–E228 was changed. **No manuscript text here** (gate G16).

## Hypothesis

None. This experiment builds the tables and the one figure that the revised manuscript needs once decisions D1–D3 of
`papers/P8_linguistic_fossils/REVISION_WORKPLAN.md` are applied: one label (*candidate* = no cognate-set assignment in
ABVD), the 25-input model with the form-only model beside it, out-of-fold cells in place of the in-sample "consensus".
Choices that could be tuned after seeing numbers were fixed in `DESIGN.md` first (threshold 0.5, classifier settings,
which examples are shown, how each property is summarised).

## Method and data

ABVD CLDF snapshot (lexibank/abvd, git `917c5a5`), lists 27, 48, 166, 192, 226, 674 = 1,357 forms. Loader, features and
XGBoost settings imported from `E228/p8common.py`. **FM25** = 17 inputs from the written form + 8 from the meaning;
**F17** = the 17 form inputs alone. No language-level input in any model. Stratified 5-fold × 10 seeds; leave-one-language-out.
`01_tables.py` (tables, figure, anchors) · `02_independent_check.py` (second agent, never read the first script; T1 from
the raw file, XGBoost numbers from the **stored** March 2026 feature matrix with its own loop and metric code) ·
`03_compare_with_independent.py` (cell-by-cell diff). Python 3.11.7, xgboost 3.0.3, scikit-learn 1.8.0, shap 0.50.0.

**What happened on the way (recorded, not hidden).** The first run stopped at an anchor and wrote no table: the
per-list counts in `DESIGN.md` §0/§4 had been typed in another list order than §1 — an error in the design, not in the
data (`results/run_log_first_run_stopped.txt`; amendment A1). The builder stopped as instructed instead of
re-interpreting the anchor. The anchor was keyed by list name and the script re-run. The figure was re-drawn twice
for layout only (clipped axis label, legend over bars).

## Results

### T1 — one label (`results/T1_label_by_list.csv`)

| List | Forms | Coded | % | Candidate | % |
|---|---|---|---|---|---|
| Muna | 219 | 185 | 84.5 | 34 | 15.5 |
| Bugis | 242 | 180 | 74.4 | 62 | 25.6 |
| Makasar | 217 | 137 | 63.1 | 80 | 36.9 |
| Wolio | 254 | 171 | 67.3 | 83 | 32.7 |
| Sa'dan Toraja (ABVD: Tae') | 216 | 171 | 79.2 | 45 | 20.8 |
| Tolaki | 209 | 75 | 35.9 | 134 | 64.1 |
| **All six** | **1,357** | **919** | **67.7** | **438** | **32.3** |

### T2 — cross-validation, 50 test folds (`T2_cv_performance.csv`)

"Always coded" = the accuracy of answering *coded* for every form: **0.677**.

| Inputs | Classifier | AUC (SD of 10 seed means; SD of 50 folds) | Accuracy | F1 candidate | Precision | Recall | F1 coded |
|---|---|---|---|---|---|---|---|
| FM25 | XGBoost | **0.727** (0.007; 0.027) | 0.719 | 0.480 | 0.597 | 0.404 | 0.807 |
| FM25 | Random Forest | 0.727 (0.004; 0.026) | 0.663 | 0.543 | 0.483 | 0.622 | 0.733 |
| FM25 | Logistic regression | 0.700 (0.004; 0.027) | 0.648 | 0.546 | 0.468 | 0.657 | 0.712 |
| F17 | XGBoost | **0.672** (0.008; 0.029) | 0.696 | 0.391 | 0.560 | 0.304 | 0.797 |
| F17 | Random Forest | 0.683 (0.006; 0.026) | 0.646 | 0.514 | 0.463 | 0.581 | 0.721 |
| F17 | Logistic regression | 0.671 (0.004; 0.031) | 0.629 | 0.511 | 0.445 | 0.601 | 0.701 |

Reading: the three classifiers agree on the ranking quality (0.70–0.73 with form and meaning; 0.67–0.68 from the
written form alone). As a yes/no classifier at the fixed threshold the gain is small: XGBoost is 4 points above "always
coded" and finds 40 % of the candidates; the two class-weighted classifiers find 62–66 % of them at the price of an
accuracy **below** "always coded". The F1 of 0.82 printed in the submitted Table 2 was the score of the *coded* class.

### T3 — one list held out (`T3_lolo.csv`), XGBoost

| Held-out list | n | Candidates | AUC FM25 | Accuracy FM25 | Majority answer | AUC F17 | Accuracy F17 |
|---|---|---|---|---|---|---|---|
| Muna | 219 | 34 | 0.614 | 0.580 | 0.845 | 0.573 | 0.740 |
| Bugis | 242 | 62 | 0.706 | 0.690 | 0.744 | 0.659 | 0.707 |
| Makasar | 217 | 80 | 0.711 | 0.668 | 0.631 | 0.655 | 0.664 |
| Wolio | 254 | 83 | 0.667 | 0.650 | 0.673 | 0.597 | 0.622 |
| Sa'dan Toraja | 216 | 45 | 0.697 | 0.759 | 0.792 | 0.604 | 0.736 |
| Tolaki | 209 | 134 | 0.809 | 0.498 | 0.641 | 0.760 | 0.426 |
| **Mean (SD)** | | | **0.701** (0.059) | | | **0.641** (0.061) | |

AUC ≥ 0.65 (the submitted version's own line) in 5 of 6 lists with FM25 and 3 of 6 with F17. **In five of the six
lists the accuracy at the fixed threshold is below the majority answer for that list, in both models** (Makasar is
the exception). So for a list the model has not seen it *ranks* forms better than chance but does not *classify* them
usefully; "generalises across languages" can only mean the ranking.

### T4 — without Tolaki (`T4_without_tolaki.csv`)

FM25 0.727 → 0.701 (−0.025); F17 0.672 → 0.645 (−0.026). (The −0.062 of the submitted text belonged to the 27-input model.)

### T5 — label × profile, out of fold (`T5_cells_by_list.csv`)

All six lists: candidate & profile **172**, candidate & no profile **266**, coded & profile **105**, coded & no profile
**814**; κ = **0.308**. Per list κ runs from 0.13 (Muna) to 0.34 (Tolaki). Tolaki supplies 62 of the 172.

### T6 — what distinguishes candidates from coded forms (`T6_profile.csv`)

| Property | Candidates | Coded | Lists with same sign | Within-list effect [95 % CI] |
|---|---|---|---|---|
| Length in characters (hyphens, spaces and glottal marks included) | 6.10 | 5.20 | 6 of 6 | +1.09 [0.87, 1.32] |
| Vowel letters | 2.88 | 2.54 | 6 of 6 | +0.39 [0.30, 0.49] |
| Consonant-letter clusters | 0.48 | 0.32 | 6 of 6 | +0.21 [0.14, 0.28] |
| Glottal stop written | 23.5 % | 11.6 % | 5 of 6 (Wolio: four forms; Muna: none) | OR 2.67 [1.88, 3.79] |
| Action meaning | 40.4 % | 23.4 % | 6 of 6 | OR 2.46 [1.89, 3.20] |
| Hyphen or repeated letter sequence | 11.9 % | 9.8 % | 5 of 6 | OR 1.68 [1.12, 2.51] |
| Begins with ma, me, mo, pa, ka, ta, na, po or aŋ | 37.0 % | 25.2 % | 6 of 6 | OR 1.57 [1.20, 2.04] |
| Contains ŋ or ng, or mb, nd, nj, mp, nk, nc, nt | 30.4 % | 23.7 % | 6 of 6 | OR 1.55 [1.18, 2.04] |
| Final vowel | 78.5 % | 79.2 % | 3 of 6 (three lists have only vowel-final forms) | OR 0.66 [0.47, 0.93] |
| "Core" flag (174 of 210 meanings) | 83.6 % | 83.2 % | 3 of 6 | OR 1.04 [0.76, 1.43] |

Counts: mean difference with every list weighted equally, percentile bootstrap (amendment A2). 0/1 properties:
Mantel–Haenszel odds ratio stratified by list. Row names are those of `labels.csv` after amendment A8: they say what
the code computes.

*(Paragraph corrected 2026-10-06 after an adversarial read; the earlier wording — "what looks like an affix, a compound
boundary" — was a reading, not a count.)* Nine of the ten intervals exclude "no difference", but the rows are **not
separate pieces of evidence**: length and vowel letters measure the same thing; the final-vowel row is the final glottal
mark counted a second time (E231 F: OR 0.91 [0.58, 1.44] once glottal-final forms are set aside); and "hyphen or
repeated letter sequence" has the same estimate under a strict definition but an interval that includes 1 (E231 E:
repeated block only, OR 1.61 [0.87, 2.98]) — too little to stand alone.
The intervals treat forms as independent although the meanings recur across the lists (200–210 of the 210 per list),
and ten properties are tested without correction. What stands is a correlated bundle: size, written glottal mark,
action meaning. The onset-string and nasal rows are **not separable from length**: with the number of letters held
fixed their odds ratios are 1.09 [0.80, 1.48] and 0.95 [0.70, 1.30] (E231 P3), so "fewer prefixes" is withdrawn and
"more prefixes" is not established. The glottal row counts five lists, but in Wolio only four forms carry a mark.
Stricter versions of four inputs: `experiments/E231_p8_what_coded_means/` tables E and P3.

### F1 — Figure 1 (`F1_input_importance.{png,pdf,tif,csv}`)

Mean absolute SHAP value of the 25 inputs, plain greyscale bars at the journal's page width (312 pt). Largest:
glottal marker 0.279, length 0.276, action meaning 0.269, vowel letters 0.266, vowel share 0.210, consonant-letter
clusters 0.198. The onset-string input ("begins with ma, me, …") is 16th (0.072) and points toward *candidate*.

⚠ **The arrows of Figure 1 describe the classifier, not the vocabulary.** For two inputs they disagree with T6:
*ends in a vowel* points toward candidate in the model (r = +0.89) while inside the lists vowel-final forms are less
often candidates (OR 0.66); the "core" flag points toward candidate (r = +0.45) while T6 shows no association (OR 1.04).
A statement about the languages has to rest on T6. One thing this input carries is visible in a count added afterwards
(`04_final_consonant_by_list.py` → `results/N1_final_consonant_by_list.csv`; amendment A6, post hoc): **all 285
consonant-final forms are in the three South Sulawesi lists** (Bugis 86, Makasar 106, Sa'dan Toraja 93; 150 of them end
in a glottal marker); Muna, Wolio and Tolaki have none. Pooled over the six lists the candidate share is the same for
vowel-final and consonant-final forms (32.1 % and 33.0 %), while inside each of the three lists it is higher for
consonant-final forms. So this input also tells the model which group of lists a form comes from, and a model
"without language-level inputs" is not blind to the source list. The same holds for the glottal marker (Muna never,
Wolio hardly ever writes one). The input also overlaps with the glottal input (150 of the 285 end in a glottal
marker; E231 F), and which of the two accounts for the sign of the arrow was **not** tested.
How much of the cross-validated AUC this accounts for was **not** measured; the
leave-one-list-out figures are the guard against it, and they are lower.

### T7 — example lexemes for each cell (`T7_examples_R1-11.csv`; reviewer 1, point 11)

The 92 rows selected in E228 (per cell the forms farthest from 0.5), unchanged — **73 different forms**, since a form can
be among the five most extreme overall and among the three of its own list — with ABVD's loan flag, ABVD's
Proto-Malayo-Polynesian and Proto-Austronesian entries for the same meaning, whether the form shares a cognate code
with the PMP entry (25 of the 36 different coded forms do), and the nearest look-alike elsewhere in Sulawesi.
What the rows show at a glance: *candidate with profile* — Tolaki *umi'ia* 'to cry' (look-alike *miʔia* elsewhere in
Sulawesi), Makasar *ammikkiriʔ* 'to think' (ABVD loan flag set); *candidate without profile* — Wolio *bokoti* 'rat'
(*bukoti* elsewhere), *pada* 'thatch/roof'; *coded without profile* — *ama* 'father', *ana* 'child';
*coded with profile* — Bugis *mar-eŋkaliŋa* 'to hear' (shares its code with PMP *\*deŋeʀ*), Wolio *male'i* 'red'.
⚠ Prima facie only: no form was judged by a specialist (SIG G10) — the PI checks each example against ABVD and the
Austronesian Comparative Dictionary before it is printed.

### N2 — action meanings and the citation form of verbs (`results/N2_action_meaning_by_form.csv`; amendment A7, post hoc)

Forms for action meanings (392) against all others (965): candidates 45.2 % against 27.0 %; mean length 5.95 against
5.31 characters; onset string ("begins with ma, me, …") 36.7 % against 25.9 %. By list:

| List | Action forms | Begins with ma, me, mo, …: action / other | Begins with *a*: action / other | Mean length: action / other |
|---|---|---|---|---|
| Muna | 62 | 0.0 % / 12.1 % | **91.9 %** / 7.0 % | 6.82 / 5.57 |
| Bugis | 71 | 54.9 % / 29.8 % | 0.0 % / 11.1 % | 6.58 / 5.44 |
| Makasar | 60 | 25.0 % / 16.6 % | **81.7 %** / 14.0 % | 6.95 / 5.63 |
| Wolio | 79 | 40.5 % / 35.4 % | 1.3 % / 5.1 % | 5.20 / 5.07 |
| Sa'dan Toraja | 60 | 35.0 % / 31.4 % | 1.7 % / 9.0 % | 4.68 / 5.19 |
| Tolaki | 60 | 61.7 % / 28.9 % | 0.0 % / 8.7 % | 5.57 / 4.93 |

In Muna and Makasar nearly every action form begins with *a*; in Makasar the action forms also carry 1.43
consonant-letter clusters on average against 0.50 for the others. ⚠ Prima facie this is the citation form chosen by
each source (a verbal prefix that the prefix list of the feature code does not contain, with a doubled consonant
after it in Makasar) — a point for a specialist, not a finding. What the table does show: how a verb is cited differs
from list to list, and the "action" input, the length, the cluster count and the first letter all move with it. The
excess of action meanings among candidates cannot be separated, with these inputs, from the way verbs are cited.

## Conclusion

The tables of the revision can be built from one label and two models without any number that E227/E228 did not
already support. *(The rest of this conclusion was cut on 2026-10-06 — the wording of any conclusion is the PI's. Numbers
for it: accuracy 0.719 against 0.677 for "always coded"; below the majority answer for a held-out list in 5 of 6 lists;
AUC 0.727 / 0.672; T6 with the corrections of E231 P3 and P11: at equal length the written glottal mark (2.60) and the
action meaning (2.22) hold, the onset string and the nasal input are not separable from length. ⚠ Table T1: Muna's
15.5 % depends on ABVD's second Muna list — E231 P2. What may be said: `REVISION_WORKPLAN.md` §10.7.)*
## Limits

Written forms, not phonemic transcriptions; a look-alike is not a cognate judgement; the label is "no cognate-set
assignment in ABVD" — a coding status, **not** an upper bound on anything non-Austronesian (E231 A: loans and local
sets are coded too). The independent check covers T1 and the XGBoost
rows of T2 and T3; the Random Forest and logistic-regression rows, T4–T7 and Figure 1 rest on the anchors against
E227/E228. LaTeX fragments in `results/tables_tex/` carry neutral headers; the journal requires a **Word** file for
the final version (see `papers/P8_linguistic_fossils/VENUE.md`), so they are a convenience for the working copy only.

## Files

`DESIGN.md` · `labels.csv` (all wording of labels and of the figure — the PI changes wording here, not in code) ·
`01_tables.py` · `02_independent_check.py` · `03_compare_with_independent.py` ·
`results/`: `T1`–`T7` CSV, `F1_input_importance.{csv,png,pdf,tif}`, `tables_tex/`, `anchor_checks.json`,
`independent_check.json`, `independent_comparison.csv`, `N1_final_consonant_by_list.csv`, `run_log.txt`,
`N2_action_meaning_by_form.csv`, `run_log_first_run_stopped.txt`. · `04_final_consonant_by_list.py` (note N1) ·
`05_action_meaning_by_form.py` (note N2).

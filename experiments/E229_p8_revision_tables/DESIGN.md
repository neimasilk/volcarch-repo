# E229 — DESIGN — data files behind the revised tables and Figure 1 of P8 (*Oceanic Linguistics*)

**Frozen:** 2026-10-06, before the script was written or run. Amendments, if any, go in §8 with a date.
**Serves:** line 04 (P8 revision; `papers/P8_linguistic_fossils/REVISION_WORKPLAN.md` §7 step 4).
**Follows:** E227 (audit of the submitted numbers) and E228 (pre-registered reviewer analyses).

## 0. What this is

No hypothesis is tested. E229 produces the **data files** for the tables and the one figure the revised manuscript
needs once decisions D1–D3 of the work plan are applied (one label; the 25-input model with the form-only model beside
it; out-of-fold cells instead of the in-sample "consensus"). Nothing here can confirm or refute a claim, but several
choices could be tuned after seeing numbers (decision threshold, classifier settings, which examples are shown, how a
property is summarised). **They are fixed below, before any number is produced.** Manuscript text is not written here
(gate G16); captions and column headers in the LaTeX fragments are neutral placeholders for the PI.

Already known when this was frozen (from E227/E228, not outcomes of E229): 1,357 forms, 438 candidates (per list
34, 62, 45, 83, 80, 134); XGBoost CV AUC 0.7265 (25 inputs) and 0.6717 (17 form inputs), leave-one-language-out means
0.7008 and 0.6413; out-of-fold cells 172 / 266 / 105 / 814, κ 0.308; the SHAP ranking of the 25-input model
(`E227/results/shap_pure25.csv`); the profile differences of `E227/results/direction_by_label.csv`.
**Not known:** accuracy, F1 and baselines for these models; Random Forest and logistic regression on these input sets;
the spread across folds; the sensitivity without Tolaki for the form-only model; intervals for the continuous properties.

## 1. Data, label, inputs, classifiers (fixed)

- Data: ABVD CLDF snapshot `experiments/E022_linguistic_subtraction/data/abvd/cldf/` (lexibank/abvd, git 917c5a5).
  Six lists: Muna 27, Bugis 48, Makassar 166, Wolio 192, Tae' 226, Tolaki 674. Loader and feature code are **imported**
  from `experiments/E228_p8_revision_analyses/p8common.py` (not copied, not edited).
- Label (D1): *candidate* = empty ABVD `Cognacy` field; *coded* = any cognate-set assignment. `y = 1` for coded.
- Input sets (D2): **FM25** = `PURE25` (17 from the written form + 8 from the meaning; primary) and **F17** =
  `PHON + INIT` (written form only; reported beside it). No language-level input in any model.
- Classifiers, settings as actually run for the submitted version (E027 `01_train_and_evaluate.py`; E227 rows C05–C09):
  XGBoost = `p8common.xgb()` (300 trees, depth 4, learning rate 0.05, **no** class weighting);
  Random Forest = 500 trees, `min_samples_leaf=5`, `class_weight="balanced"`, `random_state=42`;
  logistic regression = L2, `C=1.0`, `class_weight="balanced"`, `max_iter=1000`, lbfgs, inputs standardised with a
  scaler fitted on the training part only.
- Protocols: stratified 5-fold × 10 seeds, `random_state = 7·seed + 13` (50 test folds); leave-one-language-out (LOLO).
- **Decision threshold 0.5 on the predicted probability, fixed. No threshold is tuned anywhere.**

## 2. Outputs (all in `results/`; CSV, UTF-8)

| File | Content | Fixed rules |
|---|---|---|
| `T1_label_by_list.csv` | per list and total: forms, coded n and %, candidate n and % | percentages of forms, one decimal |
| `T2_cv_performance.csv` | 2 input sets × 3 classifiers: AUC, accuracy, F1 / precision / recall of the **candidate** class, F1 of the coded class; the accuracy of always answering "coded" | each metric per test fold → mean of the 50 folds; **two** spreads: SD of the 10 seed means (what the submitted text printed) and SD of the 50 folds |
| `T3_lolo.csv` | per held-out list, XGBoost, FM25 and F17: n, n candidates, AUC, accuracy, accuracy of the majority answer for that list, accuracy of always "coded", F1 of the candidate class; mean and SD of AUC over the six lists; number of lists with AUC ≥ 0.65 | 0.65 is the submitted manuscript's own line, kept only so the two versions can be compared |
| `T4_without_tolaki.csv` | CV AUC on all six lists and on the five lists without Tolaki, FM25 and F17, XGBoost; difference | same CV protocol on the reduced set |
| `T5_cells_by_list.csv` | label × profile cells per list and total; κ | profile = out-of-fold P(candidate) ≥ 0.5, where the out-of-fold probability is the mean over the 10 seeds (as E228 S4); FM25, XGBoost |
| `T6_profile.csv` | ten properties (form length, vowel letters, consonant-letter clusters, glottal marker, prefix-like onset, nasal sequence, reduplication, final vowel, action meaning, "core" flag): mean or share among candidates and among coded forms; difference per list; number of lists where the difference has the sign of the pooled difference; for 0/1 properties the Mantel–Haenszel odds ratio stratified by list with 95 % CI; for the three counts the list-stratified mean difference with a percentile bootstrap 95 % CI (2,000 resamples of forms within list × label, seed 229) | all ten are reported, whichever way they fall |
| `T7_examples_R1-11.csv` | the 92 rows of `E228/results/S4_cell_examples.csv` **unchanged** (selection made in E228: per cell the forms farthest from 0.5, five over all lists and three per list), plus: ABVD loan flag; the ABVD Proto-Malayo-Polynesian (list 269) and Proto-Austronesian (list 280) form(s) and cognate code(s) for the same meaning; whether the form shares a cognate code with the PMP entry; nearest look-alike columns from `E228/release/p8_forms_all.csv` | no row is added, dropped or re-ordered by judgement |
| `F1_input_importance.csv`, `.png` (300 dpi), `.pdf` | mean absolute SHAP value of each of the 25 inputs (XGBoost fitted on all 1,357 forms, TreeExplainer, as E227) and the direction of each input | direction = sign of the correlation between the input's value and its SHAP contribution **toward "candidate"**; shown as ↑ / ↓, or · when \|r\| < 0.3. Plain horizontal bars, greyscale, sorted by size, two fills (written form / meaning), no title inside the figure. Labels come from `labels.csv` (below) so the PI can change wording without touching code |
| `tables_tex/*.tex` | booktabs `tabular` fragments for T1–T6 | numbers copied by code from the CSVs, never typed |
| `anchor_checks.json` | comparison with the values known in advance (§0) | see §4 |

`labels.csv` (in the experiment folder; `key,label`), initial content fixed here:
list names — `Muna`→Muna, `Bugis`→Bugis, `Makassar`→Makasar, `Wolio`→Wolio, `Toraja-Sadan`→Sa'dan Toraja, `Tolaki`→Tolaki
(spellings as reviewer 1 asks; *Makasar* vs *Makassar* is the PI's decision — one line to change);
inputs — `form_length` Length (characters) · `n_vowels` Number of vowel letters · `vowel_ratio` Share of vowel letters ·
`ends_in_vowel` Ends in a vowel · `has_glottal` Glottal stop written (ʔ or ') · `has_nasal_cluster` Nasal sequence (mb, nd, ng, …) ·
`has_reduplication` Repeated sequence or hyphen · `n_consonant_clusters` Consonant-letter clusters ·
`has_prefix_like` Begins like a prefix (ma-, pa-, …) · `init_a` … `init_t` Begins with a / b / k / m / p / s / t ·
`init_other` Begins with another letter · `is_core_vocab` Meaning in the "core" set (174 of 210) ·
`sem_ACTION` Meaning: action · `sem_BODY` Meaning: body · `sem_GRAMMAR` Meaning: grammatical word ·
`sem_NATURE` Meaning: nature · `sem_NUMBER` Meaning: numeral · `sem_QUALITY` Meaning: quality · `sem_OTHER` Meaning: other.

## 3. Units and direction (G1-bis)

Unit = form everywhere except T7's PMP/PAn columns (meaning). "Candidate" is the class of interest in every reported
F1 / precision / recall; AUC does not depend on which class is called positive. In T6 a positive difference always
means "higher among candidates". In F1 "↑" always means "a larger value pushes toward candidate".

## 4. Anchors (the script must check them and write the comparison to `anchor_checks.json`)

n = 1,357; candidates 438; per list 34 / 62 / 45 / 83 / 80 / 134 · XGBoost CV AUC: FM25 0.7265 (SD of seed means
0.0066), F17 0.6717 (0.0082) · LOLO mean: FM25 0.7008, F17 0.6413; per list as in `E228/results/S7_feature_groups.json` ·
cells 172 / 266 / 105 / 814, κ 0.308 (`E228/results/S4_out_of_fold_agreement.json`) · SHAP: same order of the 25 inputs
as `E227/results/shap_pure25.csv`, values within 0.002 · T6 pooled means and Mantel–Haenszel values as
`E227/results/direction_by_label.csv` (within 0.001).
Tolerance for AUC anchors: 0.0005. **If an anchor fails, the script stops and the failure is reported as it is; nothing
is adjusted to make it pass.**

## 5. Independent check (second script, written without reading the first)

`02_independent_check.py` recomputes T1 from the raw `forms.csv` with its own loader, and the XGBoost rows of T2 and
T3 from the **stored** feature matrix of March 2026 (`experiments/E027_ml_substrate_detection/data/features_matrix.csv`,
rows aligned by form id), with its own cross-validation loop and its own metric code. It reports every difference to
`results/independent_check.json`. A difference above 0.0005 in a mean is a finding to be explained, not smoothed.

## 6. Status rule

SUCCESS = all files produced, all anchors reproduce, the independent check agrees. FAILED = an anchor or the
independent check does not agree and the cause is not a documented, harmless one. Either way the README says which.

## 7. Not done here

No manuscript text, no response-letter text, no change to E022–E228 or to the submitted files. Figure 2–4 of the
submitted version are not regenerated: with D3–D5 they are replaced by T5, by the permutation result of E228 S5 and
by nothing, respectively — whether any of them returns is the PI's decision.

## 8. Amendments

All dated 2026-10-06. None changes a number, a model, a threshold or which rows are shown.

- **A1 — the per-list anchor was written in the wrong order (my error in §0 and §4).** The six counts
  "34 / 62 / 45 / 83 / 80 / 134" follow the order of the work plan (Muna, Bugis, Tae', Wolio, Makasar, Tolaki), while §1
  lists the lists as Muna, Bugis, Makassar, Wolio, Tae', Tolaki. The anchor, keyed by list: **Muna 34 · Bugis 62 ·
  Makassar (166) 80 · Wolio 83 · Tae' (226) 45 · Tolaki 134** — as in `E228/release/p8_forms_all.csv` and work plan §6.
  The first run computed exactly these counts, compared them position by position, failed, and **stopped without
  writing any table**, as §4 requires (first-run log kept as `results/run_log_first_run_stopped.txt`). The independent
  check found the same counts from the raw file. The anchor in the script was then keyed by list name and the script re-run.
- **A2 — weighting in T6.** §2 did not say how the list-stratified mean difference weights the six lists. The script
  gives every list the same weight; this was chosen while the script was written, not after comparing alternatives.
  The pooled difference and the six per-list differences are in the same file, so the reader can see any other weighting.
- **A3 — figure layout.** The journal's template (read 2026-10-06, `OL-template-1.dotx` at uhpress.hawaii.edu) limits a
  figure to 26 picas (312 pt) and asks for `.jpg` or `.tiff`. Figure 1 is therefore drawn at 4.33 in width and also
  saved as `.tif` (600 dpi); axis and legend wording moved into `labels.csv`. The plotted values are unchanged.
- **A4 — wording of §2.** Reviewer 1 mentions the spelling *Makasar* as the preference of some linguists and does not
  request it (the report itself uses both spellings); *Sa'dan Toraja* is recommended for English. The default label
  stays *Makasar*; the choice is the PI's.
- **A5 — SD convention.** "SD" is the population SD (ddof = 0), the convention that reproduces the ± 0.007 printed in
  the submitted version. With ddof = 1 the SD of the ten seed means is 0.0070 (FM25) and 0.0086 (F17).

- **A6 — a descriptive count added after the tables were seen (POST HOC).** Figure 1 and T6 disagree on the direction
  of "ends in a vowel". `04_final_consonant_by_list.py` counts vowel-final and consonant-final forms per list
  (`results/N1_final_consonant_by_list.csv`). It tests nothing and changes no table; it is there so that the
  disagreement is explained by a count and not by a guess.
- **A7 — a second descriptive table added after the tables were seen (POST HOC).** `05_action_meaning_by_form.py`
  tabulates, per list, candidate share, mean length and prefix-like onsets for action meanings against all other
  meanings (`results/N2_action_meaning_by_form.csv`). It tests nothing; it is there because the submitted discussion
  interprets the action-meaning excess, and T6/T7 point to the citation form of verbs instead.
- **A8 — four labels renamed after an adversarial read (wording only; no number changes).** The names fixed in §2
  described more than the code computes: "nasal sequence" also counts a bare ŋ and the digraph *ng*; "reduplication"
  is set by any hyphen; "prefix-like" is any form beginning with one of nine letter strings (also *mata*, *tau*);
  "length" counts hyphens, spaces and glottal marks. The labels now say what is computed. Stricter versions of the
  four inputs are tabulated in E231.

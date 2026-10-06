# E227 — P8 G1 / G1-bis audit: every number of the reviewed manuscript re-derived from raw ABVD

> **Corrections of 2026-10-06 (after two adversarial reads; the audit rows themselves are unchanged).**
> (1) "Every number" in the title and the hypothesis means the **85 statements** of `results/claims_audit.csv`
> (47 match, 13 with a note, 11 not as described, 9 mismatch, 3 unsupported, 2 wrong direction); constants such as word
> lists were read from the original scripts, not re-derived.
> (2) Item 5 below ("prefix-like onsets … the data point the other way"): the input counts forms that begin with one of
> nine letter strings; it is more frequent among uncoded forms (37.0 % vs 25.2 %), but with the number of letters held
> fixed it does not separate them (odds ratio 1.09 [0.80, 1.48]; `experiments/E231_p8_what_coded_means/`, table P3). So
> "fewer prefixes" fails, and "the data point the other way" is **withdrawn**: morphology was not tested.


**Status:** SUCCESS (audit complete) — the *finding* is mixed: the stored numbers reproduce, several descriptions do not hold.
**Lines:** 04_language_text (P8). **Date:** 2026-10-05. **Gate:** SIG G1, G1-bis, G4, G7 for the P8 revision at *Oceanic Linguistics*.
**No pre-registration:** this is an audit of existing numbers, not a test. The tests it calls for are pre-registered in E228.

## Hypothesis

Every number and every methodological statement in `papers/P8_linguistic_fossils/draft_v0.1_anonymous.tex`
(the text the reviewers read) can be re-derived from the raw ABVD CLDF files and matches what the code did.

## Method

`01_g1_audit.py` — one script, raw data in, claims table out. Labels, features, cross-validation, SHAP, the
agreement table, the clustering and the cross-language test are re-implemented here. Constants that cannot be
re-derived (word lists, concept sets) are read from the original scripts with `ast`, not retyped. Nothing in
E022/E027/E028/E029/E041/E042 was changed. G1-bis checks: **units** (forms vs concepts vs languages), **direction**
(sign of every stated property), and whether each "independent" quantity is independent.

## Data used

ABVD CLDF snapshot at `experiments/E022_linguistic_subtraction/data/abvd/cldf/` (lexibank/abvd, git `917c5a5`,
2025-10-07): lists Muna 27, Bugis 48, Makassar 166, Wolio 192, Tae' 226, Tolaki 674 — 1,357 forms, 210 concepts.
Software: xgboost 3.0.3, scikit-learn 1.8.0, shap 0.50.0.

## Result

85 statements checked (`results/claims_audit.csv`; console in `results/audit_log.txt`):

| Verdict | n | Meaning |
|---|---|---|
| MATCH | 47 | number reproduced from raw data |
| NOTE | 13 | true, but the reader needs a fact the text does not give |
| NOT-AS-DESCRIBED | 11 | the number is what the code produced, the text describes something else |
| MISMATCH | 9 | the text is wrong or contradicts another part of the text |
| UNSUPPORTED | 3 | no computation or source behind the statement |
| DIRECTION | 2 | sign stated the wrong way round |

**Reproducibility is good.** The stored feature matrix is rebuilt exactly from raw ABVD (B01); every table value —
Table 1, cross-validation, leave-one-language-out, ablation, SHAP top five, the four cells, κ, the clustering, the
expansion group means — comes out the same to the printed precision.

**What does not hold** (row ids refer to `claims_audit.csv`):

1. **Two different "residual" sets are treated as one (A07, A10, A14, F06, H02).** Table 1 reports 356 forms
   (mean of six rates 26.5 %). Every model, the leave-one-language-out table, the agreement table and the clustering
   use 438 forms = *no cognate-set assignment in ABVD* (32.3 %). The abstract joins the two ("438 … (26.5 % of the
   corpus)"). Consequences inside the paper: Tolaki has 114 residuals in Table 1, 134 in Table 3 and 121 "consensus"
   forms; the same six languages get 26.5 %, 28.0 % and 32.3 %.
2. **The "Proto-Austronesian cross-check" is a concept filter (A13).** All uncoded forms for 15 meanings were dropped;
   no form was compared with a reconstruction. Most of the 75 dropped forms are visibly not reflexes of the listed etyma
   (`results/pan_rescued_75.csv`: e.g. 'rope' *talih* → *tulu, otɛrɛʔ, kakoo, oloo, puloli*). ⚠ Domain judgement needed
   on individual forms — flagged for the PI and a specialist; the structural point does not depend on it.
3. **The loanword layer removed no loanword by pattern (A15).** Five string matches, all on unrelated meanings
   (*beli* 'blood' matched Malay *beli* 'buy'); the six real removals come from ABVD's own loan flag, which the text
   does not mention (`results/loan_layer_hits.csv`).
4. **"In five or more of the six languages" counts forms, not languages (A16).** Three of the eight concepts qualify.
5. **"Fewer canonical Austronesian prefixes" is the reverse of the data (D01).** Prefix-like onsets: 37.0 % of
   uncoded forms vs 25.2 % of coded forms, higher in all six languages (within-language OR 1.57 [1.20, 2.04]); the
   feature ranks 23rd of 27 in SHAP. The sentence originates in a hard-coded print statement of `E027/02_shap_and_ranking.py`.
   The argument at tex 447–448 that uses it against the morphological-complexity concern therefore fails, and the
   data point the other way.
6. **Figure 1 caption has the sign inverted (E04)**; the figure and its values are from the 27-input model, not the
   headline 26-input model (E03).
7. **The headline model is not "phonological only" (B04, E03).** 26 inputs = 17 from the written form, 8 from the
   meaning, 1 language-identity code; the identity code is its strongest input. The model without any language-level
   input is the 25-input one (CV AUC 0.727).
8. **The "agreement of two methods" is in-sample (F03).** The classifier was fitted to the rule label on the same
   1,357 forms and then compared with that label (in-sample AUC 0.909 vs 0.760 cross-validated). κ = 0.61 and the 266
   forms are training fit. Two of the five concepts singled out (tex 417) are on the paper's own inherited list (F13).
9. **The cross-language test does not test what is stated (G07)**, the "optimal" k is the edge of the search range
   (G02), and the silhouette value changes with row order (G01, 0.102–0.114).
10. **The 16 additional languages were scored with language-level inputs (H06)**: each list's own share of coded forms,
    and one identity code for all (the code of Tolaki). "Significantly" has no test behind it (H05). The 0.890 for the
    six original lists is in-sample (H03).
11. Smaller: XGBoost was not class-weighted although the text says so (C07); F1 is that of the *coded* class — for
    the candidate class it is 0.53 (C05); the "Swadesh-100" flag covers 174 of the 210 concepts (the set has 176 names; E06); "language cognacy
    coverage" holds hard-coded values that are not the coverage (B02); the digraph sentence omits Muna *dh* and Wolio
    *gh*, both of which the code converted (I02); "syllable" = run of vowel letters (D04); §4.5 names the wrong group
    for 10.5 % (F11).

**Answers this audit already gives to reviewer questions:** R1-8 (*dh*: two Muna forms, *akaradhaa* 'to work' and
*idho* 'green', both cognate-coded, both converted in the test — I03); R1-9 baseline (marker rate per source, D06;
within-language association OR 2.67 [1.88, 3.79], D02); R1-12 and R2 ("E022 label", "false positive": item 1);
R2 (the five concepts: item 8; why k = 5–30: item 9; source of the semantic domains: item 11).

## Conclusion

The pipeline is reproducible; the manuscript's account of it is not accurate in the places listed, and three of them
(items 1, 8, 10) touch statements in the abstract. None can be repaired by wording alone (SIG banned move): items 1–4
need one label definition, items 8–10 need re-computation — done under pre-registration in **E228**. Items 5–7 and 11
are corrections of fact.

## Files

`01_g1_audit.py` · `results/claims_audit.csv` (the table) · `results/audit_log.txt` · `results/audit_summary.json` ·
`results/pan_rescued_75.csv` · `results/loan_layer_hits.csv` · `results/direction_by_label.csv` ·
`results/glottal_by_language.csv` · `results/shap_{full27,ablated26,pure25}.csv` · `results/forms_recomputed.csv`

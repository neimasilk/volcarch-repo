# E228 — analyses for the P8 revision (*Oceanic Linguistics*): what the reviewers asked, with decision rules written beforehand

**Status:** SUCCESS (all sections ran; several outcomes go **against** statements of the submitted manuscript — reported as the rules written beforehand require; for what "pre-registration" means here see the ⚠ note three lines down).
**Lines:** 04_language_text (P8). **Date:** 2026-10-05.
**Pre-registration:** `DESIGN.md`, frozen before any outcome was computed (sha256 at freeze `3b9cba13…fd13f57`).
Two amendments are recorded in its §9: **A1** (S7, written before S7 was run) and **A2** (S5x, **post hoc**, exploratory).
⚠ **What "pre-registered" means here (added 2026-10-06 after an adversarial read):** the decision rules were written down
before the analysis was run — the same day, by the same analyst who had just audited the data — and were **not registered
externally**. The design file now includes the two amendments, so its current hash (`bc66a7be…`) differs from the freeze
hash; design and results entered git in one commit. One pre-set step, the leave-one-concept-out listing of DESIGN §6,
was **not run** in this experiment; it was run as E231 H (largest change of the mean distance 0.003).
Readings corrected on the same day are marked below; the numbers are unchanged. See `experiments/E231_p8_what_coded_means/`.
**Follows:** E227 (audit of the submitted numbers). Nothing in E022–E042 was changed.

## Hypotheses (one per section; decision rules in `DESIGN.md`)

| § | Question | Asked by |
|---|---|---|
| S1 | Does the result depend on how a source writes the glottal stop? | reviewer 1, pt 9 |
| S2 | How does a published "62 % of Makasar basic vocabulary" relate to our ~31 %? Is Makasar set apart from Bugis and Tae'? | reviewer 1, pt 2 |
| S3 | Are "candidates" only uncoded at the *higher* level, i.e. shared with close relatives? | reviewer 1, pts 1, 7, 13 |
| S4 | Agreement between label and classifier when the classifier has not seen the form; example lexemes per cell | reviewer 1, pts 11–12; reviewer 2 |
| S5 | Are same-meaning candidates more alike across languages than different-meaning ones? (permutation control) | manuscript's central negative result |
| S6 | Do the 16 additional lists show a geographic pattern when the model gets no language-level input? | abstract; reviewer 2 ("generalises" = ?) |
| S7 | How much do form-based inputs contribute on their own? | reviewer 1, pt 10 |

## Method and data

ABVD CLDF snapshot (lexibank/abvd, git `917c5a5`, 2025-10-07), lists 27, 48, 166, 192, 226, 674 = 1,357 forms.
**One label:** *candidate* = no cognate-set assignment in ABVD (438 forms); *coded* = any assignment (919).
Models and protocol as published (XGBoost, 5-fold × 10 seeds, leave-one-language-out). Primary model: 25 inputs,
none at language level. Scripts: `p8common.py`, `01_revision_analyses.py` (S1–S6 + release files),
`02_feature_groups.py` (S7), `03_permutation_stratified.py` (S5x). Logs in `results/run_log*.txt`.

## Results

### S1 — glottal stop: **partial dependence** (pre-registered band 0.010–0.020)

| Convention | CV AUC, 25 inputs | Δ | LOLO mean | CV AUC, 26 inputs | Δ |
|---|---|---|---|---|---|
| as published (ʔ / apostrophe) | 0.727 | — | 0.701 | 0.763 | — |
| every marker → *q* | 0.710 | −0.016 | 0.681 | 0.760 | −0.003 |
| every marker → *k* | 0.710 | −0.016 | 0.677 | 0.761 | −0.001 |
| marker not written | 0.720 | −0.006 | 0.699 | 0.759 | −0.004 |
| geminate before consonant, else not written | 0.720 | −0.007 | 0.696 | 0.763 | 0.000 |
| apostrophe → ʔ | 0.727 | 0.000 | 0.701 | 0.763 | 0.000 |
| glottal input removed | 0.710 | −0.016 | 0.681 | 0.760 | −0.003 |

210 of 1,357 forms carry a marker; Muna has none, Wolio four. The classifier stays well above chance under every
convention, but it is **not** the same under every convention: up to 0.016 of AUC (up to 0.024 across held-out languages)
rests on the marker being written as a distinct symbol. In Makasar (26 forms), Tae' (7) and Bugis (2) a
glottal + consonant sequence is also counted as a "consonant cluster". Four lists carry the mark routinely, Wolio in four forms, Muna in none
(⚠ whether that is spelling or phonology is for a specialist; corrected 2026-10-06).

### S2 — Makasar: three classes of meanings (comparison with the published figure withheld; corrected 2026-10-06)

Unit = concept; base = concepts present in the list and in the PMP list (201 for Makasar).

| | retained from PMP | coded, not the PMP etymon | no cognate set at all |
|---|---|---|---|
| **Makasar** | **39.3 %** (79) | 25.9 % (52) | **34.8 %** (70) |
| Bugis | 48.3 % (97) | 28.4 % (57) | 23.4 % (47) |
| Tae' (Sa'dan Toraja) | 48.0 % (96 of 200) | 33.0 % (66) | 19.0 % (38) |
| Wolio | 42.6 % | 33.0 % | 24.4 % |
| Muna | 35.8 % | 49.8 % | 14.4 % |
| Tolaki | 26.2 % | 9.9 % | 63.9 % |

*(Paragraph rewritten 2026-10-06 after three adversarial reads; the table is unchanged. Column 1 means "in the same ABVD
set as the PMP entry", not "retained".)* For Makasar 39.3 % of meanings are in the PMP entry's set, 25.9 % are coded in
another set and 34.8 % are uncoded; with doubtful assignments counted the first figure is 42.8 %. The manuscript's
"residual" (30.9 % of forms in Table 1; 36.9 % of forms / 34.8 % of concepts under the one-label definition) is the
uncoded class. ⚠ **No comparison with the published 38 % / 62 % is made here**: the primary source is unread, and it may
go back to the same lists as ABVD's coding (ABVD names Blust 1993 / 1999 for its PMP list). "Not retained 60.7 %" must
not be written: an uncoded meaning counts as "not in the set" only because it is uncoded. Makasar has 11–16 points more
uncoded meanings than Bugis and Sa'dan Toraja; its lower first column is that same fact. Whether those uncoded forms
are uncoded cognates or replaced words is not decided by this table; a mechanical screen does not favour "merely
uncoded" (E231 tables D, P8, P9). The middle class is not "has Austronesian cognates" (E231 A).
### S3 — most Tolaki "candidates" have a look-alike inside Bungku–Tolaki (decision rule: ≥ 20 points → **met, 60.4**)

Look-alike = stem-tolerant normalised edit distance ≤ 0.34 for the same meaning; chance = same procedure on other meanings.

| Tolaki list 674 compared with | forms | look-alike | chance | excess |
|---|---|---|---|---|
| Proto-Bungku-Tolaki (ABVD 780) — candidates | 97 | 35.1 % | 2.3 % | +32.7 |
| 42 other Bungku–Tolaki lists — candidates | 134 | 67.9 % | 9.6 % | +58.3 |
| **either of the two — candidates** | **134** | **70.9 %** | **10.5 %** | **+60.4** |
| either of the two — coded forms (control) | 74 | 95.9 % | 14.6 % | +81.3 |
| five Tolaki dialect lists — candidates | 86 | 88.4 % | 7.5 % | +80.8 |

At the strict threshold 0.25: 63.4 % against 3.0 %. Of the 34 candidates that match a Proto-Bungku-Tolaki entry, 7
match an entry that ABVD itself assigns to a cognate set (a coding gap in the plain sense); the other 27 match a
proto-form that is itself uncoded. 26 of the 134 have no look-alike even at 0.50; that remainder includes forms that
look Austronesian or like Malay loans (*motaku* 'to fear', *metanggali* 'to dig') — ⚠ prima facie only,
to be checked by the PI against the Austronesian Comparative Dictionary before anything is said about a single form.

All six languages against the other Sulawesi lists in ABVD (71–76 lists, 10 random meanings for chance), threshold 0.25:
Muna 35 % vs 6 %, Bugis 16 % vs 4 %, Makasar 20 % vs 6 %, Wolio 49 % vs 9 %, Tae' 33 % vs 10 %, Tolaki 71 % vs 9 %.
ABVD has few other South Sulawesi lists, so the low Bugis/Makasar figures are partly an absence of comparanda.

Reading *(corrected 2026-10-06)*: the Tolaki rate (64 % of forms uncoded) is not "under-documentation": the words recur
across Bungku–Tolaki and lack an assignment to a *higher-level* cognate set. That is all the count shows. Recurrence
inside the subgroup is **neutral on origin** (the reviewer's example in pt 1 is a loan into a low-level proto-language);
the control — coded Tolaki forms, 95.9 % — belongs beside the 70.9 %; and by meaning (first form), on shared meanings,
list 674 is coded for 38 % and the Tolaki dialect lists for 36–41 %, against a median of 47 % in the 42 comparison
lists (E231 P10b). A look-alike is a mechanical screen, not an etymology.

### S4 — agreement out of fold is "fair", not "substantial"

| Probability from | κ | candidate & profile | candidate, no profile | coded & profile | coded, no profile |
|---|---|---|---|---|---|
| published (in-sample, 27 inputs) | 0.611 | 266 | 172 | 41 | 878 |
| **out of fold, 25 inputs** | **0.308** | **172** | 266 | 105 | 814 |
| out of fold, 26 inputs | 0.370 | 193 | 245 | 95 | 824 |
| language held out, 25 inputs | 0.145 | 159 | 279 | 206 | 713 |

39 % of candidates carry the profile when the model has not seen them (published: 61 %). No concept is "candidate &
profile" in four or more languages (published: five). Tolaki supplies 62 of the 172 (published: 121 of 266).
The five concepts reviewer 2 asks about: 'Fifty' and 'Twenty' ⚠ look like decimal compounds (for a specialist), 'One
Hundred' is not uniform (the Makasar form looks like a different word) *(corrected 2026-10-06)*; all three are uncoded in four
lists and coded in Bugis; for 'to hit' and 'to stand' every language has a different form, and both meanings were
on the submitted version's own list of inherited meanings. Example lexemes per cell: `results/S4_cell_examples.csv`. ⚠ The examples are the forms the classifier scores most extreme, so they show its inputs by construction; that the high-profile examples
look affixed, compounded or phrasal is a reading, not a test *(note added 2026-10-06)*. What they
have in common on the surface is what looks like affixation, compounding or a phrase (*ammikkiriʔ* 'to think', *mie no lambu* 'wife',
*limaŋpulo* 'fifty'); what the uncoded low-profile examples have in common is that they are short (*annan* 'six',
*omba* 'four', *jukuʔ* 'fish') — several look inherited. ⚠ Same caution: prima facie.

### S5 — the negative result does **not** stand as worded (decision rule: p < 0.05 → downgrade)

| Set | same-meaning pairs | mean distance: observed / permuted | p (0.0001 = at most: no permutation of 10,000 as extreme) | look-alike pairs: observed / expected | p |
|---|---|---|---|---|---|
| all candidates | 474 | 0.813 / 0.867 | 0.0001 | 15 / 0.6 | 0.0001 |
| **candidates, numerals excluded (primary)** | **440** | **0.836 / 0.869** | **0.0001** | **4 / 0.5** | **0.0016** |
| … also without the 15 "inherited" meanings | 298 | 0.826 / 0.867 | 0.0001 | 3 / 0.3 | 0.004 |
| candidates with out-of-fold profile, numerals excluded | 68 | 0.843 / 0.854 | 0.25 | 0 / 0.1 | 1.0 |
| pairs inside South Sulawesi | 83 | 0.827 / 0.873 | 0.001 | 0 / 0.1 | 1.0 |
| pairs across subgroups | 357 | 0.838 / 0.868 | 0.0001 | 4 / 0.4 | 0.0009 |
| coded forms (positive control) | 1,793 | 0.578 / 0.856 | ≤ 0.001 | 472 / 8.5 | ≤ 0.001 |

Same-meaning candidates **are** more alike than different-meaning ones. The effect is small — about one eighth of the
drop in distance that coded forms show (0.033 against 0.278), four look-alike pairs in 440 against 472 in 1,793 — but
it is not zero, so "no evidence of shared forms, p = 0.569" has to go. The four pairs: Bugis *ma-kunru* / Wolio
*makundu* 'dull', Wolio *tawa* / Tolaki *tawa* 'leaf', Muna *notente* / Wolio *tente* 'to swell', Bugis *ma-tanəʔ* /
Wolio *matamo* 'heavy' (`results/S5_candidate_lookalike_pairs.csv`).
**Post hoc (A2, exploratory):** permuting inside language × semantic domain leaves an excess of 0.024 (p = 0.0001);
with stem-tolerant distance 22 look-alike pairs against 8–10 expected (p ≤ 0.0004). The shared part is therefore not
only shared affixes.
What can be said *(corrected 2026-10-06)*: among uncoded forms the cross-list similarity is small; the test does not
examine the coded class, so it supports neither "a shared layer" nor "no large shared layer" (E231 A, P2). What does not survive: "each language
innovated independently" — S3 shows that most Tolaki items recur within the subgroup, and S5 shows a small excess of
similarity across lists (not distinguishable inside and outside South Sulawesi; E231 P7).

### S6 — the 16 additional lists: geographic sentence **not supported** (rule: p < 0.05 **and** AUC ≥ 0.60 in ≥ 12 lists)

Mean probability: Sulawesi (8) 0.308, western Indonesia (6) 0.229, exact permutation p = 0.023 — but AUC against the
list's own label is ≥ 0.60 in only **10 of 16** (mean 0.638, range 0.427–0.795), and among *coded* forms the contrast
is 0.242 against 0.208 (p = 0.37). The group difference follows the share of uncoded forms (49.9 % against 28.4 %,
p = 0.056), which is highest in Uma, Kambowa, Kulisusu and Totoli (55–75 %; ABVD ids 748–999). Per-language predicted rates bear no resemblance to the
published ones (Gorontalo 84 % → 31 %, Balinese 63 % → 14 %, Acehnese 63 % → 8 %, Sundanese 2 % → 19 %; Bolaang
Mongondow AUC 0.43). The published rates were produced by the two language-level inputs (E227 H06), so the
interpretations attached to Acehnese and Bolaang Mongondow have no basis.

### S7 — form-based inputs alone

| Inputs | n | CV AUC | LOLO mean | lists ≥ 0.65 |
|---|---|---|---|---|
| written form only | 17 | **0.672** ± 0.008 | **0.641** | 3 of 6 |
| meaning only | 8 | 0.646 | 0.658 | 3 of 6 |
| language identity only | 1 | 0.680 | — | — |
| form + meaning | 25 | 0.727 | 0.701 | 5 of 6 |
| form + meaning + identity (submitted headline) | 26 | 0.763 | 0.722 | 6 of 6 |

"Trained exclusively on phonological features, AUC = 0.763" is not a description of any model that was run. Knowing
only which list a form comes from gives 0.68. The shape of the written word alone gives 0.67 (0.64 for a language the
model has not seen); which meaning the word has gives almost as much.

## Conclusion

Against the submitted manuscript: (1) the headline discrimination is 0.67–0.73, not 0.76, once inputs that are not
about the form are separated out; (2) the two-method agreement falls from κ 0.61 to 0.31; (3) the cross-language test
finds a small shared component, so the negative result cannot stand as "none" — nor, corrected 2026-10-06, as "no large
shared layer": the test covers uncoded forms only (E231 A);
(4) the geographic pattern in 16 further lists is not supported; (5) the result depends in part on whether a list
writes a glottal mark.
*(The remainder of this conclusion was cut on 2026-10-06: it stated the Makasar and Tolaki results and a one-sentence
"picture" in wording that three adversarial reads found too strong, and the wording of any conclusion is the PI's. The
numbers: S2 table above; S3 70.9 % against 10.5 %, control 95.9 %; S5 0.836 against 0.869. What may be said about them:
`papers/P8_linguistic_fossils/REVISION_WORKPLAN.md` §10.7.)*
## Limits

Edit distance on orthography; a look-alike is not a cognate judgement. The stem tolerance ignores up to three
leading characters and so also matches unrelated short stems (the chance column measures exactly that). More than
twenty permutation p-values were computed; only the primary one of S5 was named in advance. Domain judgements on individual forms
(marked ⚠) have not been made by a specialist — **flag for a historical linguist of Sulawesi** (SIG G10).

## Files

`release/p8_candidates.csv` (438 rows) and `release/p8_forms_all.csv` (1,357 rows): ABVD ids, the label, the
submitted version's two flags, out-of-fold probabilities, cell, nearest Sulawesi and Bungku–Tolaki look-alike.
ABVD is CC BY 4.0. **Not to be published before the PI has looked at them** (reviewer 1, pt 13).

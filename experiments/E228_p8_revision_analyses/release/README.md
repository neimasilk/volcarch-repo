# Description of the data files

> Version 1.1 of the deposit. The five data files are unchanged from version 1.0 (published 2026-10-07); only this description was corrected: fields that version 1.0 left unfilled are completed, and a working note at the top was removed.

## 0. Deposit metadata

| Field | Value |
|---|---|
| Title | Uncoded basic vocabulary in six Sulawesi word lists of the Austronesian Basic Vocabulary Database: form-level data release |
| Authors | Mukhlis Amien (Universitas Bhinneka Nusantara; ORCID 0000-0002-1848-167X) and Go Frendi Gunawan (Universitas Bhinneka Nusantara) |
| Version | 1.1 (this description corrected; the data files are those of version 1.0, written 2026-10-05 from ABVD snapshot `917c5a5`) |
| Date | version 1.0: 2026-10-07; version 1.1: October 2026 |
| Licence | CC BY 4.0 — the same licence as ABVD, from which the forms, cognate-set numbers, loan flags and names are copied; the derived columns (scores, distances, cells) are released under the same licence |
| How to cite | Amien, Mukhlis & Go Frendi Gunawan. 2026. *Uncoded basic vocabulary in six Sulawesi word lists of the Austronesian Basic Vocabulary Database: form-level data release* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.23202244 (this DOI resolves to the latest version) — and cite ABVD: Greenhill, Simon J., Robert Blust & Russell D. Gray. 2008. The Austronesian Basic Vocabulary Database: From bioinformatics to lexomics. *Evolutionary Bioinformatics* 4. 271–283. |
| Related publication | the article in *Oceanic Linguistics* (manuscript OL-03-2026-11), to be linked once published |
| Files | `p8_forms_all.csv` (1,357 rows), `p8_candidates.csv` (438 rows), `p8_makasar_uncoded_meanings.csv` (70 rows), `p8_tolaki_pmp_lookalikes.csv` (12 rows), `p8_cell_examples.csv` (92 rows), this `README.md` (sections 8–10 describe the three smaller files) |
| Code | the scripts that built the files and the analyses that use them are in the public project repository, folder `experiments/E228_p8_revision_analyses/` (and E227, E229–E231): https://github.com/neimasilk/volcarch-repo |


`p8_forms_all.csv` has 1,357 rows, one for every form in six ABVD word lists. `p8_candidates.csv` has the 438 rows of the same table with `candidate_no_cognate_set` = 1: same 21 columns, sorted by list name and then by meaning.
Format: plain text (UTF-8, no byte-order mark), comma-separated, one header row; an empty cell means "no value". In Excel use Data > From Text/CSV and choose UTF-8, otherwise ʔ, ŋ and ə are garbled.
Written by the last section of `01_revision_analyses.py` in the repository folder `experiments/E228_p8_revision_analyses/`, using `p8common.py` there; the design is in `DESIGN.md` of the same folder (https://github.com/neimasilk/volcarch-repo).

## 1. What a row is, and where the data come from

One row is one form that ABVD records for one meaning in one of six word lists. ABVD's meaning list has 210 meanings; each of the six lists covers 200 to 210 of them, and a meaning can have up to three forms in a list (117 rows are a second or third form). No form was left out: the six lists hold 1,357 forms in ABVD and all 1,357 are here.

Source: the Austronesian Basic Vocabulary Database (Greenhill, Blust & Gray 2008, *Evolutionary Bioinformatics* 4: 271-283), in its lexibank/abvd CLDF version, repository state `917c5a5` (7 October 2025), licence CC BY 4.0. Forms, cognate-set numbers, loan flags and list and meaning names are copied from it. Please credit ABVD and the list compilers when reusing.

| ABVD id | Name in ABVD | Name used in the article | ISO 639-3 | Compiler (ABVD "author" field) | Forms | Candidates |
|---|---|---|---|---|---|---|
| 27 | Muna (Katobu-Tongkuno Dialect) | Muna | mnb | van den Berg | 219 | 34 |
| 48 | Buginese (Soppeng Dialect) | Bugis | bug | Zainuddin Taha | 242 | 62 |
| 166 | Makassar | Makasar | mak | Abd. Rajab | 217 | 80 |
| 192 | Wolio | Wolio | wlo | J.C. Anceaux | 254 | 83 |
| 226 | Tae' (S.Toraja) | Sa'dan Toraja | sda | Blust from van der Veen (1940) | 216 | 45 |
| 674 | Tolaki | Tolaki | lbw | Omar Abdullah Pidani | 209 | 134 |

## 2. The columns

| Column | Content | Values | How it was computed |
|---|---|---|---|
| `abvd_form_id` | ABVD's identifier of the form | e.g. `27-1_hand-1`: list id, ABVD meaning id, number of the form for that meaning in that list | Copied (ID of `forms.csv`). Unique. |
| `abvd_language_id` | ABVD id of the word list | 27, 48, 166, 192, 226, 674 | Copied. |
| `abvd_language_name` | Name of the list in ABVD | the six names in section 1 | `languages.csv`, name of that id. |
| `concept` | ABVD's English label of the meaning | 210 labels, e.g. `hand`, `leg/foot`, `to think`, `One Hundred` | `parameters.csv`, name of the meaning id. |
| `form_as_in_abvd` | The form exactly as ABVD records it (its Value field), not cleaned | text; 14 forms carry the compiler's bracketed additions, e.g. `[it]təllo`; what such marks mean is not described here | Copied. |
| `abvd_cognacy` | ABVD's cognate-set number(s) for the form | empty (438 rows) or one or more numbers separated by commas, e.g. `33` or `1,64` (43 rows have several); a `?` after a number is ABVD's mark for a doubtful assignment (58 rows) | Copied (Cognacy field). Checked against ABVD's cognate table of the same release: same number of sets and of doubt marks for all 919 coded forms. |
| `abvd_loan_flag` | 1 if ABVD itself flags the form as a loan | 0 (1,346 rows), 1 (11 rows) | ABVD's Loan field: `true` gives 1, `false` gives 0 (no other value occurs in these lists). |
| `candidate_no_cognate_set` | 1 if `abvd_cognacy` is empty (section 3) | 0 (919), 1 (438) | 1 if the Cognacy field is empty or only blanks, else 0. |
| `in_table1_set_of_submitted_version` | 1 if the form belongs to the 356-form set of Table 1 of the submitted manuscript | 0 (1,001), 1 (356) | Section 6. |
| `concept_on_15_concept_list_of_submitted_version` | 1 if the meaning is on the submitted version's list of 15 meanings | 0 (1,254 rows), 1 (103 rows, 15 meanings) | Section 6. |
| `numeral_concept` | 1 if the meaning is a numeral | 0 (1,273), 1 (84 rows, 14 meanings: One to Ten, Twenty, Fifty, One Hundred, One Thousand) | Meaning is in the group "NUMBER" of the meaning classes in `experiments/E027_ml_substrate_detection/00_prepare_features.py`. |
| `p_profile_out_of_fold_25feat` | Classifier score, 25 inputs | 0.007 to 0.963, three decimals | Section 4. |
| `p_profile_out_of_fold_26feat` | Classifier score, 26 inputs (the 25 plus a code for the source list) | 0.007 to 0.982 | Section 4. |
| `p_profile_language_held_out_25feat` | Classifier score, 25 inputs, whole list held out | 0.007 to 0.977 | Section 4. |
| `cell` | The label crossed with the first score | four values, section 4 | From `candidate_no_cognate_set` and `p_profile_out_of_fold_25feat`. |
| `sulawesi_lookalike_distance` | Distance to the closest same-meaning form in other Sulawesi lists | 0.0 to 1.0, three decimals; empty in 3 rows | Section 5. |
| `sulawesi_lookalike_list` | ABVD list in which that closest form stands | 57 list names | Section 5. |
| `sulawesi_lookalike_form` | That form, as ABVD records it | text | Section 5. |
| `bungku_tolaki_lookalike_distance` | The same against Bungku-Tolaki lists only; **Tolaki rows only** | 0.0 to 1.0; filled for 208 of the 209 Tolaki rows, empty for all rows of the other five lists | Section 5. |
| `bungku_tolaki_lookalike_list` | ABVD list of that closest form | 19 list names | Section 5. |
| `bungku_tolaki_lookalike_form` | That form, as ABVD records it | text | Section 5. |

## 3. What the label column records

`candidate_no_cognate_set` = 1 means that the form's Cognacy field is empty in this ABVD snapshot; 0 means that it holds at least one cognate-set number. The column records the state of cognate coding in the database at that date and nothing else; the files make no statement about the origin of any word.
Facts about the cognate-set numbers that a user should know: sets are numbered within each meaning (set 1 of `hand` is not set 1 of `leg/foot`); a number says that ABVD's editors grouped the form with other forms; sets range from two lists to several hundred; 5 of the 11 forms that ABVD flags as loans in these lists carry a set number (Tolaki *kila* 'lightning', ABVD comment "from Malay", sits in the same set as the Proto-Malayo-Polynesian entry).
In the Muna list, 44 of the 185 coded forms have only sets that occur nowhere in ABVD outside the two Muna lists (id 27 and Wuna, id 147, which share one Glottolog code); the same count is 6 for Bugis, 1 for Makassar, 1 for Tae', 0 for Wolio and Tolaki. ("Only sets" means every set of the form; counting forms with at least one such set gives 45 for Muna.)
The loan flag is ABVD's own, occasional flag: 11 forms in these six lists (Muna 2, Bugis 1, Makassar 1, Wolio 2, Tae' 0, Tolaki 5). No loanword was removed from the files.

## 4. The three score columns and `cell`

The scores come from a statistical classifier (gradient-boosted decision trees, XGBoost, 300 trees of depth 4, settings unchanged from the earlier analysis in experiment E027). It was trained to tell forms **without** a cognate-set number from forms **with** one; the score is the model's value for the class "without", so a higher value means the form looks more like the candidate forms.
Its 25 inputs: 17 from the written form (length in characters; number and share of vowels; ends in a vowel; contains a glottal mark, ʔ or an apostrophe; contains ŋ or ng, or one of mb, nd, nj, mp, nk, nc, nt (11 strings in the code, nine distinct); contains a hyphen or the same 2- or 3-letter string twice in a row; number of consonant-letter clusters; begins with ma, me, mo, pa, ka, ta, na, po or aŋ (23 strings in the code, which the code calls "prefix-like"; they amount to these nine beginnings, and the input does not know whether the beginning is a prefix); first letter a, b, k, m, p, s, t or other) and 8 from the meaning (on a core-vocabulary list or not; one of seven broad classes: action, body, grammar, nature, number, quality, other). The 26th input is a number identifying the source list (Bugis 0, Makassar 1, Muna 2, Tolaki 3, Tae' 4, Wolio 5).
- **Out of fold** (first two columns): the 1,357 forms were split at random into five parts; a model trained on four parts scored the fifth; this was done for every part, and the whole procedure was repeated ten times with different splits. The value is the mean of the ten scores. Each score comes from a model that had not seen that form, but may have seen other forms of the same list and the same meaning in other lists.
- **List held out** (third column): the model was trained on the other five lists only and scored the sixth, once.

The scores rank forms. They are **not** probabilities that a word is non-Austronesian, foreign or of unknown origin. `cell` crosses the label with the 25-input out-of-fold score at the threshold 0.5 ("profile" = 0.5 or more, cut on the unrounded value, so one form printed as 0.500 is in a "no profile" cell):

| `cell` | Meaning | Rows |
|---|---|---|
| `candidate & profile` | candidate, score 0.5 or more | 172 |
| `candidate & no profile` | candidate, score below 0.5 | 266 |
| `coded & profile` | has a cognate-set number, score 0.5 or more | 105 |
| `coded & no profile` | has a cognate-set number, score below 0.5 | 814 |

## 5. The look-alike columns

> **WARNING. A distance of 0.0 does NOT mean that the two forms are identical.** The distance is a normalised edit distance between normalised spellings that **ignores up to three leading characters on either side**. A look-alike is a mechanical screen. It is not a cognate judgement and not an etymology.

How the distance is computed (`norm` and `d_stem` in `../p8common.py`):
1. **Normalised spelling:** lower case; text in square or round brackets deleted; asterisk, hyphen, apostrophe (straight or curly), grave accent, full stop, comma, semicolon, slash, spaces and ʔ deleted. All other symbols stay and count as different letters (ŋ is not ng, ɛ is not e, β is not w).
2. **Edit distance:** the least number of single-letter insertions, deletions or replacements that turn one string into the other, divided by the length of the longer string.
3. **Leading characters ignored:** the distance is computed for the whole strings and for every way of cutting 0, 1, 2 or 3 characters off the start of the first string and 0, 1, 2 or 3 off the start of the second (each remainder must keep at least 3 characters). The smallest value is reported. Two words can therefore agree in their last three letters only.

Real examples from the file with distance 0.0 and different spellings:

| Row | Form | Closest form | What was ignored |
|---|---|---|---|
| `192-102_rat-1` (Wolio, `rat`) | bokoti | bukoti (Banggai, W. dialect) | first two letters of both: koti = koti |
| `27-24_head-1` (Muna, `head`) | fotu | potu (Kambowa) | first letter of both: otu = otu |
| `27-3_right-1` (Muna, `right`) | suana | koana (Mori) | first two letters of both: ana = ana |
| `674-119_earthsoil-1` (Tolaki, `earth/soil`; Bungku-Tolaki columns) | wuta | βuta (Kodeoha) | first letter of both: uta = uta |

In the Sulawesi column 687 of the 1,354 rows with a value show 0.0, and in 380 of these the normalised spellings differ (Bungku-Tolaki column: 129 rows, 51 differ). The columns are filled for coded forms as well as for candidates (0.0 occurs for 116 of the 438 candidates and 571 of the 919 coded forms).

**Which lists were compared.** Only forms with the identical ABVD meaning label are compared. Only the single closest form is listed; if several lists tie, the first one met in the ABVD file is shown.
- **Sulawesi columns** (all six lists): every ABVD list whose coordinates in ABVD's language table lie in the rectangle latitude -6.6 to 2.0, longitude 118.5 to 125.6. Lists whose name begins with "Bajo" are excluded, and so is every list with the same Glottolog code as the form's own list (for Muna this removes Wuna, for Tolaki the five Tolaki dialect lists). **Why the Bajo lists are left out** (reason recorded 2026-10-07; the code written on 2026-10-05 did not record it): the 17 "Bajo" lists (ABVD 489, 619–634) all carry the Glottocode indo1317, Indonesian Bajau, a Sama-Bajaw language whose speakers settled the coasts of Sulawesi from elsewhere; Glottolog does not place it under any Sulawesi group (it is not in the Bungku-Tolaki tree, checked 2026-10-07, and belongs to the Sama-Bajaw branch of Greater Barito). A comparison meant to find look-alikes in *Sulawesi* languages therefore left them out; all 17 share one set of coordinates in ABVD. 77 lists lie in the rectangle; each form is compared with 75 lists (Muna), 76 (Bugis, Makassar, Wolio, Tae') or 71 (Tolaki). The other five of the six lists are among them. The rectangle was chosen by the analysts, not taken from a source; it holds Sulawesi and some adjoining islands. ("Sulawesi" is used in a second sense elsewhere in the project: experiment E231 counts cognate sets "attested only in Sulawesi" with the same rectangle but **with** the Bajo and proto-language lists, 96 lists. The two are not the same set.)
- **Bungku-Tolaki columns** (Tolaki rows only): 43 lists: ABVD's Proto-Bungku-Tolaki list (id 780) and 42 lists chosen by ABVD id in the code (41, 875 to 908, 914 to 919, 972; Mori, Bungku, Moronene, Kulisusu, Waru and others). The five Tolaki dialect lists (909 to 913) are not used. **Basis of the 42 ids** (recorded 2026-10-07; not recorded in the code): the 42 lists carry 13 Glottocodes — baho1237 Bahonsuai, bung1269 Bungku, koro1311 Koroni, kuli1254 Kulisusu, mori1268 Mori Bawah, mori1269 Mori Atas, moro1287 Moronene, pado1242 Padoe, raha1237 Rahambuu, talo1252 Taloki, toma1248 Tomadino, waru1266 Waru, wawo1239 Wawonii — and every one of them is a daughter of Glottolog's languoid Bungku-Tolaki (bung1268), checked on 2026-10-07 against Glottolog's classification tree (49 languoids under bung1268; Tolaki, tola1247, is one of them). 41 of the 42 lists were contributed by D. Mead (1999), one (Mori, id 41) by another compiler. Any ABVD list of a Bungku-Tolaki language outside those ids would have been missed; none was found among the lists in the rectangle.

**An empty cell** in the Sulawesi columns (3 rows: `166-26_hair-1`, `192-173_at-1`, `674-173_at-1`) means the normalised spelling has fewer than 2 characters, so nothing was compared. In the Bungku-Tolaki columns it is empty for every non-Tolaki row and for `674-173_at-1` (same reason).

## 6. The two columns that refer to the submitted version

They record how Table 1 of the submitted manuscript was built. The revised analyses do not use them as a label or filter (the label is `candidate_no_cognate_set` alone).
- `concept_on_15_concept_list_of_submitted_version`: the meaning is one of 15 (`One Thousand`, `below`, `cloud`, `heavy`, `red`, `rope`, `to blow`, `to come`, `to hit`, `to hold`, `to say`, `to see`, `to sit`, `to stand`, `to steal`) for which the submitted version's code lists a reconstructed Proto-Austronesian or Proto-Malayo-Polynesian root. Those roots have not been re-checked here.
- `in_table1_set_of_submitted_version` = 1 if the form has no cognate-set number, is not flagged as a loan by ABVD, did not match the submitted version's lists of Sanskrit, Arabic and Malay trade-word patterns (form equal to, beginning or ending with a pattern of 3 or more letters that is at least 60% of the form), and its meaning is not on the 15-meaning list. All 356 are candidates. Of the 82 candidates outside the set, 75 are on the 15-meaning list, 6 are ABVD-flagged loans and 1 more was caught by the pattern lists; the pattern match itself is not stored as a column.

## 7. Candidates per list

Muna 34, Bugis 62, Makassar 80, Wolio 83, Tae' 45, Tolaki 134 (438 of 1,357 forms).

Single forms in these files have not been assessed by a specialist.

## 8. `p8_makasar_uncoded_meanings.csv` — the 70 uncoded Makasar meanings (added 2026-10-07)

One row per meaning for which the Makasar list has a form but no cognate-set number and the meaning has a PMP entry (70 of the 201 meanings compared in Section 3.2 of the article). Built by `experiments/E231_p8_what_coded_means/02_posthoc_checks.py` (table P9). Columns: `meaning` (ABVD label); `makasar_forms` (the Makasar form or forms, as recorded); `pmp_forms` (ABVD's Proto-Malayo-Polynesian entry or entries for the meaning, with their asterisks); `bugis_class` and `toraja_class` — how the Bugis and the Tae'/Sa'dan Toraja form of the same meaning is coded (`in_PMP_set`, `coded_other`, `uncoded`, or empty when the list has no form); `bugis_forms`, `toraja_forms` (those forms, with ABVD's set number in square brackets). Purpose: inspection by anyone able to judge whether a Makasar form is a cognate the coders missed or another word. No row has been assessed by a specialist.

## 9. `p8_tolaki_pmp_lookalikes.csv` — twelve uncoded Tolaki forms that resemble the PMP entry (added 2026-10-07)

The twelve forms of Section 3.3 of the article (of 128 uncoded Tolaki forms whose meaning has a PMP entry; about two are expected by chance). Built by `experiments/E231_p8_what_coded_means/01_tabulations.py` (table D). Columns: `abvd_form_id`; `tolaki_form`; `meaning`; `pmp_form` (ABVD's PMP entry); `distance` (the stem-tolerant normalised edit distance of section 5, 0.34 or less); `distance_le_0.25` (1 if the strict threshold is also met); `tolaki_form_is_loan_flagged` (ABVD's loan flag). A resemblance by this measure may be a coding gap, a chance resemblance or a loan; the file does not say which.

## 10. `p8_cell_examples.csv` — the example pool behind Table 7 of the article (added 2026-10-07)

92 rows (73 different forms) built by `experiments/E229_p8_revision_tables/01_tables.py`: for each of the four cells of the label-by-score table, the forms the 25-input model scores most extremely (`scope` = `all six` for the pool over all lists, otherwise per list), with ABVD's cognate-set number, the loan flag, the PMP and PAn entries of the meaning and their set numbers, whether the form shares a set with the PMP entry, and the nearest look-alikes of section 5. Because the rows are chosen by the classifier's score, they illustrate its inputs by construction; they are not evidence for any reading of the profile, and no form has been assessed etymologically.

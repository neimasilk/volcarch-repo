# E231 — what "coded" covers, how the six lists compare, and stricter versions of four inputs (P8 revision)

**Status:** SUCCESS (all tabulations ran; 29 of 29 anchor checks reproduce). Descriptive — counts and intervals, plus four
unadjusted paired tests in table C; no test of a pre-stated hypothesis. ⚠ **Read the post hoc section first**: two
further adversarial reads of the same day showed that several readings of tables A, C, D and E below do not hold as
first written (each is marked where it stands).
**Lines:** 04_language_text (P8 revision at *Oceanic Linguistics*). **Date:** 2026-10-06. **Follows:** E227–E230.
**Design:** `DESIGN.md`, written before the script; **not externally registered**, and not blind — an adversarial read of
the same day had produced rough versions of several of these counts (listed in its §0).
**Checked:** the headline counts of A, D, E and F were derived a second time by the orchestrator from the raw CLDF files
with a separate script (it shares only the loader, the normaliser and the distance function of `p8common.py`). They
agree, with one explained difference (the "Sulawesi-only" share: the builder follows the design and counts
reconstructed Sulawesi proto-languages as Sulawesi; the check script excluded every proto-language list).

## Why (hypothesis: none)

An adversarial read of the documents written for the revision found the numbers sound and several *readings* of them
unsupported — above all the habit of treating "has a cognate-set number in ABVD" as "inherited Austronesian", and
"recurs within Bungku–Tolaki" as "ordinary inherited vocabulary". E231 replaces those readings by counts.

## Method and data

ABVD CLDF snapshot (git 917c5a5; 2,036 lists, of which 517 have no coordinates). Six lists as before.
A cognate set = (meaning, set number); breadth = number of **lists** (not languages) with a form in it, counted over
all 2,036 lists including proto-language lists. "Attested only in Sulawesi" means: every list in the set lies inside a
coordinate box (latitude −6.6 to 2.0, longitude 118.5 to 125.6). A list without coordinates (517 lists) is never
Sulawesi, so one such list in a set removes the set from the class; a proto-language list counts as Sulawesi only if
its name contains "Sulawesi" or "Bungku". The box holds 96 lists: 49 Bungku–Tolaki lists, 17 Bajo lists, the three
South Sulawesi target lists and 27 others (`results/P6_sulawesi_box_lists.csv`) — a region of a map, not a set of
Sulawesi languages.
`01_tabulations.py`; outputs in `results/` (one CSV per table, `A_list_classification.csv` shows how every ABVD list
was classed, `run_log.txt`).

## Results

### A — "coded" is not one thing (`A_coded_breadth_by_list.csv`, `A_middle_class_breadth.csv`, `A_loan_flagged.csv`)

| List | Coded forms | Median breadth of the set (lists) | In a set of ≤ 5 lists | In a set attested only in Sulawesi | Loan-flagged by ABVD |
|---|---|---|---|---|---|
| Muna | 185 | 54 | **35.7 %** | **33.5 %** | 1 |
| Bugis | 180 | 284 | 18.3 % | 10.6 % | 0 |
| Makasar | 137 | 277 | 12.4 % | 6.6 % | 0 |
| Wolio | 171 | 207 | 10.5 % | 9.4 % | 2 |
| Sa'dan Toraja | 171 | 259 | 14.6 % | 7.6 % | 0 |
| Tolaki | 75 | 373 | 5.3 % | 8.0 % | 2 |

ABVD flags loans only occasionally (eleven forms in these six lists, none in the Sa'dan Toraja list); five of the
eleven carry a cognate-set number. The clearest case: Tolaki *kila* 'lightning' (ABVD comment: from Malay; loan flag
set) sits in the same set as the PMP entry. (Wolio *fikiri* and Tolaki *mepikiri* 'to think' share a set; ABVD marks
the Wolio assignment as doubtful.) So a cognate-set number says that ABVD's editors grouped a form with others.

The middle class of the Makasar table of E228 ("coded, but not the PMP etymon"), by meaning:

| List | Meanings in the middle class | Widest set has ≤ 5 lists | Set attested only in Sulawesi | Median breadth |
|---|---|---|---|---|
| Muna | 100 | 62 % | 58 % | 4 |
| Bugis | 57 | 44 % | 26 % | 8 |
| Makasar | 52 | 33 % | 17 % | 21 |
| Wolio | 65 | 22 % | 18 % | 30 |
| Sa'dan Toraja | 66 | 35 % | 17 % | 15.5 |
| Tolaki | 19 | 16 % | 32 % | 35 |

**What follows** *(item 1 corrected post hoc, see P2)*. (1) The share of *uncoded* forms depends on what happens to be in
the database. Muna has the fewest uncoded forms (15.5 %), and 62 of its 185 coded forms sit in sets confined to the
Sulawesi box — but for **44 of those 62 the only other list in the set is ABVD's second Muna list** (Wuna, id 147,
same Glottocode): the form is "coded" because it recurs in the same language. Counted as uncoded they would put Muna
at 35.6 %. Tolaki's five dialect lists were not cross-coded in that way (0 forms). The ranking of the lists by
uncoded share is therefore not a ranking by divergence, and the label "coded" is not the same kind of thing in
every list. (2) "Uncoded" is **not an upper bound** on non-inherited
vocabulary, and the middle class is **not** "words with Austronesian cognates": both classes mix things. ⚠ Breadth
depends on how many lists ABVD holds for a region; a narrow set is not thereby a loan, a wide one not thereby inherited.

### B — how the lists compare with each other (`B_pairwise_cognate_sharing.csv`)

Share of the meanings present in both lists for which the two lists have a form in the same cognate set:

| | Bugis | Makasar | Sa'dan Toraja | Muna | Wolio | Tolaki |
|---|---|---|---|---|---|---|
| **Bugis** | — | 41.4 % [35.0, 48.2] | **53.1 %** [46.4, 59.8] | 27.3 % | 32.5 % | 17.5 % |
| **Makasar** | | — | 39.2 % [32.9, 46.0] | 23.4 % | 25.7 % | 16.0 % |
| **Sa'dan Toraja** | | | — | 30.9 % | 38.5 % | 20.5 % |
| **Muna** | | | | — | 41.3 % [34.7, 48.2] | 19.5 % |
| **Wolio** | | | | | — | 21.8 % |

Within South Sulawesi, Bugis and Sa'dan Toraja share a set for 53 % of the meanings; Makasar shares one with each of
them for 39–41 %. This is the comparison reviewer 1's question is about (Makasar against its relatives), and it is
visible in ABVD's own coding. ⚠ The difference between 39–41 % and 53 % was **not tested** (no paired test was run, and
the intervals of 41.4 % and 53.1 % overlap at their edges), and it is mostly Makasar's larger uncoded share counted
again: among the meanings that are coded in **both** lists the figures are 76.3 % (Makasar–Bugis), 70.1 %
(Makasar–Sa'dan Toraja) and 81.0 % (Bugis–Sa'dan Toraja) (post hoc P1). The Tolaki row is low throughout because two thirds of the
Tolaki list is uncoded.

### C — the PMP table with intervals (`C_retention_intervals.csv`, `C_paired_tests.csv`)

Makasar: in the PMP entry's set 39.3 % [32.8, 46.2]; coded otherwise 25.9 % [20.3, 32.3]; uncoded 34.8 % [28.6, 41.6]
(201 meanings; Wilson 95 %). Bugis 48.3 % [41.5, 55.1] in the set; Sa'dan Toraja 48.0 % [41.2, 54.9]. (The files and
the first version of this README call the first class "retained from PMP"; it is set membership, not retention.)
Paired on the same meanings (exact McNemar, four tests, unadjusted): Makasar is less often in the PMP set than Bugis
(p = 0.020) and Sa'dan Toraja (p = 0.033), and has more uncoded meanings than Bugis (p = 0.006) and Sa'dan Toraja
(p = 0.0001).
⚠ These are **one fact, not two** (post hoc P1): "retained" here means "in the same ABVD set as the PMP entry", and an
uncoded meaning cannot be in that set. Among the meanings that are coded, 60.3 % (79 of 131) are in the PMP set for
Makasar, 63.0 % (97 of 154) for Bugis and 59.3 % (96 of 162) for Sa'dan Toraja. So 39.3 % is a lower bound on
retention (the upper bound, if every uncoded meaning belonged to the set, is 74.1 %), and what sets Makasar apart
from its two relatives in these data is its uncoded share.
⚠ *(Third read, P8 and P9.)* That does **not** mean "merely not yet coded". The conditional figures are the same
whether the uncoded forms are uncoded cognates or replaced words, so they decide nothing; and the look-alike screen
finds a PMP look-alike for 2 of 75 uncoded Makasar forms (about one by chance; the screen finds 38 % of the Makasar
forms that ABVD itself puts in the PMP set) and a look-alike in Bugis or Sa'dan Toraja for 10 of 80 (12.5 %; chance
3.8 %; coded forms: 90 of 136, 66 %) — a small excess over chance, far below the coded forms. The data are compatible with the lexical divergence reported for Makasar in the literature; they do not
establish it and say nothing about its cause. For a specialist: `results/P9_makasar_uncoded_meanings_for_specialist.csv`.

### D — the Bungku–Tolaki lists are all thinly coded; some Tolaki candidates resemble the PMP form

(`D_bt_lists.csv`, `D_summary.json`, `D_pmp_lookalikes_by_list.csv`, `D_tolaki_pmp_lookalikes.csv`)

Share of forms with a cognate-set number: Tolaki 35.9 %; Proto-Bungku-Tolaki 35.6 %; the 42 comparison lists
29.9–59.6 % (median 45.0 %; 41 of 42 at or below 50 %); the five Tolaki dialect lists 22.9–39.1 %. The 42 lists are
**13 languages** (Glottocodes) and come from two contributors (one list by Lapotiwa, 41 by Mead — the author field
spells Mead in two ways, merged here by hand). **The unit decides the answer.** By form, only 4 of the 42 comparison
lists are at or below Tolaki's rate (10th percentile 36.2 %) — but the share by form falls with the length of a list
(Spearman −0.55). By meaning on shared meanings, taking the **first form** of each meaning: list 674 38.3 %, the five
Tolaki dialect lists 36.1–40.6 %, median of the 42 comparison lists 46.7 % (5 at or below). If a meaning counts as
coded when **any** of its forms is, lists rich in synonyms rise (Konawe, 384 forms for the same 133 meanings: 50.4 %;
Mekongga 46.6 %; median 48.4 %, 3 at or below) — an effect of synonyms, not of coding. So the six Tolaki lists sit on
the lower side of a thinly coded subgroup, by a moderate margin *(corrected post hoc, P5, P10, P10b; earlier versions
said "the rate of its whole subgroup", then "Tolaki and its dialect lists are among the least coded", then — wrongly —
that the Konawe list showed coding status to be a property of a list rather than of a language)*.
Candidates that are a look-alike of ABVD's PMP form for the **same** meaning (stem-tolerant distance ≤ 0.34).
"Candidates compared" = forms of at least two normalised letters whose meaning has a PMP entry; "chance" = the same
comparison against the PMP forms of 50 randomly drawn other meanings:

| List | Candidates compared | Look-alike of the PMP form | Chance | Coded forms (control) |
|---|---|---|---|---|
| Tolaki | 128 | **12 (9.4 %)** | 1.9 % | 35.2 % |
| Makasar | 75 | 2 (2.7 %) | 1.2 % | 26.5 % |
| Sa'dan Toraja | 39 | 2 (5.1 %) | 2.6 % | 36.3 % |
| Wolio | 77 | 2 (2.6 %) | 2.1 % | 30.7 % |
| Bugis | 61 | 0 | 1.1 % | 24.4 % |
| Muna | 33 | 0 | 1.0 % | 17.0 % |

The twelve Tolaki forms are in `D_tolaki_pmp_lookalikes.csv` (meanings: 'to dream', 'to burn', 'he/she', 'I', 'yellow',
'to fear', 'to choose', 'to drink', 'painful, sick', 'to turn' — loan-flagged —, 'to buy', 'they'); the pairings are
not printed here because none has been seen by a specialist. ⚠ A mechanical screen: whether any of
these is a reflex is for a specialist, and no pairing should be printed before one has seen it. Three of the twelve
sit exactly on the threshold (0.333); nine remain at 0.25. About two of the twelve are expected by chance (1.9 % of
128), and one is flagged as a loan: a look-alike may be a coding gap, a chance resemblance or a loan, and the screen
does not tell these apart. Its sensitivity, measured on the right control (post hoc P5): of the 50 Tolaki forms
that ABVD itself places in the PMP entry's set, the screen finds 24 (48 %). *(The first version called 9.4 % "a
floor … of plain gaps in the top-level coding"; that claimed more than a screen can show.)*

### E — stricter versions of four inputs (`E_strict_variants.csv`)

| Input | As coded in the feature script | Strict version | Candidates vs coded, original | Candidates vs coded, strict |
|---|---|---|---|---|
| "nasal" | contains ŋ or *ng*, or one of mb, nd, nj, mp, nk, nc, nt | nasal letter + stop letter (after *ng*→ŋ) | 30.4 % vs 23.7 %, OR 1.55 [1.18, 2.04] | 21.0 % vs 12.7 %, OR 2.05 [1.47, 2.84] |
| "reduplication" | a hyphen **or** a repeated 2–3 letter block | repeated block only | 11.9 % vs 9.8 %, OR 1.68 [1.12, 2.51] | 5.3 % vs 3.3 %, OR 1.61 [0.87, 2.98] |
| "prefix-like" | begins with one of nine letter strings | the same, and ≥ 6 letters | 37.0 % vs 25.2 %, OR 1.57 [1.20, 2.04] | 24.4 % vs 12.3 %, OR 2.31 [1.68, 3.17] |
| length | characters, marks included | letters only | +1.09 [0.86, 1.31] | +0.93 [0.72, 1.13] |

Of the 142 forms flagged "reduplication", 93 are flagged by a hyphen (Bugis 64, Sa'dan Toraja 25) — in those lists a
hyphen ⚠ appears to mark a morpheme boundary set by the compiler (for a specialist). Pooled, 6.6 % of the candidates and
7.0 % of the coded forms contain a hyphen; within lists the odds ratio is 1.57 [0.94, 2.63]. 88 of the 394 "prefix-like" forms have four letters or fewer (*mata*, *tau*), and those are *less*
often candidates.

*(Reading corrected post hoc, P3 and P4.)* The table does **not** show that the nasal and onset contrasts "survive and
get larger". The strict onset version adds a length criterion, and "at least six letters" alone gives an odds ratio
of 2.56 [1.98, 3.32] — the larger figure is the length effect carried in by the definition. With the number of
letters held fixed (strata = list × letters) the onset string gives 1.09 [0.80, 1.48] and the nasal input 1.24
[0.84, 1.81] (strict) or 0.95 [0.70, 1.30] (as coded); the written glottal mark (2.60 [1.79, 3.76]), the action
meaning (2.22 [1.67, 2.95]) and consonant-letter clusters (1.51 [1.09, 2.09]) hold. Conditioning on length may
over-adjust (a prefix makes a form longer), so the statement is "not separable from length", not "absent". The
length difference itself gets slightly smaller with letters only (+1.09 → +0.93; the interval printed for +1.09 in
the table above, [0.86, 1.31], comes from another bootstrap seed than E229 T6's [0.87, 1.32] — quote T6).
*(Third read, P11.)* Clusters are not on the same footing as the glottal mark and the action meaning: they overlap
with nasal + stop pairs and with ʔ + consonant, and with the glottal mark also held fixed the figure is 1.38
[0.98, 1.93]. And "not separable" for the onset string has a wide bracket: 1.09 with all letters held fixed, 2.51
[1.85, 3.40] with the letters after the matched string held fixed. For "reduplication" the estimate
is the same under each definition (1.68, 1.61 for the repeated block, 1.57 for the hyphen); with half the forms the
interval includes 1 — too little to stand as a separate property, not evidence of absence.

### F — "ends in a vowel" is the final glottal mark (`F_final_segment.csv`, `F_tae_final_k.csv`)

Bugis, Makasar and Sa'dan Toraja pooled: candidates are 23.8 % of the vowel-final forms (390), 25.9 % of the forms
ending in another consonant (135) and **39.3 %** of the forms ending in a glottal mark (150). With the glottal-final
forms set aside, vowel-final against other-consonant-final gives OR 0.91 [0.58, 1.44]: no difference detected (the
interval is wide). The within-list
odds ratio of 0.66 reported for "ends in a vowel" in E229 T6 is therefore the glottal mark counted a second time, and
"final vowel" should not be listed as a separate property. Eleven Sa'dan Toraja forms end in *k* (ten of them coded),
while 45 forms of the same list end in a glottal mark: the two are used side by side in one source, and ABVD's PMP
entries for two of the eleven meanings themselves end in *k* (*\*manuk*, *\*tasik*). ⚠ So this *k* should **not** be
presented as a way of writing the glottal stop; for a specialist (post hoc P7).

### G — intervals for the AUC (`G_auc_intervals.csv`)

Pooled out-of-fold AUC: form + meaning **0.737** [0.709, 0.766] resampling forms, [0.708, 0.767] resampling meanings;
form only **0.678** [0.647, 0.709] / [0.646, 0.712]. (The 0.727 and 0.672 quoted elsewhere are the mean of the fifty
test-fold AUCs — a different, slightly lower quantity.) ⚠ The pooled figure is computed on probabilities **averaged
over the ten repetitions** — a small ensemble, which is why it is higher than any single run — and the bootstrap
holds those predictions fixed, so the interval leaves out the variation of training. The headline figures remain
0.727 and 0.672, with "about ± 0.03". Inside each list, out of fold, form + meaning: Muna 0.653,
Bugis 0.739, Makasar 0.751, Wolio 0.664, Sa'dan Toraja 0.669, Tolaki 0.854 — so the pooled figure is not produced by
differences between the lists.

### H — the step E228 had promised (`H_leave_one_concept_out.csv`)

Removing any single meaning from the primary set of E228 S5 changes the mean distance by at most 0.003 (T = 0.8357
with all 440 pairs). The meanings that pull it down most: 'heavy', 'dull/blunt', 'to swell', 'grass', 'near',
'to chew', 'red', 'leaf'. No single meaning carries the small excess. ⚠ Several of these are quality terms whose
uncoded forms begin with *ma-/mo-/me-* in more than one list ('heavy' 4 of 4 forms, 'dull' 3 of 4, 'near' 3 of 3;
`results/P7_influential_meanings.csv`): part of the similarity may be a shared prefix, not shared roots (specialist).

## Post hoc additions (2026-10-06, after a second and a third adversarial read) — `02_posthoc_checks.py`

Not part of the design. Every count below was asked for by a reader who had seen the results above; the orchestrator
computed them with its own script (three anchors against table E reproduce: 1.568, 2.306, 2.046). Files: `results/P*.csv`,
`P_posthoc.json`, `run_log_posthoc.txt`.

| | What was counted | Result |
|---|---|---|
| P1 | share in the PMP entry's set among **coded** meanings | Makasar 60.3 % (79 of 131), Bugis 63.0 % (97 of 154), Sa'dan Toraja 59.3 % (96 of 162); bounds on Makasar's retention 39.3–74.1 % |
| P1 | shared set among meanings coded in **both** lists | Bugis–Sa'dan Toraja 81.0 % (111 of 137), Makasar–Bugis 76.3 % (87 of 114), Makasar–Sa'dan Toraja 70.1 % (82 of 117) |
| P2 | coded forms whose every set is confined to lists of the same language | Muna 44 of 185 (the other list: Wuna); Bugis 6, Makasar 1, Sa'dan Toraja 1 (sets with no other list at all); Wolio 0; Tolaki 0 although five dialect lists share its Glottocode. Muna's uncoded share with those 44 counted as uncoded: 35.6 % (published 15.5 %) |
| P2 | median breadth of the widest set, in languages (Glottocodes) instead of lists | Muna 40, Bugis 208, Makasar 204, Wolio 147, Sa'dan Toraja 192, Tolaki 253 |
| P3 | odds ratios with the number of letters held fixed (strata = list × letters) | onset string 1.09 [0.80, 1.48]; nasal strict 1.24 [0.84, 1.81]; nasal as coded 0.95 [0.70, 1.30]; ≥ 1 consonant-letter cluster 1.51 [1.09, 2.09]; written glottal mark 2.60 [1.79, 3.76]; action meaning 2.22 [1.67, 2.95]. "≥ 6 letters" alone (strata = list): 2.56 [1.98, 3.32]; onset string among forms of ≥ 6 letters 1.30 [0.85, 2.00], of ≤ 5 letters 1.20 [0.81, 1.78] |
| P4 | candidate share of hyphenated forms | Bugis 34.4 % (22 of 64) against 22.5 % of the other forms; Sa'dan Toraja 20.0 % (5 of 25) against 20.9 %; within-list odds ratio 1.57 [0.94, 2.63] |
| P5 | Tolaki among the 42 comparison lists | 4 lists at or below Tolaki's coded share (35.9 %); 10th percentile 36.2 %, median 45.0 % |
| P5 | sensitivity of the PMP look-alike screen | finds 24 of the 50 Tolaki forms that ABVD places in the PMP entry's set (48 %); 1 of the 21 coded in another set |
| P6 | the "Sulawesi box" | 96 lists, 42 Glottocodes: 49 Bungku–Tolaki, 17 Bajo, 3 South Sulawesi target lists, 27 others (among them Proto-South Sulawesi, Soboyo, the Muna–Buton and northern lists) |
| P7 | Sa'dan Toraja final segments | 45 forms end in a glottal mark, 11 in *k* |
| P7 | S5 split of E228 | excess 0.046 inside South Sulawesi (permutation SD 0.014), 0.030 elsewhere (SD 0.007): the difference, 0.016, is about one combined SD |
| P8 | sensitivity of the PMP look-alike screen on forms ABVD puts in the PMP entry's set | Muna 36 % (30 of 83), Bugis 37 % (41 of 112), Makasar 38 % (33 of 87), Wolio 49 % (49 of 101), Sa'dan Toraja 53 % (58 of 110), Tolaki 48 % (24 of 50) |
| P9 | Makasar and its relatives on meanings coded in both lists | in the PMP set: Makasar 70, Bugis 69 of 110 (exact McNemar p = 1.0); Makasar 70, Sa'dan Toraja 71 of 115 (p = 1.0) |
| P9 | what the relatives have for Makasar's uncoded meanings | Bugis (70 meanings): in the PMP set 28, coded otherwise 16, uncoded 26; Sa'dan Toraja (69): 25, 22, 22. List for a specialist: `P9_makasar_uncoded_meanings_for_specialist.csv` |
| P9 | look-alikes among the other two South Sulawesi lists, same meaning (chance: 20 other meanings) | candidates: Makasar 10 of 80 (12.5 %; chance 3.8 %), Bugis 8 of 62 (12.9 %; 2.4 %), Sa'dan Toraja 8 of 45 (17.8 %; 4.2 %); coded forms: 90 of 136, 123 of 180, 103 of 171 |
| P10 | coded share by **meaning** (a meaning is coded if **any** of its forms is) on the meanings shared with Tolaki 674 | 42 comparison lists: median 48.4 % against Tolaki's 38.3 % on the same meanings; 3 lists at or below. Dialect lists on the same 133 meanings: Asera 40.6, Konawe 50.4, Laiwui 36.1, Mekongga 46.6, Wiwirano 39.1 % (Tolaki 38.3 %). Spearman of list length against coded share by form: −0.55 |
| P10b | the same with the **first form** of each meaning only (Konawe has more than one form for 86 of the 133 meanings, Mekongga for 62, list 674 for few) | 42 comparison lists: median 46.7 % against 38.3 %; 5 lists at or below. Dialect lists: Asera 40.6, Konawe 36.8, Laiwui 36.1, Mekongga 36.8, Wiwirano 39.1 %. The Konawe excess of P10 is an effect of synonyms |
| P11 | other conditionings | ≥ 1 consonant-letter cluster with list × letters × glottal mark held fixed: 1.38 [0.98, 1.93]; onset string with list × letters **after** the matched string held fixed: 2.51 [1.85, 3.40] |
| P12 | the classifier with the 44 same-language Muna forms relabelled as candidates | CV AUC 0.7265 → 0.7415 (form + meaning), 0.6717 → 0.6985 (form only); candidates 438 → 482 |

## What the tables show (numbers only; the wording of any conclusion is the PI's)

1. Cognate-set numbers and their absence: tables A, P2, P12 (5 of 11 loan-flagged forms coded; 44 Muna forms coded only
   through the second Muna list; the Sulawesi box of P6; the headline AUC does not rest on those 44).
2. Makasar against Bugis and Sa'dan Toraja: tables C, P1, P8, P9 (uncoded share 34.8 % against 23.4 % and 19.0 %; on
   meanings coded in both lists 70 against 69 of 110 and 70 against 71 of 115 in the PMP set; look-alikes of uncoded
   Makasar forms: 2 of 75 with the PMP form, 10 of 80 with a Bugis or Sa'dan Toraja form; 28 + 25 meanings for a specialist).
3. Tolaki: tables D, P5, P10, P10b (look-alikes within the subgroup; coded share by meaning, first form: list 674 38.3 %,
   dialect lists 36–41 %, median of the comparison lists 46.7 %; twelve look-alikes of the PMP form, about two by
   chance; sensitivity of the screen 48 %).
4. The inputs: tables E, F, P3, P4, P11 (at equal length the glottal mark and the action meaning hold; clusters are
   weaker; the onset string and the nasal input are not separable from length; "reduplication" and "final vowel" do
   not stand as separate properties).
5. AUC: table G (fold means 0.727 / 0.672; about ± 0.03).
## Limits

Counts on one snapshot of one database; breadth counts lists, not languages, and reflects ABVD's coverage and its
coders; "Sulawesi only" is a coordinate box; one language (Muna) has two lists that code each other; look-alikes are
a screen (sensitivity about one half); intervals treat forms or meanings as independent draws and, for the AUC, hold
the predictions fixed; four unadjusted paired tests; the post hoc counts were made after the results were known;
nothing here was tested against a pre-stated hypothesis. Every statement about a single form, about Sulawesi subgrouping or about what a compiler's
hyphen or final *k* means needs a historical linguist of Sulawesi (SIG G10).

## Files

`DESIGN.md` · `01_tabulations.py` · `02_posthoc_checks.py` (post hoc; `results/P*.csv`, `P_posthoc.json`) · `results/A_*.csv`, `B_pairwise_cognate_sharing.csv`, `C_*.csv`, `D_*.csv`,
`D_summary.json`, `E_strict_variants.csv`, `F_*.csv`, `G_auc_intervals.csv`, `H_leave_one_concept_out.csv`,
`anchor_checks.json`, `run_log.txt`

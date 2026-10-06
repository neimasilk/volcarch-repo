# E231 — DESIGN — what "coded" covers, how the lists compare, and stricter versions of four inputs (P8 revision)

**Written:** 2026-10-06, before the script was written or run. Amendments go in §6 with a date.
**Serves:** line 04 (P8 revision at *Oceanic Linguistics*). **Follows:** E227–E230.
**Not externally registered.** The definitions below were fixed before the tabulations were run, on the same day and by
the same project. That is all "fixed in advance" means here.

## 0. Why

An adversarial read of the documents written on 2026-10-06 (outline, READMEs of E229/E230, the note to the editor)
found the numbers sound and several *readings* of them unsupported:

1. "Coded" was being read as "inherited Austronesian" ("uncoded is an upper bound on anything non-Austronesian"; the
   middle class of the Makasar table "has Austronesian cognates"). ABVD cognate-set numbers also cover small local
   sets and shared loans.
2. The Tolaki result ("most uncoded forms recur within Bungku–Tolaki") was read as "ordinary inherited vocabulary".
   Recurrence in a subgroup says nothing about origin, and the comparison lists may be as thinly coded as Tolaki.
3. Four inputs are named after more than the code computes (nasal sequence, reduplication, prefix-like onset, length).
4. "Ends in a vowel" was explained by list membership without testing the obvious alternative (the final glottal mark).
5. No interval is given for any AUC or for the Makasar shares; the reviewer's question about Makasar's divergence from
   Bugis and Sa'dan Toraja is not answered by retention from PMP alone.
6. A step that the E228 design promised (leave-one-concept-out for the cross-language statistic) was not carried out.

**Known before this was written (not blind):** the adversarial reader's own unsaved tabulations, quoted here so that
nobody mistakes what follows for a blind test — 5 of the 11 loan-flagged forms are coded; of Makasar's coded-not-PMP
meanings roughly a third sit in sets attested in ≤ 5 lists; the Bungku–Tolaki comparison lists are 30–60 % coded;
about 12 Tolaki candidates resemble the PMP form for the same meaning; with a strict definition the nasal contrast is
about 17 % against 12 %; the letters-only length difference is about +0.93; in the three South Sulawesi lists the
candidate share is about 24 % (vowel-final), 26 % (other consonant), 39 % (final glottal mark).
E231 computes these properly, with saved code, and reports whatever comes out. **Nothing is tested; everything is a
count or an interval.** If a count contradicts a statement of the outline or of a README, the statement is changed.

## 1. Data and shared code

ABVD CLDF snapshot `experiments/E022_linguistic_subtraction/data/abvd/cldf/` (git 917c5a5): `forms.csv`,
`languages.csv`, `parameters.csv`, `cognates.csv` if needed. Six lists as before (27, 48, 166, 192, 226, 674);
loader, normalisation (`norm`), distances (`ned`, `d_stem`), features and CV from `E228/p8common.py` (imported).
A cognate set is identified by **(Parameter_ID, set number)**; a form may carry several numbers; a number followed by
`?` is a doubtful assignment (counted, and reported separately where it matters).
Sulawesi box as E228 DESIGN §4: lat −6.6…2.0, lon 118.5…125.6. Lists without coordinates and reconstructed
proto-languages are listed by name in the output and are **not** counted as "Sulawesi" unless their name says so
(e.g. Proto-Bungku-Tolaki); PMP (269) and PAn (280) are outside.

## 2. Tabulations (outputs in `results/`, CSV, UTF-8)

**A — what "coded" covers** (`A_coded_breadth_by_list.csv`, `A_middle_class_breadth.csv`, `A_loan_flagged.csv`).
Breadth of a set = number of distinct ABVD lists with a form in it. For a form with several sets: the widest.
Per list, for coded forms: n; median breadth; share with breadth ≤ 5; share with breadth ≤ 20; share whose set is
attested **only inside the Sulawesi box**; share carrying ABVD's loan flag. Then the same at the level of *meanings*
for the middle class of E228 S2 ("coded, not the PMP etymon"; S2's three classes are rebuilt from its definition and
must reproduce its counts): share of those meanings whose widest set has ≤ 5 lists, and whose set is Sulawesi-only.
The 11 loan-flagged forms with their cognate fields.

**B — how the six lists compare with each other** (`B_pairwise_cognate_sharing.csv`). For each of the 15 pairs: the
meanings present in both lists; share for which some form of one and some form of the other carry the same
(Parameter_ID, set number); Wilson 95 % interval; the same with doubtful assignments excluded; share where neither
list has any coded form.

**C — intervals and paired comparisons for the PMP table** (`C_retention_intervals.csv`, `C_paired_tests.csv`).
Wilson 95 % intervals for the three shares of E228 S2 in all six lists. Exact McNemar tests (binomial on the
discordant meanings) for Makasar against Bugis and against Sa'dan Toraja, for "retained from PMP" and for "uncoded";
four tests, unadjusted, reported as such.

**D — the Bungku–Tolaki comparison lists and the PMP look-alikes** (`D_bt_lists.csv`, `D_pmp_lookalikes_by_list.csv`,
`D_tolaki_pmp_lookalikes.csv`). For Proto-Bungku-Tolaki (780), the 42 comparison lists (ids 41, 875–908, 914–919,
972) and the five Tolaki dialect lists (909–913): name, Glottocode, author/source field, number of forms, share
coded; the number of distinct Glottocodes among the 42. For each of the six lists: the share of candidates that are a
look-alike (stem-tolerant distance ≤ 0.34; strict 0.25) of an ABVD PMP form for the **same** meaning, next to the
chance rate (the same comparison against the PMP forms of 50 randomly drawn other meanings, seed 231), and the same
for coded forms as a control. The Tolaki candidates concerned are listed (form, meaning, PMP form, distance) —
⚠ a screen for a specialist, not etymologies.

**E — stricter versions of four inputs** (`E_strict_variants.csv`), each summarised exactly like a row of E229 T6
(means or shares for candidates and coded forms, per-list differences, lists with the same sign, Mantel–Haenszel
odds ratio with 95 % CI for 0/1 properties, list-stratified mean difference with percentile bootstrap for counts;
2,000 resamples, seed 231), next to the original input:
- *nasal, strict*: after `ng`→ŋ and `ny`→ɲ, a nasal letter (m, n, ŋ, ɲ) immediately followed by a stop letter
  (p, b, t, d, c, j, k, g, q). A bare ŋ or the digraph `ng` alone does not count.
- *reduplication, strict*: a repeated two- or three-letter sequence only; the hyphen rule is reported separately
  ("contains a hyphen").
- *prefix-like, strict*: the original onset rule **and** at least six letters in the form; plus the count of
  flagged forms that have four letters or fewer.
- *length in letters*: alphabetic characters only, glottal marks (ʔ, apostrophe), hyphens, spaces and other marks
  not counted.

**F — the final segment** (`F_final_segment.csv`, `F_tae_final_k.csv`). For Bugis, Makasar and Sa'dan Toraja (the only
lists with consonant-final forms): candidate share for forms ending in a vowel / in a glottal mark / in another
consonant, per list and pooled; the Mantel–Haenszel odds ratio for "ends in a vowel" once forms ending in a glottal
mark are set aside. For all six lists: candidate share by final vowel within forms that carry / do not carry a
glottal mark. The Sa'dan Toraja forms ending in *k*, with meaning and label (reviewer 1 names *k* as one way of
writing the glottal stop) — ⚠ for a specialist.

**G — intervals for the AUC** (`G_auc_intervals.csv`). Out-of-fold P(candidate), mean of the 10 seeds, for FM25 and
F17 (FM25 must agree with the E228 release file). Pooled out-of-fold AUC; percentile 95 % intervals from 2,000
bootstrap resamples (seed 231) (i) of forms, (ii) of **meanings** (all forms of a drawn meaning go together — the
six lists share the meanings); out-of-fold AUC inside each list.

**H — the step of E228 that was not carried out** (`H_leave_one_concept_out.csv`). For the primary set of E228 S5
(candidates, numerals excluded; statistic T = mean normalised edit distance over same-meaning cross-list pairs):
T with each meaning removed in turn; the fifteen meanings whose removal raises T most, with their number of pairs
and their smallest pairwise distance. No new permutation.

## 3. Anchors (the script checks them and stops if one fails; nothing is adjusted)

1,357 forms, 438 candidates · E228 S2 counts: Makasar 79 / 52 / 70 of 201; Bugis 97 / 57 / 47 of 201; Sa'dan Toraja
96 / 66 / 38 of 200 · E228 S5 primary set: 440 pairs, T = 0.8357, 4 look-alike pairs · loan flags: 11 forms, 6 of them
uncoded · E229 T6 values for the four original inputs (within 0.001) · FM25 out-of-fold probability against the
release file (≤ 0.0006).

## 4. Units and direction

A, D-lists, E, F, G-forms: form. A (middle class), B, C, H: meaning. In E and F a positive difference or an odds
ratio above 1 means "more often among candidates".

## 5. Limits stated in advance

Breadth depends on how many lists ABVD holds for a region and on who coded them; a narrow set is not thereby a loan,
and a wide set is not thereby inherited. A look-alike is a mechanical screen. Intervals treat forms (or meanings) as
independent draws, which they are not within a list. Thirteen files of counts: no claim is made from any single one
without the others.

## 6. Amendments

(none)

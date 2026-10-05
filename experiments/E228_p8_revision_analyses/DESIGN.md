# E228 — DESIGN (pre-registration) — analyses for the P8 revision at *Oceanic Linguistics*

**Frozen:** 2026-10-05, before any of the outcomes below was computed. Amendments, if any, go in §9 with a date.
**Serves:** line 04 (P8). **Follows:** E227 (audit of the reviewed manuscript's numbers).
**Rule:** the result is reported whichever way it falls. If a result weakens a claim of the manuscript, the claim is
downgraded (SIG "banned move": no rewording of a central critique).

## 0. What was already known when this was frozen

From E227 (`experiments/E227_p8_g1_blind_rederivation/results/claims_audit.csv`), not outcomes of E228:

- The label used by every published model = "form has no cognate-set assignment in ABVD" (438 of 1,357 forms).
  Table 1 of the manuscript uses a smaller set (356) made by dropping 15 concepts wholesale and 7 loan-tagged forms.
- Glottal marking by source: Bugis 21 % (ʔ), Makassar 36 % (ʔ), Tae' 22 % (ʔ), Tolaki 14 % (apostrophe), Wolio 2 %,
  Muna 0 %. Within-language association of the marker with the label: Mantel–Haenszel OR 2.67 [1.88, 3.79].
- The published consensus (κ = 0.61, 266 forms) used in-sample probabilities. The published expansion fed the model two
  language-level inputs. The published cross-language test compared a mean of 20 with single draws from mostly cognate
  vocabulary.
- Design facts read from ABVD without looking at outcomes: the PMP list (id 269) has 261 forms / 201 concepts, all
  cognate-coded; ABVD holds a Proto-Bungku-Tolaki list (id 780, 205 forms, 73 coded) and 45 Bungku–Tolaki wordlists.

## 1. Data, label, models (common to all sections)

- Data: ABVD CLDF snapshot `experiments/E022_linguistic_subtraction/data/abvd/cldf/` (lexibank/abvd, git 917c5a5,
  2025-10-07). Six lists: Muna 27, Bugis 48, Makassar 166, Wolio 192, Tae' 226, Tolaki 674.
- **Label (one definition, used everywhere):** *candidate* = form with an empty ABVD `Cognacy` field; *coded* = any
  cognate-set assignment. This is the label the published models were trained on. Forms that ABVD flags as loans are
  kept and carry a flag. The 15-concept list and the string-matched loan lists of E022 are **not** applied (E227 A13,
  A15); a sensitivity column marks the 356-form Table-1 set.
- Features and XGBoost settings: exactly as E027 (300 trees, depth 4, learning rate 0.05, no class weighting).
  **Primary model = 25 features** (17 form-based, 8 semantic; no language-level input). Secondary = 26 features
  (adds the language-identity code), the manuscript's headline.
- CV protocol as published: stratified 5-fold × 10 seeds (`random_state = 7·seed + 13`); LOLO = each language held out.

## 2. S1 — glottal-stop orthography (reviewer 1, point 9)

Question: does the result depend on how the source writes the glottal stop?

Conventions applied to the form strings before feature extraction (all six lists):
V0 as published · V1 every glottal marker → `q` · V2 → `k` · V3 marker deleted (unwritten) ·
V4 "geminate": marker before a consonant → copy of that consonant, elsewhere deleted · V5 every marker → `ʔ`
(removes the difference between apostrophe and ʔ in the cluster count) · V6 strings as V0 but the glottal feature
removed from the model.

Outcome: CV AUC and LOLO mean AUC of the 25- and 26-feature models under each convention; Δ against V0.

Decision rule (25-feature CV AUC, the largest |Δ| over V1–V6):
- ≤ 0.010 → discrimination does not depend on the glottal convention; the manuscript's robustness sentence may stand
  for *discrimination*, and the text must still say that the glottal property is visible only where the source marks it.
- > 0.020 → the sentence "robust to orthography" is withdrawn for this property.
- between → reported as partial dependence.
Whatever the outcome, "glottal stop" stays in the profile only as "glottal marking in the four sources that write it".

## 3. S2 — Makasar: retention from PMP versus "candidate" share (reviewer 1, point 2)

Unit = **concept (meaning)**, not form. Base = concepts for which both the language and the PMP list (269) have an entry.
- *PMP-retained*: at least one form of the language for that concept shares an ABVD cognate set with a PMP form for
  that concept (primary: `Doubt = false` only; sensitivity: doubtful assignments counted).
- *coded, not PMP*: at least one form is cognate-coded, none shares a set with PMP.
- *uncoded*: no form for the concept has any cognate-set assignment.
Reported for all six languages, with exact counts. Secondary: the same against PAn (280).

Decision rule: if 100 − retention for Makasar lies within ±10 points of the 62 % that the reviewer quotes from the
literature, the two figures are the same kind of quantity and the manuscript's 30.9 % is a different, narrower one;
the revision states both and the size of the middle class. If it lies outside, the difference is reported with the
likely reasons (210- vs 200-item list, treatment of synonyms, later coding). In neither case is the published 62 %
re-derived or disputed: the source must be read by the PI first.
Second question (the reviewer's): is Makasar set apart from Bugis and Tae'? Reported as the three South Sulawesi rows
side by side; no test (three languages).

## 4. S3 — low-level inheritance and "documentation gaps" (reviewer 1, points 1, 7, 13)

Question: how many candidate forms have a same-meaning look-alike in closely related lists, i.e. are candidates only
with respect to *higher-level* cognate coding?

- Normalisation: lower case; `*`, spaces, hyphens, apostrophes, ʔ and bracketed material removed.
- Stem-tolerant distance: `d(a, b) = min over i, j ∈ {0..3}` of the normalised edit distance between `a[i:]` and
  `b[j:]`, both remainders ≥ 3 characters. **Look-alike** = `d ≤ 0.34` (primary); 0.25 and 0.50 reported.
- Chance rate: the same comparison against the forms that the same lists give for 50 randomly drawn *other* concepts
  (seed 228); reported next to every observed rate. Excess = observed − chance.
- **S3a (primary), Tolaki 674:** comparison sets (i) Proto-Bungku-Tolaki (780); (ii) the Bungku–Tolaki lists that are
  not Tolaki (ids 41, 875–908, 914–919, 972); (iii) the five Tolaki dialect lists 909–913 (reported separately — same
  language, tells only whether the word is attested independently). Reported for candidate and for coded forms
  (coded forms = positive control).
- **S3b (secondary), all six languages:** comparison set = every list inside the box lat −6.6…2.0, lon 118.5…125.6
  with a different Glottocode, Bajo lists excluded; 10 random concepts for the chance rate.

Decision rule: if the excess look-alike rate of Tolaki candidates against (i) ∪ (ii) is ≥ 20 points, the revision
states that a substantial share of Tolaki "candidates" is inherited at the Bungku–Tolaki level and uncoded in ABVD
(the reviewer's second reading of "documentation gap"); the sentence about "under-documentation" is replaced by that
number. If < 20 points, the revision says the look-alike check found little low-level sharing by this crude measure.
Either way the list goes into the release file for specialists; the mechanical look-alike is a screen, not an etymology.

## 5. S4 — agreement between label and classifier, out of fold (reviewers 1 pt 11–12, 2 "E022 label", "false positive")

- `P_cand(form)` = mean, over the 10 seeds, of the probability given by the fold model that did **not** see the form.
  Primary 25 features; secondary 26; also the LOLO probability (model never saw the language).
- Threshold 0.5 as published. Reported: the four cells, Cohen's κ, cells per language, and the AUC.
- Expectation written down beforehand: κ will be lower than the published 0.61, because 0.61 was in-sample.
  **Whatever the value, the out-of-fold figure replaces the published one.** The set "candidate and P ≥ 0.5" replaces
  the 266 forms wherever a set is needed.
- Examples for each cell: the five forms furthest from 0.5 per cell, plus the same per language, with concept and form.

## 6. S5 — are same-meaning candidates more alike across languages than different-meaning candidates?

The manuscript's negative result ("no shared word families", p = 0.569), re-tested with a permutation control
(line 04 rule 2).

- Statistic T = mean normalised edit distance (normalisation as §4, no stem tolerance) over all cross-language pairs
  of candidate forms with the same concept. Second statistic N = number of such pairs with distance ≤ 0.34.
- Null: within each language, concept labels are permuted among its candidate forms (10,000 permutations, seed 228);
  pair counts are preserved. One-sided p = share of permutations with T ≤ observed (N ≥ observed).
- Sets: (a) all candidates; (b) without the 14 numeral concepts (the manuscript itself shows the decimal compounds are
  built from inherited numerals); (c) = (b) without the 15 concepts of the E022 list; (d) candidates with out-of-fold
  P ≥ 0.5. **Primary = (b).** Positive control: the same test on coded forms (must come out near p = 0).
- Secondary split of (b): pairs inside South Sulawesi (Bugis–Makassar–Tae') versus all other pairs.

Decision rule: p ≥ 0.05 on (b) → the negative result stands and is reported with this test instead of p = 0.569.
p < 0.05 on (b) → the manuscript's statement is downgraded to "a small shared component", the concepts carrying it
are listed (leave-one-concept-out), and the South Sulawesi split says whether it is low-level inheritance.

## 7. S6 — the 16 additional languages, scored without language-level inputs

- The 25-feature model trained on the six lists is applied to the same 16 lists as the manuscript.
  Reported per language: share of candidates (label), mean P, AUC against the language's own label.
- Group contrast Sulawesi (8) vs western Indonesia (6) on mean P: exact permutation test of the difference in means
  (all C(14,6) splits), two-sided.
- Because a language with more uncoded forms may simply be a later, less coded list, the same contrast is reported on
  **mean P of coded forms only** (forms the database does treat as inherited): if the "pattern" appears there too, it
  reflects orthography or phonology of the list, not candidates.

Decision rule: the sentence about geographic patterning stays only if the contrast has p < 0.05 **and** the AUC
against the own label is ≥ 0.60 in at least 12 of 16 lists. Otherwise it is removed from the abstract and reported as
not supported.

## 8. Deliverables

`results/` — one JSON per section and the tables; `release/` — `p8_forms_all.csv` (1,357 rows) and
`p8_candidates.csv` (438 rows) with ABVD ids, flags and the out-of-fold probability, for the data release asked for by
reviewer 1 (point 13). Release only after the PI has looked at them.

## 9. Amendments

**A1 — 2026-10-05, added while script 01 was still running section S1; no S7 number had been computed.**
Reason: reviewer 1, point 10 (a semantic feature cannot be part of a *phonological* fingerprint) cannot be answered
from S1–S6. E227 (rows B04, E03) shows that the manuscript's headline model has 26 inputs of which 8 are semantic and
1 is the language's identity code, and that the identity code is its strongest input; the abstract calls it "trained
exclusively on phonological features". No model with form-based inputs only was ever run.

**S7 — what each group of inputs contributes** (script `02_feature_groups.py`, same data, label and protocol as §1):
- form-only: the 17 inputs computed from the written form (9 + 8 initial-segment indicators);
- meaning-only: the 8 inputs computed from the concept (list flag + 7 domain indicators);
- language-only: the identity code alone (CV only; LOLO is undefined for it);
- form + meaning (25) and form + meaning + identity (26) repeated for reference.
Outcome: CV AUC ± SD (seed means), LOLO mean AUC and per-language AUC.

Decision rule: the words "phonological features only/exclusively" may describe **only** the form-only model, and any
AUC quoted next to those words must be that model's. If its CV AUC is < 0.65 (the manuscript's own line), "phonological
properties alone distinguish…" is downgraded to "carry a weak signal". If meaning-only ≥ form-only, the revision says
that the meaning of a concept predicts the label at least as well as the shape of the word.

**A2 — 2026-10-05, POST HOC: written after the S5 result was seen (p = 0.0001 on the primary set). Exploratory;
it cannot rescue or sharpen the pre-registered outcome, only help to read it.**
Reason: within-language permutation of concept labels also destroys word class. Forms for the same meaning tend to
carry the same affixes across these languages (stative *ma-*, verbal *mo-/me-/maC-*), so same-meaning pairs can be
closer than permuted pairs through shared morphology alone. Among the four non-numeral look-alike pairs, one
('heavy': *ma-tanəʔ* / *matamo*) is of that kind.

**S5x** (script `03_permutation_stratified.py`), on set (b) — candidates without numerals:
- permutation within language × semantic domain (the seven domains of the model), which keeps action words with
  action words and quality words with quality words;
- the same with the stem-tolerant distance of §4 (up to three leading characters ignored);
- both statistics (T, N), 10,000 permutations, seed 228.
Reading (not a decision rule): if the excess similarity disappears under stratification and stem tolerance, the shared
component of S5 is mainly shared morphology; if it remains, it is carried by roots.

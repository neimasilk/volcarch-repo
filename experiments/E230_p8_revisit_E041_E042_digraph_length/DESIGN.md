# E230 — DESIGN (pre-registration) — the digraph ("IPA") and length checks of P8, re-derived and re-run for the models of the revision

**Frozen:** 2026-10-06, before the script was written or any outcome computed. Amendments go in §8 with a date.
**Serves:** line 04 (P8 revision at *Oceanic Linguistics*). **Revisits:** E041 (IPA approximation), E042 (syllable count).
**Rule:** the result is reported whichever way it falls. If it weakens a sentence of the manuscript, the sentence is
downgraded or withdrawn (SIG: no rewording of a valid critique).

## 0. Why

Section 3.4 of the reviewed manuscript (tex 347–359) reports that converting digraphs to single symbols, replacing
character count by "syllable" count, and dropping the length input all leave the classifier unchanged. The abstract
(tex 42) concludes from this that "the model detects phonological patterns rather than orthographic artifacts"; the
same numbers are used at tex 379, 449 ("does not depend on form length at all"), 577 and 606.
Reviewer 1 questions exactly this section (point 8: the digraph list; point 9: orthographic conventions).

What was known when this was frozen:
- E227 checked only the *counts* of that section (rows I01–I04: 75 forms changed, 54 in Muna; the sentence omits Muna
  *dh* and Wolio *gh*, which the code converted). **The AUC values of tex 352–359 were never re-derived.**
- Their baselines (0.772, 0.768) equal none of the models of Tables 2–4 (0.760, 0.763, 0.727), so they come from some
  other model variant or protocol. Which one is not yet known.
- E228 S1 showed partial dependence on the *glottal* convention (up to −0.016 AUC). Digraphs and length were not in E228.
- E042's "no length" variant removed `form_length` only; `n_vowels` and `vowel_ratio` stayed in the model.
- From the E041 README: mapping `ng`→ŋ, `ny`→ɲ (all lists); Muna `gh`→ɣ, `bh`→β, `dh`→ð; Wolio `gh`→ɣ.
Not known: any outcome below.

## 1. Part A — audit of the submitted numbers (no hypothesis)

Re-derive, by reading `experiments/E041_ipa_validation/01_ipa_approximation.py`,
`experiments/E042_syllable_validation/01_syllable_count.py` and their stored summaries, and by re-implementing them
here (their scripts are **not** re-run in place and their result files are not touched):
CV AUC 0.772 → 0.774; LOLO mean 0.724 → 0.733; "all six ≥ 0.65"; Muna +0.042; CV 0.768 → 0.769; LOLO 0.722 → 0.728;
"no length" CV 0.769, LOLO 0.732.
For each: the value reproduced, **which inputs and which cross-validation protocol produced the baseline**, and a
verdict (MATCH / MISMATCH / NOT-AS-DESCRIBED / UNSUPPORTED), in `results/A_audit.csv`.

## 2. Part B — the same questions for the models of the revision

Data, label, loader, feature code, XGBoost settings and CV protocol exactly as E229 §1 (imported from
`E228/p8common.py`; `y = 1` coded; 5-fold × 10 seeds with `random_state = 7·seed + 13`; LOLO). Models: **FM25**
(primary) and **F17** (form only — the model in which a change of the written form should show most).

Variants (applied to the form strings **before** feature extraction; every input is then recomputed by the same code):

| Id | Variant |
|---|---|
| V0 | as in ABVD (baseline; must reproduce 0.7265 / 0.6717) |
| D1 | digraphs → single symbols, mapping exactly as E041 (above) |
| D2 | as D1, but the initial-letter class is taken from the original spelling (so that only length, vowel share and cluster counts change, not the "begins with b" input for Muna *bh-*) |
| L1 | `form_length` replaced by the number of vowel groups (E042's "syllable count") |
| L2 | `form_length` removed (E042's "no length"; `n_vowels`, `vowel_ratio` stay) |
| L3 | `form_length` **and** `n_vowels` removed (both size inputs; `vowel_ratio` stays) |

Outcomes per variant and model: CV AUC, Δ against V0; the 10 **paired** seed differences (same folds) → their mean
and SD; LOLO mean and per list, Δ; number of forms changed per list (D1), and per digraph.

## 3. Decision rules (fixed)

Bands as in E228 S1, on the CV AUC Δ, read for FM25 and for F17 separately:
- |Δ| ≤ 0.010 → *no dependence shown at this resolution*. The text may then say that discrimination is unchanged under
  this conversion — **and nothing more**: an unchanged AUC under a five-per-cent change of the strings does not show
  that the model "detects phonology rather than orthography" (E228 S1 already shows one orthographic dependence).
- |Δ| > 0.020 → the model depends on this convention; the robustness sentence is withdrawn for it.
- between → partial dependence.
Noise: a Δ is called *not distinguishable from zero* when |mean of the paired seed differences| < 2 × SD / √10.

Specific sentences:
- "does not depend on form length at all" (tex 359, 449) is tested by **L3**, not by L2. If Δ(L3) ≤ −0.010 in FM25 or
  in F17, the sentence is withdrawn. If L2 is unchanged but L3 is not, that is reported as: the length information
  was still in the model through the vowel count.
- "Muna shows the largest improvement, so digraphs were adding noise" (tex 354): reported as the LOLO Δ for Muna under
  D1 and D2 in both models, next to the Δ of the other five lists. One list, no test: if Muna's Δ is not the largest
  positive one in **both** models, the sentence is withdrawn.
- The digraph list of tex 348 is replaced by the list the code applies, with counts (reviewer 1, point 8).

## 4. Anchors

V0: FM25 0.7265, F17 0.6717 (±0.0005); LOLO means 0.7008, 0.6413. D1 counts: 75 forms changed; Muna 54, Tolaki 20,
Tae' 1; two Muna forms with *dh* (E227 I01, I03). If an anchor fails the script stops; nothing is adjusted.

## 5. Units and direction

Unit = form. Δ = variant − baseline, so a negative Δ means the variant discriminates worse.

## 6. Domain caution (flag, not a result)

The symbols of the E041 mapping are placeholders. Whether Muna *bh*, *dh*, *gh* are a fricative, an implosive or a
dental stop is a fact of Muna phonology that reviewer 1 knows and this project does not (the reviewer describes *dh*
as a dental stop; E041 used fricative symbols). No output of E230 depends on which symbol is used — only on a digraph
becoming one consonant character — and the README must say so. ⚠ For the PI / a specialist before anything is printed.

## 7. Status rule and limits

SUCCESS = Part A and Part B ran and the anchors reproduce; the *finding* may go against the manuscript.
The conversion touches about 5 % of the forms, almost all in Muna and Tolaki; a null result here is weak evidence.
Six variants × two models: no correction for multiplicity is claimed; the bands above are descriptive.

## 8. Amendments

- **A1 (2026-10-06, wording only; after the results).** §6 says the reviewer describes Muna *dh* as a "dental
  stop". The reviewer's term is a voiced **interdental** stop (checked against the report). The frozen text above is
  left as written; the README uses the reviewer's word. No number depends on it.

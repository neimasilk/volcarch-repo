# E230 — the digraph ("IPA") and length checks of P8, re-derived and re-run for the models of the revision

**Status:** SUCCESS (ran as pre-registered; all 10 anchor checks reproduce). The *finding* is mixed and goes partly **against**
the submitted manuscript: two of its sentences are withdrawn by the pre-registered rules.
**Lines:** 04_language_text (P8 revision at *Oceanic Linguistics*). **Date:** 2026-10-06. **Revisits:** E041, E042.
**Design:** `DESIGN.md`, written before the script (same day; not registered externally). One amendment, A1 (2026-10-06, wording only: the reviewer's term for Muna *dh* is "interdental").
**Checked:** the baseline and the D1, L2 and L3 rows for both models (eight of the twelve rows of Part B; D2 and L1 were
not re-run) were re-derived by the orchestrator with a separate short script — identical to four decimals. That
script shares the loader, the feature code and the cross-validation routine of `p8common.py` with the builder's
script, so it checks the variant logic, not the features. The description of E041's baseline was checked by reading
`E041/01_ipa_approximation.py` lines 317–344.
**"Pre-registered" here means:** the decision rules were written down before the analysis was run, on the same day
and by the same project; nothing was registered externally.

## Hypothesis (questions and decision rules in `DESIGN.md` §2–§3)

Section 3.4 of the reviewed manuscript says that converting digraphs to single symbols, counting "syllables"
instead of characters, and dropping the length input leave the classifier unchanged, and concludes (abstract,
tex 42) that "the model detects phonological patterns rather than orthographic artifacts". Reviewer 1 asks about
exactly this section (points 8 and 9). E227 had checked only the counts of that section; E228 only the glottal stop.
**Part A:** do the printed numbers reproduce, and which model do they belong to? **Part B:** what do the same checks
give for the two models of the revision (FM25 = form + meaning, F17 = written form only)?

## Method and data

ABVD CLDF snapshot (git `917c5a5`), six lists, 1,357 forms; label, features, XGBoost settings and CV protocol as E229
(imported from `E228/p8common.py`). `01_digraph_length.py`; the E041/E042 scripts were read and re-implemented, **not**
run, and their result files were not touched. Variants: D1 digraphs → single symbols exactly as E041
(`ng`→ŋ, `ny`→ɲ in all lists; Muna `gh`, `bh`, `dh`; Wolio `gh`); D2 the same with the initial-letter inputs kept
from the original spelling; L1 length = number of vowel groups; L2 `form_length` removed; L3 `form_length` **and**
`n_vowels` removed.

## Result

### Part A — the printed numbers reproduce, but they are not numbers of "Model B" (`results/A_audit.csv`)

All fifteen values (0.772 → 0.774, LOLO 0.724 → 0.733, Muna +0.042, 0.768 → 0.769, LOLO 0.722 → 0.728, "no length"
0.769 / 0.732) reproduce to the printed precision (two differ by 0.0001 in the fourth decimal). What they are:

1. **A 26-input model that includes the language-identity code**, cross-validated with other fold seeds
   (`random_state = seed`) than the published protocol (`7·seed + 13`). That is why the baselines 0.772 and 0.768
   match no model of Tables 2–4. The same 26 inputs under the published folds give 0.763; FM25 gives 0.727.
2. **E041's "orthographic" baseline is a hybrid.** Only length, cluster count and the initial-letter class were taken
   from the original spelling; vowel count, vowel share, final vowel, glottal, nasal sequence, reduplication and
   prefix-like onset came from the *converted* string in both arms. With every input from the original spelling the
   baseline AUC is the same (0.772), but **Muna's gain is +0.015, not +0.042, and Makasar gains more (+0.033)** — in a
   list where no form was changed.
3. **E042's "no length" model kept the vowel count and the vowel share**, so length information stayed in.
4. The two inferences drawn from these numbers (rows A16, A17) have no test behind them.

Verdict by row: 13 NOT-AS-DESCRIBED, 2 MATCH, 2 UNSUPPORTED.
(The structure of E041's two arms — point 2 — was confirmed by reading its code. The "every input from the original
spelling" re-run that gives +0.015 / +0.033 is the builder's additional decomposition; it was **not** re-derived a
second time and should be quoted, if at all, with that caveat.)

### Part B — the same checks for FM25 and F17 (`results/B_variants.csv`, `B_lolo_by_list.csv`)

Baselines: FM25 0.7265 (LOLO 0.7008), F17 0.6717 (LOLO 0.6413). Δ = variant − baseline.

| Variant | FM25: Δ CV AUC | larger than fold-partition noise? | Δ LOLO | F17: Δ CV AUC | larger than noise? | Δ LOLO | Band (CV) |
|---|---|---|---|---|---|---|---|
| D1 digraphs → single symbols | −0.0003 | no | +0.005 | +0.0028 | yes (small, positive) | +0.005 | no dependence shown |
| D2 same, initial letter kept | −0.0002 | no | +0.004 | +0.0047 | yes (small, positive) | +0.008 | no dependence shown |
| L1 vowel groups instead of characters | −0.0003 | no | −0.009 | +0.0031 | no | +0.002 | no dependence shown |
| L2 `form_length` removed | +0.0001 | no | +0.001 | −0.0012 | no | +0.004 | no dependence shown |
| **L3 `form_length` and `n_vowels` removed** | **−0.0138** | yes | **−0.031** | **−0.0234** | yes | **−0.040** | FM25 partial · F17 depends |

Forms changed by D1: 75 of 1,357 (Muna 54, Tolaki 20, Sa'dan Toraja 1; none in Bugis, Makasar, Wolio). By digraph:
Muna `gh` 29 forms, `bh` 14, `dh` 2 (*akaradhaa* 'to work', *idho* 'green'), `ng` 10; Tolaki `ng` 20; Sa'dan Toraja
`ng` 1; `ny` and Wolio `gh` convert no form (`results/B_forms_changed.csv`).

### Decision rules applied (`results/decisions.json`)

- **Digraph conversion: no dependence shown at this resolution** (|Δ| ≤ 0.005 in both models). By the rule fixed in
  advance the text may say that discrimination is unchanged under this conversion — and nothing more. The conversion
  touches 5.5 % of the forms; with E228 S1 (glottal convention: up to −0.016) the sentence "phonological patterns
  rather than orthographic artifacts" cannot stand.
- **"Does not depend on form length at all" (tex 359, 449): WITHDRAWN.** Removing `form_length` alone changes nothing
  because the vowel count carries the same information; removing both costs 0.014 (FM25) and 0.023 (F17) of AUC, and
  0.031 / 0.040 for a held-out list (Muna: −0.071 and −0.099). Size is part of what the classifier uses — which is
  what T6 of E229 shows directly (candidates are about one character longer in every list).
- **"Muna shows the largest improvement, so digraphs were adding noise" (tex 354): WITHDRAWN.** Muna's held-out AUC
  rises most in FM25 (+0.017) but **falls** in F17 (−0.010), where Makasar (+0.044) and Bugis (+0.025) rise most —
  two lists in which no form changed. The per-list movements are noise of a single held-out fit, not evidence about
  digraphs.
- The digraph list of tex 348 is replaced by the list above (reviewer 1, point 8).

## Conclusion

The submitted robustness section reported real numbers for a model that is neither the headline model nor either
model of the revision, against a baseline that was not what the text says. Re-run cleanly for the revision models,
the digraph check is a weak null (unchanged, on 5 % of the forms) and the length check **fails as worded**: the
classifier does depend on word size. The honest statement for the revision is narrow: converting five digraphs (four of which occur in these lists) to
single symbols does not change discrimination; the glottal convention changes it a little (E228 S1); word size is one
of the things the classifier uses. None of this shows that the model "detects phonology rather than orthography".

## Limits and flags

Five variants (and the baseline) × two models, no multiplicity correction; the bands are descriptive. The held-out-list deltas are
single fits without intervals. The "noise" column compares a difference with the variation between the ten fold
partitions only; it is not a sampling test (the +0.003 of D1 for the form-only model passes it at about 2.4 standard
errors and should not be read as an effect).
**For the answer to reviewer 1's question about *dh*:** the reviewer supposes that no *dh* form survived the removal
of loanwords. No loanword was removed from the data the classifier saw; both *dh* forms are in it, both are
cognate-coded, and one of them (*akaradhaa* 'to work') carries ABVD's loan flag. ⚠ **Domain flag:** the symbols ɣ, β, ð are placeholders taken over from E041; whether
Muna *bh*, *dh*, *gh* are fricatives, implosives or stops is a fact of Muna phonology that reviewer 1 knows
and this project does not (the reviewer describes *dh* as a voiced interdental stop). No number here depends on the symbol — only
on a digraph becoming one consonant character — but nothing about Muna phonology may be printed from this file
without a specialist (SIG G10).

## Files

`DESIGN.md` · `01_digraph_length.py` · `results/A_audit.csv` · `B_variants.csv` · `B_lolo_by_list.csv` ·
`B_forms_changed.csv` · `decisions.json` · `anchor_checks.json` · `run_log.txt`

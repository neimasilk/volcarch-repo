# E130: Substrate Detection Interpretability

**Date:** 2026-03-30
**Status:** FAILED — method invalid: the label is among the inputs (ledger C075, downstream audit C050, 2026-10-19)
**Paper:** P8 (revision ammo), P19
**Layer:** L4 (Cosmological Overwrite)

> ⚠ **Correction 2026-10-19 (C050 audit, ledger C075) — read before anything below.** The original status line was
> "SUCCESS". It is withdrawn.
> - **Label in the inputs.** The inputs include `n_cognate_sets` and `max_cognate_set_size`, which are 0 exactly when the
>   label is 0 (cross-tab 438/0, 0/919). `concept_residual_rate` is derived from the label too. This is E027's
>   "Model A" input set (AUC 1.000, already reported as circular in E027), minus the two language-level inputs.
> - **In-sample.** The forest is fitted and scored on the same rows (script l.46, l.57).
> - **So the 438 "high-confidence substrate words" are exactly the 438 ABVD forms without a cognate-set number** (same
>   language–concept–form set; lowest probability 0.975). The classifier adds nothing. "Substrate" is a renaming of
>   "uncoded" (C044, C055).
> - **What remains** is descriptive: uncoded vs coded forms by domain and by letter features.
>   - Action meanings and glottal marks survive at equal length (C057, C066).
>   - Clusters and nasals are not separable from length (C057, C064).
> - **Muna 15.5 %** is an effect of the second Muna list (Wuna) in ABVD; uncounted, it is 35.6 % (C056). "Differential
>   Austronesian penetration" is withdrawn.
> - **Withdrawn without replacement:** the section "What Substrate Words Reveal About Pre-Indic Life", the "INVISIBLE
>   CIVILIZATION" paragraph and the link to E040 (F9).
> - Full record: `docs/research_notes/DOWNSTREAM_AUDIT_C050_20261019.md` §3.

---

## Hypothesis

Beyond detecting substrate presence (E027: AUC=0.76), the substrate WORDS themselves reveal what pre-Indic Nusantara culture was like — what people did, what they named, what mattered to them.

## Method

Re-trained RandomForest (200 trees) on E027 features matrix (1,357 lexical forms, 6 Sulawesi languages). Extracted high-confidence substrate words (prob > 0.8) and analyzed by semantic domain, language, and phonological pattern.

## Results

### 438 Substrate Words Identified

| Domain | Total | Substrate | Rate |
|--------|:---:|:---:|:---:|
| **ACTION** | 392 | **177** | **45.2%** |
| QUALITY | 177 | 67 | 37.9% |
| GRAMMAR | 157 | 59 | 37.6% |
| NUMBER | 84 | 27 | 32.1% |
| NATURE | 178 | 42 | 23.6% |
| BODY | 163 | 30 | 18.4% |
| OTHER | 206 | 36 | 17.5% |

**ACTION verbs are the most substrate-rich domain (45.2%).** These words describe: cooking, hunting, cutting, tying, hitting, seeing, thinking — the vocabulary of daily life in a pre-Indic organic civilization.

### Phonological Signature

| Feature | Substrate | Cognate | Direction |
|---------|:---:|:---:|---|
| Glottal stop | 23.5% | 11.6% | **Substrate 2x more glottal** |
| Consonant clusters | 0.48 | 0.32 | Substrate more complex |
| Word length | 6.10 | 5.21 | Substrate longer |
| Nasal cluster | 30.4% | 23.7% | Substrate more nasal |

### Language Variation

Tolaki has highest substrate rate (64.1%), Muna lowest (15.5%). This suggests differential Austronesian penetration across Sulawesi.

### What Substrate Words Reveal About Pre-Indic Life

The 438 substrate words paint a picture of daily life BEFORE Indianization:
- **Cooking, hunting, cutting, tying** — subsistence activities
- **Forest, lake, path, grass** — landscape interaction
- **Husband, wife, father, woman** — kinship terms
- **Rope, thatch/roof, dog, fish** — material culture
- **Wet, red, heavy, warm** — environmental qualities
- **Numbers (1000, 100, 8, 6)** — counting system partially non-Austronesian

This is the INVISIBLE CIVILIZATION: a culture with its own kinship system, subsistence vocabulary, environmental knowledge, and number system — all replaced or overlaid by Austronesian and then Sanskrit terminology.

## Conclusion

**SUCCESS.** 438 substrate words cataloged with semantic, phonological, and linguistic analysis. ACTION domain dominance (45.2%) is consistent with E040's finding that 63.4% of inscription mentions are organic materials — the substrate and the inscriptions describe the SAME organic civilization from different perspectives. Pre-Indic culture was action-oriented, environmentally embedded, and linguistically distinct (higher glottal frequency, more complex consonant clusters).

## Scripts

- `substrate_interpretability.py`

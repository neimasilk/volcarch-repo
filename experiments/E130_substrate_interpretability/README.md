# E130: Substrate Detection Interpretability

**Date:** 2026-03-30
**Status:** SUCCESS
**Paper:** P8 (revision ammo), P19
**Layer:** L4 (Cosmological Overwrite)

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

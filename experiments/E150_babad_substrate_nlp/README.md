# E150: Babad Tanah Jawi Substrate NLP

**Date:** 2026-03-30  
**Status:** SUCCESS  
**Paper:** P8, P19  
**Layer:** L4 (Cosmological Overwrite)

## Hypothesis

A non-DHARMA corpus should still show a strong non-Sanskrit Javanese backbone. If DHARMA monoculture is the problem, a Javanese chronicle should break it cleanly and reveal a different domain stratification from E130.

## Data

- 25 cached HTML chapters of *Babad Tanah Jawi* from Ki-Demang
- 25,743 tokens, 4,455 unique tokens after normalization
- E058 curated kakawin lexicon for native vs Sanskrit anchors
- E130 substrate domain profile for comparison

## Method

1. Parse chapter bodies from cached HTML.
2. Normalize Romanized Javanese orthography and tokenize.
3. Classify the top 150 tokens by frequency using a conservative hybrid lexicon:
   - E058 native and Sanskrit vocabulary
   - manual Javanese function-word / chronicle lexicon
   - explicit foreign-colonial token list
4. Compare native-token domain distribution to E130 substrate domains.

The top-150 window covers **13,669 tokens**, or **53.1%** of the whole corpus, so the analysis targets the high-frequency lexical backbone rather than the long tail.

## Key Results

### 1. Native Javanese Dominates the Chronicle Backbone

Within the top-150 token window:

| Class | Token mass | Share |
|------|-----------:|------:|
| **Native Javanese / non-Sanskrit** | **11,462** | **83.9%** |
| Foreign / colonial | 1,281 | 9.4% |
| Sanskrit | 898 | 6.6% |
| Unknown | 28 | 0.2% |

This is the core result: the chronicle is not lexically Sanskrit-dominated. Sanskrit survives as a thin elite overlay on top of a much thicker native Javanese matrix.

### 2. Domain Stratification is NOT the Same as E130

Native-token domain profile in *Babad Tanah Jawi*:

| Rank | E150 native domain | Share |
|-----:|--------------------|------:|
| 1 | **GRAMMAR** | **55.7%** |
| 2 | OTHER | 19.8% |
| 3 | ACTION | 10.7% |
| 4 | QUALITY | 5.2% |
| 5 | NATURE | 4.5% |
| 6 | NUMBER | 3.5% |
| 7 | BODY | 0.6% |

E130 substrate profile:

| Rank | E130 substrate domain | Rate |
|-----:|------------------------|-----:|
| 1 | **ACTION** | **45.2%** |
| 2 | QUALITY | 37.9% |
| 3 | GRAMMAR | 37.6% |
| 4 | NUMBER | 32.1% |
| 5 | NATURE | 23.6% |
| 6 | BODY | 18.4% |
| 7 | OTHER | 17.5% |

So E150 does not merely repeat E130. The substrate-heavy comparative lexicon is **ACTION-first**; the chronicle is **GRAMMAR-first** and polity-heavy. That is exactly what should happen if Indianization is a register overlay rather than a total replacement.

### 3. Top Native Content Terms

Highest-frequency native non-grammar terms include:

- `wong` (452)
- `tanah` (326)
- `tahun` (305)
- `dadi` (182)
- `jawa` (170)
- `padha` (162)
- `para` (150)
- `banget` (137)
- `bisa` (117)
- `akeh` (116)
- `menyang` (115)
- `kutha` (98)
- `bangsa` (97)
- `perang` (72)
- `jumeneng` (69)

This is a chronicle vocabulary of people, land, movement, polity, quantity, and conflict, not a Sanskrit-only court register.

## Interpretation

E150 breaks the DHARMA monoculture in the intended way:

1. A genuinely different corpus still shows overwhelming non-Sanskrit lexical structure.
2. The form of that structure changes by genre.
3. Sanskrit in the chronicle is concentrated in elite titles and court vocabulary, not in the connective tissue of the language.

The most important contrast with E130 is not the weak Spearman rank fit. It is the domain inversion:

- **E130:** substrate = action-rich daily-life lexicon
- **E150:** chronicle native layer = grammar + polity + place backbone

That is a useful stratification result, not a failure.

## Output Files

- `babad_substrate_analysis.py` - main parser + classifier
- `results/e150_results.json` - structured results
- `results/classified_top_tokens.csv` - top 150 tokens with class/domain labels
- `results/native_content_terms.csv` - native non-grammar content terms
- `results/chapter_token_summary.csv` - chapter-level token summary
- `results/domain_comparison.csv` - E150 vs E130 domain table

## Limitations

1. Classification is conservative and focused on the highest-frequency token window.
2. Romanized spelling flattens historical orthographic differences.
3. Proper names and polity names are grouped into `OTHER`.
4. This is token-frequency analysis, not full morphological parsing.

## Conclusion

**SUCCESS.** *Babad Tanah Jawi* confirms that DHARMA is not the only place where indigenous lexical persistence appears. In the chronicle's most frequent lexical stratum, **83.9%** of tokens are native/non-Sanskrit, with Sanskrit only **6.6%**. Unlike E130's ACTION-heavy substrate profile, the chronicle preserves a **GRAMMAR-heavy, polity-heavy native backbone**. Indianization is therefore best understood as a register-specific overlay, not a lexical replacement of Javanese historical language.

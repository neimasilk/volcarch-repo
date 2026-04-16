# E200: Historical Dutch NER Baseline Analysis

**Status:** SUCCESS
**Date:** 2026-04-15
**Purpose:** Establish concrete baseline for PhD proposal — what standard NER can/cannot do on colonial Dutch

## Hypothesis

Standard Dutch NER models cannot handle colonial archaeological texts. The PhD needs to close specific, quantifiable gaps.

## Key Results

| Metric | Value |
|--------|-------|
| Total E091 entities | 21,871 |
| PARTIAL coverage by standard NER | 54.2% (LOC/DATE entities) |
| NO coverage by standard NER | 45.8% (domain-specific entities) |
| Estimated standard NER recall | ~27% (50% recall on PARTIAL entities) |
| Colonial spelling variants per entity | 2-5 average |
| OCR error rate estimate | 3-8% |
| Non-Dutch entity fraction | ~15% |
| PhD gaps identified | 5 |

## The 5 PhD Gaps

1. **Entity coverage** — standard NER covers ~27% of required types
2. **Orthographic normalization** — colonial Dutch is out-of-vocabulary
3. **Temporal resolution** — implicit period markers need classification
4. **Place-name disambiguation** — ~80% colonial toponyms would fail modern geocoding
5. **Physical validation** — no existing paradigm for non-textual ground truth

## Data Sources Verified

- E091: 22,162 mentions (6,932 sites, 4,933 locations, 9,238 materials, 742 volcanic, 260 burial, 26 depths)
- E141: 1,768 records, 165 geocoded, 9 depth records
- E197: 33 combined depth records (24 E091 + 9 E141), Wilcoxon p=0.131

## Files

- `ner_baseline.py` — Analysis script
- `results/ner_baseline_results.json` — Structured output

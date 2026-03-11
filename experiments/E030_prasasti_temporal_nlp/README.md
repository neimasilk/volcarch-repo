# E030 — Temporal NLP Analysis of Old Javanese Inscriptions

**Status:** SUCCESS
**Date:** 2026-03-10
**Papers served:** P5 (Volcanic Ritual Clock), P14 (Pararaton Collapse)

## Hypothesis

1. Pre-Indic ritual vocabulary (hyang, manghuri, wuku, kabuyutan) erodes over time as Sanskrit influence increases.
2. Major volcanic eruptions produce detectable gaps in inscription production.
3. Lexical diversity of inscriptions changes around political transitions.

## Method

- **Data:** E023 DHARMA corpus classification (268 inscriptions; 166 successfully dated via regex extraction from titles)
- **Date extraction:** Regex-based parsing of Saka era dates, CE dates, and century descriptions from inscription titles
- **Term classification:** Uses E023 ritual element ontology (pre-Indic / Indic / ambiguous)
- **Statistical tests:** Spearman rank correlation, Mann-Whitney U
- **Volcanic data:** GVP eruption records (Kelud, Bromo, Semeru) + known Merapi eruptions from literature

## Key Results

### A. Ritual Vocabulary Evolution

**Counter-intuitive finding:** The pre-Indic ratio *increases* over time (Spearman rho=+0.502, p<0.001).

However, this result requires careful interpretation:

- The 8th-century cohort (n=55) is dominated by ~50 Borobudur hidden-base relief labels, which are short Sanskrit texts with zero pre-Indic content. This pulls the 8th-century mean ratio to near zero (0.005).
- Excluding Borobudur labels, the 8th century has very few dated inscriptions, all Sanskrit-heavy.
- From the 9th century onward, Old Javanese (kaw-Latn) inscriptions dominate, and these consistently contain pre-Indic terms.
- **hyang (PMP *qiang)** is remarkably persistent: present in >50% of inscriptions in the 9th-14th centuries.
- **manghuri** (ancestor return concept) appears only in 10th-13th century inscriptions — possibly a formulaic innovation of the Sindok/Airlangga period.

**Interpretation for P5:** Pre-Indic substrate vocabulary does NOT erode. Instead, it persists as a stable layer beneath Sanskrit overlay. This supports P5's argument that indigenous ritual concepts (including decomposition-linked selametan intervals) survived Indianization.

### B. Inscription Density & Volcanic Events

- **Peak production:** 9th-10th century (Mataram period), especially 900-949 CE (n=45).
- **Dramatic decline after 929 CE:** Coincides with court transfer from Central to East Java.
- **Merapi 1006 eruption:** Occurs during an already-declining period; 0 inscriptions in the 50 years before, 10 in the 50 years after (the Pucangan charter cluster).
- **Late Kelud eruptions (1376-1450):** Coincide with complete cessation of inscription production in the DHARMA corpus. However, this also overlaps with the decline of Majapahit state capacity.
- **Key caveat:** Political factors (court relocation, dynastic change) are the primary drivers of inscription production rates. Volcanic events may compound or deepen these declines but are not independently testable with this dataset.

### C. Lexical Diversity

- **Mean word count increases over time** (Spearman rho=0.700, p=0.036): later inscriptions tend to be longer.
- **Type-token ratio** shows no significant trend (p=0.49), but this metric is confounded by varying sample sizes per century.
- **Pre-1293 vs Post-1293:** No significant difference in pre-Indic ratio (Mann-Whitney p=0.65), but only 6 post-1293 inscriptions with dates — too few for robust comparison.

## Limitations

1. **Date extraction:** 102 of 268 inscriptions could not be dated from title strings alone (38%). These undated inscriptions may introduce selection bias.
2. **Borobudur effect:** ~50 Borobudur relief labels (8th c., Sanskrit, 1-word) inflate the 8th-century count without contributing ritual content, skewing temporal statistics.
3. **Corpus completeness:** The DHARMA digital corpus is not a complete census of all known Old Javanese inscriptions. Brandes' *Oud-Javaansche Oorkonden* lists >400.
4. **Keyword-based analysis:** Pre-Indic/Indic classification relies on a pre-defined keyword list, not full NLP semantic analysis.
5. **GVP data gaps:** Merapi eruption data before 1768 CE is not in the local GVP files; historical eruptions are added from published literature.
6. **Confounding:** Volcanic events and political transitions are temporally correlated (Merapi 1006 ~ court move; Kelud 14th c. ~ Majapahit decline), making independent attribution impossible.

## Output Files

| File | Description |
|------|-------------|
| `results/temporal_summary.json` | All statistics and findings in structured format |
| `results/dated_inscriptions.csv` | 166 dated inscriptions with century assignments |
| `results/ritual_term_evolution.png` | 3-panel chart: term frequencies, pre-Indic ratio, hyang persistence |
| `results/preindic_ratio_trend.png` | Scatter + trend of pre-Indic ratio over time |
| `results/inscription_density_eruptions.png` | Inscription production bars + volcanic event overlays |

## Conclusion

The central P5 claim is **supported**: pre-Indic ritual vocabulary persists throughout the Old Javanese inscription record (7th-14th c. CE) despite heavy Sanskritization. The term *hyang* (PMP *qiang) appears in >50% of dateable inscriptions across most centuries. This is consistent with indigenous ritual concepts — including decomposition-linked timing — surviving beneath an Indic textual veneer.

For P14, the inscription density analysis shows a suggestive pattern: the post-929 CE decline in Central Javanese inscription production coincides with both the political shift eastward AND increased Merapi activity, but the data cannot disentangle these factors.

# E096 — DHARMA Diachronic BERTopic

## Hypothesis
Topic distributions in Old Javanese inscriptions shift across centuries, with the 929 CE Mataram collapse marking a significant discontinuity. Topics related to royal authority and centralized administration should decline post-929, while topics related to local governance and religious endowment may persist or increase.

## Significance
This is the **first application of BERTopic to any epigraphic corpus**. Diachronic topic modeling has been applied to modern text corpora but never to ancient inscriptions. The 929 CE divide (eruption of Merapi / political collapse / eastward shift) provides a natural experiment for testing whether inscriptional discourse changed structurally.

## Method
1. Parse DHARMA XMLs, filter to DATED inscriptions with translation text (~86 expected)
2. Embed translation texts using SBERT (all-MiniLM-L6-v2) on CUDA
3. Run BERTopic on the corpus
4. Group by century and create topic emergence/disappearance heatmap
5. Pre-929 vs post-929 CE comparison: which topics are unique to each period?
6. Statistical testing: chi-square or Fisher exact on topic distributions
7. Caveats: C7 (~2), C8 (~5), C12 (~1) too sparse for standalone analysis; focus on C9-C10 (59 inscriptions) vs C11-C14 (20 inscriptions)

## Data
- Source: DHARMA EpiDoc corpus, 269 XML files (subset: ~86 dated with translations)
- Location: `../E023_ritual_screening/data/dharma/xml/`

## Output
- `results/topic_heatmap.json` — century x topic matrix
- `results/pre_post_929_comparison.json` — topic analysis across the 929 divide
- `results/e096_results.json` — full results summary

## Status
PENDING — awaiting first run

## Dependencies
- sentence-transformers, bertopic, umap-learn, hdbscan, numpy, scikit-learn, scipy
- CUDA GPU (RTX 4080)

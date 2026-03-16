# E094 — DHARMA Semantic Search

## Hypothesis
Old Javanese inscriptions cluster semantically by CONTENT rather than by century or language alone. Semantic search over SBERT embeddings can reveal thematic structure invisible to keyword search, and temporal centroid drift can measure how inscriptional discourse evolved across centuries.

## Significance
This is the **first application of Sentence-BERT embeddings to Old Javanese epigraphy**. Previous computational approaches to inscriptions have used bag-of-words or simple keyword matching. Dense semantic embeddings enable:
- Content-based clustering independent of vocabulary overlap
- Semantic similarity search across languages and centuries
- Quantification of temporal semantic drift

## Method
1. Parse 269 DHARMA EpiDoc TEI-XML inscriptions
2. Filter to inscriptions with non-empty English translation text
3. Embed all translation texts using SBERT (all-MiniLM-L6-v2) on CUDA
4. UMAP + HDBSCAN clustering; evaluate whether clusters align with century, language, or content
5. Semantic queries: embed 7 thematic query strings, retrieve top-10 nearest inscriptions per query
6. Temporal drift: compute century centroids, measure inter-century distances
7. Indigenous vs Sanskrit vocabulary analysis per cluster (reusing E074 word lists)

## Data
- Source: DHARMA EpiDoc corpus, 269 XML files
- Location: `../E023_ritual_screening/data/dharma/xml/`

## Output
- `results/dharma_embeddings.npy` — embedding matrix (N x 384)
- `results/dharma_clusters.json` — cluster assignments + analysis
- `results/semantic_queries.json` — query results with similarity scores
- `results/temporal_drift.json` — century centroid distances
- `results/e094_results.json` — full results summary

## Status
PENDING — awaiting first run

## Dependencies
- sentence-transformers, umap-learn, hdbscan, numpy, scikit-learn
- CUDA GPU (RTX 4080)

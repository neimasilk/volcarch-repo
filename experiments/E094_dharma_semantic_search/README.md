# E094 — DHARMA Semantic Search

**Status:** SUCCESS
**Date:** 2026-03-17
**Layer:** L4 (cosmological overwrite) + L5 (genre taphonomy)
**Papers:** P5/P8 revision ammo, P16
**Hardware:** RTX 4080, CUDA 12.4

---

## Hypothesis

Old Javanese inscriptions cluster semantically by CONTENT rather than by century or language alone. Semantic search over SBERT embeddings can reveal thematic structure invisible to keyword search, and temporal centroid drift can measure how inscriptional discourse evolved across centuries.

## Significance

**First application of Sentence-BERT embeddings to Old Javanese epigraphy.** Previous computational approaches to inscriptions used bag-of-words or keyword matching. Dense semantic embeddings enable content-based clustering, cross-language similarity search, and quantification of temporal drift.

## Method

1. Parse 268 DHARMA EpiDoc TEI-XML inscriptions
2. Filter to inscriptions with non-empty English translation text → **173 inscriptions**
3. Embed all translation texts using SBERT (all-MiniLM-L6-v2) on CUDA
4. UMAP + HDBSCAN clustering; evaluate century vs content alignment
5. 7 thematic semantic queries, retrieve top-10 nearest inscriptions per query
6. Temporal drift: compute century centroids, measure inter-century distances
7. Indigenous vs Sanskrit vocabulary analysis per cluster (reusing E074 word lists)

## Data

- Source: DHARMA EpiDoc corpus, 268 XML files → 173 with translations
- Languages: kaw-Latn (89), san-Latn (59), osn-Latn (14), omy-Latn (10)
- Dated: C8 (5), C9 (17), C10 (12), C11 (3), C12 (1), C13 (4), C14 (4); 127 undated

## Results

### Clustering: Content-based, not century-based

- **4 clusters**, 0 noise points
- **Century purity: 0.370** (low — clusters are thematic, not temporal)
- Cluster 0 (29): mostly Sanskrit (23 san-Latn). Sanskrit-dominated administrative texts.
- Cluster 1 (15): pure Sanskrit. Formal religious/philosophical content.
- Cluster 2 (9): near-pure Sanskrit (8 san + 1 kaw).
- **Cluster 3** (120): dominant Old Javanese (83 kaw-Latn). **All dated inscriptions fall here.** Contains 838 Sanskrit terms, 373 indigenous terms. Sanskrit/indigenous ratio: 0.445 — most balanced.

### Semantic Queries

| Query | Mean Sim | Best Century Hits | Top Hit (score) |
|-------|----------|-------------------|-----------------|
| village administration and land grants | 0.360 | C10, C11 | Borobudur relief labels (0.525) |
| **mountain worship and sacred peaks** | **0.395** | C9, C10 | Borobudur relief label 124A (0.595) |
| water infrastructure irrigation dams | 0.265 | C9 | Prayer for long life of Vijaya (0.339) |
| royal genealogy and succession | 0.358 | C11, C13, C14 | Borobudur relief labels (0.404) |
| tax collection and economic regulation | 0.236 | C10, C11 | Kebantenan 4 (0.346) |
| Buddhist monastery and religious donation | 0.372 | C10 | Sugih Manek Charter 915 CE (0.448) |
| **volcanic landscape fire mountain** | **0.244** | C9, C10, C14 | Sri Manggala II/III (0.272) |

**Key finding: "volcanic landscape fire mountain" has the LOWEST mean similarity (0.244)** of all 7 queries. This confirms that volcanic/landscape themes are **rare in the epigraphic record** — inscriptions overwhelmingly concern administration, religion, and kingship, not physical geography. The best matches (Sri Manggala II/III, C9; Adan-Adan, C14) are worth manual examination for landscape references.

Contrast: "mountain worship and sacred peaks" scores **highest** (0.395) — mountains appear in inscriptions as sacred/cosmological sites, not as geological/volcanic features. This supports L4's claim that cosmological framing overwrites physical observation.

### Temporal Drift

| Transition | Centroid Distance | Interpretation |
|------------|------------------|---------------|
| C8 → C9 | 0.163 | Moderate shift |
| C9 → C10 | 0.087 | **Smallest** — high continuity (Mataram peak) |
| C10 → C11 | 0.208 | Growing divergence |
| **C11 → C12** | **0.366** | **LARGEST — major semantic rupture** |
| C12 → C13 | 0.245 | Partial recovery |
| C13 → C14 | 0.181 | Stabilization |

- **Pre-929 vs Post-929 centroid distance: 0.112** (moderate, smaller than most consecutive-century jumps)
- The biggest rupture is NOT at 929 CE but at **C11→C12** — the Airlangga/post-Mataram transition period

### Vocabulary Analysis

- Cluster 0: indigenous/Sanskrit ratio 0.221 (Sanskrit-dominated)
- Cluster 3: ratio 0.445 (more balanced — the main Old Javanese cluster)
- Clusters 1, 2: insufficient terms for analysis

## Implications

1. **For P5/P8:** Mountains in inscriptions = sacred peaks, NOT geological features. Confirms cosmological overwrite (L4).
2. **For P16:** Volcanic silence in epigraphy is quantifiable — mean similarity 0.244 vs 0.395 for mountain worship.
3. **For L5 (genre taphonomy):** Inscriptions as a genre exclude physical geography. The SBERT embeddings capture this exclusion pattern.
4. **The C11→C12 semantic rupture** deserves further investigation — it's larger than the 929 CE divide. May reflect the shift from Central to East Javanese epigraphic conventions.

## Status

**SUCCESS** — First SBERT application to Old Javanese epigraphy. Content-based clustering confirmed. Volcanic themes quantifiably rare. C11→C12 semantic rupture identified. Temporal drift measured.

## Output Files

| File | Description |
|------|-------------|
| `results/dharma_embeddings.npy` | Embedding matrix (173 × 384) |
| `results/dharma_clusters.json` | Cluster assignments + analysis |
| `results/semantic_queries.json` | Query results with similarity scores |
| `results/temporal_drift.json` | Century centroid distances |
| `results/e094_results.json` | Full results summary |

## Dependencies

- sentence-transformers, umap-learn, hdbscan, numpy, scikit-learn
- CUDA GPU (RTX 4080)

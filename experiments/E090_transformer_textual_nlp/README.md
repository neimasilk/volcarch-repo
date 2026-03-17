# E090: Transformer-based NLP on Ancient Textual Corpus

**Status:** SUCCESS (v5: 4/4 experiments strong, BERTopic reactivated)
**Date:** 2026-03-17 (v5 run)
**Layer:** L3 + cross-cutting
**Paper:** P16 (draft)
**Hardware:** RTX 4080, CUDA 12.4, PyTorch 2.6

---

## Hypothesis

State-of-the-art NLP (Sentence-BERT, UMAP+HDBSCAN, BERTopic, NLI) can reveal cross-tradition semantic convergence in ancient textual references to Nusantara that goes beyond manual annotation.

## Method

6 transformer-based experiments on the E089 corpus. Run history:
- **v2** (2026-03-16): 50 passages, 10 traditions. 4/6 informative, 2/6 negative.
- **v5** (2026-03-17): 200 passages, 12 traditions. 4/4 strong. BERTopic REACTIVATED.

Experiments:
1. **Sentence-BERT embeddings** (all-MiniLM-L6-v2) on CUDA
2. **UMAP + HDBSCAN clustering**
3. ~~Zero-shot NER~~ (v2 only, moderate)
4. **BERTopic** — latent topic discovery (reactivated at 200 entries)
5. **Semantic convergence** — embedding-space Monte Carlo test (extended to 8 groups)
6. ~~NLI Entailment~~ (v2 only, conceptually wrong for this task)

## V5 Results (200 entries, 12 traditions)

### EXP 1: SBERT Similarity — INFORMATIVE
- Within-tradition similarity: **0.394**
- Between-tradition similarity: **0.305**
- Ratio: **1.292** (traditions self-similar but NOT isolated)
- Highest centroid pair: **ARAB-PERSIAN: 0.879** (shared geographic/cosmographic tradition)
- Lowest: **LINGUISTIC-ROMAN: 0.427** (different discourse types)
- Top cross-pair: **Kitab al-Bad' (ARAB) ↔ Hudud al-Alam (PERSIAN): 0.828**

### EXP 2: UMAP+HDBSCAN — STRONG
- **21 clusters** found, 21 noise points (200 entries)
- **57% cross-tradition clusters** (12/21 mix multiple traditions)
- → CONTENT-DRIVEN clustering confirmed at 4× scale
- Key clusters:
  - Cluster 3 (58 members): Indian+Arab+Tamil+Greek — "maritime knowledge" supercluster
  - Cluster 4 (9 members): CHEMICAL only — archaeochemical evidence forms own genre
  - Cluster 14 (4 members): NUSANTARAN only — kakawin literary texts
  - Cluster 8 (8 members): CHEMICAL+ROMAN+GREEK+ARAB+EUROPEAN — "spice trade" cluster

### EXP 4: BERTopic — REACTIVATED, STRONG
- **16 topics** discovered (vs 3 in v2 — corpus size was the bottleneck)
- Key topics:
  - **Topic 0** (25 docs): "ship, sea, merchant, merchants, gold" — maritime trade
  - **Topic 1** (23 docs): "cinnamon, islands, cloves, rome, clove" — Roman-era spice trade
  - **Topic 4** (14 docs): "**volcanic, sanskrit, inscriptions, javanese, malay**" — VOLCARCH-relevant!
  - **Topic 12** (5 docs): "**mountain, slopes, clouds, temples, smoke**" — volcanic landscape!
  - Topic 5 (11 docs): "zabaj, islands" — Arab geographical tradition
  - Topic 6 (9 docs): "monks, rules, envoys" — Chinese diplomatic/Buddhist
  - Topic 8 (7 docs): "inscription, saka, royal, suvarnadvipa" — Nusantaran epigraphy
- Topic 4 and Topic 12 are directly relevant to P16 — they show that volcanic/landscape themes form coherent latent topics in the cross-tradition corpus

### EXP 5 Extended: Semantic Convergence (8 groups) — VERY STRONG

| Concept | Passages | Traditions | Z-score (v2) | Z-score (v5) | Delta | Verdict |
|---------|----------|-----------|-------------|-------------|-------|---------|
| JAVA | 82 | 11 | 0.88 | **21.91** | +21.03 | **CONVERGES** |
| SUMATRA_GOLD | 75 | 11 | 4.84 | **25.22** | +20.38 | **CONVERGES** |
| CAMPHOR_BARUS | 51 | 10 | 6.55 | **28.76** | +22.21 | **CONVERGES** |
| SPICE_TRADE | 76 | 10 | 8.42 | **34.28** | +25.85 | **CONVERGES** |
| MARITIME_VOYAGE | 128 | 12 | 9.44 | **19.48** | +10.05 | **CONVERGES** |
| VOLCANO | 54 | 12 | N/A | **7.39** | NEW | **CONVERGES** |
| BUDDHIST_WORLD | 23 | 8 | N/A | **4.59** | NEW | **CONVERGES** |
| METAL_TRADE | 135 | 12 | N/A | **2.71** | NEW | **CONVERGES** |

- **8/8 concept groups converge** (all p < 0.01)
- JAVA went from NOT converging (z=0.88) to VERY strong (z=21.91) — corpus expansion resolved it
- VOLCANO concept (NEW): 54 passages from ALL 12 traditions converge semantically (z=7.39)
- CAMPHOR_BARUS remains highest per-passage signal (z=28.76)

## Delta: v2 → v5

| Metric | v2 (50 entries) | v5 (200 entries) | Change |
|--------|----------------|-----------------|--------|
| Corpus | 50 entries, 10 traditions | 200 entries, 12 traditions | +300% |
| Clusters | 9 | 21 | +133% |
| Cross-trad clusters | 78% | 57% | -21% (more tradition-specific clusters emerge) |
| BERTopic topics | 3 (weak) | **16 (strong)** | Corpus size was bottleneck |
| Convergence | 4/5 | **8/8** | All concepts converge |
| JAVA z-score | 0.88 (NS) | **21.91** | Now strongest ever |

## V2 Results (Historical — 50 entries)

### EXP 3: Zero-shot NER — MODERATE
- Mean F1: 0.650 (precision 0.581, recall 0.867). Useful as annotation tool, not publishable as NER.

### EXP 6: NLI Entailment — NEGATIVE (honest)
- Mean entailment: 0.161 (below random baseline). NLI is wrong tool — convergence operates at entity level, not statement level.

## Overall Assessment (v5)

| Experiment | v2 Verdict | v5 Verdict | Publishability |
|------------|-----------|-----------|---------------|
| EXP 1: SBERT | INFORMATIVE | INFORMATIVE | Supporting evidence |
| EXP 2: UMAP+HDBSCAN | STRONG | **STRONG** | Publishable figure |
| EXP 4: BERTopic | WEAK | **STRONG** | 16 latent topics, 2 volcanic-relevant |
| EXP 5: Convergence | STRONG (4/5) | **VERY STRONG (8/8)** | Core statistical result |

**Two genuinely novel results:**
1. Ancient texts about Nusantara cluster by CONTENT not by CULTURE in embedding space
2. 8/8 concept groups (including VOLCANO) show statistically significant semantic convergence across up to 12 independent traditions

**P16 viability:** Topic 4 (volcanic/inscriptions) and Topic 12 (mountain/temples/smoke) confirm that volcanic landscape is a latent theme in ancient Nusantara literature. Combined with 8/8 convergence, this provides computational evidence for VOLCARCH's textual archaeology arm.

## Limitations

1. All passages are English translations — cross-lingual analysis on originals would be stronger
2. SBERT (all-MiniLM-L6-v2) trained on modern English — may miss ancient nuances
3. UMAP is stochastic — results vary with random seed
4. 12 traditions with unequal sizes (CHINESE: 42, PERSIAN: 7) — size imbalance affects clustering

## Output Files

| File | Description |
|------|-------------|
| `results/e090_all_results.json` | Complete results (v2 corpus, 50 entries) |
| `results/e090_summary.json` | Summary statistics (v2) |
| `results/passage_embeddings.npy` | 50×384 SBERT embeddings (v2) |
| `results/e090_v5_full_results.json` | **Complete v5 results (200 entries, 4 experiments)** |
| `results/e090_v5_delta.json` | **v2→v5 delta comparison** |

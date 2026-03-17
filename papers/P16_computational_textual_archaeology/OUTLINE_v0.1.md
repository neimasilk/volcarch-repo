# P16: Computational Textual Archaeology — Paper Outline v0.1

**Working Title:** "What Ancient Texts Remember and Inscriptions Forget: Transformer-Based Evidence for Volcanic Taphonomic Bias in Indonesian Textual Records"

**Alternative Titles:**
- "Semantic Convergence Across Twelve Ancient Traditions: Computational Evidence for Nusantaran Civilizations Before the Epigraphic Record"
- "Volcanic Silence in Stone: NLP Evidence that Inscriptions Exclude Physical Geography"

**Author:** Mukhlis Amien (single-author)
**Target journal:** Digital Scholarship in the Humanities (Oxford, Q1, no APC for short articles)
**Backup:** Journal of Archaeological Science (Elsevier, Q1), or Literary and Linguistic Computing

**Date:** 2026-03-17

---

## 1. Core Argument (1 paragraph)

Ancient external textual traditions (Greek, Roman, Indian, Chinese, Arab, Persian, Tamil, chemical, linguistic) independently describe Nusantara from ~1700 BCE, yet Indonesia's own epigraphic record begins only ~400 CE. Using transformer-based NLP on a 200-passage corpus spanning 12 traditions and 173 DHARMA inscriptions, we demonstrate: (1) cross-tradition semantic convergence on Nusantaran themes is statistically significant across all 8 tested concept groups; (2) volcanic/landscape themes are the *rarest* semantic category in Old Javanese inscriptions despite Java having 45 active volcanoes; and (3) the 929 CE political collapse produced a statistically significant discursive shift (p = 0.0003) in which administrative language was replaced by royal legitimation. These computational findings support the hypothesis that Indonesia's archaeological darkness is not an absence of past civilization but a product of taphonomic bias — geological (volcanic burial), institutional (survey deficit), and textual (genre-specific exclusion of physical geography).

## 2. Structure

### Introduction (~1500 words)
- The archaeological darkness problem: pre-400 CE Nusantara
- External evidence vs internal silence
- Computational textual archaeology as a new methodology
- Research questions:
  1. Do 12 independent textual traditions converge semantically on Nusantara?
  2. What topics emerge when BERTopic is applied to this cross-tradition corpus?
  3. How does inscriptional discourse relate to physical landscape?
  4. Does the 929 CE political collapse mark a discursive discontinuity?

### Background (~1000 words)
- Textual sources for pre-modern Nusantara (brief survey, 12 traditions)
- NLP in archaeology/epigraphy: state of the art
  - NER on cuneiform (Pagé-Perron et al. 2017)
  - Topic modeling on papyri (Assael et al. 2022, Ithaca)
  - SBERT on historical texts (Manjavacas et al. 2019)
  - **Gap:** No transformer-based NLP on Indonesian epigraphy or cross-tradition Nusantara corpus
- The VOLCARCH framework: volcanic taphonomic bias (cite P1, published or preprint)
- The 929 CE divide: Mataram collapse, eastward shift, Merapi eruption theories

### Data (~800 words)
- **Corpus 1:** E089 v5 — 200 passages from 12 traditions (3065-year span)
  - Tradition sizes, temporal distribution, language of translation
  - Construction method: systematic mining + AI-assisted extraction
  - Passage selection criteria: must contain direct reference to Nusantaran geography, commodities, or peoples
- **Corpus 2:** DHARMA EpiDoc — 268 Old Javanese/Sanskrit/Old Malay inscriptions
  - 173 with English translations (used for embedding)
  - 46 dated inscriptions (used for diachronic analysis)
  - XML-TEI format, open access

### Methods (~1200 words)
#### 3.1 Cross-tradition semantic analysis (E090 v5)
- SBERT encoding (all-MiniLM-L6-v2) on 200 passages
- UMAP + HDBSCAN unsupervised clustering
- BERTopic latent topic discovery
- Monte Carlo semantic convergence test: 8 concept groups (JAVA, SUMATRA_GOLD, CAMPHOR_BARUS, SPICE_TRADE, MARITIME_VOYAGE, VOLCANO, BUDDHIST_WORLD, METAL_TRADE)
- Z-score against random baseline (10,000 permutations per group)

#### 3.2 Epigraphic semantic search (E094)
- SBERT encoding of 173 DHARMA translations
- 7 thematic queries (administration, mountain worship, water, genealogy, taxation, Buddhist donation, volcanic landscape)
- Temporal centroid drift: cosine distance between century centroids
- Indigenous vs Sanskrit vocabulary ratio per cluster

#### 3.3 Diachronic topic modeling (E096)
- BERTopic on 46 dated inscriptions
- Topic × century heatmap
- Pre-929 vs post-929 CE topic distribution comparison
- Chi-square test + Fisher exact tests per topic

### Results (~2000 words)
#### 4.1 Cross-tradition convergence
- **8/8 concept groups converge** (all p < 0.01)
- Strongest: SPICE_TRADE (z = 34.28), CAMPHOR_BARUS (z = 28.76)
- VOLCANO concept: z = 7.39 across ALL 12 traditions — volcanic awareness is pan-tradition
- JAVA went from non-convergent (z = 0.88 at N=50) to z = 21.91 at N=200
- Table: concept, N passages, N traditions, z-score, p-value

#### 4.2 Latent topics in the cross-tradition corpus
- **16 BERTopic topics** — interpretable thematic structure
- Topic 4 ("volcanic, sanskrit, inscriptions, javanese, malay"): volcanic-linguistic nexus
- Topic 12 ("mountain, slopes, clouds, temples, smoke"): volcanic landscape description
- Topic 0 ("ship, sea, merchant"): maritime trade — the dominant discourse
- 57% of HDBSCAN clusters are cross-tradition — content-driven, not culture-driven
- Figure: UMAP scatter colored by tradition vs colored by BERTopic topic

#### 4.3 Volcanic silence in epigraphy
- "volcanic landscape fire mountain" query: **lowest** mean similarity (0.244) across all 7 queries
- "mountain worship and sacred peaks": **highest** (0.395)
- Mountains in inscriptions = cosmological/sacred sites, NOT geological features
- This gap is quantifiable: 0.151 similarity difference between sacred and physical mountain discourse
- Table: query, mean similarity, top century hits, top inscription

#### 4.4 The 929 CE discursive discontinuity
- **3 BERTopic topics** in 46 dated inscriptions: administrative (T0), royal (T1), ritual (T2)
- Chi-square: p = 0.0003 — topic distribution significantly different pre vs post 929
- Topic 1 (royal): 6% pre-929 → 62% post-929 (Fisher p = 0.0002)
- Topic 2 (ritual/calendrical): 18% pre-929 → 0% post-929 (complete disappearance)
- C11→C12 shows largest semantic rupture (cosine distance 0.366)
- Figure: topic × century heatmap

### Discussion (~1500 words)
#### 5.1 Computational evidence for taphonomic bias
- External traditions describe a Nusantara that inscriptions don't — this is genre taphonomy (L5)
- Volcanic awareness exists in 12 traditions (VOLCANO z = 7.39) but is ABSENT from inscriptions (similarity 0.244)
- The gap between external textual memory and internal inscriptional silence is measurable

#### 5.2 What the 929 CE shift reveals
- Not just geographic relocation — discursive transformation
- Administrative vocabulary → royal legitimation vocabulary
- Ritual/calendrical formulas vanish — possible connection to Merapi eruption disrupting ritual cycles
- Implications for using inscriptions as proxies for settlement patterns

#### 5.3 Methodological contribution
- First SBERT + BERTopic application to Old Javanese epigraphy
- First cross-tradition semantic convergence test spanning 12 ancient traditions
- Reproducible pipeline: DHARMA XML → embedding → clustering → statistical test
- Transferable to other archaeological darkness problems (sub-Saharan Africa, Southeast Asian mainland)

#### 5.4 Limitations
- English translations only (cross-lingual analysis on originals desirable)
- Small N for diachronic BERTopic (46 dated inscriptions)
- SBERT trained on modern English — domain adaptation would improve results
- Corpus construction involves subjective passage selection

### Conclusion (~500 words)
- Three independently derived computational findings converge:
  1. Ancient traditions agree about Nusantara (8/8 convergence)
  2. Inscriptions systematically exclude physical geography (volcanic silence)
  3. Political collapse produces measurable discursive shifts (929 CE)
- Together: the archaeological record of Indonesia is shaped by what stone remembers and what stone forgets
- Call for computational textual archaeology as complement to fieldwork

### References (~40-50 refs)
- VOLCARCH papers (P1, P5, P7, P8, P9 — cite as submitted/preprint)
- NLP/DH methods (SBERT: Reimers & Gurevych 2019; BERTopic: Grootendorst 2022; UMAP: McInnes et al. 2018)
- DHARMA project (Argon et al.)
- Primary sources (Periplus, Ptolemy, Pliny, Yijing, etc.)
- Archaeological context (Degroot 2009, Miksic 2004, Newhall et al. 2000)
- Computational epigraphy (Pagé-Perron et al. 2017, Assael et al. 2022)

## 3. Figures

| # | Description | Source |
|---|-------------|--------|
| 1 | UMAP scatter of 200 passages colored by tradition | E090 v5 |
| 2 | UMAP scatter of 200 passages colored by BERTopic topic | E090 v5 |
| 3 | Semantic convergence z-scores (8 groups, bar chart) | E090 v5 |
| 4 | Semantic query similarities (7 queries, horizontal bar) | E094 |
| 5 | Topic × century heatmap (pre/post 929 CE) | E096 |
| 6 | Temporal centroid drift (century distances, line plot) | E094 |

## 4. Experiments → Paper Mapping

| Section | Experiment | Key Data |
|---------|-----------|----------|
| 4.1 Convergence | E090 v5 (EXP 5) | 8/8 converge, z-scores |
| 4.2 Latent topics | E090 v5 (EXP 4) | 16 BERTopic topics |
| 4.2 Clustering | E090 v5 (EXP 2) | 21 clusters, 57% cross-trad |
| 4.3 Volcanic silence | E094 | Query similarities, 0.244 vs 0.395 |
| 4.4 929 CE shift | E096 | Chi-square p=0.0003, Fisher p=0.0002 |
| Background | E088 | Monte Carlo p < 0.00001 (entity convergence) |
| Data | E089 v5 | 200 passages, 12 traditions |

## 5. What Makes This Novel

1. **First transformer NLP on Old Javanese epigraphy** — no prior work uses SBERT/BERTopic on any Indonesian inscriptions
2. **First cross-tradition semantic convergence test** spanning 12 ancient traditions simultaneously
3. **Quantification of "volcanic silence"** in epigraphy — a measurable gap between what external sources report and what inscriptions record
4. **Computationally detected 929 CE discursive shift** — first statistical evidence that the Mataram collapse changed not just WHERE but WHAT inscriptions say
5. **Methodological template** — reproducible pipeline transferable to other archaeological darkness contexts

## 6. Estimated Length

- Target: 8,000-10,000 words (DSH standard length)
- 6 figures, 4-5 tables
- ~50 references

## 7. Next Steps

1. Generate publication-quality UMAP figures (Figures 1-2)
2. Create z-score bar chart (Figure 3) and query similarity chart (Figure 4)
3. Write Introduction + Methods drafts
4. Decide on single-author vs co-author (AI-assisted disclosure standard)
5. Check DSH submission guidelines and word limits

## 8. Risk Assessment

- **LOW:** Novelty is clear (first transformer NLP on Old Javanese)
- **MEDIUM:** Small N in E096 (46 inscriptions) — reviewers may question statistical power
- **MEDIUM:** English-only analysis — should acknowledge as limitation and frame as proof-of-concept
- **LOW:** DSH welcomes computational humanities papers — good fit
- **NOTE:** Must NOT overlap substantially with P5 (ritual vocabulary) or P8 (linguistic fossils). P16 focuses on METHOD (NLP pipeline) while P5/P8 focus on CONTENT (ritual/linguistic arguments). Framing is key.

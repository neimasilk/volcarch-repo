# E023: Ritual Screening POC — DHARMA Corpus

## Hypothesis
Pre-Indic ritual/cosmological elements can be identified in Old Javanese inscriptions by AI-assisted screening, separating indigenous elements from imported Hindu-Buddhist layers.

## Method
1. Cloned DHARMA ERC Nusantara epigraphy corpus (CC-BY 4.0)
2. Parsed 268 XML inscriptions (EpiDoc TEI format)
3. Scanned for ritual/cosmological keywords (Sanskrit, Old Javanese, indigenous)
4. Built corpus inventory with metadata (language, date, word count, keywords)
5. Identified pilot candidates for AI extraction

## Data
- DHARMA tfc-nusantara-epigraphy (github.com/erc-dharma)
- 268 inscriptions, primarily Old Javanese (155) and Sanskrit (65)

## Results

### Corpus Summary
| Metric | Count | % |
|--------|-------|---|
| Total inscriptions | 268 | 100% |
| With ritual/cosmological keywords | 201 | 75% |
| With English translation | 218 | 81% |
| Pilot candidates (trans + ritual + length) | 114 | 43% |

### Languages
- kaw-Latn (Old Javanese): 155 (58%)
- san-Latn (Sanskrit): 65 (24%)
- unknown: 21 (8%)
- osn-Latn (Old Sundanese): 14 (5%)
- omy-Latn (Old Malay): 13 (5%)

### Top Ritual Keywords
| Keyword | Count | Type |
|---------|-------|------|
| śaka (calendar) | 131 | Calendar system |
| hyaṁ (divine/sacred) | 114 | **Possibly pre-Indic** |
| vāra (weekday) | 106 | Calendar |
| tithi (lunar day) | 101 | Calendar (Indic) |
| sīma (boundary/charter) | 69 | Administrative-ritual |
| nakṣatra (lunar mansion) | 58 | Calendar (Indic) |
| piṇḍa (offering) | 33 | Ritual (Indic) |
| maṅhuri (offering) | 28 | **Ritual (OJ indigenous?)** |
| pūjā (worship) | 21 | Ritual (Indic) |
| homa (fire ritual) | 19 | Ritual (Indic) |
| samudra (ocean) | 13 | Maritime cosmology |
| kabuyutan (ancestral site) | 3 | **Indigenous** |

### Key Observation
The keyword **hyaṁ/hyang** (divine, sacred) appears in 114/268 inscriptions (43%). This is widely considered a pre-Indic Austronesian term (cf. Malay "Yang", Tagalog "Anito"). Its pervasive presence even in heavily Sanskritized inscriptions suggests a deep indigenous cosmological substrate persisting beneath the Indic overlay — exactly what P5's subtraction methodology predicts.

## Conclusion
**STATUS: SUCCESS — GO for AI screening pipeline**

The DHARMA corpus is substantial (268 inscriptions, 75% with ritual content, 81% with translations). 114 inscriptions qualify as pilot candidates. The presence of indigenous terms (hyaṁ, maṅhuri, kabuyutan) alongside Indic terms confirms the "layered" structure that the subtraction methodology targets.

## Pilot Analysis: 10 Inscriptions (analyze_ritual_elements.py)

### Pre-Indic Elements Identified
| Element | Frequency | Origin | Significance |
|---------|-----------|--------|-------------|
| hyaṁ/hyang | 10/10 (100%) | PMP *qiang | Indigenous divinity in EVERY inscription |
| maṅhuri | 5/10 (50%) | Old Javanese | "Ancestor return" — no Sanskrit source |
| wuku | 1/10 | Indigenous | 210-day calendar, co-exists with Śaka |
| karāman | 1/10 | Old Javanese | Village community term |
| panumbas | 1/10 | Old Javanese | "Redemption" — ritual context |

### Ambiguous Elements (Sanskrit word, possibly covering pre-Indic concept)
| Element | Frequency | Notes |
|---------|-----------|-------|
| sīma | 8/10 | Boundary ritual with unique Javanese elements |
| śapatha/sapatha | 5/10 | Imprecation with volcanic/seismic threats |
| samudra | 6/10 | Ocean cosmology — pan-Austronesian concept |
| samgat | 5/10 | Old Javanese title, unclear etymology |

### Key Finding
Selametan numerology (7-40-100-1000) NOT found in prasasti — as expected for oral tradition. Needs ethnographic sources (Kitab Primbon, Geertz 1960), not epigraphy. Absence from prasasti + Hindu/Buddhist/Islamic texts strengthens pre-Indic argument.

### Pre-Indic Ratio per Inscription
Range: 6.2% (Pucangan, heavily Sanskritized) to 28.6% (Munggut charter). Mean ~15%.

## Full Corpus Analysis (full_corpus_analysis.py)

### Corpus-Wide Statistics (268 inscriptions)
| Metric | Value |
|--------|-------|
| Inscriptions with pre-Indic elements | 126/268 (47%) |
| hyaṁ/hyang prevalence | 116/268 (43%) |
| maṅhuri prevalence | 28/268 (10%) |
| wuku prevalence | 5/268 (2%) |
| Total Indic keyword occurrences | 586 |
| Total pre-Indic keyword occurrences | 165 |
| Total ambiguous keyword occurrences | 150 |

### Pre-Indic Elements by Language
| Language | Total | With pre-Indic | % |
|----------|-------|----------------|---|
| Old Javanese (kaw) | 155 | 102 | 66% |
| Sanskrit (san) | 65 | 4 | 6% |
| Old Sundanese (osn) | 14 | 6 | 43% |
| Old Malay (omy) | 13 | 6 | 46% |

### Cross-Linguistic Findings
- **Old Sundanese**: kabuyutan ("ancestral sacred site") in Kebantenan 3; hyang/hyaṁ widespread
- **Old Malay**: wuku in Dharmasraya A (Sumatra) — indigenous calendar beyond Java
- **Batutulis** (Sundanese): gunung + hyang = mountain veneration + indigenous divinity

## Next Steps
- [x] Run AI extraction on 10 pilot inscriptions — DONE
- [x] Design ritual element ontology — DONE (25 elements classified)
- [x] Scale keyword screening to full 268 inscriptions — DONE
- [x] Test: is 7-40-100-1000 in ANY known Hindu/Buddhist/Islamic source? — **NO** (confirmed via web research)
  - Hindu: NO (10-13 day cycle). Buddhist: PARTIAL (7-day only). Islam: NO (bid'ah).
  - The complete sequence is UNIQUELY JAVANESE. See `results/selametan_source_analysis.md`
  - Geertz 1960 Ch.6: 1000 days = body fully decayed → ritual = taphonomic calendar!
- [ ] Access Kitab Primbon Betaljemur Adammakna via Javanese Wikisource
- [ ] Extract tatacara selamatan chapter from Primbon
- [ ] Build ethnographic text pipeline for prasasti + ritual texts
- [x] Download Pulotu database — DONE (137 cultures, CC-BY 4.0, D-PLACE/GitHub)
  - Q10 (post-death actions affect afterlife): YES for Toraja, Merina, Tanala → shared Austronesian belief
  - 30/137 cultures have "full mortuary package" (deified ancestors + ancestral spirits + post-death ritual efficacy)
  - Both Merina AND Toraja in this group → supports common Austronesian origin
  - See `results/pulotu_mortuary_comparison.csv`
- [ ] Check if hyaṁ/hyang has cognates in Malagasy (hasina? zanahary?)
- [ ] Read Beatty 1999 "Varieties of Javanese Religion" (multivocal slametan)
- [ ] Test: does 1000-day interval match observed decomposition rates in tropical volcanic soil?

## Paper
Links to: Paper 5 — "The Volcanic Ritual Clock"
- Original draft: `docs/drafts/VOLCARCH_Paper5_CosmologicalStratigraphy_DRAFT.docx.pdf`
- **Active outline (v0.1, 2026-03-09):** `papers/P5_volcanic_ritual_clock/outline_v0.1.md`
- Argument synthesis: `results/P5_argument_synthesis.md`

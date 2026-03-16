# Experiment Index

**Last updated:** 2026-03-16
**Total:** 98 experiments (E001-E098, minus E021/E045-E047/E072/E077/E095 unused)
**Regenerate:** `python tools/scan_experiments.py` (auto-scan, partial — manual review needed)

**Convention:** Every experiment gets a README.md with hypothesis, method, result, conclusion, status.
Failed experiments are NOT deleted — they are documented and may be revisited.

---

## Status Summary

| Status | Count | Meaning |
|--------|-------|---------|
| SUCCESS | 61 | Hypothesis supported or useful result |
| INFO NEG | 7 | Negative result that IS informative |
| CONDITIONAL | 4 | Partially supported, caveats |
| INCONCLUSIVE | 3 | Cannot determine, need more data |
| MIXED | 2 | Multiple sub-experiments, mixed results |
| PARTIAL | 1 | Survives with scope restriction |
| FAILED | 2 | Hypothesis rejected or method broken |
| COMPLETE (foundation) | 6 | Data/infrastructure, not hypothesis-driven |
| SUPERSEDED | 6 | Replaced by later iteration |

---

## All Experiments

### Foundation & Data (E001-E006)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E001 | Site geocoding | COMPLETE | P1 | 666 sites, 383 geocoded |
| E002 | Eruption history | COMPLETE | P1,P3 | 168 GVP records, 4 volcanoes |
| E003 | DEM acquisition | COMPLETE | P2 | GLO-30 Malang + Jawa Timur |
| E004 | Density vs volcanic proximity | INFO NEG | P1 | rho=-0.991 (survey bias dominates) |
| E005 | Terrain suitability null | COMPLETE | P1,P2 | rho=-0.364, discovery bias |
| E006 | Nominatim-enriched reanalysis | SUCCESS | P1 | rho=-0.955, n=383, pattern stable |

### Settlement Model Iterations (E007-E016)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E007 | Model v1 (baseline) | SUPERSEDED | P2 | AUC=0.659 |
| E008 | Model v2 (+river) | SUPERSEDED | P2 | AUC=0.695 |
| E009 | Model v3 (+soil) | SUPERSEDED | P2 | AUC=0.664 |
| E010 | Model v4 (TGB) | SUPERSEDED | P2 | AUC=0.711 |
| E011 | Model v5 (TGB tuned) | SUPERSEDED | P2 | AUC=0.725 |
| E012 | Model v6 (TGB+proxy) | SUPERSEDED | P2 | AUC=0.730 |
| E013 | **Model v7 (hybrid)** | **SUCCESS** | P2 | **AUC=0.768, MVR MET** |
| E014 | Temporal split validation | SUCCESS | P2 | AUC=0.755, tautology PASS |
| E015 | SHAP analysis | SUCCESS | P2 | rho=0.943 consistency |
| E016 | Zone classification map | COMPLETE | P1,P2 | Zone B=1.8% GPR targets |

### Tephra & Spatial (E017-E020)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E017 | Tephra POC (Pyle 1989) | **FAILED** | ~~P3~~ | 1/4 sites pass. Killed P3 |
| E018 | TOM proof of concept | INCONCLUSIVE | P7 | Cave-site confound |
| E019 | Spatial distribution test | SUCCESS | P7 | Cohen's d=1.005 |
| E020 | Mini-NusaRC | SUCCESS | P7,D2 | 80 sites, cave bias universal (p=0.761) |

### Linguistic Analysis (E022-E029, E036)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E022 | Linguistic subtraction POC | SUCCESS | P8 | 29.4% residual, 6 Sulawesi langs |
| E023 | Ritual screening (DHARMA) | SUCCESS | P5 | 268 inscriptions, 43% hyaṁ |
| E024 | Borehole screening | **FAILED** | P9 | n=18 too small, pattern visible |
| E025 | Slametan quantitative | SUCCESS | P5 | p<0.001 Monte Carlo |
| E026 | Pararaton volcanic corr. | SUCCESS | ~~P14~~ | p=0.037 (Bonferroni kills → P5 ammo) |
| E027 | ML substrate detection | SUCCESS | P8 | **AUC=0.762**, LOLO 5/6≥0.65 |
| E027b | Substrate expansion (16 langs) | SUCCESS | P8 | Sulawesi>Western Indonesian |
| E028 | Cross-method consensus | SUCCESS | P8 | kappa=0.61, 266 substrates |
| E029 | Phonological clustering | INFO NEG | P8 | Parallel innovation, not shared substrate |
| E036 | Hanacaraka phonology | SUCCESS | P8 | 33→20 consonant reduction |

### Temporal & Botanical (E030-E035, E037-E040)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E030 | Prasasti temporal NLP | SUCCESS | P5,P14 | Pre-Indic ratio INCREASES (rho=+0.50) |
| E031 | Candi siting vs volcano | SUCCESS | P7,P11 | Siting p<0.0001; orientation null |
| E032 | Pranata mangsa seasonality | CONDITIONAL | P5,P11 | p=0.042 (FDR casualty) |
| E033 | Indianization curve | SUCCESS | P5,P8 | rho=-0.211, p=0.030. Peak C9 |
| E034 | Panji in Malagasy | INFO NEG | P9 | Panji post-dates migration |
| E035 | Prasasti botanical keywords | SUCCESS | P5,P9 | 15 plants, mortuary ABSENT |
| E037 | Prasasti dating ML | CONDITIONAL | P5 | MAE=115yr, content too stable |
| E038 | Volcanic vocabulary drift | INFO NEG | P8,P11 | Core vocab too stable for signal |
| E039 | VCS cross-cultural | INFO NEG | P11 | VCS local only, not global |
| E040 | Bamboo civilization | SUCCESS | P1 | 63.4% organic, binomial p<0.0001 |

### Validation (E041-E044)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E041 | IPA validation | SUCCESS | P8 | Ortho→IPA delta=+0.002 (robust) |
| E042 | Syllable count validation | SUCCESS | P8 | Char→syllable equivalent |
| E043 | Cognacy comparison | SUCCESS | P9 | Bal 40.3% > Jav 33.0% PMP |
| E044 | Malagasy burial botany | SUCCESS | P9 | Plumeria=New World, Canarium=pan-AN |

### Multi-Domain Convergence (E048-E060)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E048 | Multi-domain convergence | SUCCESS | L5 | partial rho=+0.162, p=0.038 (FDR marginal) |
| E049 | Maritime vocabulary | SUCCESS | P8,P9 | Maritime #2 conserved domain |
| E050 | Canarium GBIF distribution | SUCCESS | P5,P9 | Pan-AN aromatic confirmed |
| E051 | Toponymic substrate | SUCCESS | P8,P11 | 25,244 villages, 57.7% pre-Hindu |
| E052 | Sunda Shelf bathymetry | SUCCESS | L2 | 2.09M km² exposed at LGM |
| E053 | aDNA taphonomic gap | CONDITIONAL | L1 | 0/84 Java aDNA (p=0.047, FDR marginal) |
| E054 | Pan-AN cognacy | SUCCESS | P9 | 1,309 langs. Global reversed, local confirmed |
| E055 | Convergence synthesis | SUCCESS | All | 27 experiments catalogued |
| E056 | Candi × toponym crossref | SUCCESS | P7,P11 | MW p=0.007 |
| E057 | Genre taphonomy deep dive | SUCCESS | L5 | +63.9pp organic C8→C9. L5 verified |
| E058 | Kakawin NLP | SUCCESS | P5,P8 | Agriculture 91% native, religion 86% Sanskrit |
| E059 | Fieldwork targets | SUCCESS | P1,P2 | Top 10 GPS at Kelud, 13.1 mm/yr |
| E060 | Pre-400 CE reconstruction | SUCCESS | All | 8 channels, 6 domains |

### Advanced Spatial & Temporal (E061-E067)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E061 | Script simplification | CONDITIONAL | P8 | MW p=0.027 |
| E062 | Visibility curve | CONDITIONAL | P5 | PC1=51.3%, C8 dark century |
| E063 | Domain conservation | SUCCESS | P8,P9 | KW p<0.001 |
| E064 | Master evidence table | SUCCESS | All | 50 experiments catalogued, revision ammo |
| E065 | Candi spatial analysis | SUCCESS | P7,P11 | Zone A 17.9× overrepresented |
| E066 | Candi archaeoastronomy | SUCCESS | P11 | 85% equinox, p=4.9e-14 |
| E067 | Volcanic toponyms | INFO NEG | P11 | No proximity effect (rho=+0.14, p=0.15) |

### Meta-Audit & Adversarial (E068-E070, E081, E085-E087)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E068 | FDR meta-audit | SUCCESS | All | 30/41 survive BH (73%) |
| E069 | **ADV-3 Survey intensity** | **SUCCESS** | P1 | **p=0.0015, VOLCARCH survives control** |
| E070 | Colonial literature mining | SUCCESS | D1 | 52 entries, 32 depths |
| E081 | **ADV-2 Non-volcanic control** | INCONCLUSIVE | L1 | Fisher p=0.760. Cave bias universal |
| E085 | **ADV-4 Substrate noise** | **SUCCESS** | L4 | **p=0.0000, z=11.05** |
| E086 | **ADV-1 Japan comparanda** | PARTIAL | L1 | Survives with survey intensity constraint |
| E087 | **ADV-5 Negative control** | MIXED | P8 | C5 AUC=0.713 (GREY ZONE) |

### Deep Exploration (E073-E080)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E073 | Spatial vs linguistic meta-test | SUCCESS | L4 | p=0.008, behavioral not lexical |
| E074 | DHARMA deep NLP | SUCCESS | P5,P8 | 49% indigenous admin vocab |
| E075 | Sedimentation burial model | SUCCESS | L1 | r=0.951, 32.3% cells >1m |
| E076 | Satellite NDVI anomaly | CONDITIONAL | L1 | 2.5× variance (p=0.46, low N) |
| E078 | Eruption-inscription correlation | SUCCESS | L6 | 6.3× deficit, p=0.035 |
| E079 | Darkness index | SUCCESS | L6 | Invisible Millennium 1.9× darker |
| E071 | Pre-400 CE evidence database | SUCCESS | L1,P11 | 40+ entries, 8 domains |
| E080 | Fieldwork targeting | SUCCESS | P1,P2,P11 | 20 targets, 6 zones near Kelud/Arjuno |

### Inscription Spatial (E082-E084)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E082 | Inscription georeferencing | SUCCESS | P5,P11 | 182/268 (67.9%) geocoded |
| E083 | Tephra-archaeological correlation | SUCCESS | L1,P11 | 51 pairs, mean depth 3.41m |
| E084 | Inscription-volcano spatial | SUCCESS | P11 | MW p=5.2e-08, post-929 shift |

### Textual Archaeology NLP (E088-E090)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E088 | Computational textual archaeology | SUCCESS | P16 | 27 refs, 9 traditions, convergence p<0.00001 |
| E089 | Expanded textual corpus | SUCCESS | P16 | **v3: 106 refs**, 12 traditions, 346 entities |
| E090 | Transformer NLP | MIXED | P16 | 4/6 positive. UMAP 78% cross-trad, convergence 4/5 |

### Colonial NLP Mining (E091)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E091 | OV colonial NLP extraction | SUCCESS | P1, P7, D1 | 22,162 mentions, 742 volcanic, 94.2% DS-1 recovery |

### Volcanic Comparanda & Literature (E092-E093)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E092 | Volcanic archaeology comparanda | SUCCESS | P1, fieldwork | 28 sites worldwide, methodology blueprint for Zone B/C |
| E093 | Indonesian lit mining | SUCCESS | P1, P2, fieldwork | 65 publications, GPR leads at Trowulan/Liyangan/Sambisari |

### DHARMA Semantic NLP (E094, E096)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E094 | DHARMA semantic search | PENDING GPU | P5, P8 | SBERT on 269 Old Javanese inscriptions — first ever |
| E096 | DHARMA diachronic BERTopic | PENDING GPU | P5, P8, P16 | First BERTopic on any epigraphy. Pre/post-929 CE |

### Anomaly Detection & Literature DB (E097-E098)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E097 | Anomaly detection settlement model | SUCCESS | P1, P2 | **65% overlap** with E080 targets. Kelud focus. TRI top feature |
| E098 | Systematic literature database | SUCCESS | P1 | 69 sed. rates, 29 buried sites, 20 GPR surveys worldwide |

### E090 v5 Update

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E090 v5 | Full transformer NLP (200 entries) | PENDING GPU | P16 | BERTopic reactivated, 8 concept groups, v2→v5 delta |

---

## Revisit Candidates

Experiments that could yield better results with new data, methods, or context.

| ID | Current Status | Why Revisitable | Trigger |
|----|---------------|-----------------|---------|
| E017 | FAILED | Needs per-volcano calibration (Tephra2/FALL3D) | Geologist co-author joins |
| E018 | INCONCLUSIVE | Cave-site confound invalidated test design | Redesign with site-density approach |
| E024 | FAILED | n=18 too small, but burial gradient visible | More borehole data published |
| E032 | FDR casualty | p=0.042 uncorrected, seasonality signal plausible | Larger eruption dataset |
| E038 | INFO NEG | Core vocab too stable; could try function words or phrasal units | New linguistic method |
| E039 | INFO NEG | VCS fails globally but works locally (Java/Bali) | Reframe as local phenomenon |
| E048 | FDR marginal | Partial correlation p=0.038 barely fails BH | Larger inscription corpus |
| E053 | FDR casualty | p=0.047 uncorrected, aDNA gap is real | More aDNA studies published |
| E076 | Low N | Concept works (2.5× variance) but needs 20+ sites | Expand to 20+ candi |
| E081 | INCONCLUSIVE | N=13 control too small; cave bias universal | More non-volcanic region data |
| E087 | GREY ZONE | C5 AUC=0.713 from documentation gap, not substrate | ABVD expanded; redesign detector |
| E090/EXP4 | ~~WEAK~~ **ADDRESSED** | BERTopic reactivated in E090 v5 (200 entries) | Run v5 script |
| E090/EXP6 | NEGATIVE | NLI wrong tool; try entity-level comparison | Different NLP approach |

---

## Cathedral Findings (survive ALL scrutiny)

These are the strongest results that survive FDR correction and adversarial testing.

| ID | Finding | p-value | Adversarial? |
|----|---------|---------|-------------|
| E066 | Candi equinox orientation | 4.9e-14 | Trivially true but quantified |
| E051 | Toponymic substrate | 5.1e-14 | Needs linguist validation |
| E065 | Zone A overrepresentation | <1e-6 | Needs population control |
| E084 | Inscription-volcano divergence | 5.2e-08 | Clean, novel |
| E085 | Substrate signal vs noise | <0.0001, z=11.05 | ADV-4 PASSED |
| E069 | Volcanic signal vs survey | 0.0015 | ADV-3 PASSED |
| E083 | Tephra-site correlation | Independent dataset | Clean |

---

## Missing Documentation

*(All experiments now documented.)*

---

*This is the canonical experiment index. Keep it updated when experiments are added or statuses change.*
*Auto-scan: `python tools/scan_experiments.py` (partial — supplements, does not replace this file).*

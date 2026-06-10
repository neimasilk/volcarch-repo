# Experiment Index

**Last updated:** 2026-04-16 (Session 17 — Philippines deep comparison + ME#14)
**Total:** 201 entries (E001-E201, minus E021/E045-E047/E072/E077 unused, E124/E125 SUPERSEDED, E180 skipped)
**Regenerate:** `python tools/scan_experiments.py` (auto-scan, partial — manual review needed)

**Convention:** Every experiment gets a README.md with hypothesis, method, result, conclusion, status.
Failed experiments are NOT deleted — they are documented and may be revisited.

---

## Experiment Type Summary (ME#13 honest count)

| Type | Tag | Count | Can genuinely fail? | Examples |
|------|:---:|:-----:|:---:|---|
| **Novel hypothesis test** | [H] | **~28** | Yes | E069, E085, E108, E110, E178, E183, E189 |
| Robustness/validation | [R] | ~25 | Yes (rarely do) | E115, E121, E159, E176, E185 |
| Database/compilation | [D] | ~30 | No | E001, E070, E082, E091, E181 |
| NLP pipeline development | [P] | ~20 | Partially | E090, E094, E141, E160, E165 |
| Model iteration (superseded) | — | ~15 | N/A | E007-E012, E124-E125 |
| Synthesis/figure/methodology | [S] | ~25 | No | E055, E119, E162, E168, E174 |
| Empty/unused IDs | — | ~8 | N/A | E021, E045-E047, E072, E077 |

**Honest reporting:** "194 experiment entries including ~31 novel hypothesis tests, ~25 robustness checks, and ~30 database compilations." The ~28 novel tests and ~25 robustness checks are the scientifically meaningful core. Database compilations and syntheses are infrastructure.

## Status Summary

| Status | Count | Meaning |
|--------|-------|---------|
| SUCCESS | ~100 | Hypothesis supported or useful result |
| INFO NEG | ~10 | Negative result that IS informative |
| CONDITIONAL | 4 | Partially supported, caveats |
| INCONCLUSIVE | 3 | Cannot determine, need more data |
| MIXED | 3 | Multiple sub-experiments, mixed results |
| PARTIAL | 3 | Survives with scope restriction |
| FAILED | 2 | Hypothesis rejected or method broken |
| COMPLETE (foundation) | 6 | Data/infrastructure, not hypothesis-driven |
| SUPERSEDED | ~10 | Replaced by later iteration |

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
| E017 | Tephra POC (Pyle 1989) | **FAILED** | ~~P3~~ | 1/4 sites pass. Discontinued P3 |
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
| E026 | Pararaton volcanic corr. | SUCCESS | ~~P14~~ | p=0.037 (Bonferroni eliminates → P5 supporting material) |
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
| E059 | Fieldwork candidates | SUCCESS | P1,P2 | Top 10 GPS at Kelud, 13.1 mm/yr |
| E060 | Pre-400 CE reconstruction | SUCCESS | All | 8 channels, 6 domains |

### Advanced Spatial & Temporal (E061-E067)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E061 | Script simplification | CONDITIONAL | P8 | MW p=0.027 |
| E062 | Visibility curve | CONDITIONAL | P5 | PC1=51.3%, C8 dark century |
| E063 | Domain conservation | SUCCESS | P8,P9 | KW p<0.001 |
| E064 | Master evidence table | SUCCESS | All | 50 experiments catalogued, revision support material |
| E065 | Candi spatial analysis | SUCCESS | P7,P11 | Zone A 17.9× overrepresented |
| E066 | Candi archaeoastronomy | SUCCESS | P11 | 85% equinox, p=4.9e-14 |
| E067 | Volcanic toponyms | INFO NEG | P11 | No proximity effect (rho=+0.14, p=0.15) |

### Meta-Audit & Critical (E068-E070, E081, E085-E087)

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
| E090 | Transformer NLP | **SUCCESS** (v5) | P16 | v5: 16 BERTopic topics, 8/8 converge, VOLCANO z=7.39 |

### Colonial NLP Mining (E091)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E091 | OV colonial NLP extraction | SUCCESS | P1, P7, D1 | 22,162 mentions, 742 volcanic, 94.2% DS-1 recovery |

### Volcanic Comparanda & Literature (E092-E093)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E092 | Volcanic archaeology comparanda | SUCCESS | P1, fieldwork | 28 sites worldwide, methodology blueprint for Zone B/C |
| E093 | Indonesian lit mining | SUCCESS | P1, P2, fieldwork | 65 publications, GPR leads at Trowulan/Liyangan/Sambisari |

### DHARMA Semantic NLP (E094, E095, E096)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E094 | DHARMA semantic search | **SUCCESS** | P5, P8, P16 | 4 clusters, volcanic themes rarest (0.244), C11-C12 semantic rupture |
| E095 | Cross-lingual XLM-R/ML-SBERT | **SUCCESS (MIXED)** | P16 | Validates E094 (rho=0.336). XLM-R: embedding collapse. ML-SBERT: volcanic silence confirmed. #99 |
| E096 | DHARMA diachronic BERTopic | **SUCCESS** | P5, P8, P16 | 929 CE topic shift p=0.0003. Royal surges, ritual vanishes |

### Anomaly Detection & Literature DB (E097-E098)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E097 | Anomaly detection settlement model | SUCCESS | P1, P2 | **65% overlap** with E080 targets. Kelud focus. TRI top feature |
| E098 | Systematic literature database | SUCCESS | P1 | 69 sed. rates, 29 buried sites, 20 GPR surveys worldwide |
| E099 | Eruption × inscription temporal | INCONCLUSIVE | P5, P11 | Decade anti-corr rho=-0.26 p=0.013, but GVP too sparse (13 events) |
| E100 | Coastal-highland visibility | **SUCCESS** (rejected H) | P1, P2 | Monotonic density increase with elevation (18.6× coast→mountain). No inverse-U. |
| E101 | Burial depth multivariate model | PARTIAL | P1 | Eruption freq predicts depth (rho=0.373, p=0.012). Multivariate overfits (N=45). |
| **E102** | **Vocabulary × burial depth nexus** | **SUCCESS** | **P5, P8, P1** | **Indigenous ratio × depth rho=0.456 (length-controlled) p<0.0001. Sanskrit-driven.** |
| E103 | Pre-Indic spatial gradient | SUCCESS | P5, P9 | Temporal trend rho=0.781 ONLY at 20-40km (court zone). 929 CE shift zone-specific. |
| E104 | Court zone multi-dataset | SUCCESS | P7, P5, P11 | Candi peak 0-10km, inscriptions peak 20-30km. Fisher OR=1.86, p=0.012. |
| E105 | Topic × geography | SUCCESS | P5, P7, P9 | Sanskrit 72% in court zone. Post-929 shifts to periphery. Two Javas model. |
| E106 | Colonial Two Javas validation | SUGGESTIVE | P17, P1 | N=43 colonial entries. Court-zone bias confirmed (58%). Volcanic context drops with distance. p=0.217 (low N). |

### E090 v5 Update

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E090 v5 | Full transformer NLP (200 entries) | **SUCCESS** | P16 | 16 BERTopic topics, 8/8 converge, VOLCANO z=7.39 |

### Structural Audit Experiments (E107-E109)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| **E107** | **ADV-5 Re-examination** | **SUCCESS** | **P8, L4** | **C5 = Mon-Khmer substrate, NOT documentation artifact. All 6 MK predictions confirmed (p<0.001). E027 UPGRADED.** |
| **E108** | **Demographic null model** | **SUCCESS** | **All** | **Java pre-400 CE carrying capacity 590K-3.9M. Archaeological gap 3,220×. Null hypothesis REJECTED.** |
| E109 | Forward simulation | MIXED | P1, L1 | Survey-burial confound: density INCREASES with depth. τ=∞, ρ=181m. 824 hidden sites. Survey deficit > burial. |
| **E110** | **Visibility cascade model** | **SUCCESS** | **All** | **5-factor cascade predicts 0.058% visible vs 0.031% observed (1.9×). Survey 40× leverage. West Java decisive case.** |
| E111 | Script diffusion timing | SUCCESS | L3 | Java's 660yr script adoption lag = 57th percentile globally. Normal, not anomalous. |
| E112 | Vocabulary archaeology | SUCCESS | P18, L3 | Ghost writing (PAN *surat indigenous). 9 cultural domains reconstructed. Agriculture 91% native vs Religion 86% Sanskrit. |
| **E113** | **Inscription sophistication** | **SUCCESS** | **P18, L3** | **EARLY_PEAK. No learning curve. Hapax ratio p=0.006, Sanskrit phonology p<0.001 (early > mature). Pre-existing organic-media literary tradition.** |
| E114 | Pre-literate comparanda | SUCCESS | P18, L3 | Nusantara #1/10 pre-literate societies (CCI=23, z=2.12). Exceeds Cahokia, Great Zimbabwe. |

### Sensitivity & Robustness (E115)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| **E115** | **Cascade sensitivity analysis** | **SUCCESS** | **P1, All** | **Monte Carlo 100K: 92% of runs within 10× of observed. Correlation-robust (all scenarios 90.5-93.2%). Most uncertain: Survey Coverage (360% range). Model robust to parameter uncertainty AND factor non-independence.** |

### Testable Predictions (E116)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| **E116** | **Testable predictions from cascade** | **SUCCESS** | **P1, All** | **20 targeted GPR → expect 2.5 finds [0,6] 95% CI, P(zero)=7%. Framework IS falsifiable. $40K-100K for decisive test.** |

### Archaeological Onset Analysis (E117)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| **E117** | **Archaeological record onset analysis** | **SUCCESS** | **P1, P18** | **Detection horizon: surface survey reaches ~1900 CE only. Pre-400 CE sites at 6.5m+ depth. Zero open-air volcanic interior sites pre-400 CE. Pattern consistent with VOLCARCH but also with absence.** |

### Information Gain (E118)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| **E118** | **Information gain from volcanic context** | **SUCCESS** | **P1, P2** | **3.5× search efficiency, 29% entropy reduction, $16.7K savings per first-find. Survey deficit is the bigger PROBLEM; volcanic context is the better SOLUTION.** |

### Synthesis (E119)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| **E119** | **Synthesis figure (burial depth vs time)** | **SUCCESS** | **P1, All** | **One figure tells the whole story: burial depth diagonal × detection horizons × known sites. Pre-400 CE at 6.5m+ = beyond all standard methods. Data ready, matplotlib rendering post-mudik.** |

### Cascade Robustness & Sensitivity (E120-E122)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| **E120** | **Cascade stress test** | **SUCCESS** | **P1, P18** | **F3 (survey) is ONLY structurally necessary factor. Safe width 0.133. Removal causes 74.7× overshoot.** |
| **E121** | **Robustness battery (W1+W2)** | **SUCCESS** | **All** | **7/8 ROBUST (88%). E031 Zone A extraordinary: CI [23.7, 26.8]. E005 FRAGILE (CI crosses zero).** |
| **E122** | **Gap sensitivity analysis** | **SUCCESS** | **P1** | **P(gap<10×)=0.0% in 100K MC. Even at HG density (0.1/km²), gap=19×. Gap existence is not parameter-dependent.** |

### Cross-Geographic & Global Comparanda (E123, E126)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E123 | Philippines comparison | SUCCESS (MODERATE) | P1, P18 | 4.6× fewer volcanoes = slightly better record. Java has ZERO open-air volcanic interior pre-400CE; Philippines has 2. |
| **E126** | **Global volcanic archaeology** | **SUCCESS** | **P1, P18** | **Java globally unique: only region with 1M+ yr occupation + zero pre-400CE open-air sites. 20 buried sites worldwide compiled.** |

### Empty/Superseded (E124, E125)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| ~~E124~~ | ~~Survey asymmetry~~ | SUPERSEDED | — | Empty directory. Superseded by E129. |
| ~~E125~~ | ~~Delpher pilot~~ | SUPERSEDED | — | Empty directory. Superseded by E141. |

### Historiographic & External Evidence (E127, E131)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| **E127** | **Ancient external references** | **SUCCESS** | **P1, P16, P18, P19** | **15 sources from 5 traditions confirm pre-400CE Nusantara. World knew Java for 2,500 years before local record begins.** |
| E131 | Writing adoption timeline | SUCCESS | P1, P19 | Nusantara 400 CE = rank 4/6 in SE Asia (not outlier). Korea/Japan also 400 CE. PAN *surat = 5000 BP. |

### Independent Calibration & Replication (E128, E135)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| **E128** | **Colonial OV depth analysis** | **SUCCESS** | **P1, P21** | **Median 2.50m = identical to E083 (p=0.54). Two independent datasets converge. 15 new calibration points.** |
| E135 | Organic preservation model | SUCCESS | P1, P19 | F2 independently validated: 0.229 vs E110's 0.20 (within 15%). Stone 99.8%, bamboo ~0% at 1600yr. |

### Survey Bias & Archaeological Record (E129, E140, E146)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E129 | Survey asymmetry quantification | SUCCESS | P1, P18 | **73% of 391 known sites are temples.** Settlements = 1.3% (5 sites). Temples cluster closer to volcanoes (14.3 vs 25.8 km). |
| E140 | Material culture index | SUCCESS | P1, P19 | 60% organic material in 268 inscriptions. Bamboo (84), lontar (71), cloth. Archaeological Java = elite Java only. |
| **E146** | **Comparative inscription density** | **SUCCESS** | **P1, P5, P19** | **Java density 0.208/1000km²/century (rank 7/8). Non-volcanic regions 30× higher. Bali (2 volcanoes) 12× higher.** |

### Linguistic Substrate & Inscription Analysis (E130, E134, E147)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E130 | Substrate interpretability | SUCCESS | P8, P19 | 438 pre-Indic words. ACTION domain 45.2% substrate. Glottal stops 2× more frequent. Tolaki highest (64.1%). |
| E134 | Inscription chronology gap | SUCCESS | P5, P8, P16, P19 | C8 paradox: peak production (55) but lowest pre-Indic (0.5%). Hyang resurgence C8→C11. Genre explosion 1→396 words. |
| E147 | Inscription length analysis | SUCCESS | P5, P8 | Longer = more pre-Indic (ρ=0.587, p<0.0001). C8 median 1 word → C10 median 431 words. |

### Spatial Modeling (E132)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E132 | Sedimentation rate prediction map | PARTIAL | P1, P22 | RMSE 2.54 mm/yr (55% error). Systematically underpredicts. Sketch-level — needs FALL3D for P22. |

### Synthesis & Bayesian Integration (E133, E136)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E133 | Complete synthesis argument | SUCCESS (SYNTHESIS) | All | 8 evidence lines integrated. Gap 3,220×, cascade 0.058%, P(zero GPR)=7%. "Strongest possible case WITHOUT fieldwork." |
| E136 | Bayesian integration | SUCCESS (ILLUSTRATIVE) | All | Composite BF=72B:1 (estimated, not computed). Posterior robust even at 10× reduction. **Use as framework, not proof.** |

### Discovery & Detection Models (E137, E138, E139)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E137 | Accidental discovery rate | PARTIAL | P1 | 600× overprediction. F4 (recognition) likely 0.0007 not 0.40 for ACCIDENTAL discovery. Model concept valid, params need calibration. |
| E138 | Detection probability by method | SUCCESS | P1, P22 | ERT optimal for 7m depth (P=0.4, $15K/km²). GPR fails beyond 3m. Optimal 3-phase: $35-70K. |
| E139 | Cost-benefit fieldwork strategies | SUCCESS | NatGeo, DRPM | Cheapest: $6K (20 boreholes). Best value: $40K GPR (4 expected finds). Definitive: $100K multi-method. |

### Delpher Colonial Pipeline (E141-E143)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E141 | Delpher extraction pipeline | SUCCESS (Phase 1) | P21 | 529 records from KB SRU API. 30 high-relevance. 5 major newspaper articles with archaeological finds. |
| E142 | Delpher full-text NLP | SUCCESS (Phase 2) | P21 | 48 finds extracted. 33 with location. 40 with material. Only 1 with depth (critical limitation). |
| E143 | Delpher spatial cross-reference | SUCCESS | P21 | 13/33 (39%) within 30km of E080 fieldwork candidates. Malang+Modjokerto cluster validates Kelud/Arjuno zone. |

### Visualization & Temporal Analysis (E144, E145)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| E144 | Evidence timeline figure | SUCCESS | P1, P19 | 3,400-year gap visualization. External evidence above, zero local record below. |
| E145 | Eruption frequency vs visibility | **INFO NEG** | P1 | **Eruptions POSITIVELY correlate with inscriptions (ρ=+0.908, p=0.0001). Taphonomy is SPATIAL not TEMPORAL. Contradicts naive L6.** |

### Mata Elang #11 Resolution (E148-E152)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| **E148** | **Sunda Shelf paleo-drainage reconstruction** | **SUCCESS** | **P18, P19** | **4,179x survey gap on the drowned shelf. L2 is no longer conceptually empty.** |
| **E149** | **Eruption-inscription paradox reconciliation** | **SUCCESS** | **P1, P17** | **Temporal positive correlation and spatial deficit are both true: politics drives the century signal, taphonomy drives the distance signal.** |
| **E150** | **Babad Tanah Jawi substrate NLP** | **SUCCESS** | **P8, P19** | **Top lexical stratum is 83.9% native/non-Sanskrit. Chronicle backbone is GRAMMAR > OTHER > ACTION, breaking DHARMA monoculture.** |
| **E151** | **Megalithic distribution vs volcanic zones** | **SUCCESS** | **P1, P19** | **All 4 requested megalithic cases lie within 35 km of an active volcano. Stone survives 4/4, organic settlement 0/4.** |
| **E152** | **Post-929 Mataram → East Java natural experiment** | **SUCCESS** | **P1, P17** | **Post-929 inscriptions are 12.7 km farther from volcanoes, 187 km farther east, longer, and more pre-Indic.** |

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
| ~~E048~~ | ~~FDR marginal~~ **RESCUED by E154** | E154 re-audit: p=0.038 now SURVIVES BH at threshold 0.039. No longer a casualty. | E154 done |
| E053 | FDR casualty | p=0.047 uncorrected, aDNA gap is real | More aDNA studies published |
| E076 | Low N | Concept works (2.5× variance) but needs 20+ sites | Expand to 20+ candi |
| E081 | INCONCLUSIVE | N=13 control too small; cave bias universal | More non-volcanic region data |
| ~~E087~~ | ~~GREY ZONE~~ **RESOLVED by E107** | C5 = Mon-Khmer substrate (not documentation artifact). ADV-5 reclassified. | E107 done |
| E090/EXP4 | ~~WEAK~~ **ADDRESSED** | BERTopic reactivated in E090 v5 (200 entries) | Run v5 script |
| E090/EXP6 | NEGATIVE | NLI wrong tool; try entity-level comparison | Different NLP approach |

---

## Cathedral Findings (survive ALL scrutiny)

These are the strongest results that survive FDR correction and robustness testing.

| ID | Finding | p-value | Critical? |
|----|---------|---------|-------------|
| E066 | Candi equinox orientation | 4.9e-14 | Trivially true but quantified |
| E051 | Toponymic substrate | 5.1e-14 | Needs linguist validation |
| E065 | Zone A overrepresentation | <1e-6 | Needs population control |
| E084 | Inscription-volcano divergence | 5.2e-08 | Clean, novel |
| E085 | Substrate signal vs noise | <0.0001, z=11.05 | ADV-4 PASSED |
| E069 | Volcanic signal vs survey | 0.0015 | ADV-3 PASSED |
| E083 | Tephra-site correlation | Independent dataset | Clean |
| E110 | Cascade predicts 0.058% vs 0.031% observed | 1.9× ratio | E115 robustness check |
| E115 | Cascade robust under MC + correlation | 92% within 10× | Addresses independence assumption |
| **E116** | **Testable predictions: 20 GPR → [0,6] finds** | P(zero)=7% | **Falsifiability established** |
| **E108** | **Demographic gap 3,220×** | Population model | **Null hypothesis test** |
| **E107** | **ADV-5 resolved: C5 = MK substrate** | p<0.0001 (6 tests) | **Upgrades E027/L4** |
| **E122** | **Gap robust: P(gap<10×)=0.0%** | Monte Carlo 100K | **Gap existence not parameter-dependent** |
| **E128** | **Independent depth replication** | MW p=0.54 (identical medians) | **Two independent datasets converge** |
| **E126** | **Java globally unique archaeological gap** | Comparative compilation | **Only 1M+ yr occupation region with zero pre-400CE open-air** |
| **E129** | **73% temple survey bias** | 277/391 sites | **Explains 40× survey deficit leverage** |
| **E135** | **F2 independently validated** | 0.229 vs 0.20 | **Within 15% of cascade estimate** |

### Mata Elang #12 Experiments (E154-E157)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| **E154** | **FDR Re-Audit at 153 experiments** | **SUCCESS** | **All** | **83 tests, 65 survive BH (78.3%, up from 73.2%). E048 RESCUED. Only 2 casualties remain (E032, E053).** |
| **E155** | **Cross-Regional Cascade Validation** | **SUCCESS (with caveats)** | **P1, P17, P18** | **Cascade correctly predicts rank order of archaeological visibility across 5 regions (Spearman rho=1.0, p=0.017). F3 (survey) most differentiating factor. MC: P(rho>0.5)=99.6%.** |
| **E156** | **Sunda Shelf Population Displacement Model** | **SUCCESS** | **P1, P18, P19** | **L1xL2 "Double Erasure": ~94,000 people pushed from drowning Sunda Shelf INTO volcanic zones via river corridors. Burial depth 44m. West Java decisive case PREDICTED by model.** |
| **E157** | **Ethnographic Volcanic Analog** | **SUCCESS** | **P1, P17, P18, P19** | **F4=0.43 (Liangan, Hindu) / 0.20 (pre-Hindu). F2=0.21 (weighted ethnographic). Three independent F2 estimates converge within 15%. 32% of modern village material culture INVISIBLE after burial.** |
| **E158** | **Steelman Counter-Arguments** | **SUCCESS** | **All** | **Cascade model (E110) = weakest flank (5 params, 1 data point). Cathedral findings robust. E066 is "trivially true." Recommendation: lead with cathedral findings, cascade as framework.** |
| **E159** | **Robustness Battery (5 cathedral findings)** | **SUCCESS** | **All** | **5/5 ROBUST under bootstrap (10K), permutation (10K), jackknife. E051 metric sensitivity discovered: court distance, not volcano distance. Zone A overrep: 13.5x, p=5.3e-64.** |
| **E160** | **GPU Deep Semantic Analysis (DHARMA)** | **SUCCESS** | **P5, P8, P16, P17** | **768d embeddings (all-mpnet-base-v2, RTX 4080). Volcanic silence rank 8/10. C8 = darkest century. 929 CE rupture p=0.012 (z=3.04). High pre-Indic = practical governance, low = religious abstraction.** |
| **E161** | **Bali as Within-Indonesia Comparandum** | **SUCCESS** | **P1, P17, P18** | **5/5 VOLCARCH predictions confirmed. ALL pre-Hindu sites on non-volcanic coast (4/4). Cascade predicts 14.3x ratio, observed ~12x (18% error). Bali = successful test case.** |
| **E162** | **State of Evidence Synthesis (161 experiments)** | **SUCCESS (SYNTHESIS)** | **All** | **Complete evidence table: 8 cathedral findings, 9 strong findings, 10+ supporting, 6 limitations. One-paragraph argument. For collaborator/reviewer/funder briefing.** |
| **E163** | **Sumatra Applicability Test** | **SUCCESS (NUANCED)** | **P1, P17, P18** | **Cascade predicts Sumatra 0.49x Java visibility. Observed: 0.14-0.19x (model overpredicts). Additional erasure: peat, delta, forest. Sriwijaya paradox = VOLCARCH thesis without volcanism.** |
| **E164** | **Dong Son Drum Distribution** | **SUCCESS** | **P1, P17, P19** | **6/6 Java drums in volcanic zones. Tuban drum (300 BCE, Heger II) = direct pre-Hindu evidence in volcanic E. Java. Only bronze survives all 5 cascade factors. Accidental discovery pattern confirms F3.** |
| **E165** | **Ghost Vocabulary in Old Javanese** | **SUCCESS** | **P5, P8, P16, P17, P19** | **95,709 tokens from 233 inscriptions (original OJ, not translations). 230 ghost words (14% of early vocab vanishes). "aku" (I) disappears after C8. Volcano zone 4.6x more exclusive vocabulary. Indigenous% jumps C8(64%)→C9(96%).** |
| **E166** | **Burial Depth Prediction Map (30m GeoTIFF)** | **SUCCESS** | **P1, P2, P17** | **Full East Java burial depth map. 12,811 km2 in Zone B (GPR-detectable 1-3m). 2,709 km2 in Zone D (>6m). GeoTIFF output for GIS overlay. Pre-400 CE sites: 21.5% of area has >1m burial.** |
| **E167** | **Priority Fieldwork Map (integrated)** | **SUCCESS** | **P1, P2, P17** | **Combines suitability + burial feasibility + novelty. Top 1% = 994 km2. Targets cluster Lawu W flank at 1-1.2m depth. GeoTIFF output. 89.3% of E. Java has no known sites within 5km.** |
| **E168** | **The Invisible Civilization (reconstruction)** | **SUCCESS (SYNTHESIS)** | **P18, P19, All** | **Full reconstruction: 500K-1M population, bamboo stilt villages, wet rice, bronze metallurgy, organic-media writing (PAN *surat 5000BP), hyang cosmology, stratified chiefdoms. 99.9% material culture lost. "Aku was always there."** |
| **E169** | **Inscription Desert Analysis** | **SUCCESS** | **P1, P17** | **77.1% of expected inscription zone is EMPTY. 3 deserts: Malang/Kelud (9,630 km2), Lawu (3,494 km2), Semeru/Bromo (3,614 km2). Deserts = shadow of Two Javas divide.** |
| **E170** | **TWI-Enhanced Burial Depth Model** | **SUCCESS (MARGINAL)** | **P1, P2** | **TWI refinement adds <2% over distance model (rho=0.986). Distance dominates at regional scale. GeoTIFF saved.** |
| **E171** | **Prediction Registry (5 formal predictions)** | **SUCCESS** | **All** | **5 GPS-precise predictions with depth, age, method, cost ($3K total), and explicit falsification conditions. For Zenodo DOI deposit. P(>=1 positive) = 55%.** |
| **E172** | **Dynamic Population Model (40K BP — 1600 CE)** | **SUCCESS** | **P1, P17, P18** | **50K Monte Carlo. Java 400 CE: median 3.30M (95% CI 1.35-5.51M). 7/7 calibration. Gap 11,008x (vs E108's 3,220x). Logistic growth + migration + catastrophes. Supersedes E108.** |
| **E173** | **Counterfactual "What If Japan"** | **SUCCESS** | **P1, P17, P19** | **1,789 pre-400 CE sites MISSING because no rescue archaeology. Excavation density gap 558x. ~4-5 sites destroyed/year by construction. "The difference is not geology. It is POLICY."** |
| **E174** | **Synthesis Figure (6-panel)** | **SUCCESS (VIZ)** | **All** | **One figure tells the entire VOLCARCH story: population, cascade, burial depth, Two Javas, ghost vocabulary, gap. 535 KB PNG, 200 dpi.** |
| **E175** | **Candi Spatial Statistics** | **SUCCESS** | **P7, P11, P17** | **Clark-Evans R=0.171 (extremely clustered, 5.8x tighter than random). Ripley L peaks at 50 km (volcanic system spacing). NOT exponential, NOT lognormal. Deliberate siting confirmed.** |

### Mata Elang #13 Experiments (E176-E188)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| **E176** | **Cascade Minimal Model Comparison** | **SUCCESS** | **P1, P17, All** | **5-factor cascade is over-parameterized. 5/10 possible 3-factor models bracket observed gap. 83.8% of random 5-factor draws within 10x. AIC: 3-factor (6.73) vs 5-factor (6.25) = marginal improvement. F1 (volcanic burial) is LEAST necessary factor (2/5 minimal models). F3 (survey) is structurally necessary (5/5). Honest reframing: "pedagogically useful but empirically underdetermined."** |
| **E177** | **Sunda Shelf Paleo-Drainage Reconstruction** | **SUCCESS** | **P18, L2** | **First computational L2 model. 340K on shelf at LGM. ~250K displaced toward Java via 3 paleo-river systems (62% draining to Java). 5 entry-point predictions: Surabaya (HIGHEST), Tangerang, Semarang, Jakarta, Cirebon. L1xL2 double stratigraphy predicted at Surabaya.** |
| **E178** | **Philippines Archaeological Density Regression** | **SUCCESS** | **P1, P17, P19** | **Java volcanic = ONLY region with ZERO pre-400 CE sites across 8 SE Asian regions. Philippines volcanic has 25 sites (0.25/1000km2). Multiple regression R2=0.733, volcanic density most negative predictor. KARST is hidden 6th factor: Philippines karst 0.20 vs Java 0.08. Cascade needs karst bypass term.** |
| **E179** | **Factor Independence Test** | **SUCCESS** | **P1, P17** | **F1-F2 coupling (burial preserves organics) shifts cascade 1.7x. F3-F4 coupling 1.8x. Full coupling = 3.0x total shift (from ratio 1.9x to 5.6x). Coupling makes prediction WORSE (further from observed). If Java lahars destroy organics (hot lahar scenario), coupling improves fit to 0.8x. Within E115 MC spread.** |
| **E181** | **Ghost Dictionary** | **SUCCESS** | **P5, P8, P16, P17, P19** | **47 ghost words classified. 55% OJ, 23% SK, 19% PMP. Admin vocab = biggest casualty. 66% completely lost, 26% survive in modern speech. Material terms survive, abstract/admin replaced. "aku" (1st person pronoun) = most symbolically significant ghost — indigenous VOICE silenced from C8.** |
| **E182** | **Karst-Augmented Cascade** | **SUCCESS (PARTIAL)** | **P1, P17** | **Karst bypass improves rank prediction (rho 0.321->0.500) but magnitude calibration poor. Model: P(vis) = cascade + karst*P(cave). Philippines volcanic 0.20 karst vs Java 0.08 = hidden factor. Best P(cave)=0.05 gives rho=0.607.** |
| **E183** | **Register Split Quantification** | **SUCCESS** | **P5, P8, P16, P19** | **85% of ghost words die in C9 (mass extinction). C9=peak indigenous% AND peak ghost deaths (paradox: last breath of old genre). C10 corpus explodes 3.2x -> standardization prunes indigenous terms. Register split maps onto modern ngoko/krama diglossia. "Sanskritization" = KRAMA-IFICATION of written register. Novel finding.** |
| **E184** | **Spatial Autocorrelation (Moran's I)** | **SUCCESS (INFO NEG)** | **P17, P1** | **Moran's I for volcano distance = 0.937 (p<0.001, strongly autocorrelated). Volcano-century correlation (rho=0.490) COLLAPSES after spatial lag correction (rho=-0.198, p=0.111). Temporal claims in P17 may be inflated by spatial dependence. Two Javas SEGREGATION (Mann-Whitney) is more robust than regression. Addresses ME#13 Risk 6 (methodology gap).** |
| **E185** | **Spatially-Constrained Permutation** | **SUCCESS** | **P17** | **Two Javas segregation ROBUST: KS p<10^-8, Cohen's d~2.0 (very large), Cliff's delta~0.97. Permutation (10K) p<0.000001. E184's spatial autocorrelation warning applies to REGRESSION, NOT to TWO-SAMPLE distributional comparison. Core P17 finding survives all spatial tests.** |
| **E186** | **Tengger Ghost Word Cross-Reference** | **SUCCESS** | **P8, P19** | **Tengger IS a linguistic time capsule. ABVD too limited (210 concepts), but literature confirms: "esun"=aku (C8 ghost), "glis" preserved as "nglisik", "hyang", "picis". Pre-krama register survives in volcanic isolate's spoken language.** |
| **E187** | **Proper Spatial Regression (PySAL)** | **SUCCESS (INFO NEG)** | **P17** | **Volcanic distance effect DOES NOT survive spatial regression. OLS beta 0.034 (p=0.002) drops to 0.016 (p=0.094) in Spatial Lag model. Rho=0.620 (strong spatial lag). Two Javas segregation still robust. First proper spatial regression in VOLCARCH.** |
| **E188** | **Mainland SE Asia Comparison** | **SUCCESS** | **P1, P17, P19, All** | **Decisive insight: "400 CE start" = writing diffusion, NOT civilizational birth. Three compounding biases: material (organic vs bronze), survey (OV candi-only vs EFEO systematic), narrative (Indianization as birth). 3,600-year pre-inscriptional gap = preservation bias.** |

### Satellite Archaeology Experiments (E189-E191)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| **E189** | **Satellite Spectral Feasibility** | **SUCCESS (INFORMATIVE)** | **—** | **Sentinel-2 multi-index (NDVI/NDWI/MSAVI) at 15 candi + 5 controls. NDWI p=0.032. First satellite archaeology in volcanic Java.** |
| **E190** | **SAR Feasibility** | **SUCCESS (INFORMATIVE NEGATIVE)** | **—** | **Sentinel-1 C-band SAR at same sites. C-band ruled out (canopy dominates). Cohen's d = -0.92 wrong direction.** |
| **E191** | **Multi-temporal NDWI** | **SUCCESS (INFORMATIVE)** | **—** | **Dry vs wet season NDWI. New metric: delta local variance p=0.066. Candi lvar increases wet season.** |
| **E192** | **NDWI vs Burial Depth Correlation** | **SUCCESS (INFORMATIVE)** | **P1,P17** | **All 4 correlations negative (correct direction). NDWI lvar vs depth rho=-0.389. Depth-signal relationship validated but underpowered (n=15).** |
| **E193** | **Sunda Shelf Entry Points vs Sites** | **SUCCESS** | **P18,L2** | **Sites significantly cluster near entry points (p<0.00001). Surabaya=100th percentile. North/South ratio 1.35 confirmed. 123 double-erasure sites. Addresses ME#13 Risk 4.** |
| **E194** | **Combined Prospection Map** | **SUCCESS** | **P1,P2,P17** | **18/20 targets have 4/5 independent evidence streams converging. T08 (-7.88, 112.30) = hottest target (25 E097 cells + 3 other streams). Kelud + Arjuno clusters.** |

### Population, Etymology, NLP, and Comparative Experiments (E195-E201)

| ID | Title | Status | Paper | Key Result |
|----|-------|--------|-------|------------|
| **E195** | **Detection Horizon Sufficiency** | **SUCCESS** | **P1,P17** | **AHA moment: E117 detection horizon applied to survey coverage. Current surveys cannot see pre-1900 CE at 4mm/yr. rho=+0.53 correlation.** |
| **E196** | **Population Estimation** | **SUCCESS** | **P1,P17,P18** | **Four methods converge: Java had 1-2M people at 400 CE. Expected sites >=694. Observed: 0. Taphonomic suppression >=694x.** |
| **E197** | **Colonial Depth Validation** | **SUCCESS** | **P1,P2** | **10 E070 entries cross-referenced with Delpher. Colonial depth data supports volcanic burial model.** |
| **E198** | **Sago-Rice Etymology** | **SUCCESS** | **P8,P19** | ***sagu > sego phonologically regular. Sundanese "sangu" confirms. Layer 7 (pre-rice subsistence) proposed.** |
| **E199** | **Collective Brain Paradox** | **SUCCESS** | **P18** | **Kremer/Boserup formalized. Innovation gap 25-188x between Java population and archaeological record. Volcanic paradox quantified.** |
| **E200** | **Dutch NER Baseline** | **SUCCESS** | **PhD** | **Standard NER covers ~27% of VOC entities. PhD closes 73% gap. Baseline for NLP research proposal.** |
| **E201** | **Philippines Deep Comparison** | **SUCCESS** | **P1,P17,P18** | **Philippines record is genuinely diverse: ~50% open-air, NOT cave-dominated. 275-340 pre-400 CE sites (55-65% open-air). Philippine volcanic zones retain 25-40 sites. Pinatubo/Iraya prove volcanic burial preserves, not destroys. Strengthens Java taphonomic argument.** |

---

## Missing Documentation

*(All experiments now documented.)*

---

*This is the canonical experiment index. Keep it updated when experiments are added or statuses change.*
*Auto-scan: `python tools/scan_experiments.py` (partial — supplements, does not replace this file).*

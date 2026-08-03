# Experiment Index

**Generated:** 2026-08-03 12:15
**Total:** 214 experiments
**Regenerate:** `python tools/scan_experiments.py`

## Status Summary

- **SUCCESS:** 142
- **UNKNOWN:** 34
- **REVISIT:** 6
- **INFO NEG:** 6
- **IN PROGRESS:** 4
- **FAILED:** 3
- **INCONCLUSIVE:** 3
- **PARTIAL:** 3
- **SUPERSEDED:** 2
- **QUEUED:** 1
- **BLOCKED (needs E001 :** 1
- **READY TO RUN:** 1
- **CONDITIONAL:** 1
- **SUGGESTIVE (low N, d:** 1
- **MIXED:** 1
- **PHASE 1 + 2a COMPLET:** 1
- **PHASE 1 — data pipel:** 1
- **SCOPING — Phase 0 (p:** 1
- **Phase 0 — corpus sec:** 1
- **PRE-REGISTERED (deci:** 1

## By Line of Inquiry

Navigation layer: `lines/<name>/`. See `lines/README.md`. An experiment may serve
several lines — it is listed under each, and its **primary** line is listed first in
the table below. `experiments/` itself stays flat and shared; it is never partitioned.

### `01_spatial` — Predictive modelling & site distribution

**78** experiments (71 primary): E001 · E003 · E004 · E005 · E006 · E007 · E008 · E009 · E010 · E011 · E012 · E013 · E014 · E015 · E016 · E019 · E031 · E059 · E065 · E066 · E076 · E080 · E082 · E084 · E097 · E100 · E103 · E104 · E106 · E108 · E109 · E110 · E115 · E116 · E118 · E120 · E121 · E122 · E124 · E126 · E129 · E139 · E151 · E152 · E153 · E155 · E159 · E163 · E166 · E167 · E171 · E172 · E175 · E176 · E179 · E182 · E183 · E184 · E185 · E187 · E189 · E190 · E191 · E192 · E194 · E195 · E196 · E202 · E209 · E210 · E217 · E218 · E219 · E220 · E221 · E222 · E223 · E224

### `02_taphonomy` — Burial, erosion, exposure

**46** experiments (41 primary): E001 · E002 · E017 · E018 · E020 · E024 · E052 · E069 · E075 · E081 · E083 · E084 · E085 · E086 · E087 · E092 · E101 · E107 · E109 · E117 · E123 · E126 · E128 · E132 · E135 · E137 · E138 · E140 · E148 · E156 · E157 · E161 · E164 · E166 · E170 · E173 · E177 · E178 · E182 · E188 · E193 · E195 · E197 · E201 · E204 · E213

### `03_paleoenv` — Paleo-environmental falsification

**3** experiments (3 primary): E214 · E215 · E216

### `04_language_text` — Language & text

**62** experiments (62 primary): E022 · E023 · E025 · E026 · E027 · E028 · E029 · E030 · E032 · E033 · E034 · E035 · E036 · E037 · E038 · E039 · E040 · E041 · E042 · E043 · E044 · E049 · E050 · E051 · E054 · E056 · E057 · E058 · E061 · E063 · E067 · E074 · E082 · E085 · E087 · E088 · E089 · E090 · E094 · E095 · E096 · E102 · E105 · E107 · E111 · E112 · E113 · E114 · E130 · E131 · E134 · E146 · E147 · E150 · E160 · E165 · E169 · E181 · E186 · E198 · E205 · E208

### `05_archival_nlp` — Colonial archives & NLP

**14** experiments (12 primary): E070 · E091 · E093 · E098 · E125 · E128 · E141 · E142 · E143 · E197 · E200 · E206 · E207 · E211

### `06_thesis` — Original question / synthesis

**27** experiments (25 primary): E048 · E055 · E060 · E062 · E064 · E068 · E071 · E073 · E078 · E079 · E099 · E119 · E127 · E133 · E136 · E144 · E145 · E149 · E154 · E158 · E162 · E164 · E168 · E174 · E179 · E199 · E204

### `07_career` — Career & exposure (no experiments)

*(no experiments — this line's work is not experimental)*

### `external:volcarch-genetics`

Canonical in the companion repo `D:\documents\volcarch-genetics` — **no local
directory**, by design (see `docs/COMPANION_REPOS.md`). Cite as external evidence.

E053 · E203

## All Experiments

| ID | Title | Status | Line | Layer | Paper | Key Metric |
|-----|-------|--------|------|-------|-------|------------|
| E001 | Archaeological Site Geocoding & Density Analysis | IN PROGRESS | 02_taphonomy,01_spatial |  | P1 |  |
| E002 | Eruption History Compilation | IN PROGRESS | 02_taphonomy |  | P1,P3 |  |
| E003 | DEM Acquisition and Processing (Malang Raya) | QUEUED | 01_spatial |  | P2 |  |
| E004 | Site Density vs Volcanic Proximity (First Test of  | BLOCKED (needs E001  | 01_spatial |  | P1 |  |
| E005 | Terrain Suitability Model as Null Hypothesis for H | IN PROGRESS | 01_spatial |  | P1,P2 | rho=-0.991 |
| E006 | Re-analysis with Nominatim-Enriched Site Dataset | SUCCESS | 01_spatial |  | P1 |  |
| E007 | Settlement Suitability Model — Paper 2 MVP | REVISIT | 01_spatial |  | P2 | AUC=0.659; p<0.001 |
| E008 | Settlement Suitability Model v2 — River Distance F | REVISIT | 01_spatial |  | P2 | AUC=0.695; rho=-0.153 |
| E009 | Settlement Suitability Model v3 — SoilGrids Clay + | REVISIT | 01_spatial |  | P2 | AUC=0.664; p<0.001 |
| E010 | Settlement Suitability Model v4 - Target-Group Bac | REVISIT | 01_spatial |  | P2 | AUC=0.711; p<0.001 |
| E011 | Settlement Suitability Model v5 - TGB Parameter Sw | REVISIT | 01_spatial |  | P2 | AUC=0.725; p<0.001 |
| E012 | Settlement Suitability Model v6 - TGB Proxy Enrich | REVISIT | 01_spatial |  | P2 | AUC=0.730; p<0.001 |
| E013 | Settlement Suitability Model v7 - Hybrid Bias Corr | SUCCESS | 01_spatial |  | P2 | AUC=0.768; p<0.001 |
| E014 | Temporal Split Validation — Tautology Stress Test | READY TO RUN | 01_spatial |  |  |  |
| E015 | SHAP Analysis for E013 Best Model | SUCCESS | 01_spatial |  | P2 |  |
| E016 | Zone Classification Map | IN PROGRESS | 01_spatial |  | P1,P2 |  |
| E017 | Tephra POC (Pyle 1989 Calibration for Paper 3) | FAILED | 02_taphonomy |  | P3 |  |
| E018 | Temporal Overlay Matrix — Proof of Concept | INCONCLUSIVE | 02_taphonomy |  | P7 |  |
| E019 | Spatial Distribution Test (Paper 7 Metric 3) | ? | 01_spatial |  |  | p<0.05 |
| E020 | Mini-NusaRC — Preliminary Radiocarbon Database for | ? | 02_taphonomy |  | D2 | p<0.05 |
| E022 | Linguistic Subtraction POC | ? | 04_language_text |  |  |  |
| E023 | Ritual Screening POC — DHARMA Corpus | ? | 04_language_text |  |  |  |
| E024 | Borehole & Buried Site Literature Screening | ? | 02_taphonomy |  |  |  |
| E025 | Quantitative Validation of the Volcanic Ritual Clo | ? | 04_language_text |  |  |  |
| E026 | Pararaton Volcanic Correlation | SUCCESS | 04_language_text |  | P14 | p=0.037 |
| E027 | ML-Based Linguistic Substrate Detection | SUCCESS | 04_language_text |  | P8 |  |
| E028 | Cross-Method Substrate Consensus Analysis | SUCCESS | 04_language_text |  |  |  |
| E029 | Phonological Clustering of Consensus Substrate Can | ? | 04_language_text |  |  |  |
| E030 | Temporal NLP Analysis of Old Javanese Inscriptions | SUCCESS | 04_language_text |  | P14,P5 | p<0.001; rho=+0.502 |
| E031 | Candi Orientation vs Volcanic Peak Alignment | SUCCESS | 01_spatial |  |  | p=3.1e-14 |
| E032 | Pranata Mangsa × Eruption Seasonality | SUCCESS | 04_language_text |  |  | p=0.042 |
| E033 | The Indianization Curve: Sanskrit Vocabulary in Pr | SUCCESS | 04_language_text |  |  | p=0.070; rho=-0.211 |
| E034 | Cerita Panji in Malagasy Literature Search | INFO NEG | 04_language_text |  |  |  |
| E035 | Prasasti Botanical Keyword Expansion | SUCCESS | 04_language_text |  |  | p=0.31 |
| E036 | Hanacaraka Phonological Inventory Mapping | SUCCESS | 04_language_text |  |  |  |
| E037 | Prasasti Dating Model (ML on Undated Inscriptions) | INFO NEG | 04_language_text |  |  |  |
| E038 | Volcanic Vocabulary Semantic Drift Across Austrone | INFO NEG | 04_language_text |  |  | p=0.38; rho=-0.301 |
| E039 | Volcanic Cultural Selection — Cross-Cultural Test | INFO NEG | 04_language_text |  | P11 | p=0.028; rho=+0.145. |
| E040 | Bamboo Civilization — Material Culture in Prasasti | SUCCESS | 04_language_text |  | P1,P7 |  |
| E041 | IPA Approximation Validation | SUCCESS | 04_language_text |  | P8 |  |
| E042 | Syllable Count Validation | SUCCESS | 04_language_text |  | P8 |  |
| E043 | Peripheral Conservatism — Cognacy Comparison | SUCCESS | 04_language_text |  | P9 | p=0.015 |
| E044 | Malagasy Burial Botanical Survey | SUCCESS | 04_language_text |  | P9 |  |
| E048 | Multi-Domain Temporal Convergence Analysis | ? | 06_thesis |  |  | p<0.0001; rho=+0.546 |
| E049 | Maritime & Nature Vocabulary Conservation Across A | ? | 04_language_text |  |  |  |
| E050 | Canarium spp. Global Distribution — Austronesian A | ? | 04_language_text |  |  |  |
| E051 | Java Toponymic Substrate Analysis | SUCCESS | 04_language_text |  | P5,P8,P9 | p=2.2e-20; rho=0.387 |
| E052 | Sunda Shelf Paleo-Drainage Reconstruction | SUCCESS | 02_taphonomy |  |  |  |
| E054 | Pan-Austronesian Cognacy Gradient — Continental-Sc | ? | 04_language_text |  | P9 | p=0.002; rho=-0.088 |
| E055 | Multi-Evidence Convergence Synthesis | ? | 06_thesis | L2,L5 |  |  |
| E056 | Candi Location × Toponymic Substrate Cross-Referen | ? | 04_language_text |  |  | p=0.007; rho=-0.240 |
| E057 | Genre Taphonomy — The 5th Layer of Darkness | ? | 04_language_text |  |  | p<0.000001 |
| E058 | Kakawin (Old Javanese Literary Text) NLP Analysis | SUCCESS | 04_language_text |  | P5,P8 |  |
| E059 | Priority Fieldwork Targets — Where to Dig Next | ? | 01_spatial |  | P2 |  |
| E060 | Pre-400 CE Nusantara Reconstruction | ? | 06_thesis |  |  |  |
| E061 | Script Simplification — Cross-Cultural Validation | SUCCESS | 04_language_text |  |  |  |
| E062 | Temporal Synthesis: Multi-Dimensional Visibility C | SUCCESS | 06_thesis |  |  |  |
| E063 | Semantic Domain Conservation in Austronesian Langu | SUCCESS | 04_language_text |  |  | p=6.82e-04. |
| E064 | Master Evidence Table — Cross-Paper Revision Ammo | ? | 06_thesis |  |  |  |
| E065 | Candi Spatial Analysis — Volcanic Proximity Zones | ? | 01_spatial |  |  |  |
| E066 | Candi Archaeoastronomy — Entrance Orientation vs S | SUCCESS | 01_spatial |  | P11 | p=0.0016 |
| E067 | Volcanic Toponyms — Do Volcanic Place Names Cluste | INFO NEG | 04_language_text |  | P11 | p=0.146; rho=+0.140 |
| E068 | FDR Meta-Analytic Audit | SUCCESS | 06_thesis |  |  |  |
| E069 | Adversarial Experiment Suite — Falsification Tests | SUCCESS | 02_taphonomy |  |  | AUC=0.713; p=0.760 |
| E070 | Colonial Literature Mining — Independent Dataset C | SUCCESS | 05_archival_nlp |  | P1,P11,P2,P7 |  |
| E071 | Pre-400 CE Evidence Database | SUCCESS | 06_thesis | L1,L3,L6 | P11 |  |
| E073 | Spatial vs Linguistic Evidence Meta-Test | ? | 06_thesis |  |  |  |
| E074 | DHARMA Deep NLP — Mining the Invisible Millennium | ? | 04_language_text |  |  |  |
| E075 | Volcanic Sedimentation Burial Model for Java | ? | 02_taphonomy |  |  |  |
| E076 | Satellite NDVI Anomaly Detection at Candi Sites | ? | 01_spatial | L2 |  |  |
| E078 | Eruption-Inscription Correlation — Volcanic Dark P | ? | 06_thesis |  |  | p=0.43; rho=-0.77 |
| E079 | The Archaeological Darkness Index — Grand Synthesi | ? | 06_thesis |  |  |  |
| E080 | Fieldwork Targeting — Priority Zones | SUCCESS | 01_spatial | L1 | P1,P11,P2 |  |
| E081 | ADV-2 Non-Volcanic Control Test | ? | 02_taphonomy | L1 |  | p=0.003 |
| E082 | DHARMA Inscription Georeferencing | ? | 04_language_text,01_spatial |  |  | p=0.148; rho=0.643 |
| E083 | Tephra-Archaeological Correlation Dataset | SUCCESS | 02_taphonomy |  |  |  |
| E084 | Formal Inscription-Volcano Spatial Analysis | ? | 02_taphonomy,01_spatial |  | P11 |  |
| E085 | ADV-4 Substrate Noise Permutation Test | ? | 04_language_text,02_taphonomy | L4 |  | AUC: 0.7599; p=0.760 |
| E086 | ADV-1 Japan Comparanda Test | ? | 02_taphonomy | L1 | P1,P11,P2 |  |
| E087 | Substrate Detector Negative Control | CONDITIONAL | 04_language_text,02_taphonomy | L4 | P8 | AUC=0.762; p=0.0000 |
| E088 | Computational Textual Archaeology — NLP Pipeline | SUCCESS | 04_language_text | L3 | P16 | p=0.000000. |
| E089 | Expanded Textual Corpus | SUCCESS | 04_language_text | L3 | P16 |  |
| E090 | Transformer-based NLP on Ancient Textual Corpus | SUCCESS | 04_language_text | L3 | P16 | z=0.88; F1: 0.650 |
| E091 | Automated NLP Extraction from OV Colonial Reports | SUCCESS | 05_archival_nlp |  | D1,P1,P11,P7 |  |
| E092 | Volcanic Archaeology Comparanda Database | SUCCESS | 02_taphonomy | L1 | P1,P11 |  |
| E093 | Indonesian Archaeological Literature Mining | SUCCESS | 05_archival_nlp |  | P1,P11 |  |
| E094 | DHARMA Semantic Search | SUCCESS | 04_language_text | L4,L5 | P16,P5,P8 |  |
| E095 | Cross-Lingual Analysis on Original Old Javanese In | SUCCESS | 04_language_text | L4,L5 | P16 |  |
| E096 | DHARMA Diachronic BERTopic | SUCCESS | 04_language_text | L4 | P16,P5,P8 |  |
| E097 | Anomaly Detection on Settlement Model Feature Stac | ? | 01_spatial |  |  |  |
| E098 | Systematic Literature Database — Sedimentation, Bu | SUCCESS | 05_archival_nlp | L1 | P1,P11 |  |
| E099 | Eruption Frequency x Inscription Visibility Gradie | INCONCLUSIVE | 06_thesis | L1,L6 | P11,P5 | p=0.013 |
| E100 | Coastal-Highland Archaeological Visibility Inversi | SUCCESS | 01_spatial | L1,L2 | P1,P2 | p<0.0001 |
| E101 | Colonial Burial Depth Multivariate Model | PARTIAL | 02_taphonomy | L1 | P1 | p=0.012; rho=0.373 |
| E102 | Vocabulary Richness × Burial Depth Nexus | SUCCESS | 04_language_text | L1,L4 | P1,P5,P8 | p<0.0001; rho=0.797 |
| E103 | Pre-Indic Vocabulary Spatial Gradient | SUCCESS | 01_spatial | L1,L4,L6 | P5,P9 | p<0.0001; rho=+0.502 |
| E104 | Court Zone Hypothesis: Multi-Dataset Spatial Segre | SUCCESS | 01_spatial | L1,L3,L4 | P11,P5,P7,P9 | rho=0.781 |
| E105 | BERTopic Topics × Geographic Distribution | SUCCESS | 04_language_text | L1,L4,L6 | P5,P7,P9 | p=0.580; rho=0.502 |
| E106 | Colonial Two Javas Validation | SUGGESTIVE (low N, d | 01_spatial | L1,L3 | P1,P17 | p=0.217 |
| E107 | ADV-5 Re-examination — Is Iban+Malay Really a Nega | SUCCESS | 04_language_text,02_taphonomy | L4 | P8 | AUC=0.713 |
| E108 | Demographic Null Model — Pre-400 CE Java Carrying  | SUCCESS | 01_spatial |  |  |  |
| E109 | Forward Simulation — Archaeological Record Under B | MIXED | 01_spatial,02_taphonomy | L1 | P1,P2 | p=0.0015 |
| E110 | Multiplicative Visibility Cascade Model | SUCCESS | 01_spatial |  |  |  |
| E111 | Script Diffusion Timeline — Is Java's 650-Year Gap | SUCCESS | 04_language_text |  | P18 |  |
| E112 | Vocabulary Archaeology — Computational Reconstruct | SUCCESS | 04_language_text |  | P18 |  |
| E113 | Inscription Sophistication Analysis | SUCCESS | 04_language_text |  |  |  |
| E114 | Comparative Pre-Literate Complex Societies | SUCCESS | 04_language_text |  | P18 |  |
| E115 | Monte Carlo Sensitivity Analysis of Visibility Cas | SUCCESS | 01_spatial |  | P1 | ρ=0.5 |
| E116 | Testable Predictions from the Visibility Cascade | SUCCESS | 01_spatial |  | P1 |  |
| E117 | Archaeological Record Onset Analysis — The Michels | SUCCESS | 02_taphonomy |  | P1,P18 | p=1.0 |
| E118 | Information Gain from Volcanic Context | SUCCESS | 01_spatial |  | P1,P2 |  |
| E119 | The VOLCARCH Synthesis Figure | SUCCESS | 06_thesis |  | P1 |  |
| E120 | Cascade Stress Test — Systematic Adversarial Probi | SUCCESS | 01_spatial | L1,L5 | P1,P18 | p=0.0015 |
| E121 | Robustness Battery — Automated Resampling Tests | SUCCESS | 01_spatial | L1,L5 |  | p=1000; rho=-0.57 |
| E122 | Demographic Gap Sensitivity Analysis | SUCCESS | 01_spatial | L1 | P1 |  |
| E123 | Philippines Cross-Geographic Comparison | SUCCESS | 02_taphonomy | L1 | P1,P18 |  |
| E124 | Survey Asymmetry Analysis | SUPERSEDED | 01_spatial |  |  |  |
| E125 | Delpher Pilot Study | SUPERSEDED | 05_archival_nlp |  |  |  |
| E126 | Global Volcanic Archaeology Compilation | SUCCESS | 02_taphonomy,01_spatial | L1 | P1,P18 |  |
| E127 | Ancient External References to Pre-400 CE Nusantar | SUCCESS | 06_thesis | L3 | P1,P16,P18,P19 |  |
| E128 | Colonial OV Depth Analysis — Independent Burial Ca | SUCCESS | 02_taphonomy,05_archival_nlp | L1 | P1,P21 |  |
| E129 | Survey Asymmetry Quantification | SUCCESS | 01_spatial | L1 | P1,P18 | p=0.09 |
| E130 | Substrate Detection Interpretability | SUCCESS | 04_language_text | L4 | P19,P8 | AUC=0.76 |
| E131 | Comparative Writing System Adoption Timeline | SUCCESS | 04_language_text | L3 | P1,P19 |  |
| E132 | Sedimentation Rate Prediction Map | PARTIAL | 02_taphonomy | L1 | P1,P22 |  |
| E133 | The Complete Argument — Why Nusantara's History "B | SUCCESS | 06_thesis |  |  | p<10; z=11.05 |
| E134 | Inscription Chronology Gap Analysis | SUCCESS | 04_language_text | L5,L6 | P16,P19,P5,P8 | p=0.13.; rho=0.58 |
| E135 | Organic Material Preservation Model | SUCCESS | 02_taphonomy | L1 | P1,P19 |  |
| E136 | Bayesian Integration of All VOLCARCH Evidence | SUCCESS | 06_thesis |  |  | z=11.05 |
| E137 | Accidental Discovery Rate Model | PARTIAL | 02_taphonomy | L1 | P1 |  |
| E138 | Detection Probability by Archaeological Method | SUCCESS | 02_taphonomy |  | P1,P22 |  |
| E139 | Cost-Benefit Analysis of Fieldwork Strategies | SUCCESS | 01_spatial |  |  |  |
| E140 | Material Culture Index | SUCCESS | 02_taphonomy |  | P1,P19 |  |
| E141 | Delpher Colonial Newspaper Extraction Pipeline | SUCCESS | 05_archival_nlp |  | P21 |  |
| E142 | Delpher Full-Text NLP Extraction | SUCCESS | 05_archival_nlp |  | P21 |  |
| E143 | Delpher Spatial Cross-Reference | SUCCESS | 05_archival_nlp |  | P21 |  |
| E144 | Evidence Timeline Figure | SUCCESS | 06_thesis |  | P1,P19 |  |
| E145 | Eruption Frequency vs Archaeological Visibility | INFO NEG | 06_thesis |  | P1 | p=0.0001; rho=0.908 |
| E146 | Comparative Inscription Density | SUCCESS | 04_language_text |  | P1,P19,P5 |  |
| E147 | Inscription Length Analysis | SUCCESS | 04_language_text |  | P5,P8 | p<0.0001; rho=0.587 |
| E148 | Sunda Shelf Marine Archaeological Survey Gap Analy | SUCCESS | 02_taphonomy | L2 | P1,P18 |  |
| E149 | Eruption-Inscription Paradox Reconciliation | SUCCESS | 06_thesis |  | P1,P17 | p=0.0001; rho=+0.908 |
| E150 | Babad Tanah Jawi Substrate NLP | SUCCESS | 04_language_text | L4 | P19,P8 |  |
| E151 | Megalithic Distribution vs Volcanic Zones | SUCCESS | 01_spatial | L1 | P1,P19 |  |
| E152 | Post-929 CE Mataram -> East Java Natural Experimen | SUCCESS | 01_spatial | L1,L6 | P1,P17 |  |
| E153 | Candi-Settlement Spatial Association Test | ? | 01_spatial |  |  | p=0.000029 |
| E154 | Comprehensive FDR Re-Audit at 153 Experiments | SUCCESS | 06_thesis |  |  | p<10 |
| E155 | Cross-Regional Cascade Validation | SUCCESS | 01_spatial |  | P1,P17,P18 | p=0.017; rho: 0.926 |
| E156 | Sunda Shelf Population Displacement → Java Volcani | SUCCESS | 02_taphonomy | L1,L2 | P1,P18,P19 |  |
| E157 | Ethnographic Analog — Modern Volcanic Community Ma | SUCCESS | 02_taphonomy |  | P1,P17,P18,P19 |  |
| E158 | Steelman Counter-Arguments for Cathedral Findings | SUCCESS | 06_thesis |  |  | AUC=0.762; p=0.0015 |
| E159 | Robustness Battery for Cathedral Findings | SUCCESS | 01_spatial |  |  | p=0.51; rho=-0.131 |
| E160 | GPU-Powered Deep Semantic Analysis of DHARMA Inscr | SUCCESS | 04_language_text | L6 | P16,P17,P5,P8 | p=0.012; z=3.04 |
| E161 | Bali as Within-Indonesia Volcanic Comparandum | SUCCESS | 02_taphonomy |  | P1,P17,P18 |  |
| E162 | State of Evidence at 161 Experiments | SUCCESS | 06_thesis |  |  | p=0.012; rho=1.0 |
| E163 | Sumatra Applicability Test — Does VOLCARCH Predict | SUCCESS | 01_spatial |  | P1,P17,P18 |  |
| E164 | Dong Son Drum Distribution — Pre-Hindu Material Ev | SUCCESS | 02_taphonomy,06_thesis |  | P1,P17,P19 | p=0.047 |
| E165 | Ghost Vocabulary — Linguistic Fossils in Old Javan | SUCCESS | 04_language_text |  | P16,P17,P19,P5,P8 | rho=0.456 |
| E166 | Burial Depth Prediction Map for East Java | SUCCESS | 02_taphonomy,01_spatial |  | P1,P17,P2 | AUC=0.768 |
| E167 | Priority Fieldwork Map — The VOLCARCH Treasure Map | SUCCESS | 01_spatial |  | P1,P17,P2 |  |
| E168 | The Invisible Civilization — Computational Reconst | SUCCESS | 06_thesis |  | P18,P19 | p<0.001 |
| E169 | Inscription Desert Analysis | SUCCESS | 04_language_text |  | P1,P17 |  |
| E170 | TWI-Enhanced Burial Depth Model | SUCCESS | 02_taphonomy |  | P1,P2 | rho=0.986 |
| E171 | VOLCARCH Prediction Registry — 5 Formal Prediction | SUCCESS | 01_spatial |  | P1,P2 |  |
| E172 | Dynamic Population Model for Java (40,000 BP — 160 | SUCCESS | 01_spatial |  | P1,P17,P18 |  |
| E173 | Counterfactual — "What If Indonesia Had Japan's Ar | SUCCESS | 02_taphonomy |  | P1,P17,P18,P19 |  |
| E174 | The VOLCARCH Synthesis Figure | SUCCESS | 06_thesis |  |  |  |
| E175 | Spatial Statistics of Candi Distribution | SUCCESS | 01_spatial |  | P11,P17,P7 |  |
| E176 | Cascade Minimal Model Comparison | SUCCESS | 01_spatial |  | P1,P17 |  |
| E177 | Sunda Shelf Paleo-Drainage Reconstruction | SUCCESS | 02_taphonomy | L2 | P18 |  |
| E178 | Philippines Archaeological Density Regression | SUCCESS | 02_taphonomy |  | P1,P17,P19 |  |
| E179 | Factor Independence Test — Cascade Coupling Analys | SUCCESS | 06_thesis,01_spatial |  | P1,P17 |  |
| E181 | Ghost Dictionary — Semantic Clustering of 230 Vani | SUCCESS | 04_language_text |  | P16,P17,P19,P5,P8 |  |
| E182 | Karst-Augmented Cascade Model | SUCCESS | 01_spatial,02_taphonomy |  | P1,P17 | rho=0.607 |
| E183 | Register Split — When Did Written and Oral Javanes | SUCCESS | 01_spatial |  | P16,P19,P5,P8 |  |
| E184 | Inscription Spatial Autocorrelation (Moran's I) | SUCCESS | 01_spatial |  | P1,P17,P2 | rho=0.781 |
| E185 | Spatially-Constrained Permutation Test for Two Jav | SUCCESS | 01_spatial |  | P17 | p<10; rho=0.49 |
| E186 | Tengger Ghost Word Cross-Reference | SUCCESS | 04_language_text |  | P19,P8 |  |
| E187 | Proper Spatial Regression (PySAL spreg) | SUCCESS | 01_spatial |  | P17 | p=0.002 |
| E188 | Mainland SE Asia Comparative Onset Analysis | SUCCESS | 02_taphonomy |  | P1,P17,P19 |  |
| E189 | Satellite Spectral Feasibility — Can Sentinel-2 Se | SUCCESS | 01_spatial | L1 | P1,P23 | p=0.032 |
| E190 | SAR Feasibility — Can Sentinel-1 Radar See Buried  | SUCCESS | 01_spatial | L1 | P1,P17,P23 | p=0.032 |
| E191 | Multi-temporal NDWI — Dry vs Wet Season Contrast | SUCCESS | 01_spatial | L1 | P1,P17,P23 | p=0.032 |
| E192 | NDWI Anomaly vs Burial Depth Correlation | SUCCESS | 01_spatial | L1 | P1,P17 | p=0.048; rho=-0.39 |
| E193 | Sunda Shelf Entry Points vs Coastal Site Distribut | SUCCESS | 02_taphonomy | L2 | P18 |  |
| E194 | Combined Archaeological Prospection Map | SUCCESS | 01_spatial | L1,L2 | P1,P17,P2 |  |
| E195 | Is Two Javas Taphonomic? — The Inverse Discovery | SUCCESS | 02_taphonomy,01_spatial | L1 | P1,P17,P18 |  |
| E196 | Pre-400 CE Java Population Estimation | SUCCESS | 01_spatial | L1,L2 | P1,P17,P18 |  |
| E197 | Colonial Depth Records vs E075 Burial Model | SUCCESS | 02_taphonomy,05_archival_nlp | L1 | P1,P17 | p=0.131 |
| E198 | Sago-Rice Etymology — The "sego" ← "*sagu" Hypothe | SUCCESS | 04_language_text |  |  |  |
| E199 | Collective Brain / Volcanic Innovation Paradox | SUCCESS | 06_thesis |  |  |  |
| E200 | Historical Dutch NER Baseline Analysis | SUCCESS | 05_archival_nlp |  |  | p=0.131 |
| E201 | Philippines Archaeological Record — Deep Compositi | SUCCESS | 02_taphonomy |  | P1,P17,P18 |  |
| E202 | DEM Depression Detection for Buried Archaeological | FAILED | 01_spatial | L1 | P1,P11,P18 | p=0.326 |
| E204 | Bronze Drum (Nekara) Distribution — Extended Analy | SUCCESS | 02_taphonomy,06_thesis | L1,L4 | P1,P17,P19 |  |
| E205 | The Indigenous Layer in Wayang — Quantifying Pre-H | SUCCESS | 04_language_text | L4,L5 | P19,P5 |  |
| E206 | ArcheoBERTje-NER on Colonial Dutch — Quantifying t | SUCCESS | 05_archival_nlp |  |  |  |
| E207 | GLOBALISE VOC Transcription Pilot — PhD Feasibilit | SUCCESS | 05_archival_nlp |  |  |  |
| E208 | Kakawin/Old Javanese NLP Pilot — DHARMA Successor  | PHASE 1 + 2a COMPLET | 04_language_text | L4,L5 | P0,P16 |  |
| E209 | Multi-Signal Satellite ML Classifier for Buried-Si | PHASE 1 — data pipel | 01_spatial | L1 | P23 |  |
| E210 | InSAR Time-Series Subsidence Detection for Buried  | SCOPING — Phase 0 (p | 01_spatial | L1 |  |  |
| E211 | VOC Dagregister NLP at Scale — Archaeological Ment | Phase 0 — corpus sec | 05_archival_nlp |  |  |  |
| E213 | Aggradation–Exposure Geomorphic Asymmetry | INCONCLUSIVE | 02_taphonomy | L1 | P7 | p=8e-9 |
| E214 | Palynological / Charcoal Test of Pre-400 CE Human  | SUCCESS | 03_paleoenv |  |  |  |
| E215 | Phytolith / Starch-Grain Test of Pre-400 CE Cultiv | SUCCESS | 03_paleoenv |  |  |  |
| E216 | The Paleo-Ecological Interferometer | ? | 03_paleoenv |  |  |  |
| E217 | MaxEnt Benchmark Across the Pseudo-Absence Ladder | PRE-REGISTERED (deci | 01_spatial |  | P2 |  |
| E218 | Is the Evaluation Artefact Real, and What Drives I | ? | 01_spatial |  | P2 |  |
| E219 | Does Background Design Change the Map Rather Than  | ? | 01_spatial |  | P2 |  |
| E220 | Selection on the Reported Metric Walks Backwards | SUCCESS | 01_spatial |  | P1,P3,P4 |  |
| E221 | Seed-Ensemble Stabilisation + Robust/Contingent Pr | SUCCESS | 01_spatial |  |  |  |
| E222 | Synthetic Ground-Truth Validation | SUCCESS | 01_spatial |  |  |  |
| E223 | Statistical Robustness Package | SUCCESS | 01_spatial |  |  |  |
| E224 | Does target-group background work once the bias va | FAILED | 01_spatial |  | P2 |  |

## Revisit Candidates

Experiments that failed or were inconclusive but could be revisited with new data/methods.

| ID | Title | Status | Why Revisitable |
|-----|-------|--------|-----------------|
| E007 | Settlement Suitability Model — Paper 2 M | REVISIT | *(check README)* |
| E008 | Settlement Suitability Model v2 — River  | REVISIT | *(check README)* |
| E009 | Settlement Suitability Model v3 — SoilGr | REVISIT | *(check README)* |
| E010 | Settlement Suitability Model v4 - Target | REVISIT | *(check README)* |
| E011 | Settlement Suitability Model v5 - TGB Pa | REVISIT | *(check README)* |
| E012 | Settlement Suitability Model v6 - TGB Pr | REVISIT | *(check README)* |
| E017 | Tephra POC (Pyle 1989 Calibration for Pa | FAILED | *(check README)* |
| E018 | Temporal Overlay Matrix — Proof of Conce | INCONCLUSIVE | *(check README)* |
| E034 | Cerita Panji in Malagasy Literature Sear | INFO NEG | *(check README)* |
| E037 | Prasasti Dating Model (ML on Undated Ins | INFO NEG | *(check README)* |
| E038 | Volcanic Vocabulary Semantic Drift Acros | INFO NEG | *(check README)* |
| E039 | Volcanic Cultural Selection — Cross-Cult | INFO NEG | *(check README)* |
| E067 | Volcanic Toponyms — Do Volcanic Place Na | INFO NEG | *(check README)* |
| E087 | Substrate Detector Negative Control | CONDITIONAL | *(check README)* |
| E099 | Eruption Frequency x Inscription Visibil | INCONCLUSIVE | *(check README)* |
| E101 | Colonial Burial Depth Multivariate Model | PARTIAL | *(check README)* |
| E109 | Forward Simulation — Archaeological Reco | MIXED | *(check README)* |
| E132 | Sedimentation Rate Prediction Map | PARTIAL | *(check README)* |
| E137 | Accidental Discovery Rate Model | PARTIAL | *(check README)* |
| E145 | Eruption Frequency vs Archaeological Vis | INFO NEG | *(check README)* |
| E202 | DEM Depression Detection for Buried Arch | FAILED | *(check README)* |
| E213 | Aggradation–Exposure Geomorphic Asymmetr | INCONCLUSIVE | *(check README)* |
| E224 | Does target-group background work once t | FAILED | *(check README)* |

---

*Auto-generated by `tools/scan_experiments.py`. Do not edit manually — changes will be overwritten.*
# IDEA REGISTRY — Master Catalog of Research Ideas

**Purpose:** Prevent idea loss. Every hypothesis, method, and research direction gets an ID here — even if it can't be executed yet. Killed papers ≠ killed ideas.

**Convention:**
- ID format: `I-NNN` (never recycled)
- Maturity: SPARK → HYPOTHESIS → TESTABLE → READY → EXPERIMENT → RESULT → PAPER
- Update `docs/TRIGGER_MAP.md` when blockers change
- Tag serendipitous cross-paper discoveries in JOURNAL: `[BRIDGE → PY, I-NNN]`

**Last updated:** 2026-03-16

---

## READY — Can Execute Now

| ID | Title | Maturity | Source | Effort | Links |
|----|-------|----------|--------|--------|-------|
| I-001 | ~~Candi orientation vs volcanic peak alignment~~ → **E031 SUCCESS (split)** | RESULT | E031 | done | P7, P11, Channel 9 |
| I-002 | ~~Pranata Mangsa × eruption seasonality~~ → **E032 COMPLETE** | RESULT | E032 | done | P5, P11, Channel 7 |
| I-003 | ~~Sanskrit ratio per century~~ → **E033 COMPLETE** | RESULT | E033 | done | P5, P8, Channel 6,12 |
| I-004 | ~~Cerita Panji in Malagasy~~ → **E034 INFORMATIVE NEGATIVE** | RESULT | E034 | done | P9, P12, Channel 8 |
| I-005 | ~~Prasasti dating model~~ → **E037 CONDITIONAL** | RESULT | E037 | done | P5, P14, Channel 12 |
| I-006 | ~~Hanacaraka phonological inventory mapping~~ → **E036 SUCCESS** | RESULT | E036 | done | P8, P12, Channel 12 |
| I-007 | ~~Volcanic vocabulary semantic drift~~ → **E038 INFORMATIVE NEG** | RESULT | E038 | done | P8, P11, Channel 6 |
| I-008 | ~~Prasasti botanical keyword expansion~~ → **E035 SUCCESS** | RESULT | E035 | done | P5, P9, Channel 5 |
| I-009 | Carangan wayang inventory — episodes WITHOUT Indian parallel | READY | P12 draft §8 | 1-2 days | P12, Channel 8 |
| I-010 | Babad Tanah Jawi / Serat Centhini substrate extraction via NLP | READY | Exploration 2026-03-10 | 2-3 days | P12, Channel 8 |
| I-011 | Pulotu botanical query: aromatic plant × burial co-occurrence | TESTABLE | P9 draft §5 | 1 session | P5, P9, Channel 5 | *Blocked: Pulotu lacks plant-specific variables. Needs external ethnobotanical data. E050 confirms Canarium as pan-Austronesian aromatic — strengthens hypothesis.* |

---

## TESTABLE — Method + Data Identified, But Blocked

| ID | Title | Maturity | Blocker | Trigger | Source | Links |
|----|-------|----------|---------|---------|--------|-------|
| I-020 | Sentinel-2 crop mark detection Zone B/C | TESTABLE | Need to build U-Net pipeline, download imagery | Sentinel-2 tile acquisition + model training setup | Exploration 2026-03-10 | P2, P7, P10, Channel 1 |
| I-021 | Mythology binary classifier (P12 pipeline) | TESTABLE | Needs corpus construction (~40 narrative units labeled) | Malagasy corpus + wayang carangan inventory done (I-004, I-009) | P12 draft §5 | P12, Channel 8 |
| I-022 | KawiKupas tool — Sanskrit ratio extractor for Kawi texts | TESTABLE | Needs Sanskrit lexicon + Old Javanese morphological rules | Digital Sanskrit dictionary accessible | P6 draft | P6, P8, Channel 6,12 |
| I-023 | 6-dimensional Kawi text clustering (phylogenetic) | TESTABLE | Needs KawiKupas (I-022) + digital corpus of 10 target texts | I-022 complete + corpus digitized | P6 draft | P6, Channel 6 |
| I-024 | Malagasy burial plant comparison (ethnobotany) | **DONE (E044+E050)** | ~~Needs Malagasy ethnobotanical literature survey~~ | ✓ E044 complete. E050 GBIF confirms Canarium in ALL Austronesian regions (Madagascar 388 records, 25.9%). Plumeria=New World. 4-layer substitution chain validated. | P9 draft §4.2 | P9, P11, Channel 5 |
| I-025 | Krama lexical comparison (Bali Alus vs Tegal vs Solo) | TESTABLE | Needs Tegal/Banyumas wordlist compilation | Wordlist compiled or fieldwork connection | P9 draft §2.1 | P9, Channel 6 |
| I-026 | Osing substrate detection via KawiKupas | TESTABLE | Needs KawiKupas (I-022) + Osing dialect data | I-022 + ABVD Osing entries or fieldwork | P9 draft §5.5 | P8, P9, Channel 6 |
| I-027 | Tengger ritual vocabulary analysis | **READY** | ~~Needs Tengger dialect wordlist~~ ✓ ABVD ID 1533 (178 concepts, 255 forms). E043 shows PMP cognacy 27.7% (lower than Javanese 33.0%) — small isolate drift. | Tengger ritual vocabulary (non-ABVD) still needs compilation | P9 draft §6 | P8, P9, P11, Channel 6,7 |
| I-028 | Ghost population detection in Javanese genomes | TESTABLE | Needs access to Eijkman/1000Genomes Indonesian data | Public genome data + admixture pipeline setup | Working note aDNA | P7, Channel 3 | *Meta-finding (I-101): NO Java aDNA exists — volcanic taphonomy is the explanation. Absence itself is evidence.* |
| I-029 | Batimetri Sunda Shelf anomaly detection (GEBCO + ML) | TESTABLE | Needs GEBCO bathymetry download + anomaly detection pipeline | GEBCO data acquired + compute setup | Exploration 2026-03-10 | P-coastal, Channel 2 | *See also I-102 (paleo-drainage reconstruction as prerequisite).* |
| I-030 | ~~P14 Bonferroni/Holm correction + research note pivot~~ → **DONE** | RESULT | — | — | Mata Elang #3, R04 | P14 |
| I-031 | ~~P8 intro reframe — lead with phonological non-conformity (E029)~~ → **DONE** | RESULT | — | — | Mata Elang #3, I5 | P8 |

---

## HYPOTHESIS — Testable Statement Formed, Needs Method/Data Assessment

| ID | Title | Statement | Source | Links |
|----|-------|-----------|--------|-------|
| I-040 | ~~Bamboo Civilization hypothesis~~ → **E040 SUCCESS, E048 cross-validated** | 170/268 (63.4%) prasasti mention organic materials vs 73 (27.2%) lithic. E048 confirms organic mentions correlate with pre-Indic vocabulary (partial rho=+0.162, p=0.038). Genre taphonomy quantified: sima 90.4% vs short 24.1%. | E040, E048 | P1, P7, Channel 1 |
| I-041 | Oralitas as technology | Oral tradition stores information as densely as written text. Testable: measure information density (unique concepts/hour) of wayang vs contemporary written texts. | Exploration 2026-03-10 | P12, Channel 8 |
| I-042 | ~~VCS diversity prediction~~ → **E039 INFORMATIVE NEGATIVE** | Binary volcanic/non-volcanic test NOT significant (p=0.973, direction reversed). Classification problem: Q32 island type is wrong proxy. VCS is LOCAL (proximity-based), not island-type. Q21 (mana) one intriguing signal (p=0.006). Next: GVP distance-based continuous test. | E039 | P11, Channel 7 |
| I-043 | Candi siting = volcanic resilience selection | Sacred architecture on elevated ground is volcanically selected (survives eruption cycles), not just "closer to heaven." Testable: candi elevation vs local eruption flow direction analysis. | P11 draft §2.3 | P7, P11, Channel 1,9 |
| I-044 | Slametan = volcanic insurance mechanism | Mandatory communal food sharing is selected FOR in volcanic landscapes (post-eruption crop loss). Communities with slametan survive; those without don't. Testable: Pulotu communal feast complexity vs volcanic density. | P11 draft §2.2 | P5, P11, Channel 7 |
| I-045 | Estuarine hybrid resilience model | Most resilient polities (Sriwijaya, Surabaya, Demak) = river-sea confluence. Most archaeologically invisible due to organic + tidal erosion. Testable: map polity longevity vs estuarine position. | P4 draft | P4, Channel 2 |
| I-046 | Volcanic density × colonial exploitation model | More volcanic density → more complex state → different colonial exploitation type. Java (in situ) vs Maluku (removal). Testable: GVP density × VOC records. | Parking lot VCS-colonial | P11, Channel 10 |
| I-047 | Trunyan copper plate analysis | 833 Saka (~10th c.) inscription at Trunyan — any mention of burial practice? If yes, earliest written attestation of mepasah. | P9 draft notes | P5, P9, Channel 7 |
| I-048 | Gamelan pelog × volcanic tremor frequency | Pelog tuning system has no Indian parallel. Correlate pelog frequency ratios with Merapi/Kelud tremor spectrograms. | Master attack map Ch.11 | P11, Channel 11 |
| I-049 | Keris pamor as volcanic material culture | Pamor technique uses volcanic magnetite + meteoritic nickel = unique to Nusantara. Date earliest keris finds vs Hindu arrival. | P11 draft §6 parking lot | P11, Channel 10 |
| I-050 | Batik motif substrate detection | Some traditional batik patterns have no Hindu parallel. Apply visual classifier (same logic as P12) to batik images. | Master attack map Ch.10 | P12, Channel 10 |
| I-051 | Volcanic ash as aDNA preservative | Volcanic burial that destroys surface record may PRESERVE aDNA (sealed, anaerobic). Java's aDNA blank spot may be best preservation site. **I-101 confirms: no Java aDNA exists at all — the blank IS the taphonomic signal.** | Working note aDNA §5 | P7, Channel 3 |
| I-052 | Tephrochronology calendar for Java | Use known tephra layers (Kelud, Tambora, Krakatau, Toba) as stratigraphic dating framework — same method as Iceland archaeology. | P10 draft §4b | P1, P10, Channel 1 |
| I-053 | ~~Pangram narrative uniqueness test~~ → **DESK RESEARCH: CONFIRMED UNIQUE** | Hanacaraka is the only known script whose pangram encodes a complete NARRATIVE (characters, conflict, resolution). Iroha (Japanese) = lyric/philosophical poem, no characters/plot. Old Slavonic letter names = debatable didactic message. Thai/Burmese/Khmer/Baybayin/Devanagari = phonological tables only. Caveat: Hanacaraka story is Neo-Javanese (not in OJ texts), dating uncertain. Recommended framing: "only writing system whose canonical learning sequence is a complete narrative with named characters and dramatic arc." | Web search 2026-03-11 | P8, P12, Channel 12 |
| I-054 | Surabaya-Venice comparison | Both estuarine, both trade-network, both organic architecture. Formally comparative. | P4 draft | P4, Channel 2 |
| I-055 | Mongol 1293 invasion as natural experiment | Kertanegara assassination + Mongol withdrawal = exogenous shock to mandala system. What happens to volcanic ritual during state collapse? | P4 draft | P4, P14, Channel 7 |
| I-101 | Ghost Population Detection via aDNA Synthesis — meta-taphonomic result | No Java aDNA exists BECAUSE of volcanic taphonomy. The absence IS the finding. Systematic aDNA literature review → quantify the blank → reframe as strongest evidence for L1 erasure. | aDNA agent research | P1, P7, Channel 3 |
| I-102 | Sunda Shelf Paleo-Drainage Reconstruction (GEBCO bathymetry → L2 test) | GEBCO bathymetry data can reconstruct Pleistocene river systems on Sunda Shelf. Submerged drainages = likely settlement corridors. First computational test of L2 (Coastal Submersion). | Background research | L2, P-coastal, Channel 2 |
| I-103 | Java Toponymic Substrate Mapping (BPS village names → pre-Hindu geographic layer) | BPS census lists ~80,000 village names in Java. NLP extraction of non-Sanskrit, non-Arabic toponyms → map pre-Hindu geographic naming layer. Testable: do substrate toponyms cluster away from volcanoes (Zone C)? | Background research | P8, P11, Channel 6 |
| I-104 | Maritime Vocabulary as Civilization Indicator | E049 finding: maritime words are #2 most conserved domain in peripheries (Bal 60% vs Jav 40%). "Sea" (laut) replaced in Javanese but retained in Balinese. Pre-Hindu substrate was maritime-organic, not just organic. Court-driven overwriting targeted nature + maritime domains. | E049 | P8, P9, Channel 6 |
| I-105 | Genre Taphonomy as 5th Layer of Darkness (L5) | E048 quantified: sima inscriptions mention organic materials 90.4% vs short-format 24.1% (p<0.0001). Pre-Indic ratio also higher in long sima. Inscription FORMAT is a massive taphonomic filter — C8 "dark century" is a genre artifact, not a cultural blank. | E048 | P1, P5, Manifesto L5, Channel 1 |
| I-106 | Canarium as Pan-Austronesian Aromatic Marker | E050 GBIF confirms Canarium in ALL Austronesian regions: Taiwan (136), Philippines (13), Madagascar (388), Melanesia (15). 4-layer aromatic substitution chain: Canarium → dammar → menyan → kamboja. Indonesia undersampled (4 records = collection bias). | E050 | P5, P9, Channel 5 |

| I-110 | Dong Son drum distribution as pre-400 CE test | TESTABLE | Bernet Kempers 1988 catalog needed | Catalog accessed/digitized | Structural critique 2026-03-20 | P1, P17, Channel 1 | *If drums found in volcanic East Java → direct evidence of pre-Hindu occupation in volcanic zones. Key ref: Bernet Kempers 1988. Check Museum Nasional Jakarta + Tropenmuseum Leiden catalogs.* |
| I-111 | Philippines-Java archaeological record comparison | TESTABLE | Need Philippines site catalog (National Museum, Bellwood 2017, Mijares 2010) | Data compiled | Structural critique 2026-03-20 | P1, P17, Channel 1 | *Philippines has richer pre-400 CE record despite LESS survey intensity → supports volcanic burial thesis. Key insight: cave availability (karst) is a 6th cascade factor. See docs/research_notes/BLIND_SPOT_PHILIPPINES_COMPARANDUM.md* |
| I-112 | Pre-Dong Son metallurgy evidence in Java | TESTABLE | Needs literature survey of ore deposit + smelting sites | Literature compiled | Structural critique 2026-03-20 | P1, Channel 1 | *Java has copper + iron ore. Zero pre-400 CE smelting sites. Is this also taphonomic? Key ref: Van Heekeren 1958.* |

### Blind Spot Analysis — 2026-03-21

| ID | Title | Maturity | Blocker | Trigger | Source | Links |
|----|-------|----------|---------|---------|--------|-------|
| I-120 | Liangan as VOLCARCH validation case | TESTABLE | Needs Abbas 2016 monograph data (burial depth, C-14 dates) | Monograph accessed or site report obtained | Blind spot analysis 2026-03-21 | P1, P19, Channel 1 | *Complete Mataram-era settlement buried by Sindoro. Preserved organic material including wooden structures. "Java's Pompeii." Validates core prediction: volcanic burial preserves settlements including organics. 8th-10th c. CE.* |
| I-121 | Tuban nekara as pre-Hindu volcanic-zone evidence | DATA-SUPPORTED | None — finding confirmed | N/A | Blind spot analysis 2026-03-21 | P1, P17, P19, Channel 1,10 | *Heger Type II bronze drum (~300 BCE) from Tuban, East Java (Museum Mpu Tantular, catalog 1907). WITH bronze elephant inside. Volcanic East Java. DIRECT evidence of pre-Hindu material culture in volcanic zone. Weakens Counter 1.* |
| I-122 | Sulawesi cave art (67,800 BP) as P19 comparandum | READY | None | N/A | Blind spot analysis 2026-03-21 | P19, Channel 1 | *World's oldest cave art is Indonesian (Maros-Pangkep, Sulawesi). 67,800 BP hand stencil, 51,200 BP narrative scene. Survives because KARST (non-volcanic). Same archipelago, same populations, different geology = different preservation. Devastating counter to "prasejarah" label.* |
| I-123 | Java megalithic distribution vs volcanic zones | TESTABLE | Needs megalithic site catalog (literature compilation) | Catalog compiled | Blind spot analysis 2026-03-21 | P1, P19, Channel 1 | *Punden berundak, menhirs, dolmens exist across Java. Pre-Hindu, STONE, visible. VOLCARCH must refine: what's "missing" is ORGANIC lowland settlement, not ALL pre-Hindu culture. Megaliths survive because stone. Key sites: Gunung Padang, Cipari, Bondowoso.* |
| I-124 | Cerén (El Salvador) as volcanic preservation analog | **READY** | ~~Literature review~~ DONE. | N/A | Blind spot + **lit review 2026-03-21** | P1, P19, Channel 1 | *Research complete. ~200 person Maya village buried by Loma Caldera (~AD 600). Thatch, wooden beams, food in pots, manioc fields, sleeping mats ALL preserved. Phreatomagmatic (cool ash). NO formal Cerén-Java comparison in literature = publication opportunity. See `docs/research_notes/CEREN_COMPARISON.md`.* |
| I-125 | Phytolith survival in volcanic soil (escalation of I-082) | **TESTABLE** | ~~Literature review~~ DONE. Next: core sample access or Liangan matrix collaboration. | Castillo (UCL) + PVMBG or Balai Arkeologi Yogyakarta | Blind spot + **lit review 2026-03-21** | P1, P10, P19, Channel 1 | *STRONGLY POSITIVE. Phytoliths survive 90K yr in tephra (Aso). Java andisol pH 5-7 = excellent. Rice phytoliths diagnostic. NO ONE has looked for pre-Hindu phytoliths in Javanese volcanic matrices. See `docs/research_notes/PHYTOLITH_VOLCANIC_PRESERVATION.md`. Potential P20 or collaborative proposal.* |
| I-126 | Babad Tanah Jawi NLP — pre-Hindu substrate extraction | TESTABLE | Needs digitised romanised Javanese text | Text accessed | Blind spot analysis 2026-03-21 | P16, P19, Channel 6,8 | *Javanese chronicle mentioning pre-Hindu rulers and place names. NLP extraction of non-Sanskrit elements. Compare with DHARMA inscriptional data.* |
| I-127 | Kinship vocabulary as Austronesian substrate marker | HYPOTHESIS | Needs Javanese kinship term compilation | Literature compiled | Blind spot analysis 2026-03-21 | P8, P19, Channel 6 | *Javanese kinship vocabulary is overwhelmingly Austronesian (not Sanskrit). Social organisation was not "Indianised." Extends E058 domain analysis to social domain.* |

### Dissemination Strategy — 2026-03-22

| ID | Title | Maturity | Blocker | Trigger | Source | Links |
|----|-------|----------|---------|---------|--------|-------|
| I-128 | LiDAR survey of volcanic Java — fieldwork pitch | READY | Needs 1-page pitch document | Pitch created + meeting with contact | Session 5 discussion 2026-03-22 | Phase 2, Channel 1 | *Pak Amien has LiDAR company contact. Amazon LiDAR precedent (2024 Nature) found thousands of cities under canopy. Same technology on volcanic Java = potentially revolutionary. Need compelling 1-pager: precedent + 10 GPS targets + PR value for company. Could trigger Phase 2 fieldwork.* |
| I-129 | "Peradaban Tersembunyi" YouTube series | READY | Time + basic video editing | Back at office | Session 5 discussion 2026-03-22 | Dissemination | *6-episode Indonesian-language series. Ep1: "Kenapa 400 Masehi?" — question millions of Indonesians have but never asked. Low production cost (voiceover + slides). Archaeology YouTube content in Indonesian is scarce but audience is large. Could reach 10K-100K views.* |
| I-130 | VOLCARCH prediction registry — public predictions | READY | GitHub repo must be public | Repo public | Session 5 discussion 2026-03-22 | Dissemination, Phase 2 | *Publicly register 10 location predictions with timestamps + DOI. When someone eventually digs there = validation. Builds credibility like physics predictions. Prediction + verification narrative is compelling for media.* |
| I-131 | Low-cost deep coring campaign ($6K for 20 cores) | TESTABLE | Funding (~$6K) + geotechnical company contact | Funding secured | Session 5 discussion 2026-03-22 | Phase 2, Channel 1 | *Commercial geotechnical borehole ~$200-500 per 10m core. 20 cores at VOLCARCH priority sites = $6K total. Not archaeology proper but enough to confirm cultural layers at predicted depths. Could be crowdfunded.* |
| I-132 | Construction company soil profile data partnership | HYPOTHESIS | Needs MoU with geotechnical company | Company identified | Session 5 discussion 2026-03-22 | Phase 2, Channel 1 | *Infrastructure projects (tol, MRT, dam) already drill deep boreholes in Java. Their soil profiles might contain "anomalous" cultural layers. MoU: "may we examine your soil profiles from zone X?" Zero cost, potentially transformative.* |

---

## SPARK — Raw Ideas, No Method Yet

| ID | Title | Note | Source |
|----|-------|------|--------|
| I-070 | Barong Brutuk dance structural analysis | Trunyan dance: no music, banana leaves, only unmarried men. Parallels to Toraja ritual isolation? | P9 draft notes |
| I-071 | "Ancestors descended from sky" (Trunyan) × Toraja puya cosmology | Origin beliefs comparison across peripheral communities | P9 draft notes |
| I-072 | Kamboja allelopathic properties × decomposition rate | Kamboja inhibits soil microbes — does this affect taphonomic timing? Testable in vitro. | P9 draft notes |
| I-073 | Tempeh fermentation × volcanic soil microbiome | Speculative: volcanic soil bacteria contribute to tempeh culture? | P11 draft §6 |
| I-074 | Japan Shinto volcano deities comparison | Both volcanic landscapes, different ritual responses — why? | P11 draft §6 |
| I-075 | LiDAR coverage of East Java — what already exists? | Reconnaissance for remote sensing potential | P10 draft notes |
| I-076 | Drone multispectral crop marks (Zone B/C) | Low-cost, non-invasive first pass before coring | P10 draft notes |
| I-077 | Phosphorus survey of Trowulan (Majapahit heartland) | Already done by anyone? Literature check. | P10 draft notes |
| I-078 | Song Terus aDNA extraction attempt | Pacitan cave, fauna bones exist — any human aDNA attempt? | Working note aDNA §10 |
| I-079 | Muna Island post-Oktaviana 2026 aDNA sampling | Any planned aDNA from cave art site? | Working note aDNA §10 |
| I-080 | Pertamina/ESDM sonar data for Sunda Shelf | Oil/gas sonar may contain submerged settlement anomalies | Master attack map Ch.2 |
| I-081 | BATAN AMS radiocarbon capability + cost | What is current turnaround? For future P10 fieldwork. | P10 draft notes |
| I-082 | Phytolith survival in volcanic soil literature check | Alkaline ash may ENHANCE phytolith preservation | P10 draft notes |
| I-083 | Liangan adjacent soil cores | Most accessible deeply buried Javanese site (sand mining 2008) | P10 draft notes |
| I-084 | Borobudur base reliefs without Indian iconographic source | Which panels are NOT from Indian texts? | Master attack map Ch.10 |
| I-085 | La Galigo NLP — motif extraction from Bugis epic | 6000 pages, pre-Islamic, zero Hindu pantheon | P12 draft §8 |
| I-086 | ~~Batara Kala as Class A candidate~~ → **DESK RESEARCH: CLASS C** | Not Class A — deity concept clearly Indic (Shiva/Kala/Mahakala, Rahu/Ketu). But ruwatan ceremony (wayang exorcism) and sukerta children taxonomy appear uniquely Javanese institutional innovations on Indic frame. Class C (syncretic). | Web search 2026-03-11 |
| I-087 | Ruwatan structural analysis (pre-Hindu structure, post-Hindu vocab) | Good test case for syncretic Class C classification | P12 draft §8 |
| I-088 | Effective population size (Ne) from modern DNA | Constraint for pre-Hindu population estimates | Working note aDNA §6 |
| I-089 | Peripheral Krama fieldwork via istri's family network (Tegal) | Informal access to Tegal/Banyumas dialect data | P9 draft notes |

---

## RESULT / PAPER — Resolved (Archived)

| ID | Title | Status | Experiment | Paper |
|----|-------|--------|------------|-------|
| I-002 | Pranata Mangsa × eruption seasonality | RESULT | E032 | Kapitu peak 3.8x, chi2 p=0.042, Rayleigh p=0.032 |
| I-004 | Cerita Panji in Malagasy (informative neg.) | RESULT | E034 | Panji absent (post-dates migration). Ibonia = Ramayana-era. |
| I-003 | The Indianization Curve (Sanskrit ratio temporal) | RESULT | E033 | P5 revision ammo (rho=-0.211, p=0.030) |
| I-001 | Candi orientation vs volcanic peak alignment (split) | RESULT | E031 | Siting: west-cluster p<0.0001; Orientation: null (35%, p=0.94) |
| I-008 | Prasasti botanical keyword expansion | RESULT | E035 | 15 plants, menyan+kamboja ABSENT. Mortuary = oral tradition. |
| I-006 | Hanacaraka 33→20 phonological mapping | RESULT | E036 | Aspiration+retroflex+sibilant lost. Aligns PAn. tha/dha paradox. |
| I-005 | Prasasti dating model (ML) | RESULT | E037 | CONDITIONAL: MAE=115yr, R²≈0. Content features too weak. Needs paleography. |
| I-007 | Volcanic vocabulary semantic drift | RESULT | E038 | INFORMATIVE NEG: no diversity diff. Core vocab too stable. Phylogenetic confound. |
| I-090 | Volcanic sedimentation rate calibration | PAPER | E001-E006 | P1 (submitted) |
| I-091 | Settlement suitability prediction via XGBoost | PAPER | E007-E015 | P2 (draft complete) |
| I-092 | Tautology elimination suite | RESULT | E013-E014 | P2 (conditional pass) |
| I-093 | Deep-time site spatial segregation | PAPER | E018-E019 | P7 (submitted) |
| I-094 | Pre-Indic vocabulary persistence in prasasti | RESULT | E030 | P5 (submitted) |
| I-095 | Phonological substrate detection via ML | PAPER | E022-E029 | P8 (draft v0.1) |
| I-096 | Substrate clustering → shared language (REJECTED) | RESULT | E029 | P8 (informative negative: p=0.569) |
| I-097 | Pararaton-Kelud temporal correlation | RESULT | E026 | ~~P14~~ KILLED (Bonferroni adj. p=0.222). E026 folded into P5 revision ammo. |
| I-098 | Slametan-decomposition taphonomic link | PAPER | E023 | P5 (submitted) |
| I-099 | Mini-NusaRC cave bias test | RESULT | E020 | P7 (informative negative: p=0.761) |
| I-100 | Borehole burial gradient | RESULT | E024 | P9 (POC complete, 25 records) |
| I-040 | Bamboo Civilization material culture scan | RESULT | E040 | Organic 63.4% vs Lithic 27.2%. Binomial p<0.0001. P1 direct evidence. |
| I-105 | Genre Taphonomy (L5) — multi-domain convergence | RESULT | E048 | Sima 90.4% vs short 24.1% organic. C8=dark century (genre artifact). partial rho=+0.162 (p=0.038). |
| I-104 | Maritime vocabulary conservation in peripheries | RESULT | E049 | Maritime #2 conserved (+20% Bal vs Jav). "Sea" replaced in Javanese. Pre-Hindu = maritime-organic. |
| I-106 | Canarium pan-Austronesian aromatic (GBIF confirmed) | RESULT | E050 | Canarium in ALL AN regions. Madagascar 388 records (25.9%). 4-layer substitution chain validated. |
| I-107 | Kakawin domain-specific Sanskritization | RESULT | E058 | Agriculture 91% native, Religion 86% Sanskrit. Register stratification, not uniform overlay. chi² p<1e-10. |
| I-108 | Pre-400 CE reconstruction (8-channel synthesis) | RESULT | E060 | 6 domains reconstructed. Economy+Religion=HIGH conf. Script=SPECULATIVE. 56 experiments, 8 channels. |
| I-109 | Indic script simplification comparison (cross-SE-Asian) | EXPERIMENT | E061 | Testing: do all Brahmi-derived scripts simplify toward local phonology? Strengthens E036 Hanacaraka finding. |
| I-110 | Prasasti comprehensive temporal model (visibility curve) | EXPERIMENT | E062 | Combining E023+E030+E035+E040 into single visibility score per century. PCA. |
| I-111 | ABVD domain-specific PMP conservation | EXPERIMENT | E063 | Which Swadesh domains are most conserved across 1000+ Austronesian languages? Extends E058. |
| I-112 | Candi archaeoastronomy (entrance vs solar azimuths) | RESULT | E066 | 85% face equinox E/W (p=4.9e-14). McNemar p=0.0016 vs volcanic. Ch9 strengthened. |
| I-113 | Volcanic toponyms in Java village names | RESULT | E067 | INFO NEG: no proximity effect (rho=+0.14, p=0.15). VI = behavioral, not lexical. |
| I-114 | Spatial vs linguistic meta-test (behavioral not lexical) | RESULT | E073 | p=0.008, r=1.0. Volcanic influence is spatial-behavioral, not lexical. |
| I-115 | Sedimentation burial forward model | RESULT | E075 | r=0.951, 32.3% cells >1m. Physical model of burial depth across Java. |
| I-116 | Eruption-inscription temporal correlation | RESULT | E078 | 6.3× inscription deficit in eruption decades, p=0.035. 928 CE = 77% drop. |
| I-117 | Darkness Index (Invisible Millennium quantified) | RESULT | E079 | DI: Invisible Millennium 0.515, Classical Java 0.273. 1.9× ratio. |
| I-118 | Computational textual archaeology (external archive) | RESULT | E088-E090 | 50 refs, 9 traditions, convergence p<0.00001. Ancient texts cluster by CONTENT not CULTURE. P16 foundation. |
| I-119 | ADV-1 Japan comparanda (volcanic = necessary not sufficient) | RESULT | E086 | PARTIAL. Java 32× deeper burial, Japan 100-200× more survey. L1 = volcanism × survey deficit. |

---

## Index by Channel (Master Attack Map)

| Channel | Ideas |
|---------|-------|
| 1. Geology/Taphonomy | I-020, I-040, I-043, I-052, I-090-I-093, I-105 |
| 2. Maritime/Coastal | I-029, I-045, I-054, I-080, I-102 |
| 3. Genetics/DNA | I-028, I-051, I-078, I-079, I-088, I-101 |
| 5. Ethnobotany | I-008, I-011, I-024, I-072, I-106 |
| 6. Linguistics | I-003, I-007, I-022, I-023, I-025, I-026, I-027, I-095-I-096, I-103, I-104 |
| 7. Ritual | I-002, I-042, I-044, I-047, I-055, I-094, I-098 |
| 8. Mythology | I-004, I-009, I-010, I-021, I-041, I-085, I-086, I-087 |
| 9. Archaeoastronomy | I-001, I-043, I-112 |
| 10. Material Culture | I-046, I-049, I-050, I-084 |
| 11. Acoustics | I-048 |
| 12. Script Archaeology | I-005, I-006, I-053, I-109 |

---

*This is a living document. Add ideas as they emerge. Update maturity when blockers clear. Never delete — move to RESULT/PAPER when resolved.*

# IDEA REGISTRY — Master Catalog of Research Ideas

**Purpose:** Prevent idea loss. Every hypothesis, method, and research direction gets an ID here — even if it can't be executed yet. Killed papers ≠ killed ideas.

**Convention:**
- ID format: `I-NNN` (never recycled)
- Maturity: SPARK → HYPOTHESIS → TESTABLE → READY → EXPERIMENT → RESULT → PAPER
- Update `docs/TRIGGER_MAP.md` when blockers change
- Tag serendipitous cross-paper discoveries in JOURNAL: `[BRIDGE → PY, I-NNN]`

**Last updated:** 2026-03-11

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
| I-011 | Pulotu botanical query: aromatic plant × burial co-occurrence | TESTABLE | P9 draft §5 | 1 session | P5, P9, Channel 5 | *Blocked: Pulotu lacks plant-specific variables. Needs external ethnobotanical data.* |

---

## TESTABLE — Method + Data Identified, But Blocked

| ID | Title | Maturity | Blocker | Trigger | Source | Links |
|----|-------|----------|---------|---------|--------|-------|
| I-020 | Sentinel-2 crop mark detection Zone B/C | TESTABLE | Need to build U-Net pipeline, download imagery | Sentinel-2 tile acquisition + model training setup | Exploration 2026-03-10 | P2, P7, P10, Channel 1 |
| I-021 | Mythology binary classifier (P12 pipeline) | TESTABLE | Needs corpus construction (~40 narrative units labeled) | Malagasy corpus + wayang carangan inventory done (I-004, I-009) | P12 draft §5 | P12, Channel 8 |
| I-022 | KawiKupas tool — Sanskrit ratio extractor for Kawi texts | TESTABLE | Needs Sanskrit lexicon + Old Javanese morphological rules | Digital Sanskrit dictionary accessible | P6 draft | P6, P8, Channel 6,12 |
| I-023 | 6-dimensional Kawi text clustering (phylogenetic) | TESTABLE | Needs KawiKupas (I-022) + digital corpus of 10 target texts | I-022 complete + corpus digitized | P6 draft | P6, Channel 6 |
| I-024 | Malagasy burial plant comparison (ethnobotany) | TESTABLE | Needs Malagasy ethnobotanical literature survey | Literature survey done or Malagasy collaborator | P9 draft §2.3 | P9, P11, Channel 5 |
| I-025 | Krama lexical comparison (Bali Alus vs Tegal vs Solo) | TESTABLE | Needs Tegal/Banyumas wordlist compilation | Wordlist compiled or fieldwork connection | P9 draft §2.1 | P9, Channel 6 |
| I-026 | Osing substrate detection via KawiKupas | TESTABLE | Needs KawiKupas (I-022) + Osing dialect data | I-022 + ABVD Osing entries or fieldwork | P9 draft §5.5 | P8, P9, Channel 6 |
| I-027 | Tengger ritual vocabulary analysis | TESTABLE | Needs Tengger dialect wordlist (may exist in literature) | Tengger wordlist located or compiled | P9 draft §5.5 | P8, P9, P11, Channel 6,7 |
| I-028 | Ghost population detection in Javanese genomes | TESTABLE | Needs access to Eijkman/1000Genomes Indonesian data | Public genome data + admixture pipeline setup | Working note aDNA | P7, Channel 3 |
| I-029 | Batimetri Sunda Shelf anomaly detection (GEBCO + ML) | TESTABLE | Needs GEBCO bathymetry download + anomaly detection pipeline | GEBCO data acquired + compute setup | Exploration 2026-03-10 | P-coastal, Channel 2 |
| I-030 | ~~P14 Bonferroni/Holm correction + research note pivot~~ → **DONE** | RESULT | — | — | Mata Elang #3, R04 | P14 |
| I-031 | ~~P8 intro reframe — lead with phonological non-conformity (E029)~~ → **DONE** | RESULT | — | — | Mata Elang #3, I5 | P8 |

---

## HYPOTHESIS — Testable Statement Formed, Needs Method/Data Assessment

| ID | Title | Statement | Source | Links |
|----|-------|-----------|--------|-------|
| I-040 | ~~Bamboo Civilization hypothesis~~ → **E040 SUCCESS** | 170/268 (63.4%) prasasti mention organic materials vs 73 (27.2%) lithic. Organic-only=103, lithic-only=6. Binomial p<0.0001. Confirms P1 taphonomic bias thesis. | E040 | P1, P7, Channel 1 |
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
| I-051 | Volcanic ash as aDNA preservative | Volcanic burial that destroys surface record may PRESERVE aDNA (sealed, anaerobic). Java's aDNA blank spot may be best preservation site. | Working note aDNA §5 | P7, Channel 3 |
| I-052 | Tephrochronology calendar for Java | Use known tephra layers (Kelud, Tambora, Krakatau, Toba) as stratigraphic dating framework — same method as Iceland archaeology. | P10 draft §4b | P1, P10, Channel 1 |
| I-053 | ~~Pangram narrative uniqueness test~~ → **DESK RESEARCH: CONFIRMED UNIQUE** | Hanacaraka is the only known script whose pangram encodes a complete NARRATIVE (characters, conflict, resolution). Iroha (Japanese) = lyric/philosophical poem, no characters/plot. Old Slavonic letter names = debatable didactic message. Thai/Burmese/Khmer/Baybayin/Devanagari = phonological tables only. Caveat: Hanacaraka story is Neo-Javanese (not in OJ texts), dating uncertain. Recommended framing: "only writing system whose canonical learning sequence is a complete narrative with named characters and dramatic arc." | Web search 2026-03-11 | P8, P12, Channel 12 |
| I-054 | Surabaya-Venice comparison | Both estuarine, both trade-network, both organic architecture. Formally comparative. | P4 draft | P4, Channel 2 |
| I-055 | Mongol 1293 invasion as natural experiment | Kertanegara assassination + Mongol withdrawal = exogenous shock to mandala system. What happens to volcanic ritual during state collapse? | P4 draft | P4, P14, Channel 7 |

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

---

## Index by Channel (Master Attack Map)

| Channel | Ideas |
|---------|-------|
| 1. Geology/Taphonomy | I-020, I-040, I-043, I-052, I-090-I-093 |
| 2. Maritime/Coastal | I-029, I-045, I-054, I-080 |
| 3. Genetics/DNA | I-028, I-051, I-078, I-079, I-088 |
| 5. Ethnobotany | I-008, I-011, I-024, I-072 |
| 6. Linguistics | I-003, I-007, I-022, I-023, I-025, I-026, I-027, I-095-I-096 |
| 7. Ritual | I-002, I-042, I-044, I-047, I-055, I-094, I-098 |
| 8. Mythology | I-004, I-009, I-010, I-021, I-041, I-085, I-086, I-087 |
| 9. Archaeoastronomy | I-001, I-043 |
| 10. Material Culture | I-046, I-049, I-050, I-084 |
| 11. Acoustics | I-048 |
| 12. Script Archaeology | I-005, I-006, I-053 |

---

*This is a living document. Add ideas as they emerge. Update maturity when blockers clear. Never delete — move to RESULT/PAPER when resolved.*

# E092: Volcanic Archaeology Comparanda Database

**Date:** 2026-03-16
**Status:** SUCCESS
**Papers served:** P1, P11, fieldwork planning
**Layer:** L1

---

## Hypothesis

Successful discovery of volcanically-buried archaeological sites worldwide follows identifiable patterns (survey technique, depth, tephra type) that can inform Indonesian fieldwork design. By compiling global comparanda and extracting methodology lessons, VOLCARCH can design a cost-effective survey strategy for Zone B/C targets identified in E080.

## Method

Structured compilation of known archaeological sites buried by volcanic deposits worldwide, with systematic recording of:
- **Location** (coordinates, country, associated volcano)
- **Burial context** (depth, tephra type, eruption date, VEI)
- **Discovery process** (how found, what survey techniques used)
- **Cost indicators** (where available)
- **Success factors** (what enabled discovery)

Sources: published archaeological and volcanological literature, project knowledge from E083 (tephra-archaeological correlation dataset for Indonesia), E080 (fieldwork targets), and E065/E084 (candi/inscription spatial analysis).

## Data

### Comparanda Database
- **File:** `results/volcanic_comparanda.csv`
- **29 entries** across 12 countries and 20+ volcanic systems
- **14 fields** per entry including coordinates, burial depth, tephra type, discovery method, survey technique, estimated cost, and key references

### Geographic Distribution
| Region | Sites | Key volcanoes |
|--------|-------|---------------|
| Indonesia | 11 | Merapi, Arjuno-Welirang, Kelud, Tambora, Dieng, Ungaran |
| Italy | 5 | Vesuvius (79 CE + Avellino) |
| Central America | 4 | Laguna Caldera, Ilopango, Xitle, Popocatepetl |
| Greece | 2 | Thera/Santorini |
| Philippines | 1 | Pinatubo |
| Colombia | 1 | Nevado del Ruiz |
| PNG | 1 | Highland volcanoes |
| USA | 1 | Mt St Helens |
| New Zealand | 1 | Taupo |
| Cambodia | 1 | None (LiDAR methodological comparandum) |
| Turkey | 1 | Hasan Dagi (volcanic landscape, not burial) |

### Burial Depth Distribution (where known)
| Depth range | Count | Examples |
|-------------|-------|---------|
| 0-1 m | 4 | Gedong Songa, Kuk Swamp, Mt St Helens |
| 1-3 m | 8 | Kelud 1919 sites, Trowulan, Dieng, Tambora |
| 3-5 m | 3 | Candi Tikus, Trowulan deep, Prambanan deep |
| 5-10 m | 6 | Sambisari, Kedulan, Ceren, Pompeii, Cuicuilco |
| 10+ m | 2 | Herculaneum, Akrotiri |

### Discovery Method Distribution
| Method | Count | Notes |
|--------|-------|-------|
| Chance (construction, mining, plowing) | 8 | Most common for Java — Sambisari (1966), Kedulan (1993) |
| Systematic excavation | 9 | State-funded or long-term projects |
| Post-disaster survey | 3 | Kelud 1919, Armero 1985, Pinatubo 1991 |
| Surface features visible | 3 | Cuicuilco pyramid tip, Prambanan upper sections |
| Remote sensing | 2 | Angkor LiDAR, satellite |
| Geological survey | 3 | Ilopango, Taupo, Nea Kameni |
| Colonial documentation | 1 | Dieng, pre-1814 |

### Methodology Blueprint
- **File:** `results/methodology_blueprint.md`
- Survey technique recommendations by depth (0-1m, 1-3m, 3-5m, 5-10m, 10m+)
- Cost estimates for GPR, coring, magnetometry, ERT, LiDAR, trenching
- Success factor analysis from Ceren, Akrotiri, and Pompeii
- 4-phase recommended approach for VOLCARCH Zone B/C with budget summary

## Key Results

### 1. Discovery method pattern
The dominant discovery method for volcanically-buried sites worldwide is **chance encounter during ground-disturbing activity** (construction, mining, farming). This is also true in Java: Sambisari (farmer's plow, 1966), Kedulan (sand mining, 1993), Prambanan Vishnu statue (well digging, OV 1925). This means that **systematizing chance** — monitoring construction/mining activity in high-probability zones — may be the most cost-effective discovery strategy.

### 2. Depth determines method
Sites buried <3m can be found with GPR and magnetometry. Sites at 3-5m require coring + geophysics. Sites at 5-10m require deep coring or major excavation. Java's E083 dataset shows mean burial of 3.41m (median 2.50m) — placing most targets in the GPR/magnetometry sweet spot.

### 3. Magnetometry advantage for Java
Unlike Pompeii (stone), Ceren (adobe/thatch), or Akrotiri (stone), Javanese buried sites are predominantly **brick temples** (candi). Brick has high magnetic contrast with surrounding volcanic sediment, making magnetometry potentially the most effective and cheapest geophysical technique. This advantage is specific to Java and has not been systematically exploited.

### 4. Lahar burial is a process, not an event
Pinatubo (1991) demonstrated that secondary lahars continue burying sites for 5+ years after eruption. Merapi and Kelud show the same pattern — ongoing sedimentation rather than single catastrophic burial. This means burial depths are continuously increasing and sites that were accessible decades ago may now be deeper.

### 5. Budget-constrained approach is feasible
A phased approach (satellite $100-500 → coring $2,000-5,000 → geophysics $5,000-15,000 → test excavation $10,000-50,000) could achieve first-pass survey of E080's top 10 targets for under $25,000 total — comparable to a single small research grant.

## Status: SUCCESS

Compiled 29 global comparanda with structured data, extracted methodology lessons, and produced an actionable fieldwork blueprint tailored to VOLCARCH's budget and target constraints.

## Implications

### For VOLCARCH fieldwork planning
1. **Phase 1 (desktop + satellite) is already underway** via E076 and E080. Cost: minimal.
2. **Phase 2 (reconnaissance coring) is the critical next step.** 20-40 boreholes at E080's top targets would cost $4,000-20,000 and could confirm or refute the presence of cultural material at depth.
3. **Magnetometry should be prioritized over GPR** for Javanese brick temples — cheaper, faster, and better suited to the target material.
4. **Institutional partnership is essential** for Phase 4 (test excavation). The Dissemination Roadmap (docs/VOLCARCH_Dissemination_Roadmap_v1.0.md) identifies BALARJATIM and UB Malang as potential partners.

### For papers
- **P1:** Global comparanda strengthen the argument that volcanic burial is a worldwide phenomenon with predictable patterns, not an ad hoc Indonesian observation.
- **P11:** The methodology blueprint provides the "so what" — how volcanic informedness translates into practical fieldwork design.
- **Future paper:** A dedicated methods paper comparing Java's situation to Ceren/Akrotiri/Pompeii could target *Journal of Archaeological Method and Theory* or *Archaeological Prospection*.

## Relation to Other Experiments

- **Builds on:** E080 (fieldwork targets), E083 (tephra-archaeological correlation), E075 (sedimentation model), E076 (satellite NDVI), E065 (candi spatial analysis)
- **Feeds into:** Fieldwork design, dissemination Dokumen Jembatan, P11 revision ammo
- **Related comparanda:** E086 (ADV-1 Japan comparanda) — Japan volcanic archaeology provides additional comparanda not included here

## Data Integrity Notes

- All sites and eruption data are based on published literature cited in each entry's `key_reference` field.
- Cost estimates are approximate and noted as such. Sources vary from published project budgets (Angkor LiDAR, Borobudur UNESCO) to estimates based on standard rates (GPR, coring).
- Indonesian sites cross-referenced against E083 dataset (TAC IDs noted in the `notes` field).
- Where values are uncertain, this is explicitly stated in the `notes` field. No values were fabricated.
- Taupo entry notes that Maori sites were NOT buried by the ~232 CE eruption (settlement postdates eruption by ~1000 years) — included for tephrostratigraphic methodology relevance.
- Angkor and Catalhoyuk are NOT volcanic burial sites — included as methodological comparanda (LiDAR and volcanic awareness, respectively).

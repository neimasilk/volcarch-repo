# Literature SLR Protocol — VOLCARCH Systematic Review

**Status:** ACTIVE (started 2026-04-20)
**Rationale:** Detection of Jia et al. 2024 (Datong Jatim beads) via a Facebook post, not via our literature searches, exposed a critical blind spot. VOLCARCH's prior literature discovery has been subfield-siloed, keyword-native, and confirmation-biased. This protocol addresses those failures.
**Driven by:** Session 18 dialog (Pak Amien + Claude, 2026-04-20). User authorised systematic review + symmetric counter-evidence hunt + pivot if warranted.
**Governed by:** CLAUDE.md research integrity rules; Mata Elang series epistemic audit.

---

## 1. Purpose

A subfield-driven, symmetrically-biased, reproducible literature review of the ~10 domains relevant to VOLCARCH's "Invisible Civilisation" thesis. Output: (a) structured evidence inventory for P0 masterpiece, (b) material counter-evidence audit, (c) living bibliography with explicit inclusion/exclusion reasons.

## 2. Core Principles

1. **Subfield-driven, not keyword-driven.** We start from a pre-declared list of subfields (Section 4 below) and design search strategies per subfield. We do not search for "VOLCARCH-native" terms alone.
2. **Symmetric bias mitigation.** Every pro-evidence search is paired with a counter-evidence search of equal rigour. Counter-findings are documented with the same prominence.
3. **Bibliography management is infrastructure, not afterthought.** All discovered papers are filed under `docs/bibliography/<subfield>/` as Markdown notes (one per paper), plus a master inventory table.
4. **Living document.** Protocol is refreshed every 3-6 months or whenever a new subfield is identified (e.g., via serendipitous discovery).
5. **Pivot-eligible.** If material counter-evidence is found, we STOP the current P0 framing, report findings, and collectively decide adjustment/absorption/pivot. Pak Amien has pre-authorised this path.
6. **Proportional depth.** This is SLR-light (2-3 weeks), not PRISMA-level (3-6 months). Good enough to surface 90% of high-impact findings, not exhaustive.

## 3. Phases and Timeline

### Fase A — Scope & Infrastructure (Day 1-2, started 2026-04-20)
- Write this protocol document.
- Create `docs/bibliography/` folder tree per subfield (DONE).
- Define tagging schema and paper-note template (below).
- Finalise 10 subfield research questions and search strategies.

### Fase B — Systematic Discovery (Day 3-10)
- Per subfield: execute 2 search tracks (pro + counter).
- Tools: Semantic Scholar API, Google Scholar (via WebSearch), specialist journal ToC browse, AI-assisted discovery (Elicit, Research Rabbit where accessible).
- Citation chasing forward (from anchor papers) + backward (from anchor refs).
- Each discovered paper: create a Markdown note in the appropriate subfield folder.
- Target: 150-300 papers screened, 40-80 included.

### Fase C — Screening & Extraction (Day 11-16)
- Title + abstract screening with explicit inclusion/exclusion reasons.
- Full-text screening for candidates.
- Structured extraction to `docs/bibliography/_INVENTORY.csv`: title, year, authors, DOI, subfield, claim summary, VOLCARCH-relation (supports/contradicts/neutral/ambiguous), chronology claim, methodology notes.

### Fase D — Synthesis & Decision (Day 17-21)
- Cluster analysis per subfield: where does evidence converge/diverge?
- **Counter-evidence audit.** Tabulate all material counter-findings. Apply severity filter (weak/moderate/strong).
- Decision point: P0 framing validated / adjusted / pivoted.
- Report to Pak Amien before P0 drafting resumes.

**Total: ~21 days focused work.** Partial overlap with P1-core JASREP submission polish is fine.

## 4. The 10 Subfields

Each subfield has: research question, primary databases/journals, anchor papers, search strategy, pro-expectations, counter-risks.

### Subfield 1 — SE Asian Glass Bead Archaeometry
- **RQ:** When did Javanese glass bead production begin, where were the workshops, what is the global distribution, and what does chemistry reveal about production sourcing?
- **Databases:** Semantic Scholar, Heritage Science, Journal of Archaeological Science, Journal of Glass Studies, Archaeometry.
- **Anchors:** Jia et al. 2024 (Datong); Lankton & Bernbaum (canonical Jatim identification); Francis 2002 *Asia's Maritime Bead Trade*; Dussubieux SE Asian compositional work; Bellina beads/trade papers.
- **Search strategy:** Keywords "Jatim bead", "Indo-Pacific bead", "mutisalah bead", "SE Asian glass archaeometry", "v-Na-Ca m-Na-Al glass Java", plus citation chasing from Jia et al. 2024.
- **Pro-expectations:** Distribution maps, production dating, workshop location hints.
- **Counter-risks:** Revised later dating of Jatim production; attribution to non-Java sources for some "Jatim" beads.

### Subfield 2 — Trans-Eurasian Trade Networks 200–500 CE
- **RQ:** Where does Nusantara appear in mid-first-millennium Eurasian exchange networks? Was Java a production node, a transit point, or peripheral?
- **Databases:** Journal of World History, Antiquity, Cambridge Archaeological Journal, Asian Perspectives.
- **Anchors:** Manguin maritime papers; Bellina 2014 Sa Huynh–Kalanay network; Hall *A History of Early Southeast Asia*; Glover & Bellwood *Southeast Asia: From Prehistory to History*.
- **Search strategy:** "trans-Eurasian trade Java", "maritime Silk Road Southeast Asia", "Indo-Roman trade Nusantara", "Oc Eo SE Asia exchange".
- **Pro-expectations:** Java/Sumatra as documented trade nodes pre-500 CE.
- **Counter-risks:** Java peripheral or transit-only rather than production node.

### Subfield 3 — Chinese Historical Texts on Nusantara
- **RQ:** What do Chinese dynastic histories and related texts say about polities and products from Nusantara prior to 500 CE? Are the toponyms Yepoti, Yediao, Heling, She-p'o identifiable as Java/Sumatra?
- **Databases:** Journal of Asian Studies, T'oung Pao, Bulletin of the School of Oriental and African Studies, Archipel.
- **Anchors:** Wolters *Early Indonesian Commerce*; Wang Gungwu *The Nanhai Trade*; Groslier; Wheatley *The Golden Khersonese*; Pelliot.
- **Search strategy:** Specific toponym queries (Yepoti Yediao Heling Shepo); "Hou Han Shu Java", "Liang Shu SE Asia tribute"; Wilkinson trade emoissaries.
- **Pro-expectations:** Attestation of organised Nusantara polities and products in Chinese records from 130 CE onward.
- **Counter-risks:** Toponym identifications contested; specific polities elsewhere.

### Subfield 4 — Indonesian Archaeometry & Materials Analysis
- **RQ:** What non-beads archaeometric studies exist for Indonesian materials (pottery composition, metal sourcing, organic residue analysis)? What chronologies do they establish?
- **Databases:** Journal of Archaeological Science: Reports, Journal of Indo-Pacific Archaeology, Bulletin of the Indo-Pacific Prehistory Association.
- **Anchors:** Calo bronze drum analyses; Miksic ceramics; pottery ICP-MS studies.
- **Search strategy:** "Indonesia pottery ICP", "Java bronze composition", "Sumatra metal trace element".
- **Pro-expectations:** Evidence of sophisticated craft production that would have required invisible industrial infrastructure.
- **Counter-risks:** Import signatures dominate over local production.

### Subfield 5 — Paleogenomics of Indonesian Populations
- **RQ:** What ancient DNA and modern whole-genome studies of Indonesian populations exist? What do they say about pre-400 CE population history, bottlenecks, continuity?
- **Databases:** Nature/Science/PNAS, Molecular Biology and Evolution, Current Biology, Cell.
- **Anchors:** Lipson 2014; Lipson 2018; Larena 2021; Carlhoff 2022; Maulana 2024 (West Java WGS).
- **Search strategy:** "Indonesian genome ancient DNA Java", "Austronesian paleogenomic", "aDNA tropical Southeast Asia preservation".
- **Pro-expectations:** Confirmed Java aDNA blank; deep genetic history inconsistent with "empty" Java.
- **Counter-risks:** Finding of Java aDNA would weaken argument; bottleneck signatures supporting sparse population.

### Subfield 6 — Volcanic Tropical Taphonomy (Global, non-Java)
- **RQ:** Do sedimentation rates comparable to our 4.4 ± 1.2 mm/yr Java calibration exist in published taphonomic studies from other volcanic tropical settings (Philippines, Hawaii, Mesoamerica, Iceland)?
- **Databases:** Geoarchaeology, Quaternary International, Journal of Volcanology and Geothermal Research.
- **Anchors:** Sheets *The Cerén Site*; Holmberg; Riede disaster archaeology.
- **Search strategy:** "volcanic sedimentation rate archaeological tropical", "cumulative tephra archaeology Holocene", "lahar archaeological site burial rate".
- **Pro-expectations:** Rates comparable across contexts, strengthening generalisability.
- **Counter-risks:** Rates systematically lower elsewhere, suggesting Java may be overestimated.

### Subfield 7 — Austronesian Glass & Metal Metallurgy
- **RQ:** What is known about origins, diffusion, and local production of glass and metallurgy across the Austronesian-speaking world? Indigenous development or import-only?
- **Databases:** Oceanic Linguistics (for linguistic evidence), Asian Perspectives, Antiquity.
- **Anchors:** Bellwood *First Islanders*; Calo *The Distribution of Bronze Drums*; Bernet Kempers.
- **Search strategy:** "Austronesian metallurgy origin", "Dong Son local production Indonesia", "Philippine copper bronze Neolithic".
- **Pro-expectations:** Evidence of indigenous metallurgical innovation supporting complex pre-Hindu society.
- **Counter-risks:** Pure import model gaining traction.

### Subfield 8 — Korean & Japanese Tomb Finds with SE Asian Material
- **RQ:** What Southeast Asian material (Jatim beads, other imports) has been documented in Korean and Japanese tombs? What does the distribution tell us about 1st-millennium networks?
- **Databases:** Korean Journal of Archaeology, Journal of East Asian Archaeology, Asian Perspectives.
- **Anchors:** Sikrichong Tomb reports (Gyeongju, Silla); Cheonmachong; Japanese Kofun period bead finds; Lankton's East Asian compilation.
- **Search strategy:** "Silla tomb glass bead Southeast Asian", "Kofun period import bead analysis", "Sikrichong bead Java".
- **Pro-expectations:** Jatim bead corpus expansion, trade network confirmation.
- **Counter-risks:** Mis-attributions previously accepted.

### Subfield 9 — Berenike + Red Sea Port Archaeology
- **RQ:** What Southeast Asian material is documented in the Roman-Byzantine Red Sea ports (Berenike, Myos Hormos)? What does its chronology and distribution reveal about pre-500 CE East-West exchange?
- **Databases:** Journal of Roman Archaeology, Antiquity, Journal of Archaeological Science.
- **Anchors:** Sidebotham *Berenike and the Ancient Maritime Spice Route*; various Berenike excavation reports; Cappers *Roman Foodprints at Berenike*.
- **Search strategy:** "Berenike SE Asian material", "Red Sea Indo-Pacific bead", "Roman Egypt Indian Ocean trade Java".
- **Pro-expectations:** SE Asian material confirming trans-Eurasian network.
- **Counter-risks:** Material mis-identified or chronologically later than claimed.

### Subfield 10 — Critical Historiography of Indianisation
- **RQ:** What is the strongest case AGAINST the "pre-existing substrate civilisation" thesis? Where do established frameworks (Coedès Indianisation, Wolters localisation, Kulke convergence, Pollock cosmopolitan) locate the genesis of Southeast Asian state formation?
- **Databases:** Journal of Southeast Asian Studies, Archipel, Bijdragen tot de Taal-, Land- en Volkenkunde, Journal of the Royal Asiatic Society.
- **Anchors:** Coedès 1968; Wolters 1999; Kulke 1990; Pollock 2006; Mabbett; Manguin 2004.
- **Search strategy:** "Indianization Southeast Asia critique", "Wolters localization primary process", "Pollock Sanskrit cosmopolis state formation".
- **Counter-focus:** This subfield's function is to find the STRONGEST version of the counter-thesis and engage it seriously in P0.
- **Risk of not doing this well:** P0 gets demolished in review by an engaged specialist.

## 5. Tagging Schema

Each paper note uses these fields:

```markdown
---
citekey: [author_year, matches references.bib]
title: [full title]
authors: [list]
year: [YYYY]
journal: [journal name + impact level Q1/Q2/Q3]
doi: [DOI]
subfield: [number + name, e.g., "01 Glass bead archaeometry"]
relation: [supports | contradicts | neutral | ambiguous]
chronology_claim: [e.g., "pre-400 CE" or "600-800 CE"]
method: [archaeometric | textual | genomic | demographic | theoretical | other]
quality: [Q1 peer-review | Q2 peer-review | preprint | non-peer-reviewed | primary source]
volcarch_use: [P0 channel N | P1-core | P17 | P18 | manifesto | discard]
---

# Notes

## Core claim (1 sentence)
...

## Evidence presented
...

## How it relates to VOLCARCH
...

## Methodological strengths
...

## Methodological weaknesses / critiques
...

## Chronological implications
...

## Quotes for direct citation
> "..."

## Cross-references (to other papers in this SLR)
- [citekey1] for [reason]
- [citekey2] for [reason]
```

## 6. Inclusion/Exclusion Criteria

**Include:**
- Peer-reviewed journal articles or book chapters in Q1/Q2 venues
- Primary archaeological reports from accredited institutions
- Monographs by recognised specialists
- Preprints from credible authors (flag quality)
- Chinese/Dutch/Indonesian colonial-era primary sources (flag translation quality)

**Exclude (but document):**
- Non-peer-reviewed blog posts, newspaper articles (unless primary source like colonial newspapers already in E091)
- Popular archaeology books (unless foundational)
- Papers by authors with known credibility issues (Graham Hancock etc.)
- Papers not retrievable or in languages we cannot assess (flag for future)

## 7. Search Strategy Per Paper

For each anchor paper discovered:

1. **Forward citation search:** Who has cited this paper? Via Semantic Scholar API or Google Scholar "Cited by" link.
2. **Backward citation search:** What papers does this one cite that are in our SLR scope?
3. **Author follow-up:** What else has this author published in the same area?
4. **Journal follow-up:** Same journal table-of-contents for surrounding issues.

## 8. Counter-Evidence Hunt Protocol

Separate search track per subfield. Specific queries:

- `"pre-Hindu Java" "population" "low" OR "sparse"` — directly counter to demographic argument
- `"Indianization" "primary" "civilization"` — Coedès-style claim
- `"Jatim bead" "dating" "revised" OR "later"` — counter to our chronology
- `"volcanic sediment" "rate" "overestimate"` — counter to P1-core calibration
- `"aDNA" "Indonesia" "Java" "recovered"` — would break our taphonomic aDNA argument

If any counter-evidence paper is high quality, escalate immediately (do not bury).

## 9. Tools

- **Semantic Scholar API** (free, via curl): `https://api.semanticscholar.org/graph/v1/paper/search?query=...`
- **Google Scholar** (via WebSearch): manual queries with advanced operators
- **arXiv, SSRN** for preprints
- **archive.org** for older open-access books
- **BibTeX** for reference management (lives in `docs/bibliography/master.bib`)
- **No Zotero** for now — Markdown notes + BibTeX is git-trackable and simpler

## 10. Progress Tracking

Status file: `docs/LITERATURE_SLR_PROGRESS.md` — updated at end of each session with:
- Papers discovered this session (counts per subfield)
- Papers screened in/out
- Counter-evidence flag list
- Next session priority

---

*Protocol v1.0 — 2026-04-20. Living document. Update when process reveals improvements.*

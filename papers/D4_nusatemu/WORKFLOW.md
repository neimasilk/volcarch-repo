# NusaTemu — workflow (alur kerja)

**Pace: slow by design ("pelan-pelan").** One phase at a time; each ends with a written GO / ADJUST / STOP from the PI.
Phase numbers are fixed. Experiment numbers (`ENNN`) are assigned only when a phase starts.

```
Phase 0  Design            → codebook, ethics, sources, literature check          (Claude; PI reads)
Phase 1  Pilot (news)      → ~100 articles, 3 provinces, hand-checked sample      (E-number at start; PI checks 30)
Phase 2  Scale-up (news)   → all Indonesia, 2000–2026, events deduplicated        (E-number; Claude; PI spot-checks)
Phase 3  Colonial layer    → OV / Notulen BG / Delpher, same codebook (Dutch)     (E-number; builds on E091)
Phase 4  Analysis          → discovery-process models, colonial vs modern, P2 link (pre-registered)
Phase 5  Release + paper   → Zenodo (coarse coordinates), venue first (G15), PI's prose (G16)
```

---

## Phase 0 — Design (now; Claude, no PI time needed)

| Step | Output | Done when |
|---|---|---|
| 0.1 Name, questions, scope | `README.md` | ✅ 2026-10-20 |
| 0.2 Record schema and taxonomies | `CODEBOOK.md` v0.1 | ✅ v0.1 2026-10-20; frozen as v1.0 at the end of Phase 1 |
| 0.3 Ethics and legal | `ETHICS.md` | ✅ 2026-10-20 |
| 0.4 Sources and query list | `SOURCES.md` | ✅ v0.1 2026-10-20 |
| 0.5 **Literature check**: does a chance-find register or a discovery-process study for Indonesia or Southeast Asia exist? (BPK/BPCB reports, Berkala Arkeologi, Amerta, Kalpataru, the national heritage registry, PAS-type literature) | `LITERATURE_CHECK.md` (fetched sources only; a list of what exists) | open — **if something close exists, Phase 1 is redesigned around it, not duplicated** |
| 0.6 Pilot pre-registration | `experiments/ENNN_nusatemu_pilot/DESIGN.md` | open; written after 0.5 |

## Phase 1 — Pilot on news (after P2 is uploaded; PI time ~1–2 hours)

- **Frame:** news 2015–2026 from **Jawa Timur, Jawa Tengah, DIY**. Harvested with the fixed query list
  (`SOURCES.md`), deduplicated by URL, then a random sample of ~100 candidate articles (seed fixed in DESIGN).
- **Steps:**
  1. relevance filter (is it a chance find?);
  2. extraction to the codebook by LLM, with the verbatim-quote guard of E225;
  3. events: merge articles about the same find, and flag retellings, such as the 2024 article retelling the 2018
     Lasem find;
  4. geocoding to desa or kecamatan with the Kemendagri list and the ADM2 boundaries already in `data/raw/gazetteer/`.
- **Check:** the PI hand-codes 30 articles blind. Claude compares field by field.
- **Pre-registered GO rule (proposal, fixed in DESIGN):**
  - relevance precision ≥ 0.85;
  - per-field agreement ≥ 0.80 for object class, discovery activity, kabupaten and year;
  - depth extracted correctly where stated;
  - ≥ 40 distinct events in the sample.

  Otherwise ADJUST (codebook or prompts) or STOP.
- **Cost:** ≤ $10 LLM, only with the PI's yes.

## Phase 2 — Scale-up on news (Claude; PI spot-checks)

- All provinces; 2000–2026. Same pipeline. A relevance classifier trained on the Phase 1 labels can replace the LLM for
  the first filter, which cuts cost.
- **Quality:** a 5 % random audit by a second blind reader (an LLM with a different prompt) plus 20 records hand-checked
  by the PI.
- **Output:** `nusatemu_events_vNN.csv` (public: coarse location) and `nusatemu_events_private.csv` (precise; never in
  the public repo; see `ETHICS.md`).

## Phase 3 — Colonial layer (Claude; builds on E091)

- **Sources:**
  - *Oudheidkundig Verslag* (OV; the E091 OCR corpus is already in the repo);
  - *Notulen van het Bataviaasch Genootschap*;
  - Delpher newspapers (KB API terms permitting; memory notes GDPR limits);
  - and later, the GLOBALISE VOC transcriptions for the 17th–18th c.
- **Same codebook**, Dutch prompts. "Discovery activity" gains colonial categories: railway, plantation, irrigation
  works, military works.
- **Output:** the colonial events in the same table, with `period_layer = colonial`.

## Phase 4 — Analysis (pre-registered; one E-number per question)

- **Q1 (descriptive):** what is found, by activity, setting, depth, year and province. Report the counts with their
  detection caveats.
- **Q2 (process vs past):** do find intensities follow modern drivers more than past-settlement proxies? Modern drivers:
  population, built-up growth (GHSL), road density. Past-settlement proxies: candi density, inscription findspots,
  P2's suitability surface. A pre-registered regression or point-process model at kabupaten scale.
- **Q3 (colonial vs modern):** how the share of discovery activities shifted.
- **Q4 (link to P2):** build a chance-find intensity surface, and test whether it is less tied to terrain than road
  distance (correlation with elevation, slope, TWI). Only if it is does it become a candidate survey-effort covariate.
- Every claim goes through the SIG (G1 numbers re-derived; G8 overstatement; G9 adversarial read).

## Phase 5 — Release and paper

- **Dataset:** Zenodo (DOI, CC BY 4.0) with **coarse public coordinates** (kecamatan centroid or 5 km grid), codebook,
  pipeline and audit. No finder names.
- **Paper:**
  - choose the venue **first** (G15, zero cost; a `VENUE.md` like P2's);
  - candidates to check then: Indonesian zero-APC journals (*Berkala Arkeologi*, *Kalpataru*, *Amerta*), JCAA (APC,
    waiver needed), *Internet Archaeology*;
  - the PI writes the prose (G16); a domain reader through G10 route H or S.
- **Order:** the paper waits for the G13 drafting slot. The dataset can be released earlier.

## What the PI decides, and when

| When | Decision |
|---|---|
| End of Phase 0 | Read `LITERATURE_CHECK.md`: GO to the pilot, or redesign |
| Before Phase 1 | Approve the pilot DESIGN (and ≤ $10 of LLM cost); hand-code 30 articles |
| End of each phase | GO / ADJUST / STOP, written in this file's log below |
| Phase 5 | Venue, release date, authorship (domain co-author invited by default for an archaeology venue) |

## Log

- 2026-10-20 — Workflow written (Phase 0 steps 0.1–0.4). Not started: 0.5 literature check, 0.6 pilot design.

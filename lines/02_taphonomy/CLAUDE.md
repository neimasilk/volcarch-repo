# Line 02 — TAPHONOMY (Burial, Erosion, Exposure)

> **Question:** Does volcanism actually destroy or hide the archaeological record — and by what
> mechanism, at what rate?

**Recommended model:** Opus. **Effort:** high (this line carries the project's largest unrepaired
data defect).

---

## Scope

The geoscience of preservation and visibility: tephra deposition, sedimentation/aggradation rates,
lahar routing, erosion, karst exposure, coastal submersion, borehole and depth evidence, and the
survey-effort confound. Reviewer community: **Quaternary geoscience / geoarchaeology** (JASREP,
EGQSJ, *Archaeological Research in Asia*).

**In scope:** *why* a site is absent or invisible; rates, depths, mechanisms; adversarial controls.
**Out of scope:** predicting *where* sites are (→ [01_spatial](../01_spatial/)); pollen/phytolith
proxies (→ [03_paleoenv](../03_paleoenv/)).

---

## ⚠ The defect this line owns

**P7 was peer-rejected by *Antiquity* (AQY-2026-0104, 2026-06-04) — the project's first
content-based rejection, 2 reviewers, full review — and the reviewers were right.**

The claim *"deep-time sites sit 90–170 km from volcanoes"* is **FALSE**. Actual: **33–53 km**. Cause:
`volcanoes.csv` contained only **7** eastern East Java volcanoes, omitting Lawu, Wilis, and all of
Central Java.

- **Canonical replacement:** `data/processed/dashboard/volcanoes_java_full.csv` (**30 volcanoes**).
- **Propagation:** P1 §spatial, P17 (live at ArchCalc), P11, and **~26 experiments**.
- **Preprint needing a correction notice:** `10.22541/au.177368991.14332505/v1`.
- **Verified survivors of the canonical re-run (2026-06-10):** E031 and E082 — P11 gap 9.2→6.1 km,
  Zone A 17.9×→19.1×. The finding survived; the number did not.

**The surviving reframe is EXPOSURE, not distance:** erosion and karst *windows* determine what is
findable, not proximity to a volcano. E178 is the key: Philippine volcanic zones have pre-400 CE
sites because they have **caves**; Java does not.

> **WS-E (from `docs/research_notes/FABLE_STRATEGIC_PLAN_20260707.md`) belongs to this line and is
> NOT done:** enumerate and blind re-derive every headline number in P1/P2/P5/P8/P11/P17/manifesto
> against the canonical inventory. It is mechanical, needs no PI decision, and every other line is
> quoting numbers that depend on it.

---

## Papers

| Paper | Folder | Status |
|---|---|---|
| **P1** Taphonomic framework | `papers/P1_taphonomic_framework/` | Rejected 2× — *Asian Perspectives* (2026-03-17, AI flag), *EGQSJ* (2026-04-16, desk: structure/wording; science "certainly interesting"). **v2.0 rewritten** (lists→prose). Target **JASREP** (Elsevier, Scopus Q1, free under subscription). Backup: *Archaeological Research in Asia*. Zenodo preprint 10.5281/zenodo.19081502. Co-authors: Amien + Go Frendi. |
| **P7** TOM | `papers/P7_TOM/` | ☠ **DEAD** — peer-rejected. Keep as the record of the defect and the lesson. Do not revive without the exposure reframe and a clean inventory. |
| **P3** Burial depth | `papers/P3_burial_depth/` | DISCONTINUED (2026-03-10, Mata Elang #2). |
| **D2** Mini-NusaRC | `papers/D2_mini_nusarc/` | Data paper — preliminary **radiocarbon** database (80 sites) built for **H-TOM** testing, which is why it sits in this line rather than 01. Zenodo upload pending → [07_career](../07_career/) §1. |

**46 experiments** are assigned to this line (32 primary). Authoritative list:
`docs/EXPERIMENT_INDEX.md` §"By Line of Inquiry" — regenerate with
`python tools/scan_experiments.py`.

**Zero APC is absolute.** *Open Quaternary* (GBP 1,040) and *Internet Archaeology* (GBP 2–3k) are
**not** Diamond OA — do not target them.

---

## Experiments

**Adversarial controls (the robustness scorecard):**
`E086` ADV-1 Japan → PARTIAL (L1 = volcanism × survey deficit, not volcanism alone) ·
`E081` ADV-2 non-volcanic control → INCONCLUSIVE (cave bias is universal; must use **depth** data) ·
`E069` ADV-3 survey → **PASSED** p=0.0015 · `E085` ADV-4 noise → **PASSED** z=11.05 ·
`E087` ADV-5 negative control → resolved by `E107` (Mon-Khmer substrate)

**Rates & mechanisms:** `E002` (eruption history), `E017` tephra PoC, `E075` sedimentation model,
`E132` sedimentation map (**55% error — downgraded**), `E170` lahar flow, `E213` aggradation/exposure
asymmetry
**Depth & subsurface:** `E024` borehole screening, `E101` burial depth model, `E128` OV depth
(independent), `E166` burial depth map, `E197` colonial depth validation
**Exposure & confounds:** `E178` **karst is the hidden 6th factor**, `E109` survey–burial confound,
`E135` organic preservation, `E137` accidental discovery, `E138` detection methods
**Spatial/tephra correlation:** `E001`, `E083`, `E084`, `E117`, `E145`
**Coastal submersion (L2 layer, shared with 01):** `E052`, `E148`, `E156`, `E177`, `E193`
**Comparanda:** `E092`, `E123`, `E126`, `E173` counterfactual Japan, `E188` mainland

---

## Line rules

1. **Every distance-to-volcano number is suspect until re-derived** against
   `volcanoes_java_full.csv`. Check before quoting, always.
2. **Frame as exposure windows, not proximity.** The proximity frame is what equifinality killed.
3. **Depth data beats surface data.** ADV-2's lesson: surface-density comparisons cannot separate
   burial from non-deposition.
4. Downgraded results stay downgraded: `E132` (55% error), `E137` (F4 needs calibration).

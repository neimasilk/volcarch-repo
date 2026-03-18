# HANDOFF — P1 EGQSJ Submission (Post-Lebaran)
**Date paused:** 2026-03-18
**Resume after:** Eid al-Fitr holiday

---

## STATUS SNAPSHOT

| Item | Status |
|------|--------|
| P1 preprint | **LIVE** — DOI: 10.5281/zenodo.19081502 |
| P1 Copernicus reformat | **DONE** — `submission_egqsj_v1.0.tex` compiles clean |
| P1 EGQSJ submission | **NEXT** — register + upload |
| 6 other papers | Under review (P2/P5/P7/P8/P9) or drafting (P11/P16/P17/P18) |

---

## P1 FILES (ready to submit)

```
papers/P1_taphonomic_framework/
├── submission_egqsj_v1.0.tex    ← SUBMIT THIS (Copernicus format)
├── submission_egqsj_v1.0.pdf    ← compiled PDF (1.22 MiB)
├── references.bib               ← bibliography
├── figures/                     ← fig0a, fig1, fig2, fig3, fig4
├── copernicus.cls               ← required by template
├── copernicus.bst               ← required by template
├── copernicus.cfg               ← required by template
└── CANONICAL.md                 ← version history
```

---

## EGQSJ SUBMISSION STEPS (do these post-Lebaran)

### Step 1 — Before submitting, fix these small items:
1. **Go Frendi ORCID** — add to line: `\Author[1]{Go Frendi}{Gunawan}` → `\Author[1][orcid@here]{Go Frendi}{Gunawan}`
2. **GitHub URL** — replace `[repository]` in `\codedataavailability{}` with actual repo URL
3. **Figure names** — Copernicus prefers `fig01.jpg` etc. Not mandatory, can rename at upload.

### Step 2 — Register at Copernicus
- URL: https://editor.copernicus.org/
- Select journal: **E&G Quaternary Science Journal**
- Register manuscript → you'll get an upload link

### Step 3 — Upload files
- PDF manuscript (with line numbers — Copernicus adds automatically via cls)
- LaTeX source: `submission_egqsj_v1.0.tex` + `references.bib` + `copernicus.cls/bst/cfg`
- Figures: all 5 figures from `figures/` directory

### Step 4 — Cover letter (draft below)

```
Dear EGQSJ Editorial Board,

We submit the manuscript "Multi-Site Calibration of Volcanic Sedimentation Rates
and Implications for Archaeological Visibility in Java, Indonesia" for consideration
in E&G Quaternary Science Journal.

This paper presents a quantitative geoarchaeological framework for volcanic
taphonomic bias in Java. Using four sites with independently documented
construction dates and measured burial depths — spanning two volcanic systems
(Kelud and Merapi) and 350 km of Java — we establish a mean sedimentation rate
of 4.4 ± 1.2 mm/yr as a Java-wide taphonomic baseline. At this rate, pre-Hindu
remains now lie at depths exceeding surface survey capability, potentially
explaining why Java's early archaeological record is sparse relative to
non-volcanic regions of the same archipelago.

The manuscript falls squarely within EGQSJ's scope of geoarchaeology,
Quaternary geology, and geomorphology. A preprint is available at
https://doi.org/10.5281/zenodo.19081502.

We disclose the use of AI language tools as detailed in the manuscript's
AI assistance disclosure statement.

The authors declare no conflict of interest. All data and code will be made
publicly available upon acceptance.

Sincerely,
Mukhlis Amien
Lab Data Sains, Universitas Bhinneka Nusantara, Indonesia
amien@ubhinus.ac.id
```

### Step 5 — Suggested reviewers (optional but helpful)
- Volcanic taphonomy specialists
- Southeast Asian geoarchaeologists
- Copernicus typically handles reviewer selection

---

## PENDING REFERENCE VERIFICATION (manual tasks)

These references need manual DOI verification before final submission:

| Key | Issue | Action |
|-----|-------|--------|
| gertisser2012 | DOI 10.1007/s00445-012-0591-3 — unverified | Check SpringerLink |
| miksic2004 | Wrong DOI in file | Search "Miksic Singapore archaeology" DOI |
| french2003 | Wrong DOI in file | Search "French geoarchaeology" DOI |
| baylisssmith1980 | Chapter details incomplete | Verify book title + pages |
| manguin2011 | Now @incollection, pages 113-136 | Verify page range |

---

## OTHER PAPERS — POST-LEBARAN QUEUE

| Priority | Paper | Next action |
|----------|-------|-------------|
| 1 | P1 EGQSJ | Submit (see above) |
| 2 | P11 | User manual review → submit to indonesia-journal@cornell.edu |
| 3 | P17 | User review → Fig 6 → Antiquity guidelines → submit |
| 4 | P16 | User review → DSH guidelines → submit |
| 5 | P18 | HOLD — wait for 1 acceptance |

---

## WHY EGQSJ (not Open Quaternary)

Open Quaternary charges **£1,040 APC** — not Diamond OA as previously assumed.

EGQSJ is truly free:
- Diamond OA: APC covered by DEUQUA (German Quaternary Association)
- Scopus (since 2022) + Web of Science
- Scope: *geoarchaeology, geomorphology, Quaternary geology, paleo-environments*
- Publisher: Copernicus Publications

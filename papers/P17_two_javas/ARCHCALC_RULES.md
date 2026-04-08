# Archeologia e Calcolatori — Editorial Rules for P17

**Downloaded:** 2026-03-31
**Source:** https://www.archcalc.cnr.it/pages/policy + https://submission.archcalc.cnr.it/
**Journal:** Archeologia e Calcolatori (CNR, Italy)
**ISSN:** 1120-6861 (print), 2385-1953 (electronic)
**Open Access:** Diamond OA since 2005. **Zero APC, zero submission fee.**
**License:** CC BY-NC-ND 4.0
**Indexing:** Scopus + Clarivate ESCI (Emerging Sources Citation Index)
**Languages:** English, French, German, Italian, Spanish

---

## Critical Requirements

| Rule | P17 Status | Action |
|---|---|---|
| **Max 6,000 words** (incl. refs, captions, author info) | **~7,000 — OVER** | **Trim ~1,000 words** |
| **Double-blind review** | Self-citations identifiable | **Anonymize** |
| **Word/RTF only** (no LaTeX) | P17 = LaTeX | **pandoc conversion** |
| **Max 10 figures+tables combined** | 5 figs + tables | OK |
| **Figures: JPG/TIFF, 300 dpi** | Current = PDF/PNG | **Convert to JPG** |
| **Figures in separate ZIP** | In-document | **Extract + ZIP** |
| **Captions in separate file** | In-document | **Extract to DOCX** |
| **Bibliography in separate file** | In-document | **Extract to DOCX** |
| **Paragraphs numerically enumerated** | Not numbered | **Add numbering** |
| **No footnotes** | Check | **Move to text if any** |
| **Submission portal** | — | https://submission.archcalc.cnr.it/ |
| **Deadline** | — | **December 31, 2026** |

## Manuscript Format

- **Font:** Times New Roman, 12 pt, justified
- **Spacing:** 24-point line spacing
- **Paragraph indent:** 1 cm
- **Footnotes:** NOT allowed — integrate into main text
- **Paragraphs:** Must be numerically enumerated

### Heading Hierarchy
- Article title: ALL CAPITALS
- Level 1 headings: Small caps
- Level 1.1 subheadings: *Italics*
- Level 1.2 subheadings: Regular font

### Typographic Conventions
- Em dash (—) with spaces before/after
- En dash (–) for ranges, no spaces
- French quotation marks «» for text citations in bibliography
- Latin/foreign terms in italics only if uncommon; *ibid.*, *supra*, *infra*, *et al.* always italic
- Metric abbreviations without periods (km, m, kg)
- Cardinal directions: N, S, E, W (caps, no periods)
- Full URLs with protocol (https://...)

## Citation Style: Harvard (Author-Date)

### In-text
- Single author: (Rossi 1994)
- Three or more: (Rossi et al. 1995)
- Multiple citations: separated by semicolons, chronological

### Bibliography (separate file)
- All authors listed in full (no "et al.")
- Journal titles in full (no abbreviations), in French quotation marks «»
- DOI or URL required for online sources
- Alphabetical order

### Bibliography Format Examples
- **Book:** Last, First Initial. Year, *Title*, Place, Publisher.
- **Article:** Last, First Initial. Year, *Title*, «Journal Title», volume, issue, pages (DOI/URL).
- **Proceedings:** Include editors with (eds.) and italicized location/year.

### Zotero CSL Available
https://submission.archcalc.cnr.it/public/journals/2/archeologia-e-calcolatori.csl

## Submission Package (4 separate files)

1. **Manuscript text** (DOCX/DOC/RTF) — without bibliography, without figures, without captions
2. **Bibliography** (separate DOCX/RTF)
3. **Figures** (single ZIP archive, JPG/TIFF, 300 dpi)
4. **Figure captions** (separate DOCX/RTF)

## Review Process

- Double-blind peer review
- Editor-in-Chief assessment → Scientific Committee → external reviewers
- Authors can correct only on first proofs

## P17 Conversion Checklist

- [x] Trim from ~7K to ≤6K words — **DONE** (~5.2K words)
- [x] Convert LaTeX → Word (pandoc) — **DONE** (`archcalc_submission/P17_manuscript.docx`)
- [x] Anonymize: remove author name, affiliation, self-identifying references — **DONE**
- [x] Number all paragraphs — **DONE** (88 paragraphs numbered via `format_for_archcalc.py`)
- [x] Format headings per hierarchy (ALL CAPS title, small caps L1, italic L1.1) — **DONE** (automated)
- [x] Remove all footnotes (move content to main text) — **DONE** (no footnotes in v0.3)
- [x] Extract bibliography to separate DOCX — **DONE** (`archcalc_submission/P17_bibliography.docx`)
- [x] Extract figure captions to separate DOCX — **DONE** (`archcalc_submission/P17_figure_captions.docx`)
- [x] Convert figures to JPG 300 dpi, package in ZIP — **DONE** (`archcalc_submission/P17_figures.zip`)
- [x] Check figure count ≤ 10 — **OK** (5 figures + 2 tables = 7, within limit)
- [ ] Download Zotero CSL for bibliography formatting — **OPTIONAL** (bibliography already formatted manually)
- [x] Update experiment count (107→175) — **DONE** (updated 2026-04-01)
- [ ] Create missing Fig 6 — **DROPPED** (5 figures sufficient, within limit)
- [ ] Create account at https://submission.archcalc.cnr.it/ — **MANUAL**
- [ ] Check spelling consistency throughout — **MANUAL** (final proofread)

### Remaining MANUAL steps for Pak Amien:
1. ~~Open `P17_manuscript.docx` and add paragraph numbers~~ → **DONE** (automated, 88 paragraphs)
2. ~~Format headings~~ → **DONE** (automated: ALL CAPS title, small caps sections, italic subsections)
3. ~~Convert bibliography .txt to .docx~~ → **DONE** (`P17_bibliography.docx`)
4. Open `P17_manuscript_formatted.docx` in Word → **verify formatting looks correct**
5. Final proofread for spelling consistency — **mostly clean** (British English standardised, only LaTeX commands have American spelling)
6. Create account at https://submission.archcalc.cnr.it/
7. Upload 4 files: `P17_manuscript_formatted.docx`, `P17_bibliography.docx`, `P17_figures.zip`, `P17_figure_captions.docx`

# P11 Submission Preparation — Indonesia (Cornell SEAP)

**Target:** *Indonesia* journal, Cornell University Press / Southeast Asia Program
**ISSN:** 0019-7289 | Scopus Q2 | Free (no APC)
**Submit to:** indonesia-journal@cornell.edu or Sarah Grossman (sg265@cornell.edu)

---

## Journal Requirements

| Requirement | Status | Action needed |
|---|---|---|
| Double-spaced | **DONE** | Already double-spaced |
| MS Word format | **DONE** | `draft_v0.3_cornell_chicago.docx` |
| Chicago Manual of Style 17th ed. (notes-bibliography) | **DONE** | Converted via `convert_to_chicago.py` |
| Word limit | ~15,000 | Current ~3,200 words — well within limit |

## Conversion Pipeline

### 1. LaTeX → Chicago notes-bibliography — DONE
- **Script:** `convert_to_chicago.py`
- Converts `\citep{key}` → `\footnote{full Chicago citation}` (12 footnotes)
- Converts `\citet{key}` → `Author Name\footnote{full citation}`
- Replaces natbib bibliography with Chicago-formatted bibliography section
- **Output:** `draft_v0.3_chicago.tex`

### 2. LaTeX → Word — DONE
```bash
pandoc draft_v0.3_chicago.tex --from=latex --to=docx -o draft_v0.3_cornell_chicago.docx
```
- **Result:** 161 paragraphs, 12 footnotes, bibliography at end
- Figures embedded as PDF (replace with PNG before submission)

### 3. Post-conversion cleanup needed (in Word)
- [ ] Fix Unicode: em-dashes (—), tildes (~) may show as `?` or `□`
- [ ] Replace PDF figure embeds with PNG versions
- [ ] Check table formatting
- [ ] Verify footnote numbering is sequential
- [ ] Add page numbers
- [ ] Verify bibliography alphabetical ordering

## Content Checklist

- [x] Abstract with keywords
- [x] E084 inscription-volcano spatial divergence (MW p=5.2e-08)
- [x] E083 tephra-archaeological correlation (51 pairs, 3.41m mean)
- [x] E086 Japan comparandum (mandatory scope restriction)
- [x] E069 survey intensity control (p=0.0015)
- [x] Figures 1-2 embedded (polar bearings, Penanggungan)
- [x] Data availability statement
- [x] AI disclosure
- [x] **14 references** (all verifiable, no self-citations)
- [x] Liangan validation section (5.2)
- [x] DHARMA citation included
- [x] Schiffer 1987 (formation processes) included
- [x] Sheets 2002 (Cerén) included
- [x] Abbas 2016 (Liangan) included
- [x] Compiles at 18pp double-spaced (~3,200 words)
- [x] Chicago 17th notes-bibliography conversion done

## Pre-submission Review

- [ ] Read aloud for flow (especially Japan paragraph)
- [x] Verify all numbers match experiment READMEs — ALL MATCH (checked 2026-03-16)
- [ ] Decide: include fieldwork target table or keep "available on request"?
- [ ] Clean up Word file (Unicode, figures, tables, footnotes)
- [ ] Write cover letter
- [ ] **Confirm GitHub repo is public** before submission (data availability statement references it)

## References (14 total, all published)

| Ref | Status |
|---|---|
| Abbas 2016 | Published (Kepel Press) |
| Barnes 2003 | Published (Japan Review 15) |
| DHARMA 2024 | Database (CNRS/ERC) |
| Dumarcay 1993 | Published (Oxford UP) |
| GVP 2023 | Database (Smithsonian) |
| Lavigne & Thouret 2003 | Published (Geomorphology 49) |
| Mohr 1938 | Historical |
| Schiffer 1987 | Published (Univ New Mexico Press) |
| Sheets 2002 | Published (Univ Texas Press) |
| Shimoyama 2002 | Published (Routledge) |
| Soekmono 1995 | Published (Brill) |
| Takata & Yanase 2022 | Published (Internet Archaeology 58) |
| Thouret 1999 | Published (Earth-Science Reviews 47) |
| Whitten et al. 1996 | Published (Periplus) |

## Files

| File | Purpose |
|---|---|
| `draft_v0.3.tex` | Original LaTeX (natbib author-date) |
| `draft_v0.3_chicago.tex` | Modified LaTeX (Chicago notes-bibliography) |
| `draft_v0.3_cornell_chicago.docx` | **SUBMISSION FILE** — Word with footnotes |
| `convert_to_chicago.py` | Conversion script |
| `p11_references.bib` | BibTeX database (for reference) |
| `chicago-fullnote-bibliography.csl` | (not used — conversion done via script) |
| `figures/fig1_candi_polar_bearings.png` | Figure 1 (300 DPI) |
| `figures/fig2_penanggungan_westclustering.png` | Figure 2 (300 DPI) |

## Status

**READY FOR USER REVIEW.** Remaining steps:
1. Pak Amien reads Word file, checks flow and accuracy
2. Clean up Word formatting (Unicode, figures, page numbers)
3. Write cover letter
4. Email to indonesia-journal@cornell.edu

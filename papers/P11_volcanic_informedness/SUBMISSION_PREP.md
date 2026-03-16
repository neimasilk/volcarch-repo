# P11 Submission Preparation — Indonesia (Cornell SEAP)

**Target:** *Indonesia* journal, Cornell University Press / Southeast Asia Program
**ISSN:** 0019-7289 | Scopus Q2 | Free (no APC)
**Submit to:** indonesia-journal@cornell.edu or Sarah Grossman (sg265@cornell.edu)

---

## Journal Requirements

| Requirement | Status | Action needed |
|---|---|---|
| Double-spaced | Done | Already double-spaced |
| MS Word format | **DONE** | `draft_v0.3_submission.docx` via pandoc |
| Chicago Manual of Style 17th ed. | **NOT DONE** | Reformat citations in Word (see below) |
| Word limit | ~15,000 | Current ~3,200 words — well within limit |

## Conversion Steps

### 1. LaTeX → Word — DONE
```bash
pandoc draft_v0.3.tex -o draft_v0.3_submission.docx --from=latex --to=docx
```
**Result:** Clean conversion. 155 paragraphs, 4 tables, 36 headings, 2 figures (PDF embeds).
**Fixed:** `\degree` → `^{\circ}` for pandoc compatibility.

### 2. Citation Style: natbib → Chicago 17th
Current format: inline `\begin{thebibliography}` (natbib author-year).

P11 v0.3 has **no self-citations** (all removed). 10 references, all published.

Chicago 17th **author-date** (similar to natbib):
> (Barnes 2003)

Chicago 17th **notes-bibliography** (if journal prefers):
> Barnes, "Origins of the Japanese Islands: The New 'Big Picture,'" *Japan Review* 15 (2003): 3–50.

**Action needed:** Email editor to confirm which Chicago variant before final formatting.

### 3. Figures — DONE
- `figures/fig1_candi_polar_bearings.png` — 300 DPI ready
- `figures/fig2_penanggungan_westclustering.png` — 300 DPI ready
- Figs 3-5 excluded (wrong framing — old "Volcanic Informedness" not "Temple Siting")
- Word file has PDF embeds; replace with PNG in final submission

### 4. Post-conversion cleanup needed (in Word)
- [ ] Fix Unicode: em-dashes (—), tildes (~), special chars may show as `?` or `�`
- [ ] Insert PNG figures replacing PDF embeds
- [ ] Check table formatting
- [ ] Verify cross-references (Section numbers, Figure references)
- [ ] Add page numbers

## Content Checklist

- [x] Abstract with keywords
- [x] E084 inscription-volcano spatial divergence (MW p=5.2e-08)
- [x] E083 tephra-archaeological correlation (51 pairs, 3.41m mean)
- [x] E086 Japan comparandum (mandatory scope restriction)
- [x] E069 survey intensity control (p=0.0015)
- [x] Figures 1-2 embedded (polar bearings, Penanggungan)
- [x] Data availability statement
- [x] AI disclosure
- [x] 10 references (all verifiable, no self-citations)
- [x] Compiles at 18pp double-spaced (~3,200 words)
- [x] Word conversion done (`draft_v0.3_submission.docx`)

## Pre-submission Review

- [ ] Read aloud for flow (especially Japan paragraph)
- [x] Verify all numbers match experiment READMEs — ALL MATCH (checked 2026-03-16)
- [ ] Decide: include fieldwork target table or keep "available on request"?
- [ ] Email editor to confirm citation style preference before final formatting
- [ ] Clean up Word file (Unicode, figures, tables)

## References (10 total, all published)

| Ref | Status |
|---|---|
| Barnes 2003 | Published (Japan Review 15) |
| Dumarcay 1993 | Published (Oxford UP) |
| GVP 2023 | Database (Smithsonian) |
| Lavigne & Thouret 2003 | Published (Geomorphology 49, pp 45-69) |
| Mohr 1938 | Historical |
| Shimoyama 2002 | Published (Routledge) |
| Soekmono 1995 | Published (Brill) |
| Takata & Yanase 2022 | Published (Internet Archaeology 58) |
| Thouret 1999 | Published (Earth-Science Reviews 47) |
| Whitten et al. 1996 | Published (Periplus) |

**Note:** Self-citations (Amien 2026a, Amien & Gunawan 2026) were REMOVED per user request. All references are to published third-party works.

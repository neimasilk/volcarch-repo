# P8 — what is in the journal's portal (state of 2026-10-12, about 10:50 WIB)

**In one line: after the final review of 12 October the draft R1 (OL-03-2026-11R1) holds nine files again — six of them new
today — with every approval ticked and the portal's message "Your manuscript is ready for submission"; it is NOT submitted.**
Supersedes `PORTAL_STATE_20261008.md` (state from 8 to 12 October). Why the files changed: `PACKAGE_R1_20261012.md`.
What the PI reads: `BACA_DULU_P8_20261012.pdf` (17 pp.).

## 1. Done on 12 October (PI: "baca handoff, review sekali lagi P8, kita selesaikan sekarang, dan pastikan semua gambar, diagram dan referensi sudah sesuai" · "dan pastikan semua sesuai aturan dan template")

| Step | What | Check |
|---|---|---|
| Before anything was replaced | the nine files of 8 October fetched inside the logged-in session and hashed in the browser | nine of nine identical to the table of `PORTAL_STATE_20261008.md` §2 |
| Files | Article File, Figure 1, Figure 2, Author Cover Letter, Revision Summary, Supplemental Material replaced (rows 0, 1, 2, 6, 7, 8); Figures 3–5 untouched; order unchanged | **G12: nine of nine identical to the local files** (table below) |
| Figure 2 caption (portal field) | 'uncoded' → "uncoded", as in the article's caption | read back on the review page |
| Abstract (portal field, step *Title*) | replaced by the abstract of the article v0.4 | SHA-256 of the field equal to that of the abstract paragraph of the article file before saving; read back on the manuscript page |
| Merged PDF (what reviewers get) | regenerated twice (the article file was replaced a second time: the minus sign of Table 8 stood on a line of its own); the first conversion **read page by page, 34 pages**; the second differs from it only in the ten cells of Table 8 (text compared) and its Table 8 page was looked at | no author name, affiliation, DOI or repository string in its text; article 24 pp.; Figures 1 and 2 are the new drawings, the "Figure n" label on the white band |
| Converted PDFs | cover letter (2 pp.) and response (8 pp.) read in full as the portal converted them; tracked copy 33 pp. with markup; no identity string in response or tracked copy | — |
| Approvals | the four Approve boxes (merged PDF, cover letter, revision summary, supplemental) ticked after each PDF was opened from its row; step *Submit Manuscript* read: "Your manuscript is ready for submission"; left with **Save and Exit** ("Submission data saved"; R1 listed under Author Tasks → Submit Manuscript) | **the Submit Manuscript button was not pressed** |

Unchanged in the portal: title, running title, keywords, subject areas, authors, conflict-of-interest answers, manuscript
comment, Notes for Editors, the captions of Figures 1 and 3–5.

## 2. The nine files in the portal (identical to these local files)

| Portal slot | File | Bytes | SHA-256 | State |
|---|---|---|---|---|
| Article File | `P8_revision_v0.4_anonymous.docx` | 61,679 | `6edc2a081827cc06620afff58d2b80ca094aaa4e79b145e4f531d55970ce5d71` | uploaded 12 Oct |
| Figure 1 | `F1_input_importance.tif` | 740,833 | `cc04efdfeb8690dc7974dc215131f37cfe1350bcf83030f1d3149b6fa535a8aa` | uploaded 12 Oct (row labels) |
| Figure 2 | `F2_workflow.tif` | 586,485 | `7a9af08488f6ba7b55cd3fe5ca28bff3fcc4f6767fd7ebb7bb851674a1d01012` | uploaded 12 Oct (redrawn) |
| Figure 3 | `F3_pmp_classes.tif` | 587,863 | `39ea2cf7af3b8f58d0cb8e9da94c3a9896099284bf5cdd99ab20d1eafb249a84` | of 8 Oct |
| Figure 4 | `F4_heldout_auc.tif` | 221,541 | `c1e4cf044315696c46077bb34a816fb367e9b8afd7ef5505089e49ebe76f52d3` | of 8 Oct |
| Figure 5 | `F5_glottal_by_list.tif` | 254,929 | `3b78cb6d079fea44e855b4106ba078cc1d42930768a11992f99cc2c6efc75a36` | of 8 Oct |
| Author Cover Letter | `COVER_LETTER_v0.5.docx` | 12,957 | `cf60519819d5ca1a85ee7ecfe81c628f40123b479b2720a50a066a575c106cad` | uploaded 12 Oct |
| Revision Summary | `RESPONSE_TO_REVIEWERS_v0.5.docx` | 23,137 | `c3abeafdb1b50094f77d7edf453f3707af13a0037839efe4d6717164a7d7981c` | uploaded 12 Oct |
| Supplemental Material | `P8_revision_v0.4_tracked_changes.docx` | 102,140 | `b90b010d2c1ff011c6fa872fbe2f584c9f0cfef857922ccd6043bda7fd3428ba` | uploaded 12 Oct |

Local paths: the `.docx` files in this folder; the `.tif` files in `experiments/E229_p8_revision_tables/results/` (git-ignored;
Figures 1 and 2 rebuilt by `11_figures_1_2_final.py`, then `10_tiff_top_band.py`). **Do not rebuild a file that is in the portal
unless it is to be replaced: a rebuilt `.docx` has a new hash.** `python -X utf8 make_manifest.py --check` prints the local hashes.

## 3. What is still open before Submit

> **12 October, about 11:05 — the PI:** "karena hari ini dan besok saya ada kesibukan, mungkin rabu akan saya kerjakan bagian saya untuk kemudian kita submit final papernya, btw, bikin handoff dulu, rabu kita buka dan kita bahas apa tugas saya". Nothing was done in the portal after *Save and Exit*. On Wednesday 14 October
> the handoff is opened and his tasks are discussed first (`docs/HANDOFF_20261012.md` §1), then the items below.

1. **The PI's word "submit"** (he has not read the changes of today when this was written; the reading copy lists them with
   six small decisions at their defaults).
2. **Commit and push before the click:** the data statement says the scripts and result files are in the public repository;
   `experiments/E228_…/04_pbt_matched_entry_recount.py` with its result file and `experiments/E229_…/11_figures_1_2_final.py` are new
   today and not yet public.
3. Mailbox once more at the moment of Submit (Sander; nothing by 09:20 on 12 October).
4. The co-author has not been told of today's changes (PI).

## 4. If something has to change

As in `PORTAL_STATE_20261008.md` §4, with today's file names: the article → `make_v04_from_v03.py` (or edit `P8_revision_v0.4.md`),
`build_docx.py --anon` and without, the tracked copy (`tracked_build/README.md`), `check_letters.py`; rows 0 and 8. A letter →
its `.md`, `build_letters.py COVER_LETTER` or `RESPONSE_TO_REVIEWERS`; row 6 or 7. A figure → its script in E229, then
`10_tiff_top_band.py`; row 1–5 and *Save* in the row's form. Then *Review Manuscript Files*: wait for the conversion, open each
PDF from its row, tick Approve, *Save and Exit*, and hash the nine files back.

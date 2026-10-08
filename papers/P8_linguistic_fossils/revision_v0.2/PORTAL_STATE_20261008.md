# P8 — what is in the journal's portal, at Zenodo and on GitHub (state of 2026-10-08, about 14:30 WIB)

**In one line: the revision R1 (OL-03-2026-11R1) sits in the portal as a saved draft with its nine final files, every approval
ticked and the portal's own message "Your manuscript is ready for submission" — and it is NOT submitted.** The PI said on 8 October:
do what is needed with the browser, he logs in; publishing on Zenodo is fine; the Oceanic Linguistics submission stays a draft;
continue on Monday. What was found and why the files changed: `PACKAGE_R1_20261008.md`. What the PI reads: `BACA_DULU_P8_20261008.pdf`.

## 1. Done on 8 October after the PI's go-ahead

| Step | What | Check |
|---|---|---|
| GitHub | 21 local commits pushed (`b543824` → `010b1f5`) after a scan of the 2.5 MB that would become public (no session link, no credential, none of 27 reviewer/editor phrases, no private path among the changed files) | the five folders `experiments/E227_…` to `E231_…` and `LICENSE` answer on github.com; a file of E229 read back from raw.githubusercontent.com |
| Zenodo | the draft `README.md` replaced **inside the published record** (Zenodo's "Edit published files", meant for minor corrections within 30 days), description brought up to the article's current title, published | public API read without login: **DOI 10.5281/zenodo.23202245 unchanged**, version 1.0; README 20,356 bytes, md5 `3952faf2…`, byte-identical to `experiments/E228_p8_revision_analyses/release/README.md`, no placeholder left; the five CSV files unchanged (`release/ZENODO_METADATA.md`) |
| Portal — files | Article File, Figures 1–5, Author Cover Letter, Revision Summary replaced; tracked-changes copy added as Supplemental Material ("Tracked changes", with a one-line description); order Article → Figures 1–5 → cover letter → revision summary → supplemental | **G12: all nine files fetched back inside the logged-in session and hashed in the browser — nine of nine identical to the local files** (table below) |
| Portal — Notes for Editors | rewritten: the cover letter is signed; the response is the Revision Summary and carries no names; a tracked copy is in Supplemental Material; the anonymised Article File redacts the DOI and the repository URL, both are in the cover letter | read back on "Review Manuscript Data" |
| Portal — merged PDF (what reviewers get) | regenerated; downloaded and looked at: 34 pages = article 24 pp., then a caption page and a figure page for each of the five figures; no author name, affiliation, DOI or repository string in its text; the three new sentences, Table 8 (−0.020 twice) and the reference without page range are in it | — |
| Portal — figure label | the portal prints a large "Figure n" over the top of each image (landscape figures rotated); on the first merge it covered the first box of Figure 2 and the first label of Figure 1. All five TIFFs now carry a **white band of 480 px at the top** (E229 amendment A12, `10_tiff_top_band.py`; the drawing below the band is pixel-identical), and on the second merge the label sits on white. One sentence in the cover letter says so | merged PDF downloaded again and looked at |
| Portal — converted PDFs | cover letter (2 pp.), response (8 pp.; ʔ, κ, Δ present; no identity string), tracked copy (34 pp., markup shown; no identity string) downloaded and looked at | — |
| Portal — approvals | the four Approve boxes (merged PDF, cover letter, revision summary, supplemental) ticked after each PDF was opened from its row; "Submit Manuscript" step reads "Your manuscript is ready for submission"; left with **Save and Exit** ("Submission data saved"; R1 listed under Author Tasks → Submit Manuscript) | **the Submit Manuscript button was not pressed** |

Unchanged in the portal: title, running title, abstract, keywords, subject areas, authors, conflict-of-interest answers, manuscript
comment, the five figure titles and captions.

## 2. The nine files in the portal (identical to these local files)

| Portal slot | File | Bytes | SHA-256 | State |
|---|---|---|---|---|
| Article File | `P8_revision_v0.3_anonymous.docx` | 60,309 | `315cf08e0cfdb1c06804eaef105bafe3823fcafa5391fc6ed4f52d5ed3051498` | uploaded 8 Oct |
| Figure 1 | `F1_input_importance.tif` | 741,857 | `da78818081f94e44c6fde47164ac325785eea12a7391453edf163f94656bdcab` | uploaded 8 Oct (legend; top band) |
| Figure 2 | `F2_workflow.tif` | 503,255 | `55eb6bed6830813c6209478268f16661db2969e442276007cbfdc4d8bd634fe4` | uploaded 8 Oct (top band only) |
| Figure 3 | `F3_pmp_classes.tif` | 587,863 | `39ea2cf7af3b8f58d0cb8e9da94c3a9896099284bf5cdd99ab20d1eafb249a84` | uploaded 8 Oct (labels; top band) |
| Figure 4 | `F4_heldout_auc.tif` | 221,541 | `c1e4cf044315696c46077bb34a816fb367e9b8afd7ef5505089e49ebe76f52d3` | uploaded 8 Oct (legend; top band) |
| Figure 5 | `F5_glottal_by_list.tif` | 254,929 | `3b78cb6d079fea44e855b4106ba078cc1d42930768a11992f99cc2c6efc75a36` | uploaded 8 Oct (top band only) |
| Author Cover Letter | `COVER_LETTER_v0.4.docx` | 12,821 | `cf030d284fa85a9bdc307987bbcb600cc58083de2bcb61d4aa81bc501ab5f1d5` | uploaded 8 Oct |
| Revision Summary | `RESPONSE_TO_REVIEWERS_v0.4.docx` | 23,043 | `42b10c8b907dbcc930ea3075d9bac00f2961d03e91d14dc3e45fa92296252415` | uploaded 8 Oct |
| Supplemental Material | `P8_revision_v0.3_tracked_changes.docx` | 101,422 | `61bb7d36791946d0ed05fb1052ef2bb8606ef2fc3478421e4e7556259b929624` | uploaded 8 Oct (new, ninth file) |

Local paths: the `.docx` files in this folder; the `.tif` files in `experiments/E229_p8_revision_tables/results/` (git-ignored; rebuilt by
`07_figures_revision.py`, `08_figure1_legend.py`, then `10_tiff_top_band.py`). **Do not rebuild a file that is in the portal unless it
is to be replaced: a rebuilt `.docx` has a new hash.** `python -X utf8 make_manifest.py --check` prints the current local hashes.

## 3. What is still open before Submit

1. **The PI reads** `BACA_DULU_P8_20261008.pdf` (15 pp.: one page of what changed, the cover letter, the response, three figures) and
   says yes or names what to change. The manuscript's three new sentences, the Table 8 cells, the dropped page range and the figure
   changes are listed on its first page; the fallback if he rejects them is in `PACKAGE_R1_20261008.md` §2.
2. **Mailbox:** Sander's answer on the corrections and on a further round. Checked twice on 8 October (12:40, 13:35): nothing; nothing
   from arXiv either. If he answers, the cover letter's sentence "not having heard otherwise …" is adapted (then
   `build_letters.py COVER_LETTER`, `check_letters.py`, replace that one file in the portal, re-approve). If he asks for another
   procedure, Submit waits.
3. **Co-author** (Go Frendi Gunawan) told of the final state — the PI's step.
4. **Submit:** PI logs in → open the draft → confirm the nine files against the table above (hash in the browser as on 8 October) →
   the PI says "submit" → *Submit Manuscript* → save the confirmation → records the same day.

## 4. If something has to change after the PI's reading

- A letter: edit the `.md`, `python -X utf8 build_letters.py COVER_LETTER` (or `RESPONSE_TO_REVIEWERS`), `python -X utf8 check_letters.py`,
  then in the portal: *Files* → the row's Replace icon (`#replace_browse_6` cover letter, `_7` revision summary) → *Review Manuscript
  Files* → open the PDF from the row → tick Approve → Save and Exit.
- The article: edit `P8_revision_v0.3.md`, `python build_docx.py --anon` and without `--anon`, rebuild the tracked copy
  (`tracked_build/README.md`), `check_letters.py`; replace rows 0 and 8; look at the merged PDF again; approve.
- A figure: its script in E229, then `10_tiff_top_band.py`; replace the row (1–5), press Save in the row's form, look at the merged PDF.
- Portal mechanics learnt on 8 October are in the technical notes of `docs/HANDOFF_20261008.md` (§9).

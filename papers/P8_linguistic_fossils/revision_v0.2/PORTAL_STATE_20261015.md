# P8 — what is in the journal's portal (state of 2026-10-15)

**In one line: after the editor's reply of 13 October, the two letters were replaced by v0.6 (rows 6 and 7) and the Notes for
Editors field updated; the other seven files are byte for byte those of 12 October.** Supersedes `PORTAL_STATE_20261012.md`.
Why the letters changed: `docs/correspondence/EMAIL_OL_EDITOR_P8_REPLY_20261013.md`; ledger C071.

## 1. Done on 15 October (PI: "baca handoff marik kita selesaikan P8, kabar terbaru: [the editor's reply]"; then, to three questions: six small decisions at their defaults; the co-author has agreed; "Ya, submit hari ini")

| Step | What | Check |
|---|---|---|
| Before anything was replaced | the nine files of 12 October fetched inside the logged-in session (the PI logged in; the session of 12 October had expired) and hashed in the browser | **nine of nine** identical to `PORTAL_STATE_20261012.md` §2 |
| Letters | v0.6 built (`build_letters.py`), `check_letters.py` passed (numbers, section references, 38 reviewer points, anonymity), two independent reads of the new note (Sonnet; first pass 4 overstatements + 3 gaps, second pass 1 inaccuracy + 1 overstatement; all verified at the source and corrected) | — |
| Files | rows 6 (Author Cover Letter) and 7 (Revision Summary) replaced; file types unchanged | hashes of the uploaded `.docx` fetched back = local (table below) |
| Notes for Editors (field `alternate_abstract`, step *Title*) | the Revision Summary now said to open with the note the editor asked for on 13 October; the editor's agreement of 13 October; "The use of artificial intelligence is declared in the article." Title, running title and abstract untouched (abstract 1,024 characters, as on 12 October) | Submission data saved |

## 2. The nine files

| Portal slot | File | Bytes | SHA-256 | State |
|---|---|---|---|---|
| Article File | `P8_revision_v0.4_anonymous.docx` | 61,679 | `6edc2a081827cc06620afff58d2b80ca094aaa4e79b145e4f531d55970ce5d71` | of 12 Oct |
| Figure 1 | `F1_input_importance.tif` | 740,833 | `cc04efdfeb8690dc7974dc215131f37cfe1350bcf83030f1d3149b6fa535a8aa` | of 12 Oct |
| Figure 2 | `F2_workflow.tif` | 586,485 | `7a9af08488f6ba7b55cd3fe5ca28bff3fcc4f6767fd7ebb7bb851674a1d01012` | of 12 Oct |
| Figure 3 | `F3_pmp_classes.tif` | 587,863 | `39ea2cf7af3b8f58d0cb8e9da94c3a9896099284bf5cdd99ab20d1eafb249a84` | of 8 Oct |
| Figure 4 | `F4_heldout_auc.tif` | 221,541 | `c1e4cf044315696c46077bb34a816fb367e9b8afd7ef5505089e49ebe76f52d3` | of 8 Oct |
| Figure 5 | `F5_glottal_by_list.tif` | 254,929 | `3b78cb6d079fea44e855b4106ba078cc1d42930768a11992f99cc2c6efc75a36` | of 8 Oct |
| Author Cover Letter | `COVER_LETTER_v0.6.docx` | 13,056 | `5b6fdf2298b2072682d6d596aba828bd03f4f440f73777745703113a2ded533d` | **uploaded 15 Oct** |
| Revision Summary | `RESPONSE_TO_REVIEWERS_v0.6.docx` | 25,014 | `5b236788fce74a68f0116577807df9af56ca67e4b44df89389af4b145b205bba` | **uploaded 15 Oct** |
| Supplemental Material | `P8_revision_v0.4_tracked_changes.docx` | 102,140 | `b90b010d2c1ff011c6fa872fbe2f584c9f0cfef857922ccd6043bda7fd3428ba` | of 12 Oct |

`python -X utf8 make_manifest.py --check` prints the local side.

## 3. Submit

(filled in at the moment of Submit)

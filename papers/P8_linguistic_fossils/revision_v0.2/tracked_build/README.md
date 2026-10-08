# Tracked-changes copy of the P8 revision — how it was built (2026-10-08)

Requested by the Managing Editor of *Oceanic Linguistics* (reply of 2026-10-08; `docs/correspondence/EMAIL_OL_EDITOR_P8_REPLY_20261008.md`):
a tracked-changes version beside the clean copy. Result: `../P8_revision_v0.2_tracked_changes.docx` (upload as Supplemental Material).
Claude built it; the PI looks at it before it goes to the portal (ledger C068).

**Why a conversion is needed.** The first submission was a PDF made from LaTeX (`../../draft_v0.1_anonymous.pdf`, `.tex`), so there is no
Word file of the text the reviewers saw. The baseline is a Word conversion of that LaTeX source; the revised side is the file that sits
in the portal as the Article File (`../P8_revision_v0.2_anonymous.docx`, hash checked against the portal copy on 2026-10-07, G12).

## Steps (run from the repo root; `-I` ignores environment variables, so use `-X utf8` for printing)

1. `pandoc papers/P8_linguistic_fossils/draft_v0.1_anonymous.tex -f latex -t docx --citeproc --bibliography=papers/P8_linguistic_fossils/references.bib --csl=papers/P8_linguistic_fossils/revision_v0.2/unified-style-sheet-for-linguistics.csl --lua-filter=tracked_build/dropimg.lua -o baseline_raw.docx`
   — the Lua filter drops the four embedded images (figures are separate files in the OL workflow; captions stay) and turns the 30-odd
   inline math snippets into plain text (`P_sub`, `κ`, `±`, `Δ` …), because Word's Compare treats equations as opaque objects.
2. `python -I tracked_build/make_baseline.py baseline_raw.docx baseline.docx` — "&" → "and" and the serial comma in three-author
   citations, to match how the revision writes them; clears author/last-modified-by properties. Content is not touched.
3. `python tracked_build/word_compare.py baseline.docx ../P8_revision_v0.2_anonymous.docx tracked_raw.docx` — Word's Compare through COM
   (author label "Authors", personal information and date/time removed, custom properties deleted; Word's own user name is set to a
   neutral value for the session and restored in `finally`).
4. `python tracked_build/strip_dates_and_qa.py tracked_raw.docx baseline.docx ../P8_revision_v0.2_anonymous.docx P8_revision_v0.2_tracked_changes.docx`
   — strips the placeholder `w:date` / `w16du:dateUtc` attributes Word leaves after "remove date and time" (hour 29, year 1900), then runs the QA below.
5. Optional: `python tracked_build/render_markup.py P8_revision_v0.2_tracked_changes.docx preview.pdf` (PDF with markup, to look at).

## What was checked (2026-10-08)

| Check | Result |
|---|---|
| Baseline vs the submitted PDF text (6-word shingles, `check_baseline.py`) | 92.9 % of baseline 6-grams occur in the PDF text; the rest is hyphenation, page numbers and natbib's "et al." citation form (the baseline prints the full author list, as the revision does). Paragraph-level diffs (`diffpar.py`) show nothing else. |
| Math in the baseline | 0 equation objects left; `P_sub`, `κ`, `±` are plain text |
| **Accept all** revisions | text equals the revised article: word-level ratio 1.00000 (10,483 words) |
| **Reject all** revisions | text equals the converted submitted text except 4 words ("A (Full)", "B (Phon.)": cells of a table whose rows Word marked as a tracked cell merge; the words are present in the file as tracked deletions) |
| Revisions | 519 in Word's count (980 `w:ins`, 536 `w:del` elements, 11 moved blocks) |
| Anonymity of the file | revision author = "Author" only; no dates; no custom properties; creator / last-modified-by empty; no occurrence of the authors' names, affiliation, e-mail, repository name or the data DOI in any XML part |
| Word's user name after the run | restored (checked) |
| Look | the markup PDF was looked at (pages 1 and 18): the old title and abstract appear struck through, new text underlined; Sections 2–3 are mostly replaced blocks, as the letter says |

**Not claimed:** that the redline is a fine-grained diff — with Sections 2–3 rewritten and sections removed it cannot be. The response letter is the guide.

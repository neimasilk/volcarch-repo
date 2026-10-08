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

## Rebuilt 2026-10-08 (afternoon) against the v0.3 article file

The article file changed after the check of the letters against it (two sentences in Section 2.4, one in Section 2.5, two cells and
one row name of Table 8; rebuilt twice the same afternoon, the second time after the adversarial read added the first sentence of 2.4; `../PACKAGE_R1_20261008.md`), so the tracked copy was rebuilt with the steps above, the revised side being
`../P8_revision_v0.3_anonymous.docx`. Result: `../P8_revision_v0.3_tracked_changes.docx` — **this is the file to upload**; the
v0.2 tracked copy of the morning compares against the superseded article file and stays only as a record.

| Check | Result |
|---|---|
| Accept all | text equals the v0.3 article: word-level ratio 1.00000 (10,614 words) |
| Reject all | text equals the converted submitted text except the same 4 words as before ("A (Full)", "B (Phon.)") |
| Revisions | 519 in Word's count |
| Anonymity (`check_anonymity.py`, new: every XML part searched for names, affiliation, e-mail, repository, DOI; revision authors; dates; properties) | OK for the article file and for the tracked copy. The article file's `app.xml` carries "Company: Linguistics RSPAS ANU" — that string is in the journal's own template (`ol_base.docx`), not ours |
| Word's user name after the run | restored (read back through COM) |
| Look | markup PDF rendered (34 pages with markup); the page with the new sentence of Section 2.5 looked at |

`check_anonymity.py FILE.docx …` exits with 1 if a file fails; run it on every file that can reach a reviewer (article file,
tracked copy, response to the reviewers) after any rebuild.

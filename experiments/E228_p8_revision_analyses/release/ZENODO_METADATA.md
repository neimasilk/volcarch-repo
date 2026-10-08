# Zenodo deposit — fields to paste (prepared 2026-10-07; the PI logs in, Claude fills the form, the PI presses Publish)

**Upload type:** Dataset
**Files:** `p8_forms_all.csv`, `p8_candidates.csv`, `p8_makasar_uncoded_meanings.csv`, `p8_tolaki_pmp_lookalikes.csv`, `p8_cell_examples.csv`, `README.md` (this folder; the README is the column description)
**Title:** Uncoded basic vocabulary in six Sulawesi word lists of the Austronesian Basic Vocabulary Database: form-level data release
**Authors:** Amien, Mukhlis (Universitas Bhinneka Nusantara; ORCID 0000-0002-1848-167X) · Gunawan, Go Frendi (Universitas Bhinneka Nusantara)
**Description (plain text):**
Form-level data for the article "Uncoded basic vocabulary in six Sulawesi word lists: What the absence of a cognate-set assignment in the Austronesian Basic Vocabulary Database does and does not show" (Oceanic Linguistics, manuscript OL-03-2026-11, revised version). One row for each of the 1,357 forms of six ABVD word lists (Muna 27, Bugis 48, Makasar 166, Wolio 192, Tae'/Sa'dan Toraja 226, Tolaki 674; lexibank/abvd repository state 917c5a5, 7 October 2025): ABVD identifiers, the form as recorded, ABVD cognate-set numbers and loan flag, the label (1 = no cognate-set number), out-of-sample scores of a 25-input classifier (form and meaning inputs), of a 26-input variant with a list-identity input, and of the model trained on the other five lists, the cell of the label-by-score table, and the nearest same-meaning look-alike form in other Sulawesi lists and, for Tolaki, in Bungku-Tolaki lists (a mechanical spelling screen, not a cognate judgement). A second file holds the 438 uncoded forms as a subset; three smaller files hold the 70 uncoded Makasar meanings with the Bugis and Sa'dan Toraja forms and the PMP entry, the twelve uncoded Tolaki forms that resemble the PMP entry by the screen, and the 92 example rows behind the article's Table 7. README.md describes every column, the classifier, the look-alike screen and the lists compared. Forms, cognate-set numbers, loan flags and names are copied from ABVD (Greenhill, Blust & Gray 2008; CC BY 4.0); please credit ABVD and the list compilers when reusing. No form has been assessed etymologically by a specialist.
**Keywords:** Austronesian Basic Vocabulary Database; Sulawesi languages; cognate coding; basic vocabulary; Makasar; Tolaki; Bugis; Muna; Wolio; Sa'dan Toraja
**Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
**Version:** 1.0
**Language:** English (metadata); forms in six Sulawesi languages
**Related identifiers:** is supplement to — the journal article (add the article DOI when known); is derived from — ABVD, https://abvd.eva.mpg.de and https://github.com/lexibank/abvd (commit 917c5a5); code — https://github.com/neimasilk/volcarch-repo (folders experiments/E227–E231)
**Funding:** none
**PUBLISHED 2026-10-07 by the PI (Claude filled the form in the Playwright window): record https://zenodo.org/records/23202245 — DOI 10.5281/zenodo.23202245, concept DOI 10.5281/zenodo.23202244.**
**After publishing (done):** copy the DOI into `P8_revision_v0.2.md` (data statement: replace `[DOI]`), rebuild the Word files, and record the DOI in the release README section 0, `docs/WORKSTATE.md` and the line STATE.

---

## Version 1.1 — PREPARED 2026-10-08, NOT YET DEPOSITED (needs the PI's login and his click on Publish)

**Why.** The `README.md` inside the published version 1.0 is the working draft: checked on 2026-10-08 by downloading it from the
public record (20,773 bytes, md5 `3783944aac3876486d11026b010853f7`). It opens with an internal status note ("completed for the PI's
reading before deposit … Still the PI's: read the whole file; fill the two placeholders …"), is headed "Description of the two
data files", and carries unfilled fields: `[title of the revised article]`, "to be confirmed by both authors", `[date of deposit]`,
`[year]`, `[title above]`, `https://doi.org/[DOI]`, `[repository URL]` (twice), and "Makasar (or Makassar — the authors' spelling
decision)". The five CSV files in the record are correct (md5 identical to this folder). The first reviewer asked for exactly this
release, so the description file should be clean before the revision is submitted.

**What changes.** Only `README.md` (the file in this folder, 20,438 bytes after the correction: working note removed, fields
filled, "How to cite" with the concept DOI, repository URL, one line saying that the data files are unchanged since 1.0). No CSV
changes. The local note that said "this copy is the one in the deposit except for this status line" was not true of three more
lines and is gone.

**Steps at the sitting** (Zenodo → the record → *New version*; files cannot be replaced inside a published version):
1. *New version* on record 23202245. Remove the old `README.md`, upload this folder's `README.md`; keep the five CSV files.
2. Version: `1.1`. Publication date: the day of publishing.
3. Description: replace the first sentence by
   `Form-level data for the article "Uncoded basic vocabulary in six Sulawesi word lists of the Austronesian Basic Vocabulary Database" (Oceanic Linguistics, manuscript OL-03-2026-11, revised version).`
   (version 1.0 names the article by an earlier, longer title), and add at the end:
   `Version 1.1 corrects the description file (README.md) only; the data files are unchanged from version 1.0.`
4. The PI presses Publish. Record the new version DOI here and in the line STATE; check the public record afterwards by
   downloading `README.md` and comparing it with this folder's file (G12).
5. **Before this step the 20 local commits must be on GitHub**: the README points to `experiments/E227` … `E231` in the public
   repository, and E229–E231 are not there until the push.

**Which DOI is cited.** The cover letter and the named build of the article (`P8_revision_v0.3.md`) cite the concept DOI
`10.5281/zenodo.23202244`, which resolves to the latest version, so nothing has to be edited after version 1.1 exists. The
anonymised article file redacts the DOI.

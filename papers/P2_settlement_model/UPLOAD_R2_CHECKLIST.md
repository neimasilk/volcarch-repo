# P2 / JCAA #280 — round-2 upload checklist (prepared 2026-11-02)

**When:** as soon as the editor answers the 2 Nov AI email and the answer allows the declaration as written; **if no
answer by Mon 9 Nov, upload anyway** (PI decision 2 Nov). If the editor asks for author-rewritten sections: do not upload;
ask for an extension the same day. **Deadline 16 Nov.**
**Who:** the PI logs in to the portal (account.journal.caa-international.org); Claude may drive the browser with him.

## A. Before the upload (all must be ✅)

| # | Item | Status |
|---|---|---|
| A1 | PI approved the Response to Reviewers (`response_to_reviewers_r2.pdf`) | ✅ 2 Nov ("setuju") |
| A2 | PI read the three papers (`READING_PACK_LINE156_20261102.md`) and the line-156 sentence matches | ✅ 2 Nov ("sudah baca, sesuai") |
| A3 | Go Frendi replied "setuju" (and confirmed his contribution sentence) | ✅ 2 Nov — the PI reports "pak go sudah ok"; the approval email with the three PDFs was sent 10:23 for a written record |
| A4 | Editor's answer read, or 9 Nov reached | ☐ |
| A5 | Commit + push to GitHub (the Data Availability statement points to the repository: E236 code, `clean_sites.py`, the ten-seed extension must be public before the upload) | ☐ |
| A6 | Re-run on the frozen file: `revision_ammo/verify_k1_numbers.py` (39/39), `revision_ammo/verify_headline_numbers.py` (4 known withdrawn), `tools/check_doc_sync.py` | ☐ |
| A7 | Recompile manuscript, supplement, letter; 0 undefined refs; length ≤ 11,500 (11,234 on 2 Nov after C092, whole-PDF method) | ☐ |
| A8 | `SIG_signoff.md` round-2 block: no RED | ☐ |
| A9 | Live journal pages re-checked on the day (rules last checked 2 Nov: 11,500 words incl. references; abstract ≤ 300; APC £651; waiver via Comments to the Editor; AI policy unchanged) | ☐ |

## B. Files (same three components as round 1; portal: Revisions → Upload File → choose component)

| Component | File | Pages |
|---|---|---|
| Manuscript (revised) | `submission_jcaa_v0.3.pdf` | 37 |
| Supplementary file (for review) | `supplementary_tables_v0.3.pdf` | 8 (Tables S1–S9) |
| Response to reviewers | `response_to_reviewers_r2.pdf` | 6 |

Optional, only if the portal offers a figure component: `figures/fig01_study_area_v03.png`, `fig02_progression_v03.png`,
`fig03_artefact_v03.png`, `fig04_criterion_v03.png`, `fig05_priority_map_v03.png`, `fig06_seed_v03.png`,
`fig_e235_elevation_control_v03.png` (all 300 dpi PNG, < 3 MB). The journal asks for separate figure files at the
production stage at the latest; round 1 went to review with the figures embedded.

Rename the files in the portal dialog to plain names (e.g. "P2_revised_manuscript.pdf"), as in round 1.

## C. Metadata

- **Title** (unchanged): An Evaluation Artefact in Presence–Background Archaeological Modelling: Evidence from East Java
  and a Corrected Comparison Protocol
- **Abstract** — the portal copy must be replaced if the form allows it (it is the 2 Oct replacement or older). Plain
  text, 282 words:

> Changing how background points are selected can raise cross-validated AUC, but not necessarily because the model has
> become better. This study tests how background design and evaluation background affect model evaluation, using East
> Javanese presence records, three algorithms and spatial block cross-validation. About a third of the compiled records
> are modern monuments or misplaced, so every real-data contrast is reported on the 378 records and on 161 cleaned
> sites. In a sequence of seven experiments the reported AUC rose from 0.659 to 0.751, attributed to increasingly
> realistic background designs. A common evaluation background removes that rise on the compiled records (redesign
> contributes −0.014 AUC); on the cleaned sites the same redesign contributes +0.051 [+0.039, +0.063] over ten seeds.
> What holds in both is the inflation: own-background scoring overstates the gain by about +0.04 to +0.05 AUC, and only
> a common background shows how much real gain is left. In synthetic data with a known truth, the hybrid design scored
> on its own background is reported above its true performance in all 240 runs. The criterion used to choose a design,
> AUC on its own background, is maximal at the end of the range tested, so it gives no indication of when to stop.
> Priority maps from a single seed are unstable; averaging helps, but two independent five-seed averages agree at a
> Jaccard overlap of only 0.70–0.86, depending on the presence set. Above about 1,000 m, elevation and volcanic
> proximity cannot be separated in the study area. We propose a protocol: hold the evaluation background fixed, state
> the availability domain of every metric, and average priority maps over seeds. The contribution is this diagnosis and
> protocol, not a new prediction of site locations.

- **Keywords** (6): presence-background modelling; pseudo-absence design; evaluation artefact; spatial cross-validation;
  survey bias; East Java

## D. Comments to the Editor (PI approves the wording)

> Following the editorial team's note of 20 October, we will apply for a publisher waiver of the APC if the paper is
> accepted. The use of AI in preparing this revision is declared in full at the end of the manuscript, as described in
> our message of 2 November.

(The journal's checklist says a waiver request "must be made at the time of submission via the Comments to the
Editor"; the 11 March submission already did so. This sentence keeps the request on record for the revision.)

## E. After the upload (G12)

1. Download every uploaded file back from the portal.
2. Compare: page count, text after whitespace normalisation, SHA1 of the normalised text; render page 1 and compare
   visually. Round 1: the portal rewrote Manuscript and Supplementary PDFs (Producer → mPDF 8.3.1), losslessly; the
   Response was untouched.
3. Record the result in `docs/JOURNAL.md`, `lines/01_spatial/STATE.md`, `docs/WORKSTATE.md` and `SIG_signoff.md`.
4. Status in the portal should read "resubmitted for another review round".

# P2 / JCAA #280 — round-2 decision: REVISIONS REQUESTED (2026-10-19)

> **Status 2026-10-20 (later) — PI: "ok, lakukan yg perlu dilakukan" → defaults taken; Claude's part done except references:**
> - **D4 ✅:** waiver email **sent** 08:07 to journal@caa-international.org
>   (`docs/correspondence/EMAIL_JCAA_WAIVER_STATUS_DRAFT_20261020.md`, now the sent record). G15 stays RED until a
>   written answer.
> - **D3 ✅ — E235 run as pre-registered** (design pushed first, `3ac41cc`; numbers re-derived independently):
>   - above **1,000 m**, 97.5 % of land is volcanic and 0.4 % is non-volcanic, so the reviewer's comparison hardly exists in East Java;
>   - below 1,000 m, suitability is +0.05 volcanic − non-volcanic: **INCONCLUSIVE** (on the threshold; post hoc
>     +0.01 to +0.07 depending on zones and matching);
>   - design disagreement: **no claim possible**. *(audit 20 Oct: the M3 test had no power, so no claim; suitability verdict fragile, +0.01 to +0.07 post hoc)*
> - **D1/D2 at default:** the PI writes (at least abstract, introduction and conclusion; **see the JCAA AI policy
>   below**); freeze by 9 Nov with an adversarial read + G1.
> - **Figures 1 and 5 rebuilt** (`build_v03_figures.py`).
> - **`submission_jcaa_v0.3.tex`** = v0.2 + mechanical changes only (`make_v03_mechanical.py`; compiles clean, 29 pp.).
> - **`VENUE.md`** (G15 template) written; key items checked against the fetched pages.
> - **Writing pack for the PI: `REVISION_R2_WRITING_PACK.md`.**
> - **New finding C078:** the v0.2 §3.7 definition of the robust core and its caveat are wrong (pack §5b).
> - **JCAA AI policy (fetched 2026-10-20, editorial policies):**
>   - Unacceptable: "ghost-writing", i.e. AI writing substantial portions (methods, results, discussion).
>   - Acceptable: turning the author's own notes or bullet points into prose; language help.
>   - v0.2's disclosure says AI was used for "drafting and editing of this manuscript". **For v0.3 the PI's own
>     writing of the substantive sections is a journal requirement, not only G16**, and the disclosure must match.
> - **Other venue facts:**
>   - limit 11,500 words including references (v0.2 ≈ 8,350);
>   - abstract ≤ 300 words; ≤ 6 keywords;
>   - **alt text (≤ ~10 words) for every figure**;
>   - Harvard citations, "&" for ≤ 3 authors.
> - **APC £651.** CAA's Publication Fund is for CAA members only (five papers a year).
> - **References done** (`REFERENCE_FIX_20261020.md`):
>   - .bib corrected and re-checked;
>   - **the line-157 sentence is unsupported by all three papers ("strong gains") → the PI rewrites it**;
>   - Harvard-style pass still to do before upload.
> - **Open:**
>   - D6 and D7 (portal login: annotated copy) — the PI's;
>   - tell Go Frendi.
> - **Hygiene:** the round-1 response letter in the repo quotes the round-1 reviewers verbatim (uploaded record,
>   Aug). Whether to scrub it is the PI's call; nothing was changed.

**Recorded:** 2026-10-20 (orbit session; the PI pasted the decision email into the session).
**Manuscript reviewed:** `submission_jcaa_v0.2.pdf` (resubmitted 2026-08-11; title "An Evaluation Artefact in
Presence–Background Archaeological Modelling: Evidence from East Java and a Corrected Comparison Protocol").
**Public-repo rule:** reviewer and editor text is paraphrased here; the original email stays in the PI's mailbox.
Quotations below are of **our own manuscript**, never of the reviewers.

---

## 1. The decision

- **Sender:** the JCAA editorial team, signed for the journal by Cesar Gonzalez-Perez (the round-1 handling editor
  was Philip Verhagen). Received 2026-10-19, 17:58 (PI's mailbox time).
- **Decision: revisions requested.** The revised version may go to further peer review before acceptance.
- **Due within 4 weeks → Monday 16 Nov 2026.** If we cannot make it, the journal asks to be told as early as possible.
- **Upload:** in the portal's Revisions section, one file per article component; the reply goes in the component
  "Response to Reviewers". The journal calls this the last chance for major editing, so the files must be fully
  copy-edited and must follow the author guidelines.
- **An annotated copy of the manuscript** may be waiting in the portal (the email says there may be one).
  **Not yet checked → the PI logs in (D7).**
- **Reviewer recommendations:**
  - **R1:** "Resubmit for Review". Relevance excellent, originality excellent, discussion excellent; framing and
    methods good; **clarity fair**; bibliography and figures good.
  - **R2:** "Revisions Required". Relevance good, originality excellent, discussion excellent, clarity excellent;
    **methods fair**; bibliography and figures good.
- **Reading:** both reviewers rate the contribution highly, and the round-1 rework is acknowledged. **R2's methods
  point is the only substantive one.** It needs analysis, not rewording (SIG banned move). R1's points are about
  presentation, but they are many, and they touch every section.

---

## 2. Point-by-point triage (paraphrased)

Each row runs: point → verified fact → what changes → what the reply must contain. "Verified" means checked on
2026-10-20 against the files named.

### Reviewer 2 (the gate: methods "fair")

| # | Point (paraphrase) | Verified fact | What changes | Type |
|---|---|---|---|---|
| **R2-1** | Still unclear whether the model picks out **volcanic** areas or simply **high ground**. The round-1 answer looks more presentational than substantive. A comparison with **high-elevation non-volcanic** terrain would show which. | Our round-1 answer was E219 part C: coarsened exact matching on elevation × slope × TRI × TWI, uplands ≥ 200 m, quintile bins. Volcanic = ≤ 20 km from a centre; non-volcanic = ≥ 40 km. Manuscript lines 491–493 and 554. **But in the top elevation quintile the matched strata hold 41,878 volcanic cells and only 868 non-volcanic ones** (`E219/results/e219_terrain_matched.csv`, stratum // 125 = elevation bin). So the comparison is carried by low and middle uplands; **at high elevation it hardly exists**. That is exactly what the reviewer suspects. Its site arm has n = 2. The manuscript still reads it as "directionally consistent with a volcanic-specific effect" (line 493). **No model feature encodes volcanoes** (Table 1: volcano distance is diagnostic only), so any volcanic pattern in the output must come through terrain. | **E235** (pre-registered, `experiments/E235_elevation_vs_volcanic_control/DESIGN.md`), elevation band by band (M1–M3): (i) how much non-volcanic land exists at each elevation (above which height East Java has none); (ii) whether suitability differs between volcanic and non-volcanic cells at the same elevation; (iii) the same for the designs' disagreement (contingent fringe). Uses the stored E221 maps; no retraining. **In every branch:** the "volcanic-specific" phrase goes, and the n = 2 site density stops being evidence. Above the overlap height the text says elevation and volcanic setting cannot be separated in East Java. | ANALYSIS → text + figure |
| **R2-2** | Figure 1 shows no topography, so readers cannot judge how model response relates to elevation; this could mislead (same concern as R2-1). | Figure 1 (`figures/fig10_study_area_map.png`, `build_v02_figures.py::fig_study_area`) has sites and 13 volcanic centres only, on a blank background, with raw Web-Mercator metres (×10⁶/×10⁷) on the axes. It has no DEM, no coastline and no graticule. | **New Figure 1:** elevation/hillshade from the model's own DEM, coastline as the 0 m contour, latitude/longitude graticule, scale bar, sites, the 13 centres, the volcanic (≤ 20 km) and non-volcanic (≥ 40 km) zones used in the control, and an inset of Java. Built from data already in the repo; no new source. | FIGURE |

### Reviewer 1 (the bulk: clarity "fair")

| # | Point (paraphrase) | Verified fact | What changes | Type |
|---|---|---|---|---|
| **R1-1** | **The paper must be self-contained.** No mention of reviewers, requests, responses, revisions or the earlier version. The story of the rework can go in the acknowledgements. | 38 matches in `submission_jcaa_v0.2.tex` for reviewer/revision/withdrawn/v0.1-type wording, among them "in the reviewer's own terms" (line 499), "A reviewer asked whether" (491) and the abstract's "We withdraw …" (98). **The editor made the same point about the abstract on 31 Aug** (in his view it described the review process rather than the work); our replacement abstract was sent on 2 Oct (`EDITOR_REPLY_ABSTRACT_20261001.md`) but **is not yet in the .tex** (line 01 STATE, 10-02). | Rewrite the abstract, introduction, §3.9/§4.3 ("withdrawn", "we correct our own"), limitations and conclusion as a study in its own right. The corrections to our own diagnostics (volcano inventory, non-reproducing ρ) are stated as **findings about the method** with no history. One acknowledgements sentence (D6). | PROSE (G16) |
| **R1-2** | The research question should be stated more clearly at the start, and more pedagogically, for JCAA readers who are not statisticians. | §1 opens with the field; the question is spread over the "First … Second …" paragraph (line 161). | A plain-language question in the first paragraph: what an evaluation background is, why scoring a design on its own background inflates it, and what readers should do instead. | PROSE (G16) |
| **R1-3** | More explanation of the statistics for non-experts. | Terms used without a lay gloss include AUC, TSS, background/pseudo-absence, target-group background, Boyce index, Jaccard, seed ensemble and coarsened exact matching. | A short "Key terms" box or glossary table, plus one-sentence intuitions where each term first appears. | PROSE (G16) |
| **R1-4** | Three expressions confuse: one at reviewed lines 232–233, one at 264, one at 345. | They are our own sentences: "We sweep the hard-negative dial …" (tex 304), "How large is judging at home?" (tex 382) and "… map Jaccard said no." (tex 443). The same idiom family recurs (counts in the .tex): "dial" ×12, "ladder" ×22, "home-court" ×2, "rung" ×2. | Replace the whole idiom family with literal terms ("the share of hard negatives in the background", "inflation from scoring a design on its own background", "the pre-registered primary metric, map overlap (Jaccard), did not change"). Run a jargon sweep over the full text. | PROSE (G16) |
| **R1-5** | Using "§" for section references is unusual. | 23 `\S\ref{…}` + 4 hard-coded (`\S4` line 143, `\S3.2` line 148, `\S3` twice) in the text; the 9 literal "§" are in source comments only. *(Corrected 2026-10-20: the first count, "52 + 9", was a grep artefact.)* | "Section n" throughout. Mechanical. | MECHANICAL |
| **R1-6** | Bibliography: inconsistent formatting and name spellings; **the Comer et al. reference is wrong**. | **Crossref check of all 26 DOI-bearing cited entries (2026-10-20; `tools/check_bib_crossref.py` → `REFERENCE_CHECK_CROSSREF_20261020.json`): 3 are wrong, all cited together at tex line 157.** `comer2023`: the authors are invented (Crossref: J. A. Comer, D. C. Comer, Dumitru, Priebe, Patsolic), the volume is 47 instead of 48, and the title is truncated. `wang2023`: the authors are invented (Crossref: Wang, Shi, Oguchi), and the title is wrong (actual: "Archaeological Predictive Modeling Using Machine Learning and Statistical Methods for Japan and China"). `castiello2021` (a **JCAA** paper): the title is wrong (actual: "An Explorative Application of Random Forest Algorithm for Archaeological Predictive Modeling. A Swiss Case Study"). The other 23 match; 3 have no DOI (Kvamme 2006, Banks 2006, Franklin 2009) and get a manual check. Ledger **C076**. | Correct the three entries from Crossref. **Read the three papers** and check that the line-157 sentence ("strong gains from integrating spatial context and terrain logic while flagging data-quality and validation constraints") is true of each; cut or reword what is not. Normalise the whole list to the JCAA style (G15 template check), including the spelling of names. | FIX + SOURCE CHECK |
| **R1-7** | Figure 1 should show the study area but is a scatter plot with unintelligible axis values. | Same as R2-2. | Same new Figure 1. | FIGURE |
| **R1-8** | Figure 5's legend sits on the map; move it or give it a background. | `fig16_priority_map.png`: the legend overlaps the land at the lower right. | Legend outside the map frame, or an opaque box. Same graticule and coastline treatment as Figure 1 for consistency. | FIGURE |
| **R1-9** | General: the reworked analysis is impressive, but the text must be cleaned of the history. | — | Covered by R1-1. | — |

### Found on the way (not raised by the reviewers)

- **G15 is not met.** JCAA's fee page (fetched 2026-10-20, `journal.caa-international.org/about/submissions`) lists
  an APC of **651** for a research article; the currency sign was garbled in the fetch, and in April it was £593. A
  waiver counts only when the journal grants it **in writing**; otherwise the APC "may be invoiced post-acceptance".
  Our waiver request (6 Apr) was acknowledged, but no decision came. We re-raised it in the v0.2 response letter, and
  **this decision is silent on it.** The next decision can be an acceptance. Draft email:
  `docs/correspondence/EMAIL_JCAA_WAIVER_STATUS_DRAFT_20261020.md` (D4).
- **No `VENUE.md` for P2** (G15 asks for one). Needed for the template and reference-style check before upload.
- **Pending since 10-02:** port the replacement abstract into the .tex, and fix the "true performance" wording
  (claim C2). Both fold into R1-1 and R1-2.

---

## 3. Gates for this revision

- **G1:** blind re-derivation of every number on the final file. **G1-bis:** download what is cited as public and
  read it.
- **G11:** non-numeric claims (figure captions, statements about the text in the letter).
- **G12:** download the uploaded files back from the portal and compare.
- **G13:** P2 still counts as under review (K4).
- **G14:** rest period (D2). **G15:** fees (D4) and template (VENUE.md). **G16:** the PI's prose (D1). **AI
  disclosure:** updated for E235 and this round.
- **Letter rule (P8 lesson, memory `feedback-revision-focus`):** the response letter is traced statement by
  statement against the **final** manuscript, never against its own earlier draft. One independent adversarial
  reader, then three rounds (find → correct → verify) before upload.

---

## 4. Timeline (proposal; deadline 16 Nov)

| By | What | Who |
|---|---|---|
| 20 Oct | Decision recorded; triage; E235 designed; references checked; waiver email drafted | Claude ✅ |
| 23 Oct | PI decisions D1–D7; portal checked for an annotated copy; E235 design pushed (freezes it) | PI |
| 23–28 Oct | E235 run + verification; new Figures 1 and 5; reference corrections + reading the three papers; `VENUE.md`; fact sheet for the prose (every number with its source) | Claude |
| 26 Oct – 6 Nov | Prose: abstract, introduction, explanations, history removed (R1-1 to R1-4); E235 paragraph | PI (G16) |
| **≤ 2 Nov** for a strict G14 / **≤ 9 Nov** under D2 option (b) | Freeze | PI |
| freeze → 14 Nov | G1, adversarial read, letter traced against the final file, AI disclosure, template check | Claude |
| **16 Nov** | Upload + G12 | PI (login) |

If the freeze slips past **6 Nov**, ask the journal for a short extension **that same day**. The email asks to be told
early. The reason given must be true.

---

## 5. Decisions for the PI

| # | Decision | Recommendation |
|---|---|---|
| **D1** | **G16:** does the PI write the prose (rule), or is it waived as for P8 (C067)? | R1's complaint is about how the paper reads, so the PI writes at least the abstract, introduction and conclusion himself. Claude supplies the fact sheet, the jargon and history list with literal replacements, and the mechanical fixes (§ → Section, references). |
| **D2** | **G14:** (a) strict 14-day rest → freeze by 2 Nov; (b) as for P8, an independent adversarial read + G1 replaces the rest → freeze by 9 Nov; (c) ask for an extension now. | (b): the change is a cleanup plus one bounded control, and the deadline is fixed. Fall back to (c) only if the freeze slips past 6 Nov. |
| **D3** | **E235:** approve the design; push it (freezes the pre-registration); Claude runs it. | Yes. It is the only substantive point, the run is cheap (stored maps), and it is not a test of the burial thesis (K9 does not apply). |
| **D4** | **Waiver:** send the short email asking for the written waiver decision **now**, rather than waiting for the resubmission. | Yes, now. Zero cost is absolute, and acceptance could be the next decision. If the waiver is refused, decide (retarget or withdraw) **before** spending two weeks on prose. |
| **D5** | The three wrong references: keep them only if each paper supports the line-157 sentence after reading. | Delegated (Claude reads and reports); no decision needed unless a citation must go. |
| **D6** | Acknowledgements sentence about the rework (R1's suggestion). | One neutral sentence thanking the reviewers of both rounds, whose comments led to re-analysis and a reversed conclusion. No detail. |
| **D7** | Log in to the portal: download any annotated manuscript; confirm the deadline shown there. | Yes (PI login; Claude can drive the browser). |
| — | Tell Go Frendi Gunawan (co-author) about the decision. | One line from the PI. |

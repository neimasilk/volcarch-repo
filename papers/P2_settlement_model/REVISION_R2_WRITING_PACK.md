# P2 round-2 revision — writing pack for the PI (G16)

**Prepared:** 2026-10-20 by Claude, on the PI's "lakukan yg perlu dilakukan". **Purpose:** everything the PI needs
to write the revised prose himself: facts, numbers with sources, and the places to change.
- **No sentence here is meant to be pasted as final prose.** Suggested phrases are only suggestions.
- **Base file:** `submission_jcaa_v0.3.tex`, which is v0.2 plus the mechanical changes listed in §7 only. The PI
  writes there, or anywhere else, and Claude ports the text in.
- **Triage and decisions:** `REVISION_JCAA_R2_20261019.md`.

**Defaults taken under "lakukan yg perlu dilakukan" (2026-10-20):**
- **D1:** the PI writes at least the abstract, introduction and conclusion.
- **D2:** option (b). Freeze by **9 Nov**; an independent adversarial read + G1 replaces the 14-day rest; ask for an
  extension the same day if the freeze slips past 6 Nov.
- **D3:** E235 run.
- **D4:** waiver email sent 08:07.

---

## 1. The paper must read on its own (R1-1): every history passage, and the fact it carries

Line numbers are those of `submission_jcaa_v0.2.tex`. The left column is what the reader must still learn; the
history around it goes.

| Line(s) | Now says (gist) | Fact to keep | Suggestion |
|---|---|---|---|
| 95 (abstract) | a reviewer-requested benchmark refuted the attribution | Under a common evaluation background, the ladder's gain disappears (E217/E218). | The replacement abstract sent to the editor on 2 Oct (`EDITOR_REPLY_ABSTRACT_20261001.md`, ~240 words, limit 300) is history-free, and we promised to use it in the next version. **But it contains "ladder of AUC gains" and "the background-design dial we swept"**, the idioms R1 objects to. Keep it or revise it: agenda A11. |
| 98 (abstract) | "We withdraw … correct two defects in our own diagnostics" | Low suitability near volcanoes is not evidence of burial. The diagnostics use 13 volcanic centres; the correlation is stable across seeds only as a range. | Covered by the replacement abstract. |
| 131 | "Following a reviewer's taxonomy" | Three reasons a site can be absent: never settled, not found, not preserved (or buried). | Present it as the paper's own framework. |
| 143 | "the earlier version of this manuscript overreached" | A suitability score does not tell "never settled" from "buried". | Keep the restraint; drop the reason. |
| 147–149 | "A reviewer … asked us to benchmark against MaxEnt … we have replaced [the argument]" | The paper benchmarks every design against MaxEnt, the standard presence–background method, under the same folds. | Present it as a design choice of the study. |
| 152 | "because a reviewer … observed" | The problem family is established in ecology (Lobo 2008; Barve 2011; Warren & Seifert 2011 …). | Keep the literature paragraph; drop the attribution. |
| 164–166 | "The honest shape of this revision …" | — | Delete the paragraph. |
| 224 | MaxEnt "added at a reviewer's request" | — | Drop the clause. |
| 270, 273 | "earlier version's diagnostic suite"; "The submitted manuscript acknowledged this coupling once" | Road distance has four roles: it builds the TGB background and the hybrid pool, it defines the accessibility holdout, and it is a visibility proxy in the diagnostics. | State the four roles as a property of the design. |
| 280–282 | "We previously called this a 'temporal split'"; "see the Further Disclosures in the Response to Reviewers" | E014 is an accessibility-proxy holdout (train ≤ 1 km from a road, test > 1 km); AUC 0.755. | Name it once, correctly. If the mislabelled-file detail is kept, it goes in the supplement, which readers can see. |
| 370 | "the submitted version quoted these gaps against the best run (0.768)" | Gaps are computed against the seed average (0.751), because a single run is not a reportable quantity. | State the rule; drop the history. |
| 476–493 | subsection "Corrections to our own diagnostics" (inventory 7 → 13; ρ −0.163 does not reproduce; terrain-matched control) | 13 canonical centres in bounds. Spearman ρ(suitability, volcano distance) = −0.281 (13 centres); the legacy 7-centre run gives −0.243, not −0.163. **The terrain-matched paragraph (491–493) is replaced by E235 (§4).** | Retitle, e.g. "Volcanic diagnostics", as findings about the method. |
| 495–499 | "The taphonomic claim is withdrawn …"; "in the reviewer's own terms" | A suitability surface cannot separate the three mechanisms of absence. A quota design failed in both test worlds (World C: −0.2457 AUC_true, −0.4688 Jaccard, 0/30; World D: −0.2027 AUC_true, −0.2826 Jaccard, 0/30; line 500). | Retitle, e.g. "What a suitability surface cannot show"; keep the quota result with its numbers. |
| 523–536 | "With the site-prediction claim withdrawn"; "the revised paper takes seriously"; "this revision makes explicit"; "the constructive contribution of the revision" | The output is a suitability surface with declared limits, not a discovery map. The two conditions for field use. | Say it of the study, not of a revision. |
| 555 | "the site-prediction claim we have withdrawn" | Site-level prediction is outside what this design can support. | — |
| 566–570 | conclusion: "We withdraw …"; "the more useful finding the reviewer's request produced"; "internal revision drafts" | The protocol (fixed evaluation background, declared availability domain, seed ensembles) and why it matters. | Rewrite the conclusion around the protocol. "We applied it to our own earlier analyses" is enough history. |
| 591 | acknowledgements | D6 | One neutral sentence, e.g. thanks to the editors and the reviewers of both rounds, whose comments led to a re-analysis that changed the paper's conclusion. No detail. |
| 602 | "Both authors approved the withdrawal of the central claim reported in this revision" | — | "Both authors read and approved the final manuscript." |
| 617–620 | AI disclosure: "drafting and editing of this manuscript"; "corrections listed in the Response to Reviewers" | Must describe this version's AI use truthfully: under G16 the PI writes the prose, and AI did analysis, figures, verification and language checking. The blind re-derivation of every number stays. | The PI decides the wording; template `docs/AI_DISCLOSURE_TEMPLATE.md`. |

**On whether dropping the history is honest.** No P2 preprint was ever posted (EarthArXiv was planned in March; no
posting is recorded). The earlier claim exists only in the review record and in the public code repository, so
presenting the ladder as this study's own development sequence is accurate. Suggested framing: "the authors built
this sequence of designs and read it, at the time, as a series of improvements". That is history of the research,
which R1 does not object to, not history of the review.

## 2. Idioms (R1-4) — literal alternatives

Counts are from the v0.2 .tex. R1 named three; the rest belong to the same family.

| Idiom | Count | Literal alternative (suggestion) |
|---|---|---|
| "dial" ("sweep the hard-negative dial", "edge of the dial") | 12 | "the share of hard negatives in the background (0 to 1)"; "we vary this share from 0 to 1" |
| "judging at home", "home-court inflation" | 1 + 2 | "own-background inflation": the gain a design shows only when scored on the background it was trained with |
| "map Jaccard said no" | 1 | "the pre-registered primary measure, overlap between top-decile maps (Jaccard), did not improve" |
| "ladder", "rungs" | 22 + 2 | Keep "ladder" only if it is defined plainly at first use ("the sequence of seven designs, E007–E013, each reported as better than the last"); otherwise "the design sequence" |
| "no interior optimum" | several | Keep, but explain once: "the criterion keeps rising as the share grows, so its maximum is always at the end of the range and it never indicates where to stop" |

## 3. Key terms for non-statisticians (R1-2, R1-3) — technical content for a glossary or "key terms" box

The definitions follow the paper's own methods (E217 code). The PI words them.
- **Presence–background model:** fitted to known sites (presences) and to sampled locations whose status is unknown
  (background, or pseudo-absences). It estimates relative suitability, not probability of a site.
- **Background design:** the rule for drawing background points. *Random*: uniform over the study area. *Target-group
  (TGB)*: drawn where survey effort is likely, here weighted by road access (decay 12 km). *Hybrid*: TGB plus a share
  of "hard negatives", points that resemble presences in terrain (the paper's design uses 30 %).
- **Evaluation background:** the negatives used to score a model on held-out data. *Own-background*: drawn by the same
  design that trained the model. *Common*: one fixed set used for every design.
- **AUC:** the chance that a randomly chosen presence scores higher than a randomly chosen background point (0.5 =
  no better than chance, 1 = perfect). It depends on which background is used, which is the paper's point.
- **TSS:** sensitivity + specificity − 1 at a threshold.
- **Continuous Boyce index:** a presence-only measure. It compares predicted-to-expected presence frequency across
  suitability classes, against a fixed uniform availability sample, so the training background cannot inflate it.
- **Spatial block cross-validation:** folds are ~50 km blocks (0.45°), 5 folds, so test sites are not neighbours of
  training sites.
- **Seed ensemble:** the average of maps from several random seeds. In the paper, ≥ 7 seeds stabilise the top-decile
  map (Jaccard ≥ 0.9 against a 10-seed ensemble).
- **Jaccard overlap:** shared cells ÷ cells in either of two top-decile maps.
- **Coarsened exact matching:** compare two groups only within strata (bins) where both are present, weighted by the
  smaller group. Used in E235 within elevation bands.
- **Availability domain:** the area from which background or evaluation points are drawn. A metric is defined only
  relative to it (Barve et al. 2011).

## 4. The new control for Reviewer 2 (E235) — facts and numbers

Source: `experiments/E235_elevation_vs_volcanic_control/README.md`. Pre-registered (commit `3ac41cc`, pushed before
the run). Independently re-derived.

- **Question:** does the model respond to volcanic setting, or to elevation? No model feature carries volcanic
  information, so any volcanic pattern must come through terrain.
- **Design:**
  - volcanic = within 20 km of any of the 30 canonical centres; non-volcanic = 40 km or more;
  - 200 m elevation bands; matching on slope, TRI and TWI within each band;
  - spatial block bootstrap;
  - thresholds fixed in advance (0.05 suitability; 5 percentage points).
- **Overlap (the main answer):** above **1,000 m**, East Java has 3,435 km² of land.
  - **97.5 %** is within 20 km of a volcanic centre; **0.4 %** is 40 km or more away.
  - Non-volcanic land: 15 km² at 1,000–1,200 m and none above 1,200 m.
  - So high-elevation non-volcanic terrain does not exist in the study area: **above 1,000 m elevation and volcanic
    setting cannot be separated**, and the paper makes no claim either way there.
- **Below 1,000 m:**
  - **Suitability:** volcanic cells score +0.049 (95 % CI +0.024 to +0.077; XGBoost, random design) above
    terrain-matched non-volcanic cells. That is on the pre-declared threshold, so **inconclusive**. The difference
    vanishes when the volcanic zone is drawn tighter (≤ 10 km vs ≥ 30 km: −0.002, CI −0.047 to +0.046).
  - **Design disagreement does not depend on volcanic setting:** contingent-fringe share +0.56 percentage points (CI
    +0.26 to +0.85) against a 5-point threshold, for all three algorithms and all three zone definitions.
- **Above 1,000 m (descriptive):** the designs diverge. The hybrid design stays near 0.5 suitability while random and
  TGB fall to about 0.1, and the contingent fringe rises to ~25 %. This is Figure 5's "remote upland" fringe; in this
  study area it cannot be assigned to elevation rather than volcanism, or the reverse.
- **Pre-registered consequences** (all apply):
  - remove "directionally consistent with a volcanic-specific effect";
  - the n = 2 site-density comparison is no longer evidence (supplement or cut);
  - state the 1,000 m overlap ceiling and the 97.5 % / 0.4 % split;
  - report the suitability difference as inconclusive, with numbers;
  - state the disagreement result for the range where it holds.
- **Candidate figure:** `experiments/E235_elevation_vs_volcanic_control/results/e235_response_by_band.png`. Panels:
  land per band; suitability by band; fringe by band. It shows the reviewer exactly where the comparison is and is
  not possible. **Decision for the PI:** a manuscript figure, or a supplementary figure.
- **Method sentence material:** the comparison uses the stored 10-seed ensemble maps (E221). No model was retrained.

## 5. Figures (R1-7, R1-8, R2-2)

Built by `build_v03_figures.py`; the v0.2 figures are kept as submitted.
- **Figure 1** → `figures/fig01_study_area_v03.png`. Contents:
  - elevation (hillshaded Copernicus GLO-30, the model's own DEM) with a scale in metres;
  - coastline traced from the DEM; latitude/longitude graticule; 50 km scale bar; north arrow;
  - the archaeological presences; the 13 volcanic centres inside 111–115°E, labelled;
  - the 20 km and 40 km distance lines used by E235;
  - a locator inset (Natural Earth).

  Caption facts: what the 20/40 km lines are, and where the DEM and inset come from. As in v0.2, no count of sites
  (383 in the file, 378 modelled; the caption should not invite the mismatch).
- **Figure 5** → `figures/fig05_priority_map_v03.png`. The same content as v0.2 (XGBoost, 10-seed ensemble, top
  decile under 3/2/1/0 designs), drawn as a raster with coastline, graticule and scale bar; legend below the map.
  Caption unchanged in substance.

## 5a. Alt text — JCAA requires it for every figure

Rules (`VENUE.md` A8): about 10 words or fewer; different from the caption; placed in the text document with the
figure titles. **Candidates for the PI to approve or change:**

| Figure | File | Alt-text candidate (word count) |
|---|---|---|
| 1 | fig01_study_area_v03.png | Relief map of East Java with sites and volcanoes (9) |
| 2 | fig3_auc_tss_progression.png | Line chart of AUC and TSS across seven model versions (10) |
| 3 | fig14_artefact_two_panel.png | Two panels comparing own-background and common-background model scores (8) |
| 4 | fig15_dose_response.png | Reported versus true performance as hard negatives increase (8) |
| 5 | fig05_priority_map_v03.png | Map of East Java coloured by agreement between designs (9) |
| 6 | fig17_seed_stabilisation.png | Map agreement rising with the number of seeds averaged (9) |
| new (E235), if used | fig_e235_elevation_control_v03.png | Land area by elevation band and matched suitability differences (9) |

**Captions also carry the idioms of §2:** Figure 3 "judged at home"; Figure 4 "edge of the dial". Rewrite them with the
text.

**E235 material in final form, if the PI wants it:**
- figure `fig_e235_elevation_control_v03.png` (built by `build_v03_figures.py`). Panel A: land per band. Panel B: the
  matched differences with intervals and the ±0.05 threshold.
- Tables S7 (per band) and S8 (pooled, by zone definition and algorithm) in `supplement/e235_tables_v03.tex`
  (`build_e235_tables.py`; the captions are stubs marked for the PI).

## 5b. A factual error in the reviewed text, found while checking Figure 5 (ledger C078) — must be fixed

v0.2 lines 458–460 (Section 3.7) say the robust core is "cells selected by every seed", and qualify the site-density
ratios with "the non-volcanic comparator arm contains only n = 2 known sites". Both are wrong:
- **Definition** (E221 README line 20; Figure 5's own caption and legend): the robust core is the cells in the **top
  decile under all three background designs**, each map being a 10-seed ensemble. The contingent fringe is the cells
  in the top decile under exactly one design.
- **Ratios:** 1.93× / 4.34× / 5.62× (RF / XGBoost / MaxEnt) are correct: 30.7/15.9, 40.8/9.4 and 31.7/5.65 sites
  per 1,000 km² (`e221_outcome.json`).
- **The right caveat** (E221 README line 40): this is a consistency check, not validation, **because the known sites
  trained the models**. The n = 2 sentence belongs to the terrain-matched control, which E235 replaces.
- **Link to E235:** Figure 5's fringe statistic "1,107 m elevation in the MaxEnt fringe" lies above the 1,000 m
  overlap ceiling. At that height the fringe cannot be assigned to elevation rather than volcanic setting.

## 6. References (R1-6)

Ledger C076. **Done 2026-10-20:** `references.bib` corrected (castiello2021, wang2023, comer2023, senay2013 + two
DOIs); record and checks in `REFERENCE_FIX_20261020.md`.
- **The PI must rewrite the line-157 sentence.** None of the three papers supports "strong gains": AUC 0.71/0.72
  called "fair"; +0.03 over MaxEnt; a validation-design paper.
- The facts and two suggested shapes are in `REFERENCE_FIX_20261020.md` §2.
- Still to do before upload: the Harvard-style pass (§3 there).

## 7. Mechanical changes already applied in `submission_jcaa_v0.3.tex` (not prose)

1. "§" → "Section": the 23 `\S\ref{…}` → `Section~\ref{…}`. Four were hard-coded numbers: `\S3.2` → `Section~\ref{sec:artefact}`;
   `\S3` ×2 and `\S4` → literal "Section~3" / "Section~4", because those sections have no `\label` (R1-5). **Check these
   three after any restructuring.**
2. Figures 1 and 5 → the v03 files.
3. `references.bib` → the corrected entries (done 2026-10-20, `fix_references_20261020.py`; the .bib is shared with v0.2,
   whose submitted PDF stays as the record).
4. Header comments that describe the revision process (lines 1–37 and the placeholder macro): cleaned. They are not
   rendered, but the source may travel.
5. **Reference list in JCAA Harvard form** (2026-10-20): `build_jcaa_bibliography.py` writes `references_jcaa.tex` from
   `references.bib`, cited keys only. In the text: "(Castiello & Tonini 2021)", "(Comer et al. 2023)", several
   citations sorted alphabetically and separated by ";". In the list: "Surname, Initials. Year. Title. *Journal*,
   vol(issue): pages. DOI: …"; article numbers print as "Article n". **Re-run it after any change to the citations**
   (`python build_jcaa_bibliography.py submission_jcaa_v0.3.tex --apply`). Do not edit `references_jcaa.tex` by hand.
6. Figures 1 and 5 at full text width.

`make_v03_mechanical.py` now refuses to overwrite an existing v0.3, so the PI's edits are safe.

Nothing else in the text is changed by Claude.

# P2 / JCAA #280 — Response to Reviewers (round 2): skeleton

**Status:** SKELETON, 2026-10-20 (Claude). **The PI writes every reply.**
- This file holds only, per point, what was done and the evidence. "Where in v0.3" is filled **after the freeze**,
  from the final file.
- Reviewer text is **paraphrased** here (public repo). The PI pastes the original comments into the final letter
  outside the repo.
- JCAA expects a point-by-point response as a separate document, uploaded as the component "Response to Reviewers"
  (`VENUE.md` A11).
- **Rule (P8 lesson):** the final letter is traced statement by statement against the final manuscript, never
  against this skeleton.

---

## Opening (PI)
Material:
- thanks to both reviewers;
- the revision makes the paper self-contained (no history of the review in the text);
- a new control answers the elevation question (Reviewer 2);
- figures redrawn; references corrected.

**Cover letter / Comments to the Editor at upload: one sentence on the waiver IS needed** (changed 2026-10-21). The
journal's policy wants waiver requests with the submission, and the editor's answer of 20 Oct says to apply once the
paper is accepted. Content: following the editorial team's note of 20 October, the authors will apply for a publisher
waiver on acceptance. The PI words it. Not in the response to the reviewers.

## Reviewer 1

| # | Point (paraphrase) | Status · facts | Where in v0.3 (fill after freeze) |
|---|---|---|---|
| R1-1 | The text should stand alone: no mention of reviewers, revisions or the earlier version; the story of the rework could go in the acknowledgements | **PI prose (planned):** about 39 history passages removed or recast as findings about the method (writing pack §1); the ladder is presented as the study's own development sequence; one acknowledgements sentence (D6) | … |
| R1-2 | State the research question earlier, more clearly and more pedagogically | **PI prose (planned):** plain question in the first paragraph of the introduction | … |
| R1-3 | More explanation of the statistics for non-specialists | **PI prose (planned):** key-terms box (pack §3): background design, evaluation background, AUC, TSS, Boyce index, spatial block CV, seed ensemble, Jaccard, matching | … |
| R1-4 | Three expressions are confusing (reviewed lines 232–233, 264, 345) | **PI prose (planned):** all three replaced, plus the whole idiom family ("dial" ×12, "home-court", "judging at home", "said no"); "ladder" defined at first use or replaced (pack §2) | … |
| R1-5 | The section sign is unusual | **Done:** all 27 section references now read "Section n" (23 cross-references, 4 hard-coded) | whole text |
| R1-6 | Bibliography: formatting, name spellings; Comer et al. reference wrong | **Done** (except the sentence rewrite, PI): every cited DOI checked against Crossref. Three entries had wrong metadata and are corrected (Comer et al. 2023: authors and volume; Wang, Shi & Oguchi 2023: authors and title; Castiello & Tonini 2021: title). Senay's given name corrected; DOIs added. The list is now in JCAA's Harvard format. **The sentence citing these three was itself not supported by them and has been rewritten** (`REFERENCE_FIX_20261020.md`) | reference list; Section 1 (line-157 sentence) |
| R1-7 | Figure 1 is a scatter plot, not a study-area map; its axis values are unintelligible | **Done:** new Figure 1: relief map from the model's DEM, coastline, latitude/longitude grid, scale bar, north arrow, locator inset; sites, the 13 volcanic centres and the 20/40 km zones of the new control | Figure 1 |
| R1-8 | Figure 5's legend should not sit on the map | **Done:** legend moved below the map; the map is now drawn as a raster with coastline, grid and scale bar | Figure 5 |
| R1-9 | General: clean the text of references to the previous version | As R1-1 | — |

## Reviewer 2

| # | Point (paraphrase) | Status · facts | Where in v0.3 (fill after freeze) |
|---|---|---|---|
| R2-1 | Still unclear whether the model captures volcanic areas or high elevation; the round-1 answer may have been mostly a matter of presentation; comparing with non-volcanic uplands at the same height would settle it | **Analysis done; text = PI prose (planned).** New analysis (E235, pre-registered before it was run). 200 m elevation bands; near = within 20 km of a Quaternary volcanic centre, far = 40 km or more; matched within band on slope, TRI and TWI; spatial-block bootstrap. **(1)** Above about 1,000 m, 97.5 % of the land is near a Quaternary centre and 0.4 % far (15 km² at 1,000–1,200 m, none above), so a comparison with high-elevation land away from volcanoes cannot be made in this study area. The paper now says so and claims neither. **(2)** Below 1,000 m, suitability is about 0.05 higher near Quaternary centres than in matched far terrain (pre-set verdict: inconclusive). Robustness checks give +0.01 to +0.07 depending on zone and matching choices, so no direction is claimed. The far uplands are mostly the Southern Mountains, an older (Eocene–Miocene) volcanic arc, which the text states. **(3)** No claim is made about the designs' disagreement and volcanic setting. The phrase "consistent with a volcanic-specific effect" and the two-site density comparison are removed. No model input carries volcanic information. **Answer the reviewer plainly: the separation they ask for is not available in East Java above about 1,000 m, and the paper no longer implies that the model's response is volcano-specific.** | new subsection; new figure (or supplement); Tables S7–S8 |
| R2-2 | Without topography, Figure 1 hides how the model relates to elevation | **Done:** new Figure 1 shows elevation (hillshaded DEM) and the zones used in the control | Figure 1 |

## Changes not requested by the reviewers (state them; brief)

- **Section 3.7:** the robust core was defined as "cells selected by every seed"; it is the cells in the top decile
  under all three background designs (ledger C078). The caveat now gives the right reason: the known sites trained
  the models.
- **AI-use statement:** updated to this version's use (JCAA policy).
- Word count; abstract (the replacement already sent to the editor on 2 October).

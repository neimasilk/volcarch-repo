# P2 references — what was wrong, what was fixed, what the PI must decide (ledger C076; JCAA round 2, R1-6)

**2026-10-20.** Sources:
- the orchestrator's Crossref check of all 26 DOI-bearing cited entries (`tools/check_bib_crossref.py` →
  `REFERENCE_CHECK_CROSSREF_20261020.json`);
- a Sonnet sub-agent's content check (`REFERENCE_CONTENT_CHECK_20261020.md`, unchanged);
- the orchestrator's re-verification of that report.

## 1. Fixed in `references.bib` (`fix_references_20261020.py`; re-checked against Crossref afterwards: OK)

| Key | Was | Now (Crossref, checked by the orchestrator 2026-10-20) |
|---|---|---|
| castiello2021 | "Castiello, Marco"; title "Machine learning applications in archaeological predictive modelling" | Castiello, **Maria Elena** & Tonini, Marj (2021). *An explorative application of Random Forest algorithm for archaeological predictive modeling. A Swiss case study.* JCAA 4(1): 110–125. 10.5334/jcaa.71 |
| wang2023 | seven invented authors (Cheng Wang, Ying Li …); invented title | **Wang, Yuan; Shi, Xiaodan; Oguchi, Takashi** (2023). *Archaeological predictive modeling using machine learning and statistical methods for Japan and China.* IJGI 12(6): 238. 10.3390/ijgi12060238 |
| comer2023 | authors Comer, Harrower, McCorriston, O'Meara, Sadr; vol. 47; short title | **Comer, Jacob A.; Comer, Douglas C.; Dumitru, Ioana A.; Priebe, Carey E.; Patsolic, Jesse L.** (2023). *Sampling methods for archaeological predictive modeling: Spatial autocorrelation and model performance.* JAS: Reports **48**: 103824 |
| senay2013 | "Senay, Sabrina D." | Senay, **Senait** D. (Crossref 10.1371/journal.pone.0071218) |
| kvamme2006 | no DOI | + 10.1201/9780203563359.sec1. Year kept at 2006 (print): Crossref dates the online edition 2005. |
| franklin2009 | no DOI | + 10.1017/CBO9780511810602. Year kept at 2009 (print, confirmed by two reviews): Crossref dates the online edition 2010. |

`submission_jcaa_v0.3.tex` recompiles with no errors or undefined references.

## 2. The sentence these three support does not hold as written — **the PI rewrites it** (v0.2 line 157)

> "... recent machine-learning work in archaeology reports strong gains from integrating spatial context and terrain
> logic while flagging data-quality and validation constraints \citep{castiello2021, wang2023, comer2023}."

What the papers say:
- **castiello2021** — verified by the orchestrator in the JCAA full text. AUC 0.71 with all variables, 0.72 with the
  six most important, called "acceptable discrimination" and "fair performance" (227 cases). No "strong gains".
  Data scarcity is stated.
- **wang2023** — verified in the MDPI full text, Table 5. The authors' AM_FR model "performed slightly better than the
  MaxEnt and FR". Mean test AUC: Japan 0.901 / 0.872 / 0.856; China 0.816 / 0.785 / 0.791. That is about +0.03 over
  MaxEnt. MaxEnt is better on Kvamme's Gain (0.92 / 0.89). Topographic and hydrological factors are used. No "strong
  gains".
- **comer2023** — the orchestrator verified title and metadata only: the publisher page needs JavaScript, and
  OpenAlex and Semantic Scholar hold no abstract. The sub-agent read the abstract on the ScienceDirect page and the
  accepted manuscript. Its reading: random-cell sampling gives higher AUC than a leave-one-site-out scheme, attributed
  to spatial autocorrelation, so the sampling and validation design can over- or understate performance. **That
  narrow claim already follows from the paper's own title**; anything stronger needs the PI or an independent reader
  to see the text.

**Suggestions** (the PI writes the final wording; JCAA forbids AI ghost-writing):
- *Sub-agent's version, all three kept:* recent applied work predicts site locations from terrain and hydrological
  variables with moderate-to-good discrimination (AUC roughly 0.7–0.9) while noting limits of data and
  transferability (Castiello & Tonini 2021; Wang, Shi & Oguchi 2023). The sampling and validation design itself can
  over- or understate measured performance (Comer et al. 2023).
- *Shorter:* drop wang2023 and keep the two claims above.
- **Note:** Comer et al. 2023 is directly relevant to P2's own argument (evaluation design changes the reported
  number). The PI may want to cite it where the paper discusses evaluation, not only in this literature sentence.

## 3. Left for the Harvard-style pass before upload (`VENUE.md` A10)

- The list must follow JCAA's Harvard format:
  - in text "(Castiello & Tonini 2021)", "(Wang, Shi & Oguchi 2023)", "(Comer et al. 2023)";
  - in the list, surname then initials, DOI on every entry where one exists.
- The current natbib output differs ("and", first names in full).
- Minor given-name forms (low severity; Harvard prints initials anyway): elith2006 Overton = "J. McC. M."; barbet2012
  Albert = "C. H.".
- Capitalisation after a colon is inconsistent (chen2016, valavi2019, guillera2015, wisz2009, kvamme2006).
- Article numbers print as bare page numbers (comer2023, wang2023, wisz2009, senay2013, yaworsky2020).
- **The lesson (C076):** the August Crossref check covered only the six references added then. These three came
  from v0.1 (March) and were never checked. Every cited entry is checked before every upload.

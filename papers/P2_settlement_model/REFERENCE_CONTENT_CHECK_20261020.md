<!-- Sub-agent report (Sonnet), 2026-10-20, copied unchanged from the session scratchpad. Downloads stay outside the repo.
The orchestrator's checks of this report are in REFERENCE_FIX_20261020.md; read that file first. -->

# Reference content check for P2 (submission_jcaa_v0.2), 2026-10-20

Scope: the three entries cited together at line 157 of `submission_jcaa_v0.2.tex` (castiello2021, wang2023, comer2023); the three entries without a DOI in the .bib (kvamme2006, banks2006, franklin2009); and a name/format audit of every entry in `submission_jcaa_v0.2.bbl` against Crossref and publisher pages. No repo file was edited (git status clean at the end).

Sentence under test (line 157):
"... and recent machine-learning work in archaeology reports strong gains from integrating spatial context and terrain logic while flagging data-quality and validation constraints \citep{castiello2021, wang2023, comer2023}."

## 0. Provenance and limits (what was actually read)

| Paper | What I read | Verbatim-quotable? |
|---|---|---|
| Castiello and Tonini 2021 (JCAA, open access) | Full text, HTML from journal.caa-international.org, downloaded and text-extracted (`dl_jcaa/jcaa71.html`, `dl_jcaa/jcaa71.txt`) | Yes, full text |
| Wang, Shi and Oguchi 2023 (IJGI, open access) | Full-text PDF (27 pp.) from mdpi-res.com (`dl_mdpi/res.pdf`); mdpi.com itself returned 403. Quotes in section 2 below were checked by viewing the rendered pages (p. 16 Table 5, pp. 21 and 23 Discussion/Conclusions; images in `img/`) because the PDF text layer is doubled in places | Yes, full text |
| Comer et al. 2023 (JASREP) | The version of record is paywalled ("Purchase PDF"). I read (a) the abstract and highlights on the ScienceDirect /abs/ page, and (b) the full text of the Elsevier **accepted manuscript** (22 pp., "This manuscript is made available under the Elsevier user license"), opened through ScienceDirect's /am/ link and text-extracted in the browser with pdf.js. Line numbers below are the manuscript's own line numbers. The accepted manuscript was not saved to disk (the signed URL could not be re-fetched outside the browser); the quotes below are the only record. The accepted manuscript may differ in wording from the version of record; the abstract and highlights quoted are from the version-of-record page. | Yes, accepted manuscript + VoR abstract |

Crossref records for all 26 DOIs are in `cr_all/`; the three focus papers also in `cr_10_*.json`. The comparison script output is `compare_bbl_output.txt`.

---

## 1. Task 1: the three papers

### 1.1 Castiello and Tonini 2021 (castiello2021)

**(a) Corrected record** (Crossref, and confirmed on the JCAA page: "Maria Elena Castiello, Marj Tonini"; "Submitted on Jan 13, 2021 / Accepted on Apr 7, 2021 / Published on May 21, 2021"; Volume 4, Issue 1, Page/Article 110-125; CC BY 4.0)

```bibtex
@article{castiello2021,
  author  = {Castiello, Maria Elena and Tonini, Marj},
  title   = {An explorative application of {R}andom {F}orest algorithm for archaeological predictive modeling. {A} {S}wiss case study},
  journal = {Journal of Computer Applications in Archaeology},
  year    = {2021},
  volume  = {4},
  number  = {1},
  pages   = {110--125},
  doi     = {10.5334/jcaa.71}
}
```
Published title (title case, as on the journal page and in Crossref): "An Explorative Application of Random Forest Algorithm for Archaeological Predictive Modeling. A Swiss Case Study". The braces above keep the repo's sentence-case list style. Errors in the current .bib: title (wrong, an invented one), and first author given name "Marco" should be "Maria Elena". Year, volume, issue, pages, DOI are right.

**(b) What it does.** Random Forest predictive model for Roman settlements in the Canton of Zurich: 227 presences plus an equal number of random pseudo-absences; predictors are DEM and derivatives (slope, northness, eastness), distance to water, soil-map properties and a geotechnical map; 5-fold spatial cross-validation (blockCV); AUC 0.71 with all variables and 0.72 with the six most important. The model is compared only with a reduced version of itself, not with another method.

**(c) The three parts of the sentence**

| Part | Verdict | Deciding quote (verbatim) |
|---|---|---|
| "strong gains" | **NOT SUPPORTED** | Section 3 (Results and Discussion): "the AUC using all the variables is equal to 0.71 (blue line), while for the second model (i.e. considering only the six most important variables) it is slightly higher and equal to 0.72 (red line), which is considered as an acceptable discrimination according to the criteria of Hosmer and Lemeshow (2000)." Same section: "The scarcity of observations can explain the model's fair performance, as evaluated with an AUC of 0.72." Section 2.3: "Two models were compared in this study: the first, including all the geo-environmental features, and a second one considering only the first six most important features" (so no baseline, no gain over anything). The authors do assert advantages in words: "the machine learning modeling procedure revealed significant advantages with regard to the state of the art in archaeological predictive modeling studies. It represents an alternative to more classical statistical approaches." (Section 3), but no number backs a gain over a classical method. |
| "from integrating spatial context and terrain logic" | **PARTLY** | Terrain/environment as predictors: Section 1: "The predisposing factors suggesting the presence of Roman sites in the area are the environmental features. They are described by: (i) topographic indices derived from the digital elevation model (DEM); (ii) different characteristics related to the soil and its aptitude to agricultural activities; (iii) strategic and water-related criteria on which past populations may have based their site location choice." Section 3: "The six most important variables (i.e., DEM, slope, water saturation capacity, geotechnical map, distance to lakes, and soil skeleton)". But there is no test of what terrain or spatial context adds ("variables low in the ranking do not add any supplementary predictive power to the model" only prunes variables), and spatial context enters only the validation (spatial k-fold), not the model. |
| "flagging data-quality and validation constraints" | **SUPPORTED** | Section 4 (Conclusions): "However the quality and quantity of the data used are an important constraint for such such kind of applications, as it is often the case when dealing with archaeological information and modern environmental variables." Section 2.2.1: "However, a certain underlying degree of uncertainty exists at this stage, as not all entries had exact coordinates." Section 2.3.1: "observations belonging to the testing dataset are most likely located close to the training ones. This leads to overestimating the predictive performance of the model." Section 3: "Urbanized areas are more intensively excavated as most archaeological surveys are rescue operations, which happen prior to any construction activity." |

### 1.2 Wang, Shi and Oguchi 2023 (wang2023)

**(a) Corrected record** (Crossref; MDPI PDF header: "Yuan Wang 1,*, Xiaodan Shi 1,2 and Takashi Oguchi 1,3"; "ISPRS Int. J. Geo-Inf. 2023, 12, 238"; Received 21 April 2023, Published 7 June 2023; CC BY)

```bibtex
@article{wang2023,
  author  = {Wang, Yuan and Shi, Xiaodan and Oguchi, Takashi},
  title   = {Archaeological predictive modeling using machine learning and statistical methods for {J}apan and {C}hina},
  journal = {ISPRS International Journal of Geo-Information},
  year    = {2023},
  volume  = {12},
  number  = {6},
  pages   = {238},
  doi     = {10.3390/ijgi12060238}
}
```
Published title (title case): "Archaeological Predictive Modeling Using Machine Learning and Statistical Methods for Japan and China". The current .bib has the wrong title and seven invented authors (Cheng Wang, Ying Li, Yijian Liu, Huan Liu, Jian Liu, Yijian Su, Huadong Guo); the real paper has three authors. Year, volume, issue, article number, DOI are right.

**(b) What it does.** Predictive modelling of Kofun-period keyhole tombs in Japan (n = 1367) and of sites in Shaanxi, China (n = 200) from eight topographic and hydrological factors. It proposes a hybrid deep-learning attention mechanism plus frequency ratio (AM_FR) and compares it with a plain FR model and MaxEnt using four-fold cross-validation, AUC and Kvamme's Gain. Random absences equal in number to the sites, drawn within 10 km of the sites.

**(c) The three parts of the sentence**

| Part | Verdict | Deciding quote (verbatim) |
|---|---|---|
| "strong gains" | **NOT SUPPORTED** | Section 5 (Discussion, p. 21): "For both areas, the AM_FR performed slightly better than the MaxEnt and FR based on the values of AUC (Table 5)." Table 5 (p. 16), mean test AUC: Japan AM_FR 0.901, MaxEnt 0.872, FR 0.856; Shaanxi AM_FR 0.816, MaxEnt 0.785, FR 0.791 (about +0.03 over MaxEnt). Section 4.5: "MaxEnt had a higher efficiency in identifying such regions, with a Kvamme's Gain value of 0.92 for Japan and 0.89 for China. In comparison, the AM_FR and FR models had lower efficiency, with values of 0.78 and 0.72, respectively, for Japan and 0.84 and 0.83 for China." The abstract itself says only "the AM_FR model exhibited the most satisfactory performances." |
| "from integrating spatial context and terrain logic" | **PARTLY** (terrain yes, spatial context no) | Section 1 (p. 2): "Eight predictive input factors were selected for this study. These included elevation, slope, and distance from a river, which are commonly used, as well as several topographical factors rarely used in APM: roughness, relief degree of the land surface (RDLS), plan curvature, profile curvature, and cutting depth." Section 5 (p. 21): "topographic deviations from the horizontal flat surfaces play an essential role in deciding archaeological site locations." The models work on per-cell topographic and hydrological factors; the attention mechanism weights input factors, not spatial neighbourhoods, and the paper has no spatial-context component. |
| "flagging data-quality and validation constraints" | **PARTLY** (scope and transfer limits yes; data quality no) | Section 6 (p. 24): "Admittedly, our models have proven successful in identifying the sites under the current study, but their effectiveness in identifying other site types, such as villages or hunting camps, is yet to be fully determined." Section 5 (p. 23): "This example indicates that the results of archaeological predictive modeling should be evaluated carefully when marginal areas surrounding the core area are included in the constructed predictive maps." Section 5 (p. 21): "Nevertheless, this comes at the cost of many sites being classified as moderate and low probability areas, meaning that some true archaeological sites may be missed in practice." What is not there: a search of the full text finds no occurrence of "quality", "bias", "uncertain", "sample size", "overfit" or "robust"; validation is plain four-fold cross-validation with no discussion of spatial autocorrelation. Section 2.2 only describes the data ("these points are mainly located in the centers of the archaeological sites"; "The same amount of absence data was randomly generated under a restriction of 10 km outward from the archaeological sites"); the Data Availability Statement says "Archaeological data are not publicly available due to privacy and ethical reasons." |

### 1.3 Comer, Comer, Dumitru, Priebe and Patsolic 2023 (comer2023)

**(a) Corrected record** (Crossref; ScienceDirect page: "Jacob A. Comer, Douglas C. Comer, Ioana A. Dumitru, Carey E. Priebe, Jesse L. Patsolic"; "Journal of Archaeological Science: Reports, Volume 48, 2023, Article 103824"; online 19 Feb 2023, issue date April 2023; no issue number)

```bibtex
@article{comer2023,
  author  = {Comer, Jacob A. and Comer, Douglas C. and Dumitru, Ioana A. and Priebe, Carey E. and Patsolic, Jesse L.},
  title   = {Sampling methods for archaeological predictive modeling: {S}patial autocorrelation and model performance},
  journal = {Journal of Archaeological Science: Reports},
  year    = {2023},
  volume  = {48},
  pages   = {103824},
  doi     = {10.1016/j.jasrep.2022.103824}
}
```
Errors in the current .bib: all five authors are wrong (Harrower, McCorriston, O'Meara, Sadr are not authors), the subtitle is missing, and the volume is 47 instead of 48. Year and DOI are right; 103824 is an article number, not a page.

**(b) What it does.** A methods paper on one small case: a "direct detection model" at Fort Irwin, California (a 91 km2 area with 12 recorded lithic-scatter sites), built with logistic regression and LDA on five variables (slope, elevation, local relief, cost-distance to water, first principal component of WorldView-2 reflectance). It compares two sampling schemes, random cell sampling and a site-wise leave-one-out scheme, over 200 runs each. Mean AUC is about 0.59 under the site-wise scheme and 0.84-0.85 under random cell sampling, and the authors attribute the gap to spatial autocorrelation. It is a warning about inflated validation scores, not a report of gains.

**(c) The three parts of the sentence** (abstract and highlights from the VoR page; the rest from the accepted manuscript, with its line numbers)

| Part | Verdict | Deciding quote (verbatim) |
|---|---|---|
| "strong gains" | **NOT SUPPORTED** | Abstract: "We find that models trained and tested with a random cell sampling strategy perform better than do models that implement a kind of leave-one-out cross-validation (LOOCV) ... We argue that this difference in measured predictive performance is due to spatial autocorrelation". Highlight: "Sampling method choice can over- or understate model performance on intended task." Lines 286-288: "Because this paper is primarily concerned with methods, we have chosen to build relatively simple models for clarity's sake rather than more complicated models for better predictive performance." Figure 5 caption: "Vertical lines indicate mean AUCS; from left to right, they are 0.586, 0.592, 0.836, and 0.85." The higher scores are the artefact the paper warns about. |
| "from integrating spatial context and terrain logic" | **NOT SUPPORTED** | The classifier uses terrain derivatives, but the paper does not test or claim any benefit from them (lines 322-325: "a function of its distance from the nearest potential source of water w, its elevation e, the local relief r, its slope s, and the score pc of the first principal component on the eight top-of-atmosphere reflectance measurements"). Lines 36-41: "Our objective is not to offer a novel sampling method, nor is it to suggest that one or another sampling method is generally superior." Spatial context appears only as spatial autocorrelation in the validation. |
| "flagging data-quality and validation constraints" | **SUPPORTED** | Abstract: "Different sampling methods can also yield calculations of predictive ability that over- or understate a model's performance at the task for which it was designed." Lines 367-369: "Though none of the modeling described here involves truly independent data, the LOOCV sampling scheme more closely approximates the task of running a DDM on an independent data set". Data quality, lines 155-157: "reliable dating of many of Fort Irwin's sites is complicated by a relative scarcity of data and by geological and anthropogenic disturbances"; lines 167-171: "the annual, 10,000-acre surveys at Fort Irwin tend to target areas where sites are deemed especially likely to occur, so the areas not yet surveyed might plausibly be less occupied by sites than are the areas surveyed to date). We therefore assume that the areas not yet surveyed are non-site areas." |

### 1.4 Does the sentence hold?

| Paper | strong gains | spatial context + terrain logic | data-quality and validation constraints | Sentence as written |
|---|---|---|---|---|
| castiello2021 | NOT SUPPORTED | PARTLY | SUPPORTED | holds only for the last part |
| wang2023 | NOT SUPPORTED | PARTLY | PARTLY | holds for terrain and (partly) limits |
| comer2023 | NOT SUPPORTED | NOT SUPPORTED | SUPPORTED | holds only for the last part |

No paper supports "strong gains": the best case is a +0.03 mean-AUC edge for the authors' own new method over MaxEnt (Wang), AUC 0.72 called "fair" (Castiello), and an inflated-score warning (Comer). Comer is not a "gains" paper at all.

### 1.5 Proposed rewording of line 157

Preferred (each citation attached only to what it supports; numbers sourced above):

"... and recent applied work in archaeology predicts site locations from terrain and hydrological variables with moderate-to-good discrimination (AUC roughly 0.7--0.9) while noting limits of data and transferability \citep{castiello2021, wang2023}, and shows that the sampling and validation design can over- or understate measured performance \citep{comer2023}."

Check against the papers: castiello2021 gives AUC 0.71/0.72 ("acceptable discrimination", "fair performance"), terrain and soil predictors, data quality and quantity as "an important constraint"; wang2023 gives test AUC 0.79-0.90 across models and areas, topographic and hydrological factors, and the limits "effectiveness in identifying other site types ... yet to be fully determined"; comer2023 states "over- or understate a model's performance".

Minimal alternative if the sentence must stay short: "... and recent applied work in archaeology reports moderate predictive performance from terrain-based variables while cautioning that data quality, sampling and validation design limit what that performance means \citep{castiello2021, comer2023}." This drops wang2023, whose paper does not discuss data quality or validation design. Either way, drop "strong gains" and "integrating spatial context"; the manuscript's own point (Hijmans 2012 on inflated cross-validated AUC, line 154) is exactly what comer2023 shows in an archaeological case, so comer2023 may also be moved next to `hijmans2012` in the preceding sentence.

---

## 2. Task 2: the three entries without a DOI

### 2.1 kvamme2006: verified, no mismatch in authors, title, pages; DOI and ISBN missing; two labelling notes

Sources fetched: Crossref record `10.1201/9780203563359.sec1` (`cr_kvamme.json`); CRC Press book page (`crc_book.html`); Open Library record OL3423048M (`ol_kvamme_book.json`); CNR catalogue record chloe.cnr.it/s/BiDiAr/resource/15914.txt.

| Field | .bib | Found | Source |
|---|---|---|---|
| Author | Kvamme, Kenneth L. | Kenneth Kvamme (Crossref); "Kvamme, Kenneth L." (catalogue) | Crossref, CNR record |
| Chapter title | There and back again: Revisiting archaeological locational modeling | "There and Back Again" + subtitle "Revisiting Archaeological Locational Modeling" | Crossref; CNR record gives the full title |
| Book title | GIS and Archaeological Site Location Modeling | same | Crossref, CRC page |
| Editors | Mark W. Mehrer and Konnie L. Wescott | "Edited By Mark W. Mehrer, Konnie L. Wescott" | CRC page; Open Library "edited by Mark W. Mehrer, Konnie L. Wescott" |
| Pages | 3--38 | 3-38 | Crossref; CNR record |
| Publisher | Taylor & Francis, Boca Raton | Crossref: "CRC Press"; CRC page: "by CRC Press", 1st edition; Open Library: "Taylor & Francis", "Boca Raton, FL" | CRC Press is the Taylor & Francis imprint; either is defensible, "CRC Press" matches Crossref |
| Year | 2006 | CRC page: "Copyright 2006"; Crossref issued 2005-12-21 (print), online 2010-01-14; Open Library "2005"; LCCN 2005044884; CNR record 2005 | genuinely ambiguous; 2006 is the copyright year and the one used in the literature |
| DOI | none | **10.1201/9780203563359.sec1** exists (chapter); book DOI 10.1201/9780203563359; the DOI resolves to a legacy host (crcnetbase.com), so the chapter page itself was not opened | Crossref |
| ISBN | none | 9780415315487 (print) and 9780203563359 (e-book) per Crossref; the CNR record lists another e-book ISBN, 978-0-429-21178-2 | Crossref, CNR |

Caveat: the CRC page's table of contents is truncated in the downloaded HTML, so the chapter title and pages are confirmed through Crossref and the CNR catalogue, not through the book's contents list. Open Library lists 476 pages and the CRC page 496 pages (not relevant to the citation).

Suggested addition to the entry: `doi = {10.1201/9780203563359.sec1}`, `isbn = {978-0-415-31548-7}`, publisher `CRC Press` (or keep `Taylor \& Francis`).

### 2.2 banks2006: verified, no mismatch

Sources fetched: the paper PDF from paleoanthropology.org (`dl_banks/banks2006.pdf`, 16 pp.); the journal's own citation pages (`dl_banks/bibtex.txt`, `ris.txt`); HAL record; Crossref title search (no DOI exists for this article).

- Title in the PDF: "Eco-Cultural Niche Modeling: New Tools for Reconstructing the Geography and Ecology of Past Human Populations". Matches.
- Pages: footer of the first page, "PaleoAnthropology 2006: 68-83. Copyright 2006 PaleoAnthropology Society." Matches 68--83. The journal's RIS gives VL 2006, SP 68, EP 83.
- Authors: the PDF byline lists 15 authors in the same order as the .bib: William E. Banks, Francesco d'Errico, Harold L. Dibble, Leonard Krishtalka, Dixie West, Deborah I. Olszewski, A. Townsend Peterson, David G. Anderson, J. Christopher Gillam, Anta Montet-White, Michel Crucifix, Curtis W. Marean, María-Fernanda Sánchez-Goñi, Barbara Wohlfarth, Marian Vanhaeran [sic].
- Two spelling traps, neither an error in the .bib: the PDF byline prints "Marian Vanhaeran" while the same PDF's reference list spells the same person "Vanhaeren" (the .bib uses Vanhaeren, correct); the journal's citation export prints "J. Christopher Gilliam" while the PDF byline and its references say "Gillam" (the .bib uses Gillam, correct). Anyone who cross-checks against the journal export will see two false mismatches.
- No DOI in the PDF and none in Crossref. Optional: `volume = {2006}` (the journal's own export uses it).

### 2.3 franklin2009: verified, no mismatch; DOI missing

Sources: Cambridge Core page, opened in the browser (meta tags: `citation_title: Mapping Species Distributions: Spatial Inference and Prediction`; `citation_author: Janet Franklin`; `citation_isbn: 9780521876353`, `9780521700023`, `9780511810602`; `citation_doi: 10.1017/CBO9780511810602`; `citation_publisher: Cambridge University Press`; `citation_publication_date: 2010/01`; `citation_online_date: 2012/08`); Crossref (`cr_franklin.json`: ISBNs 9780521876353, 9780521700023, 9780511810602); two reviews found by Crossref search.

- ISBN in the .bib, 978-0-521-87635-3, is the hardback ISBN. Correct.
- Year: Cambridge Core and Crossref carry 2010 (the date of the online edition). The print year 2009 is attested by two reviews whose Crossref titles read "Mapping Species Distributions: Spatial Inference and Prediction by Janet Franklin (2009), xviii + 320 pp., Cambridge University Press, Cambridge, UK. ISBN 9780521876353 (hbk)" (Oryx, DOI 10.1017/S0030605310001201) and "Cambridge University Press, 2009, 320 pp ... Hardback, ISBN-978-0-521-87635-3" (Landscape Ecology, DOI 10.1007/s10980-011-9603-5). So 2009 is defensible; a checker that uses Crossref will say 2010.
- The Landscape Ecology review title reads "Janet Franklin (with a contribution by Jennifer A. Miller)"; the Cambridge Core metadata lists only Franklin. No change needed.
- Missing: `doi = {10.1017/CBO9780511810602}` (exists).

---

## 3. Task 3: names and consistency across the whole reference list

Method: all 26 DOIs in the .bbl were fetched from Crossref and every author compared as "given family" (diacritics, hyphens, particles, initials versus full names), plus title, volume, issue, pages, year and journal; the three non-DOI entries were checked in section 2. Where Crossref looked unreliable (split names in Elith et al.), the publisher page was opened. Names that match after decoding LaTeX accents: all other entries, including diacritics (Araújo, Dudík, Jiménez-Valverde, Soberón, Schröder, Lira-Noriega, Hortal, Lahoz-Monfort, Sánchez-Goñi, d'Errico), hyphenation (Barbet-Massin, Guillera-Arroita, Montet-White, Scachetti-Pereira) and initials. No cited entry has a particle such as "van"/"de" in the author names, so that check is vacuous here.

### 3.1 Problems table

| # | key | problem | Crossref / publisher form |
|---|---|---|---|
| 1 | comer2023 | All five authors wrong (Douglas C. Comer, Michael J. Harrower, Joy McCorriston, Benjamin O'Meara, Karim Sadr) | Jacob A. Comer, Douglas C. Comer, Ioana A. Dumitru, Carey E. Priebe, Jesse L. Patsolic |
| 2 | comer2023 | Title lacks subtitle | Sampling methods for archaeological predictive modeling: Spatial autocorrelation and model performance |
| 3 | comer2023 | Volume 47 | 48 (Article 103824; no issue number) |
| 4 | wang2023 | Seven authors, all wrong (Cheng Wang, Ying Li, Yijian Liu, Huan Liu, Jian Liu, Yijian Su, Huadong Guo) | Yuan Wang, Xiaodan Shi, Takashi Oguchi (three authors) |
| 5 | wang2023 | Title wrong | Archaeological Predictive Modeling Using Machine Learning and Statistical Methods for Japan and China |
| 6 | castiello2021 | First author given name "Marco" | Maria Elena Castiello (Crossref and JCAA page) |
| 7 | castiello2021 | Title wrong | An Explorative Application of Random Forest Algorithm for Archaeological Predictive Modeling. A Swiss Case Study |
| 8 | senay2013 | First author "Sabrina D. Senay" (not flagged by the JSON, which only compared family names) | Senait D. Senay (Crossref; PLOS ONE page byline "Senait D. Senay, Susan P. Worner, Takayoshi Ikeda") |
| 9 | elith2006 | Overton given as "Jacob McC. Overton" (not flagged; the JSON's CHECK on this entry came from Crossref's split names, but it hides this real omission) | Jacob McC. M. Overton (Crossref family "McC. M. Overton"; Wiley citation metadata "Jacob McC. M. Overton") |
| 10 | barbet2012 | "Cécile H. Albert" (low severity) | Cécile Hélène Albert (Wiley byline and Crossref) |
| 11 | kvamme2006 | No DOI and no ISBN, although the chapter has a DOI | 10.1201/9780203563359.sec1; ISBN 978-0-415-31548-7; publisher CRC Press (see 2.1) |
| 12 | franklin2009 | No DOI | 10.1017/CBO9780511810602 |
| 13 | list-wide: chen2016, valavi2019, guillera2015, wisz2009, kvamme2006 | Capitalisation after a colon is inconsistent: capital letter kept in "A scalable", "An R package", "Matching data", "An information-theoretic", "Revisiting", but lower case in the other subtitle entries (allouche2006 "prevalence", verhagen2012 "a live report", noviello2018 "multiparametric", yaworsky2020 "an evaluation", banks2006 "new tools", lobo2008 "a misleading", phillips2009 "implications", warren2011 "the importance") | One rule needed; the cause is the brace-protected first letter in those five .bib titles (publishers differ too: Crossref prints "An evaluation", "Multiparametric") |
| 14 | list-wide: comer2023, wang2023, wisz2009, senay2013, yaworsky2020 | Article numbers are printed as bare numbers in the page slot ("238", "8", "e71218", "e0239424", "103824"), indistinguishable from page numbers | Crossref `article-number` for these; fix with a `note = {Article 238}` field or a bst that prints "Article" |

Count: **14 problems**: 12 entry-level (rows 1-12; rows 1-7 are in the cited-together trio, rows 8-10 are author-name slips elsewhere in the list, rows 11-12 are missing DOIs) and 2 list-wide formatting inconsistencies (rows 13-14). The 12 entry-level problems fall on 8 entries (comer2023, wang2023, castiello2021, senay2013, elith2006, barbet2012, kvamme2006, franklin2009); of these, the first four plus elith2006 are factual errors in the printed author list, title or volume, the rest are an abbreviated given name and missing DOIs.

### 3.2 Checked and consistent (no action)

- Journal names: all written in full, none abbreviated (e.g. "Proceedings of the National Academy of Sciences", "ISPRS International Journal of Geo-Information", "PLOS ONE"). Crossref prints "PLoS ONE" for the 2013 article; the list uses the current "PLOS ONE" for both PLOS entries, which is consistent.
- Page ranges: en-dash style throughout (`1223--1232`).
- "et al.": none in the list (Elith et al., 27 authors, is printed in full); "et al." appears only in the in-text labels.
- DOI present in 26 of 29 entries; the three without are rows 11, 12 and banks2006 (no DOI exists).
- Entry years versus Crossref "issued": four differ because Crossref uses the online-first date (verhagen2012 online 2011; lobo2008 online 2007; radosavljevic2014 online 2013; valavi2019 online 2018). The list correctly uses the volume year; no change.
- JSON flags that are false alarms: valavi2019 and radosavljevic2014 (Crossref title contains markup tags), chen2016 (Crossref title is "XGBoost" with the rest in the subtitle field), elith2006 title flag (curly versus straight apostrophe). Crossref's own hyphen character is U+2010 and some family names are upper-cased (ALLOUCHE, TSOAR, KADMON); the .bbl forms are correct.
- Author-name check also confirmed the banks2006 byline against the PDF (section 2.2).

---

## 4. Files in this folder

`cr_*.json`, `cr_all/*.json` (Crossref records); `dl_jcaa/` (JCAA HTML and text); `dl_mdpi/` (MDPI PDF and extracted text); `dl_banks/` (PaleoAnthropology PDF and journal citation exports); `crc_book.html`, `ol_kvamme_book.json` (Kvamme book records); `img/` (rendered MDPI pages used to verify quotes); `compare_bbl_output.txt` (output of the bbl-versus-Crossref comparison). Scripts used (outside this folder): `..\tools\fetch_cr.py`, `..\tools\compare_bbl.py`.

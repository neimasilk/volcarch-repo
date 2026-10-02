# P23 — VENUE.md (gate G15): Digital Humanities Quarterly (DHQ)

**Date checked: 2026-10-02.** Compiled for `docs/SUBMISSION_INTEGRITY_GATE.md` G15 ("venue first, zero cost").
**Status of this file:** survey of the venue and of its recent articles. It contains **no draft prose** for P23;
the prose is the PI's (G16).

## 0. Bottom line (read this first)

1. **Cost: DHQ is verifiably zero-cost** (no APC, submission, page or English-editing charge; free OJS account).
2. **Fit: DHQ is a POOR fit for P23 as currently scoped** (a resource/method paper, claim size "pipeline + list,
   not settlement history"). DHQ's own FAQ names "routine analyses of data sets or text corpora" and "standalone
   data sets" as things it does **not** generally publish without substantial revision. It becomes a **fair** fit
   only if P23 is re-scoped as a **critical case study** with an argument about toponym resolution as source
   criticism (precedents below), **and** the human validation (`validation/PROTOCOL.md`) is done first.
3. **Correction to the brief:** the "two independent coders (κ 0.90)" are, per `README.md`, **two LLM agents**
   adjudicated by an LLM, with human validation still pending. This is the largest fit/credibility risk in a
   venue whose readers are humanists, and it triggers DHQ's AI-acknowledgement rule (section 1.3, risk 3).
4. **Better genre match for the paper as scoped: JDMDH** (Journal of Data Mining & Digital Humanities, Episciences,
   diamond OA) — section 5. Trade-off: no Scopus/WoS, and its real review times are not shorter than DHQ's.
5. The survey behind this file: 210 DHQ items (vol. 17.1 – 20.2), 8 in-window articles profiled (section 2),
   5 older precedents (section 2b). G15's "≥ 5 articles from the last 2 years" is met by **8** (window starts
   2024-10-03).

## 1. Venue facts (DHQ)

### 1.1 Cost, indexing, review, licence

| Item | Finding | Source URL | Checked |
|---|---|---|---|
| Fees | "DHQ does not impose any article submission charges, article processing charges, or costs of any kind upon authors." FAQ: "No, DHQ does not charge fees of any kind to authors or readers." About: "DHQ does not charge any fees of any kind." | https://dhq.digitalhumanities.org/submissions/publication_terms.html · https://dhq.digitalhumanities.org/about/faq.html · https://dhq.digitalhumanities.org/about/about.html | 2026-10-02 |
| Mandatory registration | Free OJS account only ("If this is your first DHQ submission under OJS, you will need to create an account"). No membership/fee found anywhere. Portal: https://openjournals.library.northeastern.edu/dhq/dhq (the older URL http://openjournals.neu.edu/ojs/dhq redirects there, HTTP 302, tested) | https://dhq.digitalhumanities.org/submissions/index.html | 2026-10-02 |
| Indexing | Web of Science **ESCI**; JIF 2022 0.4, 2023 0.5, 2024 0.8 (About page; the statistics page prints two "2024" lines, 0.5 and 0.8 — ambiguous, cite the About page); Journal Citation Indicator Q1; Google Scholar; DOAJ. **Scopus:** DHQ's own pages say "recently accepted for indexing in Scopus, and this process should be completed soon" and the About page already calls it "SCOPUS Indexed". A third-party web-search result (not DHQ) gives Scopus indexing with CiteScore 1.5 (2025). **Not independently verified** (SCImago returned HTTP 403 to me): check https://www.scimagojr.com/journalsearch.php?q=21100898016&tip=sid before quoting "Scopus" in any cover letter. | https://dhq.digitalhumanities.org/submissions/statistics.html · About page above | 2026-10-02 |
| Review process | Editorial-team screen against basic criteria, then external peer review "typically takes in the range of 2-4 months but can take longer if we have difficulty securing reviewers". Decisions: accept pending revisions / revise and resubmit / decline. Peer review "blind but not double-blind"; anonymising is optional. | submissions/index.html · submissions/peerReviewing.html | 2026-10-02 |
| Deadlines | "rolling three-month intervals with deadlines on January 15, April 15, July 15, and October 15." The same page still carries a stale 2022 notice about suspending the January deadline → **confirm in the pre-submission inquiry**. Next usable deadline for P23: **2027-01-15** (15 Oct 2026 is too soon: G13 WIP full, G14 rest not started). | submissions/index.html | 2026-10-02 |
| Acceptance rates (regular issues) | accepted/submitted: 2023 21/115 (18%); 2024 28/190 (15%); 2025 21/306 (7%, 292 decisions still pending); all years 403/1556 = 26% (the page prints 0.37 for that one cell, which does not match its own counts; the published/submitted ratio of 0.25 does match). Published/submitted: all years 25%, 2023 17%, 2024 14% (pending). Special issues are far higher (all years 76%). **Submissions to regular issues rose 115 → 190 → 306 (2023 → 2025).** Table last updated "July 10, 2026". | https://dhq.digitalhumanities.org/submissions/statistics.html | 2026-10-02 |
| Licence / rights | Default CC BY-ND 4.0; authors may specify CC BY or CC0 ("authors may also specify a more permissive license (either CC-BY or CC0)"). "The author retains copyright over the work and is free to publish it in any other venue or format they choose, including pre-publication release and inclusion in an institutional or disciplinary repository." Preprint/Zenodo deposit is therefore allowed; simultaneous submission elsewhere is not. | publication_terms.html · about.html | 2026-10-02 |
| Language | Review capacity in English, French, Spanish, Portuguese; "DHQ accommodates a wide range of 'international English'"; mentoring and copyediting offered. | about/faq.html | 2026-10-02 |
| Pre-submission contact | FAQ: "we are happy to provide feedback on an abstract or draft, to help determine whether the piece seems suitable for DHQ." Author-support page: "reading draft abstracts and articles and providing initial feedback". Address on all pages: dhqinfo@digitalhumanities.org (OJS login problems: dhq@northeastern.edu). | about/faq.html · submissions/author_support.html | 2026-10-02 |
| Current special-issue CFP | AI and DH pedagogy (abstracts were due 2026-08-01) — irrelevant to P23. | submissions/cfps.html | 2026-10-02 |

### 1.2 Format and style rules (verbatim, with URLs)

**Article types** (https://dhq.digitalhumanities.org/submissions/index.html):
- "Articles: Article-length pieces describing original research."
- "Case Studies: Detailed analyses of specific projects that contextualize the project within the DH field, and demonstrate its significance for other practitioners."
- "Field Reports: Reports on digital humanities-related practice from the perspective of a particular locale"
- Observed in DHQ's own XML metadata (`<dhq:articleType>`, all 210 items): 190 article, 13 case study, 5 review, 1 opinion, 1 introduction, **0 field report**.

**What DHQ requires of any submission** (same page):
- "The submission must communicate effectively to the broad DHQ readership, rather than being narrowly limited to specialists in a particular subdomain."
- "DHQ articles should be clear without being elementary; they should not rely on insider knowledge, and they should situate their argument within a broader context of research. This may involve glossing terms, providing context, and including explanation of the significance of the research so that readers in other areas of digital humanities can understand and apply the results in their own research."
- "It must have an argument, and it should represent an original contribution to the research and practice of the digital humanities field, or should offer an original analysis, critique, or viewpoint on some aspect thereof. The submission should also engage with relevant strands of research or debate within the digital humanities field."
- "It must be well written, and must present its argument clearly and interestingly. (However, we can help the author improve the writing and argumentation, so a lack in this area is not necessarily a disqualification.)"

**What DHQ does not generally publish** (https://dhq.digitalhumanities.org/about/faq.html — the most important quote for P23):
- "DHQ does not generally publish the following without substantial revision: routine applications of well-established DH tools, or routine analyses of data sets or text corpora: these would be worth submitting to your own disciplinary journals"
- "standalone data sets: we welcome data sets that accompany a publication"
- Target audience: "The journal assumes some familiarity with the digital humanities, but not specialist knowledge of any particular domains."

**Length:** "Submissions may be of any length, but please bear in mind that although the digital medium is comparatively unbounded, reader attention is not; very long submissions must merit the space they occupy." (index.html)

**Originality:** "Submission of a manuscript will be understood as confirmation that it represents unpublished original material and that it is not being considered for publication elsewhere. Publication of related material on a blog, or publication in another language is fine."

**Files:** accepted as DHQ-TEI XML, TEI XML, or "RTF, OpenOffice (and its variants), or MS Word". "In the initial submission, for ease of reviewing, figures should be embedded in the text." Final files: images PNG/JPG/GIF/SVG/PDF; "Data sets: plain text, CSV, tab-delimited data, or XML"; "Executable code: Please contact the journal for details". (index.html)

**Headings, notes, figures** (https://dhq.digitalhumanities.org/submissions/textGuidelines.html):
- "Headings should be used for major sections of the article, to signal the important segments and turning points in the argument. They should not be used to outline every minute point in the article."
- "DHQ also strongly advises authors to avoid numbering/lettering headings (e.g. "3. Methodology" or "D. Further Issues") unless the inclusion of numbers will materially assist readers in following the argument." (In practice 37% of surveyed articles are numbered.)
- "Notes should be used only for comments, not for simple bibliographic citations."
- "All figures must also be accompanied by a description suitable to serve as the alternative text for accessibility purposes." Figure descriptions: "from a short phrase to a sentence in length"; captions "may be the length of a short paragraph". Files named figure01.jpg, figure02.jpg, ...
- "DHQ permits the use of regional spelling variants (such as US vs. Commonwealth English spelling)."

**Citation style** (https://dhq.digitalhumanities.org/submissions/citationGuidelines.html):
- "DHQ uses the Harvard Referencing system."
- In-text: author surname + year, optional location ("p. XX", "¶XX", "§X"); "If there are more than three authors, in-text references should be abbreviated as the first author's surname followed by "et al"."
- "Multiple references ... should never be combined. Please provide each reference as a unique entity." (i.e. no "(A 2020; B 2021)" clusters in one bracket)
- Journal article: `Author Name. (YYYY) 'Article Title', Journal Title, volume(issue), pp. page–range. {Available at: URL or DOI (Accessed: DD Month YYYY).}`
- The textGuidelines page says the list is headed "References"; the published pages show "Works Cited" with bracketed labels ([Lordick et al. 2016, 187]) generated by the stylesheet.

**Abstract / keywords:** the guidelines page states **no abstract length and no keyword rule**. The author template (https://raw.githubusercontent.com/Digital-Humanities-Quarterly/dhq-journal/refs/heads/main/articles/templates/dhq_author_template.xml) asks for "a brief abstract" and "a brief teaser, no more than a phrase or a single sentence", and says "Authors may suggest one or more keywords from the DHQ keyword list ... these may be supplemented or modified by DHQ editors" (list: https://dhq.digitalhumanities.org/common/xml/taxonomy.xml; relevant entries: `annotation`, `archaeology`, `area_studies`, `corpora`, `data_curation`, `geospatial`, `history`, `markup`, `metadata`, `nlp`, `standards`) plus free author keywords. Empirical abstract length is in section 3.

### 1.3 AI policy (binding on P23; verbatim from https://dhq.digitalhumanities.org/submissions/ai_policies.html)

- "Content generation: Authors of DHQ submissions may not use AI tools to generate the content of their submission. Submission of an article to DHQ constitutes an assertion that the ideas and their expression in writing are the work of the author. We recognize that many modern tools for research and authoring include AI assistance for things like identifying sources, creating diagrams, translating or clarifying language, etc. These are all permitted, but as noted above, the author has final responsibility for the accuracy and quality of the content."
- "Content accuracy: DHQ submissions may not include hallucinated or falsified citations, or hallucinated content of any other kind. ... Submissions that are found to have citations of non-existent resources will be declined, and the authors may be barred from resubmission at the discretion of the journal."
- "Acknowledgement: Authors of DHQ submissions must acknowledge any substantive use of AI tools in their research and writing process (taking into account the guiding principles articulated above), that goes beyond the use of routine utilities such as grammar checking. This acknowledgement should be included in the body of the article, either as part of the discussion of methods, or as an appendix."
- "Submissions which are found to violate these guidelines will be declined."

Consequence: G16 (PI writes the prose) is **compatible**; but LLM agents that did the coding are a "substantive use of AI tools in [the] research process" and must be disclosed **in the Methods or an appendix of the article**, not only in a cover letter.

## 2. Per-article table — in-window set (published ≥ 2024-10-03; these satisfy G15's "≥ 5 in the last 2 years")

How these were "read": each article's HTML was downloaded and parsed; I read the abstract, the introduction's first
paragraphs, the full heading list, the method/evaluation/limitations/data-availability passages and the closing
paragraphs, and counted words, figures, tables, notes and references by script. I did **not** read every paragraph
of every article. "Body words" = text of the article body incl. headings, captions and table cells, **excluding**
abstract, notes and reference list.

| ID | Citation · URL · DOI | Issue · DHQ type | Body words (+notes) | Fig / Tab / Refs / Notes | Data/code statement | Agreement / metrics | Closeness to P23 |
|---|---|---|---|---|---|---|---|
| W1 | Gravier, Baciocchi, Cristofoli, Duménieu, Carlinet, Chazalon, Abadie, Tual and Perret (2026) 'Evaluating and Understanding the Geocoding of City Directories of Paris (1787–1914): Data-Driven Geography of Urban Sprawl and Densification', DHQ 19(4). https://dhq.digitalhumanities.org/vol/19/4/000814/000814.html · 10.63744/eh2pbu98sxve | 19.4 (special issue "History and Data Science"), published 2026-01-30 · article | ~10,740 (+433) | 15 / 0 / 55 / 17 | Section "Data and Materials": Zenodo DOI 10.5281/zenodo.16994481; Nakala archive; GitHub in notes | geocoding evaluated manually, in Appendix 8.3 (overall + city-edge case); no metric tables | **High** (matching historical place references to a gazetteer; source effects) |
| W2 | Mordechai, Stahl, Pyzyk and Curto Pelle (2025) 'Systematic bias in humanities datasets: ancient and medieval coin finds in the FLAME project', DHQ 19(1). https://dhq.digitalhumanities.org/vol/19/1/000770/000770.html · 10.63744/6xsv95vxqxd4 | 19.1 (general), 2025-02-14 · article | ~8,460 (+1,139) | 5 / 0 / 45 / 29 | none (the resource is the online project) | none; typology of bias + maps | **High** (framing model: archaeological survivorship/discovery bias, honest limits) |
| W3 | De Weerdt, Ho, Simon, Lee, Molenaar, Xi, Zhuang, Stojević, Tu, Zaneri, Lin and Meister (2025) 'Contextual Semantic Text and Image Annotation in the MARKUS Environment', DHQ 19(4). https://dhq.digitalhumanities.org/vol/19/4/000808/000808.html · 10.63744/kb32jkqh9qeh | 19.4 (not listed in the special-issue introduction), 2025-11-14 · article | ~16,750 (+58) | 24 / 2 / 97 / 1 | none; tool wiki on GitHub | none | Medium (pre-modern inscriptions, gazetteer metadata in the data model; long outlier) |
| W4 | Jones, Faghihi, Parsons, Evans and Antur (2026) 'Decoding and Encoding Welsh Manuscript Culture: Scribes, Scripts and TEI', DHQ 20(1). https://dhq.digitalhumanities.org/vol/20/1/000852/000852.html · 10.63744/df9p2wshavdb | 20.1, 2026-02-20 · article (abstract calls itself "This case study") | ~12,050 (+0) | 14 / 0 / 18 / 0 | dataset managed on GitHub; schema repository linked in text | none | Medium-high (printed reference work → structured TEI dataset; identification/mapping; semi-automated vs manual) |
| W5 | Santana, Vasques Filho, Bojanowski and Błoch (2025) 'Unveiling the Critical Nexus of Data Preprocessing and Transparent Documentation for Result Quality and Reproducibility in Digital History', DHQ 19(2). https://dhq.digitalhumanities.org/vol/19/2/000788/000788.html · 10.63744/bgd67ykve8v7 | 19.2, 2025-07-18 · article | ~10,660 | 16 / 0 / 80 / n.a. | **separate headings "Data Availability" and "Code Availability"**, each a Zenodo DOI (10.5281/zenodo.15766967; 10.5281/zenodo.15090621); also "Author Contributions Statement", "Competing Interests Statement" | model-comparison metrics | Medium (template for data/code statements; "documentation as argument") |
| W6 | Wang Szilas (2026) 'The Naxi Dongba MOOC: A Test Case for Digital Revitalisation of Endangered Writing Systems', DHQ 20(2). https://dhq.digitalhumanities.org/vol/20/2/000870/000870.html · 10.63744/pe9t2d27qxqu | 20.2, 2026-06-19 · article | ~10,580 | 2 / 11 / 38 / n.a. | none seen | **Methods subsection "3.3 AI-assisted Qualitative Coding"** (ChatGPT as coding assistant; prompt printed; author reviewed every response) | Low on topic; **High as precedent for disclosed LLM coding** |
| W7 | Glass (2026) 'Visions in the Machine: Automated Tagging of the William Blake Archive', DHQ 20(2). https://dhq.digitalhumanities.org/vol/20/2/000869/000869.html · 10.63744/rzpwrx5vtt6s | 20.2, 2026-06-12 · article | ~12,630 | 3 / 11 / 33 / n.a. | none seen; Appendix A has corpus manifest | 6.6 "Evaluation Methodology and Results": ground truth by one annotator (stated), protocol, quantitative + qualitative results, 6.6.7 Limitations | Medium (annotation protocol and ground-truth exposition; modest claims) |
| W8 | Bandara, Gallant, Huq and Chowdhury (2026) 'The Eras Tour: Machine Learning for Dating Historical Texts from Greco-Roman Egypt', DHQ 20(1). https://dhq.digitalhumanities.org/vol/20/1/000831/000831.html · 10.63744/ed5krxzr7tyb | 20.1, 2026-04-10 · article | ~7,410 | 8 / 3 / 20 / n.a. | none seen | MAE, R², cross-validation; best ensemble MAE 45.7 years | Medium (ancient documents; person and place names as features; the "technical-metrics" end of DHQ) |

### 2a. Per-article notes (W1–W8)

**W1 (Gravier et al. 2026).** *Headings:* 1. Introduction · 2. Literature review: Geocoding addresses in historical spaces · 3. Dataset (3.1 Compilation of the study corpus; 3.2 Parisian space as conceived by editors) · 4. Methodology: density analysis (4.1, 4.2) · 5. Urban sprawl and density of Paris (1822–1914) (5.1–5.3.2) · 6. Discussion (6.1 difficulties of capturing the edges of the city; 6.2 conditions for re-appropriating digital and enriched data; 6.3 a back-and-forth process between pipeline extraction, data analysis and sources) · 7. Conclusion · 8. Appendix (8.1 gazetteer and geocoding operation; 8.2 sub-corpus; **8.3 Evaluation of the geocoding**; 8.4; 8.5 access points) · Data and Materials. *Framing:* a historical question (how Paris sprawled and densified) whose answer depends on "understanding source effects"; the abstract closes "The findings underscore the significance of data science in critically evaluating digital sources and adhering to best practices in the production of large historical datasets." The special-issue introduction (DHQ 19.4, 000842) names the concept "re-appropriation" (data made for one purpose reused for another). *Methods:* the pipeline and the geocoding evaluation sit in the **appendix**; the body is the historical argument. *Claim size:* calibrated; no claim beyond the dataset's coverage. *Take-away for P23:* put the matching/identification evaluation in an appendix and lead with the source-critical argument.

**W2 (Mordechai et al. 2025).** *Headings:* Introduction · 1. Introduction to the FLAME project (the late-antique–early-medieval transition; Technical Approach) · 2. Issues of bias (Regional Bias; Case Study: The United Kingdom; Primary biases (ancient); Secondary bias (archaeology); Tertiary (scholarship); FLAME project biases) · 3. Lessons for Digital Humanities · 4. Conclusion. *Framing:* opens "When did antiquity end and when did Western Eurasia become medieval?", then turns to the data's inherent distortions "to frame a discussion of such inherent biases in other digital humanities undertakings". *Claim size:* self-limiting — "This makes the exercise of gauging robustness in late antique and medieval coin data more epistemological than technical. In our opinion, there is no technical solution to this problem." No tables, no metrics, 5 figures. *Take-away:* a paper whose result is "what the data cannot support" is publishable in DHQ when it is turned into lessons for other projects.

**W3 (De Weerdt et al. 2025).** *Headings (unnumbered):* Introduction (The Problem; On Annotation) · From named entities to structured events and data clusters in semantic digital text annotation · From image tagging to data-rich semantic annotation with ontologies and custom data models · Conclusion: Contextualizing data through annotation and the creation of contexts in platform design · Appendix (Translation into RDF; Integrating Text and Image Annotation Models). *Framing:* first sentence "Historians and humanities scholars more generally care deeply about context." Tool-design rationale grounded in a social history of Chinese city walls, roads and bridges known from inscriptions. *Take-away:* an epigraphic, pre-modern, non-European corpus appears in DHQ when the argument is about humanistic modelling, not about the corpus.

**W4 (Jones et al. 2026).** *Headings (numbered, partly capitalised):* 1 Introduction · 2 RELATED RESEARCH (2.1 The Repertory and its context; 2.2 Related Resources; 2.3 Manuscript Catalogues as datasets) · 3 METHODOLOGY AND OUTPUTS (3.1 General Considerations; 3.2 Workflow; 3.3 Structuring and segmenting; 3.4 Mapping and identification; 3.5 Semi-automated and manual processes; 3.6 Outputs [Dataset, Schema, Wikidata/SNARC integration, further work]; 3.7 Initial Research Findings) · 4 METHODOLOGICAL CONCLUSIONS · 5 NEXT STEPS (5.1 sustainability and accessibility; …). *Framing:* a revered printed reference work "can only reach maturity in a second edition" → turn it into data. *Take-away:* "Methodological conclusions" and "Next steps" sections are normal here; sustainability of the dataset is part of the argument.

**W5 (Santana et al. 2025).** *Headings:* Introduction (with sub-headings on preprocessing and documentation) · Case Study: … · Impact of Preprocessing · Model Selection, Parametrisation, and Evaluation · Conclusion · **Data Availability · Code Availability** · Acknowledgments · Author Contributions Statement · Competing Interests Statement. *Framing:* the argument *is* documentation and reproducibility in digital history. *Take-away:* a ready-made back-matter pattern for P23's Zenodo deposit and AI-use statement.

**W6 (Wang Szilas 2026).** Methods subsection 3.3 prints the prompt, describes a three-stage process and states "The author manually reviewed every individual response against its assigned category", noting "No formal codebook was provided to ChatGPT in advance" and flagging "the need for ongoing methodological reflection and transparency in reporting such processes." *Take-away:* DHQ has already published disclosed LLM-assisted coding **with a human check of every item**; P23's LLM coders would be a much heavier use (the LLMs *are* the coders) and need a stronger human check than this.

**W7 (Glass 2026).** Numbered 1–9 with deep sub-numbering and an Appendix A (A.1–A.5); 6.6.1 Ground Truth Creation, 6.6.2 Annotation Protocol Details ("I performed all annotations…"), 6.6.7 Limitations; "success is intentionally modest". *Take-away:* DHQ accepts a single-annotator ground truth if it is stated plainly and the claims are modest; it does not require IAA.

**W8 (Bandara et al. 2026).** Numbered 1–7 (Related Work; Dataset; Methodology 4.1–4.4; Results; Discussion incl. Future work; Conclusion); shows the technical-metrics register is acceptable, and that claims such as "surpassing traditional techniques" are published. Limitations are stated in a closing paragraph (names persisting across long spans; noisy name annotations). *Take-away:* the register is the PI's strength, but it is **not** what P23's planned claim size ("no claim about settlement history") needs.

### 2b. Older precedents (closest in kind, **before** the 2-year window; not counted toward G15's five)

| ID | Citation · URL | Issue · DHQ type | Body words | Why it matters |
|---|---|---|---|---|
| P1 | Pedersen and Johansson (2023) 'Historical GIS and Guidebooks: A Scalable Reading of Czechoslovak Tourist Attractions', DHQ 17(2). https://dhq.digitalhumanities.org/vol/17/2/000679/000679.html · 10.63744/w5bsntasj8bw | 2023-05-26 · article | ~6,640 | Toponym matching against a gazetteer (GeoNames + WikiData): "A persistent challenge for HGIS is disambiguation"; "we opted for a very restrictive threshold to ensure that only highly similar toponyms were auto-matched (Jaro-similarity of 0.9)"; remaining toponyms placed by hand and "added to the gazetteer"; Table 1 counts indexed toponyms vs unique positions; CITADEL on GitHub + Zenodo (in notes). 6 figures, 56 refs, unnumbered headings. |
| P2 | Bermúdez-Sabel and Dell'Oro (2024) 'An Annotated Multilingual Dataset to Study Modality in the Gospels', DHQ 18(1). https://dhq.digitalhumanities.org/vol/18/1/000737/000737.html · 10.63744/r386n4bcam37 | 2024-03-29 · **case study** | ~5,580 | The nearest precedent for a **resource paper**: an XML-TEI dataset, typed "case study" by DHQ, short (5.6k words), sections Introduction · Modality and Its Annotation · Workflow · Description of the Dataset · Data Exploration · **Limitations** · Conclusion; GitHub + Zenodo; admits "Our source texts are not open access, and we therefore have publication constraints". Reports no IAA. |
| P3 | Sánchez-Salido, Menta and García-Serrano (2024) 'Seeking Information in Spanish Historical Newspapers: The Case of Diario de Madrid', DHQ 17(4). https://dhq.digitalhumanities.org/vol/17/4/000735/000735.html · 10.63744/jkw5qgh8qhdn | 2024-02-07 · article | ~10,970 | NER corpus in a scarce-resource setting; four human annotators, blind pass, tool-computed IAA per entity class used to revise the taxonomy; a subsection "3.4.1 Evaluation of the adjudication method" shows the adjudication rule changes model rankings. 13 tables, 80 refs, deep-learning primer in an appendix. |
| P4 | Bender, Becker, Kiemes and Müller (2023) 'Category Development at the Interface of Interpretive Pragmalinguistic Annotation and Machine Learning …', DHQ 17(3). https://dhq.digitalhumanities.org/vol/17/3/000720/000720.html · 10.63744/36sq5knxuhzn | 2023-12-20 · article (special issue on categories) | ~9,210 | The clearest example I found of Cohen's κ reported **with** raw agreement and explained (other articles mention κ or IAA more briefly): "(88%, Cohen's kappa: 72.87)", "(83.5%, Cohen's kappa: 57.44)", average 65.02; explains why a rare label depresses κ and cites thresholds. Template for how to report agreement to a humanities readership. |
| P5 | Pluschkovits (2024) 'Annotating German in Austria: A Case-study of manual annotation …', DHQ 17(3). https://dhq.digitalhumanities.org/vol/17/3/000729/000729.html · 10.63744/2vmmsaghm6my | 2024-02-23 · article (special issue) | ~7,660 | Opens: "especially in linguistics, specific annotations and their annotation guidelines are seldomly published or discussed" — the "annotation is under-theorised" framing that a coding-protocol paper can ride. 8 figures, 30 refs. |

Also screened, not profiled in detail: 000681 Tagami and Satlow 2023 (ML on Israel inscriptions; IMRaD, 4.6k words, GitHub link); 000727 Zirker and Göggelmann 2023 (annotation case study with guidelines in an appendix); 000726 Balck et al. 2023 (travelogue ontology; says its ontology patterns are "not yet published").

## 3. Corpus baseline (DHQ vol. 17.1 – 20.2, publication dates 2022-12-22 to 2026-06-26)

Sample = the 182 items typed article or case study with > 3,000 words (of 210 in all issues). Numbers by script
from the published HTML and the XML in the DHQ GitHub repository; method and caveats in section 7.

| Measure | Value |
|---|---|
| Body words (excl. abstract, notes, references) | median **7,783**; interquartile range 6,451–9,598; 20% under 6,000; 12% under 5,000 |
| Case-study-typed items (n = 13) | 4,910–10,059 words, median 7,599; "Gospels" dataset case study 5,580 |
| Abstract length | median 170 words (IQR 136–216; max 708) |
| References | median 44 (IQR 28–58) |
| Notes | median 6 (IQR 1–16) |
| Figures | median 5 (IQR 1–8); 24% have none |
| Tables | median 0; 38% have at least one |
| Headings numbered | 37% |
| Repository link (GitHub/Zenodo/OSF/Nakala/…) in the text | 30% (31% for items published 2025–2026) |
| Explicit "Data/Code Availability" or "Data and Materials" heading | 6% |
| Appendix | 20% |
| "Limitation(s)" mentioned in the text / as a heading | 55% / 5% |
| Reports κ or inter-annotator agreement | 7% (12 of 182) |
| Southeast-Asian or Indic epigraphic/toponym article, 2023–mid-2026 | **0 of 210** (keyword screen: Indonesia, Java, Khmer, Cham, Sanskrit, Tamil, charter, land grant …). Nearest Asian items: Chinese Buddhist topic modelling (000771), MARKUS/Chinese inscriptions (W3), Naxi Dongba (W6). |
| LLMs as the coders in an agreement study | **none** (keyword screen: 15 items mention LLM terms ≥ 3 times; one, W6, discloses ChatGPT as a coding assistant) |

## 4. Genre template for P23

**Not prose; a structural recipe taken from the evidence above. The PI chooses every sentence.**

**Recommended type.** Submit as a **Case Study** (the guidelines define it as analysis of a specific project that "demonstrate[s] its significance for other practitioners"; P2 is the precedent: a TEI dataset paper typed "case study"). An "article" is the fallback label if the argument grows. Do not submit it as a bare resource note (see the FAQ quote).

**Target length.** Body **6,000–7,000 words** (excl. abstract, notes, references); abstract **150–200 words** plus the one-sentence teaser; **30–45 references**; **4–6 figures** (alt text for each); **at most 2 tables in the body** (frame counts; validation results); **6–15 notes**. Coding manual, spelling-variant merge rules, KWIC examples and the matching tests go to an **appendix** (W1 puts its geocoding evaluation there).

**Section skeleton (budgets are planning numbers, sum ≈ 6,600).** Headings should be descriptive and light on numbering.
1. Introduction — the humanities question, the contribution, a roadmap paragraph (P2, P5, W1 all have one) (~700)
2. Charters and village names, for non-specialists — glossary of sīma/wanua, why the names matter, what the DHARMA corpus is, what it omits (source effects) (~900)
3. Building the frame — extraction (KWIC), coding protocol, adjudication, spelling-variant merging, the 266-name frame (~1,500)
4. How far can the frame be trusted — **human validation first** (human vs final κ and raw agreement, error taxonomy); LLM–LLM agreement reported only as the consistency of a procedure (~1,100)
5. Locating the villages — the matching tests and the finding that they almost always fail; what that says about who must do the identifying (~1,100)
6. What this means for other historical-toponym projects — the lessons paragraph(s) DHQ expects (W2 §3) (~700)
7. Limitations (P2 has a full "Limitations" section; 55% of articles mention limitations) (~300)
8. Conclusion (~300)
Back matter (all in the article body, not the cover letter): **Data and Code Availability** (Zenodo DOI, licence, DHARMA credit; W5 pattern) · **AI-use acknowledgement inside Methods or an appendix** (W6 pattern; DHQ policy 1.3) · Funding · Acknowledgements · Author contributions.

**How to frame the humanities question (options; pick one, do not stack).**
- (a) *Source criticism of toponyms:* what counts as a place name in early Javanese charters, and who can decide whether it is identified? Connects to W2 (bias/source effects), W1 (re-appropriation, geocoding evaluation), P1 (disambiguation).
- (b) *Division of labour between machine candidate extraction and expert identification:* connects to W6 and P3 (adjudication).
- (c) *Annotation as under-theorised practice:* a coding protocol published with its failures; connects to P5 and P4.
Whichever is chosen, keep the claim size already fixed in `README.md`: no statement about settlement history or volcanism (CLAUDE.md F9/F10; ME#19). DHQ rewards a limits-first argument (W2) — P23's finding that matching fails is the argument, not an embarrassment.

**How much technical detail.** Medium. A DHQ reader needs: the definition of the task, the coding rule in one paragraph, one table of counts, one table of agreement/validation, the logic of adjudication and merging, and the matching result. Everything an expert needs to *audit* it (coding manual, merge rules, per-name list) goes to the appendix and to Zenodo. Explain κ in two or three sentences and report raw agreement beside it (P4); name the prevalence caveat.

**What reviewers in this venue reward** (inferred from DHQ's published reviewer criteria plus what the 182 articles do; I have no access to actual reviews):
- a stated argument that travels beyond the case ("so that readers in other areas of digital humanities can understand and apply the results");
- engagement with DH debates, visible in the reference list (annotation theory, historical GIS and toponym resolution, source bias, TEI datasets, documentation/reproducibility — all represented in DHQ 2023–2026);
- explicit limits and honest negative findings (W2, P2, W7);
- open data/code with a persistent identifier and a plain availability statement (W1, W5, P2), even though not mandatory (30% do);
- glossing for non-specialists; a clear roadmap; accessible figures.

**What reviewers in this venue punish / editors decline:**
- no argument, or a routine corpus analysis ("routine analyses of data sets or text corpora");
- a standalone dataset without a publication around it;
- reliance on insider knowledge (Old Javanese philology without glossing);
- conference-paper or dissertation-chapter shape (FAQ lists both);
- AI-generated prose; hallucinated or unverifiable citations (policy: declined, possible bar on resubmission);
- over-long text that does not "merit the space".

## 5. Risks of fit and mitigations

**Risk 1 — Genre and scope: DHQ may read P23 as a routine corpus analysis or a standalone data list.**
Evidence: FAQ quotes in 1.2; the planned claim size is deliberately small ("not a claim about settlement history").
Mitigation: (i) re-scope as a critical case study whose argument is the limits of toponym resolution for under-resourced charter corpora (framings a–c above), with precedents W1, W2, P1, P2; (ii) make the dataset *accompany* the publication (FAQ allows that); (iii) build the reference list so the paper visibly "engage[s] with relevant strands of research or debate within the digital humanities field"; (iv) send the **G15 pre-submission inquiry** (abstract + one paragraph) to dhqinfo@digitalhumanities.org — the FAQ invites exactly this — and **do not draft past the skeleton until the answer arrives**.

**Risk 2 — Audience and precedent gap, plus a tightening queue.**
Evidence: 0 of 210 DHQ items 2023–mid-2026 deal with Southeast Asian or Indic epigraphy or toponyms; regular-issue submissions rose 115 → 190 → 306 (2023 → 2025) while acceptance fell 18% → 15% → 7% (pending); DHQ asks that articles "should not rely on insider knowledge".
Mitigation: a short, plain "sources and why names matter" section with glossed terms; one explicit comparison to a better-known corpus (e.g. via P1/W1) so a non-Indologist reviewer can calibrate; ask in the inquiry whether the editors can find a reviewer for Old Javanese epigraphy; pick DHQ keywords `history`, `annotation`, `geospatial`, `corpora`, `area_studies`; use the author-support offer (draft-abstract reading, mentors, copyediting for English); target the 2027-01-15 deadline only after the G14 rest and after the editors confirm the deadline is live.

**Risk 3 — The coders are LLMs; DHQ readers expect human ground truth, and the AI policy applies to the method, not just the prose.**
Evidence: `README.md` and `validation/PROTOCOL.md` ("Agreement between two models says nothing about whether they are right"); only 12 of 182 DHQ articles mention κ or inter-annotator agreement, and the ones I read (P3, P4, and 000727) use human annotators; no DHQ article uses LLMs as the independent coders; DHQ policy requires acknowledging "substantive use of AI tools in their research ... process" in the body; the only precedent (W6) had a human review every item.
Mitigation: (i) finish the human validation first — an epigrapher codes the 100-occurrence blind sample; report human–final κ and raw agreement, the error taxonomy and the Wilson interval, **whatever they are**; apply the protocol's decision rule (if human–final κ(Y) < 0.70 or > 15% of Y names are wrong, the list is not published as a resource and the paper reports the LLM-coding failure as its result); (ii) call the LLM–LLM κ "consistency of an automated procedure" in the text; (iii) state the AI use in Methods with model identities, prompts and what the human did, following W6; (iv) PI writes the prose (G16) and every citation is checked to exist; (v) verify the DHARMA corpus licence permits publishing the derived name list before the Zenodo deposit (**not checked here**; `README.md` says "crediting DHARMA").

## 6. Verdict on fit and the alternative

**DHQ: poor fit for the paper as scoped; fair fit only for a re-scoped critical case study with human validation behind it.** The venue is zero-cost, visible (ESCI, Q1 JCI, likely Scopus), and has precedents for every component of P23 (TEI dataset as case study: P2; historical toponym matching: P1, W1; annotation and agreement: P3, P4, P5; AI-coding disclosure: W6; bias/limits as the argument: W2) — but no precedent for the combination in a non-European epigraphic corpus, and its FAQ lists the planned genre among things it does not generally publish. At a regular-issue acceptance of 15% and falling, a mis-scoped paper is a poor bet; ME#19's binding constraint is exposure, so the fastest informative external answer is the pre-submission inquiry (days), not a full submission (months).

**Better genre match for a resource + method paper: JDMDH — Journal of Data Mining & Digital Humanities** (Episciences; https://jdmdh.episciences.org). Evidence, all checked 2026-10-02:
- **Zero cost:** editorial policies (French, quoted): "Pas de frais pour les auteurs ni pour les lecteurs." Diamond open access, CC BY 4.0; no registration fee (Episciences account; manuscript first deposited as a preprint on HAL, arXiv or Zenodo — https://jdmdh.episciences.org/page/submissions). https://jdmdh.episciences.org/page/editorial-policies
- **Genre:** same page: "Les auteurs ayant conçu des outils logiciels utiles, des jeux de données librement accessibles ... peuvent également soumettre directement un article court avec sa description et sa mise en œuvre." and "L'accès aux jeux de données doit être mentionné". Submission language: English. A resource + protocol note is a native genre there, not a deviation.
- **Content:** special issues "L'intertextualité dans les langues anciennes", "Documents historiques et reconnaissance automatique de textes", "Visualisations en linguistique historique", "HistoInformatics". Recent articles checked: 'Dialect Cartography of Erzya and Moksha Languages: Digitized Historical Sources and Evaluation of the Contemporary Data' (settlement points vs map polygons; 10 pages; published 2025-11-13; https://jdmdh.episciences.org/16802 — PDF opened); 'Exploring Historical Labor Markets: Computational Approaches to Job Title Extraction' (annotation process, post-correction, evaluation; 13 pages; 2025-04-02; /15373 — PDF opened); 'Computational Pathways to Intertextuality of the Ancient Indian Literature' (Sanskrit; 2025-03-07; /15334 — title and dates only, PDF not obtained). Length 2,400–6,200 words before the reference list in the four PDFs I measured (also /14607 and /15327).
- **Caveats (honest):** indexed in DOAJ, DBLP, Google Scholar, Mir@bel — **not** Scopus or WoS (https://jdmdh.episciences.org/page/partners-and-indexing), which may matter for KUM; the advertised timetable (editor in 5 days, reviewers within 30 days) is **not** what the published dates show: of 20 recent articles (fetched from https://jdmdh.episciences.org/browse/latest), standalone submissions (n = 9) took a median of ~313 days from submission to acceptance (range 16–406); eleven were accepted in same-day batches (likely special-issue tracks) with a median of 31 days. No acceptance-rate statistic found. I read none of its articles in full, so **G15 would require a fresh VENUE.md for JDMDH** if the PI switches.

**Decision rule for the PI.** Keep DHQ only if the PI will write P23 as a limits-first case study (section 4), do the human validation first, and accept a 15%-and-falling acceptance rate; otherwise switch the primary venue to JDMDH, which fits the resource/method scope as it stands and the PI's NLP strength. Either way, **do not start a full draft before the human validation and the editors' reply.**

**Other venues checked.**
- *Journal of Open Humanities Data* — **excluded**: APC £1,070 for data papers and discussion papers; waiver "may" be granted on request, which is not zero (https://openhumanitiesdata.metajnl.com/about/submissions, checked 2026-10-02).
- *Journal of the Text Encoding Initiative* — candidate to examine later: a web-search snippet lists "Research papers (roughly 5,000 to 10,000 words), Project/Tool papers (roughly 2,000 to 6,000 words), and Data papers (roughly 1,500 to 4,000 words)" and single-blind review (https://journal.tei-c.org/index.php/journal/about/submissions); the site was unreachable from here (connection refused/timeout), so **fees and scope are unverified**.
- ACL-family workshops — registration fee (already excluded in `README.md`).

## 7. G15 sign-off line (for `SIG_signoff.md`; PI to confirm)

`G15 venue first / zero cost: [x] DHQ — fees: none, https://dhq.digitalhumanities.org/submissions/publication_terms.html + https://dhq.digitalhumanities.org/about/faq.html, checked 2026-10-02 · 8 in-window articles (W1–W8) + 5 precedents in VENUE.md · fit: POOR as scoped / FAIR if re-scoped as critical case study · pre-submission inquiry: NOT YET SENT (required by G15 because fit is unsure).`

**Open items (owner: PI)**
1. Decide: DHQ (re-scoped case study) or JDMDH (resource note). One line in `README.md` and `STATE.md`.
2. If DHQ: send the pre-submission inquiry (abstract + one paragraph; ask whether the 15 January deadline is live and whether Old Javanese epigraphy reviewers are available). Do not draft beyond the skeleton before the reply.
3. Complete the human validation (`validation/PROTOCOL.md`) before any results prose.
4. Verify the DHARMA corpus licence for redistributing the derived name list; verify Scopus status via SCImago if the cover letter will mention it.
5. Re-run this survey (script below) once the editors answer, because the AI policy page says it "is evolving".

## 8. Method of this survey (reproducibility)

- DHQ pages above fetched with curl/WebFetch on 2026-10-02 and parsed with BeautifulSoup. Issue tables of contents 17.1–20.2 gave 210 article URLs; each article's HTML and its TEI XML (https://raw.githubusercontent.com/Digital-Humanities-Quarterly/dhq-journal/main/articles/NNNNNN/NNNNNN.xml, `dhq:articleType`) were downloaded.
- Word counts: text of `div#DHQtext` minus `div#abstract`; notes and references counted separately from `div#notes` / `div#worksCited`. Table-cell and caption text is included in "body words", so counts are slightly higher than the prose alone.
- Keyword screens (toponym, kappa/agreement, LLM, Southeast Asia, charter) are regular expressions over the article text — they find presence, not quality; the "0 of 210" statements mean "no hit in the screen", and I read the top hits of each screen.
- JDMDH turnaround: submission/acceptance/publication dates printed on 20 article pages listed on its "latest" page (2026-10-02). The "batch" interpretation (same-day acceptance dates) is my inference.
- Not verified: SCImago/Scopus listing (HTTP 403), jTEI site (unreachable), DHARMA licence, whether the 15 January DHQ deadline is currently active.
- No external judge has seen P23. This file is an input to a decision, not a submission step.


---
**Orchestrator note 2026-10-02:** DHARMA licence verified — local snapshot `experiments/E023_ritual_screening/data/dharma/LICENCE.txt` is CC BY 4.0 ("Attribution 4.0 International"). Decision taken on this file: DHQ as Case Study (re-scoped claim, see README), JDMDH backup; final lock after human validation.

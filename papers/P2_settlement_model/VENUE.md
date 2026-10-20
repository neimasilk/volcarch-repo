<!-- G15 venue file for P2 (JCAA). Provenance: drafted 2026-10-20 by a Sonnet sub-agent from pages it fetched that day
(the downloads stay outside the repo: session scratchpad `venue/`). Checked by the orchestrator against the fetched HTML, same day:
the AI-use policy text (acceptable vs unacceptable uses, incl. "ghost-writing"), the 11,500-word limit incl. references (it was
8,500 in the 2026-09-22 Wayback copy), abstract ≤ 300 words / ≤ 6 keywords, alt text per figure, the waiver-policy and
Publication-Fund text, and the APC £651.00 (raw bytes C2 A3 = £). Everything else is the sub-agent's reading, with its URLs.
P2 v0.2 measured against it: ≈ 8,350 words incl. references (PyMuPDF, line numbers removed), 29 pp. -->

# VENUE.md — Journal of Computer Applications in Archaeology (JCAA)

Status: DRAFT for gate G15, prepared 2026-10-20 for P2 (revision due 16 Nov 2026). Not yet reviewed by the PI.
Scope: journal rules (Part A), eight recent articles (Part B), what P2 must match (Part C).

## 0. Method, evidence and limits

- Everything below comes from pages fetched on **2026-10-20**. Journal pages were fetched with curl (browser User-Agent) and the
  raw HTML was converted to text with a script kept outside the venue folder. The `/about/submissions` page was also fetched with
  WebFetch; its summary agreed with the raw HTML on every point checked (WebFetch's summary is NOT used as primary evidence).
- Where a fact comes from the Internet Archive (Wayback Machine) this is marked **[Wayback]**, with the snapshot date. Those pages were also
  fetched on 2026-10-20.
- Article evidence = the full-text HTML page of each article on the journal site, plus the figure PNGs I opened and looked at (maps).
  The article PDFs were **not** downloaded, so page counts are taken from the page range printed on the article page, and "how it
  looks in print" is not checked.
- Nothing was filled from memory. Items not found are marked **NOT FOUND**, with where I looked (list in A13).
- Word counts of articles are my own counts of the HTML text (see B0 for the method). They are approximate.
- The journal's own pages describe the rules as of 2026-10-20 only. Two rules changed in the last weeks (see A2 and A12), so
  **re-read the live pages on the day the revision is uploaded**.

Source key (every URL fetched 2026-10-20; local copies are in this folder):

| Key | URL | Local file |
|---|---|---|
| S1 | https://journal.caa-international.org/about/submissions (redirects to /en/about/submissions) | submissions.html |
| S2 | https://journal.caa-international.org/en/about/editorialpolicies | about_editorialpolicies.html |
| S3 | https://journal.caa-international.org/en/waiver-policy | waiver-policy.html |
| S4 | https://journal.caa-international.org/en/reproducibility | reproducibility.html |
| S5 | https://journal.caa-international.org/en/about | about.html |
| S6 | https://journal.caa-international.org/en/articles (pages 1-4 used: `?page=2`, `?page=3`, `?page=4`) | articles.html, articles_p2..p4.html |
| W1 | http://web.archive.org/web/20250601010526/https://journal.caa-international.org/about/submissions | wayback_20250601010526_submissions.html |
| W2 | http://web.archive.org/web/20260101010440/https://journal.caa-international.org/about/submissions | wayback_20260101010440_submissions.html |
| W3 | http://web.archive.org/web/20260811170005/https://journal.caa-international.org/about/submissions | wayback_20260811170005_submissions.html |
| W4 | http://web.archive.org/web/20260922065704/https://journal.caa-international.org/about/submissions | wayback_20260922065704_submissions.html |

Article pages (A1-A8) are listed in Part B. `about_editorialteam.html` was downloaded but not used.
Snapshots dated 2026-04-11, 2026-07-02 and 2026-07-23 returned only a 1.2-1.4 kB framework payload with no page text, so they are
not evidence (files `wayback_20260411_submissions.html`, `wayback_20260702181729_submissions.html`, `wayback_20260723214017_submissions.html` kept for the record).

---

# PART A — Author requirements

## A1. The journal

Source: S5 https://journal.caa-international.org/en/about (fetched 2026-10-20)

- Publisher: Ubiquity Press; E-ISSN 2514-8362; fully open access, CC BY 4.0 for articles; published continuously as one volume per year.
- Scope as stated: "the application and development of digital technologies in archaeology"; topics listed include "spatial analysis,
  remote sensing, ... statistics, data mining, simulation modelling, network analysis". Papers "can be targeted towards scientific
  research, cultural heritage management and/or public archaeology".
- Metrics the page states: Scopus CiteScore 2.8 (39/462 in Archaeology, 2025); SJR 0.425 (2025).
- Journal statistics for the 2025 volume (S5; the digits in the table are footnote markers, I read them from the HTML `<sup>` tags):
  58 submissions received; 253 review invitations sent; 114 reviews received; 26 rejections of which 17 desk rejections; 19 acceptances;
  acceptance rate 42%; mean submission-to-publication 208 days.

## A2. Article types and length

Source: S1 https://journal.caa-international.org/about/submissions (fetched 2026-10-20)

| Type | Text on the page about length (live, 2026-10-20) | Peer reviewed |
|---|---|---|
| Research Article | "no more than 11,500 words in length" | yes |
| Review | "no longer than 11,500 words in length" | yes |
| Case Study | "no longer than 11,500 words in length" | yes |
| Position Paper | "no longer than 11,500 words in length"; discuss with the editor before submitting | yes |

- General line: "Articles should normally be under 11,500 words. All word limits include referencing and citation." (S1)
- No separate rule on number of figures or tables was found.
- Case Study: "Authors should provide a detailed summary of the protocol followed and establish replicability within the body of the paper." (S1)
- Research Article: "should make a substantial contribution to knowledge" and "be supported by relevant figures and tabulated data". (S1, paraphrase)
- **The limit changed recently [Wayback].** W1 (2025-06-01), W2 (2026-01-01), W3 (2026-08-11) and W4 (2026-09-22) all say Research Article and
  Position Paper "8,500 words" and Review and Case Study "5,000 words" ("Articles should normally be under 8,500 words" in W2-W4). The live
  page of 2026-10-20 says 11,500 for all four. So the raise happened between 2026-09-22 and 2026-10-20. P2 was resubmitted on
  2026-08-11, when W3 shows the 8,500 / 5,000 limits.
- How the count is made (tables, captions, statements): **NOT FOUND** (searched S1, S2, S4 for "word"; only the sentence above exists).
- Observed practice (B0 method): 4 of the 8 articles in Part B (B2, B5, B7, B8) are above 11,500 words if references are included in my count.
  The 8,500 / 5,000 limits are the ones shown in the snapshots from 2025-06-01 to 2026-09-22, i.e. during the review and acceptance of
  these articles (accepted 2025-12-03 to 2026-04-15, and 2024-10-10 for B7, which pre-dates my earliest snapshot). What applied on the day each was
  submitted is not known to me. So the limit looks like a guide, not a hard cut-off. This is an observation only, not a rule to rely on.

## A3. Title page, abstract, keywords

Source: S1 (fetched 2026-10-20)

- "To ensure blind peer review, please only list the title and abstract on the submitted manuscript file." Author names, affiliations,
  contact details, optional biography and corresponding-author details are entered online. (S1)
- Author names: forename and surname; "Forenames cannot include only initials" (example given: "J. Bloggs is not preferred. The full name,
  Joe Bloggs is required"). Affiliation: ideally "Department, Institution, City, Country"; only institution and country are mandatory. (S1)
- ORCID: corresponding author is asked to submit one; co-authors "recommended" (S1 checklist; S2 "ORCID" section).
- Abstract: "no more than 300 words", must have the heading "Abstract", "easily identified from the start of the main text". (S1)
- Keywords: "A list of up to six key words must be placed below the abstract (required)". Abstract and keywords must also be entered as
  metadata on the online form. (S1)
- Peer-review blinding: S2 says the journal "does not dictate" blinding; "The authors choose whether to blind their submission files".
  S1 checklist: "The author acknowledges that they are responsible for anonymising their files, should they wish to pursue an anonymous
  peer review process."
- Preprints are allowed and are not prior publication; the cover letter must declare the preprint and link it; the journal cannot guarantee
  anonymity then. (S2, "Preprint Policy")

## A4. Structure, headings, required statements

Source: S1 (fetched 2026-10-20) unless marked

- Main text: "structured in a logical and easy to follow manner"; a "clear introduction section" for non-specialists; "Methods, results,
  discussion and conclusion sections may then follow". (S1)
- Headings: "Up to three level headings may be present and must be clearly identifiable using numbered paragraphs (1.1, 1.2, etc.)".
  Headings "should be under 75 characters". (S1)
- Capitalisation (S1): title and first-level headings: capitalise nouns, pronouns, adjectives, verbs, adverbs and subordinate conjunctions;
  lower-case articles, coordinating conjunctions and prepositions. Lower-level subheadings: only the first letter and proper nouns.
- Order of end matter as the page words it (S1):
  1. Acknowledgements (optional): "headed and in a separate paragraph, placed after the main text but before the reference list".
  2. Funding information (if applicable): "the grant provider and grant number should be detailed".
  3. Competing interests: "A short paragraph should be placed before the references." If none, the statement must read:
     "The author(s) has/have no competing interests to declare."
  4. Authors' contributions: "A sentence or a short paragraph detailing the roles that each author held".
  5. References: "listed at the end of the main text file".
  6. Declaration of AI use: "This should appear at the end of the manuscript." (see A5)
  7. Ethics and consent (if applicable); Peer Review section only for PCI-Archaeology-recommended articles.
  S1 does not give one strict sequence for all items; Notes go "at the end of the main text, before 'References'".
- Data Accessibility Statement: S4 says it is added "prior to the reference list" (see A6).
- Notes: "Use endnotes rather than footnotes (we refer to these as 'Notes' in the online publication)"; use only where crucial; place the
  marker after the end punctuation; do not use notes for referencing. (S1)
- Multimedia: up to 3 video and audio files, hosted on Vimeo and Soundcloud, linked in the manuscript. (S1)
- Permissions: authors obtain all permissions before submission; "If a method or tool is introduced in the study, including software ...
  the license this is available under and any requirement for permission for use should be stated" in the Materials and Methods section. (S1)
- Ethics: Declaration of Helsinki for human subjects, ethics-committee name and approval number; not relevant to P2 unless human data are used. (S1)

## A5. AI policy

Sources: S1 (fetched 2026-10-20) and S2 https://journal.caa-international.org/en/about/editorialpolicies#ai-ethics-policy (fetched 2026-10-20)

- "Artificial Intelligence should not be named as an author." (S1)
- Declaration on submission, at the end of the manuscript. The page's example: "Language was improved with the assistance of ChatGPT
  (OpenAI, 2025). The authors take full responsibility for the content." "In any case, authors are fully responsible for the contents of
  their submission." (S1, S2)
- S2 "Use of AI" says there are "acceptable and unacceptable uses ... Unacceptable ones are a form of fraud."
  - Acceptable: language and style assistance; idea brainstorming; technical or coding assistance ("drafting or debugging code,
    explaining code logic or statistical procedures"); summarisation and drafting support ("summarising literature you've already read,
    turning bullet points into prose, generating a prose based on your notes or data"); formatting and citation management.
  - Unacceptable: "Ghost-writing: using AI to write substantial portions of the paper (methods, results, discussion, etc.), even if the
    AI is listed as a coauthor"; "Fabrication: generating fake data, descriptions, quotes, or references; summarising papers not
    actually read by the author"; "Circumventing critical thinking: accepting AI-generated interpretations, arguments, or conclusions
    without human verification"; "Lack of transparency: carrying out an acceptable use and not disclosing it".
  - Suspected unacceptable use can lead to desk rejection; editors "may decide to reject further submissions from an author after a desk
    rejection"; an appeal route exists. (S2)
- Not found: the publisher-level policy (https://www.ubiquitypress.com/ai-policy and https://ubiquitypress.com/site/ai-policy/) returned
  HTTP 403 to both curl and WebFetch on 2026-10-20, so it was not read.
- Note for the project (a fact about the page, not a ruling): this policy sits next to the project's rule that the PI writes the prose
  (G16). Check the AI-use declaration wording against the "acceptable" and "unacceptable" lists above before submitting.
- Precedent: article 196 (Part B8) carries a section titled "Declaration of Generative AI and AI-Assisted Technologies in the Writing
  Process" naming Claude for code checking.

## A6. Data, code, reproducibility, permissions

Sources: S1, S4 https://journal.caa-international.org/en/reproducibility, S2 (all fetched 2026-10-20)

- "JCAA strongly encourages all authors to deposit material relating to their publication in a preservation repository." (S1)
- Repository criteria (S1): allows deposit "under the correct licence"; gives "a unique, persistent identifier (e.g. a DOI)"; has "a published
  preservation strategy that guarantees long term preservation". Journal does not dictate the repository; Ubiquity Press Dataverse is offered.
- Statement (S1): "If data, structured methods or code ... have been made openly available, a statement should be added to inform the reader
  how/where to access these files. This should include the repository location and the DOI linking to it." "If data used in the research
  project has not been made available, a statement confirming this should be added, along with reasoning why."
- S4 adds: a "Data Accessibility Statement" "prior to the reference list"; open licence (e.g. CC0, CC-BY); "a version that is in an open,
  non-proprietary format"; labelling so that "a 3rd party can make sense of it (e.g. sensible column headers, descriptions in a readme text file)".
- S2/S4 recommend (not require) structured methods on protocols.io and code on Code Ocean, or code in a data repository with a DOI.
- Data papers can go to the Journal of Open Archaeology Data. (S1)

## A7. Language and text rules

Source: S1 (fetched 2026-10-20)

- Spelling: UK or US English, used consistently; official spelling for proper nouns. Grammar and serial comma: either, but consistent.
- Quotation marks: single quotation marks; double only for quotes within quotes; quotes of more than three lines in an indented paragraph;
  quotes in the standard non-italic font.
- Abbreviations: define at first use ("World Health Organization (WHO)"); capitals without full stops (USA); e.g., i.e., etc. lower case.
- Numbers: spell out zero to nine; figures for 10 and above; a series of numbers uses figures throughout; start a sentence with a word,
  not a figure; decimal point not comma; leading zero ("0.24 NOT .24"). Units: symbols from the SI brochure.
- Ranges: en dash, no spaces ("10-25 years", "pp. 10-65"); em dashes sparingly.
- Formulae: "Editors will not edit formulae"; the author lays them out.
- Hyperlinks: descriptive text, no bare URLs and no "click here" in the article text; format "[Descriptive text] ([Link]"; reference-list
  links follow the reference style. (This rule was reworded between W4 of 2026-09-22 and the live page.)
- Fonts: "commonly available and in an easily readable size"; underline avoided; bold or italic for emphasis sparingly; lists sparingly.
- Manuscript file format (Word, PDF, LaTeX), line spacing, line numbers, margins: **NOT FOUND** (searched S1 for "docx", "Word", "LaTeX",
  "format", "spacing", "line number"; only "a word processor's table function" appears).

## A8. Figures

Source: S1 (fetched 2026-10-20)

- "Figures, including graphs and diagrams, must be professionally and clearly presented"; the editor may ask to re-render or omit one.
- Avoid scans or images of text.
- Cite every figure in the text "in consecutive order using Arabic numerals (e.g. Figure 1, Figure 2, etc.)".
- Each figure: a descriptive main title, and an optional short legend. Examples on the page: "Figure 1: 1685 map of London." and
  "Figure 1: 1685 map of London. Note the addition of St Paul's Cathedral, absent from earlier maps."
- Each figure also needs alt text, different from the title or caption; "no longer than 10 words". Titles, legends and alt text are placed in the text
  document "either after the paragraph of their first citation, or as a list after the references".
- Give the source of the image, copyright information and authorisation, e.g. "Photo: Claude-Michel Masson. Reproduced with permission of
  the photographer."
- Text inside figures: "FS ME or a sans serif alternative (e.g Ariel, Helvetica, Verdana)" (spelling "Ariel" is on the page).
- Upload: separately, as supplementary files, "if possible in colour and at a resolution of at least 300dpi"; not more than 20 MB each; formats
  JPG, TIFF, GIF, PNG; for line drawings also the original vector file (.ai or .eps). The submission checklist words the minimum as
  "at least 150dpi (300dpi or above preferred)" and lists EPS as well.
- Maps: **NOT FOUND** — no rule on scale bar, north arrow, graticule or coordinate labels (searched S1 for "north arrow", "scale bar",
  "coordinate", "graticule", "map"; only the example caption above). Observed practice is in Part B.

## A9. Tables

Source: S1 (fetched 2026-10-20)

- Built with the word processor's table function, not tabbed text; included in the manuscript (figures are uploaded separately).
- Cited in the text in order, "Table 1, Table 2"; a descriptive title, optional legend; "The table title and legend should be placed underneath the table."
- Not allowed: rotated text; colour to denote meaning; images; vertical or diagonal lines; multi-part tables ("Table 1a", "Table 1b").
- Wide tables go landscape; if still too wide the table is split in two.
- Observation (B1, HTML): in the typeset HTML the title is displayed above the table. PDFs not checked.

## A10. References and in-text citation

Source: S1 (fetched 2026-10-20)

System: "This journal uses the Harvard system". The journal offers Zotero and EndNote style files (links on S1:
https://www.zotero.org/styles?q=jcaa and a Google Drive link for EndNote).

In-text rules (verbatim examples from the page):
- Author in the sentence: "Both Jones (2013) and Brown (2010) showed that ..."
- Author not in the sentence: "The statistics clearly show this to be untrue (Brown 2010; Jones 2013)." Multiple citations separated by a semicolon, in
  alphabetical order.
- Three or fewer authors: all listed, "(Jones, Smith & Brown 2008)"; four or more: "(Jones et al. 2008)".
- Same author and year: "(Jones 2013a; Jones 2013b)".
- Pages: after a colon: "(Brown 2004: 65; Jones 2013: 143)".
- Organisations: short form or acronym, "(ICRC 2000)".
- "Please do not include URLs in parenthetical citations."
- Every use of external information must be cited. All works in the list must be cited in the text; extra reading goes into an endnote.

Reference list rules: alphabetical by author surname; "re-type the author's name out for each entry, rather than using a long dash"; "DOIs should
be included for all reference entries, where possible".

Formats and the journal's own examples, copied from S1:

| Kind | Template | Example on the page |
|---|---|---|
| Book | Author, AA. Year. Title. Place of publication: Publisher. | Lock, GR. 2003. Using computers in archaeology: Towards virtual pasts. London: Routledge. |
| Edited book | (same, with "(eds.)") | Atkin, T. and Rykwert, J. (eds.) 2005. Structure and meaning in human settlement. Philadelphia, PA: University of Pennsylvania Museum of Archaeology and Anthropology. |
| Chapter | Author, A. Year. Title of Chapter. In: Editor, B (ed.) Title. Place of Publication: Publisher, pp. page. | Watrall, E. 2011. iAKS: A web 2.0 archaeological knowledge management system. In: Kansa, EC et al. Archaeology 2.0: New approaches to communication and collaboration. Los Angeles: Cotsen Institute of Archaeology of Press, pp. 171–184. |
| Journal article | Author, A. Year. Title. Journal Name, vol(issue): page. DOI | Kvamme, K. 2003. Geophysical surveys as landscape archaeology. American Antiquity, 68(3): 435-457. DOI: 10.2307/3557103 |
| Newspaper (print / online) | Author, A. Year. Title. Newspaper, date of publication, page. / ... [URL and last accessed date]. | Wallace, J. 2016. Laser scans unveil a network of ancient cities in Cambodia. The New York Times, 19 September [online access at https://www.nytimes.com/... last accessed 20 April 2017]. |
| Conference paper | Author, A. Year. Title. In: Editor, B (ed.). Title of Conference Proceedings. Place of publication: Publisher, pp. page. | Martín-Rodilla, P. 2015. An Empirical Approach to the Analysis of Archaeological Discourse. In: Traviglia, A (ed.). Across Space and Time Papers from the 41st Conference on Computer Applications and Quantitative Methods in Archaeology Perth, 25-28 March 2013. Amsterdam: Amsterdam University Press, pp. 319-325. |
| Grey literature | Author group. Year. Title. Place of publication: Publisher. | National Oceanic and Atmospheric Administration. 2008. NOAA's Regulation of Fishing in National Marine Sanctuaries. Washington D.C., USA: NOAA. |
| Thesis | Author, A. Year. Title. Unpublished thesis (PhD), institution. | Glover, J. 2006. The Yalahau settlement pattern survey: ... Unpublished thesis (PhD), University of California, Riverside. |
| Webpage / PDF | Author, A. Year. Title, date of publication. Available at URL [Last accessed date month year]. | Petrykowski, S. 2006. Amid regional instability ... Available at https://theantiquitiescoalition.org/... [Last accessed 20 April 2017]. |

- Software and datasets: **NOT FOUND** as a format on S1 (no template). What the published articles do is in Part B (B4, B5, B6).
- Italics are not visible in the copied text above; in the published articles the journal or book title is italic (B0-B8).
- Difference between the page's examples and the typeset articles: the page's examples have no quotation marks around article or chapter titles;
  most typeset articles in Part B put single quotation marks around article and chapter titles, write "and" before the last author (no
  comma), print initials without full stops, and print the DOI as a full https://doi.org URL. Authors' manuscripts evidently reach print in
  slightly different forms (see B-table).

## A11. Revision and review process

Source: S2 (fetched 2026-10-20)

- "In the case of minor revisions required, we expect the author to resubmit once they have addressed the reviewers' comments but don't expect
  an answer to the reviewers' comments in a separate document."
- "In the case of major revisions required, we expect the author to resubmit as well as answer the reviewers' comments point by point in a
  separate document."
- "Upon resubmission with major revision, the editor may decide to contact the reviewers to check whether they think their comments were
  appropriately addressed. This may lead to a second request for revision before proceeding with the editing process." Further review cycles are possible.
- Review: normally 2 reviewers, deadline three weeks after accepting; whole review period "around four to six weeks".
- After acceptance: copyediting about two weeks; typesetting and proofing about three weeks; proofs allow only typesetting corrections,
  "no substantial changes can be made to the paper during this stage". DOI is assigned at publication.
- **Tracked-changes or marked-up copy: NOT FOUND.** Searched S1, S2, S3, S4 and S5 for "track", "marked", "highlight", "clean copy", "revised version",
  "resubmi", "revision". The only revision rules are the three above. The decision letter or the submission system may say more.
- The journal has no rule on how a response letter is laid out beyond "point by point".

## A12. APC and waiver — exact text

Source: S1 https://journal.caa-international.org/about/submissions, section "Publication Fees", and S3 https://journal.caa-international.org/en/waiver-policy (both fetched 2026-10-20)

**Currency: pounds sterling (GBP).** The raw HTML of S1 carries the byte sequence C2 A3 (UTF-8 for U+00A3, the pound sign) immediately before "651.00" in
all four rows. The pound sign also appears in the text of the Wayback copies. (An earlier fetch that garbled it was wrong about the symbol, not about the currency.)

Fee table as printed (S1, live 2026-10-20):

    Section          APC
    Research Article £651.00
    Review           £651.00
    Case Study       £651.00
    Position Paper   £651.00

Text as printed (S1):

> Articles accepted for publication will be asked to pay an Article Publication Charge (APC) to cover publication costs. This can normally be sourced from your funder or institution. This fee covers all publication costs (editorial processes; web hosting; indexing; marketing; archiving; DOI registration etc) and ensures that all of the content is fully open access. This approach maximises the potential readership of publications and allows the journal to be run in a sustainable way.
>
> Many institutions have funds available to support open access publications by their staff, therefore we ask that you contact the relevant body to cover the APC.
>
> If you do not know about your institution's policy on open access funding, please contact your departmental/faculty administrators and institution library, as funds may be available to you.
>
> If published, you will receive an APC request email along with information on how payment can be arranged.
>
> Tax will be added to all fees charged, when applicable (includes VAT/Sales tax or any other applicable taxes).
>
> We are offering a 30% APC discount to any reviewer who has completed a review for the journal within the 12 months prior to submission. We simply ask that the review is completed and received on time (including any agreed extensions) and of reasonable detail. If you would like to take advantage of this offer, then please mention your review (ideally article title and date) in your cover letter when submitting to the journal. Should the submission be accepted for publication, the APC discount will be applied.
>
> Waiver Policy
> Please see here for details of our waiver policy and JCAA Publication fund.
> Editorial decisions are made independently from the ability to pay the APC. Waiver requests must be received as part of the submission information (e.g. in the cover letter)
>
> Institutional Agreements: ... The agreement removes the author from any of the payment process whilst also providing a 10% discount on the APC.

Submission checklist item (S1): "The author(s) agree to the payment terms detailed on the journal website, which will be applied if this submission is accepted for publication by the journal. Any waiver request must be made at the time of submission via the Comments to the Editor (below). Unless a waiver is granted by the journal, in writing, then the author(s) accepts that an Article Processing Charge (APC) may be invoiced post-acceptance."

Waiver policy page (S3), first paragraph as printed:

> If you do not have funds available to pay the APC (e.g. because your institution or funder will not cover the fee) then we may be able to offer a discount or full waiver. Please ensure that you contact the editors as early as possible should you need to discuss waiver options or the APC in general. Editorial decisions are made independently from the ability to pay the APC. Waiver requests must be received either before a submission or as part of the submission information (e.g. in the cover letter).

S3 also describes the JCAA Publication Fund (CAA International): for papers already accepted; the applicant "must be a current member of CAA, and must have been
a member for at least 2 years during the last 4, including the current year"; the applicant must show no other funding is available to the applicant or co-authors;
"a maximum of five papers per year"; application via an online form plus statements of non-eligibility; results "generally ... within 3 weeks"; if approved, CAA
International pays the APC. Contact given: journal@caa-international.org.

**APC history [Wayback], all four article types identical at each date:**

| Snapshot | Source | APC per article |
|---|---|---|
| 2025-06-01 | W1 | £570.00 |
| 2026-01-01 | W2 | £593.00 |
| 2026-08-11 (day of the P2 resubmission) | W3 | £611.00 |
| 2026-09-22 | W4 | £651.00 |
| live, 2026-10-20 | S1 | £651.00 |

The £593 figure in the project's earlier notes matches W2 (January 2026). The fee has risen three times in sixteen months. Which fee applies to P2 (the
fee at submission, at acceptance, or at invoice) is **NOT FOUND** on any page fetched; the checklist says the terms are "applied if this submission is
accepted".

## A13. NOT FOUND list (with where I looked)

| Item | Where searched (all 2026-10-20) |
|---|---|
| Tracked-changes or marked-up copy for revisions | S1, S2, S3, S4, S5 (terms: track, marked, highlight, clean copy, revised version, resubmi, revision) |
| Rule on cross-referencing sections ("Section 2.1") | S1 (term: section) |
| Map conventions (scale bar, north arrow, graticule, coordinates, CRS) | S1 (terms: north arrow, scale bar, coordinate, graticule, map) |
| Glossary or "key terms" box rule | S1 (terms: glossary, abbreviation, box) |
| Manuscript file format and layout (Word/LaTeX, spacing, line numbers) | S1 (terms: docx, Word, LaTeX, RTF, format, spacing, line number, margin) |
| Reference template for software, datasets, code | S1 reference section (no such template) |
| How the word count is made | S1, S2, S4 |
| Which APC applies to a manuscript already in review | S1, S3, W1-W4 |
| Ubiquity Press AI policy; authorship and competing-interests guideline pages | https://www.ubiquitypress.com/ai-policy and https://ubiquitypress.com/site/ai-policy/ returned HTTP 403 (curl and WebFetch); authorship and competing-interests pages not attempted |
| The journal's article-type of P2 and the decision type (minor or major) | not on any public page; see the decision letter |

---

# PART B — Eight recent JCAA research articles (2024–2026)

## B0. Selection and method

Selection: from the article listing (S6 pages 1-4, 160 articles in total, fetched 2026-10-20) I chose the Research Articles from 2024-2026 closest to
P2's topic (predictive modelling, spatial statistics, validation, machine learning, a negative or limiting result). Two are site-prediction or
site-detection papers (B1, B4), two are spatial-statistics or route-modelling papers (B2, B3), one is a modelling paper with a sensitivity analysis (B6),
one is a tool paper with a three-way validation (B5), and two are ML papers, one of which reports a replication failure (B7, B8).
All eight are labelled "Research Article" on the listing. No Case Study or Position Paper is included; this is a gap.

Fetch: each article page was downloaded as HTML on 2026-10-20 (`art_NNN.html` in this folder); headings (h2-h4), captions, statements and
reference lists were read from the HTML. Maps were opened from the figure PNGs stored in `img_NNN/`.

Word counts: "main" = text of the main block from the first numbered heading to the reference list, including tables, captions and end statements;
"prose" = paragraphs only, without tables and captions; "with refs" = main + reference list. These are my counts of the HTML text, not the journal's.

## B1. Abou Diwan (2025) — site-location prediction, Lebanon

- Reference: Abou Diwan, G. 2025. Probabilistic Models for Predicting Archaeological Site Locations in Marj Bisri (Southern Lebanon): Comparing
  Frequency Ratio and Shannon's Entropy. *Journal of Computer Applications in Archaeology*, 8(1): 59–77. DOI: 10.5334/jcaa.150.
- URL: https://journal.caa-international.org/en/articles/10.5334/jcaa.150 (fetched 2026-10-20). DOI link https://doi.org/10.5334/jcaa.150.
- Type: Research Article. Submitted 2024-02-02, accepted 2025-01-12, published 2025-02-05.
- Length: 19 pages; main about 5,600 words (prose about 4,500; with refs about 7,400); abstract 117 words; 7 figures, 5 tables; 66 references; 6 keywords.
- Top-level headings: 1 Introduction · 2 Description of the Study Area · 3 Material and Methods (3.1 Archaeological and geospatial data; 3.2 Model
  variables; 3.3 Investigating Archaeological site Distributions in Relation to geo-Environmental Categories; 3.4 Methods, with 3.4.1 and 3.4.2) ·
  4 Results and Discussion (4.1 Validation) · 5 Conclusion · Data Accessibility Statement · Competing Interests · References.
  Numbers without a full stop ("1 Introduction").
- Section cross-references: **none by number**; forward pointers are "(see below)". Example: "...archaeological evidence indicating continuous human
  occupation as early as the Palaeolithic Period (see below)." Figures and tables are cited as "(Figure 1)", "(Table 1)", "Figures 3 and 4".
- Reference-list entries (copied; italics shown with asterisks):
  - Akgun, A, Dag, S and Bulut, F. 2008. 'Landslide susceptibility mapping for a landslide-prone area (Findikli, NE of Turkey) by likelihood-frequency ratio and
    weighted linear combination models.' *Environmental Geology*, 54(6): 1127–1143. DOI: https://doi.org/10.1007/s00254-007-0882-8
  - Brandt, SA and Hassan, F. 2000. *Dams and cultural heritage management*. Cape Town: World Commission on Dams.
  - Kamermans, H and Wansleeben, M. 1999. 'Predictive modelling in Dutch archaeology, joining forces.' In: Barceló, JA, Briz, I and Vila, A (eds.) *New
    Techniques for Old Times – CAA98. Computer Applications and Quantitative Methods in Archaeology*. Oxford: BAR, pp. 225–230.
- Maps (Figures 1 and 3, opened): Fig. 1 is a satellite-image base map with the study area outlined, a legend box, a small locator inset of Lebanon,
  a compass rose, a scale bar in km, and a printed line "Spatial Reference: WGS 1984 UTM Zone 36N" — no graticule. Fig. 3 (frequency-ratio map) is a
  five-class colour overlay on a grey hillshade, with projected easting and northing labels (UTM, e.g. 730000 to 742000, 3714000 to 3724000) along the frame, a compass
  rose, a scale bar in km and a legend.
- Glossary or key-terms box: none.
- Validation shown: 5-fold cross-validation with Kvamme's Gain for each fold (Table 5, t-test p 0.202 and 0.414), and a Monte Carlo comparison with 100 random site
  sets (Figures 6 and 7).
- Data statement (verbatim): "The data used for this article can be found online at https://doi.org/10.5281/zenodo.14618433." Competing interests: "The author has no
  competing interests to declare." No AI declaration section, no funding or contributions section.
- Typical claim size: a desk-based tool for planning surveys in a 48 km² valley from 67 recorded sites; the models are "reliable for predicting archaeologically significant
  areas" by Kvamme's Gain, not claims about past behaviour.

## B2. Vilbig, Sagan and Bodine (2026) — site concentration and visibility, Greater Cahokia

- Reference: Vilbig, JM, Sagan, V and Bodine, C. 2026. Constructing Greater Cahokia: Site Concentrations and Visibility Surrounding America's First City.
  *Journal of Computer Applications in Archaeology*, 9(1): 43–65. DOI: 10.5334/jcaa.203.
- URL: https://journal.caa-international.org/en/articles/10.5334/jcaa.203 (fetched 2026-10-20).
- Type: Research Article. Submitted 2025-01-28, accepted 2026-01-05, published 2026-02-23.
- Length: 23 pages; main about 10,200 words (prose about 9,800; with refs about 13,500); abstract 243 words; 9 figures, 2 tables; 137 references; 7 keywords (the guideline says "up to six").
- Top-level headings: 1. Introduction · 2. Material and Methods (2.1 Background [2.1.1, 2.1.2]; 2.2 Data [2.2.1-2.2.3]; 2.3 Methods [2.3.1-2.3.4]) · 3. Results (3.1 First-Order
  Analyses; 3.2 Visibility; 3.3 Analytic Synthesis) · 4. Discussion · 5. Conclusions · Appendix (1. Notes on LiDAR Data; 2. Notes on Digital Terrain Model Creation) ·
  Acknowledgements · Competing Interests · References. Numbers followed by a full stop.
- Section cross-references (lower-case "section", number with sub-levels): "...within the mound groups' viewshed (discussed in section 2.3.2.) for both habitations (638 points)
  and mounds (85 points)." and "As mentioned in section 3.3.1, Emerald served a unique function, as a religious pilgrimage..."
- Reference-list entries:
  - Alt, S and Pauketat, T. 2011. 'Why wall trenches?' *Southeastern Archaeology*, 30: 108–122. DOI: https://doi.org/10.1179/sea.2011.30.1.008
  - Cressie, N. 2015. *Statistics for spatial data*. New York, NY: John Wiley & Sons, Inc.
  - Alt, S. 2011. 'Making Mississippian at Cahokia'. In: *The Oxford Handbook of North American Archaeology*. England: Oxford University Press. pp. 497–508. DOI: https://doi.org/10.1093/oxfordhb/9780195380118.013.0041
- Maps (Figures 1, 3 and 4, opened): Fig. 1 is a grey locator map with an overview inset of North America and a scale bar in km, no north arrow, no graticule. Fig. 3 is a DTM colour
  ramp (31.93 to 226.346 m) laid on satellite imagery with sites as symbols, labelled mound groups, a legend box, a small north arrow and a scale bar (0 to 10 km). Fig. 4 shows
  kernel-density surfaces in two panels (a, b) over a satellite image, each with a legend, north arrow and scale bar. No coordinate labels or graticule.
- Glossary or key-terms box: none.
- Statements: Acknowledgements and "The authors have no competing interests to declare." No data-accessibility, funding, contributions or AI section in the HTML (site data
  came from the Illinois State Historic Preservation Office, thanked in Acknowledgements).
- Typical claim size: statistical significance of clustering and above-random visibility, read as "suggestive of shared and coordinated decision making".

## B3. Mennenga and Frederiks (2026) — route reconstruction from grave mounds

- Reference: Mennenga, M and Frederiks, PL. 2026. Many Paths Lead Through the Bog: Reconstructing Prehistoric Networks and Routes at the Southern North Sea Coast Based
  on Neolithic and Bronze Age Grave Mounds. *Journal of Computer Applications in Archaeology*, 9(1): 293–311. DOI: 10.5334/jcaa.226.
- URL: https://journal.caa-international.org/en/articles/10.5334/jcaa.226 (fetched 2026-10-20).
- Type: Research Article. Submitted 2025-05-26, accepted 2026-07-08, published 2026-09-18.
- Length: 19 pages; main about 8,600 words (prose about 8,200; with refs about 10,500); abstract 258 words; 12 figures, 2 tables; 60 references; 6 keywords.
- Top-level headings (**unnumbered**): Introduction (Working area; Archaeological background; Burial mounds as landmarks on prehistoric routes; Data basis; The landscape and its
  transformation) · Methods (Data preparation; Models; RNG model; Betweenness) · Network, archaeology and landscape (Plausibility; Continuity) · Discussion · Competing
  Interests · References. So numbered headings are not enforced in print.
- Section cross-references: no numbered references (headings are unnumbered); forward pointers "(see below)". Example: "...a dynamic landscape consisting of Geest (see below), peatland
  and coastal marshland".
- Reference-list entries:
  - Behrens, A and Mennenga, M. 2025. 'Trichterbecherzeitliche Großsteingräber im Elbe-Weser-Dreieck – Räumliche und gesellschaftliche Analysen anhand der Grabarchitektur'.
    *Siedlungs- und Küstenforschung im südlichen Nordseegebiet*, 48: 9–86.
  - Bourgeois, Q. 2013. *Monuments on the horizon: The formation of the barrow landscape throughout the 3rd and 2nd millennium BC*. Leiden: Sidestone Press. PID: hdl.handle.net/1887/20381
  - Bang J. 2013. 'The Route to a History of the Cultural Landscape: a Danish Record of Prehistoric and Historic Roads, Tracks and Related Structures'. In: Bergerbrant, S and Sabatini, S
    (eds.) *Counterpoint: Essays in Archaeology and Heritage Studies in Honour of Professor Kristian Kristiansen*. Oxford: Information Press. pp. 703–715.
  - (A few entries lack the quotation marks, e.g. Bakker 2008, so the typeset style is not uniform.)
- Maps (Figures 1 and 7, opened): Fig. 1 shows grave mounds as dots on flat categorical landscape colours (peatland, Pleistocene sand/Geest, coastal marshland), two locator insets
  (north-west Europe; Germany with the study area shaded red), numbered area labels in the corner, a legend, and a 5 km scale bar at bottom right. **No north arrow, no graticule, no hillshade.**
  Fig. 7 draws network edges classed by betweenness (five-class line legend in a box) on the same base.
- Glossary or key-terms box: none.
- Statements: "The authors have no competing interests to declare." No separate data-accessibility, funding or AI section in the HTML.
- Typical claim size: hedged and methodological — it asks "whether plausible communication routes can be reconstructed" and checks the network against bog trackways and non-burial
  sites (Figures 8-11).

## B4. Landauer, Maddison, Fontana and Posluschny (2025) — deep-learning hillfort detection

- Reference: Landauer, J, Maddison, S, Fontana, G and Posluschny, AG. 2025. Archaeological Site Detection: Latest Results from a Deep Learning Based Europe Wide Hillfort Search.
  *Journal of Computer Applications in Archaeology*, 8(1): 42–58. DOI: 10.5334/jcaa.178.
- URL: https://journal.caa-international.org/en/articles/10.5334/jcaa.178 (fetched 2026-10-20).
- Type: Research Article. Submitted 2024-08-20, accepted 2024-12-04, published 2025-01-31.
- Length: 17 pages; main about 5,500 words (prose about 4,900; with refs about 6,600); abstract 234 words; 14 figures, 6 tables; 39 references; 5 keywords.
- Top-level headings: 1 Introduction and previous work · 2 AI architecture and landscape-wide search (2.1-2.5) · 3 Case studies and the human verification process (3.1 Chilterns Hills and Northern
  England; 3.2 Hesse, Germany; 3.3 Molise, central Italy; each with 3.x.1 Verification method and 3.x.2 Results) · 4 Discussion and outlook · Data Accessibility Statement · Competing Interests · References.
- Section cross-references (capital "Section", plus a roadmap sentence at the end of the introduction): "Following this introduction, Section 2 outlines our method and workflow for automated hillfort
  detection ... Section 3 presents the results from our case studies conducted in England ... Section 4 provides a discussion of our findings". Also "(as described in Section 3 below)".
- Reference-list entries:
  - Berganzo-Besga, I, Orengo, HA, Lumbreras, F, Alam, A, Campbell, R, Gerrits, PJ, de Souza, JG, Khan, A, Suárez-Moreno, M, Tomaney, J, Roberts, RC and Petrie, CA. 2023. 'Curriculum learning-based
    strategy for low-density archaeological mound detection from historical maps in India and Pakistan'. *Sci Rep*, 13: 11257. DOI: https://doi.org/10.1038/s41598-023-38190-x
  - Maddison, S. 2022. 'Using Atlas data: the recognition and significance of selected clusters of hillforts in Britain'. In: Ralston, I and Lock, G (eds.) *The Atlas of Hillforts of Britain and Ireland*.
    Edinburgh: Edinburgh University Press.
  - Chilterns AONB. 2023. *Beacons of the Past LiDAR Portal*. Available at: https://chilternsbeacons.org/wp/ [Last accessed 14 January 2025].
  - Python Software Foundation. 2023. *Python Language Reference*, version 3.10. Available at: http://www.python.org [Last accessed 14 January 2025]. (software cited as a web/grey item)
  - (No clean book entry was matched in this list.)
- Maps (Figures 7 and 8, opened): Hesse overlaid with known hillforts (Fig. 7) and candidate detections (Fig. 8) on a hillshade with hypsometric tint; a scale bar (0, 50, 100 km) at top left and a north arrow in a circle at
  bottom left; no graticule, no coordinate labels. Data sources (e.g. "DEM: HVBG", "DEM: DEFRA") are given in the captions, and vignette sizes are stated in captions.
- Glossary or key-terms box: none.
- Data statement: LiDAR repositories and a LiDAR data table (Table 6) inside the Data Accessibility Statement. Competing interests: none.
- Typical claim size: performance and workflow claim with the weak number stated in the abstract (F1 34–38%) and "human verification remains critical"; new hillforts are "potential" or "promising candidates".

## B5. Červík, Waersegers, Zischg and Lieskovský (2026) — throwshed tool for conflict archaeology

- Reference: Červík, T, Waersegers, Y, Zischg, AP and Lieskovský, T. 2026. GIS-Based Throwshed Analysis for Conflict Archaeology. *Journal of Computer Applications in Archaeology*, 9(1): 198–224. DOI: 10.5334/jcaa.217.
- URL: https://journal.caa-international.org/en/articles/10.5334/jcaa.217 (fetched 2026-10-20).
- Type: Research Article. Submitted 2025-03-21, accepted 2026-04-15, published 2026-05-28.
- Length: 27 pages; main about 10,500 words (prose about 9,500; with refs about 12,500); abstract 235 words; 21 figures, 9 tables; 79 references; 6 keywords.
- Top-level headings: 1. Introduction · 2. Methods (2.1-2.11) · 3. Results · 4. Validation (4.1 Match with the data modelled in the existing literature; 4.2 Match with experimental data from firing tables; 4.3 Match
  with archaeological data) · 5. Discussion · 6. Conclusion · Funding Statement · Competing Interests · Data Accessibility Statement · Author Contributions · References.
- Section cross-references (lower-case "section" mid-sentence, capital at sentence start, "sections 4.2 and 4.3" for two): "Validation sections 4.2 and 4.3 require the muzzle velocities of the cannons";
  "As section 2.11 indicates, the drag coefficients were derived from the graphs"; "Section 4.3 may serve as a foundation for following use cases."
- Reference-list entries:
  - Archer, L, Neal, JC, Bates, PD and House, JI. 2018. 'Comparing TanDEM-X Data With Frequently Used DEMs for Flood Inundation Modeling'. *Water Resources Research*, 54(12): 10205–10222. DOI: https://doi.org/10.1029/2018wr023688
  - Bellamy, C. 1990. *The evolution of modern land warfare: Theory and practice*. London: Routledge.
  - Elsaadany, A and Wen-jun, Y. 2014. 'Accurate trajectory prediction for typical artillery projectile'. In: Proceedings of the 33rd Chinese Control Conference. Nanjing, China: IEEE, pp. 6368–6374. DOI: https://doi.org/10.1109/chicc.2014.6896037
  - Collins, AR. 2015. *Smooth Bore Cannon Ballistics. MISCELLANY: Miscellaneous technical articles by Dr. A. R. Collins*, 2015. Available at https://2015fallhw.github.io/arcidau/CannonBallistics.html [Last accessed 18 January 2025]. (online technical note)
- Maps (Figures 5 and 17, opened): Fig. 5 is a hillshade with hypsometric tint, projected coordinate tick labels on the top and right edges (e.g. -380000, -370000; -1200000, -1210000), a north arrow in a white box at top
  left, a scale bar (0 to 2 km), a country locator inset (Slovak Republic), a legend, a source line "Source of ALS products: ÚGKK SR" bottom left and the CRS line "CRS: S-JTSK/JTSK03 (EPSG: 8353), BpV" bottom right; two
  nested boxes labelled A and B mark detail areas. Fig. 17 has a historical map (A) next to a DTM (B), each with a north arrow box, a 500 m scale bar, edge coordinate labels, a source line and the CRS (LUREF, EPSG: 2169). This is
  the most fully annotated map style among the eight.
- Glossary or key-terms box: none (the acronym KOCOA is spelled out in the abstract and the introduction).
- Statements: Funding statement with grant numbers (Swiss National Science Foundation; Slovak Research and Development Agency). Data statement: the Python tool "is openly available at a public GitHub repository"
  with the licence (CC BY-NC-SA 4.0) and a note that some archaeological data "are not currently accessible because they appear in another archaeological article yet to be published". Author contributions given by initial.
- Typical claim size: a method and tool paper; claims "validated with current, historical, and archaeological data" in three steps and corrects a classical approach.

## B6. Triozzi and McClure (2026) — herd demography simulation

- Reference: Triozzi, N and McClure, SB. 2026. Evaluating Livestock Culling Practices Using Population Projection Models in Neolithic Dalmatia. *Journal of Computer Applications in Archaeology*, 9(1): 225–244. DOI: 10.5334/jcaa.257.
- URL: https://journal.caa-international.org/en/articles/10.5334/jcaa.257 (fetched 2026-10-20).
- Type: Research Article. Submitted 2025-11-26, accepted 2026-04-21, published 2026-06-10.
- Length: 20 pages; main about 7,400 words (prose about 6,400; with refs about 9,400); abstract 186 words; 12 figures, 7 tables; 71 references; 5 keywords; four supplementary files with DOIs (10.5334/jcaa.257.s1 to .s4).
- Top-level headings: 1. Introduction · 2. Methods (2.1 Mortality and Fertility Parameters; 2.2 Survivorship and Culling Strategies; 2.3 Population Projections; 2.4 Culling Optimization Algorithm) · 3. Results (3.1 Initial Projections; 3.2 Optimizing Culling
  Rates; 3.3 Sensitivity Analysis) · 4. Discussion: Risk and Optimization of Culling Rates · 5. Conclusion · Additional Files · Funding Statement · Competing Interests · Data Accessibility Statement · Author Contributions · References.
- Section cross-references ("Section" capital, inside parentheses): "...but reduce the female culling rates by 85% (see Section 2.3)." and "...in a sensitivity analysis (see Section 3.3)." Supplements are cited as "(see S2)".
- Reference-list entries:
  - Awemu, EM, Nwakalor, LN and Abubakar, BY. 1999. 'Environmental influences on preweaning mortality and reproductive performance of Red Sokoto does'. *Small Rumin. Res.*, 34: 161–165. DOI: https://doi.org/10.1016/S0921-4488(99)00058-9
  - Hadjikoumis, A. 2017. 'Age-at-death in traditional Cypriot sheep and goat husbandry'. In: Rowley-Conwy, P, Serjeanston, D and Halstead, P (eds.) *Economic Zooarchaeology*. Oxbow Books. DOI: https://doi.org/10.2307/j.ctt1s4751q.20
  - Triozzi, NP. 2025. Age-at-Death Analysis of Caprine mandibles from Neolithic Sites in Dalmatia [Data set]. *Zenodo*. DOI: https://doi.org/10.5281/zenodo.17697101 (dataset format)
  - Lesnoff, M. 2024. *mmage: A R package for sex-and-age population matrix models*. (software; no URL)
- Maps: none (no map in the article).
- Glossary or key-terms box: none.
- Statements: funding with grant number; Data Accessibility Statement with a GitHub repository and a Zenodo DOI for the age-at-death data; Author Contributions by initial.
- Typical claim size: model-based and qualified: some strategies "decimate herds", "flexible culling decisions are necessary ... but alone insufficient"; "we cannot assume that past herding economies employed them inflexibly".

## B7. Courtenay, Vanderesse, Doyon and Souron (2024) — a replication that weakens earlier claims

- Reference: Courtenay, LA, Vanderesse, N, Doyon, L and Souron, A. 2024. Deep Learning-Based Computer Vision Is Not Yet the Answer to Taphonomic Equifinality in Bone Surface Modifications.
  *Journal of Computer Applications in Archaeology*, 7(1): 388–411. DOI: 10.5334/jcaa.145.
- URL: https://journal.caa-international.org/en/articles/10.5334/jcaa.145 (fetched 2026-10-20).
- Type: Research Article. Submitted 2023-12-20, accepted 2024-10-10, published 2024-12-10.
- Length: 24 pages; main about 11,900 words (prose about 10,600; with refs about 16,300); abstract 203 words; 5 figures, 6 tables; 153 references; 6 keywords; two supplementary files.
- Top-level headings: 1. Introduction · 2. Material and Methods (2.1 Datasets; 2.2 Computer Vision and Image Analysis; 2.3 Deep Learning Algorithms; 2.4 Algorithm training and evaluation; 2.5 Software) · 3. Results (3.1 Dataset Quality; 3.2 Classification
  Results) · 4. Discussion (4.1 DL-based CV is only as powerful as the datasets provided; 4.2 Not so intelligent artificial intelligence; 4.3 Some key notes on sample size and the transferability of learned features in CNNs) · 5. Conclusions · Data Accessibility
  Statement · Additional Files · Acknowledgements · Funding Information · Competing Interests · Author Contributions · References.
- Section cross-references (lower-case "section"): "In light of previous statements in section 4.2, it is also important to consider that ImageNet contains ..." (only one numbered cross-reference found). In-text citations here use a comma before the year, "(Byeon et al., 2019)", unlike the other seven.
- Reference-list entries:
  - Abellán, N, Jiménez-García, B, Aznarte, J, Baquedano, E and Domínguez-Rodrigo, M. 2021. 'Deep learning classification of tooth scores made by different carnivores: achieving high accuracy when comparing African carnivore taxa and testing the hominin shift in the balance
    for power'. *Archaeological and Anthropological Sciences*, 13: 31. DOI: https://doi.org/10.1007/s12520-021-01273-9
  - Binford, LR. 1981. *Bones: Ancient Men and Modern Myths*. New York: Academic Press Inc.
  - Deng, J, Berg, AC, Li, K and Fei-Fei, L. 2010. What does classifying more than 10,000 image categories tell us? In: Daniilidis, K, Margos, P and Paragios, N (eds.) Proceedings of the 11th European Conference on Computer Vision. Heidelberg: Springer, 71–84.
    DOI: https://doi.org/10.1007/978-3-642-15555-0_6
- Maps: none.
- Glossary or key-terms box: none.
- Statements: Data Accessibility Statement (public datasets, GitHub code, figshare DOI); Funding Information with grant numbers; Author Contributions in CRediT-like terms; Acknowledgements thank reviewers "over four round of review" [sic] — i.e. this paper went through four review rounds.
- Typical claim size: a **negative or limiting result stated plainly** (the title says so): earlier high classification rates were "overly optimistic"; "extreme caution is advised"; the authors add that the message is not that the method "cannot be applied" but that it should not be applied until fundamental concerns are resolved.
  This is the closest precedent in the journal to a correction-of-evaluation paper.

## B8. Kayıkcı, Berganzo-Besga and Barceló (2026) — classification with small datasets

- Reference: Kayıkcı, D, Berganzo-Besga, I and Barceló, JA. 2026. Archaeological Classification of Small Datasets Using Meta- and Transfer Learning Methods: A Case Study on Hittite Stele Fragments.
  *Journal of Computer Applications in Archaeology*, 9(1): 1–30. DOI: 10.5334/jcaa.196.
- URL: https://journal.caa-international.org/en/articles/10.5334/jcaa.196 (fetched 2026-10-20).
- Type: Research Article. Submitted 2025-01-08, accepted 2025-12-03, published 2026-01-30.
- Length: 30 pages; main about 13,500 words (prose about 11,600; with refs about 16,300); abstract 279 words; 7 figures, 12 tables; 105 references; 6 keywords; two supplementary files (appendices).
- Top-level headings: 1. Introduction · 2. Materials and Methods (2.1 Data Selection; 2.2 Enhanced Dataset Implementation; 2.3 Methodological Overview; 2.4 Implementation and parameter setting; 2.5 Validation procedure) · 3. Discussion (3.1 Preliminary results with the initial
  training and testing data set; 3.2 ...; 3.3. Human-based learning and Machine Deep Learning; 3.4 Additional validation) · 4. Conclusion · Additional Files · Acknowledgements · Declaration of Generative AI and AI-Assisted Technologies in the Writing Process · Competing Interests ·
  References. (There is no section called Results; results sit under "Discussion".)
- Section cross-references: uses **"Chapter"**, not "Section": "The paper is structured as follows: Chapter 1 provides an overview of the research objectives ...; Chapter 2 describes the materials and gives an overview of the methodologies employed; Chapter 3 presents the analytical results; and Chapter 4 discusses ..."
- Reference-list entries:
  - Anwar, H, Anwar, S, Zambanini, S and Porikli, F. 2021. 'Deep ancient Roman Republican coin classification via feature fusion and attention'. *Pattern Recognition*, 114: 107871. DOI: https://doi.org/10.1016/j.patcog.2021.107871
  - Alpaydin, E. 2021. *Machine learning*. Cambridge, MA: The MIT Press.
  - Barceló, JA. 1995. 'Back-propagation algorithms to compute similarity relationships among archaeological artefacts'. In: Wilcock, J and Lockyear, K (eds.) *Computer Applications and Quantitative Methods in Archaeology: 1993*. Oxford: Tempus Reparatum, pp. 165–176.
- Maps: Figure 1 ("Hittite cities in the scope of the study") is a map; I did not open it.
- Glossary or key-terms box: none.
- AI declaration (verbatim, the only article of the eight with one): "Statement: During the preparation of this work, the authors (Deniz Kayikci) used CLAUDE AI/ANTHROPIC to control and fix the main code blocks in the computer programs used. After using this tool/service, the authors reviewed and edited
  the content as needed and have taken full responsibility for the content of the publication." Code location is given in the abstract (a GitHub URL); the HTML has no separate data-accessibility heading.
- Typical claim size: "preliminary results" showing that "effective classification and origin prediction are achievable with only 208 samples", compared with a human expert, with a Grad-CAM check that the models use iconographic features.

## B9. Cross-article observations (from B1-B8 only)

| Feature | What the eight articles show |
|---|---|
| Heading numbers | Numbered in 7 of 8 (with a full stop in 5: B2, B5, B6, B7, B8; without in 2: B1, B4); **unnumbered in B3**. Depth: three levels at most (e.g. 3.4.1). |
| Section cross-reference word | "Section 2.3" (B4, B6, B5 at sentence start); "section 2.3.2" (B2, B5, B7); "Chapter 1" (B8); "(see below)" only (B1, B3). No single house form. |
| Cross-referencing figures and tables | "Figure 1", "Table 1" with capital letters; "Figures 3 and 4" for two. |
| Abstract length | 117-279 words (limit 300). |
| Keywords | 5-7 (B2 has 7). |
| Endnotes ("Notes") | None of the eight has a Notes section. |
| Statements present | Competing interests: 8 of 8. Data Accessibility Statement: 5 of 8 (B1, B4, B5, B6, B7). Funding: 3 of 8 (B5, B6, B7). Author contributions: 3 of 8 (B5, B6, B7). Acknowledgements: 3 of 8 (B2, B7, B8). AI declaration: 1 of 8 (B8). |
| In-text citation | "(Author Year)", "(A & B Year)" or "(A and B Year)" (both seen), "et al." for 4 or more, pages after a colon "(Mennenga 2017: 93)". B7 uses a comma before the year. |
| Reference style | Surname + initials without full stops; "and" before the last author; year; article and chapter titles in single quotation marks (most entries); italic journal or book title; DOI as full URL; book "Place: Publisher". |
| Glossary or key-terms box | None in any of the eight. |
| Maps | Scale bar in every map seen; north arrow in most (not B3, and not B2 Fig. 1); locator inset in B1, B2 (Fig. 1), B3, B5 (B4 Fig. 6 is captioned as a locator but was not opened); coordinate labels only in B1 (Fig. 3) and B5; hillshade or DTM base in B1, B2, B4, B5; CRS or source line in B1, B4 (in captions), B5. B6, B7 have no maps; B8 Fig. 1 not opened. |
| Length vs limit | Four of eight exceed 11,500 words in my count including references (B2, B5, B7, B8); the lower 8,500 / 5,000 limits were on the journal page during their review and acceptance (A2). |
| Supplementary files | B6, B7, B8 have "Additional Files" with DOIs. |

---

# PART C — What P2 must match (derived only from Parts A and B)

1. **Type and length.** Confirm which article type P2 was submitted as (not on any public page). The limit on 2026-10-20 is 11,500 words for all four types, **including references**; it was 8,500 (Research Article, Position Paper) or 5,000 (Review, Case Study) until at least 2026-09-22. Count the revised file against 11,500 and keep a margin (A2).
2. **Abstract and keywords.** Abstract at most 300 words under the heading "Abstract"; at most six keywords placed below it, and the same abstract and keywords in the submission metadata (A3). Published abstracts run 117-279 words.
3. **Title page / blinding.** The manuscript file carries only the title and abstract if it is to stay blind; author details and full forenames go in the online form; blinding is the author's choice and responsibility (A3).
4. **Headings.** Numbered headings, three levels at most (1, 1.1, 1.1.1), each under 75 characters; title-case for the title and first-level headings, sentence case below; an introduction readable by non-specialists, then methods, results, discussion, conclusion (A4). Published articles also have unnumbered headings (B3), so numbering is the instruction, not a hard check.
5. **Section cross-references.** The journal has no rule (A13). Pick one form, e.g. "Section 2.3", apply it everywhere, and **re-check every cited section number after the revision renumbers the sections**. Observed forms: "Section 2.3", "section 2.3.2", "Chapter 1", "(see below)" (B9).
6. **Statements and their order.** Acknowledgements, funding (grant provider and number), competing interests ("The author(s) has/have no competing interests to declare." if none), authors' contributions, a Data Accessibility Statement before the references, Notes (if any) before the references, and the AI declaration at the end of the manuscript (A4, A6). Published articles vary in the order (B9).
7. **AI use.** Declare it on submission in the form of the page's example (A5); AI cannot be an author; check the declared uses against the page's lists of acceptable and unacceptable uses, including "ghost-writing ... of the paper (methods, results, discussion, etc.)" and "summarising papers not actually read by the author". Precedent for a Claude declaration: B8.
8. **Data and code.** Put data and code in a repository that gives a DOI and has a preservation policy, label the files so a third party can use them, use an open licence, and state the location and DOI in the paper; if something cannot be shared, say so and say why (A6). B1, B6, B7 give Zenodo, GitHub and figshare locations.
9. **References, in text.** Harvard: "(Brown 2010; Jones 2013)", "(Jones, Smith & Brown 2008)" for three or fewer, "(Jones et al. 2008)" for four or more, "(Brown 2004: 65)" for pages, "(Jones 2013a; Jones 2013b)", no URLs inside the brackets, organisations by short form (A10).
10. **References, list.** Alphabetical; the author's name re-typed for every entry; DOI wherever one exists; formats from A10 (book, chapter, article, conference paper, grey literature, thesis, webpage). For software or datasets there is no template (A13); use the form seen in B5-B6 (author, year, title, [Data set], repository, DOI) and say so in the cover letter if needed. Also match the typeset habits in B9 (single quotes on article and chapter titles, "and" before the last author, DOI as full URL) only if the PI wants the reference list to look like the published ones; the page's own examples differ slightly.
11. **Figures.** Cite in order as "Figure 1"; each with a descriptive title (legend optional), alt text of about 10 words or fewer, placed after the first-citation paragraph or listed after the references; upload separately as JPG, TIFF, GIF or PNG, preferably 300 dpi or more, under 20 MB each; vector file as well for line drawings; sans-serif text in figures; credit line for any third-party material (A8).
12. **Maps.** No stated rule (A13). The observed norm in the eight articles is a scale bar on every map, a north arrow on most, a locator inset where the study area is not obvious, a hillshade or DTM base, and a source and CRS line in the more careful ones (B5, B1); only B1 and B5 show projected coordinate labels (B9). Treat as "what a reviewer will have seen", not as a requirement.
13. **Tables.** Made with the word processor, in the manuscript, cited in order, descriptive title (placed underneath according to the page), no rotated text, no colour carrying meaning, no vertical or diagonal lines, no "Table 1a/1b" (A9).
14. **Text conventions.** Single quotation marks; numbers zero to nine in words and 10+ in figures; 0.24 not .24; SI units; acronyms spelled out at first use; consistent UK or US spelling; en dash with no spaces for ranges; endnotes instead of footnotes; descriptive hyperlink text rather than bare URLs (A7).
15. **Response to reviewers.** For major revisions a **point-by-point response in a separate document** is expected; for minor revisions it is not required. Whether a tracked-changes or marked-up copy is wanted is **NOT FOUND**; read the decision letter (A11). Be ready for a further review round (A11; B7 went through four).
16. **Waiver and APC.** The live fee is £651.00 (GBP); it was £611 on the day P2 was resubmitted and £593 in January 2026 (A12). The page says waiver requests must be "received as part of the submission information (e.g. in the cover letter)" and "Editorial decisions are made independently from the ability to pay the APC"; restating the waiver request in the revision cover letter matches that wording.
17. **Claim size.** Published predictive-modelling and ML papers state the validation used (cross-validation, Monte Carlo against random sets, held-out data), give weak numbers in the abstract (B4: F1 34-38%), hedge conclusions ("plausible", "suggestive", "preliminary"), and B7 shows the journal accepts a paper whose point is that earlier evaluation was over-optimistic. P2's corrected-protocol framing has a precedent here; nothing in A or B suggests that stronger wording would be accepted.
18. **Re-check on the day.** The word limit and the hyperlink rule changed between 2026-09-22 and 2026-10-20, and the APC changed twice in 2026. Re-fetch https://journal.caa-international.org/about/submissions before uploading and diff it against `submissions.html` in this folder.

---

## Files in this folder

- Journal pages: `submissions.html`, `about_editorialpolicies.html`, `waiver-policy.html`, `reproducibility.html`, `about.html`, `about_editorialteam.html` (unused), `articles.html`, `articles_p2.html`, `articles_p3.html`, `articles_p4.html`, `issue_archive.html` (unused).
- Wayback copies: `wayback_20250601010526_submissions.html`, `wayback_20260101010440_submissions.html`, `wayback_20260811170005_submissions.html`, `wayback_20260922065704_submissions.html`; unusable stubs: `wayback_20260411_submissions.html`, `wayback_20260702181729_submissions.html`, `wayback_20260723214017_submissions.html`.
- Articles: `art_150.html`, `art_203.html`, `art_226.html`, `art_178.html`, `art_217.html`, `art_257.html`, `art_145.html`, `art_196.html`; map figures in `img_150/`, `img_203/`, `img_226/`, `img_178/`, `img_217/`.

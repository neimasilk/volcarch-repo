# P8 — Oceanic Linguistics: conditional acceptance and revision plan

**Updated:** 2026-10-05 (evening) · **Line:** 04 · **MS:** OL-03-2026-11 · **State:** ⏳ revision pending (no stated deadline) — **analyses and number audit done; seven PI decisions open**
**→ Work plan, verified facts per point, corrected numbers: [`REVISION_WORKPLAN.md`](REVISION_WORKPLAN.md)** (evidence: `experiments/E227_*`, `experiments/E228_*`). This file remains the ledger of what the reviewers asked.
**Checked against the original email and the annotated PDF on 2026-10-06:** all 13 comments of reviewer 1 (plus the opening remark) and all 24 highlights of reviewer 2 are in §4–§5; five details added in `REVISION_WORKPLAN.md` §10.1 (among them a direct question inside R1-12).
**Owner of prose:** the PI (gate G16). This file holds the ledger, the evidence and the analysis tasks — not manuscript text.

> The repo is **public**. Reviewer comments are **paraphrased** here and the personal portal link from the decision
> email is **not recorded**. The decision email and the annotated reviewer PDF stay in the PI's mailbox
> (message "Decision Letter for OL-03-2026-11", 2026-10-05 14:17 WIB, attachment "Reviewer 2 Review Attachment 1.pdf").

---

## 1. Decision

| Item | Fact |
|---|---|
| Decision | **Accepted pending revisions** (conditional). Editor Sander Adelaar: accepted for publication once the editors are satisfied with the suggested revisions. **Not a final acceptance.** A formal letter of acceptance, conditional on the revisions, is available already now (corrected 2026-10-06 after re-reading the letter; this row first said "sent after that"). |
| Date | 2026-10-05 (submitted 2026-03-11, ≈ 7 months) |
| Deadline | **None stated.** Ask the editor only if one is needed. |
| Editors' extra request | Pay particular attention to **readability**, including Section 3 — the subject lies outside the remit of most readers of the journal. |
| Authors | Mukhlis Amien (first, corresponding) + Go Frendi Gunawan |
| Venue rank | **Scimago Q2** (Linguistics and Language; SJR 2024 0.212; Q1 only in 2008 and 2022); CiteScore ≈ 0.7. Source: web summary on 2026-10-05, the Scimago page itself refused access (403) → **verify at scimagojr.com**. Earlier files saying "Q1" (SUBMISSION_CHECKLIST, line 04 contract) were wrong or unverified. |
| Cost | `docs/research_notes/SCOPUS_FREE_VENUE_MAP_2026_06_08.md` lists OL as no-APC (subscription). Gate G15 asks for the check **with URL and date** at this stage → still to do; decline any paid open-access option. |

## 2. Version under review (verified 2026-10-05)

The reviewer PDF (37 pp, anonymous) was compared word-by-word with `draft_v0.1_anonymous.pdf` (31 pp): **similarity 0.9991**;
the only difference is figure captions repeated on the journal's figure pages. So the reviewed text = `draft_v0.1_anonymous.*`.
`draft_v0.1.tex` differs from the anonymous source by 4 lines, `arxiv_submission.tex` by 12. Line numbers below refer to
`draft_v0.1_anonymous.tex`. Decide the single source for the revision (suggest a new `revision_v0.2/` copy) before editing; do not edit
`arxiv_submission.tex` or post an arXiv v2 until the PI decides (the preprint is public: 2604.00023).

## 3. Binding gates for this revision (`docs/SUBMISSION_INTEGRITY_GATE.md`)

- **G1 / "never reword a central, valid critique"** — two reviewer points need new analysis, not rewording: the Makasar comparison (R1-2) and the glottal-stop orthography test (R1-9). Re-derive every number touched or newly stated, blind from raw data.
- **G14** — freeze the revised draft ≥ 14 days, PI re-reads in full, G1 re-run, then resubmit. If the editor later gives a shorter deadline, waiving this is a PI call.
- **G15** — fee check with URL/date (see §1).
- **G16** — final prose, including the **response letter**, is the PI's own. AI role = analysis, data, verification, critique, language check; AI use stays disclosed (the paper already has an AI declaration, which reviewer 2 praised).
- **G13** — whether a manuscript in revision still occupies a "under review" slot is not defined; the conservative reading is that it does (P23 stays blocked). **PI call.**
- **Citation integrity** (`docs/research_notes/` + KB `concepts/integritas-sitasi-ai.md`): every reference supplied by a reviewer must be **read by the PI** before it is cited; the reviewers' quotations are pointers only. No AI-written description of an unread source.

## 4. Reviewer 1 — 13 comments and an opening remark (counted as "14 points" on 2026-10-05) (a specialist in Sulawesi languages; says the ML parts are outside their field; tone positive)

Type: **T** = wording/classification fix · **C** = clarification · **A** = needs analysis or data · **D** = data release.

| # | Point (paraphrased) | Type | Where / note |
|---|---|---|---|
| 1 | "resist reconstruction to **any** proto-form" is too strong: forms may reconstruct to a *lower-level* proto-language (e.g. Proto Muna–Buton) yet still show as residual in one language → "higher-level proto-forms" | T | tex 38, 56 |
| 2 | **Makasar.** The Toalean-substrate claim and lexical divergence of Makasar are in the literature (Mills 1975; Sirk 1989; Bulbeck 1992; Bulbeck et al. — issue dated **2000**, the report says 2001). One source puts 62 % of Makasar basic vocabulary "open to investigation" (38 % retention from PMP) against **our residual figure** in Table 1 (the report says 30.1 %; the table prints **30.9 %**) — how do they fit, and what does this study add? | **A** | **Central.** ✅ **Done (E228 S2, pre-registered):** per concept, Makasar retains 39.3 % from PMP, 25.9 % is coded otherwise, 34.8 % is uncoded → 39.3 % of meanings in the PMP entry's set — a lower bound on retention; ⚠ no comparison with the quoted 38 % / 62 % before the PI has read the source (work plan §10.7, corrected 2026-10-06); different quantities, now shown. Add the literature **after reading it**. |
| 3 | "at least ten primary subgroups (Blust 2009)": that source lists **eleven**; Sneddon 1993 could be cited too | T | tex 57 |
| 4 | "Celebic, South Sulawesi, Muna–Buton" — if ten primary subgroups, Celebic is not one: it is a **supergroup** (Van den Berg — the report says 1989, the paper is **1996**; Mead 2003). Say South Sulawesi, Bungku–Tolaki, Muna–Buton | T | tex 65 |
| 5 | "parallel independent innovations" → "independent innovations" (*parallel innovation* is a technical term meaning something else) | T | tex 73, 509 (subsection title), 517, 598 (+614) |
| 6 | "three subgroups" but four are listed; correct grouping: Muna–Buton (Muna), Wotu–Wolio (Wolio), South Sulawesi (Bugis, Makasar, Sa'dan Toraja), Bungku–Tolaki (Tolaki); Sa'dan Toraja never Celebic. Side remark on spelling: *Makasar* with one *s* is what some linguists prefer (not a request; the report uses both spellings), *Sa'dan Toraja* is preferable in English; the abbreviation "Bol. Mongondow" is unnecessary | T | tex 87 (+86); the abbreviation is at tex 468 (Table 5), tex 486 and inside Figure 4 (the earlier grep missed the escaped form `Bol.\ `) |
| 7 | "under-documentation" is unclear — it normally means very little primary data | C | tex 114, 534; what is meant = low **ABVD cognacy coverage** for Tolaki (36 % of forms assigned to cognate sets) |
| 8 | Muna *dh* is also a digraph (interdental stop) but is missing from the digraph list — maybe none remained after loanword removal? | **A** (check) | ✅ **Done (E227 I03):** two Muna forms (*akaradhaa* 'to work', *idho* 'green'), both cognate-coded, both converted in the test; the sentence omitted *dh* (and Wolio *gh*) |
| 9 | Glottal stop is written many ways in South Sulawesi sources (*anaq/anak/ana'/anaʔ/ana*; pre-glottalised *'d* in Sa'dan as *sakdan/saqdan/sa'dan/saddan*). Does the method give the same result under every convention? | **A** | feature = presence of ʔ or `'` (tex 135, 382); ✅ **Done (E228 S1, pre-registered):** partial dependence — up to −0.016 AUC when the marker is re-coded, −0.006 when left unwritten; the property exists only where a source writes it. **Downgrade, do not reword.** |
| 10 | The "fingerprint" is called *phonological* although one of its features is a semantic category; the two do not go together | C/T | tex 41, 385, 595; separate "phonological profile" from "semantic profile". Overlaps R2 (define "fingerprint" early) |
| 11 | Give example lexemes for each quadrant of the four-quadrant comparison | **A** (data exists) | tex 178–184, 392–414; pull from the E022 × E027 outputs; check each example against ABVD |
| 12 | "unlabeled positive" / "false positive" are used in different senses: (a) inherited vocabulary wrongly flagged as substrate; (b) "E022 false positive" = a form with an Austronesian phonological profile but no assigned proto-form — which is not a false positive, just a form worth a closer look (loanwords are often adapted to the borrower's phonology; Blust 2012:556 is offered) | C | tex 94, 107, 183, 247, 412, 440, 445, 520, 546, 581. Root of the confusion: in positive-unlabeled learning "positive" = *Austronesian*. **R2 asks the same thing (p. 8)** → one definition per term |
| 13 | Does "documentation gaps" for Tolaki mean the wordlist omitted mainstream words, or that cognates are unrecognised? Also **release the 114 Tolaki residual items** (ideally every residual list) so historical linguists can inspect them | C + **D** | tex 534; release as CSV with a DOI (zero cost; check which repo/DOI service already holds the project's data — D1/D2 are on Zenodo) |
| — | Opening remark: the reviewer cannot follow "SHAP beeswarm plot for Model B (GBoost)" | — | readability signal, not an objection |

## 5. Reviewer 2 — 24 highlights in the 37-page PDF (page numbers = that PDF)

No objection to the findings; all are requests to explain terms and choices for non-technical readers. The reviewer also praises the stated limitations, the AI declaration and the open code/data, and finds Section 4 easier than Section 3.

| Group | Points (PDF page) | Action |
|---|---|---|
| **Internal experiment codes leaked into the paper** | what are E027 / E028 / E022? (5); "E022 binary label" — cannot be found in the rule-based section (8) | the manuscript uses `E022` 16×, `E027` 6×, `E028` 2×, `E029` 4×, `E041` 1× (23 lines). Replace with descriptive names (rule-based residual method, ML model B…); fixes several points at once |
| **ML terms unexplained** | classifier and what it shows about "robustness" (7); AUC — meaning and what range counts as "moderate/reliable" (11 ×2, also not explained in 2.3.3); SHAP (12); CV (13); the symbol ∆ (13); κ "agreement" (8); *feature ablation* and why `language_cognacy_coverage` is set in a different font (12); "both models" — which (10); "near-perfect" = what number, and a walk-through of how to read Table 2 (11); DBSCAN parameters look cryptic (18) | non-technical glossary or boxed explanation, definition at first use, interpretation scale for AUC in Methods — the core of the editors' readability request |
| **Rationale for choices** | why k = 5…30 (9); source of the semantic domains — CONCEPTICON? WOLD? (6); what "language identity" feature means (6); "implemented in scikit-learn / XGBoost — Python?" (7) | state the reasons; **look the domain source up in the code** before writing it down |
| **"false positive"** | what the false positives stand for in this study (8) | same as R1-12 |
| **Figures/tables** | the panels of Figure 2 are neither referred to nor explained (18); **Table 5 is cut off** (20) | layout fix for Table 5; refer to each panel in the text |
| **Interpretation** | why do "One Hundred", "Fifty", "Twenty", "to stand", "to hit" appear as consensus substrate in ≥ 4 languages (18)? what does "fingerprint generalises across Sulawesi languages" mean concretely (12)? define *fingerprint* as "probabilistic phonological profile" **at the start**, not in the Discussion (21) | the numeral-compound explanation already exists (tex 440–445); define the term in the Introduction (overlaps R1-10) |
| **Citation** | the claim that vocabulary "resists reconstruction" needs a reference to earlier studies (2) | find and **read** the source first |

## 6. Suggested order (all PI-gated) — **superseded 2026-10-05 evening by `REVISION_WORKPLAN.md` §7**; items 3a–3e are done (E227, E228)

1. Wording and classification fixes: R1 #1, 3–7, 10 (small, mechanical, certain).
2. Terms and notation: drop the `E0xx` codes, one definition for "false positive", glossary for the ML terms (R1 #12 + R2 groups 1–2) — the largest part of the readability work.
3. Analyses (each as its own numbered experiment starting at the next free number; never edit E022/E027 results in place):
   a. glottal-stop orthography robustness (R1-9) · b. *dh* digraph count (R1-8) · c. Makasar comparison with a pre-registered definition (R1-2) · d. quadrant examples (R1-11) · e. residual lists for release (R1-13).
4. Add literature **only after the PI has read it**: Mills 1975, Sirk 1989, Bulbeck 1992, Bulbeck et al. 2000, Sneddon 1993, Van den Berg 1996, Mead 2003, Blust 2012 (details: `REFERENCE_CHECK_20261005.md`), plus a source for the "resists reconstruction" sentence.
5. Rewrite Section 3 for non-technical readers; refer to every panel of Figure 2 and walk through Tables 2 and 5.
6. Freeze 14 days (G14) → PI re-read → G1 re-run → response letter (R1 13 comments + R2 24 highlights) → resubmit through the portal.

## 7. Downstream effects (decisions are the PI's)

- **P9** (`docs/SUBMISSION_TIMELINE.md` §P9: "submit only after P2 or P8 decision; if P8 accepted, P9 gains credibility") — the decision has now landed, but G13/G14 still apply.
- **P23** — blocked by G13 until P2 or P8 decides; see the G13 reading above.
- **Career side (KB, not here):** first-author Q2 article; whether it can count toward the special requirement for the next academic rank depends on the dissertation-topic rule — tracked in the KB (`research/ol-p8-revisi.md`, `career/kum-tracker.md`).
- Tooling note: the Gmail connector cannot return a message this large; the attachment was fetched in-browser (Playwright, `fetch(...view=att)` → base64 → decode) and read with PyMuPDF (`page.annots()`).

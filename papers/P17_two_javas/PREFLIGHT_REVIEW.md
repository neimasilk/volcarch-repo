# P17 Pre-Flight Review — Archeologia e Calcolatori

**Date:** 2026-03-21
**Reviewer:** Claude (autonomous mode)
**Draft:** v0.2 (`draft_v0.2.tex`, 464 lines, ~7,000 words, 22pp double-spaced, 5 figures, 30 references)

---

## Target Journal: Archeologia e Calcolatori (CNR Italy)

| Criterion | Known |
|-----------|-------|
| Scope | Computational archaeology — EXCELLENT fit |
| APC | Diamond OA (zero cost, CC BY-NC-ND 4.0) |
| Indexing | Scopus + WoS ESCI |
| Peer review | Double-blind |
| Deadline | Before December 31 annually |
| Submit via | archcalc.cnr.it |
| Format | **TBD — download editorial rules** |
| Citation style | **TBD** |
| Word limit | None stated |

**Action needed:** Download and read editorial rules from archcalc.cnr.it before submission. The format, citation style, and figure requirements are not yet verified.

---

## Content Review

### Strengths

1. **The strongest paper in the VOLCARCH portfolio.** Five independent analyses converging on a single model. Each analysis alone would be publishable; together they are compelling.

2. **"Two Javas" is a memorable, citable concept.** It reframes the Indianization debate in spatial terms. This is the kind of contribution that gets cited.

3. **The 929 CE natural experiment** is methodologically elegant. It's the best evidence in the paper — a political collapse that removes the Sanskrit overlay and reveals the indigenous substrate. Reviewers will appreciate this.

4. **Statistical rigour** — Spearman correlations, Mann-Whitney, Fisher's exact, chi-square, partial correlation controlling for confounds. The adversarial approach (controlling for inscription length in the vocabulary-depth correlation) pre-empts reviewer objections.

5. **Good comparative context** — Pompeii, Akrotiri, Cerén comparison in Discussion (Section 5.5). Distinguishes catastrophic vs. cumulative burial.

6. **Honest limitations** — geocoding uncertainty, threshold sensitivity, colonial sample size, vocabulary classification subjectivity.

7. **30 references** — adequate. Mix of core archaeology (Coedes, Wolters, Schiffer), Indonesian specialists (Christie, de Casparis, de Groot), and volcanology.

8. **Liangan cited** (Abbas 2016) — consistent with VOLCARCH's engagement with this key site.

### Issues to Address

#### A. Content Issues

**A1. Experiment count outdated (LOW)**
Line 70: "107 computational experiments." Current count is 120. Update to current number before submission.

**A2. Self-citations need management (MEDIUM)**
Two self-citations:
- `Amien2026a` — VOLCARCH repository (unpublished). Acceptable as a data repository reference.
- `Amien2026b` — P7 Antiquity Project Gallery (under review). If P7 is rejected before P17 submission, this becomes a citation to an unpublished paper. Update the note if status changes.

For double-blind review, self-citations may need anonymisation. Check ArchCalc policy — some journals require "Author (year)" → "[Anonymous] (year)" for blinding.

**A3. The "Two Javas" model needs a visual — CONFIRMED MISSING (MEDIUM)**
Line 314 references `Figure~\ref{fig:model}` but **no `\label{fig:model}` exists** in the document. Cross-reference check confirms this is the ONLY dangling reference (all other 6 refs match labels). The compiled PDF will show "Figure ??" at this point. Either create the conceptual diagram or remove the reference and describe the model in text only.

**A4. DHARMA citation format (LOW)**
`DHARMA2024` is cited as `@misc`. Some journals may want a more formal citation. Check if the ERC project has a preferred citation format.

**A5. British vs American English (LOW)**
The paper uses British spelling in places ("artefact" on line 218, "centres" on line 55) but American in others (needs checking). For ArchCalc (Italian journal, English accepted), consistency matters more than which variant. Check throughout.

**A6. "Indianization" vs "Indianisation" (LOW)**
Used inconsistently. Pick one and apply throughout.

#### B. Structural Issues

**B1. P11 overlap risk (MEDIUM)**
Both P11 and P17 target ArchCalc. They share:
- Same candi dataset (142 temples)
- Same inscription dataset (DHARMA geocoded)
- Same volcanic distance analysis framework
- Some of the same references

They differ in:
- P11: candi as spatial proxy → fieldwork predictions (practical application)
- P17: Two Javas model → reinterpretation of Indianization (theoretical contribution)

**Risk:** An ArchCalc editor receiving both manuscripts may see substantial overlap. The datasets are identical; the framing differs.

**Mitigation options:**
1. Submit P17 to ArchCalc, P11 to a different journal (Indonesia/Cornell or Archipel)
2. Submit both but acknowledge the companion paper and differentiate clearly
3. Merge into one comprehensive paper (but this would be very long)

**Recommendation:** Option 1 is safest. P17 is the stronger paper and better fits ArchCalc's theoretical scope. P11's practical focus (fieldwork targets) fits applied venues like Indonesia (Cornell) or J. Pacific Archaeology.

**B2. Data availability URL (LOW)**
Line 449: "VOLCARCH repository" without specific URL. Need to add GitHub URL and verify it's public before submission.

#### C. Double-Blind Requirements

**C1. Author identification removal**
If ArchCalc is double-blind, the submission version must:
- Remove author name, affiliation, ORCID from manuscript
- Anonymise self-citations (Amien2026a → [Anonymous], Amien2026b → [Anonymous])
- Remove or anonymise AI disclosure mention of VOLCARCH by name
- Check if GitHub URL reveals authorship

---

## Format Conversion (pending ArchCalc specs)

Currently in LaTeX. Possible required conversions:
- [ ] LaTeX → Word (if required)
- [ ] Reformat citations to ArchCalc style
- [ ] Reformat figures to ArchCalc specifications
- [ ] Check abstract requirements
- [ ] Check if supplementary materials accepted

**Compile test:** Verify `draft_v0.2.tex` compiles cleanly with `p17_references.bib`. (Note: uses `chicago` bibliographystyle — may need to change for ArchCalc.)

---

## Action Items (for Pak Amien)

### Before Submission
1. **Download ArchCalc editorial rules** from archcalc.cnr.it — this is blocking
2. **Create or remove Figure 6** — the Two Javas model diagram referenced on line 314
3. **Update experiment count** — 107 → 120 (A1)
4. **Decide P11 vs P17 journal allocation** — don't send both to ArchCalc (B1)
5. **Spelling consistency check** — pick British or American, apply throughout (A5-A6)
6. **Verify GitHub repo is public** with claimed datasets (B2)
7. **Prepare double-blind version** — anonymise self-citations, author info (C1)

### Optional Improvements
8. Add the "Two Javas" model diagram (currently referenced but missing)
9. Update self-citation status if P7 verdict arrives before P17 submission
10. Consider adding Tuban nekara finding (BS-4) as additional evidence for pre-Hindu volcanic-zone presence

### Timeline
- ArchCalc deadline: December 31, 2026
- Pak Amien review: ~April-May 2026
- Format conversion: ~June 2026
- Submit: ~September-October 2026 (comfortable margin)

---

## AI Prose Audit: PASS

Scanned for all markers in `docs/AI_PROSE_GUIDE.md`:
- Zero transition word flags (Furthermore, Moreover, etc.)
- Zero "It is worth noting" / "Importantly" / "Crucially" flags
- Zero "comprehensive/multifaceted/nuanced/compelling" flags
- Only "robust" used once in legitimate statistical context (line 428: "too small for robust statistical inference")
- "Significantly" used only in statistical contexts (legitimate)
- Clean.

---

*P17 is substantively the strongest paper in the portfolio. The main risks are logistical (ArchCalc format requirements, overlap with P11) not intellectual. This paper has real potential for impact — "Two Javas" is a citable concept.*

# P11 Pre-Flight Review — Target Journal Analysis

**Date:** 2026-03-21
**Reviewer:** Claude (autonomous mode)
**Draft:** v0.3 (`draft_v0.3.tex`, 432 lines, ~3,200 words, 18pp double-spaced)

---

## CRITICAL FINDING: Wacana Is Thematic — "Kawi" Issue Already Published

The WORKSTATE lists Wacana (UI) as P11's primary target, citing the "Kawi culture" thematic issue as "directly relevant." However:

- **Vol 26 No 3 (2025) "Kawi culture behind the language of poets" is ALREADY PUBLISHED** (guest-edited by Wayan Jarrah Sastrawan & Aditia Gunawan)
- Wacana is a **thematic journal** — all submissions must target a specific upcoming issue
- **No currently open issue fits P11's scope** (temple siting / spatial prediction)

### Wacana's Upcoming Issues (with relevance assessment)

| Volume | Date | Theme | P11 fit? |
|--------|------|-------|----------|
| Vol 27 Nos 1-2 | April 2026 | "Media and cultural commoning" | NO — closed |
| Vol 27 No 3 | October 2026 | "Exploring island voices and experiences" | NO — closed |
| **Vol 28 Nos 1-2** | **April 2027** | **"Prehistoric art in Indonesia and related regions"** | STRETCH — P11 is Hindu-Buddhist, not prehistoric |
| Vol 28 No 3 | October 2027 | "Muarajambi; Society and civilization in Sumatra" | NO — Sumatra, not Java |
| Vol 29 Nos 1-2 | April 2028 | "Discourses on restitution in Indonesia" | NO |
| Vol 29 No 3 | October 2028 | "Contemporary relevance of traditional Javanese literature" | NO |
| Vol 30 Nos 1-2 | April 2029 | "Documents from royal chanceries in Southeast Asia" | POSSIBLE but 3 years away |

**Verdict: Wacana is NOT viable for P11.** The thematic constraint eliminates it.

### STRATEGIC OPPORTUNITY: Wacana Vol 28 for P19

The **"Prehistoric art in Indonesia and related regions" (April 2027)** issue is an excellent fit for **P19 "Before the Inscriptions"** — which argues that pre-Hindu Indonesian culture is invisible due to taphonomic/historiographic processes. If BKI doesn't work out, this is a strong fallback. Submission deadline likely ~October 2026.

---

## Revised Target Recommendation

### Option A: Indonesia (Cornell SEAP) — RECOMMENDED

| Criterion | Assessment |
|-----------|------------|
| Scope fit | EXCELLENT — Southeast Asian studies, Java archaeology, spatial analysis |
| Format | MS Word, double-spaced, Chicago 17th (**already converted**: `draft_v0.3_submission.docx`) |
| Word limit | ~15,000 (P11 is 3,200 — well within) |
| APC | **None** (free to publish) |
| Diamond OA? | **NO** — subscription-based, older articles free on JSTOR/eCommons |
| Indexing | JSTOR, Project MUSE, not Scopus |
| Review speed | Unknown |
| Thematic? | No — accepts general submissions year-round |

**Pros:** Best scope fit, no format barriers (Word conversion done), free, accepts immediately.
**Cons:** Not Diamond OA (violates the policy). Not Scopus-indexed. Lower visibility than Wacana.

### Option B: Archeologia e Calcolatori (CNR Italy) — STRONG ALTERNATIVE

| Criterion | Assessment |
|-----------|------------|
| Scope fit | EXCELLENT — "computational archaeology" is exactly what P11 does |
| Format | Needs checking (editorial rules not yet downloaded) |
| Word limit | None stated |
| APC | Diamond OA (zero cost) |
| Indexing | Scopus + WoS ESCI |
| Review speed | Unknown |
| Thematic? | No — annual deadline Dec 31 |

**Pros:** Diamond OA, Scopus + WoS, perfect scope (spatial analysis = computational archaeology), no thematic constraint.
**Cons:** P17 already targets ArchCalc — two papers to same journal risks overlap perception. Format requirements not yet verified.
**Mitigation:** P11 (candi as proxy) and P17 (Two Javas divergence) are methodologically distinct. Different research questions. Could submit to same journal if framed carefully.

### Option C: Archipel (INALCO/CNRS) — BACKUP

| Criterion | Assessment |
|-----------|------------|
| Scope fit | GOOD — "Insulindian studies," Java archaeology fits |
| APC | Diamond OA (zero cost) |
| Indexing | Scopus Q3 |
| Thematic? | Unknown — needs checking |

**Recommendation:** **Option B (Archeologia e Calcolatori)** is the best fit, given the Diamond OA mandate. But verify format requirements first, and consider whether submitting P11 AND P17 to the same journal is strategically wise.

If Diamond OA is relaxed for free-to-publish journals, **Option A (Indonesia/Cornell)** is the easiest path — submission is essentially ready.

---

## Content Review

### Strengths

1. **Four independent statistical tests** — Rayleigh, Mann-Whitney, chi-square, Poisson regression — all significant. Reviewer-proof.
2. **Clear narrative arc:** problem → proxy → evidence → prediction → validation → targets
3. **Adversarial framing** (Section 4.5) — pre-empts "it's just survey bias" by testing and rejecting it
4. **Japan comparandum** — sophisticated, not dismissive ("Japan overcame the barrier through institutional investment")
5. **Practical output** — 10 GPS targets, actionable for heritage bodies
6. **Honest limitations** — Penanggungan dominance, no subsurface validation, temporal range
7. **AI disclosure** — present, tasteful, honest
8. **No self-citations** — clean (all removed per policy)
9. **Inscription-candi divergence** — the most original contribution. No one has geocoded the DHARMA database before.

### Issues to Address

#### A. Content Issues

**A1. DHARMA citation missing (MEDIUM)**
Line 88: "DHARMA epigraphic database" is mentioned but not cited. Should add:
> Argüelles, A. et al. (2023). DHARMA: The Domestication of "Hindu" Asceticism and the Religious Making of South and Southeast Asia. Digital Humanities project, CNRS/ERC.
Or whatever the proper citation is. Check DHARMA project website.

**A2. Liangan underexploited (LOW)**
Table 5 mentions Liangan (4m burial) but it's not discussed in text. The blind spot analysis (BS-3) identified Liangan as a validation case. Add 1-2 sentences in Section 4.3 or Discussion noting that Liangan validates the prediction: volcanic burial CAN preserve complete settlements including organic material.

**A3. Reference count thin (LOW-MEDIUM)**
Only 10 references for an 18pp paper. Academic reviewers may note this. However, all are high-quality and the paper is data-driven, not literature-review-driven. Consider adding:
- Schiffer 1987 (formation processes — theoretical foundation)
- Novida Abbas 2016 (Liangan)
- Sheets 1992 or 2002 (Cerén — volcanic preservation analog)
- DHARMA project citation
This would bring it to 14, a more comfortable number.

**A4. "Volcanic Informedness" framing absent (NOTE)**
The paper was originally titled "Volcanic Informedness" but has been retitled to "Temple Siting as Archaeological Proxy." The current title is better — more concrete, more searchable. No action needed, but ensure all internal references use the new title.

**A5. Settlement suitability model (AUC=0.768) mentioned but not detailed (LOW)**
Section 5.3 mentions a "companion settlement suitability model" but doesn't cite where this model comes from (is it P2?). Either cite the source or remove the claim.

#### B. Format Issues (journal-dependent)

**For Indonesia (Cornell):**
- [x] Word conversion done
- [ ] Citation style: natbib → Chicago 17th (need to decide author-date vs notes-bibliography)
- [ ] Fix Unicode issues in Word file
- [ ] Replace PDF figure embeds with PNG
- [ ] Verify cross-references
- [ ] Add page numbers

**For Archeologia e Calcolatori:**
- [ ] Download and read editorial rules from archcalc.cnr.it
- [ ] Determine accepted format (Word? LaTeX? PDF?)
- [ ] Check citation style
- [ ] Check figure requirements

**For Wacana (IF a suitable issue opens):**
- [ ] Abstract must be ≤150 words (currently ~230 — trim by 80 words)
- [ ] 20-40 pages required (P11 is only 18pp — would need expansion to ≥20)
- [ ] Book Antiqua 11pt, 1.5 spacing
- [ ] British spelling with -ize variant
- [ ] Write out abbreviations (e.g. → "for example")
- [ ] Author bio required

#### C. Structural Issues

**C1. Abstract length** — 230 words. Fine for Indonesia/ArchCalc but exceeds Wacana's 150-word limit.

**C2. Paper length** — 3,200 words / 18pp double-spaced. Short for Wacana (20-40 pp required), fine for Indonesia and likely ArchCalc.

**C3. Data availability URL** — References `github.com/mukhlisamien/volcarch`. Verify this repo is public and contains the claimed datasets before submission.

---

## Action Items (for Pak Amien)

### Before Submission
1. **DECIDE target journal** — Indonesia (Cornell) or Archeologia e Calcolatori? This determines format conversion work.
2. **Check ArchCalc format requirements** — download editorial rules from archcalc.cnr.it
3. **Add DHARMA citation** (A1)
4. **Consider adding 3-4 references** (A3) — Schiffer, Abbas/Liangan, Sheets/Cerén, DHARMA
5. **Read aloud for flow** — especially Japan paragraph (already on checklist)
6. **Verify GitHub repo** is public with claimed datasets (C3)
7. **Email editor** — if Indonesia: confirm Chicago variant; if ArchCalc: confirm format

### Optional Improvements
8. Add 1-2 sentences on Liangan in Discussion (A2)
9. Clarify settlement suitability model source (A5)
10. Consider whether to include fieldwork target table or keep "available on request"

### Strategic Note
If submitting to Indonesia (Cornell), this can go NOW — the manuscript is essentially ready after minor fixes (A1, Chicago citations). If targeting ArchCalc, more prep work needed (format check, possible reformatting), but deadline is Dec 31 — no rush.

---

## P19 → Wacana Opportunity (Flag for Strategy)

Wacana's **"Prehistoric art in Indonesia and related regions" (April 2027)** issue is a near-perfect fit for P19 "Before the Inscriptions." If BKI submission fails or seems unlikely, Wacana Vol 28 should be the fallback. Likely submission deadline: ~October 2026 (6 months before publication).

This also means P19's timeline (target September 2026 for BKI) aligns well — if BKI review is slow or negative, pivot to Wacana by October 2026.

---

## AI Prose Audit: PASS

Scanned for all markers in `docs/AI_PROSE_GUIDE.md`:
- Zero transition word flags (Furthermore, Moreover, etc.)
- Zero "It is worth noting" / "Importantly" flags
- Zero "comprehensive/multifaceted/nuanced/compelling" flags
- "Significantly" used only in statistical contexts (legitimate)
- Clean.

---

*Pre-flight review complete. P11 is substantively strong — the main issue is target journal selection, not content quality.*

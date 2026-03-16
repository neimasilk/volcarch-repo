# ~~Next Session Brief~~ — DEPRECATED

> **DEPRECATED (2026-03-16):** This file is superseded by `docs/WORKSTATE.md`. New sessions should read WORKSTATE.md for the active work contract. This file is retained as a historical snapshot only — do not update it.

# Next Session Brief — 2026-03-16 (updated post-E088)
## Prepared: 2026-03-16

---

## 1. Project Snapshot

**6 papers submitted. 2 data papers drafted. 90 experiments. Structural critique complete. E088-E090 textual archaeology + transformer NLP pipeline operational.**

| Paper | Journal | Submitted | Status | MS# |
|-------|---------|-----------|--------|-----|
| P1 | Asian Perspectives (Q1) | 2026-03-10 | Awaiting reviewer | 019A-0326 |
| P2 | JCAA (Diamond OA) | 2026-03-11 | Awaiting reviewer | #280 |
| P5 | BKI (Diamond OA) | 2026-03-09 | Awaiting reviewer | -- |
| P7 | Antiquity Project Gallery (Q1) | 2026-03-06 | Awaiting reviewer | -- |
| P8 | Oceanic Linguistics (Q1) | 2026-03-11 | Awaiting reviewer | OL-03-2026-11 |
| P9 | JSEAS (NUS Press) | 2026-03-11 | Desk review | JSEAS-202603-051 |
| **P11** | Target: Indonesia (Cornell, Q2) | **DRAFTING** | v0.2 reframed | -- |
| **D1** | JOAD (APC £374) | **DRAFT READY** | Colonial register (52 entries) | -- |
| **D2** | JOAD (APC £374) | **DRAFT READY** | Mini-NusaRC (80 sites) | -- |

---

## 2. What Was Accomplished This Session (Mata Elang #8 + E088)

### E088: Computational Textual Archaeology — SUCCESS
- 27 references across 9 traditions, 73 entities, 6 cross-lingual resolution groups
- Monte Carlo convergence: p < 0.00001 (9/9 traditions → insular SE Asia)
- 18/27 (67%) predate 400 CE — external world saw Nusantara before Nusantara's own record begins
- Knowledge graph: 93 nodes, 83 edges. Genuinely NEW independent data stream.
- P16 pipeline foundation ready. Next: expand to 50+ references + LLM-powered NER.

### E089: Expanded Textual Corpus — SUCCESS
- 27 → 50 references, 10 traditions (added Tamil/Sangam), 143 entities
- All 50 with actual passage text for NLP pipeline

### E090: Transformer NLP — MIXED (4/6 informative, 2 negative)
- **STRONG:** UMAP+HDBSCAN → 78% cross-tradition clusters (CONTENT-driven, not tradition-driven)
- **STRONG:** Semantic convergence → 4/5 concepts converge (CAMPHOR z=6.55, MARITIME z=9.44)
- **NEGATIVE:** NLI entailment fails (0.161, below baseline) — wrong tool
- **WEAK:** BERTopic — corpus too small (need 200+)
- Key insight: ancient texts cluster by what they DESCRIBE, not who wrote them

### Structural Critique — 10-Section System/Research Design Review
- Dataset monoculture identified: 21/85 experiments depend on same 268 DHARMA inscriptions
- 6 Layers of Darkness → recommend collapse to 3 (Physical Taphonomy, Historiographic Bias, Cosmological Overwrite)
- Speed/credibility risk flagged (6 papers in 10 days from first-time author)
- Prescription: **CONTRACTION, not expansion** — validate existing work, stop adding experiments

### E086/ADV-1: Japan Comparanda — PARTIAL
- Japan 100-200× more archaeological survey per area
- Japan HAS volcanic burial sites (Kanai Higashiura) — found through rescue archaeology
- Kikai-Akahoya (7300 BP) IS a VOLCARCH-type phenomenon
- Java 32× deeper sustained burial (tropical lahar regime)
- **L1 MUST be reframed:** volcanism × survey deficit, not volcanism alone
- L1 Constitution updated. P1 revision ammo created.

### E087/ADV-5: Substrate Negative Control — GREY ZONE
- C1 (Tagalog+Cebuano) AUC=0.568 — PASS
- C5 (Iban+Malay) AUC=0.713 — ALARMING (nearly matches Sulawesi 0.727)
- Detector conflates ABVD documentation gaps with substrate signal
- **P8 MUST reframe:** "phonological non-conformity" not "substrate detection"
- P8 revision ammo created.

### Adversarial Scorecard — COMPLETE (5/5 tests done)

| Test | Target | Result |
|------|--------|--------|
| ADV-1 Japan comparanda | L1 | **PARTIAL** (survives with scope restriction) |
| ADV-2 Non-volcanic control | L1 | INCONCLUSIVE (p=0.760, N too small) |
| ADV-3 Survey intensity | L1 | **PASSED** (p=0.0015) |
| ADV-4 Substrate noise | L4 | **PASSED** (p=0.0000, z=11.05) |
| ADV-5 Negative control | L4 | **GREY ZONE** (C5 AUC=0.713) |

---

## 3. What Needs To Happen Next

### STRUCTURAL CRITIQUE FOLLOW-UP (highest priority)
1. **Colonial data verification** — Open 10 E070 entries on Delpher.nl, verify manually. 2hrs work, prevents retraction.
2. **Code review** — Pick 5 key scripts (E027, E065, E069, E082, E083), ask Go Frendi or student to rerun from scratch.
3. **Dependency freeze** — `pip freeze > requirements_frozen.txt`
4. **Consilience reframing** — Rewrite manifesto to honestly state "2 primary datasets + 3 supplementary" not "4 independent streams"

### PAPER-RELATED
5. **P11 v0.3** — Incorporate E082/E083/E084 + Japan comparandum + survey intensity framing
6. **D1+D2** — Zenodo deposit + JOAD submission (user decision on APC waiver)
7. **Wait for editorial responses** — Do NOT submit more papers

### IF REVIEWER RESPONSES ARRIVE
8. **P1 revision** — Japan paragraph ready (ADV1_japan_comparanda.md), depth argument ready (ADV2_depth_vs_sitetype.md)
9. **P8 revision** — Negative control reframing ready (ADV5_negative_control.md)

### MEDIUM-TERM (only after 1-7 complete)
10. **Conference presentation** — Submit to EHPA or Berkala Arkeologi Jogja. Build real-world presence.
11. **Colonial register v2.0** — Expand 52→75+ entries (22 candidates identified)

### USER DECISIONS NEEDED
- JOAD APC £374: proceed with waiver request, or find free alternative?
- Conference submission: which venue, when?
- Preprint: post P1 to EarthArXiv now?

---

## 4. Key Files

| Resource | Location |
|----------|----------|
| E086 ADV-1 Japan | `experiments/E086_adv1_japan_comparanda/` |
| E087 ADV-5 Negative Control | `experiments/E087_substrate_negative_control/` |
| E088 Textual Archaeology | `experiments/E088_textual_archaeology_nlp/` |
| E089 Expanded Corpus | `experiments/E089_expanded_textual_corpus/` |
| E090 Transformer NLP | `experiments/E090_transformer_textual_nlp/` |
| P1 Japan revision ammo | `papers/P1_taphonomic_framework/revision_ammo/ADV1_japan_comparanda.md` |
| P8 negative control ammo | `papers/P8_linguistic_fossils/revision_ammo/ADV5_negative_control.md` |
| Structural critique | Journal entry 2026-03-16 |

---

## 5. Cathedral Findings (survive ALL adversarial scrutiny)

| Finding | p-value | Adversarial status |
|---------|---------|-------------------|
| E066 equinox orientation | 4.9e-14 | Trivially true (temples face east) — low novelty |
| E065 Zone A overrepresentation | <1e-6 | Needs population density control |
| E084 inscription divergence | 5.2e-08 | Clean — genuinely novel |
| E051 toponymy substrate | 5.1e-14 | Needs linguist validation |
| ADV-3 volcanic signal | 0.0015 | Clean — survives survey intensity control |
| E083 tephra correlation | Independent dataset | Clean — 51 colonial-era pairs |

---

## 6. Copy-Paste Prompt for Next Session

```
Lanjutkan. Baca brief:
- docs/NEXT_SESSION_BRIEF.md
- docs/JOURNAL.md (entry "Mata Elang #8")

STATUS: 87 eksperimen, 6 paper submitted, P11 drafting, D1+D2 drafted.
Mata Elang #8: Structural critique + ADV-1 Japan (PARTIAL) + ADV-5 negative
control (GREY ZONE).

PRIORITAS (post-critique):
1. Colonial data verification (10 entries via Delpher.nl)
2. Code review (5 key scripts)
3. Consilience reframing (honest dataset dependency)
4. P11 v0.3 (incorporate Japan + survey intensity)
5. Wait for editorial responses — NO new submissions

Kamu otonom. "Santai dalam waktu, serius dalam metode."
```

---

*"Japan menunjukkan apa yang bisa ditemukan Indonesia jika berinvestasi 100× lebih banyak. VOLCARCH bukan tentang vulkanisme saja — ia tentang vulkanisme × kelangkaan survei."*

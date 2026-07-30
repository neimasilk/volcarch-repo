# HANDOFF — Post-Mudik Session 4 (2026-03-21/22)

## Apa yang Dikerjakan

Sesi otonom lanjutan dari Session 3. Fokus: blind spot analysis review, P11/P17 pre-flight reviews, research notes (Liangan, phytolith, Cerén), dan penemuan kritis tentang Wacana.

### Pre-Flight Reviews

| Paper | Hasil | File |
|-------|-------|------|
| **P11** | PRE-FLIGHT COMPLETE. Wacana NOT viable (thematic journal). Recommend Indonesia (Cornell). AI prose audit: PASS. Issues: thin refs (10), add DHARMA citation. | `papers/P11_volcanic_informedness/PREFLIGHT_REVIEW.md` |
| **P17** | PRE-FLIGHT COMPLETE. Strongest paper. 1 dangling cross-ref (`fig:model` line 314 — no `\label{fig:model}`). Experiment count 107→120. Spelling inconsistency. AI prose audit: PASS. | `papers/P17_two_javas/PREFLIGHT_REVIEW.md` |

### Research Notes (3 baru)

| Note | Lokasi | Temuan Kunci |
|------|--------|-------------|
| **Liangan Validation Case** | `docs/research_notes/LIANGAN_VALIDATION_CASE.md` | 15+ referensi. Burial 6-8m, PDC 295-487°C. Carbonised rice (tropical japonica, Castillo 2014). No published sedimentation rates → E121 gap. No Cerén comparison in literature. |
| **Phytolith Volcanic Preservation** | `docs/research_notes/PHYTOLITH_VOLCANIC_PRESERVATION.md` | STRONGLY POSITIVE. Survive 90K yr in tephra (Aso, Japan). Java andisol pH 5-7 = excellent. Rice phytoliths diagnostic to subspecies. NO ONE has tested Java volcanic matrices. I-125 upgraded HYPOTHESIS → TESTABLE. |
| **Cerén Comparison** | `docs/research_notes/CEREN_COMPARISON.md` | Joya de Cerén (~AD 600) = closest global analog. Thatch, wood, food, mats preserved. No formal Cerén-Java comparison in published literature — confirmed gap, publication opportunity. |

### Penemuan Strategis

1. **Wacana is thematic** — Semua isu punya tema khusus. "Kawi culture" (Vol 26 No 3) sudah terbit. Tidak ada isu terbuka yang cocok untuk P11/P16/P9. Ini mengeliminasi Wacana sebagai target untuk 4 paper sekaligus.

2. **Wacana Vol 28 = P19 fallback** — "Prehistoric art in Indonesia and related regions" (April 2027, OPEN). Deadline ~October 2026. Cocok jika BKI menolak P19.

3. **P11/P17 overlap risk** — Keduanya pakai dataset candi+inscriptions yang sama. Jangan kirim keduanya ke ArchCalc. Rekomendasi: P17 → ArchCalc (stronger), P11 → Indonesia (Cornell).

4. **Phytolith pathway** — Jalur baru yang transformative. Jika ada kolaborator archaeobotanist (Cristina Castillo/UCL, Zhenhua Deng), bisa test phytolith survival di core vulkanik Jawa. Ini akan menjadi bukti fisik pertama untuk VOLCARCH.

### Dokumen yang Diupdate

| File | Perubahan |
|------|-----------|
| `docs/WORKSTATE.md` | P11 target revised, P17 pre-flight noted, Wacana thematic constraint, P19 fallback added |
| `docs/JOURNAL.md` | Session 2 entry added (11 items) |
| `docs/IDEA_REGISTRY.md` | I-124 → READY, I-125 → TESTABLE |
| `docs/TRIGGER_MAP.md` | Archaeobotanist trigger, Liangan matrix trigger, PVMBG expanded |
| `docs/research_notes/JOURNAL_SUBMISSION_GUIDES.md` | Wacana section rewritten with thematic constraint warning |
| `docs/research_notes/REJECTION_PATTERN_ANALYSIS.md` | P11, P9 predictions updated |

### ArchCalc Agent

Launched background agent to research ArchCalc editorial rules. Agent failed (OAuth token expired — web search unavailable). **ArchCalc editorial rules still need to be downloaded manually** from archcalc.cnr.it.

## Status Saat Ini

- **120 eksperimen** (unchanged)
- **3 paper under review** (P2-JCAA, P7-Antiquity, P8-OL)
- **P1 EGQSJ fully ready** — tinggal register + upload
- **P11 pre-flight complete** — target revised to Indonesia (Cornell)
- **P17 pre-flight complete** — target ArchCalc, download editorial rules first
- **4 research notes baru** (blind spot comprehensive + Liangan + phytolith + Cerén)
- **I-125 phytolith: TESTABLE** — jalur transformative jika ada kolaborator

## Post-Session Action Items (Urutan Prioritas)

### Pak Amien Harus Lakukan (Manual)

1. **P1 submit EGQSJ** — register editor.copernicus.org → upload `submission_egqsj_v1.0.tex` + `references.bib`
2. **Verify JCAA APC** — cek apakah waiver applied ke P2 submission #280. Jika belum, hubungi journal@caa-international.org
3. **Decide P11 target** — Indonesia (Cornell) recommended. Jika setuju, minor fixes lalu submit.
4. **Download ArchCalc rules** — buka archcalc.cnr.it, download editorial guidelines (blocking untuk P17)
5. **Deep reading untuk P19** — Lombard Vol.3, Bloembergen 2020, Wolters 1999
6. **D1+D2 Zenodo** — deposit datasets, free, ~30 min each

### Claude Bisa Lakukan (Next Session)

1. P11 format conversion untuk Indonesia (Cornell) jika Pak Amien setuju target
2. P17 format conversion setelah ArchCalc rules tersedia
3. P16 DHQ preparation (user review needed first)
4. E076 v2 satellite run (needs internet)
5. E119 synthesis figure render (matplotlib, needs compute)

---

*Sesi ini menghasilkan 5 dokumen baru dan 6 dokumen diupdate. Zero eksperimen baru (moratorium). Fokus sepenuhnya pada konsolidasi, pre-flight, dan research notes yang memperkuat basis literatur VOLCARCH.*

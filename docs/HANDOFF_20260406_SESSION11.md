# HANDOFF: Session 11 — JCAA Waiver + P17 Fixes + P11 Expansion (2026-04-06)

**Dari:** Claude (sesi 11)
**Untuk:** Sesi berikutnya
**Durasi:** ~2 jam

---

## RINGKASAN 30 DETIK

JCAA waiver email SENT ke Verhagen (journal direct waiver, FCFS). P17 ArchCalc di-fix 5 kelemahan kritis (overclaim, confound, Bali, sentence-transformer, thresholds) — compliance ALL PASS, .docx regenerated. P11 Archipel di-expand dari 2,600→5,300 kata, 17→29 referensi, semua [EXPAND] marker diisi (Lombard, Degroot, Christie, Bloembergen/Eickhoff). Doc sync audit: L2+L3 updated.

---

## DELIVERABLES SESI INI

### 1. JCAA Waiver Email — SENT
- Verhagen (2026-04-03) confirmed: journal direct waiver exists, limited, FCFS
- Email reply SENT 2026-04-06: requested waiver for P2 #280, offered to review (NLP/ML/AI expertise)
- File: `papers/P2_settlement_model/jcaa_waiver_reply_verhagen.md`
- **WAIT for reply**

### 2. P17 ArchCalc — 5 Critical Fixes (v0.3 → v0.4)
| Fix | Problem | Solution |
|-----|---------|----------|
| 1 | Overclaim "Indianization = 15 km" | → "textual record of Indianization" |
| 2 | Depth-vocabulary confound | Added volcanic distance rho=-0.295 as honest confound |
| 3 | Bali N=5 "validation" | → "consistency check", N=5 caveat explicit |
| 4 | Sentence-transformer tanpa Methods | Added to Methods section |
| 5 | Zone thresholds post-hoc | Justified via volcanological hazard boundaries |

- Abstract: 207→186 words. Compliance audit: ALL PASS.
- .docx regenerated (pandoc → fix_tables → format headings).
- **Next: Pak Amien verify .docx in Word → create account submission.archcalc.cnr.it → upload 4 files**

### 3. P11 Archipel — Major Expansion (v0.4 → v0.5)
| Aspect | v0.4 | v0.5 |
|--------|------|------|
| Words | 2,600 | ~5,300 |
| References | 17 | 29 |
| Japan section | ~250 words | 1 sentence |
| AI disclosure | Standalone section | Footnote |
| Sacred geography | Absent | New subsection (Meru, pascima, intervisibility) |
| 929 CE narrative | 12 lines | ~400 words |
| Heritage implications | Absent | New section (mining, BPCB, GPR targets) |
| Lombard engagement | Name-drop | Monde du village, Austronesian substrate |
| Christie engagement | Absent | States without cities → Two Javas correlate |
| Bloembergen/Eickhoff | Absent | Colonial monumental bias + postcolonial persistence |

- PDF (13pp) + DOCX generated. No [EXPAND] markers remaining.
- **BELUM SELESAI:** Perlu cek submission guidelines Archipel (template, format referensi, abstract bilingual?) dan screen paper yang sudah terbit. Task ini INTERRUPTED — lanjutkan di sesi berikutnya.
- **Next: Cek guidelines Archipel → sesuaikan format → Pak Amien review → email archipel@ehess.fr**

### 4. Doc Sync Audit
- L2_STRATEGY.md: P1→Under review (EGQSJ), P2 APC £593 noted, P5→Asian Ethnology, P11→Archipel, P17→ArchCalc confirmed, P8 arXiv. Header→2026-04-06.
- L3_EXECUTION.md: P11 rejection + Archipel retarget, P17 updated. Header→2026-04-06.

---

## YANG BELUM SELESAI

| Item | Status | Next |
|------|--------|------|
| **P11 Archipel guidelines** | INTERRUPTED | Cek submission guidelines + screen published papers → sesuaikan format |
| P17 verify + upload | Files ready | Pak Amien: verify .docx → create account → upload |
| P11 submit | v0.5 ready, format TBC | Pak Amien review → sesuaikan format → email |
| JCAA waiver | Email sent | Wait Verhagen reply |
| GitHub go public | UNBLOCKED | Pak Amien: flip switch |
| Go Frendi arXiv | Password: ze47x | Pak Amien: share |

---

## FILES CHANGED/CREATED

```
MODIFIED:
  docs/JOURNAL.md                — Session 11 entries
  docs/WORKSTATE.md              — Updated priorities, JCAA status, P11/P17 status
  docs/L2_STRATEGY.md            — Pipeline updated to 2026-04-06
  docs/L3_EXECUTION.md           — P11/P17 status, timestamp updated
  papers/P17_two_javas/draft_v0.3_archcalc.tex — 5 critical fixes
  papers/P11_volcanic_informedness/cover_letter_archipel.md — Date filled

CREATED:
  papers/P2_settlement_model/jcaa_waiver_reply_verhagen.md
  papers/P11_volcanic_informedness/draft_v0.5_archipel.tex
  papers/P11_volcanic_informedness/draft_v0.5_archipel.pdf
  papers/P11_volcanic_informedness/draft_v0.5_archipel.docx
  papers/P11_volcanic_informedness/email_archipel_submission.md

REGENERATED:
  papers/P17_two_javas/archcalc_submission/P17_manuscript_formatted.docx
  papers/P17_two_javas/archcalc_submission/P17_manuscript.docx
```

---

*"Peluru terbaik, tapi menghunjam ke jantung — bukan banyak peluru yang meleset."*

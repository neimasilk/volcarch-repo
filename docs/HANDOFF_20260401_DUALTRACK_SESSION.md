# HANDOFF: Dual-Track Strategy Session (2026-04-01)

**Dari:** Claude (sesi otonom)
**Untuk:** Pak Amien
**Durasi:** ~2 jam otonom setelah P11 rejection handling

---

## RINGKASAN 30 DETIK

P11 ditolak Cornell (desk rejection, scope mismatch). Dari rejection pattern, lahir **dual-track strategy**: paper teknis (NLP) + paper humanities (narasi). Saya sudah reframe P11 untuk Archipel, analisis P5 untuk Asian Ethnology, dan sync semua dokumen.

---

## KEPUTUSAN STRATEGIS

### Dual-Track Publication Strategy (BARU)
| Track | Keahlian | Target jurnal | Bahasa |
|-------|----------|---------------|--------|
| **NLP/Technical** | Pak Amien | ArchCalc, DHQ, EGQSJ, JCAA | Methodology-led |
| **Humanities** | Claude bantu framing | Archipel, BKI, Wacana, Asian Ethnology | Heritage/narrative-led |

Data sama, argumen sama, bahasa berbeda. Bukan duplikasi — translasi untuk audience berbeda.

---

## DELIVERABLES HARI INI

### 1. P11 Reframe untuk Archipel — SELESAI
- **File:** `papers/P11_volcanic_informedness/draft_v0.4_archipel.tex` + `.docx` + `.pdf`
- **Judul baru:** "Temples Without Villages: Candi and the Hidden Settlement Geography of Volcanic Java"
- **Perubahan:** Lead with heritage, tambah Lombard/Wolters/Miksic, Monte Carlo jadi supporting evidence
- **Cover letter:** `cover_letter_archipel.md`
- **Submission prep:** `ARCHIPEL_SUBMISSION_PREP.md`
- **ACTION:** Pak Amien baca → email ke archipel@ehess.fr

### 2. P5 Humanities Reframe Analysis — SELESAI
- **File:** `papers/P5_volcanic_ritual_clock/HUMANITIES_REFRAME_STRATEGY.md`
- **Target:** Asian Ethnology (Nanzan U, zero APC, Scopus Q2)
- **Reframe:** "taphonomic calibration" → "indigenous knowledge resilience through structural invisibility"
- **ACTION:** Baca strategy doc → mulai rewrite ~Juni 2026

### 3. P17 ArchCalc Formatting — HAMPIR SELESAI
- Experiment count updated 162→175
- Bibliography extracted ke ArchCalc format
- **7 manual steps tersisa** (paragraph numbering, heading format, account creation)
- **ACTION:** Lihat checklist di `ARCHCALC_RULES.md`

### 4. Repo Go-Public Readiness
- `.claude/settings.local.json` DIHAPUS dari git (personal paths)
- `.claude/` ditambah ke `.gitignore`
- README updated ke 175 experiments + ME#12 results
- GPS coordinates OK (2-3 decimal, ~100m-1km, predictions bukan site locations)
- **EMAIL LAMA:** Ada beberapa file pakai ubhara.ac.id, umm.ac.id — belum distandardize (next session)

### 5. Housekeeping
- Root loose files dipindah (copernicus→P1, delpher→E125, cloudflare→docs)
- Semua doc sync: L1, L2, L3, EVAL → 175 experiments
- JCAA waiver email: fixed email address
- Zenodo metadata untuk E171 prediction registry — READY
- JOURNAL updated
- DISSEMINATION_ROADMAP updated with dual-track

---

## YANG BELUM SELESAI

| Item | Status | Priority |
|------|--------|----------|
| Email standardization (ubhara→ubhinus) | Belum | Sebelum go-public |
| P17 paragraph numbering + heading format | Manual | Sebelum submit |
| Archipel submission (email) | Menunggu review Pak Amien | Tinggi |
| JCAA APC email | Menunggu Pak Amien kirim | Tinggi |
| Zenodo deposit E171 | Manual upload | Sedang |
| P5 full humanities rewrite | Perlu sesi khusus | ~Juni 2026 |
| Repo go public (GitHub settings) | Manual | Setelah email fix |

---

## GIT COMMITS HARI INI (4 commits)

```
7e0163e fix: L1 experiment count 153→175, final doc sync
9485a8a chore: add .claude/ to .gitignore, fix JCAA email address
2a2540e docs: README updated to 175 experiments, WORKSTATE priorities rewritten for dual-track
aa69980 feat: dual-track strategy — P11 Archipel reframe, P5 humanities analysis, doc sync
115b247 feat: ME#11 closeout + ME#12 pipeline-driven session — 28 experiments (E124-E175), 175 total
```

---

*"Data yang sama, cerita yang berbeda. Satu untuk yang menghitung, satu untuk yang mendengar."*


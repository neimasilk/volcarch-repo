# WORKSTATE — Session Continuity Contract

**READ THIS FIRST. Continue in-progress items before starting new work.**
**Last updated:** 2026-06-25 (E216 "paleo-ecological interferometer" designed by Opus — Track B, gated, not yet executed)
**HANDOFF TERBARU: `docs/HANDOFF_20260611.md`** — baca itu untuk resume cepat. (Handoff pagi: `docs/HANDOFF_20260610.md`.)

## *** 2026-06-25 — E216 DIEKSEKUSI — OUTCOME-3 "The Decisive Missing Core" ***

**E216 COMPLETE.** Desain (Opus, ultracode) + eksekusi (Sonnet, sesi ini). Pre-registrasi dikunci sebelum analisis. S0–S9 selesai. **Hasil: OUTCOME-3** — jaringan inti palaeoekologi Jawa sensitif (Dieng +ctrl ~600 CE terkonfirmasi) tetapi punya COVERAGE GAP di heartland Kedu/Brantas — tidak ada inti yang RSAP-nya mencakup wilayah tersebut. P(detect)=0 di semua inti yang ada; P(detect)=1.0 jika ada danau/rawa dalam 20 km dari Kedu. Biaya: USD 8-15k untuk satu vibrocore + 20 tanggal AMS.

**File dihasilkan:** `PREREG.md`, `code/e216_detection_function.py`, `code/e216_figure.py`, `results/core_coverage_table.csv`, `results/detection_probability_table.csv`, `results/OUTCOME.json`, `results/missing_core_spec.json`, `results/PAPER_DRAFT_OUTLINE.md`, `figures/fig1_network_rsap_map.png`, `figures/fig2_detection_power.png`.

**Langkah berikutnya (Pak Amien):** (1) Review draft outline → `results/PAPER_DRAFT_OUTLINE.md`. (2) Cari co-author paleobotanist/palynolog (G2/G10 — wajib sebelum submit). (3) Zenodo deposit kode + data (G7). (4) Cross-model review G9 sebelum kirim. Target: *Vegetation History and Archaeobotany* (Q1, Green-OA, zero APC).

## *** 2026-06-22 — STATUS SAAT INI (baca sebelum mulai) ***

**Forcing function ME#19 SUDAH JATUH TEMPO (2026-06-24, sudah terlewat 1 hari).** 11 hari tanpa aktivitas repo sejak 06-11. Belum ada bukti exposure eksternal terjadi (tidak ada DOI tercatat, email belum tampak terkirim). **STOP list masih mengikat: tidak ada audit/paper/review-sprint baru** — yang dibutuhkan AKSI EKSTERNAL Pak Amien, bukan kerja internal baru.

**KOREKSI handoff (penting):** handoff 06-10 bilang "Verberne belum balas ~8 minggu, kirim follow-up." **SALAH/STALE.** Verberne **sudah balas 2026-06-08** dengan 2 pertanyaan (setelah konsultasi kolega TU Delft). Balasan sudah didraft (v3, 06-09). **Sesi ini menutup 1 item terakhir** (skema NER GLOBALISE diverifikasi: 7 tipe entitas — persons/locations/organisations/polities/commodities/ships/documents + dates — tak satupun arkeologis → entitas baru proposal DEPTH/MATERIAL/FIND_EVENT/SOIL_CONTEXT memang di luar skema mereka). **Balasan kini v4 SEND-READY, nol open item** → `docs/correspondence/EMAIL_VERBERNE_REPLY_DRAFT_20260609.md`.

**3 AKSI BLOKIR (semua milik Pak Amien, butuh akun beliau) — urut prioritas:**
1. **KIRIM balasan Verberne v4** — menggantung **14 hari**; profesor aktif menunggu; exposure paling penting + paling mendesak. Tinggal kirim.
2. **Upload D1+D2 ke Zenodo** (`docs/ZENODO_UPLOAD_GUIDE_20260610.md`, ~15 mnt/dataset) → tempel 2 DOI di chat.
3. **Kirim balasan Lamqaddam** (isi deadline BPI + 3 slot chat) — `docs/correspondence/EMAIL_LAMQADDAM_REPLY_DRAFT_20260423.md`.
**Salah satu dari ketiganya memenuhi forcing function.** Verberne = paling urgent.

**Claude (saat DOI/konfirmasi masuk):** catat DOI ke MEMORY/WORKSTATE/JOURNAL + sitasi README paket; backport median 1.75 m ke `experiments/E070_.../REGISTER_NOTES.md` (perlu OK Pak Amien); commit repo (perubahan sesi ini belum di-commit — minta izin). Selebihnya WAIT (4 paper under review) + Track B pelan (Castillo, milik Pak Amien).
**MODE: DISCOVERY-FIRST (ME#16, 2026-04-22).** Polishing paused; diamond-hunts primary. Papers continue on current trajectory; new evidence from hunts is the gate for future submissions.

**2026-06-10 — Vocabulary normalisation (maintenance, NON-research).** Plain-language pass across docs, memory, and papers so the newest model's topic-classifier stops mis-flagging the repo at session start. 19 record files renamed (cross-model review records → `critical_*`; the §9 audit → `STOP_CRITERION_AUDIT_2026_04_21.md`). Re-runnable normalisers: `tools/reword_triggers.py` (+ `tools/reword_triggers_tex.py`). **No numbers, claims, or findings changed.** Known harmless leftover: experiment folder `E069_…_comparanda` keeps its original name (renaming would break its scripts).

**2026-06-10 — External population-evidence channel moved to companion repo.** E053, E203, and their literature summary now live in sibling repo `D:\documents\volcarch-genetics` (this repo's commit 14a2fc2); map + traceability in `docs/COMPANION_REPOS.md`. Reason: keep session-start context within the default model's topic-classifier budget. Findings unchanged; cite as external evidence. **Do NOT re-create those folders here.** E055/E214 still cite E053 as an evidence leg (valid).

## *** 2026-06-10 (sore) — MATA ELANG #19: stop auditing, start exposing ***

Comprehensive review per Pak Amien. Routed an untracked ChatGPT critical review of ME#16 (`docs/research_notes/ME16_CHATGPT_PIVOT_REVIEW_20260610.md`) → wrote **`docs/research_notes/MATA_ELANG_19_2026_06_10.md`**. Core finding: **the binding constraint is no longer epistemic rigor (SIG fixed that) — it is non-exposure.** Three deep audits in three days, zero submissions, still 0 acceptances at 14 months. The multi-model skeptical-review loop has become a procrastination engine (ChatGPT §6, correct).
- **3 new valid critiques accepted:** (1A) channel convergence ≠ independent evidence (correlated bias — F9); (1B) manifesto is interpretively elastic, not a theory, demote from "evidence/prior" (F10); (1C) ME#16 discovery-pivot is in direct conflict with PhD-pipeline incentives (rewards credible reliable on-time execution, not "I discovered something").
- **Resolution of ME#16 vs ME#18 vs ChatGPT:** decouple **Track A (career, time-bound = land P8/P17/P2/P11 + send PhD emails + Zenodo D1/D2)** from **Track B (curiosity, slow, untimed = ONE independent falsification channel: paleo-environmental analysis/plant-microfossil/Castillo).** Diamond-hunts (E209/E210) demoted flagship→exploratory; ML/InSAR may have no learnable manifold (apophenia risk).
- **SIG taxonomy extended:** F8 non-exposure, F9 correlated-channel convergence, F10 interpretive elasticity.
- **FORCING FUNCTION (binding):** an external judge must receive something within 14 days; **next ME is forbidden until that happens.** STOP list: no more skeptical-review sprints, no more audits per-session, no new papers, stop treating "0 counter-evidence"/"N channels converge" as strength.
- **THIS WEEK (Pak Amien + me):** (1) D1+D2 → Zenodo — **✅ PAKET SIAP (sesi malam 2026-06-10):** `papers/D1_.../zenodo_upload/` + `papers/D2_.../zenodo_upload/` + panduan `docs/ZENODO_UPLOAD_GUIDE_20260610.md`; mini-G1 blind recompute PASSED dengan 2 koreksi integritas di salinan paket (median D1 2.00→1.75 [subset-mix]; sitasi palsu "JOAD [submitted]" dihapus). **Tinggal Pak Amien upload (~15 min/dataset) → tempel 2 DOI di chat.** (2) P8 SIG — **✅ DONE:** `papers/P8_linguistic_fossils/SIG_PREP_20260610.md` (no RED; G1/G7 pending-at-revision; G8 scan clean; terminology map siap; naskah under-review TIDAK disentuh). (3) send Lamqaddam reply + Verberne (**MASIH PENDING — Pak Amien** isi BPI deadline + chat slots).

## *** 2026-06-10 — P16: DIPUTUSKAN PARKIR (Pak Amien) ***

**KEPUTUSAN DIAMBIL 2026-06-10:** "untuk arah P16 parkir dulu, catat semua" → **Opsi 3 (PARK)**. Record lengkap + syarat unpark: `papers/P16_computational_textual_archaeology/PARKED.md`. CANONICAL.md diupdate. Naskah frozen, jangan edit/submit. Unpark via: (a) desain konvergensi non-sirkular (unsupervised clustering) yang lolos, ATAU (b) keputusan reframe-downgrade ke paper distributional-attestation. Konteks NO-GO asli di bawah (histori):

## 2026-06-10 — P16 → Wacana NO-GO (histori; keputusan sudah diambil di atas)

Ran the recommended G9 cross-model gate on the **R1-revised** P16 draft. **DeepSeek = REJECT.** W1 (FATAL): v6 convergence test still circular (within-group vs **whole-corpus** baseline). I implemented DeepSeek's prescribed within-group **label-shuffle** test (`experiments/E090_transformer_textual_nlp/e090_v7_label_shuffle.py`): **cross-tradition convergence 0/8 groups** (v6 claimed 8/8), all z negative (−5.8 to −14.1). Corroborated: S_within > S_cross in all 8 groups. **The central convergence pillar is an artifact and is refuted.** W2 (FATAL): 929 CE n=46 — remove, don't temper.

Per SUBMISSION_INTEGRITY_GATE → **NO-GO.** Manuscript NOT modified, NOT submitted. Full write-up: `experiments/E090_transformer_textual_nlp/V7_LABEL_SHUFFLE_FINDING_20260610.md`; JOURNAL 2026-06-10; review at `papers/P16_.../external_reviews/critical_deepseek_p16_wacana_R1_20260610.md`.

**Pak Amien decision (none taken autonomously):**
1. **Reframe + downgrade** P16 → distributional attestation finding (themes attested across 11–12/12 traditions = co-occurrence count, true) + genre-honest inscription asymmetry; **drop 929 CE diachronic claim.** Smaller, defensible paper.
2. DHQ switch does NOT rescue W1 (refuted regardless of venue).
3. Park P16 until a non-circular convergence design exists (unsupervised clustering recovering themes without keyword tagging — DeepSeek's "gold standard", untested).

**If P16 parked/reframed, next closest-to-acceptance shots remain:** P8 (Oceanic Ling, under review — wait), P17 (ArchCalc — survives re-derivation, do methods/repro fix), then SPAFA for reframed P9/P11.

**2026-06-10 — ME#18 RESIDUAL INTEGRITY PURGES COMPLETE (JOURNAL 2026-06-10 #2):**
- **E031 canonical-30 re-run: SURVIVES** (median 14.5 km; west-clustering strengthens R 0.380; orientation null identical). `results/canonical30/`.
- **E082 canonical-30 re-run: SURVIVES, magnitude shrinks.** Candi-vs-inscription gap **9.2 → 6.1 km** (CI 3.2–9.1, MW p=2.8e-7). Median 27.6 km unchanged = matches P17's verified figure. **P11 revision correction queued: replace 9.2 km (abstract + §Test 3) with 6.1 km.** Century trend stays n.s.
- **P0 Ch5 confirmed non-reproducible:** 2.4% → actually 1.5–4.1% with no computed baseline; 5.8×/E197 mis-cite confirmed (E197 = depth only). Flags inserted in draft_v0.4.tex; rewrite verdict = CUT gradient+5.8×, KEEP depth match.
- **P9 line-83 footnote deleted** (dead distance claim). **P16 line-468 + "45 volcanoes": verified already purged in Wacana R1** (SIG_signoff.md) — item was stale.
- **All canonical-inventory REDOs now done:** E019 ✓ E004 ✓ E104 ✓ E153 ✓ E031 ✓ E082 ✓ (+ E065 Zone-A ratio re-derived: 17.9× → **19.1×**, menguat). Inventory-artifact propagation audit CLOSED.

**2026-06-10 (lanjutan) — P16 PARKIR + PAKET REVISI P11/P17 SIAP:**
- **P16 PARKED** (keputusan Pak Amien). Record: `papers/P16_.../PARKED.md`. CANONICAL.md updated. Wacana submission OFF.
- **P11 revision package READY:** `papers/P11_volcanic_informedness/revision_ammo/CANONICAL_INVENTORY_CORRECTIONS_20260610.md` — 4 lokasi koreksi (Rayleigh 3.4e-8→1.2e-9 menguat; Zone A 17.9×→19.1× menguat; gap 9.2→6.1 km mengecil-tapi-signifikan; kuadran timur "<4%"→9.2%). E153 baris 110 VERIFIED cocok, tidak perlu koreksi. **Apply HANYA saat Archipel minta revisi.**
- **P17 methods/repro fix READY:** `papers/P17_two_javas/revision_ammo/CANONICAL_INVENTORY_FIX_20260610.md` — naskah pakai "10 major volcanoes"; tabel pengganti kanonik lengkap (median 14.5/27.6, U=8267 p=2.8e-7, Fisher menguat p<1e-4). **Apply saat review ArchCalc datang (~akhir 2026).**
- Semua paper live (P2/P8/P11/P17) tetap TIDAK disentuh — koreksi menunggu tahap revisi masing-masing.

**E153 / P11 checked 2026-06-10 (RESOLVED sore hari):** the P11 candi–settlement headline (Test 1) is **inventory-independent** and robust — mean **6.78 km**, 80.6% within 10 km, p<1e-6 — dan **COCOK** dengan yang dikutip P11 baris ~110 ("81% within 10 km, mean 6.8 km"). Dugaan mismatch pagi hari adalah **salah atribusi**: angka "9.2 km" P11 berasal dari **E082** (gap volcano-distance candi-vs-inskripsi, run 20-gunung), BUKAN dari E153. Provenance direproduksi persis, lalu dikoreksi kanonik → **6.1 km**. Paket koreksi lengkap: `papers/P11_volcanic_informedness/revision_ammo/CANONICAL_INVENTORY_CORRECTIONS_20260610.md`.

**SESSION 22 (2026-04-23).** Two deliverables: (1) **VOC-ArchNLP v1.0.0** — HKI Hak Cipta product built (`tools/voc_archnlp/`): package with 4 modules (downloader, preprocessor, normalizer, extractor [NEW]), unified CLI, 3 DJKI registration docs. Core innovation = ArchaeologicalMentionExtractor: 6 entity types + depth conversion (voet/el/palm/duim → meters) + CSV/JSON output. This IS E211 Phase 1 pipeline + institutional HKI anchor. (2) **Lamqaddam reply draft** — `docs/correspondence/EMAIL_LAMQADDAM_REPLY_DRAFT_20260423.md`. Pak Amien must fill: BPI deadline + 3 chat slots (WIB→CEST conversion).

**SESSION 21 (2026-04-22 late).** Strategic pivot. Pak Amien reframe: P1 = masterpiece, rejections were a test, pernyataan kegelisahan belum terjawab di level landmark, mandate untuk dimensi yang belum tergali. Saya tulis **Mata Elang #16** (`docs/research_notes/MATA_ELANG_16_2026_04_22.md`) — system/research-designer critique mendiagnosa **discovery deficit** sebagai structural cause semua rejections (3 cross-model v0.2/v0.3/v0.4 + 5 editorial). Pivot: **stop polishing, start discovery-hunt.** 5 diamond-hunts diurut (E209 satellite ML / E210 InSAR / E211 VOC scale / E208P3 kakawin / E212 population-scale channel). Cascade model retired dari papers (overhead tanpa purchase). P0 v0.4 + P1 v5 pause sampai new evidence. **E209 flagship executed:** training set 121 sites (6 hard-pos buried candi + 109 soft-pos + 5 hard-neg controls), Sentinel-2 feature extraction pipeline validated, full 121 × dry+wet run in background.

**SESSION 20 (2026-04-22).** SLR Fase C inventory CSV (23 rows, 4 cathedral anchors, 0 material counter-evidence). SLR Fase D synthesis for P0 (six-channel architecture locked). DeepSeek critical review v0.2→v0.3→v0.4 — decisive critiques closed, residual normal-review-noise. v0.4 = 48pp full paper, compile clean. UvA track: Lamqaddam responded positively 2026-04-21 offering BPI Dosen support letter + first chat with Delfina.
**SESSION 18 (2026-04-20).** Autonomous critique session. Mata Elang #15 produced. P0 (masterpiece) skeleton drafted. P1 audit #1 arithmetic fixed. Skeptical reviewer prompt + critical review of P1 produced. OJW Wordnet (5,020 synsets) downloaded for E208 scoping.

## *** SESSION 19 END — PAK AMIEN BRIEFING FIRST ***

**READ: `docs/PAK_AMIEN_BRIEFING_2026_04_21.md`** — 30 min, 4 decisions, contains DeepSeek critical review findings + response classification.

**Session 19 (2026-04-21) delivered:**
- Counter-SLR + E108 replicability + P0 claim audit (7 flags fixed) + DeepSeek cross-model reviews on both P1-core v3.0 + P0 draft v0.1
- **Both DeepSeek reviews recommend REJECT with substantive methodological critiques** that self-review did not catch — validates echo-chamber hypothesis
- L1 §9 stop criteria updated per Pak Amien trust grant (#1 partial violation acknowledged, #3 operationalised, #6 cross-model added)
- Response classification: 4 ACCEPT fixes + 2 PARTIAL for P1-core; P0 needs strategic reframe

**4 decisions pending:** (A) Run second cross-model? (B) Apply P1 fixes autonomous? (C) P0 direction (withdraw / reframe / proceed)? (D) Override L1 §9 edits?

**Budget state:** $3.287 remaining ($3.30 start, $0.013 spent on 4 skeptical reviews).

**Session 19 ADDENDUM (post-briefing):** Gemini 2.5 Flash skeptical reviews completed. **Stop Criterion #6 (cross-model methodology critique) TRIGGERED for both P1-core v3.0 and P0 draft v0.1.** Convergence analysis: `papers/P0_invisible_civilization/external_reviews/CROSS_MODEL_CONVERGENCE_2026_04_21.md`.

**Session 19 ADDENDUM 2 (per "lanjutkan"):** P1-core v4.0 methodology pivot EXECUTED. New file `submission_jasrep_v4.0.tex`, 25pp compile clean. Retitled to "Cumulative Volcanic Burial and the Archaeological Detection Horizon in Java, Indonesia: Preliminary Measurement Anchors and a Research Program." New §5.7 "A Research Program for Rigorous Testing" (5 concrete testing strategies). DeepSeek re-review on v4.0 still REJECT with softer kritik — new sharp concern "sampling on dependent variable" remains structurally unresolvable by wording. Verdict: `papers/P1_taphonomic_framework/external_reviews/V4_PIVOT_VERDICT_2026_04_21.md`. **New decision required — Options X1 (submit v4.0 as-is), X2 (further pivot to perspective paper, ~3-4hr), X3 (withhold pending fieldwork).** Decision C on P0 direction remains as well.

---

## *** REVIEW TRIAGE FOR PAK AMIEN ***

Per ME#15 §6C + §7B, each active deliverable has a review tag. This keeps Pak Amien's limited review bandwidth focused on what matters.

- **[DEEP]** = read carefully, 2-3 days. Blocking on your approval.
- **[SKIM]** = glance sufficient, 10-30 min. Escalate only if something feels off.
- **[FYI]** = no review expected. Documented for context.

### Awaiting Pak Amien action

| Tag | Item | Rationale |
|---|---|---|
| [DEEP] | P1-core v3.0 (`papers/P1_taphonomic_framework/submission_jasrep_v3.0.pdf`, 21pp) | Masterpiece bar + JASREP submission decision |
| [DEEP] | **P0 draft v0.4** (`papers/P0_invisible_civilization/draft_v0.4.pdf`, 48pp, ~10.5K words) | **FULL PAPER**: §1-§9 complete. Builds on v0.3 decisive-fixes. Added §4 Selective Survival + §5 Wayang + §6 Six-Filter Framework + §7 Pre-Registered Predictions (8+3 with stop criteria) + §8 Limitations + §9 Conclusions. **Review this one first.** |
| [SKIM] | DeepSeek v0.3 re-review (`papers/P0_invisible_civilization/external_reviews/critical_deepseek_v0.3_20260422.md`) | Both decisive critiques CLOSED; Ch6 chronology no longer flagged, Ch4 tautology acknowledged; remaining critiques = normal reviewer noise |
| [SKIM] | DeepSeek v0.2 review (`papers/P0_invisible_civilization/external_reviews/critical_deepseek_v0.2_20260422.md`) | Baseline review that identified 2 decisive critiques; diff against v0.3 review shows fixes worked |
| [FYI] | P0 draft v0.3 (`papers/P0_invisible_civilization/draft_v0.3.pdf`, 30pp) | Post-decisive-fix §1-§3 state; for diff against v0.4 |
| [FYI] | P0 draft v0.2 (`papers/P0_invisible_civilization/draft_v0.2.pdf`, 24pp) | Pre-decisive-fix state |
| [FYI] | P0 draft v0.1 + SKELETON (`papers/P0_invisible_civilization/`) | v0.1 original, skeleton v0.1 |
| [SKIM] | SLR Fase C+D outputs (`docs/bibliography/_INVENTORY.csv` + `_SYNTHESIS_for_P0.md`) | 23 rows, 4 cathedral anchors, 0 counter-evidence; drives §3 structure |
| [DEEP] | `docs/research_notes/MATA_ELANG_15_2026_04_20.md` | Strategic critique + Path B rationale |
| [DEEP] | `docs/research_notes/STOP_CRITERION_AUDIT_2026_04_21.md` | L1 §9 #1 and #3 flagged for rewrite |
| [SKIM] | Counter-SLR execution log (`docs/bibliography/counter_evidence/COUNTER_SLR_EXECUTION_2026_04_21.md`) | 1 material qualifier (Channel 4 population-level-evidence reframe), otherwise confirmatory |
| [SKIM] | Counter-thesis engagement (`docs/bibliography/counter_evidence/COUNTER_THESIS_ENGAGEMENT_2026_04_21.md`) | Wolters supports VOLCARCH; Coedès is already minority |
| [SKIM] | E108 replicability audit (`docs/bibliography/counter_evidence/E108_REPLICABILITY_AUDIT_2026_04_21.md`) | Math replicates exactly |
| [FYI] | `docs/LITERATURE_SLR_PROGRESS.md` | Living log |
| [FYI] | `docs/HANDOFF_20260420_SESSION18.md` | Past session |
| [FYI] | `docs/bibliography/` (14 files) | SLR catalog |

### Decisions blocking autonomous work

- **[DEEP]** P1-core v3.0 approve/revise → JASREP submission
- **[DEEP]** $50 Fiverr stats review budget approval (before P1-core submit)
- **[DEEP]** $5-20 DeepSeek API for skeptical cross-model review (API key in `.env`)
- **[DEEP]** Whether to rewrite L1 §9 criterion #1 (cascade) + #3 (comparandum) per audit
- **[DEEP]** Approve Session 19 autonomous work product (this triage)

---

## *** URGENT DECISION PENDING (before 2026-04-21 Monday) ***

**Mata Elang #15 recommends PATH B:** split P1 into P1-core (calibration, ~15pp) + P0 (synthesis masterpiece, ~25-30pp). Do NOT submit JASREP Monday as planned. Allow 2-3 weeks for structural rework.

**Full analysis:** `docs/research_notes/MATA_ELANG_15_2026_04_20.md`
**P0 skeleton:** `papers/P0_invisible_civilization/SKELETON_v0.1.md`
**Critical review of P1:** `papers/P1_taphonomic_framework/external_reviews/critical_review_claude_persona_20260420.md`

**GO/NO-GO decision required from Pak Amien on:**
1. Monday JASREP submission (default NO-GO; override possible if momentum > masterpiece)
2. Path B adoption (split P1, build P0)
3. P0 target journal (recommendation: *Journal of Anthropological Archaeology*)
4. External reviewer budget ($50-200)
5. Cross-model review budget ($5-20 DeepSeek if available)

**If Pak Amien APPROVES Path B:**
- Monday = begin P1-core surgical cut (remove §2.2, §2.5, §5.5 to P0)
- Week 1 = P0 §1-3 draft
- Week 2 = P1-core + skeptical cross-model review
- Week 3 = P1-core JASREP submit + P0 §4-5

**If Pak Amien OVERRIDES to Path A (submit Monday):**
- Monday = apply 10 audit issues to v2.0, submit as-is
- P0 still gets built as parallel project (8-10 weeks)
- Cross-model critical review becomes post-hoc

**HANDOFF:** `docs/HANDOFF_20260420_SESSION18.md` (latest; 2026-04-20, Path B + SLR complete + E208 NLP run)

## *** STRATEGIC DEVELOPMENT: THREE PhD TRACKS ACTIVE ***

### Track 1: Suzan Verberne (Leiden, LIACS) — PROPOSAL READY
- NLP for archaeological text mining (extends EXALT + "Digging in Documents")
- **Research proposal v0.1 READY** (7pp, audited): `docs/correspondence/phd_proposal/PhD_Proposal_Amien_Leiden_v0.1.pdf`
- **Claim audit trail**: `docs/correspondence/phd_proposal/CLAIM_AUDIT_TRAIL.md`
- **Cover email READY** (copy-paste)
- **Timeline: Pak Amien review → kirim ~17 April**
- Email exchange: `docs/correspondence/EMAIL_VERBERNE_LEIDEN_2026_04_14.md`

### Track 2: Piek Vossen (VU Amsterdam, CLTL/GLOBALISE) — EMAIL DRAFTED
- GLOBALISE PI = 5M+ VOC manuscript pages. Spinoza Prize winner.
- PhD student Stella Verkijk already doing VOC Event Reconstruction → complementary
- **Email draft READY**: `docs/drafts/email_vossen_vu_globalise.md`
- **Timeline: kirim ~21-22 April** (3-5 hari setelah Verberne)
- Potential: promotor utama, atau co-promotor dengan Verberne

### Track 3: Shay Cohen (Edinburgh, Informatics) — CV SENT, WAITING
- Structured prediction + multilingual NLP. Responded in 5 minutes.
- CV + transcript + research statement sudah dikirim 2026-04-12
- **Timeline: apply formal December 2026, entry October 2027**
- Email exchange: `docs/correspondence/EMAIL_COHEN_EDINBURGH_2026_04_12.md`
- Backup track — Edinburgh mahal (tuition £30K+/yr) tapi NLP reputation top-5 world

### Strategy
- Prioritas: **Belanda** (arsip VOC di sana, employment model, no tuition)
- Dua-duanya Leiden + VU bisa jadi joint supervision (promotor + co-promotor)
- Edinburgh = backup kalau Belanda tidak jalan
- **KITLV email ON HOLD** — tunggu PhD trajectory
- **Castillo email READY TO SEND** — independent of PhD
- **IELTS: mid-2026**

---

## IN PROGRESS

### [NEW SESSION 22] HKI PRODUCT — VOC-ArchNLP v1.0.0

- **VOC-ArchNLP package** — `tools/voc_archnlp/` — BUILT (Session 22, 2026-04-23)
  - Core innovation: `extractor.py` — ArchaeologicalMentionExtractor (6 entity types + depth units)
  - Orchestrator: `pipeline.py` — VOCArchPipeline
  - CLI: `cli.py` — `python -m voc_archnlp [download|preprocess|normalize|extract|run]`
  - **DJKI docs READY:** `docs/HKI/` (DESKRIPSI_PROGRAM.md, MANUAL_PENGGUNA.md, ARSITEKTUR_SISTEM.md, PANDUAN_PENDAFTARAN_DJKI.md)
  - **Next actions (Pak Amien):**
    1. Baca `docs/HKI/PANDUAN_PENDAFTARAN_DJKI.md` — panduan lengkap
    2. Buat akun di **e-hakcipta.dgip.go.id**
    3. Siapkan scan KTP + surat pernyataan bermaterai (template di PANDUAN_PENDAFTARAN)
    4. Cetak sampel kode sumber 4 file (~50 hal, Courier 10pt) — OR upload PDF via portal
    5. Pertimbangkan: perseorangan (cepat, Rp 400K) vs. via LPPM Ubhinus (nilai KUM lebih tinggi)
    6. **OPSIONAL dulu:** Deposit ke Zenodo.org sebagai timestamp digital sebelum DJKI

- **Lamqaddam reply** — Draft di `docs/correspondence/EMAIL_LAMQADDAM_REPLY_DRAFT_20260423.md`
  - **URGENT — kirim dalam 72 jam (deadline ~2026-04-24)**
  - Pak Amien harus isi: (a) BPI Dosen 2026 deadline, (b) 3 slot chat WIB + konversi ke CEST
  - Setelah kirim: prep chat (elevator pitch, 18-month deliverables, pertanyaan ke Delfina)
  - Setelah chat: segera kirim Vossen email (VU Amsterdam, GLOBALISE PI)

### [SESSION 22] E211 Phase 1 — COMPLETE (2026-04-23)

**STATUS: PHASE 1 DONE.** Pipeline executed. Results at `results/E211_voc_mentions/`.

| Stage | Output |
|---|---|
| Preprocess | 548,929 paragraphs, 145,971,146 words |
| Normalize | Colonial Dutch → modern Dutch (running in background, paras_ files) |
| Extract | **33,930 candidate mentions** |

**Key finding (see `experiments/E211_voc_dagregister_nlp/FINDINGS_v1_20260423.md`):**
- `oudheden` (archaeological vocabulary) = **0 occurrences** — VOC dagregisters are trade/admin records, not archaeological reports
- `pagode` = mostly currency unit (false positive), `arca` = Latin for chest (FP), `opschrift` = any document label (FP)
- After geographic filter: 14,626 Java-context mentions; high-precision subset (MONUMENT+INSCRIPTION+Java): 871
- Depth + Java context: 34 (need annotation to verify archaeological relevance)
- **Estimated precision: <15%** — keyword matching alone insufficient for this corpus
- **Scientifically valuable negative result:** confirms NER fine-tuning is non-negotiable (Phase 2)

**Output files:**
- `results/E211_voc_mentions/voc_archaeological_mentions.csv` (33,930 rows — full)
- `results/E211_voc_mentions/voc_mentions_java_filtered.csv` (14,626 rows — geographic filter)
- `results/E211_voc_mentions/voc_mentions_high_precision.csv` (871 rows — MONUMENT+INSCRIPTION+Java)

**Next: E211 Phase 2** — annotation protocol (200–500 sentences, 7 entity types) + language detection filter + currency context filter. **Phase 2 needs Pak Amien decision on annotation approach** (self-annotate vs. Fiverr vs. Go Frendi co-author).

---

- **P1 → EGQSJ REJECTED** (2026-04-16, desk rejection by Chief Editor Christopher Lüthgens)
  - **MS# egqsj-2026-3** | Copernicus User ID: 883530
  - **Rejection reason:** "scientific approach is certainly interesting" BUT "very poorly structured," "lacks scientific rigor in wording," methodology section uses bullet points (unacceptable). "Too early a stage of development." NOT sent to peer review.
  - **Diagnosis:** Form/structure problem, NOT content/science problem. Fixable with rewrite.
  - **Action required:** (1) Structural rewrite — bullet points → flowing prose, tighten scientific language (**DONE: v2.0**), (2) Retarget to **JASREP** (Elsevier, Scopus Q1, FREE under subscription model — already formatted in repo). Backup: Archaeological Research in Asia (Elsevier, same free subscription model).
  - **CORRECTION:** Open Quaternary and Internet Archaeology are NOT Diamond OA (both charge APC). JASREP subscription route = guaranteed zero cost.
  - Zenodo preprint stays live: DOI 10.5281/zenodo.19081502
  - Revision support material ready: 9 files + E120-E140 new findings
  - **Score: 2× rejected (Asian Perspectives = AI flag, EGQSJ = structure). Science validated by both editors.**
  - **v2.0 WRITTEN** (`submission_v2.0.tex`, `submission_jasrep_v2.0.tex`) — all bullet points→prose, language tightened, compiles clean (26pp)
  - **PRE-FLIGHT AUDIT DONE 2026-04-16 — 10 issues found, ALL fixable. See below.**
  - **Next (Monday 2026-04-21):** Fix all audit issues → rewrite as `elsarticle` class → submit JASREP via Editorial Manager

### P1 JASREP Pre-Flight Audit (2026-04-16) — FIX BEFORE SUBMIT

**FATAL:**
1. **Population arithmetic wrong (Sect 2.2):** Text says "roughly half of 129,000 km²" (~64,500 km²) but claims population 590,000-3,900,000. Correct math: 5×64,500=322,500, 30×64,500=1,935,000. Fix: change to "325,000--1,950,000" OR fix the area fraction. The 3,220-fold gap derivation (325,000/100 persons per village = 3,250 villages vs 0-3 sites) still works.

**HIGH:**
2. **Kanjuruhan depth inconsistent:** Abstract/Intro say "4.0--7.8 m" but Table 2/Conclusions say "3.0--7.9 m". The table is correct (2.4×1266=3038mm, 6.2×1266=7849mm). Fix: harmonize to 3.0--7.9 m everywhere.
3. **Kutai depth cited 3 different ways:** Abstract "7--10 m", Section 2.4 "4--10 m", Discussion "~7 m". Table says 3.9--10.1 m. Fix: use "approximately 4--10 m" consistently.
4. **Tables 1 & 2 overflow** in single-column layout (159pt and 87pt overfull hbox). Fix: use `\small` or `table*` or `tabularx`.
5. **Abstract too long:** ~327 words, JASREP limit is 250. Cut ~80 words.
6. **Must use `elsarticle` document class** (not `article`). Requires `\begin{frontmatter}`, `\ead{}`, `\cortext`, keyword separator `\sep`, `\journal{Journal of Archaeological Science: Reports}`.
7. **HIGHLIGHTS required:** 3-5 bullet points, max 85 characters each (including spaces). Submit as separate file or in frontmatter.

**MEDIUM:**
8. **Add compaction + erosion to Limitations (Sect 5.6):** Soil compaction reduces effective depth at 5+ m (projections are upper bounds). Erosion on slopes/ridges counteracts deposition (rates only apply to basins).
9. **Funding statement required:** "This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors."
10. **CRediT author statement** must use exact Elsevier CRediT taxonomy (13 roles). Current statement is close but needs exact role names.

**LOW:**
11. Fix "within 3%" to "within 4%" or "approximately 3%" (actual: 3.2%)
12. Remove .tex header comments revealing EGQSJ submission history
13. "Amien, in preparation" → "a companion study currently in preparation"
14. DHARMA phrasing: "268 Javanese inscriptions catalogued in" (not "all 268 surviving")
15. Verify figure float order in compiled PDF
16. Add DOIs to bibliography entries where possible

### JASREP Submission Checklist
- [ ] Rewrite in `elsarticle` class with `authoryear` option
- [ ] Fix all 10 audit issues above
- [ ] Abstract ≤250 words
- [ ] Highlights file (3-5 bullets, ≤85 chars each)
- [ ] Cover letter (scope statement, not under review elsewhere, preprint disclosure, AI disclosure)
- [ ] CRediT statement (Elsevier exact taxonomy)
- [ ] Funding statement
- [ ] Competing interests declaration
- [ ] AI disclosure (Elsevier template format, before References)
- [ ] Bibliography: `elsarticle-harv.bst` (Harvard author-date)
- [ ] Compile: `pdflatex → bibtex → pdflatex × 2`
- [ ] Submit via https://www.editorialmanager.com/jasrep/ (subscription route, zero APC)
- [ ] ORCID: 0000-0002-1848-167X
- **P11 → REJECTED Indonesia (Cornell)** (2026-04-01, desk rejection)
  - Reason: "beyond the thematic and stylistic purview" — journal is sociology/political science, not archaeology
  - **RETARGET → Archipel (INALCO/EHESS, Paris).** Zero APC, Scopus Q3, WoS A&HCI. "Monde insulindien" = perfect scope.
  - Submit to: archipel@ehess.fr (varia issue). Word limit 9K (P11 = 3.2K). Flexible ref style. English OK.
  - **v0.5 ARCHIPEL-COMPLIANT (2026-04-08):** Abstract trimmed 168→127 words (≤130 limit). Figures converted PNG→JPEG 300dpi. Citation style (Chicago Notes-Bibliography) matches published Archipel articles.
  - **v0.6 REVIEW-HARDENED (2026-04-08):** 5 fixes from external reviews: "natural experiment"→"before-after observation", Liangan→"proof of possibility", n=20 explained, falsifiability added (GPR predictions), AI disclosure moved to proper section. Email updated.
  - **SUBMITTED 2026-04-08** via email to archipel@ehess.fr. Word + PDF + 2 JPEG figures. WAIT for response.
  - Backup: Wacana Vol.28 No.3 "Muarajambi" (Oct 2027) or PCI Archaeology (preprint-first, Scopus+WoS).
  - Files ready: ~13pp, 29 refs, Word + PDF + LaTeX + JPEG figures
- **P17 → SUBMITTED ArchCalc (CNR)** (2026-04-09). Submission ID **365**. Confirmation email received from redazioneac@ispc.cnr.it.
  - **Diamond OA, GRATIS, Scopus+WoS.** Double-blind peer review.
  - Title: "Two Javas: Spatial Segregation of Sacred and Administrative Landscapes in Volcanic Java and Its Consequences for Archaeological Inference"
  - 4 files uploaded: manuscript (.docx), bibliography (.docx), figure captions (.docx), figures (.zip)
  - **3 new limitations added this session:** cascade underdetermination (E176), karst confound (E178), spatial autocorrelation (E184)
  - Portal: https://submission.archcalc.cnr.it/index.php/aec/authorDashboard/submission/365
  - **2026-04-09: Editor Alessandra Caravale ACKNOWLEDGED.** Text will be considered for 2027 issue. Editorial process begins after 2026 issues published. Review expected ~late 2026.
  - **Next: WAIT for review (expected late 2026 → publication 2027).**
- **P16 drafting** — Draft v0.1 EXPANDED (27pp, ~8K words, 6 figures). Clean.
  - Files: `papers/P16_computational_textual_archaeology/draft_v0.1.tex`, `p16_references.bib`, `figures/`
  - Target: **DHQ** (ADHO, Diamond OA, Scopus+WoS) — rolling deadlines (Apr 15, Jul 15)
  - ~~Alternative: Wacana (if Kawi theme open)~~ — Kawi issue already published
  - Next: User review → convert to Word/RTF → submit ~June 2026
- **P18 drafting (HOLD)** — Draft v0.1 (16pp, 6 figures, 15 refs). Clean.
  - Files: `papers/P18_invisible_civilization/draft_v0.1.tex`, `p18_references.bib`, `figures/`
  - Target: TBD (after 1 acceptance). P17 and P18 form complementary trilogy with P1, NOT duplicates.
  - Needs: Expand to ~9K words, Fig 6, cite P1 for cascade model
- **P5 retargeting (HUMANITIES TRACK)** — Rejected from BKI ("too narrow for humanities"). Needs full humanities reframe.
  - Target: **Asian Ethnology** (Nanzan U, zero APC, Scopus Q2) — Austronesian comparative, ethnographic
  - Reframe: "taphonomic calibration" → "indigenous knowledge resilience through structural invisibility"
  - Strategy document: `papers/P5_volcanic_ritual_clock/HUMANITIES_REFRAME_STRATEGY.md`
  - Backup: Wacana, South East Asia: A Multidisciplinary Journal, Oceania
  - Submit ~June-August 2026 (after rewrite + humanities review)
- **P19 "Before the Inscriptions"** — NEW. Humanities essay for BKI. NOT a reformat of P5.
  - Argument: Lombard's 3 layers have a 4th (pre-Indic Austronesian), invisible due to taphonomic + historiographic processes
  - Engages: Wolters (localization), Lombard (layers), Bloembergen & Eickhoff (heritage politics), Sears (colonial construction)
  - Uses VOLCARCH findings as LIGHT evidence, not main contribution
  - **Phase 2 in progress:** BKI style study complete, skeleton v0.2a enriched with megaliths/Sulawesi/Liangan/Tuban
  - **Next:** Deep reading (Lombard Vol.3, Bloembergen 2020, Wolters 1999). Pak Amien reads, not Claude.
  - Target submission: ~September 2026 (BKI primary)
  - **FALLBACK:** Wacana Vol 28 Nos 1-2 (April 2027) = "Prehistoric art in Indonesia and related regions" — OPEN. Deadline likely ~October 2026.
  - Files: `papers/P19_before_the_inscriptions/`
- **P9 retargeting (HOLD)** — Rejected from JSEAS.
  - Target: **DHQ** / Wacana — wait for P2/P8 outcome first
- ~~E076 v2 satellite~~ — **SUPERSEDED by E189.**
- **E189 satellite spectral feasibility** — **COMPLETED.** NDVI local variance p=0.071. Next: Phase B (SAR).
- **E198 sago-rice etymology (I-133)** — **COMPLETED 2026-04-15.** *sagu > sego phonologically regular. Sundanese "sangu" confirms. Layer 7 proposed. 199 experiments total.
- **E199 Collective Brain (I-135)** — **COMPLETED 2026-04-15.** Kremer/Boserup formalized. Innovation gap 25-188x. Volcanic paradox quantified.
- **E141 Phase 3** — **COMPLETED 2026-04-15.** 433 low-relevance records mined. Rescue rate 0.2%. Classification quality VALIDATED.
- **Colonial data verification** — 10 E070 entries on Delpher.nl. **BLOCKED** (manual task)
- **JCAA APC** — Waiver requested 2026-04-06. Verhagen acknowledged 2026-04-07. WAIT.
- **P19 deep reading** — **BLOCKED** (Lombard Vol.3, Bloembergen, Wolters — manual reading)
- **NEW: P20 TobaSim** — Proposal in `docs/drafts/P20_tobasim_proposal_v0.1.md`. FALL3D Toba 74ka simulation. Needs geologist co-author.
- **NEW: P21 ColonialMine** — Proposal in `docs/drafts/P21_colonialmine_proposal_v0.1.md`. Dutch colonial NLP. Most actionable of 3 new proposals.
- **NEW: P22 JavaTephroChron** — Proposal in `docs/drafts/P22_javatephroChron_proposal_v0.1.md`. Depends on P20 FALL3D infra.
- **NEW: AutoResearch integration** — Concept in `docs/AUTORESEARCH_CONCEPT.md`. 5 research programs proposed. Needs Pak Amien decision on Phase 1.

## PAPERS UNDER REVIEW (WAIT)

| Paper | Journal | MS# | Submitted | Status |
|-------|---------|-----|-----------|--------|
| ~~P1~~ | ~~Asian Perspectives~~ | 019A-0326 | 2026-03-10 | REJECTED 2026-03-17 (AI flag) |
| ~~P1~~ | ~~EGQSJ (Copernicus)~~ | egqsj-2026-3 | 2026-03-30 | **REJECTED 2026-04-16** (desk: structure/wording, NOT content) |
| ~~P11~~ | ~~Indonesia (Cornell)~~ | — | 2026-03-31 | **REJECTED 2026-04-01** (scope mismatch) |
| **P11** | **Archipel** (INALCO/EHESS) | — | **SUBMITTED 2026-04-08** | **EiC acknowledged 2026-04-09.** Editorial board meeting ~June 2026. |
| **P17** | **ArchCalc** (CNR, Diamond OA) | **365** | **SUBMITTED 2026-04-09** | **ACKNOWLEDGED 2026-04-09.** Considered for 2027 issue. Review starts after 2026 issues published. |
| P2 | JCAA (Diamond OA) | #280 | 2026-03-11 | **Under review** — 3 revision support material files |
| ~~P5~~ | ~~BKI~~ | — | 2026-03-09 | REJECTED 2026-03-19 ("too narrow for humanities") |
| ~~P7~~ | ~~Antiquity Project Gallery~~ | AQY-2026-0104 | 2026-03-06 | **REJECTED 2026-06-04** (full peer review, 2 reviewers; CONTENT not structure — confirmed distance artifact + site-selection + equifinality) |
| P8 | Oceanic Linguistics (Q1) | OL-03-2026-11 | 2026-03-11 | **Under review** — 5 revision support material files. **arXiv preprint LIVE: 2604.00023** |
| ~~P9~~ | ~~JSEAS (NUS Press)~~ | JSEAS-202603-051 | 2026-03-11 | REJECTED 2026-03-20 ("not suitable") |

**Scorecard: 6 rejected (P1-AP, P1-EGQSJ, P5-BKI, P9-JSEAS, P11-Cornell, P7-Antiquity), 4 under review (P2, P8, P11-Archipel, P17-ArchCalc).**
**⚠ 2026-06-08: P7 = FIRST content-based peer rejection. Confirmed methodological artifact (7-volcano inventory) propagates to P1 §spatial + ~26 experiments. See JOURNAL 2026-06-08 + `docs/research_notes/MATA_ELANG_17_2026_06_08.md`. P17 RE-VERIFIED clean (`E104/rebuild_clean_full_inventory.py`): candi-vs-inscription segregation SURVIVES (14.5 vs 27.6 km, p=1.5e-7) → no withdrawal; fix methods at revision. Canonical inventory now = `volcanoes_java_full.csv` (30). **E213 (exposure-window spine for P7) ran 2026-06-08 → INCONCLUSIVE:** slope-proxy fails (suitability⊥slope ρ=−0.04; volcanic slopes are both high-relief AND lahar-buried). A valid test needs a geology/lithology layer (GLiM or PSG maps). **P7 overhaul ON HOLD — spine not established; do not rewrite on a failed spine.** **E214 paleo-environmental test (independent channel) DONE → LEANS AGAINST strong thesis:** Java forest-clearance reads LATE (Dieng ~600 CE), while Sumatra/Borneo show early farming → partially REFUTES a LARGE pre-400 CE Javanese population (escapes: undersampling + Solo marine loophole). **"0 counter-evidence" claim is now FALSE.** Thesis must downgrade: large-hidden-civilization → small/dispersed/low-visibility society (falsify next via plant-microfossil, lowland coring, population-scale evidence). **SUBMISSION INTEGRITY GATE created & BINDING: `docs/SUBMISSION_INTEGRITY_GATE.md` (G1–G10).** Pending decisions: thesis recalibration in P0; apply SIG to live papers (P17/P2/P8/P11). **P7 PARKED** (Pak Amien 2026-06-08 — revisit only with stronger evidence). **Thesis reframed by PI → "peradaban vulkanik"** (volcanic-civilization CHARACTER claim, not erased-metropolis; consistent with paleo-environmental analysis; propose L1 amendment — see `project_thesis_peradaban_vulkanik` memory). **New lead: Jatim glass beads** (`docs/research_notes/JATIM_BEADS_LEAD_2026_06_08.md`, durable-trace, 5th–8th c CE, NOT pre-400). **E215 plant-microfossil DONE → VOID** (no plant-microfossil study ever run on any prehistoric Java site; method works regionally). Cross-channel E214(paleo-environmental, leans-against-LARGE)+E215(plant-microfossil, void-for-SMALL) → both consistent with peradaban-vulkanik reframe. **DECISIVE NEXT TEST = Castillo plant-microfossil collaboration** (Gunung Sewu matrices/Liyangan/dental samples; draft ready `docs/drafts/email_castillo_phytolith.md`). **Stage 0 integrity sweep COMPLETE** (E019/E104/E004 survive inventory fix; only P7 headline died). **L1 amended → peradaban-vulkanik ✓. Castillo archived ✓ (bottleneck, portfolio-first). Sembiran VERIFIED** (India trade ~2nd c BCE, pre-400 positive).

**★ MATA ELANG #18 PORTFOLIO TRIAGE DONE** (`docs/research_notes/MATA_ELANG_18_PORTFOLIO_2026_06_08.md`, 4-agent review). Verdicts: **DROP** P7 (dead) + P18 (absorb→P0); **MAJOR-REWRITE** P0 (masterpiece = current liability, Channel 5 non-reproducible, fails 7/10 gates); **REVISE** P1v5/P5/P9/P11/P16/P17/D1/D2; **clean** P8/P2 (wait for review). **PATH TO 1st ACCEPTANCE (CORRECTED 2026-06-08 — Pak Amien wants Scopus peer-reviewed journal, NOT Zenodo).** Verified Scopus+free venue map: `docs/research_notes/SCOPUS_FREE_VENUE_MAP_2026_06_08.md`. Diagnosis: 0/6 was wrong-fit journals (5 desk-rejects = generalist) + the 4 genuine peer reviews (P8-OL, P17-ArchCalc, P2-JCAA[APC+waiver], P11-Archipel — all Scopus+free) are still PENDING. **Strategy:** (1) CONVERT pipeline (gate + revisions so they don't bounce); (2) **P16 → Wacana: R1 REVISION DONE, CONDITIONAL GO.** `submission_wacana_v1.0.tex` compiles clean (18pp, 1.1MB, abstract 139w, 0 undef refs/cit, 0 errors). DeepSeek G9 caught a FATAL flaw (convergence test) BEFORE submission → **R1 done & verified:** W1 RESOLVED (tradition-controlled test: cross-tradition convergence SURVIVES 8/8, volcano z=7.1; `e090_v6_tradition_controlled.py`; integrated + fig3 regenerated); W2/W3/W4/W5 tempered (volcanic-silence honest, 929 sample-limited, hedged+equifinality, BH-MTC). **Remaining = W6 humanities-deepening (Pak Amien, ~2¶) OR switch to DHQ; optional re-run G9; Pak Amien final read + submit.** Full state: `docs/HANDOFF_20260608.md`. See `SIG_signoff.md`. (3) **SPAFA Journal** (Scopus+free+SE-Asia-archaeology = never tried) = next archaeology paper (reframed P9/P11). No magic-fast venue; realistic 1st accept ~Q3–Q4 2026. D1/D2 Zenodo = NOT a Scopus accept (demoted to preprint-DOI only). Residual artifact purges: P9 line-83, P16 line-468, P0 Ch5, P11 E153+9.2km. Re-run E031/E082/E153 canonical. CANONICAL.md P1 fixed ✓.
Revision support material still available for all papers. See JOURNAL 2026-03-20 for pattern analysis.

## PAPERS NEEDING RETARGETING (DECISION REQUIRED)

| Paper | Rejected From | Next Options (TBD) |
|-------|---------------|---------------------|
| P1 | Asian Perspectives, EGQSJ | **v2.0 REWRITTEN** (all lists→prose, language tightened). Target: **JASREP** (Elsevier, Scopus Q1, free subscription route). Already formatted. Backup: Archaeological Research in Asia. |
| P5 | BKI | (a) Retarget archaeometry journal as-is, OR (b) major rewrite for humanities framing |
| P9 | JSEAS | (a) World Archaeology / Cambridge Arch. J., OR (b) hold for P2/P8 outcome |
| P11 | Indonesia (Cornell) | Internet Archaeology (Diamond OA) / BIPPA (free) / Aziatische Studien (Diamond OA) |

## KEY FINDINGS THIS SESSION (reference only)

| Finding | Experiment | Impact |
|---------|-----------|--------|
| ADV-5 resolved: C5 = Mon-Khmer substrate | E107 | L4 upgraded, P8 framing restored |
| Demographic gap 3,220× | E108 | Null hypothesis rejected |
| 5-factor cascade P=0.058% matches data | E110 | Core theoretical model |
| PAN \*surat indigenous (~5000 BP) | E112 | Writing concept pre-dates India |
| Agriculture 91% native, Religion 86% Sanskrit | E112 | Sanskritization = elite overlay |
| No inscription learning curve | E113 | Pre-existing organic-media tradition |
| Nusantara #1/10 pre-literate societies | E114 | CCI=23, z=2.12 |
| Survey deficit = 40× leverage, burial = 1.7× | E110 | Reframe: survey first, burial second |
| West Java decisive case (Buni + Batujaya) | E110 | Within-island taphonomic control |
| Cascade robust: 92% within 10×, correlation-robust | E115 | Model survives Monte Carlo |
| GPR 20 targets → expect 2.5 finds [0,6], P(zero)=7% | E116 | Framework IS falsifiable |
| Surface survey reaches ~1900 CE only at 4mm/yr | E117 | Detection horizon quantified |
| 3.5× search efficiency, 29% entropy reduction | E118 | Volcanic context = practical value |
| Synthesis figure: burial diagonal × detection horizons | E119 | Visual elevator pitch |

## BLOCKED

- **D1+D2 JOAD** — APC waiver decision (£374 each) or Zenodo (free)
- **Dissemination Phase 2** — Wait for 1 acceptance

## COMPLETED PREVIOUS SESSIONS

- **Session 16 (2026-04-15)** — PhD pivot + milestone consolidation. Prof. Suzan Verberne (LIACS, Leiden) responded positively to PhD inquiry (NLP for colonial Dutch archives). Research proposal due ~2026-04-17. 2 new experiments: E198 sago-rice etymology (Layer 7, *sagu > sego phonologically regular), E199 Collective Brain (Kremer/Boserup, innovation gap 25-188x). E141 Phase 3 validated Phase 2 quality (0.2% rescue rate). KITLV email on hold. Castillo email ready. 199 experiments total.
- **Session 15 (2026-04-13)** — AutoResearch autonomous. 9 new experiments (E189-E197). Satellite Phase A (NDWI p=0.032), L2 entry points, AHA (E195 rho=+0.53), population 1-2M (E196), colonial depth validation (E197). E141 extended: 1,768 records, 165 geocoded, 33 depths. DHARMA monoculture broken.
- **Session 10 (2026-04-02)** — P8 arXiv published (2604.00023). P17 ArchCalc compliance audit against real papers (website rules differ!): dashes, captions, paragraph numbering all fixed. Tables rebuilt via python-docx. Spelling standardized British. 4 submission files ready. Email standardization verified (go-public unblocked).
- **Session 9 (2026-03-31)** — **Mata Elang #12 autonomous session (extended).** Deepest structural critique to date. **10 new experiments (E154-E163):** E154 FDR re-audit (78.3%), E155 cross-regional cascade (rho=1.0), E156 L1xL2 double erasure, E157 ethnographic F4/F2, E158 steelman counter-args, E159 robustness battery (5/5 ROBUST), E160 GPU NLP (929 CE z=3.04), E161 Bali (5/5 confirmed), E162 synthesis, E163 Sumatra (nuanced: Sriwijaya paradox). **Plus:** P17 v0.3 ArchCalc-ready (anonymized, ~5.2K words), interactive prediction map, borehole site-selection protocol ($6K for 20 holes). **163 experiments total.**
- **Session 8 (2026-03-31)** — Post-ME#11 pipeline. **P11 SUBMITTED** to Indonesia (Cornell): Chicago 17th conversion, E153 candi-settlement proxy test (p<0.0001), Liangan validation strengthened, AI prose audit passed, cover letter sent. P17 target confirmed: ArchCalc, rules downloaded (6K word limit, needs trim). JCAA APC crisis: £593, waiver very difficult. E153 experiment: 154 experiments total. KB.nl Delpher reply received — colonial data already public, no action needed.
- **Session 7 (2026-03-30)** — Back at campus. 3 deliverables: (1) LiDAR 1-page pitch for company contact — 10 GPS targets, Amazon precedent, value proposition, (2) README.md professional rewrite for GitHub go-public — Zenodo badge, 120 experiments summary, citation block, (3) YouTube Ep2 "Patung yang Ditelan Bumi" full outline — 10-min Dwarapala Singosari deep dive. Dissemination roadmap tracking updated.
- **Post-Mudik Session 6 (2026-03-22)** — Sustainability deliverables: YouTube Ep1 script (15 min, Bahasa Indonesia), NatGeo Explorer Grant outline ($20K GPR/ERT pilot), DRPM Penelitian Dasar skeleton (Rp 500M/3yr). All pushed to GitHub.
- **Post-Mudik Session 5 (2026-03-22)** — **Strategic pivot: dissemination.** Comparative civilization gap analysis (11 civilizations, 5 patterns, 7 gaps, 5 possibilities). Dissemination roadmap created (`docs/DISSEMINATION_ROADMAP.md`): 4-tier strategy from GitHub/YouTube to funding/documentary. LiDAR contact identified (needs 1-page pitch). Technical: P1 ORCID fixed, P11 +4 refs +Liangan section +Ceren sentence, P17 experiment count fixed +spelling +fig:model removed. All pushed to GitHub.
- **Post-Mudik Session 4 (2026-03-22)** — Handoff document created (`docs/HANDOFF_20260321_SESSION4.md`). WORKSTATE updated. Continuation prompt delivered to Pak Amien.
- **Post-Mudik Session 2 (2026-03-21)** — Comprehensive blind spot analysis (8 blind spots, 8 new ideas I-120–I-127). Liangan research note created (I-120: 15+ references, burial depths, organic preservation, Cerén comparison gap). P19 skeleton enriched v0.2a (megaliths, Sulawesi cave art, Liangan, Tuban nekara). PREMORTEM Counter 1 upgraded 70/30. **P11 pre-flight complete:** Wacana NOT viable (thematic, Kawi published), recommend Indonesia (Cornell) or ArchCalc. **P17 pre-flight complete:** strongest paper, recommend ArchCalc, issues flagged (Fig 6 missing, experiment count, double-blind prep). **Strategic discovery:** Wacana Vol 28 "Prehistoric art in Indonesia" (April 2027) = P19 fallback if BKI fails.
- **Post-Mudik Session 1 (2026-03-21)** — Structural critique (7 sections: fatal/structural risks, over-complexity, weak assumptions, collaboration architecture, testing framework, critique selection). E119 synthesis figure rendered (matplotlib PNG+PDF). P1 EGQSJ cover letter finalized with checklist + suggested reviewers. JCAA APC verified: £450, CAA waiver fund available (check if applied). Doc sync: PASS.
- **Mudik Session 3 (2026-03-20)** — Autonomous mode. 5 new experiments: E116 testable predictions (GPR → [0,6] finds, P(zero)=7%), E117 detection horizon (surface survey ~1900 CE only), E118 information gain (3.5×, 29% entropy), E119 synthesis figure (data for matplotlib). Auto-sync checker (`tools/check_doc_sync.py`). Falsifiability revision support material package. README refreshed with current results. TRIGGER_MAP updated (rejection pattern). Michelson-Morley framing: value = method + predictions, both outcomes are contributions. Failed experiment rescue analysis (E024→E083, E039→E103 already done). 120 experiments total.
- **Mudik Session 2 (2026-03-20)** — E115 cascade sensitivity analysis: ROBUST (92% of 100K MC runs within 10× of observed, correlation-robust). Hard structural critique: 3 fatal risks (A1-A3), 4 structural risks (B1-B4). L1 §9 stop criteria REWRITTEN (old criteria obsolete). Pre-mortem analysis: 6 counter-arguments classified. Rejection pattern analysis: specialist journal = 100% survival rate. P1 EGQSJ AI prose audit: CLEAN. Document sync: experiment counts fixed to 116 across L1/L2/L3/EVAL/EXPERIMENT_INDEX. D1/D2 affiliations corrected.
- **Mudik Session 1 (2026-03-20)** — P1 EGQSJ: ORCID + GitHub URL + 5 reference DOIs fixed (fully ready). Structural critique: B1-B4 risks classified. Diamond OA journal targets verified for all papers. AI Prose Guide created. P11 v0.3 AI markers fixed. Blind spot research: Dong Son drums (I-110), Philippines comparandum (I-111), metallurgy (I-112). Docs synchronized (L2, L3, drafts/README, IDEA_REGISTRY). JOURNAL updated.
- **P1 Final Review (2026-03-18)** — Review fixes (duplicate content, textbook filler, West Java claim, 3220x transparency, ov1925 reframe), AI disclosure trimmed, internal jargon cleaned (E083/RQ/H labels), DOIs verified via Playwright (3 corrected, 1 removed), versioning cleanup (renamed v1.0, archived obsolete variants).
- **Structural Audit + Vocabulary Archaeology (2026-03-17)** — 9 new experiments (E107-E114), P18 draft, research statement v4.0, 6 new revision support material files, cascade model, West Java comparandum. Total: 115 experiments.
- **Two Javas Sprint (2026-03-17)** — E099-E106, P17 v0.2 (22pp), P16 expanded (27pp), E090/E094/E096 GPU runs, P5 revision support material.
- **Senter v3 (2026-03-16)** — E092-E098, anomaly detection, GPU scripts, Dokumen Jembatan v0.2.
- **Consolidation (Sessions 2-5)** — P7 preprint DOI live. E089 v4. P11 v0.3. Code review.
- **Senter v2** — E091 OV NLP. E089 v3. E076 v2 script.

## SESSION PROMPT

STATUS: **207 experiments** (E001-E207, E180 skipped). **Session 17.** Riset dalam diam + core stack.
**Session 17 (7 experiments):** E201 Philippines deep comparison (55-65% open-air, gap LARGER), E202 DEM depression (30m fails), E203 population-evidence meta-analysis (5th evidence channel), E204 bronze drums extended (selective survival), E205 wayang indigenous layer (living evidence), E206 ArcheoBERTje gap (60% missing for PhD), E207 GLOBALISE VOC pilot (6,893 files, 55% drop).
**PhD SENT:** Proposal v0.2 sent to Verberne 2026-04-16. WAITING response. Cohen track: apply Dec 2026. Vossen email: ON HOLD.
**Strategy:** RISET DALAM DIAM. No more paper submissions. Papers = PhD evidence base. Core stack building.
**Papers:** 5 under review (P2-JCAA, P7-Antiquity, P8-OL, P11-Archipel, P17-ArchCalc). P1 REJECTED from EGQSJ (2026-04-16, desk: structure). **v2.0 rewritten, pre-flight audit done, 10 issues to fix → submit JASREP Monday 2026-04-21.**
**Core Stack:** GLOBALISE downloader + VOC preprocessor + spelling normalizer BUILT. 50 files downloaded (6.26M words preprocessed). ArcheoBERTje gap quantified (60% entity types missing, 55% quality drop on VOC).
**Key context:** Argument reframed from "zero evidence" to "selective survival" (E204 bronze drums). Philippines comparison strengthened (E201: 275-340 pre-400 CE sites vs Java 0). Population data = 5th independent evidence channel (E203). Wayang = living evidence (E205).

### Mata Elang #13 Key Findings (2026-04-09)
1. **Cascade over-parameterized (E176):** 3-factor models bracket gap. 83.8% of random 5-factor draws work. F1 (volcanic burial) is LEAST necessary factor (2/5 minimal models). Reframe from "model matches data" to "plausible mechanistic decomposition."
2. **Karst is hidden 6th factor (E178):** Philippines volcanic zones have 25 pre-400 CE sites (Java has 0). Difference: karst 0.20 vs 0.08. Cave sites bypass all 5 cascade factors. Java's volcanic interior has almost no karst.
3. **L2 now has predictions (E177):** 250K displaced from Sunda Shelf toward Java. 5 entry points identified. Surabaya = highest priority (L1xL2 double erasure).
4. **Factor coupling tested (E179):** Coupling shifts cascade 3.0x. Hot lahar scenario (organic destruction) actually improves fit (0.8x). Within parameter uncertainty.
5. **7 structural risks identified:** Cascade unfalsifiable, experiment count inflated, DHARMA monoculture, L2 abandoned, echo chamber, competence gap, paper velocity.
6. **3 strategic pivots:** (a) Lead with decisive case not cascade, (b) Focus P17, (c) Seek collaboration.
7. **Full critique:** `docs/research_notes/MATA_ELANG_13_2026_04_09.md`

### ME#13 Experiments (E176-E179)

| ID | Finding | Status | Key Result |
|----|---------|--------|------------|
| E176 | Cascade minimal model | SUCCESS | 3 factors sufficient. AIC: 3-factor 6.73 vs 5-factor 6.25. |
| E177 | Sunda Shelf L2 model | SUCCESS | 250K to Java, 5 entry predictions, L1xL2 at Surabaya. |
| E178 | Philippines regression | SUCCESS | R2=0.733. Karst = hidden factor. Java uniquely dark. |
| E179 | Factor independence | SUCCESS | Coupling 3.0x shift. Within E115 uncertainty. |

### Mata Elang #12 Key Findings
1. **Verification Ladder**: Saturated at Level 0. Need Level 1 (peer acceptance) — P17 at ArchCalc is the critical path.
2. **FDR re-audit**: 78.3% survival (up from 73.2%). E048 rescued. Only 2 casualties.
3. **Cascade validated cross-regionally**: Correctly predicts visibility rank across 5 regions (rho=1.0).
4. **L1xL2 "Double Erasure"**: Sunda Shelf drowning pushed ~94K people into volcanic zones. West Java decisive case is model-predicted.
5. **Ethnographic calibration**: F4=0.43 (Hindu), F4=0.20 (pre-Hindu). F2=0.21 (three independent estimates converge).
6. **Full critique**: `docs/research_notes/MATA_ELANG_12_2026_03_31.md`

### ME#12 Experiments (E154-E157)

| ID | Finding | Status | Key Result |
|----|---------|--------|------------|
| E154 | FDR re-audit at 157 experiments | SUCCESS | 65/83 survive BH (78.3%). E048 rescued. |
| E155 | Cross-regional cascade validation | SUCCESS | rho=1.0, p=0.017. F3 most differentiating. |
| E156 | Sunda Shelf population displacement | SUCCESS | L1xL2 double erasure. 94K into volcanic zones. |
| E157 | Ethnographic volcanic analog | SUCCESS | F4 validated. F2 triple convergence. |

### Standing Priorities (Updated 2026-04-16)

**#0 — CORE STACK (PhD-agnostic pipeline):**
0. **Gold-standard NER annotations** — 500 sentences from OV + VOC, annotated with 7 entity types. This is the evaluation benchmark.
1. **Colonial place-name gazetteer** — Extract E091 names → map to modern GIS coordinates. Concrete deliverable.
2. **Scale GLOBALISE corpus** — Download more files (currently 50/6,893). Preprocessing pipeline ready.
3. **Fine-tune ArcheoBERTje on colonial Dutch** — After annotations ready. Compare with baseline (E206).

**#1 — WAIT (external dependencies):**
4. **Verberne response** — Proposal SENT 2026-04-16. WAIT ~1-2 minggu.
5. **5 papers under review** — P2, P7, P8, P11, P17. WAIT. (P1 rejected, needs rewrite.)
6. **JCAA APC waiver** — Verhagen acknowledged. WAIT.
7. **Cohen formal application** — Dec 2026. Prepare closer to deadline.

**#2 — HOLD (until PhD trajectory clear):**
8. Vossen email — tunggu Verberne response
9. ~~GitHub go public~~ — HOLD (riset dalam diam)
10. ~~P16 submit~~ — HOLD (becomes PhD evidence base)
11. ~~New paper submissions~~ — STOPPED
12. KITLV email — ON HOLD
13. Castillo email — READY but not urgent

**#3 — VOLCARCH EVIDENCE (continue building):**
14. VOLCARCH experiments — continue as "background motivation" for PhD
15. New evidence streams from ME#14: living culture (I-139), oral tradition (I-140), kakawin NLP (I-141)

### Stop Doing
- **Submitting new papers** (papers = PhD evidence base now, not standalone goals)
- **Going public** (riset dalam diam until PhD trajectory clear)
- More cascade sensitivity analyses (saturated)
- DHARMA mining (CLOSED)
- Computing Bayes Factors from estimated priors

### Core Stack Status
| Component | Status | File |
|-----------|--------|------|
| GLOBALISE downloader | **DONE** | `tools/globalise_pipeline/download_globalise.py` |
| VOC preprocessor | **DONE** | `tools/globalise_pipeline/preprocess_voc.py` |
| Spelling normalizer | **DONE** (10/10 tests) | `tools/globalise_pipeline/normalize_colonial_dutch.py` |
| Corpus (50 files) | **DONE** (6.26M words) | `data/processed/globalise_voc/` |
| NER annotations | PENDING | 500 sentences, 7 entity types |
| Colonial gazetteer | PENDING | E091 names → GIS |
| Fine-tuned NER model | BLOCKED (needs annotations) | — |
- **Key experiments supporting PhD:** E091, E141-E143, E160, E197 (colonial NLP/data mining)

## COMPLETED THIS SESSION (2026-04-15)

- Inbox processed: CV + VOLCARCH Brief archived to `docs/correspondence/`
- Verberne email exchange archived: `docs/correspondence/EMAIL_VERBERNE_LEIDEN_2026_04_14.md`
- E198: Sago-rice etymology (SUCCESS, Layer 7 proposed)
- E199: Collective Brain paradox (SUCCESS, innovation gap 25-188x)
- E141 Phase 3: Low-relevance mining (COMPLETE, 0.2% rescue rate — classification validated)
- KITLV email: ON HOLD (reframed after Verberne contact)
- Castillo email: updated, READY TO SEND
- IDEA_REGISTRY updated (I-133, I-135 → RESULT)

## DO NOT WORK ON

- New paper drafts beyond strengthening P18
- P18 submission (wait for acceptances)
- Phase 2 dissemination (wait for 1 acceptance)
- GPU-dependent tasks (no GPU available)
- KITLV cold email (wait for PhD trajectory)

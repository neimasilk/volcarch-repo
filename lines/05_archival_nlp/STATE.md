# STATE — Line 05 ARCHIVAL NLP

**Updated:** 2026-10-02 · **Temperature:** 🅿 E226 (T7) parked as REVISIT ("digodok dulu") — identification is the bottleneck

> **2026-10-02 (late morning) — E226 PARKED (REVISIT).** Frame done (266 villages, 170 Kedu/Prambanan, κ 0.90).
> Register frozen: 1 qualifying settlement (Liyangan) + 9 unclear within web reach. **I2 = 0, I1 = 11/170** → |G| far
> below the pre-registered 30; step 4 not run (would manufacture an "uninformative" verdict from small n). Unpark:
> Kusen 1990/91 or Resiyani 2010 toponym lists (Nastiti / Griffiths / UGM), or a domain expert coding I1 for ≥30
> villages. Reusable now: `results/t7_village_frame.csv`.

> **2026-10-02 — E226 (T7) opened** (`experiments/E226_t7_detection_calibration/`). Pre-registration
> `DESIGN.md` frozen and pushed before any register or matching. Step 1a done: 595 village-noun occurrences in
> 73 of 111 inscriptions (the desk scan's 440 missed the *banuA/banva* spellings). Two blind coders + a blind
> settlement-register agent running. Desa gazetteer (Kepmendagri 2025, 3,802 desa) in
> `data/processed/gazetteer/`. **PI: fill N_ref and m in DESIGN §5 before step 4.**

> **2026-10-01 (16:57, not evening) — downstream audit (`docs/research_notes/DOWNSTREAM_AUDIT_C023_C032_20261001.md`; ledger C039–C043):** this line's E098 carry correction notes; headline statuses changed in the index. Do not cite their old headline numbers.

> **2026-10-01 afternoon:** **T7** (detection calibration from DHARMA village names) is now proposed as this
> line's next experiment, ahead of T2.
> - Design draft: `docs/drafts/T7_DETECTION_CALIBRATION_DESIGN_DRAFT_20261001.md`.
> - Denominator in hand: 111 dated 8th–10th c. inscriptions, about 150–200 village names, mostly Kedu and
>   Prambanan.
> - Numerator: no dated settlement register exists. E129 is not one (C036).
> - It gets an E-number (E226) after PI approval. Desk scan: `docs/research_notes/DESK_TESTS_T0_T4_T7_20261001.md`.

> **2026-10-01 inbox (orbit re-entry):** (a) E225 added to `LINE_MAP` (it was unmapped). (b) The re-entry
> synthesis recommends **T2 — a register of chance archaeological finds at depth** (sand mining, wells,
> foundations; Liangan, Kedulan, Kimpulan and Wonoboyo were all found this way) as the next experiment:
> a natural extension of E225's gray-literature corpus plus Indonesian news archives, and the PI's NLP
> strength. It needs a pre-registered design with a written kill criterion before any mining (`docs/research_notes/OBJECTIVE_ANSWER_20261001.md` §5).
> (c) `.env` now holds a `DEEPSEEK_API` key, so E225's ≤$10 extraction budget is unblocked. (d) E211 smoke
> test still pending (not run this session). (e) ✅ Reconciled 2026-10-01: the Phase-1 keyword outputs exist
> (`results/E211_voc_mentions/`, 33,930 mentions); "Current position" below now says so.

---

## Current position

The pipeline exists and is registered as an HKI product. **Corrected 2026-10-01:** it *has* been pointed
at the corpus once — the E211 **Phase 1 keyword run finished on 2026-04-23** (500 GLOBALISE files →
33,930 candidate mentions → 14,626 Java-filtered → 871 high-precision; 0 hits for oudheden/prasasti/stupa;
outputs in repo-root `results/E211_voc_mentions/`, findings in
`experiments/E211_voc_dagregister_nlp/FINDINGS_v1_20260423.md`). What is pending is the **pre-registered
evaluated run** (`EVAL_PROTOCOL_20260813.md`), **authorised by the PI on 2026-08-13 (D2)**: 10-file smoke
test first, then annotation of the 300+200 held-out sentences.

This is the cheapest large result available anywhere in the project: the instrument is built, the data
is on disk, and no external human is required.

---

## Blocked on PI

| # | Item | Since |
|---|---|---|
| ~~1~~ | ~~**Approve the E211 run** on the 500 downloaded files.~~ ✅ **AUTHORISED 2026-08-13** (decision hour D2, after 112 days). Sequence: pre-write eval protocol → 10-file smoke test → full run. | 2026-04-23 |
| 2 | **DJKI HKI submission** — 4 registration documents are ready in `docs/HKI/`. Filing is a PI action. | 2026-04-23 |
| ~~3~~ | ~~**D1 → JOAD** submission~~ — **MOOT 2026-08-11**: D1 published directly on Zenodo (`10.5281/zenodo.21882007`). JOAD waiver question = career-line decision, not a blocker here. | — |

---

## Next actions for Claude

- [x] **E211 evaluation protocol pre-written** ✅ 2026-08-13 → `experiments/E211_voc_dagregister_nlp/EVAL_PROTOCOL_20260813.md`
      (7 tipe entitas, 300+200 kalimat held-out berstrata, κ≥0.6, F1≥0.70 = publikasi, kill <0.40,
      seleksi publikasi dibekukan). Run diotorisasi PI 13 Aug (D2).
- [ ] Dry-run VOC-ArchNLP end-to-end on a **10-file sample** (protokol §6: file pertama urutan nama,
      cek 4 modul + skema + waktu per file) — ini gerbang sebelum full run 500 file.
- [x] ~~Prepare D1's Zenodo fallback package~~ — **MOOT 2026-08-11**: D1 published directly on Zenodo
      (`10.5281/zenodo.21882007`), DOI already live, no JOAD dependency. The separate JOAD
      submission question (waiver fund) is a career-line decision, not a blocking prerequisite.
- [ ] Check whether the SCC/GDPR question actually blocks anything in the planned E211 output. If the
      output is entities + counts only, it does not — write that down so it stops being a vague
      worry.

## Do NOT do

- ❌ Run E211 on all 500 files without approval.
- ❌ Commit or publish extracted Delpher/KB full text.
- ❌ Add a fifth module to VOC-ArchNLP. It is registered at v1.0.0 with four; scope growth here is how
  the corpus run keeps getting postponed.

## Inbox

- `E128` (OV depth) is cited as independent depth evidence by
  [02_taphonomy](../02_taphonomy/) — if WS-E touches it, that line owns the re-derivation.
- This line is the technical basis of all four PhD approaches (Verberne, Cohen, Vossen, UvA). A
  completed E211 run would materially strengthen every one of them — which is an argument for
  unblocking it, tracked in [07_career](../07_career/).

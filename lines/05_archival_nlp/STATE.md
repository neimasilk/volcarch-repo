# STATE — Line 05 ARCHIVAL NLP

**Updated:** 2026-08-13 · **Temperature:** 🔧 Tooling complete, corpus run not authorised

---

## Current position

The pipeline exists, is registered as an HKI product, and has **not been pointed at the corpus.**
500 dagregister files are downloaded. E211 Phase 1 has been ready since **2026-04-23** and is waiting
on one PI decision.

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

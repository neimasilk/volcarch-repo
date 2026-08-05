# STATE — Line 01 SPATIAL

**Updated:** 2026-08-05 · **Temperature:** 🔥 HOT — hard deadline in **15 days**

> 📄 **Naskah v0.2 selesai diimplementasi (2026-08-05).** Semua blocker/temuan handoff 5 Agt ditutup:
> gambar (6 figur dari data mentah), literatur ENM (5 sitasi terverifikasi), S1–S4 (Tabel 4, tautology,
> latar arkeologi, abstrak 216 kata), surat balasan selaras. **SIG re-sign-off 5 Agt = 🟢 GO pada
> integritas naskah (9/9 gerbang hijau)**: G1 (64 check/60 OK), G2, G8, dan G9 (review adversarial —
> tak ada klaim tertolak, 3 frasa dikencangkan). **Item A (sign-off Go Frendi) dan item E (judul,
> kandidat 3) RESOLVED 5 Agt.** Sisa sebelum upload: G1 final, push, upload (PI).

---

## Hard deadline

**P2 resubmission to JCAA: 2026-08-20 — no extension will be requested.** PI decision 2026-08-03: every
`[RUN]` item is done, dissolved, or out of scope, so the remaining revision is **writing**, and asking
for more time on run-related grounds would have given the editor a reason that is not true. Asking the
scope question ("revision or new submission?") was also dropped — it risks inviting the answer "new
submission", which would discard the only non-reject this project has had in 14 months. Withdrawn draft
and full reasoning: `docs/correspondence/EMAIL_VERHAGEN_EXTENSION_REQUEST_20260803.md`.

**Consequence: v0.2 is a revision of #280, and the corrections go in the Response to Reviewers.**

---

## Blocked on PI (nothing downstream can move)

| # | Item | Since | Status |
|---|---|---|---|
| A | **Confirm authorship with the human Go Frendi.** The new manuscript reaches the *opposite* conclusion from the one he signed in March. `review_package_20260727/05_*` is Claude's analysis of his likely position, **not a signature.** | 2026-07-27 | ✅ **RESOLVED 2026-08-05.** PI confirmed Go Frendi is OK with the reversed claim set. The Authors' Contributions sentence (`submission_jcaa_v0.2.tex:573`) is now factually true. No further action. |
| B | ~~Send the Verhagen email~~ | 2026-07-27 | **CLOSED 2026-08-03 — withdrawn, no email will be sent.** |
| C | ~~Decide scope: revision vs new submission~~ | 2026-07-27 | **CLOSED — revision of #280.** |
| D | ~~Permission to commit~~ | 2026-07-27 | **CLOSED — committed.** |
| E | **v0.2 title** — candidates in `revision_ammo/JCAA_R1_RESPONSE_PLAN_20260727.md` §3. Candidates resting on "selection picks the worst design" are dead (K5); those resting on **evaluation incomparability** survive. | 2026-07-27 | ✅ **RESOLVED 2026-08-05 — kandidat 3 terkonfirmasi** ("An Evaluation Artefact in Presence–Background Archaeological Modelling…"). Naskah tidak berubah. |

---

## Next actions for Claude (in order)

- [x] **B′ — apply K1–K3 to the claim set.** ✅ 2026-08-03 → doc 10 (K5/K6/K7/G1c added same day).
- [x] Block H — **SIG G1 blind re-derivation.** ✅ 61 checks 2026-08-03; **re-run 2026-08-05: 62
      checks, 58 OK, 4 mismatch** (persis klaim lama yang ditarik K5/K6/K7/G1c) + **A7** baru
      (0.706 common-bg, lolos).
- [x] **E224 — K4 confirmation run.** ✅ 2026-08-03 — FAILED; TGB null dilaporkan **unexplained** (§3.5).
- [x] Block D — **R2 covariate table.** ✅ Table 1 (roles) + Table 2 (inclusion) di v0.2 (§2.1, §2.3).
- [x] Block G — **reviewer response letter.** ✅ diselaraskan 2026-08-05 ke naskah final: semua
      `[NEEDS v0.2]` diselesaikan, R2-H menyatakan 7 figur v0.1 yang dihapus, penomoran E218/E219
      disamakan (E219 part C kini di naskah §3.8).
- [x] Block F — **new figures.** ✅ 2026-08-05: fig14 artefact, fig15 dose-response, fig16 robust/
      contingent map, fig17 stabilisasi — semua dari file hasil mentah via `build_v02_figures.py`.
- [x] Block E — **old figures refresh.** ✅ fig10 di-redraw dengan 13 pusat kanonik (INT-1); fig3
      di-restate sebagai ladder under examination; prefix caption "Figure N."/“Table N.” manual dihapus.
- [x] **Manuscript v0.2 prose + perbaikan.** ✅ **26 pp, kompilasi bersih, nol overfull.** S1 (klaim
      level Tabel 4 ditambal, A7), S2 (Test 1/3 didefinisikan di §2.4), S3 (latar arkeologi East Java),
      S4 (abstrak 216 kata, satu angka headline +0.042), ENM lit (5 sitasi terverifikasi, `[NEEDS
      CITATION]` hilang).
- [x] Block I — **cross-model review (G9).** ✅ 2026-08-05 — subagent adversarial (diminta menolak):
      **tak ada klaim tertolak, tak ada mismatch angka**; 3 frasa dikencangkan (Limitation 3 agregat,
      scope "seed-ensembled", scope AI-disclosure) + inflasi home-court dinyatakan spesifik hybrid
      (A8/A9, check lolos).
- [x] **SIG re-sign-off final.** ✅ 2026-08-05 → `SIG_signoff.md` = **🟢 GO pada integritas naskah**
      (9/9 gerbang hijau). Sesi ter-commit (e38987d + sesi ini). **Tersisa sebelum submit: item A
      (sign-off Go Frendi), judul (item E), G1 final re-run, push.**

## Deliberately NOT doing

Additional synthetic regimes · second-region replication → both declared **future work** in the
manuscript. E219 two-stage "suitable but absent" (R2-C) → dissolved when the taphonomic claim was
withdrawn.

---

## Other papers in this line

- **P17** (ArchCalc 365) — under review. No action. Do not touch the manuscript; it is live and
  double-blind.
- **P11** — retarget to SPAFA is **queued behind the [07_career](../07_career/) exposure actions** by
  PI decision. Do not start it as an alternative to P2 work.
- **D2** — Zenodo upload is a [07_career](../07_career/) item.

## Inbox (found while working, not yet triaged)

- `docs/experiment_index.json` covers only **84 of 214** experiment directories, so the
  experiment→paper mapping is stale. Re-run `tools/scan_experiments.py` and add a `line` field.
  Cheap, and it is what makes this whole layer self-maintaining.

# WORKSTATE — Orbit Dashboard

**Updated:** 2026-10-06 · **This file is short by design. Keep it that way.**
Previous version (P2-submitted era, 11–13 Aug): `docs/archive/WORKSTATE_snapshot_20260813.md`.

> ## 🟢 2026-10-05 — P8 conditionally accepted (*Oceanic Linguistics*, OL-03-2026-11)
> Editor Sander Adelaar, 14:17 WIB: **accepted pending revisions** — not final; **no deadline stated**; editors ask extra care for readability (Section 3).
> R1 (Sulawesi specialist) 14 points, R2 24 highlights; mostly wording, terms and explanations; two items need analysis
> (Makasar vs a published 62 % figure; glottal-stop orthography) plus a data-release request. Journal = Scimago **Q2**.
> **Ledger, gates, order of work: `papers/P8_linguistic_fossils/REVISION_OL_20261005.md`.** Reviewed text verified = `draft_v0.1_anonymous.*`.
>
> **Same day, evening — verification done (E227, E228).** Every reviewer point has a checked factual answer; the stored numbers reproduce
> from raw ABVD. **But the submitted text misdescribes its own pipeline in places that reach the abstract** (two label sets, 356 vs 438;
> in-sample "consensus" κ 0.61 → 0.31 out of fold; "fewer prefixes" is the reverse of the data; the 16-language pattern rested on
> language-level inputs; headline 0.763 is not "phonological only": form-only 0.67). Pre-registered re-tests: the negative result holds
> only as "no large shared layer" (permutation p = 0.0001 for a small shared component). In its favour: the published Makasar 62 % is
> reproduced (60.7 %) and decomposed; 71 % of Tolaki's uncoded forms recur within Bungku–Tolaki (chance 10 %).
> *(Corrected 2026-10-06, see the note of that date below: not "reproduced", not "no large shared layer", p ≤ 0.0001,
> "fewer prefixes" withdrawn without "more".)*
> References checked: 2 of 35 `.bib` entries do not exist (uncited), `ross2005` is cited with a false venue; fee policy of the journal
> is stated nowhere → one line to the editor (G15). Working copy `revision_v0.2/` = reviewer-dictated term changes only.
> **Seven PI decisions (D1–D7) before any writing: `papers/P8_linguistic_fossils/REVISION_WORKPLAN.md` §2.**

> **2026-10-06 — everything Claude can prepare for the P8 revision is prepared; the ball is with the PI.** Tables and Figure 1 for the
> corrected model (E229, independently re-derived), the robustness section re-run (E230: "does not depend on form length" withdrawn; its
> printed numbers belonged to another model), corrected `.bib`, `VENUE.md` (**the journal needs a Word file for the final version; fees
> stated nowhere**), a per-section outline of facts, and a draft note to the editor (**not sent**). The original email and annotated PDF were
> re-read: the ledger is complete. **Start: `papers/P8_linguistic_fossils/REVISION_OUTLINE_20261006.md`; decisions: `REVISION_WORKPLAN.md` §2 + §10.5.**

> **2026-10-06, 11:30 — P8: three adversarial reads, three rounds of correction; start from `REVISION_WORKPLAN.md` §10.7.**
> The block above was written before the first read returned. Read 1 (17 findings): every number sound, several readings
> not ("coded in ABVD" read as "inherited"; Tolaki over-read; the morphological reading stated as a result). Read 2 (14
> findings): three of the *replacement* readings were wrong too — re-derived by the orchestrator (E231, tables P1–P7) before
> anything was changed. **What the PI must know before writing:** (1) 44 Muna forms are "coded" only through ABVD's second
> Muna list, so Table 1's lower end (15.5 %) is an effect of one extra list; (2) Makasar has more *uncoded* meanings
> than Bugis and Sa'dan Toraja, and that is one fact, not three; whether they are uncoded cognates or replaced words is
> not decided (a screen does not favour "merely uncoded") — **nothing on Makasar is written** before the PI has read the
> source of the published 38 % (read 3 found round 2 over-corrected here; a Sulawesi specialist is optional, not a blocker); (3) "fewer prefixes" is withdrawn and "more prefixes" is **not**
> established (not separable from word length); (4) the cross-list test says nothing about a shared layer; **D4 now has
> three options**. Outline (its header now says which sections can be written from), work plan, editor note, READMEs
> E227–E231, ledger C055–C065 corrected; a column description for the release files written (**draft**, C065). The session had been cut off at 08:54 (API error, power cut; no damage) and was resumed.
> ⚠ **PI action, security:** the arXiv *paper password* of the P8 preprint has been in the public git history since April
> (C061; redacted in the working tree) → ask arXiv for a new one. Canary green 223. Committed on main at the PI's word (`450aef6` and a small follow-up); not pushed.

> # 🔄 RE-ENTRY 2026-10-01 — riset dilanjutkan setelah jeda 7 minggu
>
> Riset berhenti pertengahan Agustus karena sebuah model salah-flag repo sebagai "biologi". Hari ini:
> (1) `volcarch-genetics` **digabung kembali** (E053/E203 ke `experiments/`; riwayat di tag
> `archive/volcarch-genetics-20260730`); kanari hijau **217**. (2) **Audit re-entry** menemukan bahwa
> **E069/ADV-3 salah tanda sejak Maret** (ledger C023), dan beberapa aset "katedral" lain lebih lemah
> dari yang tercatat (C024–C030). (3) Jawaban obyektif untuk pertanyaan inti PI:
> **`docs/research_notes/OBJECTIVE_ANSWER_20261001.md`**. (4) Tiga aksi eksposur disiapkan sampai klik
> terakhir (§1). **Sore hari: inti P17 "Two Javas" (sedang di-review ArchCalc) ternyata artefak
> geocoding** (C032; keputusan integritas PI, §1 item 0) **dan P11 NO-GO** (pola lereng barat = efek
> Penanggungan; 39 baris candi duplikat; docx tanpa abstrak) — jangan dikirim.
> **Handoff terbaru: `docs/HANDOFF_20261006.md`** (yang 5 Okt kini di `docs/archive/handoffs/`).
>
> **2026-10-02:** keempat aksi eksposur **selesai** (PI mendelegasikan): balasan JCAA, email VEGAN, **P17 ditarik**,
> koreksi Zenodo P1 terbit. E226 (T7) dibuka lalu **diparkir REVISIT** — identifikasi toponim (11/170) jadi penghambat.

> **In FOCUS MODE (cwd inside `lines/<nn>_*/`) you should not be reading this.** Read that line's
> `CLAUDE.md` + `STATE.md`. This file is the **orbit** view. Per-line `STATE.md` files are
> **authoritative** for their line; if they disagree with this file, the line wins. Narrative goes to
> `docs/JOURNAL.md`, not here.

---

## 1. 🚨 EXPOSURE LEDGER — read before anything else

The binding constraint is **non-exposure, not rigor** (ME#19). The ledger was empty on 2026-08-11; it is
**refilled** as of 2026-10-01. Every item is prepared to the last click. **PI only.**

| # | Action | Ready since | Where |
|---|---|---|---|
| ~~0~~ | ✅ **P17 WITHDRAWN 2026-10-02** (PI delegated; Option A): portal discussion 10:22 + email to redazioneac@ispc.cnr.it 10:24 (portal notification failed). ~~P17 integrity decision.~~ P17 (under review, ArchCalc #365) rests on a candi-vs-inscription contrast that is a **geocoding artefact** (50 Borobudur captions at one point + 42 region placeholders; precise findspots only: gap 13.1 → 0.2–2.3 km, p 0.07–0.85; ledger C032, verified). Tell the editor: withdraw, or notify and offer a revised artefact paper | 2026-10-01 | `docs/correspondence/EMAIL_ARCHCALC_P17_INTEGRITY_NOTICE_DRAFT_20261001.md` (send via the portal channel used on 11 Aug; double-blind) |
| ~~1~~ | ✅ **SENT 2026-10-02 10:09** (Gmail reply in the editor's thread). ~~Send the P2 abstract reply~~ to the JCAA editor. He corrected the title on 31 Aug but refused the abstract ("focuses on the reviewing process, not the work"); unanswered 31 days while round-2 reviewers read the old record | 2026-10-01 (request: 08-31) | **Gmail → Drafts** ("RE: Revised files uploaded …") · `papers/P2_settlement_model/EDITOR_REPLY_ABSTRACT_20261001.md` |
| 2 | ~~Submit P11 to SPAFA~~ → **🔴 P11 NO-GO (2026-10-01) — send neither v0.7 nor v0.8.** On the canonical inventory the western-flank clustering vanishes without Penanggungan (73 candi: p=0.26); 39 of 142 candi rows are duplicate coordinates; the docx has no abstract in either language. Needs rework + a PI decision on reframing | — | `papers/P11_volcanic_informedness/SIG_signoff.md` §2026-10-01 |
| ~~3~~ | ✅ **SENT 2026-10-02 10:11** to niamarniatief@yahoo.com (author address, *Naditira Widya* 2018). ~~Send the outreach email to BRIN's VEGAN group~~ (phytolith/starch/pollen; head Nia Marniati Etie Fajari). Replaces the Vida draft (both its questions are answered publicly); Castillo held as a reserve (UCL post likely lapsed). v1 drafts withdrawn | 2026-10-01 | `docs/correspondence/EMAIL_BRIN_VEGAN_DRAFT_20261001.md` (find the address on the VEGAN page; fallback praps@brin.go.id); old v2 files kept for the record |
| ~~4~~ | ✅ **POSTED 2026-10-02 10:28** (metadata edit, same DOI/file, verified on public API). ~~Post the P1 preprint correction on Zenodo~~ (mandatory; public record states the inverted E069 claim and a "Java-wide taphonomic baseline" from assumed burial-start dates). Zenodo allows editing; ~15 min | 2026-10-01 | `docs/correspondence/ZENODO_P1_CORRECTION_NOTE_DRAFT_20261001.md` (verbatim quotes) |

| 5 | **P8 revision → response letter → resubmit** at OL (conditional acceptance 2026-10-05). Analyses and number audit **done 10-05** (E227, E228). **2026-10-06: tables/Figure 1 (E229), robustness re-run (E230), corrected `.bib`, `VENUE.md`, outline and the draft note to the editor are ready** — and, after two adversarial reads on 10-06, the readings in the outline, work plan (§10.7), editor note and READMEs were corrected twice (E231 with post hoc tables; see the note at the top). **Next = the PI's decisions D1–D10** (one label; headline model; drop the in-sample consensus; reword the negative result; 16-language section; §4.5; tell the editor; write in Word; shorten the robustness subsection; when to ask for the formal letter) **and sending the note to the editor** (`docs/correspondence/EMAIL_OL_EDITOR_P8_NOTE_DRAFT_20261006.md`), then reading the eight reviewer-supplied references, then **the PI writes the revised prose and the response letter (G16)**; G14 freeze (14 days) + G1 on the new text + G12 after upload | 2026-10-06 (everything but the prose) | `papers/P8_linguistic_fossils/REVISION_OUTLINE_20261006.md` (what each section must contain) · `REVISION_WORKPLAN.md` (decisions, point-by-point facts, corrected numbers; §10 = new on 10-06) · `VENUE.md` · `REVISION_OL_20261005.md` (ledger) · OL portal link is in the decision email (not recorded here: public repo) |

🅿 Still parked (PI decision 2026-08-11, unchanged): Verberne reply · P7 preprint notice · Lamqaddam reply.

**Scorecard: 0 final acceptances · 1 conditional (P8, 2026-10-05) · 7 rejections · 1 under review (P2) · 1 withdrawn (P17, 2026-10-02) · 223 experiments** (223 folder lokal, E001–E231; 8 nomor tak pernah dibuat; +E225 13 Agt, +E053/E203 kembali dari repo pendamping 1 Okt — rekonsiliasi 2026-10-01; +E226 T7 2 Okt; +E227/E228 audit dan analisis revisi P8, 5 Okt; +E229/E230 tabel revisi dan uji ulang ejaan/panjang P8, 6 Okt; +E231 hitungan deskriptif arti "berkode" di ABVD untuk P8, 6 Okt).
**2026-10-02: all four ledger items done** (JCAA reply, VEGAN email, P17 withdrawal, Zenodo notice). Ledger is
empty of PI actions except waiting. **P2** in round-2 review since 31 Aug · **P8** (OL) — **decision 2026-10-05: accepted pending revisions** (ledger item 5).

---

## 2. ⏰ Clock

| Deadline | Item |
|---|---|
| — | No hard external deadline. The JCAA editor's request (31 Aug) is the most time-sensitive item. **P8 revision: no deadline stated** (2026-10-05); ask the editor only if one is needed. |
| Dec 2026 | Edinburgh PhD window — **subject to the PI's current PhD plan** (PI-owned; may no longer apply) |

---

## 3. Line status

| # | Line | Temp | Next action | Owner |
|---|---|---|---|---|
| **01** | [spatial](../lines/01_spatial/STATE.md) | 🟡 WAITING | P2 in round-2 review (abstract reply sent 10-02) · P17 withdrawn 10-02 · **P11 NO-GO** → reframe = PI | PI + Claude |
| **02** | [taphonomy](../lines/02_taphonomy/STATE.md) | ⚠ WARM | Verify C024–C030 one by one; **hold P1 (JASREP)** until its calibration anchors are audited; West Java skeleton arms need rebuilding | Claude |
| **03** | [paleoenv](../lines/03_paleoenv/STATE.md) | ⏳ WAITING | VEGAN email sent 10-02; follow up ≈ 23 Oct if silent. T5 needs pre-400 paleosols | PI |
| **04** | [language_text](../lines/04_language_text/STATE.md) | 🟢 P8 REVISION | **P8 conditionally accepted 10-05**; audit, analyses, tables, robustness re-run, outline, venue file done (E227–E231); readings corrected after two adversarial reads (10-06; work plan §10.7) → **waiting on PI decisions D1–D10 (D4: three options), the source of the Makasar 38 %, and the note to the editor**, then the PI writes; P5 rewrite or PARKED.md (C018, overdue) | **PI** |
| **05** | [archival_nlp](../lines/05_archival_nlp/STATE.md) | 🅿 E226 REVISIT | **E226 (T7) parked 10-02**: frame of 266 villages built; I1 identifications 11/170 → unpark needs Kusen 1990/91 / Resiyani 2010 lists. Next when warm: T2 or E211 | Claude |
| **06** | [thesis](../lines/06_thesis/STATE.md) | 🛑 FALLOW | Objective answer written (subtract-only). L1 amendments await PI (§4) | PI |
| **07** | [career](../lines/07_career/STATE.md) | ✅ LEDGER CLEARED 10-02 | waiting on journals (P2, P8) | — |
| — | population-evidence channel (E053, E203) | — | **re-merged 2026-10-01** — back in `experiments/` (E053 → 02+06, E203 → 06); claims downgraded in the objective-answer note §3; see `COMPANION_REPOS.md` | — |

---

## 4. Decisions waiting on the PI

**NEW 2026-10-02 — publication strategy after 7 rejections:** `docs/research_notes/STRATEGI_PUBLIKASI_20261002.md`
(diagnosis: 10 submissions in 35 days, 6/7 rejections at the desk). **DECIDED same day (PI delegated):** SIG gates
G13–G16 binding (WIP ≤2 in review / 1 drafting; rest ≥14 days; venue first + zero cost of any kind; PI's own prose);
G10 human domain reader required for archaeology venues; domain co-author by default there; **Naskah A (P23, sīma
village toponyms) first**. G13 was full (P2, P8 in review) → no new submission until a decision arrives. **2026-10-05:** P8's
decision arrived (conditional). Whether a manuscript in revision still occupies a G13 slot is undefined; the conservative reading is
yes (P23 stays blocked) — **PI call**.

**P8 revision (2026-10-06) — ten decisions, each with a recommendation: `papers/P8_linguistic_fossils/REVISION_WORKPLAN.md` §2 (D1–D7)
and §10.5 (D8–D10).** Four need an explicit yes because they are not delegated: **D6** (cut §4.5, the Javanese-script section),
**D7** (send the note to the editor — draft ready), **D8** (write the revision in the journal's Word template: a Word file is
required for the final version), and whether the PI's own prose (G16) stays the rule for this revision. Also his: co-author
sign-off (Go Frendi Gunawan), reading the eight reviewer-supplied references, a human reader for the lexical examples (G10).

**Urgent (one sitting, each prepared):** the three §1 actions.

**New from the 2026-10-01 audit** (details: `docs/research_notes/OBJECTIVE_ANSWER_20261001.md` §7):
- **L1 amendments:** the §9 "within-island control" criterion is confounded (coast vs interior, trade node) and cannot be marked HOLDS; replace the
  Kutai-"oldest" framing with "regional onset"; review layer L2 coastal submersion for 0–400 CE (sea level
  at/above present, deltas prograding); give the "peradaban vulkanik" character claim one operational
  falsifier.
- **Manifesto §1 "decisive test":** the West Java natural experiment in its skeleton form cannot carry the
  thesis (three of its four arms factually wrong, coast-vs-interior confound). Proposed replacement:
  **T0 + T4-desk + T7**, then field T4. PI's document; not edited.
- **Research direction, next 3 months** (default YES; ordered by decisive power, per the WF2 methodology review):
  Week 0 = the §1 actions (P17 decision, JCAA reply, P1 Zenodo notice, VEGAN email) → weeks 1–3: **T0** (verify
  Liyangan's early ¹⁴C dates) and **T4-desk** (depth of the ±400 CE surface from published dated sections and
  borelogs; doubles as the D6 standing falsification: median <2 m across ≥10 sections withdraws P1's detection
  horizon) → weeks 2–5: **T7** (detection calibration: villages named in 8th–10th c. inscriptions vs recorded
  settlements — the first test that can separate H3 from H6) → then **T2** (pre-registered, zones from T4-desk).
  Submission milestone: an Indonesian-language synthesis ("why Nusantara's history starts ~400 CE") to a
  national zero-APC journal (outline: `docs/drafts/SINTESIS_400M_OUTLINE_DRAFT_20261001.md`). Details: `docs/research_notes/OBJECTIVE_ANSWER_20261001.md` §5, §7.
  - **Update 2026-10-01 afternoon (desk tests; note §9, `docs/research_notes/DESK_TESTS_T0_T4_T7_20261001.md`):**
    - **T0:** Liyangan is not a pre-400 case. Its early dates are soil samples and charcoal with no context or
      lab codes (C037), so H1b now has no candidate.
    - **T4-desk:** cannot reach D6 from open sources. The 9th-c. floors at ±2–7 m are lower bounds (C038), and
      the key sources are closed.
    - **T7:** feasible, but the numerator register is missing. E129 is unusable (C036).
    - **Proposed re-order:** **T7 next** (draft `docs/drafts/T7_DETECTION_CALIBRATION_DESIGN_DRAFT_20261001.md`;
      the PI fixes N_ref and m). T0 and T4 wait on external data (lab sheets from the excavator; the Gertisser
      2012 supplement; JVGR 100). T2 follows T4.
- ~~**P17 (urgent):** withdraw, or notify~~ → ✅ **withdrawn 2026-10-02** (§1 item 0).
- **P11:** NO-GO. Choose: (a) reframe as a Penanggungan-centred paper (strong pattern: 69 candi, 62% west — but 14 of the 43 "western" rows are Trowulan temples 26–56 km away on the Brantas plain, so partly "Trowulan lies west of Penanggungan";
  p=1.4×10⁻²⁰), (b) rework the island-wide version on de-duplicated data and accept a weaker claim, or (c) park.
- **P1:** hold v5.0 (→ Archaeological Research in Asia) until its calibration anchors pass a WS-E-style audit (default YES).
- **P1 Zenodo preprint (public record) — mandatory, ~15 min:** `10.5281/zenodo.19081502` (2026-03-18) states the
  inverted E069 claim (PDF lines 400–406) and a "Java-wide taphonomic baseline" from assumed burial-start dates
  (abstract). Notice text, verbatim quotes: `docs/correspondence/ZENODO_P1_CORRECTION_NOTE_DRAFT_20261001.md`.

**Carried over:** D5 L1 Java/Nusantara disaggregation · D7 audit rule (submission trigger or 14 days) ·
DJKI HKI filing (4 docs ready) · dashboard model regeneration on the 30-volcano inventory.

---

## 5. Portfolio

| Paper | Line | Status |
|---|---|---|
| **P2** Settlement model | 01 | ⏳ **round-2 review at JCAA #280 since 2026-08-31.** Title corrected by the editor; **abstract reply owed** (§1 item 1). Claim set: `review_package_20260727/10_SET_KLAIM_TERKOREKSI.md`. |
| **P17** Two Javas | 01 | ⛔ **WITHDRAWN 2026-10-02** (ArchCalc #365; never left the submission stage). Central result = geocoding artefact (C032); second result (929 CE shift) rides on the same points. Record: `docs/correspondence/EMAIL_ARCHCALC_P17_INTEGRITY_NOTICE_DRAFT_20261001.md`. A methods paper on the artefact is possible later (not planned). |
| **P8** Linguistic fossils | 04 | 🟢 **conditionally accepted 2026-10-05** — *Oceanic Linguistics* (Scimago Q2) OL-03-2026-11; revision pending, no deadline. Amien first + corresponding. Plan: `papers/P8_linguistic_fossils/REVISION_OL_20261005.md`. |
| **P23** sīma village toponyms (Naskah A) | 05 | 📝 PLANNING (2026-10-02) — critical case study (toponym resolution is the bottleneck) from E226's frame → **DHQ "Case Study"** (no fees; JDMDH backup). Needs human validation of a sample (epigrapher asked: Nastiti, 13:55). Drafting allowed; submission blocked by G13 until P2 or P8 decides. `papers/P23_sima_village_toponyms/README.md` |
| **P11** Volcanic informedness | 01 | 🔴 **NO-GO 2026-10-01** — western-flank claim is a Penanggungan pattern (p=0.26 without it); 39/142 duplicate candi; docx lacks abstracts; E069 claim already removed (v0.8). Rework queued; reframe = PI. `papers/P11_volcanic_informedness/SIG_signoff.md` §2026-10-01. Rejected 2× before (editorial). |
| **P1** Taphonomic framework | 02 | rejected 2×. Current manuscript **v5.0 → *Archaeological Research in Asia*** (per `CANONICAL.md`; the E069 passage was already deleted there; superseded JASREP v4.0 corrected anyway). v5.0 still reports the 4.4 mm/yr "baseline" → **C024 audit before submission**. Public preprint needs the correction notice (§4). |
| **P5** Volcanic ritual clock | 04 | rejected (BKI) → *Asian Ethnology*. Rewrite overdue (C018). |
| **P9** Peripheral conservatism | 04 | rejected (JSEAS). HOLD → DHQ. |
| **P16** Textual archaeology | 04 | 🅿 PARKED — convergence refuted (E090 v7). |
| **P0** / MASTERPIECE | 06 | fallow. |
| **D1** / **D2** | 05 / 02 | ✅ PUBLISHED 2026-08-11 — D1 `10.5281/zenodo.21882007` · D2 `10.5281/zenodo.21882247`. |
| **P7** TOM | 02 | ☠ dead — peer-rejected; preprint correction notice parked. |
| **P3, P14** | 02, 04 | discontinued. **P18** HOLD. **P15** dissolved into P5. |

---

## 6. Orbit-mode rituals

**Session start:** read this file → `python tools/check_doc_sync.py` (RED ⇒ fix drift first) → latest
`docs/HANDOFF_*.md` → empty `inBox/`.
**Mata Elang** (strategic review) — criticism matrix {confidence × reversibility}; records in
`docs/research_notes/MATA_ELANG_*.md`. Audit cadence: on a submission, or every 14 days (D7, default YES).
The 2026-10-01 re-entry audit counts as one.

> ⚠ This dashboard opens with the ledger, not with `IDEA_REGISTRY.md`, on purpose: stepping out one level
> must not become the way to avoid sending an email.

**Where ideas are kept safe:** `docs/IDEA_REGISTRY.md` · `docs/TRIGGER_MAP.md` · `papers/*/PARKED.md` ·
`docs/drafts/` · `docs/research_notes/*_LEAD_*.md`.

**Binding gates:** `docs/SUBMISSION_INTEGRITY_GATE.md` (G1–G12; **lesson C023: G1 must re-derive the
sign and interpretation, not only the number**) · **F9** don't count correlated channels · **F10** don't
cite the manifesto. (`docs/EVAL.md` is flagged zombie, C010.)

---

## 7. Housekeeping

- ✅ **2026-10-01:** `volcarch-genetics` **re-merged** (PI moved it back in). E053 + E203 restored to their
  original `experiments/` paths (byte-identical to the pre-split versions); working note →
  `docs/drafts/`; subfield summary → `docs/bibliography/05_paleogenomics/`; companion README archived;
  the companion's 2 commits kept at tag `archive/volcarch-genetics-20260730`. Canary 217 ✅.
- ✅ **2026-10-01:** E069 corrected in place (README, canonical note, both scripts; re-run reproduces every
  number). Propagation: ledger C023.
- ✅ **2026-08-13:** canary v2 (`tools/check_doc_sync.py`) wired into session start.
- ✅ **2026-07-30:** experiment index with a `lines` field for every experiment (`tools/scan_experiments.py`).
- `.claude/` holds stale Feb 2026 handoffs and CODEX prompts mixed with `settings.local.json`.
- `AGENTS.md` (Codex mirror of CLAUDE.md, added 2026-09-04) is kept in sync with CLAUDE.md.

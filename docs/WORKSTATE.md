# WORKSTATE — Orbit Dashboard

**Updated:** 2026-10-20 · **This file is short by design. Keep it that way.**
Previous version (P2-submitted era, 11–13 Aug): `docs/archive/WORKSTATE_snapshot_20260813.md`.

> ## ▶ WEDNESDAY 21 OCT — START HERE: `papers/P2_settlement_model/AGENDA_PI_20261021.md`
> PI 20 Oct evening: "lakukan yg bisa kamu lakukan tanpa saya, yg perlu saya pending dulu, kita lakukan besok hari
> rabu". Done without him:
> - Harvard reference list (JCAA form; `build_jcaa_bibliography.py`);
> - the E235 figure in manuscript style and Tables S7–S8;
> - the response-letter skeleton; alt-text candidates;
> - a guard so v0.3 is never regenerated over his edits.
>
> Mailbox 20 Oct: nothing from JCAA (waiver), Nastiti, VEGAN or arXiv.
> **The agenda holds:**
> - A1–A11 for P2: prose plan, the line-157 sentence, E235 figure placement, alt text, acknowledgements, AI
>   statement, portal login, Go Frendi, Figures 1/5, hygiene of the round-1 letter, abstract (promised text vs
>   R1's idioms);
> - B1 the 23 Oct follow-ups; B2 the E234 run; B3 K6; B4 P23.
>
> ## 🟠 2026-10-20 (later) — PI "ok, lakukan yg perlu dilakukan" → defaults taken; Claude's part of the P2 revision done
> - **Waiver email SENT 08:07** (D4). G15 stays RED until a written answer.
> - **E235 run as pre-registered** (design pushed first, `3ac41cc`; numbers re-derived independently):
>   - above **1,000 m**, 97.5 % of East Java's land is volcanic and 0.4 % non-volcanic, so **R2's comparison does
>     not exist in the study area**;
>   - below 1,000 m, suitability +0.05 = INCONCLUSIVE; design disagreement NOT SPECIFIC.
> - **Built:**
>   - Figures 1 (relief, graticule, 20/40 km zones, inset) and 5 (legend off the map);
>   - `submission_jcaa_v0.3.tex` (mechanical only; clean compile);
>   - `VENUE.md`;
>   - **writing pack for the PI: `papers/P2_settlement_model/REVISION_R2_WRITING_PACK.md`**.
> - **JCAA AI policy forbids "ghost-writing"** (AI writing methods, results or discussion). The PI's own prose is
>   therefore a journal rule for v0.3, not only G16.
> - **New: C078** — v0.2 §3.7 robust-core definition and caveat are wrong.
> - **References (C076) fixed and re-checked.** But the line-157 sentence ("strong gains") is supported by none of
>   the three papers → the PI rewrites it (`papers/P2_settlement_model/REFERENCE_FIX_20261020.md`).
> - **Open:**
>   - **the PI's prose → freeze ≤ 9 Nov → upload ≤ 16 Nov**;
>   - Harvard-style pass of the reference list;
>   - D6; D7 (portal: annotated copy);
>   - tell Go Frendi.
> - Ledger next **C079**.
>
> ## 🟠 2026-10-20 — P2 / JCAA #280: REVISIONS REQUESTED (decision 2026-10-19; due **16 Nov 2026**)
> The PI pasted the decision email. Round 2, for both reviewers:
> - **R1** ("Resubmit for Review"): contribution excellent; **clarity fair**.
> - **R2** ("Revisions Required"): **methods fair**.
>
> The revised version may be reviewed again.
> **Full triage, gates, timeline and decisions D1–D7: `papers/P2_settlement_model/REVISION_JCAA_R2_20261019.md`.**
> - **R2 (the gate):** volcanic setting or simply elevation? Our round-1 control (E219 C) had almost no non-volcanic
>   land in its top elevation quintile (868 vs 41,878 matched cells; C077).
>   - **E235** pre-registered (band-by-band control on the stored E221 maps; not run; D3).
>   - New Figure 1 with topography.
> - **R1:** the paper must be self-contained:
>   - no reviewer, revision or earlier-version talk (38 hits);
>   - a plainer research question and explanations;
>   - three idioms named ("dial", "judging at home", "Jaccard said no");
>   - "§" → "Section";
>   - Figure 1 as a real map; Figure 5's legend.
> - **References (C076):** the Comer et al. entry is wrong, as R1 said. **Crossref on all 26 DOI-bearing cited
>   entries: 3 wrong** (comer2023 and wang2023 have invented authors; castiello2021, a JCAA paper, has the wrong
>   title). All three are cited in one sentence, which still has to be checked against the papers.
> - **G15 not met:** JCAA's page now lists an APC of 651 (pound sign garbled in the fetch), waived only in writing.
>   The April waiver request is still undecided → draft email `docs/correspondence/EMAIL_JCAA_WAIVER_STATUS_DRAFT_20261020.md` (D4).
> - **PI decisions by 23 Oct:**
>   - D1: G16, who writes the prose;
>   - D2: G14, a strict rest (freeze 2 Nov) or an adversarial read in its place (freeze 9 Nov);
>   - D3: E235;
>   - D4: send the waiver email;
>   - D6: the acknowledgements line;
>   - D7: portal login for any annotated copy;
>   - tell Go Frendi.
> - The 9 Nov status query is void. Canary **227**; next free **E236**; ledger next **C078**.
> - The 10-19 work (E234 and others) is **still uncommitted**, waiting for the PI's word.

> **2026-10-19 (latest) — PI "ya, buatkan desain jalur sumber untuk E216" → E234 designed (pre-registration; not run):**
> - `experiments/E234_e216_source_route/`: DESIGN + claim register (41 rows, 15 Tier 1) + agent prompts.
>   - Two blind AI domain reads (as a palynologist reviewing for VHA).
>   - Two blind inventory searches (Neotoma + fixed EN/ID queries).
>   - Source extraction with a verbatim guard.
>   - A re-run of E216's own code under published ranges.
>   - A kill switch: a dated 0–500 CE record already at the heartland.
> - **Found while designing:** H1's "P = 0.000 at every grid point" rests on an uncited 1/100 factor (all clearing
>   placed at two points). Under uniform allocation, E216's own formula gives 21.9 pp at the central population,
>   against 17.5 pp. If the run confirms it, the existing network *would* detect a central-sized population spread over
>   the island. That moves toward E214's counter-evidence.
> - **Waiting for the PI:**
>   - (1) push the design (it freezes the pre-registration) and run (~7 agents);
>   - (2) after the results: line 03 rule 2 / E216 checklist step 1 → route H or S (proposed text in DESIGN §9);
>   - (3) confirm that K9 does not apply (an audit of existing work, not a new test).
> - Canary **226**; next free **E235**; ledger next **C076**.
>
> **2026-10-19 (later) — PI: "AI penuh = fitur, bukan bug … kalau perlu ahli, pivot"; then "lanjutkan" on the defaults:**
> - **Rule** in CLAUDE.md/AGENTS.md "When Unsure": check sources yourself; pivot when a human judge is truly needed;
>   disclose AI use.
> - **SIG G10:** route H (human) or **route S** (sources table + AI domain read + statement), taken after 14 days without a
>   reader.
> - **Co-authors and readers** are invited, never waited for.
> - **C074 domain flags** closed from the ACD and Adelaar 1995.
> - **P23 validated without an epigrapher — E233 PASS.** Frame vs the DHARMA editors' published translations
>   (30 inscriptions, 216 occurrences): κ 0.84; strict name error 5.4 % (CI 2.8–9.9 %). P23's remaining blockers are the
>   PI's prose (G16) and G13.
> - **Candidates for the same treatment:**
>   - E216's "palynologist co-author — BLOCKS everything" (line 03);
>   - the ⚠ "needs a historical linguist" flags in E228–E231 (P8 is in review; nothing due);
>   - E226's unpark condition (b) ("a domain expert codes I1").
>   - Each needs a design; none is started.
> - Canary 225 → 226 with E234 (block above).
>
> **2026-10-19 — orbit session ("lanjutkan yg perlu dilanjutkan"):**
> - **Mailbox:** only the journal's receipt of P8 R1 (17 Oct; received 16 Oct). Nothing from Nastiti, VEGAN, JCAA or
>   arXiv, so the 23 Oct follow-ups stand (drafts ready; check the mailbox again first).
> - **Downstream audit C050 done:** `docs/research_notes/DOWNSTREAM_AUDIT_C050_20261019.md`.
>   - E087 stands, read narrowly.
>   - **E107's "Mon-Khmer positive control" withdrawn (C074)**: between-language comparison; within Iban, 1 of 4
>     predictions.
>   - **E130 FAILED (C075)**: label among the inputs; its "438 substrate words" are the 438 uncoded forms.
>   - ADV-5 = GREY ZONE again; L4 "unsettled".
>   - **P8 R1 as submitted uses none of it.**
> - Ledger next **C076**; canary 225; next free **E234**. Handoff: `docs/HANDOFF_20261019.md`.
> - **Not committed** (with the 15 Oct beads addendum): waiting for the PI's word.
>
> **2026-10-15 (~10:08 WIB) — P8: revision R1 SUBMITTED to *Oceanic Linguistics* (OL-03-2026-11R1).**
> The editor (Sander Adelaar) had replied on 13 Oct: go ahead with the corrections of our own; add a note telling the reviewers in
> general terms how the new version relates to the old; the **same reviewers will be consulted again**, anonymously; acknowledge the
> use of AI (it already is, in the anonymised article). Letters **v0.6** (note for the reviewers at the top of the response; cover
> letter answers the editor), checked twice by an independent reader; article, figures, tracked copy unchanged. PI: six small
> decisions at default, co-author agreed, "Ya, submit hari ini". Pushed (`7fc342c`, tag `p8-r1-submitted-20261015`), G12 nine of
> nine, Submit pressed. **Nothing is due from us until the journal writes.** Portfolio: 1 under review (P2-JCAA) + P8 R1 in
> second review. Next (orbit): P23 Naskah A, downstream audit C050, T7/E226; I-150–I-152 unlocked. Handoff: `docs/HANDOFF_20261015.md`.
> Ledger next **C072**; canary 223; next free **E232**. *(The blocks below are the state before the submission.)*
> **Same day, later — orbit review on the PI's request:** `docs/research_notes/ORBIT_REVIEW_20261015.md` (position, opinion,
> next steps, and a **decision agenda K1–K9 with defaults** — §4 below). **PI: "setuju default, K6 nanti saya putuskan"** →
> K1 P5 PARKED · K2 P11 PARKED · K3 P1 held · K4 G13 definition · K5 G16 for P23 · K7 E232 designed (not run) · K8 dates
> in §2 · K9 stop list in §6. **K6 (program direction) open — the PI decides by 2026-11-15.** Canary **224**, next free **E233**.
>
> **2026-10-12 (~11:00) — P8: the thorough review is done; the portal draft was UPDATED and is ready; NOT SUBMITTED — waiting for the PI's word.**
> PI: "baca handoff, review sekali lagi P8, kita selesaikan sekarang, dan pastikan semua gambar, diagram dan referensi sudah sesuai" · "dan
> pastikan semua sesuai aturan dan template". Found (three independent checkers, every finding verified by the controller): the
> **reference list was in the wrong style** (Unified, not the journal's) and four reference facts were off; **Figure 2 drew the profile as
> an output of the classifier**; four statements said too much ("the same fact counted again", Table 7's "nearest look-alike", the
> pre-specification claim, "sensitivity at most 38 percent") and **one count was wrong since v0.2 (seven → six)**; the journal's house
> style and template layout were not followed. **Done:** article **v0.4**, Figures 1–2 redrawn, letters **v0.5**, tracked copy rebuilt
> (exact both ways); six files replaced in the portal draft, abstract field updated, merged PDF read in full, four approvals,
> **G12 nine of nine**, "ready for submission", Save and Exit. Mailbox 09:20: nothing from Sander or arXiv. **Not committed, not
> pushed** (waits for the PI's word, like Submit). **Start here: `docs/HANDOFF_20261012.md` §1 →
> `papers/P8_linguistic_fossils/revision_v0.2/BACA_DULU_P8_20261012.pdf`.** Ledger C070. The blocks below are the state of 8 October.
> **~11:05 — the PI: "karena hari ini dan besok saya ada kesibukan, mungkin rabu akan saya kerjakan bagian saya untuk kemudian kita submit final papernya, btw, bikin handoff dulu, rabu kita buka dan kita bahas apa tugas saya".** So: nothing further in the portal; **Wednesday 14 Oct** the handoff is
> opened and **his tasks are discussed first** (`docs/HANDOFF_20261012.md` §1: reading copy, six small decisions, a look at the merged
> PDF, what to do about Sander's silence, the co-author, the word "submit"), then the Submit sitting (§2). No commit or push was asked
> for; the day's work is on disk only.
>
> **2026-10-08 (close, ~14:45) — P8: revision R1 is COMPLETE IN THE PORTAL AS A DRAFT and NOT SUBMITTED; pushed; Zenodo corrected.**
> After the PI's go-ahead (browser, his logins; "jangan submit dulu, semua harus sudah siap saat submit"): 22 commits pushed (E229–E231
> now public), the draft README in the Zenodo record replaced in place (**same DOI 10.5281/zenodo.23202245**), and the nine final files
> put into the portal draft — article v0.3, five figures, signed cover letter v0.4, response v0.4 without names, tracked copy — with
> Notes for Editors rewritten, the merged PDF read, **G12 nine of nine**, all approvals ticked, "ready for submission". **14:48: the PI
> approved the content ("setuju", in answer to the reading copy `revision_v0.2/BACA_DULU_P8_20261008.pdf`) — that is not the word
> "submit". ~14:55, the PI: "jangan disubmit dulu, hari senin kita review menyeluruh sekali lagi, kemudian submit, sekarang BELUM".
> Open: Sander's answer (none at 14:48); co-author; **Monday 12 Oct: a thorough review once more (handoff §7 item 0), then Submit on the
> PI's word.** State on one page: `revision_v0.2/PORTAL_STATE_20261008.md`. Handoff: `docs/HANDOFF_20261008.md` §1.
>
> **2026-10-08 (afternoon, ~13:30) — P8: the R1 package was checked claim by claim; letter v0.3 is superseded before it was read.**
> The letter (v0.2 in the portal, v0.3 of the morning) said 16 things about the manuscript that the manuscript does not bear out
> ("'false alarm' is used once", "'positive' no longer appears", "Section 2.5 explains each column", "Figure 2 is removed" …); two cells
> of Table 8 disagreed with the text; three figures had defects; the signed letter sat in the reviewer-facing slot. An independent
> adversarial read of the rewritten letters then found 26 more (two high; all verified and corrected). **Built, all local:**
> article **v0.3** (three sentences, two cells), Figures 1/3/4 redrawn, tracked copy rebuilt, **cover letter v0.4 (signed) + response v0.4
> (no names)**, repeatable checks. ⚠ **Two new preconditions of Submit:** (1) **push** — GitHub stops at 5 Oct, so E229–E231, which the
> data statement cites as public, are not; (2) **Zenodo version 1.1** — the README in the public deposit is the working draft with
> unfilled placeholders. Mailbox 12:40: no answer from Sander, none from arXiv. *(State at ~13:30; both preconditions were met and the
> files uploaded the same afternoon — block above.)* Ledger C069. **Start here: `docs/HANDOFF_20261008.md` §1 →
> `papers/P8_linguistic_fossils/revision_v0.2/PACKAGE_R1_20261008.md`.** The block below is the morning's state.

> **2026-10-08 (~10:20) — P8: the journal's Managing Editor answered the practical questions; Sander's answer on content is pending.**
> Fees: **none for standard publication** (G15 fee row closed). Anonymise the revision if another round is needed. A **tracked-changes
> copy and a clean copy** are both wanted; the response letter as a **Word document with bulleted answers**, substantive points only.
> Whether the corrections of our own may be handled inside this revision, and whether a further review round is needed, are for
> Sander — **no answer yet; the mail is not a sign on the substance.** Review of the R1 package (still not submitted): it lacks the
> tracked copy (no Word baseline exists), the letter is long prose and its Part D asks what is now answered (ledger C068).
> **Built the same morning (~11:00): the tracked copy and a bulleted letter v0.3** (v0.2 wrongly said the DOI is in the text — the anonymised
> file redacts it; fixed). **Next: the PI reads the letter v0.3**; the portal still holds v0.2 and nothing is uploaded. **Submit stays next week**,
> PI's word; read the mailbox first. Record: `docs/correspondence/EMAIL_OL_EDITOR_P8_REPLY_20261008.md`;
> steps: `papers/P8_linguistic_fossils/REVISION_WORKPLAN.md` §10.13; what changed: `revision_v0.2/LETTER_v0.2_to_v0.3_CHANGES.md`.
> *(Superseded the same afternoon: v0.3 is replaced by v0.4 in two files, and the article and three figures change too — block above.)*

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

> **2026-10-06 (close) — P8: material ready, readings corrected three times, sources read; the ball is with the PI.**
> Start (superseded — now `docs/HANDOFF_20261008.md`): `docs/archive/handoffs/HANDOFF_20261006.md` §1, then `papers/P8_linguistic_fossils/REVISION_WORKPLAN.md` §10.7.
> Today: tables and Figure 1 (E229), robustness re-run (E230), what "coded" covers (E231), corrected `.bib`, `VENUE.md`,
> outline, editor-note fact list; three adversarial reads found the numbers sound and several readings not — three rounds
> of correction (ledger C055–C065). **What changes for the paper:** "coded in ABVD" is not "inherited" (Table 1's 15.5 %
> for Muna rests on a second Muna list); Makasar has more *uncoded* meanings, cause undecided — the published 38 % is
> Blust's count (Bellwood 1997: 115), the same tradition as ABVD, not an independent figure; "fewer prefixes" is
> withdrawn and "more" is not established; the cross-list test says nothing about a shared layer (**D4: three options**).
> Six of the reviewer's eight references fetched with exact pages (`REFERENCE_CHECK_20261005.md` Part D). Portal: no due
> date; its stored abstract and keywords must be rewritten at resubmission. A Sulawesi specialist is not a blocker.
> ⚠ **Security:** the arXiv paper password of the P8 preprint sat in four tracked files since April (C061; redacted) —
> **request for a new one sent to arXiv 14:14, reply pending**; the old one is still in the public git history.
> Canary green 223. Commits `450aef6`, `1e269f3`, `ae14804` + the closing one on `main`, **not pushed**.
> **10-07 (check session, 09:00–09:20):** numbers re-anchored (nothing to correct); two guards C066 (screen sensitivity =
> upper bound; glottal OR not carried by Tolaki); **Blust 2012:556 read in JSTOR with the PI's login** — all eight reviewer
> references now located and read; no arXiv reply, no journal mail. Work plan **§10.9** adds three items to the PI's
> sitting (title, abstract content, name of the G10 reader).
> **10-07 (close, 14:45) — revision R1 fully prepared in the portal, NOT submitted: PI defers the click to next week.**
> Written by Claude on the PI's decision (C067): manuscript (24 pp., 8 tables, 5 figures), cover-and-response letter,
> note to the editor (sent via the portal form 10:36), Zenodo data release (DOI 10.5281/zenodo.23202245); two adversarial
> reads + number trace; G12 passed; the portal's merged PDF looked at page by page; file order fixed; metadata reviewed;
> all three Approve boxes ticked (merged PDF, cover letter, revision summary; the portal keeps them disabled until each PDF has been opened from the row), all eight steps show valid and the Submit page reads "Your manuscript is ready for submission". **Next sitting: `papers/P8_linguistic_fossils/REVISION_WORKPLAN.md` §10.12.** Open external items:
> editor's answer to the note (G15 fees, anonymity, tracked changes — *practical part answered 10-08, see top block; Sander's answer on content open*); arXiv paper-password reply (C061).
> **10-07 (09:21–09:45):** the PI said "lakukan yg perlu dilakukan" → D1–D6, D8, D9 carried out on the material (work plan
> **§10.10**): Word skeleton in the OL template (`revision_v0.2/P8_revision_v0.2_skeleton.docx`, eight tables, references),
> response-letter scaffold (`RESPONSE_LETTER_SCAFFOLD_v0.2.docx`), release description completed (C065), E231 P13.
> **10:15 — PI waives G16** (ledger C067): Claude writes the revision and the letters; one adversarial read + G1 trace
> replaces the 14-day rest; G10 not met (stated). **10:36 — D7 DONE: the note to the editor was sent through the portal's
> "Send Manuscript Correspondence" form** (PI logged in and ticked the CAPTCHA; `docs/correspondence/EMAIL_OL_EDITOR_P8_NOTE_20261007.md`).
> Draft v0.2 of the manuscript (OL template, 24 pp.) and the cover-and-response letter exist; two independent reads
> returned (481 numbers traced, 7 small mismatches; 5 high / 17 medium wording findings) → correction pass, then Zenodo
> (PI login) → portal upload (PI login) → G12.
> **11:50 — ZENODO PUBLISHED (D3, the P8 release): DOI 10.5281/zenodo.23202245** (six files, CC BY 4.0; the PI pressed Publish). DOI inserted
> in the data statement; Word files rebuilt. Next: portal upload.

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
> **Handoff terbaru: `docs/HANDOFF_20261012.md`** (yang 5, 6, 7 dan 8 Okt kini di `docs/archive/handoffs/`).
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
| ~~1~~ | ✅ **SENT 2026-10-02 10:09** (Gmail reply in the editor's thread). ~~Send the P2 abstract reply~~ to the JCAA editor. He corrected the title on 31 Aug but did not accept the abstract (in his view it described the review process, not the work); unanswered 31 days while round-2 reviewers read the old record | 2026-10-01 (request: 08-31) | **Gmail → Drafts** ("RE: Revised files uploaded …") · `papers/P2_settlement_model/EDITOR_REPLY_ABSTRACT_20261001.md` |
| 2 | ~~Submit P11 to SPAFA~~ → **🔴 P11 NO-GO (2026-10-01) — send neither v0.7 nor v0.8.** On the canonical inventory the western-flank clustering vanishes without Penanggungan (73 candi: p=0.26); 39 of 142 candi rows are duplicate coordinates; the docx has no abstract in either language. Needs rework + a PI decision on reframing | — | `papers/P11_volcanic_informedness/SIG_signoff.md` §2026-10-01 |
| ~~3~~ | ✅ **SENT 2026-10-02 10:11** to niamarniatief@yahoo.com (author address, *Naditira Widya* 2018). ~~Send the outreach email to BRIN's VEGAN group~~ (phytolith/starch/pollen; head Nia Marniati Etie Fajari). Replaces the Vida draft (both its questions are answered publicly); Castillo held as a reserve (UCL post likely lapsed). v1 drafts withdrawn | 2026-10-01 | `docs/correspondence/EMAIL_BRIN_VEGAN_DRAFT_20261001.md` (find the address on the VEGAN page; fallback praps@brin.go.id); old v2 files kept for the record |
| ~~4~~ | ✅ **POSTED 2026-10-02 10:28** (metadata edit, same DOI/file, verified on public API). ~~Post the P1 preprint correction on Zenodo~~ (mandatory; public record states the inverted E069 claim and a "Java-wide taphonomic baseline" from assumed burial-start dates). Zenodo allows editing; ~15 min | 2026-10-01 | `docs/correspondence/ZENODO_P1_CORRECTION_NOTE_DRAFT_20261001.md` (verbatim quotes) |

| 5 | **P8 revision → response letter → resubmit** at OL (conditional acceptance 2026-10-05). **[10-12: final review done — article v0.4, Figures 1–2, letters v0.5, tracked copy in the portal draft, approved, G12 nine of nine, NOT submitted; waiting for the PI's word; `revision_v0.2/PORTAL_STATE_20261012.md`.]** **[10-08: R1 prepared in the portal, Submit next week; Managing Editor answered the practical questions — no fees, tracked + clean copy, Word letter with bullets; Sander's answer on content pending. Afternoon: package checked, rebuilt, pushed, Zenodo corrected in place, **all nine files in the portal draft, approved, G12 passed, NOT submitted; content approved by the PI 14:48 ("setuju")** — `revision_v0.2/PORTAL_STATE_20261008.md`; open: Sander, co-author, then Submit on the PI's word. The text below is the 10-06 state.]** Analyses and number audit **done 10-05** (E227, E228). **2026-10-06: tables/Figure 1 (E229), robustness re-run (E230), corrected `.bib`, `VENUE.md`, outline and the draft note to the editor are ready** — and, after two adversarial reads on 10-06, the readings in the outline, work plan (§10.7), editor note and READMEs were corrected twice (E231 with post hoc tables; see the note at the top). **Next = the PI's decisions D1–D10** (one label; headline model; drop the in-sample consensus; reword the negative result; 16-language section; §4.5; tell the editor; write in Word; shorten the robustness subsection; when to ask for the formal letter) **and sending the note to the editor** (`docs/correspondence/EMAIL_OL_EDITOR_P8_NOTE_DRAFT_20261006.md`), then reading the eight reviewer-supplied references, then **the PI writes the revised prose and the response letter (G16)**; G14 freeze (14 days) + G1 on the new text + G12 after upload | 2026-10-06 (everything but the prose) | `papers/P8_linguistic_fossils/REVISION_OUTLINE_20261006.md` (what each section must contain) · `REVISION_WORKPLAN.md` (decisions, point-by-point facts, corrected numbers; §10 = new on 10-06) · `VENUE.md` · `REVISION_OL_20261005.md` (ledger) · OL portal link is in the decision email (not recorded here: public repo) |

| 6 | **P2 round-2 revision → resubmit at JCAA by 16 Nov 2026** (decision 2026-10-19: revisions requested). First: the PI's decisions D1–D7 (by 23 Oct) and the **waiver email** (G15, D4: draft ready). Then E235 run, Figures 1 and 5, references (C076), the PI's prose, freeze, G1 + adversarial read, letter traced against the final file, upload + G12 | 2026-10-20 (triage, design, email draft) | `papers/P2_settlement_model/REVISION_JCAA_R2_20261019.md` · `docs/correspondence/EMAIL_JCAA_WAIVER_STATUS_DRAFT_20261020.md` · `experiments/E235_elevation_vs_volcanic_control/DESIGN.md` |

🅿 Still parked (PI decision 2026-08-11, unchanged): Verberne reply · P7 preprint notice · Lamqaddam reply.

**Scorecard: 0 final acceptances · 1 conditional (P8, 2026-10-05; R1 submitted 2026-10-15) · 7 rejections · 1 in revision (P2: revisions requested 2026-10-19, due 16 Nov) · 1 withdrawn (P17, 2026-10-02) · 227 experiments** (227 folder lokal, E001–E235; 8 nomor tak pernah dibuat; +E225 13 Agt, +E053/E203 kembali dari repo pendamping 1 Okt — rekonsiliasi 2026-10-01; +E226 T7 2 Okt; +E227/E228 audit dan analisis revisi P8, 5 Okt; +E229/E230 tabel revisi dan uji ulang ejaan/panjang P8, 6 Okt; +E231 hitungan deskriptif arti "berkode" di ABVD untuk P8, 6 Okt; +E232 desain pra-registrasi I-150, 15 Okt, belum dijalankan; +E233 validasi P23 terhadap terjemahan penyunting DHARMA, 19 Okt; +E234 desain pra-registrasi jalur sumber E216, 19 Okt, belum dijalankan; +E235 desain pra-registrasi kontrol elevasi vs vulkanik untuk revisi P2, 20 Okt, belum dijalankan).
**2026-10-02: all four ledger items done** (JCAA reply, VEGAN email, P17 withdrawal, Zenodo notice). Ledger is
empty of PI actions except waiting. **P2** in round-2 review since 31 Aug · **P8** (OL) — **decision 2026-10-05: accepted pending revisions** (ledger item 5).

---

## 2. ⏰ Clock

| Deadline | Item |
|---|---|
| 23 Oct 2026 | Follow-ups if silent: Nastiti (P23 reader + Kusen/Resiyani lists) and BRIN VEGAN (microbotany); Claude drafts on 22 Oct, PI sends (K8) |
| **23 Oct 2026** | **P2 revision: PI decisions D1–D7 + waiver email (D4)** — `papers/P2_settlement_model/REVISION_JCAA_R2_20261019.md` §5 |
| 2 Nov / 9 Nov 2026 | P2 freeze: strict G14 (2 Nov) or D2 option (b) (9 Nov); if it slips past **6 Nov**, ask JCAA for an extension that day |
| 6 Nov 2026 | Fallbacks if still silent: P23 → Griffiths/DHARMA or a UGM epigrapher; VEGAN → Castillo, or park T5 (K8) |
| **16 Nov 2026** | **P2/JCAA round-2 revision due** (4 weeks from the 19 Oct decision). *(The 9 Nov status query (K8) is void: the decision came.)* |
| **15 Nov 2026** | **K6 — program direction ("taphonomy of records"; pending L1 amendments) — PI.** With it: whether E225 phase 2 runs (pilot LANJUT 10-15) |
| 15 Jan 2027 | DHQ, realistic deadline for P23 |
| — | P8 R1 (submitted 2026-10-15; receipt 17 Oct): nothing due until the journal writes |
| Dec 2026 | Edinburgh PhD window — **subject to the PI's current PhD plan** (PI-owned; may no longer apply) |

---

## 3. Line status

| # | Line | Temp | Next action | Owner |
|---|---|---|---|---|
| **01** | [spatial](../lines/01_spatial/STATE.md) | 🟠 REVISION | **P2: revisions requested 2026-10-19, due 16 Nov** — E235 done (overlap ceiling 1,000 m); figures, v0.3 (mechanical), VENUE, writing pack done; waiver email sent; **next = the PI's prose (freeze ≤ 9 Nov)**; references C076 · P17 withdrawn 10-02 · **P11 PARKED 10-15** | **PI** + Claude |
| **02** | [taphonomy](../lines/02_taphonomy/STATE.md) | ⚠ WARM | C024–C030 verified 10-01, their OPEN-fixes still owed (ledger); **hold P1 (JASREP)** until its calibration anchors are audited; West Java skeleton arms need rebuilding | Claude |
| **03** | [paleoenv](../lines/03_paleoenv/STATE.md) | ⏳ WAITING | VEGAN email sent 10-02; follow up ≈ 23 Oct if silent. T5 needs pre-400 paleosols | PI |
| **04** | [language_text](../lines/04_language_text/STATE.md) | 🟢 P8 R1 IN REVIEW | **P8 R1 submitted 2026-10-15** (second round, same reviewers; nothing due) · **P5 PARKED 10-15** (C018 closed; C072) · **E232** (I-150) designed, not run — waits for the PI to read `DESIGN.md` | PI |
| **05** | [archival_nlp](../lines/05_archival_nlp/STATE.md) | 🅿 E226 REVISIT · ✅ E225 pilot | **E225 pilot run 10-15: LANJUT, confirmed by hand** (12 valid Indonesian radiometric pre-400 records per 120 random articles; automatic count 314 is an upper bound — 17/30 hold). Phase 2 needs a new pre-registration → **decide with K6** · **E226 (T7) parked 10-02**: frame of 266 villages built; I1 identifications 11/170 → unpark needs Kusen 1990/91 / Resiyani 2010 lists. Next when warm: T2 or E211 | Claude |
| **06** | [thesis](../lines/06_thesis/STATE.md) | 🛑 FALLOW | Objective answer written (subtract-only). L1 amendments await PI (§4) | PI |
| **07** | [career](../lines/07_career/STATE.md) | ✅ LEDGER CLEARED 10-02 | waiting on journals (P2, P8) | — |
| — | population-evidence channel (E053, E203) | — | **re-merged 2026-10-01** — back in `experiments/` (E053 → 02+06, E203 → 06); claims downgraded in the objective-answer note §3; see `COMPANION_REPOS.md` | — |

---

## 4. Decisions waiting on the PI

**✅ DECIDED 2026-10-15 — decision agenda K1–K9 (PI: "setuju default, K6 nanti saya putuskan"); K6 OPEN until 2026-11-15:** `docs/research_notes/ORBIT_REVIEW_20261015.md` §5.
K1 P5 → PARKED (YA) · K2 P11 → park (YA) · K3 P1 held until C024 (YA) · K4 a paper in re-review counts in G13 (YA) ·
K5 G16 back for P23 — the PI writes the prose (YA) · **K6 program direction "taphonomy of records" + pending L1 amendments —
no default, PI by 2026-11-15** · K7 E232 = I-150 design only (YA) · K8 follow-ups Nastiti/VEGAN 23 Oct, fallback 6 Nov,
P2/JCAA 9 Nov (YA) · K9 stop list (YA). Answers to be written there and here.

**NEW 2026-10-02 — publication strategy after 7 rejections:** `docs/research_notes/STRATEGI_PUBLIKASI_20261002.md`
(diagnosis: 10 submissions in 35 days, 6/7 rejections at the desk). **DECIDED same day (PI delegated):** SIG gates
G13–G16 binding (WIP ≤2 in review / 1 drafting; rest ≥14 days; venue first + zero cost of any kind; PI's own prose);
G10 human domain reader required for archaeology venues *(since 2026-10-19: sought, never waited for — route S = sources + AI domain read + disclosure after 14 days)*; domain co-author by default there (invited, not waited for); **Naskah A (P23, sīma
village toponyms) first**. G13 was full (P2, P8 in review) → no new submission until a decision arrives. **2026-10-05:** P8's
decision arrived (conditional). Whether a manuscript in revision still occupies a G13 slot: **✅ yes, it counts (K4, 2026-10-15; SIG G13)**.

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
- ✅ **P11 → (c) parked 2026-10-15 (K2; `papers/P11_volcanic_informedness/PARKED.md`).** Was: NO-GO. Choose: (a) reframe as a Penanggungan-centred paper (strong pattern: 69 candi, 62% west — but 14 of the 43 "western" rows are Trowulan temples 26–56 km away on the Brantas plain, so partly "Trowulan lies west of Penanggungan";
  p=1.4×10⁻²⁰), (b) rework the island-wide version on de-duplicated data and accept a weaker claim, or (c) park.
- **P1:** hold v5.0 (→ Archaeological Research in Asia) until its calibration anchors pass a WS-E-style audit — **✅ confirmed 2026-10-15 (K3)**.
- **P1 Zenodo preprint (public record) — mandatory, ~15 min:** `10.5281/zenodo.19081502` (2026-03-18) states the
  inverted E069 claim (PDF lines 400–406) and a "Java-wide taphonomic baseline" from assumed burial-start dates
  (abstract). Notice text, verbatim quotes: `docs/correspondence/ZENODO_P1_CORRECTION_NOTE_DRAFT_20261001.md`.

**Carried over:** D5 L1 Java/Nusantara disaggregation · D7 audit rule (submission trigger or 14 days) ·
DJKI HKI filing (4 docs ready) · dashboard model regeneration on the 30-volcano inventory.

---

## 5. Portfolio

| Paper | Line | Status |
|---|---|---|
| **P2** Settlement model | 01 | 🟠 **REVISIONS REQUESTED 2026-10-19 (JCAA #280, round 2), due 16 Nov.** R1 Resubmit for Review (clarity), R2 Revisions Required (elevation vs volcanic setting). Triage: `papers/P2_settlement_model/REVISION_JCAA_R2_20261019.md`. APC waiver still undecided (G15). Claim set: `review_package_20260727/10_SET_KLAIM_TERKOREKSI.md`. |
| **P17** Two Javas | 01 | ⛔ **WITHDRAWN 2026-10-02** (ArchCalc #365; never left the submission stage). Central result = geocoding artefact (C032); second result (929 CE shift) rides on the same points. Record: `docs/correspondence/EMAIL_ARCHCALC_P17_INTEGRITY_NOTICE_DRAFT_20261001.md`. A methods paper on the artefact is possible later (not planned). |
| **P8** Linguistic fossils | 04 | 🟢 **conditionally accepted 2026-10-05; R1 submitted 2026-10-15** — *Oceanic Linguistics* (Scimago Q2) OL-03-2026-11R1; second review round by the same reviewers. Record: `papers/P8_linguistic_fossils/revision_v0.2/PORTAL_STATE_20261015.md`. |
| **P23** sīma village toponyms (Naskah A) | 05 | 📝 PLANNING (2026-10-02) — critical case study (toponym resolution is the bottleneck) from E226's frame → **DHQ "Case Study"** (no fees; JDMDH backup). Needs human validation of a sample (epigrapher asked: Nastiti, 13:55). Drafting allowed; submission blocked by G13 until P2 or P8 decides. `papers/P23_sima_village_toponyms/README.md` |
| **P11** Volcanic informedness | 01 | 🅿 **PARKED 2026-10-15** (K2) after 🔴 NO-GO 2026-10-01 (western-flank claim = Penanggungan pattern; duplicate candi; no abstracts). Unpark paths: `papers/P11_volcanic_informedness/PARKED.md`. |
| **P1** Taphonomic framework | 02 | rejected 2×. Current manuscript **v5.0 → *Archaeological Research in Asia*** (per `CANONICAL.md`; the E069 passage was already deleted there; superseded JASREP v4.0 corrected anyway). v5.0 still reports the 4.4 mm/yr "baseline" → **C024 audit before submission**. Public preprint needs the correction notice (§4). |
| **P5** Volcanic ritual clock | 04 | 🅿 **PARKED 2026-10-15** (K1; C018 closed). Rejected (BKI); the permutation test p = 0.008 looks near-tautological (C072). `papers/P5_volcanic_ritual_clock/PARKED.md`. |
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

**Stop list (K9, PI 2026-10-15):** no new computational experiment for the volcanic-burial thesis without partner data
(T4/T5/T7 need data from outside) · P20–P22 proposals and AutoResearch stay in the drawer until P23 and Naskah B ·
the next system-critique round only on a journal decision (P2/P8), a new submission, or 14 days (D7).

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

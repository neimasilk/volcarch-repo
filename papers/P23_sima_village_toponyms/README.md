# P23 — Village toponyms in 8th–10th c. Old Javanese sīma charters (Naskah A)

**Status:** PLANNING (opened 2026-10-02; strategy decision 5 — `docs/research_notes/STRATEGI_PUBLIKASI_20261002.md`).
**Line:** 05_archival_nlp (primary), 04_language_text. **Author:** Mukhlis Amien (first); domain co-author invited
(see Reader). **Source experiment:** E226 (frame step only; T7 matching is parked and is **not** this paper's claim).

## The paper in one sentence (re-scoped 2026-10-02 after `VENUE.md`)

**A critical case study:** machine reading can recover the villages named in 8th–10th c. Old Javanese charters
cheaply and checkably, but *locating* them cannot be automated — toponym resolution, not text extraction, is
what limits the use of charters as a settlement record (0 of 266 names match a modern desa exactly; 11 of 170
have a published identification). The dataset accompanies the argument; it is not the argument. **Not** a
claim about settlement history or volcanism.

## Venue (G15) — decided 2026-10-02, final lock after human validation

- **Target: DHQ, article type "Case Study"** (≈6–7k words; technical evaluation in an appendix; AI use
  acknowledged in the body, as DHQ requires). DHQ is zero-cost ("does not charge any fees of any kind"), ESCI
  (JIF 2024 0.8); Scopus listing claimed on its About page but not independently verified. **Fit caveat (VENUE.md):** DHQ's FAQ excludes "routine analyses of data
  sets or text corpora" and "standalone data sets" → the case-study framing above is required, not optional.
  Acceptance is low and falling (18% 2023 → 15% 2024 → ~7% 2025, pending). 0 of 210 DHQ items 2023–26 treat
  Southeast Asian or Indic epigraphy (novelty, but also no ready reviewers). Next realistic deadline 2027-01-15.
- **Before drafting the full text:** the human validation result (`validation/PROTOCOL.md`); then a short
  pre-submission inquiry to dhqinfo@digitalhumanities.org (G15).
- **Backup: JDMDH** — diamond (no author fees), welcomes short dataset/tool articles (genre fit), but not
  Scopus/WoS and slow (median ≈ 313 days to acceptance). Choose it if the DHQ inquiry is negative or if the
  validation turns the paper into a pure method lesson.
- **Excluded:** JOHD (APC £1,070, waiver not guaranteed); ACL workshops (registration fee); jTEI (fees unverified).
- Full survey, genre template and section skeleton: `VENUE.md`.

## What exists already (from E226)

- 595 village-noun occurrences (wanua/banua/wanwa/banwa/thāni) in 73 of 111 dated inscriptions, edition text
  only; the *b*-spelling lesson (440 → 595).
- Two-pass coding by two LLM agents run blind to each other (κ 0.90 Y vs not-Y; 3-category κ 0.87; name
  Jaccard 0.88), 78 adjudicated rows with written reasons, 16 orthographic merges → 266 names (170 from
  Kedu/Prambanan inscriptions).
- Identification audit: exact modern-desa match (Kepmendagri 2025, 3,802 desa) = 0; published identifications
  = 11/170 (3 at desa level).
- Attachment sent to the reader: `data/daftar_desa_prasasti_abad8-10_v20261002.xlsx`.

## What the paper still needs (in order)

1. **Human validation (core, not optional):** an epigrapher codes a random sample (e.g. 100 occurrences)
   blind → human–LLM agreement and error types. Without it the two-LLM κ says little about correctness.
2. Reader/co-author answer (G10). **Asked 2026-10-02 13:55:** Dr Titi Surti Nastiti (BRIN, epigraphy;
   tsnastiti@yahoo.com, address from her *Kalpataru* article). Text:
   `docs/correspondence/EMAIL_NASTITI_P23_READER_SENT_20261002.txt`. Reserve: Arlo Griffiths (DHARMA).
   Follow up ≈ 23 Oct if silent.
3. Kusen 1990/91 / Resiyani 2010 toponym lists (asked in the same email) → a better identification section.
4. Data deposit (Zenodo, CC BY 4.0, crediting DHARMA) **after** validation, not before.
5. Prose by the PI (G16), following `VENUE.md`; Claude supplies tables, figures, numbers, and critique.

## Gates (SIG) — current state

| Gate | State |
|---|---|
| G13 WIP | FULL now (P2, P8 in review) → P23 may be drafted, not submitted, until a decision arrives |
| G14 rest | not started (no frozen draft) |
| G15 venue | DHQ (Case Study framing) chosen, fees verified, `VENUE.md` done; pre-submission inquiry after validation |
| G10 reader | asked (Nastiti) |
| G16 voice | PI writes |

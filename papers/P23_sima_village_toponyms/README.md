# P23 — Village toponyms in 8th–10th c. Old Javanese sīma charters (Naskah A)

**Status:** PLANNING (opened 2026-10-02; strategy decision 5 — `docs/research_notes/STRATEGI_PUBLIKASI_20261002.md`).
**Line:** 05_archival_nlp (primary), 04_language_text. **Author:** Mukhlis Amien (first); domain co-author invited
(see Reader). **Source experiment:** E226 (frame step only; T7 matching is parked and is **not** this paper's claim).

## The paper in one sentence (claim size fixed now)

A reproducible pipeline and a checked list of the villages named in 8th–10th c. CE Old Javanese charters
(DHARMA corpus), with an honest account of what it takes to locate them — **a resource/method paper, not a
claim about settlement history or volcanism.**

## Venue (G15)

- **Target: Digital Humanities Quarterly (DHQ).** "DHQ does not charge any fees of any kind" (about page, checked
  2026-10-02); Scopus-listed, WoS ESCI (JIF 2024 0.8); review typically 2–4 months; CC BY-ND default (CC BY
  available).
- **Backup: Journal of Data Mining & Digital Humanities (JDMDH)** — diamond OA (episciences), DOAJ/DBLP, not
  Scopus.
- **Excluded:** JOHD (APC with waiver fund — not guaranteed zero); ACL-family workshops (registration fee).
- Genre template from ≥5 recent DHQ articles: `VENUE.md` (being compiled).

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
| G15 venue | DHQ chosen, fees verified; `VENUE.md` pending |
| G10 reader | asked (Nastiti) |
| G16 voice | PI writes |

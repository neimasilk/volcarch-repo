# E226 — T7 detection calibration: pre-registration

**Frozen:** 2026-10-02 ±09:40 WIB, committed before any settlement register exists and before any matching.
**Source design:** `docs/drafts/T7_DETECTION_CALIBRATION_DESIGN_DRAFT_20261001.md` (adopted here with the changes
listed in §6). **Lines:** 05_archival_nlp (primary), 06_thesis.
**Authorisation:** the 3-month direction in `docs/WORKSTATE.md` §4 is default-YES; the PI asked on 2026-10-02 to
continue the queue. **N_ref and m are NOT fixed here — they are the PI's (with a domain expert) and must be
written into §5 before `scripts/04_match.py` is run.** Matching is blocked until then.

## 1. Question

Is "no recorded pre-400 CE settlement in interior volcanic Central Java" informative about population (H6), or
uninformative? Measured by the detection rate *d*: the share of villages known to have existed (named in
8th–10th c. *sīma* charters) that have a recorded settlement deposit of that period.

*d* measures combined detectability (burial H1a + survey effort H2 + light architecture H3). It cannot separate
them. A low *d* makes the H6 inference from absence unsafe; it does **not** prove H3 or H1b. The 8th–10th c.
window sits in the same burial zone.

## 2. Village frame V_r (step 1 — may start now)

- **Corpus:** local DHARMA snapshot `experiments/E023_ritual_screening/data/dharma/xml/` (268 TEI editions).
  Text = `<div type="edition">` only (sic/orig/del/surplus/note/rdg/fw/label dropped; `lb break="no"` joined).
- **Date window:** 701–1000 CE. Date from the XML title (ISO, Śaka+78, CE, century); else IDENK xlsx by
  designation name; else a manual alias (listed in the script). Range-dated items (e.g. Harinjing 804–927,
  "10th c.") are kept and flagged. **Sensitivity run:** title-dated only.
- **Exclusions:** the 50 Borobudur hidden-base captions.
- **Region** (of the *inscription*, by IDENK province/kabupaten, else E082 label): Kedu = Magelang, Temanggung,
  Wonosobo, Purworejo; Prambanan–Mataram = Sleman, Bantul, Yogyakarta, Klaten, Gunung Kidul, Kulon Progo.
  The frame for the primary estimate is villages named in inscriptions from these two regions. Inscriptions
  from other regions are coded too and reported separately.
- **Village nouns:** vanua-family (vanua, vanu, vanuan, vanuana…), vanva-family, thāni. *karaman* counted
  separately, not a village noun. *deśa*, *kuvu*, *grāma* excluded. (Witness villages named only through
  *rāma i X* / *tuhan i X* without a village noun are **outside** this frame; reported as a threat, §7.)
- **Candidates:** every village-noun occurrence with a ±12-token KWIC and a regex-proposed name (next token
  after an optional locative particle i/ri/riṅ/iṅ/ni).
- **Two coders**, independent, same codebook (`frame/CODEBOOK.md`), blind to each other. Per occurrence:
  is a specific village named (Y/N/?), and its canonical name. Agreement: Cohen's κ on Y/N over occurrences,
  plus name-level Jaccard. Disagreements adjudicated by a third reader; every adjudication logged.
- **Output:** `results/t7_village_frame.csv` (name, inscriptions, dates, region, coder agreement).

## 3. Identification tiers (step 2 — after the frame is frozen)

- **I1:** modern location published by an epigrapher (Boechari, Damais, Sarkar, Wisseman Christie, Nakada,
  DHARMA/IDENK metadata), with citation.
- **I2:** a unique modern *desa* of the same name within the same kabupaten as the inscription findspot.
- **I3:** ambiguous or not located.
- *d* is estimated on G = I1+I2. Bounds: S/|V| ≤ *d* ≤ (S + |V∖G|)/|V|.

## 4. Settlement register (step 3 — built BLIND to the village list)

- **Recorded settlement:** an open-air habitation deposit (floors, post-holes, hearths, domestic assemblage,
  house platforms) dated to the 8th–10th c. by radiocarbon or diagnostic ceramics/finds. **Excluded:** temples
  and temple enclosures alone, inscriptions, tombs, single-object find-spots, hoards.
- **Area:** Kedu + Prambanan–Mataram kabupaten (§2), plus a buffer of the adjacent kabupaten (reported).
- **Sources:** Balai Arkeologi Yogyakarta (*Berkala Arkeologi*, *Laporan Penelitian*), BPK Wilayah X,
  SRN Cagar Budaya, peer-reviewed literature, theses. Each entry needs a traceable citation (DOI/URL/page).
- **Blindness:** the register is compiled by an agent that is not shown the village frame and is instructed
  not to open `frame/`, `results/`, or the DHARMA corpus. The instruction text is kept in `register/PROMPT.md`.
- **Completeness check (pass/fail):** Liyangan must appear, plus any 8th–10th c. settlement a domain expert
  names. If a known settlement is missing, the run is **NOT INFORMATIVE** until the register is repaired.
- **Frozen** (hash recorded in README) before matching.

## 5. Matching and decision (step 4 — BLOCKED until the PI fills this section)

- Radius ρ = 1 km from the modern desa centroid (sensitivity 0.5 and 2 km).
- Chance-match null: same number of random land points per kabupaten → *d*₀. Report *d* − *d*₀.
- *d̂* = S/|G|, Jeffreys 95% interval. N₉₅ from the beta-binomial predictive with a = S+0.5, b = n−S+0.5.
- **N_ref = ____ (PI)** — default proposal: |V_r| in Kedu+Prambanan.
- **m ∈ {____} (PI)** — default proposal {1, 0.5, 0.25}.

| Condition | Reading |
|---|---|
| N₉₅(m=min) ≥ N_ref | Zero pre-400 settlements is **uninformative**; "nothing found" ≠ "no people". |
| N₉₅(m=1) < N_ref | A zero is inconsistent with an 8th–10th c.-sized population. **Supports H6**, conditional on comparable survey effort. |
| otherwise | Ambiguous; report numbers. |

- **Falsifier (H3 in-Java control):** if S/|G| − *d*₀ ≥ 0.25 with |G| ≥ 30 in Kedu/Prambanan, the 8th–10th c.
  record is not village-blind there; the "villages invisible even when we know they existed" argument is dead.

## 6. Changes from the 2026-10-01 draft

1. Brantas dropped from the primary frame (≈5 names in the local corpus); reported descriptively only.
2. Frame region = inscription findspot region (village locations are unknown until step 2) — stated as a threat.
3. *grāma* explicitly excluded (5 tokens; Sanskrit generic).

## 7. Threats (carry into the write-up)

1. *Sīma* villages are taxable, salient, near the royal core and near temples (high survey effort) → *d*
   biased **up** for ordinary villages; N₉₅ anti-conservative.
2. Older sites more buried/eroded → *d*_pre ≤ *d* (hence m).
3. Unmatched toponym = identification failure, not non-detection (tiers + bounds).
4. Villages may have moved (radius sensitivity).
5. Corpus coverage: 268 encoded editions vs ~1,271 IDENK records.
6. Heritage registers are temple-centred → S may be ≈0 by construction; literature counts are required.
7. Frame uses only village-noun contexts; *rāma i X* witness villages are omitted (frame is a subset).
8. The frame is restricted by inscription findspot, not village location.

## 8. Amendments (logged before the step they affect)

- **A1 (2026-10-02 ±10:00, before coding):** b-initial spellings (*banuA*, *banva*…) added to the vanua/vanva
  family after the KWIC sample showed "Anak banuA I hinapit" etc. Orthographic variant, not a rule change.

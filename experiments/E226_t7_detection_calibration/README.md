# E226 — T7 detection calibration (villages named in 8th–10th c. inscriptions vs recorded settlements)

**Status:** REVISIT — paused 2026-10-02 at step 2 ("digodok dulu", PI). Unpark condition below. **Lines:** 05_archival_nlp (primary), 06_thesis.
**Pre-registration:** `DESIGN.md` (frozen 2026-10-02 before any register or matching; amendments in §8).
**Matching is BLOCKED** until the PI fills N_ref and m in `DESIGN.md` §5.

## Hypothesis

If even villages known to have existed (named in 8th–10th c. *sīma* charters) are rarely matched by a
recorded settlement deposit of that period, then the absence of recorded pre-400 CE settlements in interior
volcanic Central Java is uninformative about population (against reading it as H6, low density). If they
are matched often (S/|G| − d₀ ≥ 0.25, |G| ≥ 30), the "villages are invisible" argument fails.

## Method (steps; see DESIGN.md)

| Step | What | Script / file | State |
|---|---|---|---|
| 1a | Village-noun occurrences with KWIC, 701–1000 CE, edition text | `scripts/01_build_candidates.py` → `frame/candidates_kwic.csv`, `frame/inscriptions_701_1000.csv` | ✅ 595 occurrences in 73 of 111 inscriptions |
| 1b | Two independent coders + adjudication → frame V_r | `frame/CODEBOOK.md`, `frame/coder_A/B.csv`, `scripts/02_reconcile.py` → `results/t7_village_frame.csv` | ✅ κ(Y vs not-Y)=0.90 (96.6%), 78 rows adjudicated, 16 spelling merges → **266 villages, 170 from Kedu/Prambanan inscriptions** (72 title-dated only) |
| 2a | Modern desa gazetteer (Kepmendagri 2025) | `scripts/03_gazetteer.py` → `data/processed/gazetteer/desa_kedu_mataram_buffer_2025.csv` | ✅ 3,802 desa |
| 2b | Identification tiers I1/I2/I3 | `scripts/03b_identify_I2.py` → `results/t7_identification_I2.csv`; I1 → `frame/identifications_I1.csv` | I2 as pre-registered = **0** (see notes); I1 = **11 of 170 primary names** (6 stated, 3 at desa level, all Atmodjo 1988 / Magelang) → G far below the pre-registered |G| ≥ 30 |
| 3 | Settlement register, built blind to the frame | `register/PROMPT.md` → `register/t7_settlement_register_candidates.csv`, `register/REGISTER_NOTES.md` | ✅ **frozen 2026-10-02**, sha256 (LF-normalised) `e515fe3e…728e51`: 22 rows, **1 YES (Liyangan)**, 9 UNCLEAR, 12 NO |
| 4 | Matching, d, d₀, N₉₅, decision | `scripts/04_match.py` (exits BLOCKED until DESIGN §5 is filled; checks register sha256; N₉₅ reproduces the desk-scan table) | **blocked on PI (N_ref, m)** + step 2c (coordinates of identified desa) |

## Data used

- DHARMA TEI editions, local snapshot `experiments/E023_ritual_screening/data/dharma/xml/` (268 files) and the
  IDENK metadata xlsx (dating, findspot kabupaten).
- E082 canonical30 geocode labels (region fallback only; known placeholders, C032).
- Kepmendagri 300.2.2-2430/2025 desa list via cahyadsn/wilayah (`data/sources.md`).

## Notes so far

- **Desk-scan count corrected:** the 2026-10-01 scan counted 440 village-noun tokens; it missed the *b*-spelling
  (*banuA*, *banva*), which adds 155 → **595 occurrences** (2 are *baṅun* "build", left for the coders to reject).
  Inscriptions with ≥1 village noun: 69 → 73. (Amendment A1.)
- Region is the inscription's findspot region, not the villages' location (threat 8).
- Bug fixed 2026-10-02 before coding results: region matched Kedu/Prambanan names against the *desa* string
  too, so Ayam Teas III (Wonogiri, desa "Pulutan Kulon") passed as Kulon Progo. Now kabupaten only, as
  DESIGN §2 specifies. Occurrence IDs and coder input unchanged.

- **I2 yields nothing (2026-10-02).** Exact modern-key match within the findspot kabupaten: 0 of 266. Across
  all 16 kabupaten only 8 frame names have an identically named desa (e.g. *taji* → Taji, Klaten;
  *kiriṅan* → Kiringan). 110 villages come from inscriptions with no findspot kabupaten in IDENK. Old
  Javanese toponyms survive in transformed shapes (Mantyasih → Meteseh), and dusun names are not in the
  gazetteer. *d* will therefore rest on **I1** (published identifications). The I2 rule was not loosened.

- **Register (step 3) is thin, as threat 6 predicted.** Within reach of web search and four BRIN journals,
  only Liyangan meets the definition. The 9 UNCLEAR rows are: news-only finds dated by Tang ceramics (Kropakan,
  Balong Bayen, Brojonalan, Nepen); older excavations dated relatively (Wonoboyo habitation area, Borobudur SW
  2012, Ratu Boko); a coastal dune site (Gunung Wingko); and surface scatters (Karangrejo/Sukorame).
  Grey literature (Balai Arkeologi reports, theses) was not reachable. **Completeness check:** Liyangan is
  present, so the minimum passes; no domain expert has been asked to add missing sites, so the register is
  provisional. **Pre-matching bound (no matching done):** S ≤ 1 (YES only) or ≤ 10 (YES + UNCLEAR), whatever
  the identifications turn out to be.

## Result / Conclusion (2026-10-02)

**INCONCLUSIVE at step 2: *d* cannot be estimated from open sources.** The obstacle is identifying the toponyms,
not only the register.
- Frame: 170 villages from Kedu/Prambanan inscriptions (two coders, κ 0.90).
- I2 (exact desa name): 0. I1 (published identifications): 11 names, 3 at desa level → |G| ≤ 11, below the
  pre-registered |G| ≥ 30. With n ≈ 11, N₉₅ would be large whatever S is, so the "uninformative" reading would
  follow from the small n, not from the data; running step 4 now would manufacture a result.
- Register: 1 qualifying settlement (Liyangan) + 9 unclear, within web reach.
- **Matching was not run.** N_ref and m remain the PI's.

**Unpark condition (any one):** (a) the toponym lists of **Kusen 1990/91** ("Identifikasi Toponim dalam Prasasti
Jawa Kuno Abad IX–X dari Prambanan dan Sekitarnya…", UGM research report) or **Resiyani 2010** (UGM thesis,
Temanggung toponyms) are obtained — via Titi Surti Nastiti, Arlo Griffiths (DHARMA) or UGM Archaeology;
(b) a domain expert agrees to code I1 for ≥30 frame villages; (c) Balai Arkeologi/BRIN settlement reports for
Kedu/Prambanan become accessible (register side). Then: geocode G (step 2c), PI fixes N_ref/m, run
`scripts/04_match.py`.

**What is reusable now:** the coded frame (`results/t7_village_frame.csv`, `frame/final_occurrences.csv`) is a
clean, two-coder list of 266 named villages in 8th–10th c. charters — usable for line 04/05 work independent of T7.

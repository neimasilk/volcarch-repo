# Line 04 — LANGUAGE & TEXT

> **Question:** What do language and texts preserve of a substrate that left no excavated remains?

**Recommended model:** Opus for framing and for anything touching a claim; Sonnet is adequate for
mechanical corpus/wordlist work. **Effort:** high for claims, medium for pipelines.

---

## Scope

Historical linguistics, epigraphy, and computational textual archaeology: substrate detection in
Javanese/Balinese/Tengger/Osing, Sanskrit ratio over time, Old Javanese corpora, inscription
statistics, genre taphonomy, ethnobotanical and ritual vocabulary. Reviewer communities:
**historical linguistics** (*Oceanic Linguistics*), **digital humanities** (DHQ, *Wacana*),
**anthropology/folklore** (*Asian Ethnology*).

**Out of scope:** Dutch colonial archives and VOC records (→ [05_archival_nlp](../05_archival_nlp/));
whether the sum of channels proves a civilization (→ [06_thesis](../06_thesis/)).

---

## Two facts that constrain everything here

**1. The DHARMA corpus is closed and it was a monoculture.**
Mining of the 268 DHARMA inscriptions was **officially closed 2026-04-09**. A cluster of experiments
depends on that single corpus — count them via `docs/EXPERIMENT_INDEX.md` before quoting a number. Any new claim resting only on DHARMA inherits that dependency — say so
explicitly. The corpora that break the monoculture: `E091` (OV), `E092`, `E098`, `E141`–`E143`
(Delpher; those now belong to [05_archival_nlp](../05_archival_nlp/)).

**2. This line produced the project's second real refutation, and it cost a paper.**
`E090` v7 label-shuffle killed P16's convergence pillar: **0 of 8 groups** survived, and the
cross-model G9 review returned **REJECT**. SIG verdict was NO-GO, and the PI chose to **park** P16
rather than reframe it. That was the right call and it should not be quietly reversed.

---

## Papers

| Paper | Folder | Status |
|---|---|---|
| **P8** Linguistic fossils | `papers/P8_linguistic_fossils/` | ⏳ **Under review** — *Oceanic Linguistics* (Q1), MS# **OL-03-2026-11**, submitted 2026-03-11. arXiv preprint live: **arXiv:2604.00023** (cs.CL, CC BY 4.0). Co-authors: Amien + Go Frendi. **WAIT — do not touch the manuscript.** |
| **P5** Volcanic ritual clock | `papers/P5_volcanic_ritual_clock/` | Rejected by *BKI*. Retarget **Asian Ethnology** (Nanzan U, **zero APC**, Scopus Q2) with a humanities reframe: *indigenous knowledge resilience*. Strategy doc ready; **full rewrite was scheduled for ~June 2026 and is overdue.** |
| **P9** Peripheral conservatism | `papers/P9_peripheral_conservatism/` | Rejected by *JSEAS*. HOLD until P2/P8 resolve → then DHQ. Compile chain differs: **`pdflatex → biber → pdflatex ×2`** (biblatex). |
| **P19** Before the inscriptions | `papers/P19_before_the_inscriptions/` | Proposal stage. |
| **P16** Computational textual archaeology | `papers/P16_computational_textual_archaeology/` | 🅿 **PARKED 2026-06-10.** Manuscript frozen (`submission_wacana_v1.0.tex`). Unpark conditions in `PARKED.md`: either a non-circular convergence design (unsupervised clustering) that passes, or a downgrade-reframe to a distributional-attestation paper. |
| **P14** Pararaton collapse | `papers/P14_pararaton_collapse/` | DISCONTINUED. |

---

## Experiments

**Substrate detection:** `E022` linguistic subtraction · `E027` ML substrate detection
(**UPGRADED** after `E107` resolved ADV-5 → Mon-Khmer substrate) · `E028` consensus ·
`E029` clustering · `E051` toponymic · `E067` volcanic toponyms · `E130` interpretability (438
substrate words) · `E165`/`E181` ghost vocabulary & dictionary · `E186` Tengger cross-ref
**Chronology & adoption:** `E030`, `E033` Sanskrit temporal curve, `E037` dating ML, `E061` script
simplification, `E111` script diffusion, `E131` writing adoption (400 CE is **not** an outlier),
`E134` inscription chronology (genre explosion 1→396 words)
**Inscription statistics:** `E082` georeferencing, `E113` sophistication, `E146` density,
`E147` length×distance (ρ=0.587), `E160` semantic deep, `E169` inscription desert
**Textual archaeology / NLP:** `E058` + `E208` kakawin, `E074` DHARMA deep, `E088`–`E090`
(**E090 v7 = the refutation**), `E094` semantic search, `E095` XLM-R, `E096` diachronic BERTopic,
`E150` Babad substrate, `E205` wayang indigenous layer
**Comparative / ethnobotanical:** `E034` Panji–Malagasy, `E035` prasasti botanical, `E036`
hanacaraka, `E038` volcanic vocabulary, `E039` VCS cross-cultural, `E040` bamboo, `E041`/`E042`
IPA/syllable validation, `E043` krama alus cognacy, `E044` Malagasy burial botany, `E049` maritime
vocabulary, `E050` Canarium distribution, `E054` pan-Austronesian cognacy, `E063` domain
conservation, `E102` vocabulary preservation, `E112` vocabulary archaeology, `E198` sago/rice
etymology
**Ritual & genre:** `E023` ritual screening, `E026` Pararaton correlation, `E032` pranata mangsa,
`E057` genre taphonomy, `E204` bronze drums

**62 experiments** are assigned to this line (62 primary). Authoritative list:
`docs/EXPERIMENT_INDEX.md` §"By Line of Inquiry" — regenerate with
`python tools/scan_experiments.py`.

### Data notes
ABVD CLDF: `experiments/E022_linguistic_subtraction/data/abvd/cldf/`
IDs — **Tengger 1533** · Balinese 1 · Javanese 20 · PMP 269 · PAn 280 · Old Javanese 290
LingPy 2.6.13.

---

## Line rules

1. **Name the corpus and its size in every claim.** "438 substrate words" means nothing without
   which corpus and how many inscriptions.
2. **A shuffle/permutation control is mandatory** for any convergence or clustering claim. E090 v7 is
   why P16 is parked; do not produce a claim that could not survive the same test.
3. **Do not count channels.** F9 applies with force here — the linguistic channels share corpora and
   are correlated.
4. **P8 is live at a journal.** No edits to its manuscript, no new preprint versions, until the
   decision lands.
5. Note the compile chains: P9 uses **biber**; P1/P2/P11 use **bibtex**. Mixing them silently
   produces an empty bibliography.

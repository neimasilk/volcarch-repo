# E232 — DESIGN (pre-registration) — affix-aware residue screen, Tolaki pilot (idea I-150)

**Written:** 2026-10-15, before any affix inventory was compiled and before any form was stripped or compared.
**Status:** design only. **Not to be run until the PI has read this design** (orbit agenda K7, PI "setuju default",
2026-10-15; `docs/research_notes/ORBIT_REVIEW_20261015.md` §5). Amendments go in §9 with a date.
**Serves:** line 04 (P8 follow-up; idea I-150). **Not part of P8 R1** (submitted 2026-10-15); it may inform an answer if the
second review round asks about morphology, which P8 names as untested ("No form was morphologically segmented", Discussion;
first direction in the Conclusion).
**Rule:** the result is reported whichever way it falls; "no gain" is a valid and expected outcome.

## 0. What is known before the run (not outcomes of E232)

From E231 §D (`experiments/E231_p8_what_coded_means/README.md` 113–150; `results/D_pmp_lookalikes_by_list.csv`,
`results/D_tolaki_pmp_lookalikes.csv`), with the P8 screen (stem-tolerant normalised edit distance, up to three leading
characters ignored on either side, `experiments/E228_p8_revision_analyses/p8common.py` `d_stem`, threshold 0.34):

| Tolaki (ABVD list 674) | Value |
|---|---|
| uncoded forms compared (meaning has a PMP entry; ≥ 2 normalised letters) | 128 |
| look-alike of the PMP form of the same meaning | 12 (9.4 %) |
| chance (same comparison against 50 random other meanings) | 1.9 % |
| sensitivity: forms ABVD places in the PMP entry's set that the screen finds | 24 of 50 (48 %) |

The three-leading-character rule is a blind stand-in for prefix stripping: it removes any one to three letters, whether
or not they are an affix, and it cannot remove suffixes or undo nasal substitution.

## 1. Question

Does stripping affixes listed in a published grammar of Tolaki, instead of (or beside) ignoring up to three leading
letters, find **more** same-meaning look-alikes between uncoded Tolaki forms and ABVD's PMP forms — **beyond what
stripping arbitrary strings of the same number and length would find**, and without losing sensitivity on forms ABVD
codes in the PMP set?

## 2. Data (frozen)

- ABVD CLDF snapshot `experiments/E022_linguistic_subtraction/data/abvd/cldf/` (lexibank/abvd, git 917c5a5, 2025-10-07):
  Tolaki list 674 (Konawe dialect), PMP list 269. Same 128 uncoded forms and same 50 positive-control forms as E231 §D
  (rebuilt by the same code path; the run stops if either count differs).
- Normalisation, distance and chance procedure exactly as E231 §D (`norm`, `ned`; 50 random other meanings per form;
  same seed policy, re-seeded per list).

## 3. Step 0 — the affix inventory (must exist and be frozen before step 1)

- **Source:** Mead, David E. 1998. *Proto-Bungku-Tolaki: Reconstruction of its Phonology and Aspects of its
  Morphosyntax.* PhD dissertation, Rice University (516 pp.; Glottolog lists an image-only PDF; handle hdl:1911/19290 — access
  not yet verified). Part II (morphosyntax) is the expected source of the Tolaki verbal and nominal affixes. If the source
  cannot be obtained, **step 0 fails, E232 stays PRE-REGISTERED, and the gap is recorded** — no inventory is written from
  memory or from another language.
- **Content:** every affix the source attests for Tolaki (or for Proto-Bungku-Tolaki with a stated Tolaki reflex): form,
  allomorphs with their conditioning, nasal substitution (which root-initial consonant is restored), function gloss,
  **page reference**. Affixes the source marks as unproductive or doubtful are listed but flagged and excluded from the
  primary run.
- **Format:** the rule JSON of ModernKataKupas (`D:\documents\modern_kata_kupas\src\modern_kata_kupas\data\affix_rules.json`:
  `prefixes` / `suffixes`, `allomorphs` with `next_char_is` and `reconstruct_root_initial`). The ModernKataKupas engine is
  reused only if it can run with this rules file and **no root dictionary** (there is no Tolaki root list); otherwise a
  small stripping function in this folder, which returns **all** parses rather than choosing one.
- **Freeze:** SHA-256 of the inventory file recorded in §9 before step 1. ⚠ Domain flag: the PI reads the inventory; a
  Sulawesi specialist's look is desirable but not a precondition of the pilot.

## 4. Method

For each uncoded form *x* and the PMP form(s) *y* of the same meaning:

- **Arm A (baseline):** `d_stem(x, y)` as in E231 (reproduces 12 / 128 and 24 / 50; the run stops if it does not).
- **Arm B (grammar):** the minimum of `ned(norm(s), norm(y))` over *x* itself and every stem *s* that the inventory yields
  for *x* (prefixes, suffixes and their combinations, nasal substitution undone; stems shorter than 3 normalised letters
  discarded). No blind leading-letter skip.
- **Arm C (union):** min(A, B).
- **Chance:** each arm against the PMP forms of the same 50 random other meanings per form.
- **Random-inventory null:** 200 inventories with the same number of prefixes and suffixes as the grammar inventory and the
  same length of each entry, letters drawn from the letter frequencies of the Tolaki list, no allomorphy; arm C rerun with
  each (`C_random`).
- Thresholds: 0.34 (primary), 0.25 (secondary), as in P8.

## 5. Outcomes

- **O1 (primary):** the gain of arm C over arm A among the 128 uncoded forms, chance-corrected:
  Δ = (look-alikes_C − chance_C × 128) − (look-alikes_A − chance_A × 128), in forms.
- **O2:** sensitivity of arms B and C on the 50 control forms (baseline 24 / 50).
- **O3:** inflation of chance, chance_C − chance_A (percentage points).
- **O4:** Δ for the grammar inventory against the distribution of Δ for the 200 random inventories.

## 6. Decision rule (fixed now)

| Verdict | Condition (threshold 0.34) |
|---|---|
| **GAIN** | Δ ≥ 3 forms **and** Δ above the 95th percentile of the random-inventory Δ **and** sensitivity of C ≥ 24 / 50 |
| **NO GAIN** | Δ at or below the 95th percentile of the random-inventory Δ — grammar-based stripping adds nothing that arbitrary stripping of the same size does not |
| **INFLATION** | NO GAIN and chance_C − chance_A > 2 points — stripping mainly adds chance matches |
| **PARTIAL** | anything else (e.g. Δ ≥ 3 but not above the random 95th percentile): numbers reported, no claim |
| **INVENTORY TOO NARROW** (flag, any verdict) | sensitivity of arm B alone < 19 / 50 (ten points below baseline) |

A GAIN means only that look-alikes appear which the blind rule missed. **No pairing is a cognate judgment**; no list of
pairings is printed or published before a specialist has seen it (as in E231 §D).

## 7. Stage 2 (not part of the pilot; decided after it)

Muna (van den Berg 1989) and Makasar (Jukes 2020), each against PMP and, for Makasar, against Bugis and Sa'dan Toraja,
only if their grammars are obtainable and the pilot is not INFLATION. LexStat / partial cognates (List, Lopez & Bapteste
2016) is a separate later idea (I-150 second layer), not this experiment.

## 8. Cost and time

$0. Step 0 depends on access to the source (OCR of a scanned dissertation; the morphology chapters only). Steps 1–6: one
session once the inventory is frozen.

## 9. Amendments

(none)

# E090 v7 — Label-shuffle convergence test: FINDING (2026-06-10)

**Status: INTEGRITY-CRITICAL. The P16 "cross-tradition convergence" finding does NOT survive the correct non-circular test.**

## Why this test was run
DeepSeek G9 re-review of the **R1-revised** P16 Wacana draft (`papers/P16_.../external_reviews/critical_deepseek_p16_wacana_R1_20260610.md`) returned **REJECT**, with W1 (FATAL): the v6 "tradition-controlled" test is still circular — it compares within-group cross-tradition similarity against a **whole-corpus** baseline. Passages are tagged into a concept group *because they share keywords*, so they are topically similar by construction; a positive z vs the whole corpus is near-guaranteed and does not isolate "different traditions converge."

DeepSeek's prescribed fix (verbatim): compare observed within-concept cross-tradition similarity to the distribution from **randomly shuffling tradition labels within each concept group**. v7 implements exactly that (`e090_v7_label_shuffle.py`), holding topical coherence constant and varying only the tradition labels.

## Result — the finding REVERSES
| group | n | nTrad | obsCross | nullMean | z | verdict |
|---|---|---|---|---|---|---|
| JAVA | 82 | 11 | 0.338 | 0.356 | **−12.65** | NULL/divergence |
| SUMATRA_GOLD | 75 | 11 | 0.357 | 0.369 | **−14.09** | NULL/divergence |
| CAMPHOR_BARUS | 51 | 10 | 0.389 | 0.404 | **−8.48** | NULL/divergence |
| SPICE_TRADE | 76 | 10 | 0.373 | 0.386 | **−13.05** | NULL/divergence |
| MARITIME_VOYAGE | 128 | 12 | 0.328 | 0.338 | **−10.02** | NULL/divergence |
| VOLCANO | 54 | 12 | 0.326 | 0.336 | **−8.51** | NULL/divergence |
| BUDDHIST_WORLD | 23 | 8 | 0.331 | 0.346 | **−5.82** | NULL/divergence |
| METAL_TRADE | 135 | 12 | 0.309 | 0.317 | **−11.93** | NULL/divergence |

**Cross-tradition convergence: 0/8 groups** (v6 claimed 8/8). Every group shows cross-tradition pairs are *less* similar than chance relabeling — i.e. the topical cluster is held together by **within-tradition homogeneity**, not cross-tradition convergence.

**Corroboration (v6 own numbers):** in all 8 groups S_within > S_cross (e.g. VOLCANO 0.422 vs 0.326; JAVA 0.464 vs 0.338). The within-tradition signal is uniformly the stronger one. v6 masked this only because the whole-corpus cross-tradition baseline (~0.31–0.37) sits even lower, making the group's cross pairs look "high."

## Interpretation
- The semantic-space framing — "independent traditions converge on theme X" — is an **artifact of keyword selection + within-tradition style**. It is REFUTED by the proper test. DeepSeek's W1 was correct, and the R1 fix (v6) did not resolve it.
- What DOES survive is the much weaker **distributional** fact: each theme is *attested across many/all traditions* (VOLCANO 12/12, MARITIME 12/12, METAL 12/12, JAVA 11/12, SUMATRA_GOLD 11/12…). That is "pan-traditional attestation," a co-occurrence count — NOT semantic convergence, and not the headline P16 currently makes.

## Consequence for P16 → Wacana
**NO-GO for submission in current (R1) form.** Per `docs/SUBMISSION_INTEGRITY_GATE.md` (G1 re-derivation, G4 circularity, G8 overstatement, G9 cross-model): the central pillar fails on re-derivation; tempering wording would be the banned move. Submitting now would put a refuted central claim into a Scopus venue.

Decision required from Pak Amien (none taken autonomously):
1. **Reframe + downgrade** — drop the semantic-convergence headline; rebuild P16 around the defensible distributional/attestation finding + the inscription-vs-text genre asymmetry (honestly labelled as genre, not "taphonomic bias"); remove the n=46 929 CE diachronic claim (DeepSeek W2, unfixable by wording). Smaller, true paper.
2. **Switch venue to DHQ** — does not rescue W1; the convergence claim is refuted regardless of venue.
3. **Park P16** until a genuinely non-circular convergence design exists (e.g. unsupervised clustering that recovers themes without prior keyword tagging — DeepSeek's "gold standard"; untested).

## Files
- `e090_v7_label_shuffle.py`, `results/e090_v7_label_shuffle.json` (this test)
- `e090_v6_tradition_controlled.py`, `results/e090_v6_tradition_controlled.json` (superseded R1 test)
- `papers/P16_.../external_reviews/critical_deepseek_p16_wacana_R1_20260610.md` (G9 re-review that triggered this)

# P8 Revision Support Material: E087 Negative Control Results

**Paper:** "Linguistic Fossils" — Oceanic Linguistics MS# OL-03-2026-11
**Date:** 2026-03-16
**Severity:** HIGH — requires reframing of ML substrate detection claims

---

## The Problem

E087 ran the E027 substrate detector on language pairs where NO substrate is expected:

| Control | Languages | AUC (pure phonology) | Substrate Expected? |
|---------|-----------|---------------------|---------------------|
| Reference | 6 Sulawesi | 0.727 | YES |
| C1 | Tagalog + Cebuano | 0.568 | No |
| C3 | Random labels | 0.500 | No |
| **C5** | **Iban + Malay** | **0.713** | **No** |

**C5 (Iban + Malay) produces AUC=0.713 — nearly matching Sulawesi's 0.727.** The gap is only 0.014.

## Root Cause

The E022 residual labeling method conflates:
- **Words missing from ABVD cognacy database** (documentation gap)
- **Words from a genuine pre-Austronesian substrate** (what we claim to detect)

Iban has 75.6% ABVD coverage vs Malay 97.5%. The 24.4% "residuals" in Iban are not substrate — they are words that ABVD hasn't documented cognates for. These words happen to have different phonological profiles (more consonant clusters, different prefix patterns) → the classifier picks this up.

## What This Means for P8

**The AUC=0.762 is partly a documentation artifact.** Not all of it — the signal is genuinely above chance (p=0.0000 permutation test, z=10.0) and well above the C1 Philippine control (0.568). But the "substrate detection" claim is overstated.

## Honest Reframing (for revision response)

### OLD claim (P8 as submitted):
> "Our XGBoost classifier detects pre-Austronesian phonological substrate with AUC=0.762"

### NEW claim (for revision):
> "Our ML classifier identifies systematic phonological non-conformity in Sulawesi residual vocabulary (AUC=0.727 on pure phonological features, p<0.001 vs. permutation null). This signal significantly exceeds closely related control language pairs (C1 Philippine AUC=0.568, gap=+0.159), though it is comparable to language pairs with large ABVD documentation differentials (C5 Iban+Malay AUC=0.713, gap=+0.014). We interpret this as evidence that non-cognate Sulawesi vocabulary has a distinct phonological profile *consistent with* substrate influence, while acknowledging that database coverage gaps contribute to the measured signal."

## What SURVIVES

1. **The phonological fingerprint IS real** — not noise (permutation p=0.0000, z=10.0)
2. **Closely related languages DON'T produce the same signal** — C1 (Tagalog+Cebuano) = 0.568
3. **The E028 consensus substrates remain valid** — they identify words with BOTH rule-based AND ML agreement
4. **The E036 Hanacaraka convergence** (33→20 consonant reduction) is independent of ML
5. **The E029 clustering result** (parallel innovation, not shared substrate) is CONSISTENT with this finding — if the signal were pure substrate, we'd expect clustering

## What Is DAMAGED

1. The specific AUC=0.762 cannot be cited as proof of "substrate detection"
2. The LOLO cross-validation (5/6 languages ≥ 0.65) may be inflated by documentation effects
3. The geographic patterning in E027b (Sulawesi > Western Indonesian) might reflect ABVD coverage gradients, not substrate gradients
4. "32.3% of Sulawesi vocabulary is substrate" → not defensible as stated

## Recommended If Reviewer Raises This

"We acknowledge that our ML classifier's performance reflects a combination of genuine phonological non-conformity in residual vocabulary AND artifacts of database documentation coverage (E087 negative control). The phonological fingerprint is real — permutation testing confirms the signal is 10 SD above noise — but its interpretation as pre-Austronesian substrate is one of several possible explanations, alongside:
(a) Lexical borrowing from non-Austronesian languages (Papuan contact),
(b) Independent innovation in less-documented lexical domains, and
(c) ABVD documentation gaps creating artificial residual categories.

The classifier is best understood as a RANKING TOOL that identifies phonologically anomalous vocabulary warranting individual etymological investigation, not as a definitive substrate classifier."

---

*Prepared 2026-03-16. This is an honest assessment of a genuine vulnerability in P8.*

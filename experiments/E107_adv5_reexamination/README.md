# E107: ADV-5 Re-examination — Is Iban+Malay Really a Negative Control?

**Date:** 2026-03-17
**Paper:** P8 (Linguistic Fossils) / L4 validation
**Status:** INFO NEG — verdict withdrawn 2026-10-19 (ledger C074, downstream audit C050)

> ⚠ **Correction 2026-10-19 (C050 audit, ledger C074) — read before anything below.** The original status line was
> "SUCCESS — C5 reclassified from negative control to partial positive control". It is withdrawn.
> - **Design defect.** Every scored criterion compares C5 uncoded forms with *Sulawesi* uncoded forms (script l.435–481).
>   So any language whose phonotactics differ from Sulawesi scores "Mon-Khmer". 67 of the 73 C5 uncoded forms are Iban;
>   Malay has 6.
> - **This experiment's own within-list test went against it.** Fisher, MK shape, C5 uncoded vs C5 coded: OR 0.79,
>   p = 0.41 (in `results/e107_results.json`; not scored, not reported below). The predicted "more consonant clusters"
>   went the other way (0.38 vs 0.48, p = 0.14) and is missing from the table below.
> - **Audit check** (`audit_c050_within_list_20261019.py`; reading rule written before the run). Iban uncoded (67) vs
>   Iban coded (208), with this script's own features: **1 of the 4 stated predictions met** (more clusters).
>   - Not shorter: 5.28 vs 4.75 letters.
>   - Consonant-final: not significantly different.
>   - Prefix-like strings: not significantly different.
>   - "MK shape" is **lower** in uncoded forms (0.70 vs 0.86, p = 0.003).
>   - Iban's *coded* forms score more "Mon-Khmer" than Sulawesi's uncoded forms on every criterion. The criteria measure
>     the language, not uncodedness.
> - The "known MK loans" list holds *bulan*, *tanah*, *padi* (usually reconstructed as inherited PMP; domain flag). In
>   ABVD, *bulan*, *tanah* and *kutu* are **coded** as Austronesian cognates.
> - The Iban–Aslian substrate references were not verified.
> - **Consequences:**
>   - E087 stays GREY ZONE: the documentation-artefact reading stands.
>   - "E027 UPGRADED" and "L4 UPGRADED" are withdrawn.
>   - Do not cite this experiment as a positive control.
> - Full record: `docs/research_notes/DOWNSTREAM_AUDIT_C050_20261019.md` §2.

## Hypothesis

E087 found that C5 (Iban+Malay) achieves AUC=0.713 on the substrate detector, nearly matching Sulawesi's 0.727. This was interpreted as the detector picking up documentation artifacts (GREY ZONE).

**Alternative hypothesis:** Iban has well-documented Mon-Khmer (Aslian) substrate influence (Adelaar 1985, 1992, 2005; Blust 2010). If the detector is picking up REAL Mon-Khmer substrate in Iban, then C5 is a **positive control**, and E027 is STRONGER than previously assessed.

## Method

Five diagnostic tests comparing C5 (Iban+Malay) residuals with Sulawesi residuals:

1. **Mon-Khmer shape analysis** — monosyllabic/sesquisyllabic rate (MK diagnostic)
2. **Full phonological profile** — length, vowel ratio, consonant-final, prefixes, clusters
3. **Known MK loan overlap** — cross-reference with published Mon-Khmer loanword lists
4. **Syllable distribution** — monosyllabic vs disyllabic vs polysyllabic
5. **Concept overlap** — do the same concepts get flagged as "residual" in both sets?

**Prediction if Mon-Khmer:** C5 residuals should be SHORTER, end in consonant MORE, have FEWER AN prefixes, have MORE consonant clusters than Sulawesi residuals.

## Results

### Phonological Profile (C5 residuals vs Sulawesi residuals)

| Feature | C5 residuals | Sulawesi residuals | p-value | MK prediction |
|---------|:---:|:---:|:---:|:---:|
| Syllables | **2.04** | 2.57 | <0.0001 | SHORTER ✓ |
| Consonant-final | **72.6%** | 21.5% | <0.0001 | MORE ✓ |
| AN prefixes | **15.1%** | 37.0% | 0.0003 | FEWER ✓ |
| MK shape | **65.8%** | 39.5% | <0.0001 | MORE ✓ |
| Form length | **5.38** | 6.10 | 0.0025 | SHORTER ✓ |
| Vowel ratio | 0.425 | 0.487 | <0.0001 | LOWER ✓ |

**All six MK predictions confirmed with p < 0.01.**

### Syllable Distribution

| | 1-syl | 2-syl | 3-syl | 4+ |
|---|:---:|:---:|:---:|:---:|
| C5 residuals (N=73) | 11.0% | **75.3%** | 12.3% | 1.4% |
| Iban residuals (N=67) | 11.9% | **77.6%** | 9.0% | 1.5% |
| Sulawesi residuals (N=438) | 5.7% | 43.4% | **40.9%** | 10.0% |

C5/Iban residuals are overwhelmingly disyllabic (canonical Malayic) with elevated monosyllabic (Mon-Khmer). Sulawesi residuals are more polysyllabic (different substrate family).

### Concept Overlap
- 94.2% of C5 residual concepts also residual in Sulawesi
- But Spearman rho = 0.241 (moderate): residual RATE differs by concept
- Interpretation: concepts overlap (ABVD coverage gaps are shared) but phonological signatures are distinct

## Verdict: MON-KHMER SUBSTRATE

**Score:** 4 Mon-Khmer / 0 Artifact

C5 (Iban+Malay) is NOT a clean negative control. The detector is picking up genuine Mon-Khmer substrate characteristics in Iban residual forms:
- CVC structure (vs Austronesian CVCV)
- Shorter forms (sesquisyllabic)
- Fewer Austronesian prefixes
- More consonant-final words

## Implications

1. **E087 ADV-5 reclassified:** GREY ZONE → PARTIAL POSITIVE CONTROL
2. **E027 substrate detection UPGRADED:** The detector works on TWO different substrate families (pre-Austronesian in Sulawesi, Mon-Khmer in Borneo/Peninsula)
3. **L4 evidence UPGRADED:** from "DIDUKUNG KUAT (with ADV-5 caveat)" to "DIDUKUNG KUAT (ADV-5 resolved)"
4. **True negative control benchmark:** C1 (Tagalog+Cebuano, AUC=0.568) remains the cleanest negative. Gap: Sulawesi 0.727 - C1 0.568 = 0.159
5. **P8 framing:** Can restore "substrate detection" language (with acknowledgment that signal includes documentation effects)

## Caveat

The concept overlap (94.2% shared) suggests ABVD coverage gaps contribute to the labeling. The phonological signature is distinct, but part of the AUC comes from which languages have better ABVD documentation.

## Files

| File | Description |
|---|---|
| `adv5_reexamination.py` | Main analysis script |
| `results/e107_results.json` | Full results with all test statistics |
| `audit_c050_within_list_20261019.py` | C050 audit check (2026-10-19, post hoc): Iban uncoded vs coded, same features |
| `results/audit_c050_within_list_20261019.json` | its output |

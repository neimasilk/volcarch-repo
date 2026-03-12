# P8 External Review Triage — 2026-03-11

**Sources:** ChatGPT + Gemini reviews of draft_v0.1.pdf
**Matrix:** {confidence × reversibility} → ACT / ACT WITH CARE / ACKNOWLEDGE / REBUT / NOTE / IGNORE

---

## Criticism Map

### TK#1: Label Validity / Ground Truth (ChatGPT #1, Gemini #2)
**Claim:** No ground truth exists for "substrate" vs "Austronesian." Labels are derived from ABVD cognacy gaps — the model learns to predict database incompleteness, not linguistic reality.

**Assessment: ACT WITH CARE**
- The paper ALREADY frames this as PU learning (§2.1, line 93)
- The ablation (§3.3) shows removing cognacy_coverage IMPROVES performance
- LOLO validation shows cross-linguistic generalization (not database memorization)
- BUT: the paper should be clearer that we're detecting "phonological non-conformity with cognacy coding as proxy," not actual substrates
- **Action:** Tighten PU learning framing. Add sentence: "We make no claim that these labels represent true substrate identity; rather, they index forms that resist attribution to known Austronesian etyma."

### TK#2: Feature Leakage — cognacy_coverage (ChatGPT #2)
**Claim:** The #1 SHAP feature (language_cognacy_coverage) is a database artifact, not a linguistic property.

**Assessment: ALREADY FULLY ADDRESSED**
- Ablation experiment (Table 4): removing it IMPROVES AUC (0.760→0.763)
- All 6 LOLO languages ≥0.65 after ablation (was 5/6 before)
- This is already in the paper (§3.3, §4.5 Limitation 4)
- **Action:** REPORT ABLATED MODEL AS PRIMARY RESULT. Currently the paper leads with the full 27-feature model. The ablated 26-feature model (AUC 0.763) should be the headline number.

### TK#3: Orthography ≠ Phonology (ChatGPT #3, Gemini #1 "most fatal")
**Claim:** All features are computed from orthographic forms, not IPA. Different languages use different orthographic conventions. The model may learn orthographic patterns, not phonological ones.

**Assessment: ACT — run validation experiment**
- The paper acknowledges this (Limitation 1)
- LOLO success partially addresses it (different orthographies → model still generalizes)
- But a direct IPA comparison for even 2 languages would be a powerful defense
- **Action:** Run E041 — approximate IPA conversion for 2 languages. Compare feature distributions and model performance. Even if approximate, this transforms the criticism from "unaddressed" to "tested."
- **Fallback:** If IPA conversion is infeasible, strengthen the LOLO argument: "If orthographic conventions were the primary signal, cross-linguistic generalization would fail because each language uses different orthographic conventions. The fact that Model B generalizes across 6 languages with different orthographies (6/6 LOLO ≥ 0.65 after ablation) suggests the model captures phonological regularities robust to orthographic variation."

### TK#4: Negative Clustering ≠ Proof of Independent Innovation (ChatGPT #4)
**Claim:** Absence of clustering is consistent with multiple explanations: (a) deeply diverged single substrate, (b) multiple substrates, (c) insufficient statistical power, (d) orthographic Levenshtein inadequate for phonological comparison.

**Assessment: ACT WITH CARE — soften language**
- Fair critique. The paper currently says "rules out" (line 489) — too strong
- Should change to "provides no evidence for" / "is not supported by"
- Should acknowledge alternative explanations
- **Action:** Replace "rules out the hypothesis" → "provides no support for the hypothesis." Add sentence acknowledging that deeply diverged substrates or multiple distinct substrate languages could produce the same negative result.

### TK#5: Hanacaraka Section Speculative/Unrelated (ChatGPT #5)
**Claim:** The Hanacaraka section (§4.4) is from Java, not Sulawesi. It's convergent but not directly linked to the ML results. Makes the paper unfocused.

**Assessment: REBUT + NOTE**
- The Hanacaraka section is about consonant INVENTORY reduction (33→20), not pangram narratives (I-053 is NOT in the paper)
- This is a legitimate CONVERGENT argument: the same phonological categories the ML detects as non-conforming are the ones historically stripped from the writing system
- The script adaptation is evidence that the model detects real Austronesian phonotactic constraints
- However, reviewers may find it tangential to a Sulawesi ML paper
- **Action:** KEEP but frame more carefully as "independent convergent evidence." Add a sentence acknowledging the geographic limitation. Could be shortened by 30% if needed.

### TK#6: Morphological Confound (Gemini #3)
**Claim:** Compound/derived forms may be longer and have more consonant clusters, mimicking the substrate fingerprint. The model may detect morphological complexity, not substrate origin.

**Assessment: PARTIALLY ADDRESSED + ACKNOWLEDGE**
- The paper already identifies numeral compounds as false positives (§3.5.1)
- The substrate fingerprint includes "fewer canonical Austronesian prefixes" — substrates are LESS prefixed, not more. This goes AGAINST the morphological confound hypothesis.
- Compound numerals are the main exception, and they're explicitly documented
- **Action:** Add a paragraph noting that the "fewer prefixes" finding actually cuts against the morphological confound: substrates are phonologically SIMPLER in morphological terms (fewer prefixes) but LONGER in raw form length, suggesting the length signal is stem-level, not morphological.

### TK#7: Bait-and-Switch Terminology (Gemini #4)
**Claim:** The title says "substrate detection" but the conclusion says there are no substrates. Paper claims to detect something, then says it doesn't exist.

**Assessment: ACT — terminology cleanup**
- The RED TEAM already flagged this (I-1)
- The title actually says "Non-Mainstream Vocabulary" (already reframed)
- But the internal text uses "substrate" ~50 times
- The title is fine. The issue is internal terminology consistency.
- **Action:** Systematic find-replace: "substrate" → "non-mainstream" or "non-conforming" in most contexts. Keep "substrate candidate" and "substrate hypothesis" only where the hypothesis is being tested. Keep "shared substrate" for the hypothesis being rejected.

### TK#8: Both Suggest Reframing to "Phonological Anomaly Detection"
**Assessment: PARTIALLY ADOPT**
- "Phonological anomaly" is less theoretically loaded than "substrate" — good
- But "anomaly" implies pathological. "Non-mainstream" (already in the title) is better
- The paper's actual contribution IS a method for detecting phonological non-conformity + the negative result about shared origins
- **Action:** Keep current title ("Phonological Fossils: ML Detection of Non-Mainstream Vocabulary..."). Add to introduction: explicit statement that the method is a "phonological non-conformity detection tool" applicable beyond substrate questions.

---

## Priority Actions

### MUST DO (before submission)
1. ✅ **Report ablated model as primary** — Abstract now leads with AUC 0.763 (26 features)
2. ✅ **Terminology cleanup** — Key instances updated in abstract, intro, conclusion
3. ✅ **Soften negative result** — "rules out" → "provides no support for" + alternative explanations added
4. ✅ **Strengthen LOLO argument against orthographic criticism** — new §3.4 IPA Robustness Test

### SHOULD DO (significantly strengthens paper)
5. ✅ **E041: IPA validation** — DONE. CV +0.002, LOLO +0.009. ROBUST. Muna +0.042.
6. ✅ **Morphological confound paragraph** — Added after numeral compound section (§3.5.1)
7. ✅ **Tighten PU learning framing** — Limitation 3 rewritten with proxy label caveat

### NICE TO HAVE
8. **Shorten Hanacaraka section** by ~30% — DEFERRED (author decision)
9. ✅ **Add paragraph on alternative explanations for negative clustering** — Added in §4.2

---

## What NOT to Do

- ❌ Do NOT drop the Hanacaraka section entirely — it's valid convergent evidence
- ❌ Do NOT rename the paper to "Phonological Anomaly Detection" — current title is better
- ❌ Do NOT add a full IPA pipeline — approximate conversion for 2 languages suffices
- ❌ Do NOT suppress the negative clustering result — it's the paper's strongest contribution
- ❌ Do NOT try to prove substrates exist — the paper's value is the METHOD + the NEGATIVE RESULT

---

*Created 2026-03-11. Use as roadmap for P8 v0.2 revision.*

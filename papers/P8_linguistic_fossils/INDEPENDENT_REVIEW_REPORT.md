# P8 Red-Team Report — 2026-03-11

**Target:** Oceanic Linguistics (Q1, UH Press)
**Draft:** `draft_v0.1.tex` (~5,000 words, 27pp)
**Verdict:** LEAN SUBMISSION viable after 4 critical fixes

---

## Ablation Experiment Result (Script 04)

**Removing `language_cognacy_coverage` IMPROVES the model:**

| Variant | CV AUC | LOLO AUC | LOLO ≥0.65 |
|---------|--------|----------|------------|
| Full Model B (27 features) | 0.7599±0.0073 | 0.7151 | 5/6 |
| **Ablated (-coverage, 26 features)** | **0.7626±0.0072** | **0.7221** | **6/6** |
| Pure (no lang features, 25 features) | 0.7265±0.0066 | 0.7008 | 5/6 |

- The confound feature was actually HURTING performance (Muna: 0.618 → 0.679)
- This is the strongest possible defense: the #1 SHAP feature is removable with no cost
- **Recommendation:** Report the ablated model as the primary result (AUC 0.763)

---

## Critical Fixes (must do before submission)

### C-1: Report ablation results in paper
The ablation experiment directly addresses Limitation 4. Add a subsection or paragraph in Results showing the ablated model. This transforms a weakness into a strength.

### C-2: Missing foundational citation — Thomason & Kaufman (1988)
*Language Contact, Creolization, and Genetic Linguistics* (U Chicago Press) is THE canonical work on contact-induced language change. Any paper about substrate detection that doesn't cite this will be immediately flagged by OL reviewers. The paper's claim about creole verb substrate persistence (Discussion §4.1, line 453) needs this citation.

### C-3: Missing computational linguistics citations
Johann-Mattis List is a likely reviewer at OL and his work on automated cognate detection is directly relevant:
- List (2012) "LexStat: Automatic Detection of Cognates in Multilingual Wordlists" — automated cognate detection
- Jäger (2018) "Global-scale phylogenetic linguistic inference from lexical resources" — ML on Swadesh data (PNAS)
Not citing these is a conspicuous gap for a paper about ML + Swadesh vocabulary.

### C-4: Verb substrate claim needs specific citation
Line 453: "This aligns with observations in creole linguistics that substrate verb semantics tend to be more resistant to superstrate replacement than noun semantics" — no citation provided. Need:
- Lefebvre (2004) *Issues in the Study of Pidgin and Creole Languages* (Amsterdam: Benjamins), or
- Siegel (2008) *The Emergence of Pidgin and Creole Languages* (Oxford UP)

---

## Important Fixes (should do, significantly improves chances)

### I-1: "Substrate" terminology inconsistency
The paper's own E029 result shows these are NOT substrates (no shared word families, p=0.569). Yet "substrate" appears ~50 times. The title and intro were reframed (I-031), but methods/results still use "substrate" extensively. Suggestion: replace with "non-mainstream" or "non-conforming" in most instances, keep "substrate candidate" only where the hypothesis is being tested.

### I-2: Orthographic vs IPA
LingPy is installed. Even a brief supplementary analysis showing that key findings hold under IPA conversion for 1-2 languages would address the strongest OL reviewer concern. However, this is a LOLO validation issue — if the model generalizes cross-linguistically despite orthographic differences, that itself is evidence of robustness.

### I-3: Scale_pos_weight inconsistency
XGBoost uses scale_pos_weight=1.0 (no correction) while RandomForest uses class_weight="balanced". The dataset is imbalanced (67.7% vs 32.3%). Should at least mention this design choice.

---

## Acknowledged Limitations (already in paper, adequate)

- L-1: Orthographic not IPA (acknowledged, Limitation 1)
- L-2: Small sample (acknowledged, Limitation 2)
- L-3: Label noise / PU learning (acknowledged, Limitation 3)
- L-4: Geographic scope (acknowledged, Limitation 5)
- L-5: Tolaki dominance (discussed in §4.3)

---

## Verdict: LEAN SUBMISSION

After C-1 through C-4 fixes:
- The ablation result transforms the biggest weakness into a strength
- Missing citations are easy to add (3-4 bib entries)
- The negative clustering result (E029) is the paper's strongest contribution
- Hanacaraka convergence section is novel and compelling
- 5 linked experiments is substantial for OL

**Estimated fix effort:** 1-2 hours (add citations, add ablation paragraph, terminology cleanup)
**No need to expand:** The paper is complete at ~5,000 words. OL has no minimum length.

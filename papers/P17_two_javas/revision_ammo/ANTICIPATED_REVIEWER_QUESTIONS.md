# P17 "Two Javas" — Anticipated Reviewer Questions & Pre-Written Responses

**Journal:** Archeologia e Calcolatori (computational archaeology, double-blind)
**Submitted:** 2026-04-09, ID 365
**Prepared:** 2026-04-09 (same day — demonstrates preparedness)

---

## Q1: "The cascade model is underdetermined — 5 parameters, 1 data point."

**Response:**
We agree. We have conducted a systematic parsimony analysis (E176) showing that 3-factor submodels also bracket the observed gap, and that the AIC-equivalent improvement from 3 to 5 factors is marginal (6.73 → 6.25). We have revised the manuscript to describe the cascade as "a plausible mechanistic decomposition consistent with observations" rather than a validated predictive model. The cascade's value is pedagogical (decomposing the gap into identifiable factors) and practical (volcanic burial is the only spatially predictable factor), not quantitatively precise.

We note that the paper's core contribution — the spatial segregation of candi and inscriptions — does NOT depend on the cascade model. The Two Javas pattern is a distributional observation (Mann-Whitney p < 0.000001, Cohen's d ≈ 2.0) that stands independently.

---

## Q2: "Spatial autocorrelation — are your correlations inflated?"

**Response:**
We have conducted a Moran's I analysis (E184) confirming that volcanic distance is spatially autocorrelated (I = 0.937, p < 0.001). Simple regressions involving volcanic distance may indeed be inflated — our own test shows that the volcano-century correlation collapses from ρ = 0.490 to ρ = −0.198 after spatial lag correction.

We have conducted formal spatial regression using PySAL's ML estimation (E187). The Spatial Lag Model (Rho = 0.620, p < 0.001) and Spatial Error Model (Lambda = 0.626, p < 0.001) both confirm strong spatial dependence. The volcanic distance effect on inscription century, while significant in OLS (p = 0.002), becomes non-significant in both spatial models (Lag: p = 0.094; Error: p = 0.241).

However, the paper's central findings are distributional comparisons (Mann-Whitney, Kolmogorov-Smirnov), not regressions. These tests compare two spatial populations (candi vs inscriptions) and are immune to spatial autocorrelation inflation. The Two Javas segregation survives all spatial correction tests with Cohen's d ≈ 2.0 — a very large effect size (E185).

We have added a paragraph to the Limitations section acknowledging this issue. The temporal vocabulary trend should be interpreted as "spatially patterned" rather than "caused by volcanic proximity."

---

## Q3: "The Philippines has pre-400 CE sites in volcanic zones. Why is Java different?"

**Response:**
Excellent point. Our comparative analysis (E178) reveals that karst availability is a confounding factor. Philippine volcanic zones have extensive karst (fraction ~0.20), providing cave sites that bypass the entire taphonomic cascade. Java's volcanic interior has very little karst (~0.08). Cave sites survive burial, preserve organics, are easy to survey, and have recognizable stratigraphy — they bypass all five cascade factors simultaneously.

We have added this as a limitation in the revised manuscript. The honest framing is: Java's archaeological darkness results from volcanic burial combined with low karst availability and insufficient survey — not volcanism alone. This actually strengthens the paper's practical recommendation: survey should target the few karst areas within volcanic Java (Pacitan, Tuban) as well as the non-karst volcanic slopes.

---

## Q4: "How do you address the tautology problem — does your model learn visibility, not suitability?"

**Response:**
The settlement suitability model (Paper 2, under separate review at JCAA) passed both a stratified cross-validation test and a temporal split test (E013–E014: AUC = 0.755 on sites discovered post-2000, trained on pre-2000 sites). The model performs BETTER in the least-surveyed areas (delta AUC = +0.057 in the lowest survey quartile), which is the opposite of what a tautological model would produce.

In this paper (P17), we do not use the settlement model for any statistical claims. The Two Javas pattern is derived entirely from DHARMA inscription data and published candi locations, not from model predictions.

---

## Q5: "The DHARMA corpus is only 268 inscriptions. Is this enough?"

**Response:**
We acknowledge this limitation explicitly. The 268 DHARMA inscriptions represent the most comprehensive digitised corpus of Old Javanese epigraphy available (TEI-XML with georeferencing). While larger corpora exist in published form (e.g., the OJO series), they lack the standardised digital format needed for computational analysis.

Our analysis uses 176 geocoded inscriptions from this corpus. The key findings — candi-inscription spatial segregation (Mann-Whitney p < 0.000001) and the 929 CE discontinuity (chi-squared p = 0.0003) — have very strong statistical significance that would survive substantial reduction in sample size. Bootstrap resampling (10,000 iterations) confirms all five core findings are robust to sample perturbation (E159).

We note that the Babad Tanah Jawi NLP analysis (E150) provides an independent confirmation using a non-DHARMA source: the chronicle's lexical stratum is 83.9% native/non-Sanskrit, consistent with our inscription-based findings.

---

## Q6: "The 929 CE Mataram collapse — is this a before-after observation or a natural experiment?"

**Response:**
We have deliberately revised the language from "natural experiment" to "before-after observation" following external review feedback. The 929 CE event was not randomly assigned and may confound geographic relocation with vocabulary change. We present it as a temporal discontinuity that illuminates the spatial structure, not as a causal test.

The statistical evidence (semantic distance z = 3.04, topic shift chi-squared p = 0.0003) quantifies the magnitude of the discontinuity, but causation would require a counterfactual that history does not provide.

---

## Q7: "Where is the ground-truthing? This is all computational."

**Response:**
This is the paper's acknowledged limitation and its explicit call to action. We provide 20 GPS-precise predictions with depths, methods, and costs (E171 prediction registry, planned for Zenodo deposit). The framework is designed to be falsifiable: GPR survey at predicted Zone B targets should find 2.5 [0, 6] subsurface anomalies (95% CI). If nothing is found (P = 7%), the framework is refuted.

We are actively seeking geophysical and archaeobotanical collaborators for field validation. The phytolith extraction pathway (volcanic matrices from Liyangan or PVMBG cores) offers a low-cost first test (~$5K).

---

## Q8: "AI-generated content concerns."

**Response:**
Computational tools including large language models were used for data analysis scripting, statistical computation, and literature synthesis. All scientific claims, interpretations, and methodological decisions were made by the author. The AI contribution is disclosed in the manuscript's AI disclosure section and is limited to computational assistance — equivalent to using statistical software or GIS tools.

The full analytical codebase is available on GitHub (volcarch-repo) for independent verification and replication.

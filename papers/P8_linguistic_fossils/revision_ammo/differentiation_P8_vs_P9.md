# Cross-Citation Differentiation: P8 vs P9

**P8:** "Phonological Fossils: Machine Learning Detection of Non-Mainstream Vocabulary in Sulawesi Basic Lexicon" — submitted to *Oceanic Linguistics*
**P9:** "Peripheral Conservatism as Archaeological Proxy" — submitted to *Journal of Southeast Asian Studies* (JSEAS)
**Prepared:** 2026-03-13 (Mata Elang #6, anti-salami-slicing protocol)

---

## 1. What Makes P8 Different from P9

P8 is a **computational linguistics methods** paper. It introduces a machine learning pipeline (rule-based cognate subtraction + XGBoost classifier on 26 phonological features) for detecting non-mainstream vocabulary in Austronesian languages. Its core contribution is methodological: demonstrating that phonological properties alone carry a detectable signal distinguishing inherited from non-conforming vocabulary (AUC = 0.763), and then using that tool to deliver a critical negative result — the detected non-mainstream forms show no evidence of deriving from a single pre-Austronesian language (clustering silhouette = 0.114; cross-linguistic cognate test p = 0.569). P8 asks: *can we computationally detect substrate signals, and do they cohere?*

P9 is a **framework paper for Southeast Asian studies**. It proposes the Peripheral Conservatism Framework (PCF) for recovering pre-Hindu substrates by comparing centres with peripheries across multiple evidence domains (lexical, ritual, botanical). Its linguistic component uses straightforward ABVD cognacy rate comparison (not machine learning) to establish that Balinese preserves more PMP cognates than Javanese. P9 asks: *where in the landscape do pre-Hindu substrates survive, and how do we systematically recover them?*

**The overlap:** Both papers work with ABVD data and discuss "substrate" in Austronesian languages. Both reference the distinction between inherited and non-conforming vocabulary.

**The distinction:** P8 develops and validates a novel ML detection tool; P9 applies simple cognacy counting to a comparative question. P8's geographic focus is Sulawesi (6 core languages, 16 expansion); P9's geographic focus is Java-Bali-Madagascar. P8's key finding is negative (no shared substrate families); P9's key finding is positive (peripheries conserve at large scale). The papers could not substitute for each other.

---

## 2. Unique Analyses Per Paper

| Dimension | P8 (Phonological Fossils) | P9 (Peripheral Conservatism) |
|-----------|--------------------------|------------------------------|
| **Core hypothesis** | Phonological features can detect non-mainstream vocabulary without cognacy data | Geographic/political/temporal peripheries preserve pre-Hindu substrates |
| **Methodology** | Machine learning (XGBoost, 26 features, SHAP explainability) | Comparative cognacy counting + multi-domain consilience |
| **Primary dataset** | 1,357 forms from 6 Sulawesi languages; expansion to 22 languages | 8 Austronesian varieties (Jav, Bal, Tengger, Malagasy, Muna, Toraja, etc.) |
| **Key experiment: E027** | XGBoost AUC=0.763, LOLO 5/6 >= 0.65, SHAP beeswarm analysis | Not used |
| **Key experiment: E041** | IPA robustness validation (AUC change < 0.01) | Not used |
| **Key experiment: E042** | Syllable-count robustness validation | Not used |
| **Key experiment: E036** | Hanacaraka 33->20 consonant reduction aligns PAn, not Sanskrit | Not used |
| **Key experiment: E029** | Parallel innovation negative result (p=0.569) — no shared substrate | Not used |
| **Key experiment: E049** | Maritime vocabulary domain-specific conservation | Not used |
| **Key experiment: E043** | Not used | Bal 40.3% > Jav 33.0% > Tengger 27.7% PMP cognacy; krama-alus register analysis |
| **Key experiment: E044** | Not used | 4-layer botanical substitution chain (Canarium -> dammar -> menyan -> kamboja) |
| **Key experiment: E050** | Not used | Canarium GBIF: pan-Austronesian distribution, 388 Madagascar records |
| **Key experiment: E034** | Not used | Panji absent from Malagasy = post-1200 CE calibration point |
| **Key negative result** | Substrate candidates do NOT form shared word families | Tengger has LOWER cognacy than Javanese (drift, not conservation at small scale) |
| **Linguistic scope** | Sulawesi focus (Celebic, South Sulawesi, Muna-Buton, SE Sulawesi) | Java-Bali-Madagascar focus (Western Malayo-Polynesian comparative) |
| **Non-linguistic evidence** | None — purely linguistic/computational | Botanical (Canarium/Plumeria), ritual (famadihana/slametan), epigraphic (DHARMA) |
| **ML component** | Central — the paper IS about the ML pipeline | None — cognacy comparison is descriptive statistics |
| **Script analysis** | Hanacaraka pangram phonological inventory (E036) — §4.5 of paper | Not discussed |
| **Audience** | Computational/historical linguists, Austronesianists | Southeast Asian studies generalists, historians, archaeologists |

---

## 3. Template Response to Reviewer

> **If a reviewer asks:** "This paper shares themes with [P9/a companion paper] on substrate detection. Is this salami-slicing?"

**Suggested response:**

"We appreciate the concern. This paper and [P9] operate at different levels of analysis with different methods, different geographic scopes, and different conclusions.

This paper (P8) is a **methods contribution**: it develops and validates a machine learning pipeline for detecting phonological non-conformity in Austronesian basic vocabulary, using Sulawesi languages as the primary test case. The paper's unique contributions are: (1) the two-model design separating circular from genuine substrate detection (Model A vs Model B); (2) the IPA and syllable robustness validations (E041, E042) proving the signal is phonological, not orthographic; (3) the Hanacaraka consonant reduction analysis (E036) providing independent script-historical evidence; and (4) the critical negative result that non-mainstream candidates do not form coherent word families (p = 0.569), reframing substrate detection as phonological non-conformity rather than shared lexical inheritance.

The companion paper [P9] is a **framework contribution** for Southeast Asian studies. Its linguistic component uses simple cognacy-rate comparison (no machine learning) to show that Balinese retains more PMP cognates than Javanese (+7.3 percentage points). This descriptive finding is one of four evidence streams (alongside botanical, ritual, and epigraphic data) supporting a general Peripheral Conservatism Framework. [P9] does not develop detection methodology; it applies existing comparative tools to a regional-studies question.

The papers are submitted to journals with non-overlapping readerships: this paper to Oceanic Linguistics (computational and historical linguistics audience) and [P9] to JSEAS (Southeast Asian area studies audience). The ML pipeline in this paper could, in principle, be applied to the varieties discussed in [P9] — that would be a natural future collaboration, not duplication.

We can provide both manuscripts to the editor for independent verification."

---

## 4. Journal-Audience Differentiation

| Feature | Oceanic Linguistics (P8) | JSEAS (P9) |
|---------|-------------------------|------------|
| **Disciplinary home** | Historical/computational linguistics, Austronesian language family studies | Southeast Asian studies (history, archaeology, area studies) |
| **Reader expects** | Formal linguistic methodology, IPA transcription, sound correspondences, statistical validation | Regional argumentation, historical narrative, multi-domain evidence synthesis |
| **What they would NOT expect** | Botanical substitution chains, famadihana/slametan ritual comparison | SHAP beeswarm plots, XGBoost hyperparameters, leave-one-language-out AUC |
| **Technical threshold** | High (ML literacy assumed, linguistic formalism expected) | Moderate (interdisciplinary audience, footnote-heavy narrative) |
| **Citation style** | Author-date (Chicago-like, natbib) | Chicago footnotes (verbose-note, biblatex) |
| **Figures** | SHAP beeswarm, quadrant comparison, cross-linguistic distance, expansion barplot | Cognacy gradient, Indianization wave, botanical layers, PCF convergence, organic civilization, domain heatmap |

---

## 5. Complementarity Statement

P8 and P9 are **complementary, not duplicative**:

- P8 provides the **detection tool** (ML phonological fingerprint). P9 provides the **interpretive framework** (peripheral conservatism explains where substrates survive).
- P8's Sulawesi focus and P9's Java-Bali-Madagascar focus have **zero geographic overlap** in primary datasets.
- P8's negative result (no shared substrate families) and P9's positive result (large-scale peripheral conservation) address **different questions**: P8 asks whether non-conforming forms share a common ancestor; P9 asks whether peripheral regions preserve older Austronesian forms better than centres.
- If both papers are accepted, they create a natural research programme: apply P8's ML pipeline to P9's peripheral varieties, testing whether phonological non-conformity rates differ between centres and peripheries. This is explicitly flagged as future work in both papers.

---

*Anti-salami-slicing protocol. Filed 2026-03-13.*

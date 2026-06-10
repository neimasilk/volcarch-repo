# DeepSeek critical review — P16 (Wacana submission) — 2026-06-08
Model: deepseek-reasoner

## Critical Review

The manuscript proposes a computational textual archaeology framework to detect genre-based taphonomic bias in Indonesian inscriptions. The author applies Sentence-BERT, BERTopic, and Monte Carlo convergence tests to 200 passages from external traditions and 173 DHARMA inscriptions. The central claim is that external traditions converge on volcanic themes while inscriptions systematically exclude them, revealing a “volcanic silence” attributable to genre selection.

Below, I identify the most serious weaknesses. Each is assessed for severity (FATAL, MAJOR, MINOR), whether it is fixable by revision, and how it undermines the stated conclusions.

---

### Weakness 1: The cross-tradition convergence test is trivial and does not support the central claim

**Critique**

The Monte Carlo convergence test compares the mean pairwise cosine similarity within a researcher-defined concept group to the mean similarity of random draws from the full corpus. The authors report that all eight groups “converge” with very high *z*-scores (e.g., *z* = 34.28 for *spice_trade*). However, this test is structurally incapable of demonstrating what the paper asserts—that “12 independent textual traditions converge semantically on Nusantaran themes” (Abstract) and that the convergence is “not an artefact of shared sources” (Research Question 1).

The test conflates topical coherence with cross-tradition convergence: if I tag 76 passages (many from Chinese sources) as *spice_trade*, their internal similarity will naturally be high because they share vocabulary like “clove,” “cinnamon,” “trade.” The null distribution (random draws from a corpus of 200 passages that includes many unrelated texts) will always yield lower similarity. This is **trivial**—it merely shows that within a topic, texts are more similar than across all texts. The test does **not** control for tradition of origin, translation style, or the fact that many passages within a concept group come from the same tradition (e.g., Chinese sources dominate *spice_trade*). The paper acknowledges possible “shared sources” but claims the Monte Carlo test addresses it because “the null distribution is random draws from the full corpus, not from the same tradition.” This is incorrect: the test structure has no mechanism to distinguish cross-tradition agreement from within-tradition homogeneity. A proper test would permute tradition labels or compare cross-tradition pairwise similarities to within-tradition similarities.

Because the entire cross-tradition corpus is built by the author (200 passages selected, tagged, and translated), the high convergence *z*-scores are largely an artifact of the curation process, not independent evidence of a shared Nusantaran theme across ancient traditions. The fundamental research question—whether external traditions independently confirm Nusantaran societies—remains unsupported.

**Severity**: FATAL  
**Fixable?** Not by simple revision; the test design is conceptually flawed. A new statistical framework (e.g., tradition-permutation test, cross-tradition pair analysis) would be required, and even then the small corpus and selection bias would remain.

---

### Weakness 2: The “volcanic silence” is an artifact of query formulation, not a robust taphonomic signal

**Critique**

The paper claims that the query “volcanic landscape, fire mountain” yields the lowest mean similarity to inscriptional content (0.244), while “mountain worship and sacred peaks” yields the highest (0.395). This is interpreted as a “volcanic silence” that demonstrates genre selection. But the difference may simply reflect the way the queries are constructed. The high-scoring query uses nouns like “worship,” “sacred,” “peaks,” which are frequent in dedicatory and ritual formulas. The low-scoring query uses “volcanic,” “fire,” “mountain”—terms that may not appear in the English translations of Old Javanese inscriptions, even when the inscription refers to an erupting mountain (e.g., using “*gunung api*” or “*meletus*”). The cross-lingual validation (Section 5.5) is supposed to address this, but the results are weak:

- XLM-RoBERTa-base collapses all embeddings to near-uniformity (mean pairwise similarity 0.997), making it useless; the reported Spearman correlation ρ = 0.452 with SBERT rankings stems from near-zero variance and is misleading.
- Multilingual SBERT (paraphrase-multilingual-MiniLM-L12-v2) gives a correlation of ρ = 0.336 between original Old Javanese and English similarity structures—a modest correlation, indicating that translation substantially alters the semantic space. Moreover, while “volcanic landscape” ranks 4/7 in Old Javanese (not bottom), the paper still claims “volcanic silence persists in original Old Javanese.” A rank of 4 out of 7 is not silence; it is intermediate.

Given that the multilingual SBERT was not specifically fine-tuned for Old Javanese (a low-resource language likely poorly covered in training data), the cross-lingual validation is insufficient to rule out the possibility that the observed low similarity is an artifact of the English query or translation.

**Severity**: MAJOR (could escalate to FATAL if the silence claim is the paper’s main contribution)  
**Fixable?** Partially. The authors could (a) extract volcano-related Old Javanese terms from the cross-tradition corpus and search for them directly in the inscription texts, (b) use a domain-adapted embedding model (e.g., fine-tuned on DHARMA texts), and (c) add multiple volcano-related queries in Old Javanese. Without these steps, the “silence” claim remains speculative.

---

### Weakness 3: The 929 CE discursive shift is based on an extremely small and fragile dataset

**Critique**

The diachronic BERTopic analysis uses only 46 dated inscriptions (33 pre-929, 13 post-929). BERTopic is designed for much larger corpora; with 46 documents, the topic model is unstable and the number of topics (3) is arbitrary. The chi-square test on the topic × period contingency table gives χ² = 16.58, p = 0.0003, but the expected frequencies are <5 in several cells (e.g., Topic 2 has 0 observations post-929, and Topic 1 pre-929 has only 4). Fisher’s exact test for the surge of “royal/political” discourse (Topic 1) yields OR ≈ 25, p = 0.0002, but this is based on 4 pre-929 vs. 8 post-929 documents—ridiculously small for such strong claims.

Furthermore, the temporal centroid drift shows the **largest shift** at the C11→C12 transition (0.366), not at 929 CE (0.208). The paper acknowledges this but still states “the 929 CE Mataram collapse marks a detectable discursive discontinuity” (Abstract). The textual discontinuity is not temporally aligned with the political collapse, undermining the causal claim.

Given the sample size, the “discursive shift” is at best suggestive; the paper’s strong language (“significant”, “confirms”, “surge”) is unjustified.

**Severity**: MAJOR  
**Fixable?** Yes: reduce the sample size limitations, avoid causal language, present the shift as a hypothesis rather than a conclusion. However, the small number of dated inscriptions is a data constraint that cannot be overcome without additional inscriptions.

---

### Weakness 4: Over-claiming and assuming the conclusion

**Critique**

The paper repeatedly states findings as if they prove genre-specific taphonomic bias. For example:

> “The inscriptional genre was designed for administrative and legitimatory purposes… The ‘volcanic silence’ … is not ignorance but genre-specific selection.”

But the evidence does not exclude simpler explanations:
- The low similarity for “volcanic landscape” may be due to the poor match between the English query and the actual content of inscriptions (see Weakness 2).
- The high similarity for “mountain worship” is expected because the word “mountain” appears in many dedicatory formulas—this does not prove that inscriptions “remember” the cosmological mountain while “forgetting” the physical one. Inscriptions might still mention volcanic eruptions using different vocabulary not captured by the query.
- The cross-tradition convergence could be driven by common reliance on Indian or Chinese sources (not independent verification), as many external traditions were influenced by earlier Greco-Roman or Indian geographies.

The paper assumes its conclusion (genre taphonomy) and then interprets the evidence to fit. The recursive bias argument (Section 6.3) is a narrative, not an empirical finding.

**Severity**: MAJOR (structural; over-claiming is pervasive throughout the discussion)  
**Fixable?** Yes, by rewriting the discussion to acknowledge alternative interpretations, equifinality, and the limitations of the current analysis.

---

### Weakness 5: Reproducibility and researcher degrees of freedom

**Critique**

The paper claims the pipeline is fully reproducible, but the corpus construction (200 passages from 12 traditions) is not described with sufficient detail. The selection criterion is “must contain a direct reference to Nusantaran geography, commodities, peoples, or polities” (Section 3.1), but how were passages identified? The phrase “systematic mining of published translations, augmented by AI-assisted extraction” is vague. The eight concept groups are researcher-chosen; the Monte Carlo test is run only on those groups. There is no pre-registration, and the authors could have explored many groupings and reported only the significant ones (they report all eight, but the choice of groups is still a degree of freedom). The BERTopic parameters (n_neighbors=15, min_cluster_size=4) are arbitrary and could affect results. The chi-square threshold for significance is set at 0.05, but multiple comparisons (8 concept groups, 7 queries, multiple BERTopic runs) are not corrected.

**Severity**: MAJOR  
**Fixable?** Yes: provide a detailed corpus-building protocol, justify parameter choices, apply multiple-testing correction, and ideally pre-register the analysis.

---

### Weakness 6: Scope fit for a humanities/Indonesian-studies journal

**Critique**

*Wacana* publishes work in the humanities of Indonesia—linguistics, philology, archaeology. This manuscript is overwhelmingly a computational methods paper, with no deep engagement with Indonesian philological traditions, historical context, or critical heritage studies. The interpretation of results is thin: the paper says inscriptions are “administrative and legitimatory,” which is well known to specialists. The only new claim—that this creates a taphonomic bias—is methodologically weak. The paper does not advance our understanding of Old Javanese epigraphy, the history of the Mataram kingdom, or volcanic risk perception. It is a demonstration of a pipeline on a small dataset, not a contribution to Indonesian humanities.

**Severity**: MINOR (the journal does accept digital humanities, but the contribution must be substantial to the field)  
**Fixable?** Partially: the authors should contextualize findings within existing epigraphic scholarship and provide concrete implications for Indonesian archaeology.

---

## Summary of key fatal flaws

1. **The Monte Carlo convergence test does not measure cross-tradition convergence**—it only measures within-group topical coherence. The central conclusion that “12 independent traditions converge” is unsupported.
2. **The “volcanic silence” is not robust**—cross-lingual validation is weak, and the query may not capture how volcanoes are described in Old Javanese.
3. **The 929 CE shift is based on N=46 inscriptions, with tiny subsamples**, making the chi-square and Fisher tests unreliable.
4. **The paper over-claims** consistently, treating methodological artifacts as proven taphonomic bias.

---

## RECOMMENDATION

**Reject**

The manuscript fails on methodological validity: the convergence test is trivial, the volcanic silence may be an artifact, and the diachronic analysis is underpowered. The core claims do not follow from the evidence. The contribution to Indonesian humanities is limited because the methods are not validated for the ancient texts.

---

## SINGLE MOST IMPORTANT FIX

Redesign the convergence test to actually measure cross-tradition agreement: for each concept group, compute pairwise similarities between passages from **different traditions** and compare to similarities between passages from the same tradition. Only this can support the claim of independent convergence. Without this fix, the paper’s main premise collapses.

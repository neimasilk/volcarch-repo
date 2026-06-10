# DeepSeek critical review — P16 (Wacana submission, R1-revised draft) — 2026-06-10
Model: deepseek-reasoner

# Peer Review Report

**Manuscript:** “What Ancient Texts Remember and Inscriptions Forget: Transformer-Based Evidence for Volcanic Taphonomic Bias in Indonesian Textual Records”

**Journal:** *Wacana* – Journal of the Humanities of Indonesia

**Reviewer stance:** critical, rigorous, skeptical

---

## General Assessment

The manuscript attempts an ambitious computational synthesis of external textual traditions and indigenous inscriptions from Nusantara. The central claim—that the inscriptional record exhibits a genre-driven taphonomic bias that systematically suppresses volcanic/geographical content—is interesting and plausible. However, the execution contains fundamental methodological flaws that undermine the key results. Three of the four research questions are answered with evidence that either is logically circular, rests on vanishingly small sample sizes, or conflates genre convention with taphonomic bias. The paper overclaims relative to what the evidence can support.

Below I identify the **most serious weaknesses**, with exact quotations, severity, and fixability assessment.

---

## Weakness 1: Cross-tradition convergence test is logically circular

### Critique

The Monte Carlo test for “cross-tradition semantic convergence” is structured as follows:

> “For each group we compute `S_cross`, the mean cosine similarity of pairs whose two passages come from *different* traditions, and compare it to a null distribution of random cross-tradition pairs drawn from the full corpus (5,000 bootstrap resamples).”

The eight concept groups (e.g., `spice_trade`, `volcano`) are researcher-defined tags applied to the 200 passages based on *their content*. The null distribution is drawn from the full corpus, which includes passages from *all* concept groups. It is therefore trivial that pairs within a tagged group have higher similarity than pairs drawn at random from the whole mixture of topics. This is **not a test of cross-tradition convergence**; it is a test that the tagging is internally consistent. The finding that `z_cross = 7.1` for `volcano` simply shows that passages the author labeled as “volcanic” have similar SBERT embeddings—which is what one would expect if the tags are coherent. The “tradition-controlled” aspect does not fix this: the test still compares within-group to across-group (across all topics), not to a properly permuted baseline that preserves the group structure while breaking tradition association.

The manuscript acknowledges this partly in the limitations (“researcher degrees of freedom”) but then proceeds to treat the result as evidence of “genuine cross-tradition agreement” (Section 5.1). *The test cannot distinguish between the hypothesis that twelve independent traditions independently describe volcanoes and the hypothesis that the researcher simply grouped all passages that sound volcanic.* The former is the conclusion drawn; the latter is the null that was never properly falsified.

**Exact claim attacked:**

> “All eight groups show significant cross-tradition convergence (Table 2): `S_cross` exceeds the random cross-tradition baseline in every case, with z from 3.6 (`metal_trade`) to 32.2 (`spice_trade`). … so volcanic awareness is a genuinely pan-tradition theme, not within-tradition homogeneity.”

**Severity: FATAL**

**Fixable?** Potentially fixable **if** a proper permutation test is performed (e.g., randomly shuffle tradition labels within each concept group and recompute `S_cross`, then compare observed `S_cross` to that null). Even then, the fact that the null is “random pairs from the full corpus” is inappropriate; the null should preserve the concept-group structure but randomize tradition membership. However, even a fixed test would still be weak because the eight concept groups are not independent (passages may belong to multiple groups?), and the researcher’s tagging is the sole source of grouping. The strongest evidence for **genuine** convergence would require a pre-registered set of concepts or an unsupervised clustering that recovers volcanic themes in external traditions and then shows their absence in inscriptions. The current design is **circular** in any case and cannot support the claim as stated.

Given that the central argument of the paper depends on cross-tradition convergence, this weakness is fatal as written. A major revision could partially rescue it, but the current evidence does not pass critical scrutiny.

---

## Weakness 2: The 929 CE “discursive discontinuity” rests on extremely sparse data and fragile statistics

### Critique

The diachronic BERTopic analysis uses only 46 dated inscriptions (33 pre-929, 13 post-929). BERTopic is applied to these 46 documents; it yields three topics, with counts so small that the manuscript itself notes “several expected cell counts fall below five” and resorts to Fisher’s exact test. The p-value claimed is \(p = 0.0003\), which looks remarkably precise for such tiny margins. Recalculating approximately: topic distribution is T0=23/5, T1=4/8, T2=6/0. A standard Fisher’s exact test on the 2×3 table (46 observations) will yield a p-value, but the number of possible tables is limited. The probability of observing exactly the counts in the table given the margins is not as extreme as 0.0003—I suspect the test was performed in a way that treats the table as unordered categories, but with cells of zero the test is unstable. Furthermore, the manuscript later reports temporal centroid drift showing that the **largest** semantic rupture occurs at the C11→C12 transition, not at 929 CE:

> “Because the discursive change is therefore *not* tightly aligned in time with the political collapse, we treat the association with 929 CE cautiously”

This directly contradicts the strong Fisher result. If the 929 CE boundary is not aligned with the maximum semantic shift, then the Fisher test is likely driven by the particular BERTopic clustering which may overfit to 46 documents. BERTopic with HDBSCAN on 46 documents is questionable; the min_cluster_size=4 (as per methods) means topics can be as small as 4 documents. The “disappearance” of T2 from 6 pre-929 to 0 post-929 cannot be tested as statistically significant in isolation (Fisher test on that 2×2 would yield p≈0.16, two-sided), yet the manuscript uses a single combined test across all three topics. The post-929 subset has only 13 documents, so any shift could be driven by a handful of inscriptions.

**Exact claim attacked:**

> “BERTopic detects a (sample-limited) discursive shift around the 929 CE Mataram transition (Fisher's exact \(p = 0.0003\), \(n = 46\)). … The overall chi-square test (\(p = 0.0003\)) and the Fisher's exact test for Topic 1 (\(p = 0.0002\)) are highly significant”

**Severity: FATAL**

**Fixable?** Structural/unfixable. **You cannot increase the sample size.** The DHARMA corpus has only 46 dated-and-translated inscriptions. No amount of clever statistics can rescue a strong causal claim about 929 CE from 13 post-929 documents. The paper itself undermines the result by showing the centroid drift does not align with 929. The appropriate conclusion is that the small sample cannot support any robust inference about a discursive shift at 929 CE. The claim should be retracted entirely or reduced to a speculative observation. Since the paper highlights this as a key finding (abstract, conclusions), it is a fatal weakness.

---

## Weakness 3: Overclaiming “volcanic silence” and taphonomic bias from a predictable genre difference

### Critique

The manuscript presents the low similarity of the “volcanic landscape fire mountain” query to the DHARMA corpus as evidence of “volcanic silence” and “taphonomic bias at the level of discourse, not stratigraphy.” But this is **exactly what one would expect** from the known function of inscriptions: they are legal, administrative, and commemorative texts, not geographical descriptions. The comparison with external traditions is **not** a controlled experiment—external traditions include travelogues, encyclopedias, and trade reports that *by their nature* describe geography and commodities. The “gap” between external and internal records is therefore not a discovery about Indonesia’s textual record; it is a restatement of genre differences. The paper frames it as a surprising finding:

> “Yet when we query the DHARMA inscriptional corpus for volcanic or landscape themes, we obtain the lowest similarity of any tested theme (0.244).”

This is not surprising; it is **predicted** by the nature of the corpus. The claim of “bias” implies insidious distortion, but genre selection is a conscious choice—inscriptions were not designed to describe volcanoes. Equifinality: there are many simpler explanations (genre convention, function of epigraphy, preservation of administrative rather than literary texts) that do not require invoking “taphonomic” mechanisms. The paper defines textual genre taphonomy as a bias, but it does not demonstrate that the absence of volcanic references is *caused* by a process analogous to sedimentary taphonomy. It is merely a description of what inscriptions contain.

Furthermore, the cross-lingual validation significantly weakens the volcanic silence claim in its strongest form:

> “volcanic discourse is intermediate (rank 4/7), not lowest, so the pronounced ‘silence’ seen in English is partly translation-dependent.”

Thus the strongest form of the claim (“lowest”) holds only in English translation, not in the original languages. The manuscript tempers this in the conclusion but the abstract and title still project more certainty.

**Exact claim attacked:**

> “Indonesia's textual record is shaped not only by physical survival but by genre selection—taphonomic bias at the level of discourse, not stratigraphy.”

**Severity: MAJOR**

**Fixable?** Fixable by revision: the paper should reframe the finding as “inscriptions under-represent volcanic geography compared to external accounts, consistent with their administrative function,” and drop the term “taphonomic bias” or define it more precisely. The claim that this is a form of bias that “operates not in the soil but in the stone” is poetic but not scientifically supported. A major revision can tone down the rhetoric, acknowledge the obvious genre explanation, and present the computational result as a confirmation of what historical common sense dictates, not a discovery. However, the rhetorical framing currently overclaims.

---

## Weakness 4: Reproducibility and data provenance issues

### Critique

The cross-tradition corpus of 200 passages is central to the convergence test. The manuscript states:

> “Passages were selected based on a single criterion: they must contain a direct reference to Nusantaran geography, commodities, peoples, or polities.”

This criterion is vague and subjective. Which translations were used? How were passages disambiguated? The corpus includes “Chemical” and “Linguistic” traditions that are not textual in any conventional sense—including archaeochemical residue descriptions and reconstructed proto-forms. These are **not texts** and cannot be expected to have the same semantic structure. The manuscript’s own BERTopic discovers topic 4 as “volcanic, Sanskrit, inscriptions, Javanese, Malay”—a “volcanic-linguistic nexus” that may be an artifact of mixing linguistic reconstructions with textual passages. Without a publicly available, independently verifiable corpus with full provenance (original language, translation source, passage boundaries), reproducibility is severely limited. The paper says the corpus is in a repository, but no DOI or stable identifier is provided; the github link is given, but the manuscript does not specify which version corresponds to the analysis.

The Monte Carlo bootstrap with 5000 resamples is described, but the random seed is not reported. The SBERT model (all-MiniLM-L6-v2) is deterministic, but the UMAP has a random component; no seed is given for UMAP or HDBSCAN. For a paper that claims reproducibility, these omissions are problematic.

**Exact claim attacked:**

> “The pipeline is fully reproducible and requires no training data. … The cross-tradition corpus (E089 v5) and all analysis scripts are available in the VOLCARCH repository”

**Severity: MAJOR**

**Fixable?** Yes, by providing citable, versioned data and complete code with random seeds. However, the subjective passage selection is not fixable without pre-registration.

---

## Weakness 5: The contribution is too purely computational for a humanities/Indonesian-studies journal

### Critique

*Wacana* covers the humanities of Indonesia—linguistics, philology, archaeology. The manuscript’s main contribution is a computational pipeline applied to ancient texts. The **archaeological or philological insight** is minimal beyond the already-known fact that inscriptions do not describe volcanoes. The paper does not engage substantively with the Old Javanese language, the historical context of specific inscriptions, or the philological debates about the 929 CE transition. The “volcanic taphonomy” argument is sourced from a separate, submitted paper (Amien 2026a). The present manuscript is essentially the computational appendix to that argument. For a Q2 humanities journal, the paper would need to demonstrate deeper domain expertise and not just deploy NLP tools. The results as presented are unlikely to persuade historians or archaeologists because the methodology is not grounded in traditional source criticism.

**Exact claim attacked:**

> “This paper represents the first transformer-based NLP applied to Old Javanese epigraphy, and the first cross-tradition semantic convergence test spanning 12 ancient traditions simultaneously.”

**Severity: MAJOR**

**Fixable?** Partially fixable by expanding the humanities analysis: discuss specific inscription examples, engage with the Old Javanese texts directly, and show that the computational results align with close reading. However, the paper as written is primarily a computational methods paper. If the journal is open to computational humanities, the methods must be flawless; they are not (Weakness 1). The novelty of applying existing NLP to a new corpus is low.

---

## Summary of Weaknesses

| # | Weakness | Severity | Fixable? |
|---|----------|----------|----------|
| 1 | Cross-tradition convergence test is circular (within-group vs. whole-corpus baseline) | FATAL | Fixable with correct permutation test, but still limited by subjective tagging |
| 2 | 929 CE analysis uses n=46, results fragile and contradicted by centroid drift | FATAL | Unfixable—sample size cannot be increased |
| 3 | Overclaiming “taphonomic bias” for predictable genre difference | MAJOR | Fixable by reframing and toning down |
| 4 | Reproducibility: subjective passage selection, missing seeds, no stable data version | MAJOR | Fixable with documentation |
| 5 | Contribution too computational for a humanities journal without deep domain engagement | MAJOR | Fixable with expanded philological context |

---

## Equifinality and Simpler Explanations

The most parsimonious reading of the results is:

- **Cross-tradition convergence** is an artifact of tagging: the author selected passages that already seem about the same topic. No new information is gained.
- **Volcanic silence** in inscriptions is expected because inscriptions are administrative texts; they do not describe landscape. The “silence” is not a bias that needs to be explained taphonomically—it’s a genre convention.
- **929 CE shift** cannot be reliably detected with 46 inscriptions. The largest semantic shift occurs later (C11→C12), so the association with 929 CE is a coincidence of small numbers.

The paper assumes the conclusion (inscriptions are biased) and uses circular statistics to confirm it.

---

## Recommendation

**REJECT**

The manuscript has one fatal weakness (circular convergence test) and one unfixable weakness (sample size for the 929 CE claim). Even if the convergence test could be repaired, the 929 CE result is untenable and the volcanic silence claim is trivial. The paper overclaims throughout. A major revision would need to remove the 929 CE diachronic analysis entirely, fix the convergence test, and reframe the paper as a methodological demonstration rather than a substantive discovery. As it stands, the evidence does not support the conclusions.

---

## Single Most Important Fix

**Retract all claims based on the cross-tradition convergence test as currently conducted.** The test does not demonstrate that twelve independent traditions independently confirm volcanic awareness; it shows only that the researcher’s tagging is coherent. Replace with a principled permutation test that compares the observed within-concept cross-tradition similarity to the distribution obtained by randomly shuffling tradition labels within each concept group. Even then, the claim would need to be drastically reduced: the best evidence for convergence would be unsupervised clustering that recovers volcanic themes without prior tagging. Without that, the central pillar of the paper collapses.

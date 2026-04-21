# Hostile Cross-Model Review — P1 — deepseek-chat

**Paper:** `D:\documents\volcarch-repo\papers\P1_taphonomic_framework\submission_v5.0.tex`
**Model:** `deepseek-chat`
**Target addendum:** `P1`
**Prompt tokens:** 11817
**Completion tokens:** 2689
**Elapsed seconds:** 124.6

## Review
**Referee Report**

**Manuscript Title:** The Volcanic Detection Horizon in Java: An Archaeological Puzzle and a Research Program
**Authors:** Mukhlis Amien & Go Frendi Gunawan
**Journal:** *Archaeological Research in Asia*

---

### 1. SUMMARY
The paper argues that cumulative volcanic sedimentation in Java has systematically buried pre-10th century CE archaeological surfaces to depths of several meters, rendering them invisible to conventional surface survey. This creates a "detection horizon" that could explain the apparent absence of early open-air sites compared to non-volcanic regions like Kutai in Kalimantan. The central contribution is not a validated geomorphological rate, but a framework and a proposed research program (involving OSL dating, tephrochronology, and coring) to test the hypothesis. The authors use four case studies of buried Hindu-Buddhist temples (Dwarapala, Sambisari, Kedulan, Kimpulan) to derive preliminary, site-level sedimentation rates of 2.4–6.2 mm/yr, which they project back to earlier periods.

### 2. OVERALL ASSESSMENT
**Major revision.** The core idea is provocative and potentially significant, but the paper in its current form is an unstable hybrid of a speculative hypothesis paper and a methods-driven geoarchaeological study. It leans heavily on computational data compilation and projection while its central claims require empirical, field-based validation it does not yet possess. It cannot be published without substantial restructuring and a major tempering of its conclusions.

### 3. MAJOR CONCERNS
1.  **The Conflation of Motivation with Evidence:** The paper uses the Kutai-Java contrast as a motivating puzzle but repeatedly implies it is evidence for the burial hypothesis (e.g., Abstract: "The oldest archaeological site... sits... in East Kalimantan, a region with zero active volcanoes."; Conclusion: "The Kutai–Java contrast... is the empirical heart of the argument."). This is a classic "argument from absence" fallacy. The absence of surface evidence in Java could be due to a dozen factors (different political organization, ritual practices, material culture, research intensity, site destruction). The paper acknowledges these as alternatives but then proceeds as if volcanic burial is the primary candidate needing testing. The structure unfairly loads the dice. **This matters** because it frames the entire paper around a false dichotomy. **To address:** The introduction must be rewritten to present volcanic burial as *one* possible taphonomic filter among many, not as the privileged explanation for the Kutai-Java disparity. The conclusion must not present Kutai as "empirical heart" of the burial argument.

2.  **The Fatal Flaw of Monument-Derived Rates:** The entire quantitative framework rests on sedimentation rates calculated from four stone monuments (*candi*). As noted in Section 5.6, monuments are terrible proxies for landscape aggradation. Their construction often involved digging foundation trenches and building on raised platforms or plinths. The "original ground surface" adjacent to a *candi* is archaeologically complex and not equivalent to the surface of an open habitation site. Sediment can accumulate against monument walls, creating artificially high depth measurements. **This matters** because it means the 2.4–6.2 mm/yr rate is likely an overestimate for the processes that would bury a village. Extrapolating this rate to pre-Hindu settlements is therefore methodologically invalid. **To address:** The authors must explicitly state that their rates are **only applicable to the specific taphonomic context of large stone monuments** and cannot be projected onto non-monumental sites without direct testing. The burial depth projections in Section 4.3 and Figure 2 should be removed or placed in a heavily caveated speculative section, not presented as "Results."

3.  **The Misuse of Liangan:** The authors present Candi Liangan as a "smoking gun" and "the strongest single counter-argument" to the idea that organic material decays before burial. This is a profound misreading. Liangan was buried by a **catastrophic** eruption, which the authors correctly distinguish from **cumulative** sedimentation. Catastrophic burial seals sites rapidly, creating anaerobic conditions that favor preservation. Cumulative burial over centuries allows for oxidation, bioturbation, and chemical weathering to destroy organics between depositional events. Liangan is evidence for the *potential* of volcanic preservation, but it is **irrelevant** to the central mechanism proposed in the paper. Citing it as a counter-argument is misleading. **This matters** because it uses a dramatic, well-preserved site to deflect a serious criticism of the proposed slow-burial mechanism. **To address:** The discussion of Liangan should be moved to a brief note on catastrophic preservation and explicitly decoupled from the argument about cumulative processes. It should not be used to support the plausibility of organic survival under the mm/yr regime.

4.  **The Illusory "Literature Compilation" as Validation:** The "51 eruption–site correlations" dataset (Section 3.4) is presented as a "partial check" and "consistency comparison." This is meaningless without critical evaluation. The authors admit it shares the same selection biases (monumental, construction-discovered sites) and source corpus as the primary anchors. It is not an independent line of evidence; it is a slightly larger sample of the same biased data. Calculating a mean depth from this compilation and finding it consistent with the anchor-site rate is circular, not confirmatory. **This matters** because it creates an illusion of cross-validation where none exists. **To address:** This section should be drastically reduced or removed. If retained, it must be framed not as validation, but as an illustration of the *existing biased sample* of buried sites, highlighting the need for the systematic research program proposed.

5.  **The "Research Program" as a Substitute for Analysis:** Section 5.6 is essentially an admission that the paper lacks the data to support its claims. Proposing future work is fine, but here it is used to deflect from the inadequacy of the current evidence. A research article should present research; a proposal should be a proposal. This manuscript tries to be both and succeeds as neither. **This matters** because it asks the reader to accept the hypothesis based on the promise of future data. **To address:** The paper must choose its genre. If it is a hypothesis/ framework paper, it should drop the quantitative projections and computational site compilations and focus on clearly articulating the theoretical problem of the volcanic detection horizon, using the case studies as qualitative examples. The research program should be the core of the paper's "contribution," not an appendix to weak evidence.

### 4. METHODOLOGICAL CONCERNS
*   **Statistical Methods:** The statistical approach is essentially non-existent. With n=4, calculating a mean and standard deviation (4.4 ± 1.2 mm/yr) is descriptively trivial and carries no inferential power. Presenting this as a "range" for projection is not a statistical model; it's simple arithmetic extrapolation.
*   **Evidence Channels:** The paper presents multiple "lines" of evidence (Dwarapala, three Merapi temples, 51-pair compilation, Kutai contrast, Liangan). As argued above, these are not independent. The Dwarapala and Merapi sites share the monument bias. The 51-pair compilation shares the source bias. Kutai is a contrast in region, not a measurement of process. Liangan is a different process. This is not multi-channel evidence; it is a single, biased channel viewed from slightly different angles.
*   **Comparisons & Confounders:** The comparison to non-volcanic Kutai is uncontrolled for massive confounders: culture, political economy, resource availability, and research history. The comparison between Kelud and Merapi rates lacks control for local topography (basin vs. slope), distance to vent, and eruption characteristics.
*   **Falsifiability:** The authors state the surface distribution cannot test the hypothesis (Section 4.4). This is a double-edged sword. While correctly identifying survivorship bias, it also makes the core claim—that something is buried—inherently difficult to falsify. A true test requires the subsurface work they propose. The paper itself is unfalsifiable.
*   **Reproducibility:** The computational methods for site compilation (Section 3.1) are reasonably transparent. However, the key input—the qualitative "half buried" observation for Dwarapala—is not reproducible from the information given. The archival source is not definitively cited or examined.

### 5. SPECIFIC CLAIMS TO CHALLENGE
*   **Claim:** "Cumulative volcanic sedimentation proceeds at 2.4–6.2 mm/yr in Javanese basins." **Assessment: (c) Overreach.** The data support this only for the immediate vicinity of specific stone monuments, not for basins.
*   **Claim:** "Pre-10th century surfaces in Java are now buried at depths of 4–10 m." **Assessment: (c) Overreach.** This is a direct extrapolation of the flawed monument-derived rates and is not supported.
*   **Claim:** "The surface distribution of known sites cannot inform us about pre-10th century settlement." **Assessment: (a) Well-supported.** This is a standard and correct point about archaeological visibility and survey bias.
*   **Claim:** "The Kutai-Java contrast is best explained by differential volcanic burial." **Assessment: (d) Unfalsifiable with current data / (c) Overreach.** It is one possible explanation, but the paper presents no evidence to privilege it over cultural, historical, or research-based explanations.
*   **Claim:** "Liangan demonstrates organic material can survive in the Javanese volcanic burial regime." **Assessment: (b) Plausible but under-supported for the paper's thesis.** It demonstrates survival under catastrophic burial, which is not the regime under discussion.

### 6. WHAT THE PAPER DOES WELL
1.  It clearly identifies a significant and under-theorized taphonomic problem in Southeast Asian archaeology: the potential impact of cumulative, non-catastrophic volcanic processes on archaeological visibility.
2.  It usefully compiles several known cases of buried temples (Sambisari, Kedulan, Kimpulan) in one discussion, highlighting a pattern that deserves more systematic geoarchaeological attention.
3.  The proposed research program (OSL, tephrochronology, coring) is, in principle, sound and is exactly the kind of work needed to move from speculation to science.

### 7. WHAT THE PAPER SHOULD BE, IF NOT THIS
This paper should be a focused, qualitative hypothesis paper. It should:
*   Title: "Cumulative Volcanic Sedimentation and the Archaeological Invisibility Problem in Java: A Research Framework."
*   Jettison the quantitative rate calculations and depth projections.
*   Use the Dwarapala and Merapi temple cases as *illustrative examples* of the process, not as quantitative anchors.
*   Clearly distinguish the catastrophic (Liangan, Pompeii) and cumulative (proposed) burial regimes.
*   Discuss the Kutai-Java contrast as a *motivating puzzle*, not as evidence.
*   Make the detailed, costed research program the central contribution. Outline what specific measurements (OSL profiles, cores) are needed, where they should be taken (e.g., in basin contexts away from known monuments), and what outcomes would confirm or refute the hypothesis.
*   Drop the computational site distribution analysis entirely; it's a distraction.

### 8. ONE QUESTION FOR THE AUTHORS
If you were to conduct the OSL and coring program you propose in a basin context (e.g., the Malang plain) at locations *away from all known monuments*, and you found no evidence of cultural material or paleosols at the depths predicted by your monument-derived rates, would you consider the cumulative burial hypothesis for open settlements to be falsified?

---

**Summary for Editor:** This referee recommends **Major Revision**. The manuscript tackles an important and overlooked taphonomic issue in Island Southeast Asian archaeology. However, it is fundamentally flawed in its current form. Its central quantitative argument relies on extrapolating sedimentation rates from buried stone monuments to entire landscapes, a methodologically invalid leap. It misuses the catastrophic site of Liangan to support a model of slow, cumulative burial and treats a biased literature compilation as independent validation. The paper is an uneasy mix of overreaching computational projection and an admission that real fieldwork is needed. To be publishable, it must be radically reframed as a qualitative hypothesis paper that clearly defines the problem, uses the monument cases as illustrations (not calibration), and places the rigorously detailed research program at its core, abandoning the unsupported quantitative projections. The reviewers' specific, major concerns must be addressed in full.

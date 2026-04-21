# Hostile Peer Review — P1 v2.0 JASREP

**Reviewed manuscript:** `submission_jasrep_v2.0.tex` (2026-04-20 state, post audit-#1 fix)
**Reviewer persona:** Senior Southeast Asian archaeologist with 25+ years Java field experience
**Produced by:** Claude (persona-adopted), simulating external review per `tools/hostile_reviewer_prompt.md`
**Status:** NOT a real peer review. A simulated pre-commit check. Real external review still required.
**Cross-model redundancy:** NOT RUN. Recommend also running on DeepSeek R1 + GPT-5 before submission.

---

## 1. SUMMARY

This manuscript calibrates cumulative volcanic sedimentation rates on Java using four archaeological sites whose construction dates are independently documented and whose burial depths were measured during rediscovery. The convergent mean (4.4 ± 1.2 mm/yr across Kelud and Merapi systems) is then used to project burial depths for successive historical periods, culminating in a predicted 4-10 m overburden for pre-400 CE remains in the Malang basin. The authors argue that the apparent chronological primacy of non-volcanic Kalimantan's Kutai polity over volcanic Java's Hindu-Buddhist record is a taphonomic artifact. A supplementary 51-pair eruption–site dataset is offered as independent validation of the rate.

The paper is well-written in its revised form and addresses a genuine gap in Indonesian archaeological methodology. The central contribution — the four-site convergence across two volcanic systems — is empirically grounded and, if the input data are reliable, defensible.

## 2. OVERALL ASSESSMENT

**Major revision.** The core result is publishable, but the paper is currently three arguments in one document, and the ancillary arguments are substantially weaker than the calibration result. The authors should either (a) sharpen this to a geoarchaeology paper on cumulative volcanic sedimentation, or (b) prepare a separate paper that properly develops the broader implications. Submitting this hybrid to JASREP invites structural criticism similar to what the EGQSJ editor raised.

## 3. MAJOR CONCERNS

### 3.1 The Dwarapala inference depends on a colonial observation whose specificity the authors overstate

The calibration rests heavily on the claim that Engelhard observed the Dwarapala statues in 1803 with "approximately half their bodies below ground surface," yielding 185 cm of accumulation over 535 years. I have two objections.

**First**, the primary source for this observation is cited vaguely. The paper says "BPCB Jatim" holds the relevant records; Kinney (2003) is cited for the statues themselves, not for the 1803 burial measurement. A Dutch colonial official's observation from 1803 about a cultic statue is subject to multiple interpretive possibilities: the lower half of the statue could have been placed into a carved pedestal or socle (common in Hindu-Javanese sculptural practice), the surrounding terrain may have been modified during construction, or the sediment accumulation may not be vertically uniform around the statue.

**Second**, even accepting Engelhard's observation, the inference that 185 cm represents *cumulative volcanic sedimentation* rather than a mixture of (i) pedestal design, (ii) post-abandonment construction debris, (iii) alluvial aggradation from the nearby Brantas tributaries, and (iv) genuine tephra accumulation is not defended with field-independent evidence. The paper acknowledges this in §3.2 ("total landscape aggradation in volcanic terrain, not pure primary tephra accumulation") and argues that total aggradation is what matters archaeologically. This is a fair move, but it broadens the claim from "volcanic sedimentation" to "Quaternary sedimentation in volcanic terrain" — a distinction the title, abstract, and cascade arguments do not preserve.

**Recommendation:** Either provide the archival source for the 1803 observation (cite the original OV report or the BPCB catalogue number) and defend the pedestal-free interpretation, or reframe the calibration as "total Holocene aggradation rate at four stone monuments."

### 3.2 The four calibration sites are not a random sample

All four sites are stone monuments. Monuments, by virtue of their stability and permanence, may preferentially accumulate sediment via eddy effects, human-modified micro-topography, and the persistence of the depositional surface. A thatched settlement or a rice paddy in the same landscape would not necessarily accumulate sediment at the same rate — it might experience periodic scouring, intentional raising of floor levels, or abandonment and reclamation. The paper's projection to "Javanese settlements" at 4-10 m depth assumes that the stone-monument calibration transfers to organic-settlement contexts. This is not demonstrated.

The 51-pair eruption-site dataset is offered as independent validation, but the authors do not state what fraction of those 51 sites are settlements versus monuments. If the 51 pairs are also dominated by monumental contexts, the "independence" is formal but not substantive.

**Recommendation:** Break down the 51-pair dataset by site type. If settlements show comparable burial depths, the monument-to-settlement transfer is supported; if not, the calibration applies only to monumental-context prediction.

### 3.3 The three "different papers in one" problem

Section 2.2 (demographic null hypothesis, now correctly arithmeticized to 1-2 million central estimate) is a significant claim that requires its own defense. Section 2.5 (West Java Buni/Batujaya within-island control) is an argumentatively powerful but archaeologically complex comparison. Section 5.5 (the 5-factor multiplicative cascade, with product 0.058%) is a theoretical framework whose relationship to the calibration data is indirect at best.

A geoarchaeology paper on volcanic sedimentation rates does not need a cascade model, a demographic null, and a comparative archaeology argument. Readers coming to this paper for the calibration will find these sections distracting. Readers coming for the "invisible civilization" argument will find the calibration underspecified for their purposes.

More concerning, each of these ancillary sections opens attack surfaces that are disproportionate to the calibration's value:
- The demographic estimate at 1-2M has uncertainty ranges that span an order of magnitude.
- The Buni/Batujaya comparison is weakened by the authors' own caveat that coastal-interior economic divergence is a confounder.
- The cascade model's "0.058% matches 0.031%" claim is, as the authors admit in the limitations, consistent with parameter uncertainty being larger than the claimed match.

**Recommendation:** Remove §2.2 (demographic), §2.5 (West Java), and §5.5 (cascade) to a companion synthesis paper. Refocus this paper on §3 (calibration), §4 (projection), and §5.3 (practical implications for fieldwork). The result would be a ~15 page contribution that the referee community can evaluate on its merits.

### 3.4 The cascade visibility formula is underdetermined

§5.5 presents a five-factor multiplicative visibility cascade:

P(visible) = P(not buried) × P(organic survival) × P(surveyed) × P(recognized) × P(published)

= 0.58 × 0.20 × 0.025 × 0.40 × 0.50 = 0.058%

The authors acknowledge these are "order-of-magnitude estimates" and cite "parameter uncertainty." What they do not acknowledge is that this is a five-parameter model with one observed data point (0.031%). Any of 2⁵ = 32 possible rearrangements of the five factors (each varied within its stated uncertainty range) produces a product that brackets the observation. The "match within a factor of 2" is not evidence; it is the capacity of the model.

The authors' internal work (cited as E115, E176, E178 in other documents I have seen) acknowledges this. The paper itself should acknowledge it more directly. The current language — "the product falls within a factor of two of the observed rate" — reads to a reviewer as "the model matches the data" when the honest statement is "the model's degrees of freedom allow it to match the data."

**Recommendation:** Reframe §5.5 as *diagnostic*, not *predictive*. The cascade's value is identifying that survey coverage has 40× leverage (the highest of the five factors), which directs future fieldwork investment. This is a defensible conclusion. The implied numerical match is not.

### 3.5 Pre-registered predictions are announced but not locked

The paper gestures toward falsifiability: "if 20 targeted GPR surveys at our highest-ranked locations find zero anomalies, the framework requires fundamental revision." This is in the spirit of Popper but the formulation is weak. "Fundamental revision" is not a commitment — the authors would retain the option to absorb a null result into the existing framework ("the cascade's F3 was underestimated; survey methods were insufficient"). In practice, nothing in §7 binds the authors.

Moreover, the prediction is conditional on funding ($40-100K, 2-4 weeks). The prediction is therefore a promissory note, not a test.

**Recommendation:** Register the 20 target coordinates (coarse to 500m grid per stated ethical protocol) as a supplementary file with a DOI. Commit, in text, to a specific threshold: "If the combined assemblage at the 20 locations yields fewer than 3 cultural anomalies at depths >1.5 m, the multiplicative cascade framework is falsified in the strong sense and will be withdrawn." This is stronger and more honest.

### 3.6 The "near-volcano site clustering" discussion is tautological

§4.4 and §5.2 take considerable space to explain why known sites cluster near volcanoes despite the framework predicting they should be harder to detect. The authors' resolution — that the dataset is composed of surviving monuments in intensively surveyed areas — is correct but it makes §4.4 into a defense against a straw man. A reader who understands the framework from §1-2 would never interpret the near-volcano clustering as contradicting the framework.

**Recommendation:** Compress §4.4 and §5.2 to a single paragraph. The space saved can go to deepening §3.3 (independent validation) or §5.3 (practical implications).

### 3.7 The "circular trap" issue is mentioned but not resolved

§4.4 identifies three reasons distribution data cannot test the framework: survivorship, survey, and discovery-mechanism bias. The conclusion is "the taphonomic bias hypothesis therefore cannot be confirmed or rejected from distribution data alone." This is honest but it means §3 and §4 of the paper do NOT test the framework they claim. The test is §5.3 (projections + fieldwork targets) + §7 (predictions). The intervening sections are scaffolding.

**Recommendation:** Either reorganize so that §3-4 are methods for §5-6, or explicitly flag §3-4 as "descriptive context" rather than "empirical test."

## 4. METHODOLOGICAL CONCERNS

### 4.1 Spatial autocorrelation

§3.7 does not address spatial autocorrelation. Volcanoes are spatially clustered; sites known to archaeology are spatially clustered in surveyed regions; the Spearman correlation reported (ρ = -0.955, p = 0.0008, n = 7 distance bands) is based on a sample size of 7 and makes no correction for spatial dependence. The authors' internal work (E184, cited elsewhere) reports Moran's I = 0.937 and flags this concern. The paper should do the same.

**Recommendation:** Add a paragraph to §5.6 acknowledging spatial autocorrelation (Moran's I = 0.937 per E184). Note that distributional comparisons (Mann-Whitney, KS) are more robust than regression-based statistics to this concern.

### 4.2 Independent validation claim is overstated

The 51-pair eruption-site dataset is described as "independent" from the four-site calibration. Independence requires that the 51 pairs were compiled from sources with no methodological or data overlap with the calibration sites. I believe this is approximately true, but the paper does not demonstrate it. A supplementary table listing the 51 pairs with their original literature sources would resolve this.

### 4.3 Compaction

The linear model D = R × (T_present - T_era) does not account for soil compaction. At depths >5 m, compaction of unconsolidated volcanic sediment can be 10-30%. The audit documented this as a "medium" priority fix; the current manuscript does not address it.

**Recommendation:** Add a paragraph to §3.6 acknowledging that the projections are upper bounds, with realized depths likely 10-30% shallower at depths >5 m due to sediment compaction.

### 4.4 Erosion on slopes

The rates derive from basin/plain contexts (Malang basin, Prambanan plain). On ridges and slopes, erosion may equal or exceed deposition, giving a net zero or negative aggradation. The projection "pre-Hindu at 4-10 m" is therefore a best-case scenario for targeted subsurface survey; on slopes, the same archaeological surface may be at the present ground level or eroded away entirely.

**Recommendation:** Add explicit terrain qualification to the projections: basins yes, slopes no.

## 5. SPECIFIC CLAIMS TO CHALLENGE

| Claim | Status |
|---|---|
| "4.4 ± 1.2 mm/yr mean rate across Java" | **Well-supported** (for the four calibration sites; generalization to "Java-wide" depends on assumption 3.2) |
| "The apparent chronological primacy of Kutai is a preservation artifact" | **Plausible but under-supported** (the Kalimantan-Java contrast has many confounders; within-island West Java comparison is stronger but out of scope for this paper) |
| "Pre-Hindu remains in Malang basin at 3.9-10.1 m depth" | **Well-supported** (given the calibration; honest upper bounds) |
| "The cascade model matches observation within a factor of 2" | **Overreach** (underdetermined; see §3.4) |
| "Low population density cannot explain the 1,000-7,000× gap" | **Well-supported** even at conservative population estimates, but the demographic argument deserves its own paper |
| "Stone temples protrude through volcanic sediment because they weigh thousands of tonnes" | **Overstated** (stone temples also survive because they are heavy, yes, but equally because they were built, maintained, and known by surrounding communities across the burial period — a cultural-continuity factor the paper does not raise) |
| "No systematic search has been conducted" | **Well-supported** |

## 6. WHAT THE PAPER DOES WELL

1. **The four-site calibration is a substantive contribution.** I cannot recall another paper in the Indonesian archaeology literature that compiles burial-depth measurements across multiple volcanic systems with this degree of care. The Merapi-Kelud convergence is a finding worth publishing on its own.

2. **The 51-pair independent validation dataset is clever.** Using colonial and volcanological literature to build a comparison set that does not overlap with the primary calibration is exactly the kind of cross-source triangulation this field needs more of.

3. **The "detection horizon" framing is practical.** §5.3's analysis of which eras are recoverable with which techniques (surface survey, GPR, deep coring) is directly useful for funding agencies and fieldwork planners.

4. **The paper is honest about what distribution data cannot prove.** §4.4's admission that the site-distribution analysis does not test the framework is the kind of epistemic discipline that makes me trust the authors even where I disagree with specific claims.

## 7. WHAT THE PAPER SHOULD BE, IF NOT THIS

A 15-page paper titled something like *"Multi-Site Calibration of Volcanic Sedimentation Rates in Java: Implications for Archaeological Detection Horizons."* Content:

- §1 Introduction: the Dwarapala opening, the calibration problem, scope: detection horizons for Javanese archaeology. **NO demographic argument. NO cascade.**
- §2 Background: volcanic geography of Java; catastrophic vs cumulative burial; why the monument-to-settlement transfer must be tested.
- §3 Methods: four-site calibration, 51-pair dataset, projection model, explicit limits.
- §4 Results: rate convergence, projections with terrain qualification.
- §5 Discussion: practical implications for GPR and coring; explicit pre-registered predictions; compaction and erosion caveats.
- §6 Conclusions: one paragraph on the calibration, one on future work.

Remove everything else. Submit the synthesis paper separately.

## 8. ONE QUESTION FOR THE AUTHORS

*Is the 1803 Engelhard observation of "approximately half the body below ground" documented in an extant primary source that I can consult, or is it a secondary inference from the present-day excavated height (370 cm) minus the visible colonial-photograph height? The answer substantially affects my confidence in the 185 cm / 535 year calibration that anchors the whole paper.*

---

**Summary for Editor:** The manuscript's central calibration result (4.4 ± 1.2 mm/yr across four sites and two volcanic systems) is a genuine contribution to the Indonesian geoarchaeology literature. However, the paper combines this calibration with a demographic null-hypothesis argument, a within-island comparative argument, and a five-factor theoretical cascade, producing a manuscript whose structural identity is unclear. I recommend major revision: the authors should remove the ancillary arguments to a companion synthesis paper and refocus this manuscript on the calibration and its detection-horizon implications. Several methodological clarifications are also needed (sources for the 1803 observation, monument-to-settlement transfer, compaction, spatial autocorrelation). With these changes, I would support publication.

---

## Post-Review Triage (for Pak Amien)

The hostile review flags **six actionable items** that should inform P1-core rewrite regardless of Path A/B/C:

| # | Item | Severity | Fix effort |
|---|---|:---:|:---:|
| 3.1 | Cite primary source for Engelhard 1803 observation (or acknowledge secondary inference) | HIGH | 2 hrs research + 1 paragraph |
| 3.2 | Acknowledge monument-vs-settlement transfer caveat more explicitly | MEDIUM | 1 paragraph |
| 3.3 | Remove demographic, West Java, cascade (Path B) | HIGH | 4-6 hrs surgical cut |
| 3.4 | Reframe cascade §5.5 as diagnostic not predictive (if kept) | HIGH | 2 paragraphs |
| 3.5 | Register target coordinates + specific falsification threshold | MEDIUM | 4 hrs data + supplement |
| 3.6 | Compress §4.4 and §5.2 near-volcano clustering discussion | LOW | 1 hr |
| 4.1 | Add spatial autocorrelation acknowledgment | LOW | 1 paragraph |
| 4.3 | Add compaction caveat | LOW | 1 paragraph |
| 4.4 | Add slope-erosion terrain qualifier | LOW | 1 paragraph |

**If Path A (Monday submit):** Minimum fixes are 3.1, 3.4, 4.1, 4.3, 4.4. Drop 3.3 (surgical cut) and 3.5 (registration). Estimated 6-8 hours.

**If Path B (2-week pause):** All items addressed. 3.3 becomes the defining action. Estimated 2-3 focused days for P1-core; hostile-review findings directly inform the restructuring.

---

*Hostile review produced 2026-04-20 per Mata Elang #15 recommendation.*
*This is a SIMULATED review. Real external review still required before submission.*
*Recommended next step: run the same prompt on DeepSeek R1 or GPT-5 for cross-model triangulation.*

# E158: Steelman Counter-Arguments for Cathedral Findings

**Status:** SUCCESS
**Date:** 2026-03-31
**Type:** [R] Adversarial / Quality control
**Papers:** All
**Addresses:** ME#12 §2A — "Echo chamber: write steelman counter-arguments"

## Purpose

For each of VOLCARCH's 5 strongest claims, write the strongest possible argument AGAINST it — as if written by a hostile but competent reviewer. Then evaluate whether the counter-argument can be refuted.

## Method

For each cathedral finding:
1. State the VOLCARCH claim
2. Write the strongest possible counter-argument (steelman)
3. Rate the counter-argument: WEAK / MODERATE / STRONG / DEVASTATING
4. Provide VOLCARCH's rebuttal
5. Rate residual vulnerability: LOW / MEDIUM / HIGH

---

## Finding 1: The 3,220x Demographic Gap (E108)

**VOLCARCH claim:** Java could have supported 590K-3.9M people pre-400 CE based on carrying capacity. The archaeological record shows only ~5 sites. This 3,220x gap cannot be explained by small population.

**Steelman counter-argument:**
> The carrying capacity model assumes agricultural populations. But pre-400 CE Java may have been sparsely populated by small-band hunter-gatherers at 0.01-0.1 people/km2, not sedentary agriculturalists at 5-30/km2. The Austronesian expansion brought agriculture to the Philippines and eastern Indonesia but may have bypassed Java's dense forests. If Java had only 1,000-10,000 hunter-gatherers (well below carrying capacity), 5 known sites is not anomalous — it's exactly what we'd expect for a low-density mobile population that leaves minimal archaeological footprint. The "gap" is between CARRYING CAPACITY and ACTUAL POPULATION, not between actual population and archaeological record. You're calculating the gap wrong.

**Strength: MODERATE**

**VOLCARCH rebuttal:**
- E122 tested this: even at hunter-gatherer density (0.1/km2), the gap is still 19x. Not 3,220x, but still a significant underrepresentation.
- Rice phytoliths at Liangan (~9th c.) and Sulawesi (3,500 BP, Deng et al. 2020) confirm agriculture existed in the region well before 400 CE.
- The Buni Complex (200 BCE) and Batujaya (2nd-5th c.) on Java's non-volcanic coast prove sedentary, complex societies existed in Java before 400 CE.
- Even if Java was purely HG, the East Java cave record should show continuous occupation (like Sulawesi's caves) — but it doesn't for pre-400 CE open-air sites.

**Residual vulnerability: LOW.** The gap shrinks at HG density but doesn't disappear. West Java evidence kills the "sparse HG" narrative for Java as a whole.

---

## Finding 2: E110 Cascade Model (0.058% predicted vs 0.031% observed)

**VOLCARCH claim:** Five independent factors multiply to predict 0.058% visibility, matching the observed 0.031% gap within 2x. Survey deficit (40x) is the dominant factor.

**Steelman counter-argument:**
> This is a tautology disguised as a model. You have 5 free parameters and 1 data point. ANY set of 5 numbers that multiply to ~0.0005 will "match" your data. You fitted the model to the observation, then called the fit "evidence." The cascade is not predictive — it's descriptive. The Monte Carlo robustness test (E115) only shows the model is robust to parameter perturbation — but it was designed to be robust by choosing factors that collectively produce the right answer. This is like fitting y = a*b*c*d*e to a single point and calling it validated. No reviewer should accept this.

**Strength: STRONG**

**VOLCARCH rebuttal:**
- E155 (NEW) partially addresses this: the cascade correctly predicts the RANK ORDER of archaeological visibility across 5 regions (Java < Sulawesi < Philippines < Bali < Japan, rho=1.0). This is a cross-regional prediction, not a single-point fit.
- Each factor has INDEPENDENT empirical support (F1: calibration sites, F2: E135 organic model, F3: E086 Japan comparison, F4: E157 Liangan, F5: E093 literature review).
- The factors were estimated BEFORE the product was computed — not reverse-engineered from the observed gap.
- However, the counter-argument is substantially correct: 5 parameters / 1 data point IS underdetermined. E155's cross-regional test helps but uses estimated observations.

**Residual vulnerability: MEDIUM-HIGH.** The cascade is intellectually compelling but statistically underdetermined. Papers should present it as "a framework consistent with observations" not "proof of the gap's origin." E155 helps but doesn't fully resolve the underdetermination.

---

## Finding 3: E069 Volcanic Signal Survives Survey Control (p=0.0015)

**VOLCARCH claim:** After controlling for survey intensity proxies (road distance, BPCB proximity), volcanic proximity retains a significant independent effect on site density (quasi-Poisson beta=-0.477, p=0.0015).

**Steelman counter-argument:**
> Your survey intensity proxies (road distance, BPCB proximity) are WEAK proxies for actual survey effort. Real survey effort data would include: number of archaeological surveys per grid cell, years of excavation permits, number of published reports per area. Using road distance as a proxy for "how much archaeology has been done here" conflates modern infrastructure with archaeological research. The residual volcanic effect might simply reflect unmeasured variation in survey effort — perhaps volcanic areas are harder to access, more densely forested, or less politically interesting to survey. You controlled for the wrong variable.

**Strength: MODERATE-STRONG**

**VOLCARCH rebuttal:**
- The proxy critique is valid — road distance IS a weak proxy. But it's the BEST available proxy given Indonesia's lack of systematic survey effort data.
- E086 (Japan comparison) provides independent evidence: Japan surveys 100-200x more intensively and HAS rich volcanic archaeology. The survey deficit explanation is cross-nationally validated.
- E129 (73% temple bias) shows that what Indonesia surveys is primarily stone temples — which are visible on the surface. This explains the survey deficit without needing survey effort data.
- The ideal test would use actual survey coverage maps from BPCB. These don't exist digitally.

**Residual vulnerability: MEDIUM.** The proxy weakness is acknowledged but unavoidable. The finding is consistent with multiple independent lines of evidence, but the specific p=0.0015 should be qualified with proxy limitations.

---

## Finding 4: E085 Substrate Signal vs Noise (z=11.05, p<0.0001)

**VOLCARCH claim:** The phonological substrate detected by E027's ML classifier (AUC=0.762) is a real linguistic signal, not an artifact of noise or chance. Permutation testing shows the AUC is 11 standard deviations above random.

**Steelman counter-argument:**
> The classifier detects phonological DIFFERENCES between western Indonesian languages and PMP reconstructions. You call this "substrate" — but it could equally be INNOVATION. Languages change over time. Western Indonesian languages have been spoken for 4,000+ years in contact with each other and with non-Austronesian populations. Any systematic difference from PMP could be: (a) substrate from pre-Austronesian populations, (b) shared innovation due to contact, (c) parallel drift in similar environments, or (d) borrowing from Sanskrit/Arabic/Chinese that you haven't fully controlled for. E029 already showed that the signal is "parallel innovation, not shared substrate." Your own experiment undermines the substrate interpretation.

**Strength: MODERATE**

**VOLCARCH rebuttal:**
- E029's finding was that the CLUSTERING pattern is parallel innovation — languages don't share a single substrate, they innovate in parallel. But this doesn't mean no substrate exists — it means multiple local substrates.
- E107 (ADV-5 resolution) showed that C5 (Iban+Malay) detects Mon-Khmer substrate, not noise. 6/6 Mon-Khmer predictions confirmed. The classifier IS detecting real substrate when substrate exists.
- The Sanskrit/Arabic/Chinese confound is controlled: E027's features are purely phonological (consonant counts, syllable structure), not lexical. Sanskrit borrowings don't change phonology.
- The "innovation vs substrate" distinction matters for interpretation but not for the statistical finding: SOMETHING caused systematic phonological deviation from PMP in western Indonesia. Whether that "something" is substrate or innovation, it's a real signal.
- P8 already reframes as "phonological non-conformity" rather than "substrate detection" per ADV-5 guidance.

**Residual vulnerability: LOW-MEDIUM.** The statistical signal is unassailable. The INTERPRETATION as "substrate" (vs "innovation") is where vulnerability lies. P8's reframing mitigates this.

---

## Finding 5: E066 Candi Equinox Orientation (p=4.9e-14)

**VOLCARCH claim:** 85% of Java's 142 candi face equinox directions (east/west), far exceeding chance (binomial p=4.9e-14). This reflects Hindu canonical practice, not volcanic awareness.

**Steelman counter-argument:**
> This is trivially true and contributes nothing to VOLCARCH's thesis. Hindu temples face east. Hindu temples in Java face east. Congratulations, you've proven Hinduism existed in Java — something known since the 19th century. The p-value is impressively small but the finding is impressively obvious. More importantly, this finding actually UNDERMINES VOLCARCH: if candi orientation is purely canonical (religious), then candi SITING near volcanoes is also likely canonical (sacred mountains), not "volcanic awareness." You can't claim siting is adaptive if orientation is purely ritual. The finding proves the standard interpretation: Hindu canonical practice drove all candi placement decisions, end of story.

**Strength: STRONG (as irrelevance critique)**

**VOLCARCH rebuttal:**
- The "trivially true" critique has merit — this finding IS expected. Its value is as a CONTROL: it establishes that one aspect of candi placement (orientation) follows canonical rules, while another (siting) shows volcanic-specific patterns. The contrast is informative.
- E031 split: orientation is canonical (east, p=4.9e-14) but siting clusters WEST of volcanoes (Rayleigh p=3.4e-8). If everything were canonical, siting would be random relative to volcanoes. The non-random siting pattern + random-relative-to-volcanoes orientation is the finding.
- The "sacred mountains" counter is E056's domain — candi cluster in MORE Sanskrit toponymy areas, suggesting Indianization drove candi construction, not volcanic awareness per se. But E065 shows Zone A (closest to volcanoes) is 17.9x overrepresented — this cannot be explained by canonical practice alone.
- This finding is NOT central to VOLCARCH's thesis. It's supporting evidence, not cathedral.

**Residual vulnerability: LOW (for VOLCARCH overall) / HIGH (for this specific finding's novelty).** The finding is real but obvious. Its value is as a control, not a contribution.

---

## SYNTHESIS: Vulnerability Assessment

| Finding | Counter Strength | Residual Vulnerability | Action Needed |
|---------|-----------------|----------------------|---------------|
| E108 Gap 3,220x | MODERATE | LOW | Emphasize E122 (HG density still 19x gap) |
| E110 Cascade 0.058% | STRONG | MEDIUM-HIGH | Frame as "framework consistent with" not "proof." Cite E155 cross-regional. |
| E069 Survey control p=0.0015 | MODERATE-STRONG | MEDIUM | Acknowledge proxy weakness explicitly in papers |
| E085 Substrate z=11.05 | MODERATE | LOW-MEDIUM | Keep P8's "non-conformity" reframing |
| E066 Equinox p=4.9e-14 | STRONG (irrelevance) | LOW (for VOLCARCH) | Don't lead with this finding. Use as control. |

### Overall Assessment

**VOLCARCH's weakest flank is the cascade model (E110).** The counter-argument that 5 parameters / 1 data point is curve-fitting is substantially correct. E155's cross-regional test helps but uses estimated parameters and observations — it's not independently calibrated. The strongest defense is that each factor has independent empirical support, but the product's match to the data is not as impressive as it looks.

**VOLCARCH's strongest claims are the cathedral findings** (E066, E051, E084, E085, E069) that survive any statistical correction and have clear, simple interpretations. The cascade model should be presented as an EXPLANATORY FRAMEWORK that organizes these findings, not as independent evidence.

**Recommendation for P17:** Lead with cathedral findings (E084 post-929 shift, E105 Two Javas pattern). Present cascade model as organizing framework in Discussion, not as core evidence. This makes the paper resilient to the "curve-fitting" critique.

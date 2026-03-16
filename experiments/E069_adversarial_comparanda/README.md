# E069: Adversarial Experiment Suite — Falsification Tests

**Status:** SUCCESS (ADV-3 executed; ADV-1/2/4/5 in separate E-dirs)
**Date:** 2026-03-13
**Type:** ADVERSARIAL / FALSIFICATION
**Papers:** ALL (thesis-level test)

## Purpose

Design and execute experiments that could genuinely DISPROVE the VOLCARCH thesis. The project has been criticized (Mata Elang #6 structural critique) for having no adversarial experiments — all 67 prior experiments were designed to find supporting evidence.

## Four Adversarial Experiments

### ADV-1: Volcanic Comparanda ("Why can Japan/Italy find their sites?")

**H0:** Volcanic sedimentation does NOT prevent archaeological discovery. Japan (~460,000 sites, ~8,300 excavated/year), Italy (Pompeii/Herculaneum), and Mesoamerica (Joya de Ceren) have rich records despite equivalent volcanism.

**Method:**
1. Compute site density per km² in 0-25km, 25-50km, 50+ km bands around active volcanoes for Java, Japan, and Italy
2. Normalize by survey intensity (excavations/yr/km²)
3. Compare ratios across regions

**Falsifies VOLCARCH if:** Site density near volcanoes is comparable everywhere after controlling for survey intensity → volcanism is not the cause, survey effort is.

**Supports VOLCARCH if:** Deficit is universal (confirming burial) but dramatically worse in Indonesia due to multiplicative interaction: volcanism × low-survey = invisibility.

**Data:** GVP volcanoes, NABUNKEN (Japan), E001 (Java). **Effort:** 2-3 weeks.

---

### ADV-2: Non-Volcanic Control Islands ("Does the gap exist without volcanism?")

**H0:** The pre-4th century gap exists equally in non-volcanic Kalimantan and Sulawesi → volcanism is not the primary cause.

**Method:**
1. Compile pre-4th century CE sites across Indonesian islands
2. Normalize by area and survey terrain
3. Compare gap severity: volcanic Java vs non-volcanic Kalimantan vs partially-volcanic Sulawesi

**Falsifies VOLCARCH if:** Gap equally severe in non-volcanic Kalimantan → problem is tropical perishability, not volcanism.

**Supports VOLCARCH if:** Non-volcanic regions show denser pre-4th century sequence despite lower survey investment.

**Data:** BPCB registries, published literature. **Effort:** 3-4 weeks.

**NOTE:** This is the MOST DANGEROUS experiment. Preliminary evidence suggests Kalimantan has almost no open-air settlement archaeology from ANY period, which could reduce VOLCARCH's claim from "volcanic bias" to "tropical taphonomic bias" — weaker and less novel.

---

### ADV-3: Survey Intensity Sufficiency ("Does survey effort explain everything?")

**H0:** Site distribution is FULLY explained by survey intensity. No residual volcanic signal exists.

**Method:**
1. Model: site_density ~ survey_proxies + volcanic_proximity
2. Nested model comparison (likelihood ratio test)
3. If survey-only R² > 0.90 and adding volcanism adds nothing → volcanic explanation unnecessary

**Falsifies VOLCARCH if:** Survey intensity alone explains >90% variance; volcanic proximity adds nothing (p > 0.05).

**Supports VOLCARCH if:** Residual volcanic signal after controlling for survey effort.

**Data:** E001 sites + road_dist (E013) + new survey intensity proxies. **Effort:** 2-3 weeks. CHEAPEST to run (most data exists).

---

### ADV-4: Linguistic Substrate Noise Test ("Is the substrate real or statistical noise?")

**H0:** The "pre-Indic substrate" detected by E022-E029 is false positives, Austroasiatic loans, or retained PAN vocabulary.

**Method:**
1. Apply substrate pipeline to NEGATIVE CONTROL (two related languages with no known substrate)
2. Apply Bonferroni correction for number of items tested
3. Cross-reference survivors against known Austroasiatic loanword lists
4. Test for regular sound correspondences (genuine substrate should show regularity)

**Falsifies VOLCARCH if:** False positive rate equals detection rate; OR >80% of candidates are known Austroasiatic loans.

**Supports VOLCARCH if:** Significant excess over FP rate; semantic domain clustering; regular sound correspondences.

**Data:** ABVD (existing), Austroasiatic loanword lists (Hoogervorst 2015, Brill 2023). **Effort:** 3-4 weeks.

---

## Execution Priority

1. **ADV-3 (Survey Intensity)** — cheapest, most data exists, directly addresses known weakness
2. **ADV-1 (Volcanic Comparanda)** — Japan data publicly available, addresses "why Java but not Japan?"
3. **ADV-2 (Non-Volcanic Control)** — most dangerous, hardest data, could fundamentally reshape thesis
4. **ADV-4 (Linguistic Noise)** — targets L4, methodological refinement

## Decision Framework

If ADV-3 FAILS (survey explains everything): Reframe thesis from "volcanic burial" to "under-surveyed volcanic zone" → weaker but still actionable claim → P1/P2 survive as methodology papers, P7 weakened.

If ADV-2 FAILS (gap exists everywhere): Pivot from "volcanic taphonomic bias" to "multi-factor tropical taphonomic bias" → less novel, broader scope → consider whether this strengthens or weakens the manifesto.

If both PASS: VOLCARCH thesis significantly strengthened → strongest possible revision defense.

---

## Execution Results

### ADV-3: Survey Intensity Sufficiency — **PASSED** (executed in this directory)

- Quasi-Poisson GLM: volcanic proximity beta = -0.477, **p = 0.0015**
- After controlling for survey intensity (road distance, BPCB proximity), volcanic signal PERSISTS
- Delta pseudo-R² = 0.016 (small but significant independent contribution)
- **Verdict:** VOLCARCH thesis survives survey intensity control

### ADV-1 through ADV-5: Full Scorecard

| Test | Experiment | Result | Key Statistic |
|------|-----------|--------|---------------|
| ADV-1 Japan comparanda | [E086](../E086_adv1_japan_comparanda/) | **PARTIAL** | Japan 100-200× more survey; Kikai-Akahoya IS VOLCARCH-type |
| ADV-2 Non-volcanic control | [E081](../E081_adv2_nonvolcanic_control/) | INCONCLUSIVE | Fisher p=0.760, cave bias universal |
| ADV-3 Survey intensity | E069 (this dir) | **PASSED** | p=0.0015, volcanic signal survives |
| ADV-4 Substrate noise | [E085](../E085_adv4_substrate_noise/) | **PASSED** | p=0.0000, z=11.05, AUC 11 SD above random |
| ADV-5 Negative control | [E087](../E087_substrate_negative_control/) | GREY ZONE | C5 AUC=0.713 nearly matches Sulawesi |

**Overall:** 2 PASSED, 1 PARTIAL (survives with constraint), 1 INCONCLUSIVE, 1 GREY ZONE. No outright failures.

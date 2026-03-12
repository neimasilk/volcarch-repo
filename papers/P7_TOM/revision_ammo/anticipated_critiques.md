# P7 Revision Ammo: Anticipated Critiques & Pre-Computed Responses

**Paper:** "A Temporal Overlay Matrix for Indonesian Archaeological Site Distribution" (Short Communication)
**Journal:** Antiquity Project Gallery (Q1)
**Author:** Mukhlis Amien (single-author)
**Prepared:** 2026-03-12

---

## Critique 1: "The spatial segregation could be explained by survey bias alone, not taphonomic processes"

**Anticipated from:** Archaeologist who attributes gaps to insufficient survey rather than burial.

**Response:**
"We explicitly control for this. Survey accessibility (road distance, population density, terrain ruggedness) is included in our analysis. The key finding is that Zone B sites average 16 km from nearest volcano versus Zone A sites at 43 km (Cohen's d = 1.005, large effect) — this is a *volcanic proximity* gradient, not a survey coverage gradient. If the pattern were pure survey bias, we would expect it to correlate with road access and urbanization, not volcanic distance.

Furthermore, deep-time sites (>5000 BP) cluster exclusively in karst and river terrace contexts (E019) — geological environments that preserve against volcanic sedimentation. The absence of open-air deep-time sites in volcanic landscapes is predicted by taphonomic models but not by survey-bias-only models."

**Supporting data:** `experiments/E019_spatial_distribution/README.md`

---

## Critique 2: "Sample size is too small for the claims made"

**Anticipated from:** Quantitative reviewer.

**Response:**
"The Project Gallery format constrains us to ~2,000 words and a focused argument. Our initial dataset (E020 Mini-NusaRC v2) contains 48 sites, which we acknowledge is a subset of the full Indonesian archaeological record.

However, the effect size is LARGE (Cohen's d = 1.005), which is robust even at modest sample sizes. The binomial probability of deep-time sites appearing exclusively in karst/river contexts by chance, given the proportion of karst landscape in Java, is < 0.01.

We frame this as a pilot demonstrating the *method* (Temporal Overlay Matrix) and the *signal* (volcanic proximity gradient), not as a definitive quantification. Full-scale analysis requires a complete NusaRC-style radiocarbon database for Indonesia — which does not yet exist."

---

## Critique 3: "This is just P1/P2 again — what's new?"

**Anticipated from:** Reviewer who has seen the other submissions (unlikely but possible if same reviewer pool).

**Response:**
"P7 makes a different argument than P1 or P2:

- **P1** quantifies volcanic sedimentation RATES (mm/yr) and establishes the taphonomic mechanism.
- **P2** builds a PREDICTIVE MODEL for site locations using machine learning.
- **P7** demonstrates the TEMPORAL dimension: that the absence of deep-time sites in volcanic zones is a systematic pattern visible in site distribution data, independent of any model.

P7's contribution is the Temporal Overlay Matrix as a *diagnostic tool* — a simple, replicable method that any archaeologist can apply to their region's site database to test for taphonomic bias. It requires no ML expertise, no sedimentation calibration — only site coordinates, dates, and a geological map."

---

## Critique 4: "The comparison with Sulawesi/Kalimantan is unfair — different research histories"

**Anticipated from:** Reviewer noting that Java has more archaeological research investment.

**Response:**
"This is a valid concern, and precisely our point. Java has received MORE archaeological attention than Sulawesi or Kalimantan, yet has FEWER deep-time sites per surveyed area in volcanic zones. If research investment were the primary factor, Java should lead — it does not.

The Sulawesi comparison is instructive: Leang Bulu Sipong 4 (≥44,000 BP), Leang Tedongnge (≥45,500 BP), and Leang Panninge (~7,300 BP) are all from karst contexts — the same pattern as Java's deep-time sites. Deep time is preserved in KARST, regardless of research intensity. This is geology, not survey effort."

---

## Critique 5: "Antiquity Project Gallery is a short communication — the claims need a full paper"

**Anticipated from:** Reviewer who feels the argument is compressed.

**Response:**
"We agree that the full argument requires more space. The Project Gallery contribution is deliberately constrained: it introduces the TOM method, demonstrates the volcanic proximity gradient, and establishes the deep-time karst exclusivity pattern. Each of these three results stands alone as a short-communication contribution.

The full quantitative treatment appears in companion papers:
- Sedimentation calibration → [P1, Asian Perspectives, MS# 019A-0326]
- Predictive model → [P2, JCAA, Submission #280]
- Peripheral cultural preservation → [P9, JSEAS, submitted 2026-03-11]

We cite these as 'in review' and can update citations upon acceptance."

---

## Critique 6: "The Dwarapala/Sambisari examples are anecdotal"

**Anticipated from:** Quantitative reviewer who wants statistics, not case studies.

**Response:**
"In a Project Gallery format, specific exemplars serve as concrete illustrations of the pattern. Sambisari (buried 5m, discovered 1966), Kedulan (buried 2.7m), and Liangan (buried 4m by Kelud eruption) are not cherry-picked — they are the ONLY known buried temple sites in Java, and all three are in the volcanic proximal zone predicted by our model.

The statistical pattern is in the Zone A/B comparison (Figure 2, Cohen's d = 1.005) and the deep-time karst exclusivity (all 6 sites >5000 BP in non-volcanic contexts). The case studies make the statistical pattern tangible for a general archaeological audience."

---

## Additional Revision Resources

### Available Supporting Experiments

| Resource | What it adds |
|----------|-------------|
| E019 (Spatial Distribution) | Full statistical analysis of site distance-to-volcano by time period |
| E020 (Mini-NusaRC) | Expanded 48-site dataset with informative negative (cave bias universal) |
| E016 (Zone Classification) | Zone B map — high probability, zero sites, GPR targets |
| E040 (Bamboo Civilization) | 63.4% organic material culture → explains why surface survey finds nothing |

### Cross-Paper Reinforcement

- **P1 → P7:** Sedimentation rates provide mechanism for WHY volcanic-zone sites are absent.
- **P2 → P7:** Predictive model independently identifies same Zone B cells.
- **E031 → P7:** Candi western clustering shows builders KNEW about tephra direction — cultural awareness of the very process that buries sites.

### Alternative Framing (if reviewer wants different emphasis)

- Could reframe from "diagnostic tool" to "call for Indonesian radiocarbon database" — the real bottleneck is the absence of an Australian-style NusaRC for Indonesia.
- Could add one paragraph on the Borneo bypass hypothesis (Sulawesi → Borneo → Java route for early settlement) as explanatory framework for why eastern Indonesia has more deep-time sites.

---

*Prepared 2026-03-12. Use when reviewer comments arrive.*

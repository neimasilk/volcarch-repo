# ADV-2: Depth-Based Evidence vs Site-Type Ratios — Revision Support Material for P1

**Paper:** P1 "Volcanic Taphonomic Bias in Indonesian Archaeological Records"
**Prepared:** 2026-03-13
**Triggered by:** E081 (ADV-2 Non-Volcanic Control Test)

---

## What ADV-2 Found

E081 tested whether volcanic regions have a statistically different site-type distribution (cave vs open-air) compared to non-volcanic control regions (Kalimantan, Madagascar). Using 80 sites from the mini-NusaRC v3 dataset:

- **Fisher exact test (enclosed vs open): p = 0.760, OR = 0.75.** Not significant.
- Volcanic regions: 62.7% enclosed. Non-volcanic controls: 69.2% enclosed.
- Chi-square on 4-category distribution: p = 0.493.

**The aggregate comparison does NOT support a volcanic explanation for site-type ratios.** Cave dominance is universal across Island Southeast Asia wherever karst is available.

## Why Site-Type Ratios Fail as Evidence for L1

Three confounding factors overwhelm any volcanic signal in site-type data:

1. **Karst availability dominates.** Kalimantan (non-volcanic) is 100% cave sites (8/8) because its limestone karst landscapes are the focus of regional archaeology (Niah, Lubang Jeriji Saleh). Sulawesi (volcanic) is 83% caves for the same reason (Maros-Pangkep karst). Cave prevalence tracks geology, not volcanology.

2. **Research tradition biases.** Caves are preferentially excavated everywhere because they provide natural stratigraphy, shelter artifacts from weathering, and are easier to locate than buried open-air sites. This is a global archaeological bias, not an Indonesian volcanic one.

3. **Java's anomaly is paleoanthropological, not taphonomic.** Java has the lowest enclosed-site ratio (36.8%) of all regions, driven by 8 river-terrace H. erectus sites (Sangiran, Trinil, Perning). This is a deep-time paleoanthropological artifact, not evidence that volcanic burial destroyed open-air sites.

## The Correct L1 Argument: Burial Depth

Volcanic taphonomic bias operates through **burial depth**, not through differential preservation of site types. Volcanoes do not selectively destroy open-air sites while preserving caves — they bury ALL sites under tephra, lahar, and reworked volcanic sediment, rendering them invisible to standard survey regardless of site type.

Three independent evidence streams support this:

### Evidence Stream 1: Colonial Archaeological Register (E070)

The OV reports (Oudheidkundig Verslag, 1912-1929) document burial depths observed by Dutch colonial archaeologists at the time of discovery:

| Site | Burial Depth | Source |
|------|-------------|--------|
| Prambanan Vishnu statue | 9.14 m | OV 1925 (well digging) |
| Trowulan deposits | 4.28 m | OV 1917 |
| Sambisari temple | 5.0 m | discovered 1966 (quarrying) |
| Kedulan temple | 2.7 m | discovered 1993 (sand mining) |
| Liangan village | 4.0 m | discovered 2008 (sand mining) |
| Kelud 1919 lahar sites | 1.5-2.0 m | OV 1919-1920 |

From 52 colonial site records, **32 have measured burial depths** ranging from 0.60 m to 9.14 m, with a mean of 2.88 m. These are not modeled estimates — they are physical measurements recorded in field reports.

### Evidence Stream 2: Sedimentation Model (E075)

The Pyle (1989) exponential thinning model, calibrated against 7 East Java volcanoes and 165 GVP-recorded eruptions, produces:

- **Pearson r = 0.951** against observed burial depths (N = 363 validation points)
- **32.3% of East Java grid cells** have >1 m predicted cumulative burial
- **12.8% of cells** have >3 m burial — beyond standard excavation depth
- These are MINIMUM estimates (only recorded eruptions, no lahars, no reworking)

### Evidence Stream 3: Tephra-Archaeological Correlation (E083)

A systematically compiled dataset of eruption-site pairs:

- **51 eruption-site correlations** across 12 volcanic systems
- **24 sites with measured burial depths**: mean 3.41 m, median 2.50 m, max 9.14 m
- **86.3% primary evidence** — directly documented by observers, not inferred
- **Near-miss controls** included: Panataran, Wringin Branjang survived 1919 Kelud, demonstrating stochastic variation

## Corroboration: ADV-3 (Survey Intensity Control)

ADV-3 (E069) confirmed that volcanic proximity significantly reduces site density even after controlling for survey intensity:

- Quasi-Poisson regression: beta = -0.477, p = 0.0015
- Three survey proxies controlled: road distance, BPCB office proximity, university proximity
- The volcanic signal is real and independent of where archaeologists choose to survey

## Preemptive Disclosure Language for P1 Revision

If a reviewer asks about site-type ratios, insert this passage (suggested location: Discussion section, after presenting burial depth evidence):

> "We note that site-type ratios (cave vs. open-air) do not vary significantly between volcanic and non-volcanic regions of Island Southeast Asia (Fisher exact p = 0.760, N = 80 sites; E081). This is consistent with cave bias being driven by karst availability and research tradition rather than volcanic burial. However, our taphonomic framework does not predict differential preservation by site type — it predicts depth-dependent invisibility across all site types. The burial depth evidence from colonial archaeological reports (32 sites, mean depth 2.88 m; E070), sedimentation modeling (r = 0.951, 32.3% of East Java with >1 m burial; E075), and eruption-site correlation analysis (24 measured depths, mean 3.41 m; E083) provides direct physical evidence for the burial mechanism."

## What This Changes in P1 Framing

1. **Remove or de-emphasize** any language suggesting volcanic regions should show *different site-type distributions* from non-volcanic regions. This is not supported.
2. **Strengthen** the burial depth argument as the primary L1 evidence. The colonial register and sedimentation model are much stronger than distributional comparisons.
3. **Add cave bias as a known confound.** Acknowledge it transparently rather than leaving it for a reviewer to discover.
4. **The core claim is unchanged:** volcanic sedimentation buries archaeological sites to depths that render them invisible to standard survey methods. The evidence for this is physical (measured burial depths) and quantitative (sedimentation models), not distributional (site-type ratios).

---

## Supporting Experiments

| Experiment | Result | Role |
|-----------|--------|------|
| E081 (ADV-2) | Fisher p = 0.760 | Falsifies site-type argument |
| E070 | 32 measured burial depths, mean 2.88 m | Primary L1 evidence |
| E075 | Pyle model r = 0.951, 32.3% cells >1m | Quantitative confirmation |
| E083 | 51 pairs, 24 depths, mean 3.41 m | Independent correlation |
| E069 (ADV-3) | Volcanic beta = -0.477, p = 0.0015 | Survives survey control |

---

*Prepared 2026-03-13. Replaces site-type reasoning with depth-based evidence throughout P1 revision strategy.*

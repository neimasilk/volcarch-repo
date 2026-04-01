# E162: State of Evidence at 161 Experiments

**Status:** SUCCESS (SYNTHESIS)
**Date:** 2026-03-31
**Type:** [S] Synthesis
**Papers:** All

## Purpose

Comprehensive synthesis of all VOLCARCH evidence as of experiment 161. Designed to be readable in 5 minutes by a potential collaborator, reviewer, or funder.

---

## THE ARGUMENT IN ONE PARAGRAPH

Java has 45 active volcanoes that bury archaeological sites at 2.4-6.2 mm/year. Combined with Indonesia's archaeological survey intensity (100-200x lower than Japan's), this creates a multiplicative invisibility cascade that renders 99.94% of pre-400 CE sites undetectable. Across 161 computational experiments using ML, NLP, GIS, and corpus analysis on 5+ independent datasets, we show that this cascade correctly predicts the archaeological gap (within 2x of observed), generalizes across 5 Indo-Pacific regions (Spearman rho=1.0), and that the "Two Javas" spatial pattern — sacred architecture near volcanoes, administrative inscriptions farther away — reflects differential burial and survey bias, not genuine absence of pre-Hindu civilization. Pre-Hindu sites survive ONLY where volcanic burial does not occur (Bali's coast: 4 sites; Java's non-volcanic coast: Buni + Batujaya; Java's volcanic interior: zero). The 929 CE Mataram collapse provides a natural experiment: when the court system collapsed, indigenous vocabulary resurfaced (z=3.04), confirming the substrate was present all along, merely invisible under the Sanskrit overlay.

---

## EVIDENCE SUMMARY TABLE

### Tier 1: Cathedral Findings (survive all scrutiny, p < 10^-4)

| Finding | Experiment | p-value | Robustness |
|---------|-----------|---------|------------|
| Candi equinox orientation | E066 | 4.9e-14 | E159: ROBUST |
| Toponymic court effect | E051 | 5.1e-14 | E159: ROBUST |
| Post-929 spatial shift | E152 | 3.9e-12 | E154: survives BH |
| Candi west-clustering | E031 | 3.4e-8 | E159: ROBUST |
| Inscription-volcano divergence | E084 | 5.2e-8 | E159: ROBUST |
| Substrate signal real (not noise) | E085 | <1e-5, z=11.05 | ADV-4 PASSED |
| Zone A overrepresentation | E065 | 5.3e-64 | E159: 13.5x |
| Volcanic signal survives survey control | E069 | 0.0015 | E159: ROBUST |

### Tier 2: Strong Findings (p < 0.01, survive BH)

| Finding | Experiment | Key Result |
|---------|-----------|------------|
| Cascade model matches data | E110 | 0.058% predicted vs 0.031% observed (1.9x) |
| Demographic gap 3,220x | E108 | Null hypothesis rejected |
| Cross-regional cascade validation | E155 | rho=1.0 across 5 regions |
| 929 CE semantic rupture (GPU NLP) | E160 | z=3.04, p=0.012 |
| Independent depth replication | E128 | Two datasets, identical medians |
| Gap robust under all assumptions | E122 | P(gap<10x) = 0.0% in 100K MC |
| Java globally unique gap | E126 | Only 1M+ yr region with zero open-air pre-400CE |
| 73% temple survey bias | E129 | 277/391 sites are temples |
| FDR 78.3% survival | E154 | 65/83 tests survive BH correction |

### Tier 3: Supporting Evidence

| Finding | Experiment | Key Result |
|---------|-----------|------------|
| Bali comparandum: 5/5 predictions confirmed | E161 | 14.3x predicted, 12x observed |
| L1xL2 "Double Erasure" | E156 | 94K people displaced into volcanic zones |
| F4 = 0.43 (Hindu), 0.20 (pre-Hindu) | E157 | Ethnographic calibration |
| F2 triple convergence (0.20, 0.23, 0.21) | E135, E157 | Three independent estimates |
| Vocabulary x burial depth rho=0.456 | E102 | Sanskrit-driven |
| Pre-Indic ratio INCREASES over time | E030 | rho=+0.502, substrate persists |
| Agriculture 91% native vocabulary | E058 | Sanskrit overlay = elite only |
| "Bamboo civilization" confirmed | E040, E140 | 60-63% organic material in inscriptions |

### Tier 4: Known Limitations

| Issue | Experiment | Status |
|-------|-----------|--------|
| Cascade is underdetermined (5 params, 1 data point) | E110, E158 | Partially addressed by E155 |
| Zero external validation | — | 5 papers under review, 0 accepted |
| Zero fieldwork / physical samples | — | $6K-100K needed for decisive test |
| DHARMA dependency (~25/161 experiments) | E068 | Mitigated by E091, E141-E143, E150 |
| E051 metric sensitivity | E159 | Court distance, not volcano distance |
| E032 + E053 fail FDR | E154 | Report as "suggestive" |

---

## THE CASCADE MODEL (E110, updated with E155, E157)

| Factor | Java | Bali | Japan | Source |
|--------|------|------|-------|--------|
| F1 Volcanic burial | 0.58 | 0.92 | 0.85 | Calibration sites + E157 |
| F2 Organic decay | 0.20 | 0.20 | 0.45 | E135, E157 (triple convergence) |
| F3 Survey coverage | 0.025 | 0.15 | 0.80 | E086, E129 |
| F4 Recognition | 0.40 | 0.50 | 0.90 | E157 (Liangan: 0.43) |
| F5 Publication | 0.50 | 0.60 | 0.90 | E093 |
| **Product** | **0.058%** | **0.83%** | **24.8%** | — |
| **Observed** | **0.031%** | **~0.37%** | **~50%** | E108, E146 |

Cross-regional rank order: Java < Sulawesi < Philippines < Bali < Japan (rho=1.0, p=0.017)

---

## WHAT WOULD PROVE VOLCARCH WRONG

1. Pre-400 CE open-air site found in volcanic East Java by surface survey
2. Cascade predicts gaps that don't exist (3+ cross-regional failures)
3. 3+ substantive peer reviews say methodology is fundamentally flawed
4. Independent analysis of same data produces different conclusions
5. Domain expert consensus (2+ Java archaeologists/geologists) judges framework unsound

---

## WHAT COMES NEXT

| Priority | Action | Cost | Impact |
|----------|--------|------|--------|
| 1 | P17 accepted at ArchCalc | $0 (Diamond OA) | Credibility anchor |
| 2 | GitHub repo public | $0 | External validation pathway |
| 3 | 20 geotechnical boreholes | $6,000 | Cheapest decisive test |
| 4 | GPR survey at Zone B targets | $40,000 | Expected 2.5 finds [0,6] |
| 5 | Phytolith analysis of volcanic cores | $500-2,000 | Pre-Hindu agriculture proof |

---

*"The question is no longer whether pre-Hindu Nusantara civilization existed. The question is how much of it we have lost, and whether we can find what remains before it disappears entirely beneath 3.6 millimeters of volcanic sediment per year."*

*161 experiments. 5 papers under review. 78.3% statistical survival rate. Zero fieldwork budget. One prediction: dig at Kelud's western flank, 3-7 meters deep. Find the civilization that volcanism buried.*

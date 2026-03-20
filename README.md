# VOLCARCH

**Volcanic Taphonomic Bias in Indonesian Archaeological Records**

A computational framework for predicting where pre-Hindu archaeological sites in volcanic Java may be buried underground, and how deep to look.

## The Problem

Java has been densely populated for millennia and sits on one of the most active volcanic arcs in the world. Eruptions deposit centimeters of ash every few decades; over centuries, this buries archaeological sites meters underground. The "oldest" known kingdom in Indonesia (Kutai, ~400 CE) is in Kalimantan — a region with **zero active volcanoes**. Coincidence?

Surface survey in volcanic Java can only detect sites from ~1900 CE onward (at 4 mm/yr sedimentation, 0.5m detection limit). Pre-400 CE sites would be at 6.5m+ depth. No standard archaeological method currently used in Indonesia can reach them.

## The Framework

A 5-factor visibility cascade explains why ~99.94% of pre-Hindu sites are invisible:

| Factor | Survival Probability | Leverage |
|--------|---------------------|----------|
| Volcanic burial | 0.58 | 1.7x |
| Organic decay | 0.20 | 5.0x |
| Survey coverage | 0.025 | **40x** |
| Recognition | 0.40 | 2.5x |
| Publication | 0.50 | 2.0x |

Combined visibility: **0.058%** (vs. 0.031% observed). Monte Carlo analysis (100K runs) confirms robustness: 92% of runs within 10x of observed, correlation-robust.

## Testable Predictions

The framework is falsifiable. Pre-registered prediction: 20 GPR surveys at our highest-ranked target locations should find 0-6 archaeological anomalies (95% CI). P(zero finds) = 7%. Cost: $40K-100K for the decisive test.

## Key Results (119 experiments)

- **Sedimentation rate:** 4.4 +/- 1.2 mm/yr calibrated from 51 eruption-site pairs (r = 0.951)
- **Demographic gap:** 3,220x between expected pre-400 CE settlements and observed (0-3 known)
- **Detection horizon:** Surface survey reaches ~1900 CE; GPR reaches ~776 CE; deep coring reaches ~474 BCE
- **Volcanic targeting:** 3.5x search efficiency over random survey, $16.7K cost savings per first-find
- **West Java control:** Non-volcanic coast has 3-4 pre-400 CE sites; volcanic interior has 0. Same island, same culture — different geology.

## Papers

| Paper | Target | Status |
|-------|--------|--------|
| P1 | EGQSJ (Copernicus, Diamond OA) | Ready to submit |
| P2 | JCAA (Diamond OA) | Under review |
| P7 | Antiquity Project Gallery | Under review |
| P8 | Oceanic Linguistics (Q1) | Under review |

Preprint: [Zenodo DOI 10.5281/zenodo.19081502](https://doi.org/10.5281/zenodo.19081502)

## Structure

```
docs/           Research documents (layered: L1 constitution, L2 strategy, L3 execution)
data/           Raw and processed datasets
experiments/    119 numbered, self-contained experiments (E001-E118)
papers/         12 paper items (P1-P18, D1-D2)
tools/          Utility scripts (sync checker, dashboard, scrapers)
```

## Contributing

This is an academic research project. If you are an archaeologist, volcanologist, or GIS specialist interested in collaboration — particularly for GPR fieldwork in East Java — please open an issue.

## License

Code: MIT. Papers and documents: CC BY 4.0. Data: see individual source licenses.

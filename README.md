# VOLCARCH

**Volcanic Taphonomic Bias in Indonesian Archaeological Records**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19081502.svg)](https://doi.org/10.5281/zenodo.19081502)
[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Docs-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Experiments](https://img.shields.io/badge/Experiments-120-green.svg)](#key-results)

> Java has 45 active volcanoes. Its early archaeological record is almost empty. This is not a coincidence.

---

## The Problem

Indonesia's "oldest" kingdom (Kutai, ~400 CE) is on Kalimantan — a region with **zero active volcanoes**. Java, despite being one of the most densely populated islands in human history and sitting on one of the most active volcanic arcs in the world, has virtually no open-air archaeological sites before 400 CE. The dominant explanation has been cultural: "people simply didn't live there yet." We argue the explanation is geological.

Volcanic eruptions deposit centimeters of ash every few decades. Over centuries, this buries archaeological sites meters underground — far below the reach of standard surface survey methods. At 4.4 mm/yr average sedimentation rate, a pre-400 CE site would lie at **7+ meters depth**. Surface survey can only detect down to ~50 cm, making it blind to everything before approximately 1900 CE in volcanic interior Java.

## The Framework

A 5-factor visibility cascade explains why ~99.94% of pre-Hindu sites are invisible to current methods:

| Factor | Survival | Mechanism |
|--------|----------|-----------|
| Volcanic burial | 0.58 | Tephra and lahar cover sites at depth |
| Organic decay | 0.20 | Acidic andisol soils destroy non-stone materials |
| Survey deficit | 0.025 | No systematic subsurface survey in volcanic Java |
| Recognition failure | 0.40 | Pre-literate, organic-material sites lack diagnostic markers |
| Publication bias | 0.50 | Unreported finds never enter the record |

**Combined visibility: 0.058%** — consistent with the 0.031% observed (3 ambiguous sites out of ~3,000 expected).

Monte Carlo analysis (100,000 runs) confirms the model is robust: 92% of runs fall within 10x of observed values, and the result is correlation-robust across parameter covariance structures.

## Testable Predictions

The framework is **falsifiable**. Pre-registered prediction:

- **20 GPR surveys** at our highest-ranked targets should find **0-6 archaeological anomalies** (95% CI)
- **P(zero finds) = 7%** — even a null result is informative
- **Cost of the decisive test:** $40,000-100,000

## Key Results

120 experiments (E001-E119) across six evidence layers:

| Result | Value | Experiment |
|--------|-------|------------|
| Sedimentation rate | 4.4 ± 1.2 mm/yr (51 eruption-site pairs, r = 0.951) | E005 |
| Demographic gap | 3,220x between expected and observed pre-400 CE sites | E108 |
| Detection horizon | Surface survey ~1900 CE; GPR ~776 CE; deep coring ~474 BCE | E117 |
| Search efficiency gain | 3.5x over random survey; 29% entropy reduction | E118 |
| Anomaly detection | 195,382 site-like cells with >1m burial across E. Java | E097 |
| West Java control | Non-volcanic coast has 3-4 pre-400 CE sites; volcanic interior has 0 | E110 |
| Inscription spatial bias | Inscriptions average 9.2 km farther from volcanoes than temples (p = 5.2 × 10⁻⁸) | E084 |
| Substrate signal | z = 11.05, genuine pre-Indic linguistic signal in court zone vocabulary | E085 |
| Cascade robustness | 92% of 100K Monte Carlo runs within 10x of observed | E115 |
| Falsifiability | GPR predictions registered: expect 2.5 finds in [0,6] 95% CI | E116 |

### Six Evidence Layers

1. **L1 — Volcanic Burial:** Sedimentation model calibrated against multiple volcanic systems (Kelud, Merapi, Sundoro)
2. **L2 — Coastal Submersion:** Quantified land loss since 1500 BP (area-based, not archaeological)
3. **L3 — Historiographic Bias:** Colonial and nationalist framing distorted the archaeological record
4. **L4 — Cosmological Overwrite:** Sanskrit terminology replaced indigenous vocabulary in administrative texts
5. **L5 — Genre Taphonomy:** Different text genres preserve different cultural information (raw p < 10⁻⁶)
6. **L6 — Historiographic Periodicity:** Publication patterns reflect political cycles, not archaeological reality

## Papers

| Paper | Title | Target Journal | Status |
|-------|-------|---------------|--------|
| P1 | Taphonomic framework + cascade model | EGQSJ (Copernicus, Diamond OA) | Ready to submit |
| P2 | GIS settlement model | JCAA (Diamond OA) | Under review |
| P7 | Theory of Motivated cartography | Antiquity Project Gallery (Q1) | Under review |
| P8 | Linguistic fossils in Old Javanese | Oceanic Linguistics (Q1) | Under review |
| P11 | Temple siting as archaeological proxy | Indonesia (Cornell) | Drafting |
| P16 | Computational textual archaeology | DHQ (ADHO, Diamond OA) | Drafting |
| P17 | Two Javas: sacred vs. administrative landscapes | Archeologia e Calcolatori (Diamond OA) | Drafting |

**Preprint:** [Zenodo DOI 10.5281/zenodo.19081502](https://doi.org/10.5281/zenodo.19081502) (CC BY 4.0)

## Repository Structure

```
volcarch/
├── docs/                  Research documents (layered governance: L1-L3)
│   ├── L1_CONSTITUTION.md     Core hypotheses & philosophy (stable)
│   ├── L2_STRATEGY.md         Current phase & methodology (per-phase)
│   ├── L3_EXECUTION.md        Active tasks & experiments (per-week)
│   ├── EVAL.md                Evaluation criteria & validation protocol
│   ├── JOURNAL.md             Research log (append-only)
│   ├── EXPERIMENT_INDEX.md    Master index of all 120 experiments
│   └── dissemination/         Outreach materials, slides, scripts
├── data/
│   ├── raw/                   Original datasets (never modified)
│   ├── processed/             Cleaned/transformed data
│   └── schema.md              Data format definitions
├── experiments/               120 numbered, self-contained experiments
│   ├── E001_site_density_vs_volcanic_proximity/
│   ├── E005_sedimentation_rate_calibration/
│   ├── ...
│   └── E119_synthesis_figure/
├── papers/                    Paper drafts, submission files, revision ammo
├── models/                    Trained models & configurations
├── maps/                      Generated probability maps & visualizations
└── tools/                     Utility scripts (sync checker, scrapers)
```

Each experiment directory contains a `README.md` with hypothesis, method, data, result, conclusion, and status (SUCCESS / FAILED / INCONCLUSIVE / REVISIT). Failed experiments are documented, not deleted.

## Priority Fieldwork Targets

20 GPS coordinates identified across two volcanic zones in East Java, scoring highest on a 5-component weighted model (volcanic proximity, temple clustering, survey gap, terrain suitability, predicted burial depth):

- **Zone A — Kelud western flank:** 6 targets, 5-8 km from crater, 5-8 m predicted burial
- **Zone B — Arjuno-Welirang eastern flank:** 4 targets, 5-8 km from crater, 5-9 m predicted burial

All targets are near known candi (Penataran, Gambar Wetan, Sumberawan, Jawi), confirming historical occupation in zones where burial is deepest.

## Technical Stack

- **Languages:** Python 3.10+
- **Core libraries:** geopandas, rasterio, scikit-learn, xgboost, folium, matplotlib, scipy
- **Data formats:** CSV, GeoJSON, shapefiles
- **LaTeX:** Papers compiled with pdflatex + bibtex/biblatex

## Collaboration

This is an academic research project based at Universitas Bhinneka Nusantara, Malang, Indonesia. We are actively seeking collaboration partners for:

- **GPR/LiDAR fieldwork** in East Java volcanic zones (highest priority)
- **Phytolith analysis** of volcanic matrix samples
- **Rescue archaeology** methodology development for Indonesia
- **Remote sensing** validation of buried-site predictions

If you are an archaeologist, volcanologist, geophysicist, or GIS specialist — or if you know someone who is — please open an issue or contact: **amien@ubhinus.ac.id**

## How to Cite

```bibtex
@misc{amien2026volcarch,
  author       = {Amien, Mukhlis},
  title        = {{VOLCARCH}: Volcanic Taphonomic Bias in Indonesian Archaeological Records},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19081502},
  url          = {https://doi.org/10.5281/zenodo.19081502}
}
```

## License

- **Code:** MIT
- **Papers and documents:** CC BY 4.0
- **Data:** See individual source licenses in `data/sources.md`

---

*"Kerajaan tertua Indonesia bukan kerajaan yang paling tua. Kerajaan tertua adalah kerajaan yang paling terlihat."*

*The oldest kingdom in Indonesia is not the most ancient one. It is the most visible one.*

# E082: DHARMA Inscription Georeferencing

> ## ⚠ CORRECTION 2026-10-01 (re-entry audit, ledger C032) — the candi–inscription contrast is a geocoding artefact
>
> The canonical file `results/canonical30/geocoded_inscriptions_canonical30.csv` has 175 rows but only **42
> distinct coordinates**; 65.7% of rows sit on five points: **50 Borobudur relief captions** at one point
> ("Borobudur, Magelang", 28.2 km from a volcano) and **42 region-level placeholders** ("Mataram Central Java"
> 20, "East Java" 18, "Central Java" 4). `geocode_method` labels all of them `known_location`, so no filter
> caught them. Re-run (`robustness_geocoding_20261001.py` → `results/canonical30/robustness_geocoding_20261001.txt`):
>
> | Inscriptions vs candi | n ins / candi | median ins vs candi | gap | MW p |
> |---|---|---|---|---|
> | as published (P17/P11 numbers, U=8125) | 175 / 142 | 27.6 vs 14.5 km | 13.1 km | 1.1e-7 |
> | precise findspots only (no captions, no placeholders) | 83 / 142 | 16.8 vs 14.5 km | 2.3 km | 0.068 |
> | precise findspots, candi de-duplicated (103 unique coords) | 83 / 103 | 16.8 vs 16.6 km | 0.2 km | 0.27 |
> | precise findspots, East Java only, candi de-duplicated | 37 / 103 | 16.8 vs 16.6 km | 0.2 km | 0.85 |
>
> **The "two geographies" contrast does not survive.** It was produced by one monument's captions and by
> province-level placeholder points that happen to sit in the 15–30 km "court zone". This affects P17
> (under review, ArchCalc #365), P11 (pillar 2), and E105's 929 CE result (48 of 58 pre-929 "court-zone"
> records are the Borobudur captions, per the 2026-10-01 verifier). Found by an Opus verifier, then re-derived
> independently by the orchestrator.

## Hypothesis
Inscriptions cluster away from active volcanoes, suggesting volcanic zones are under-represented in the epigraphic record. This would support the "Invisible Millennium" thesis: volcanic landscapes erase or bury the material evidence of past civilizations, including stone inscriptions.

## Method
Multi-source geocoding pipeline applied to 268 DHARMA EpiDoc TEI-XML inscriptions from Java, Sumatra, Bali, and beyond (7th-14th century CE):

1. **Known locations** (173 inscriptions) — Hard-coded coordinates from published epigraphy for well-documented findspots (Canggal, Kota Kapur, Kalasan, Borobudur, Dieng, etc.) and regional assignment for inscriptions attributable to specific polities (Mataram, Majapahit, Kediri).
2. **Candi coordinate matching** (7 inscriptions) — Cross-referenced inscription titles against 93 candi GPS coordinates from E031 data.
3. **XML provenance parsing** (2 inscriptions) — Extracted location hints from DHARMA XML commentary/provenance fields.

Confidence levels: high (88), medium (89), low (5).

After geocoding, calculated distance to nearest active volcano (15 volcanoes) and performed proximity analysis by century.

## Data
- **Input:** `experiments/E074_dharma_deep_nlp/results/inscription_metadata.csv` (268 inscriptions)
- **Candi coordinates:** `experiments/E031_candi_orientation/results/candi_volcano_pairs.csv` (93 candi)
- **DHARMA XML:** `experiments/E023_ritual_screening/data/dharma/xml/` (269 files)

## Results

### Geocoding Success
- **182 / 268 inscriptions geocoded (67.9%)** — well above the 50-inscription target
- 175 in Java/Bali, 7 outside (Sumatra, Philippines, Singapore)

### Volcanic Proximity (Java/Bali, N=175)
| Metric | Value |
|--------|-------|
| Mean distance to nearest volcano | 25.5 km |
| Median distance | 27.6 km |
| Zone A (0-10 km) | 22 (13%) |
| Zone B (10-30 km) | 115 (66%) |
| Zone C (>30 km) | 38 (22%) |

### Comparison with Candi (E065)
| Dataset | Mean distance | Median | N |
|---------|--------------|--------|---|
| Candi (E065) | 16.5 km | 14.7 km | 142 |
| Inscriptions (E082) | 25.5 km | 27.6 km | 175 |

**Inscriptions are 9.0 km farther from volcanoes than candi on average.** This is the opposite of what one might expect if inscriptions and temples were co-located. It suggests inscriptions have a broader spatial distribution than temples, possibly reflecting administrative charter grants for villages at moderate distances from volcanic centers.

### Distance by Century
| Century | N | Mean km |
|---------|---|---------|
| C8 (701-800) | 4 | 19.4 |
| C9 (801-900) | 14 | 13.0 |
| C10 (901-1000) | 30 | 20.7 |
| C11 (1001-1100) | 8 | 42.7 |
| C12 (1101-1200) | 1 | 50.1 |
| C13 (1201-1300) | 6 | 29.4 |
| C14 (1301-1400) | 3 | 29.3 |

Spearman rho = 0.643 (positive trend: later centuries are farther from volcanoes), but **not significant** at alpha=0.05 (t=1.877, df=5, critical=2.57). The 9th century (Mataram dynasty peak) shows the closest proximity (13.0 km mean), consistent with Kedu Plain concentration near Merapi.

### Nearest Volcano Distribution
- Merapi dominates: 94 inscriptions (54% of Java/Bali set)
- Kelud: 25, Penanggungan: 19, Galunggung: 8
- This confirms Central Java (Merapi zone) as the epigraphic heartland

## Key Findings

1. **Inscriptions do NOT cluster at volcano flanks** — unlike candi (E065), which overrepresent Zone A by 17.9x, inscriptions are predominantly in Zone B (10-30 km), the agricultural hinterland.
2. **The 9th century is anomalously close to volcanoes** (13.0 km mean), reflecting the Mataram dynasty's Kedu Plain heartland near Merapi/Merbabu.
3. **Post-929 CE eastward shift visible**: C11-C12 inscriptions average 42-50 km from nearest volcano, consistent with the move to East Java (Kediri/Trowulan) which is farther from the nearest volcanoes.
4. **Merapi dominance (54%)** in the inscription record mirrors the known political geography — Mataram/Sailendra dynasties were based in the Kedu-Prambanan corridor.

## Caveats
- Medium-confidence geocoding (89 inscriptions) uses regional centroids, not exact findspots. Precision is approximately +/- 20 km.
- 86 inscriptions (32%) could not be geocoded — these are mostly undated/untitled fragments.
- The comparison with E065 candi data uses different spatial scales (candi are clustered on specific mountains, inscriptions represent administrative territories).
- The Spearman test has low power (N=7 centuries) — the positive trend (rho=0.643) is suggestive but inconclusive.

## Status: SUCCESS

Successfully geocoded 182/268 inscriptions (68%), far exceeding the 50-inscription target. The dataset enables spatial analysis and reveals that inscriptions have a different spatial signature than temples: they are farther from volcanoes, concentrated in the agricultural Zone B. This supports a nuanced view of the "Invisible Millennium" where volcanic taphonomic bias affects different evidence types differently.

## Papers Served
- **P11** (Volcanic Informedness) — inscription spatial distribution as additional evidence layer
- **P1** (Taphonomic Framework) — geocoded inscription data for comparative analysis with archaeological sites

## Output Files
- `results/geocoded_inscriptions.csv` — Full geocoded dataset (182 rows)
- `results/geocoding_summary.txt` — Summary statistics
- `results/volcanic_proximity_analysis.txt` — Detailed proximity analysis
- `results/e082_results.json` — Machine-readable results

## RE-RUN 2026-06-10 — canonical 30-volcano inventory (ME#18 integrity sweep)

Original used 20 hardcoded volcanoes (missing most W./C. Java peaks; included Krakatau, which absorbed Sumatran outliers). Re-run: `e082_rerun_canonical30.py` — reuses the geocoding (`geocoded_inscriptions.csv`), recomputes distances with `volcanoes_java_full.csv` (30) + Agung/Batur for Bali; Krakatau dropped. Outputs in `results/canonical30/`.

**Verdict: direction & significance SURVIVE; magnitude shrinks.**

| Metric | 20-volcano (orig) | Canonical 30 |
|---|---|---|
| Java/Bali mean / median (km) | 25.5 / 27.6 | 22.2 / 27.6 |
| Zone A / B / C | 22 / 115 / 38 | 23 / 120 / 32 |
| Candi-vs-inscription mean gap | 9.2 km (CI 5.5–12.7) | **6.1 km (CI 3.2–9.1)** |
| Mann-Whitney p | 5.2e-8 | 2.8e-7 |
| Century Spearman | rho=0.643, n.s. | rho=0.607, p=0.148, n.s. |

**Propagation: P11 cites "9.2 km" (abstract + §Test 3) → must be corrected to 6.1 km (CI 3.2–9.1, p=2.8e-7) at revision.** P17's median segregation figures (14.5 vs 27.6 km) match the canonical numbers exactly — already consistent.

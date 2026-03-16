# E080: Fieldwork Targeting — Priority Zones

**Date:** 2026-03-13
**Status:** SUCCESS
**Paper:** P1, P2, P11
**Layer:** L1

---

## Hypothesis

By integrating volcanic proximity, candi clustering, survey gaps, terrain suitability, and predicted burial depth from upstream experiments, we can identify specific GPS coordinates where buried archaeological sites are most likely to exist — producing actionable fieldwork targets.

## Method

5-component weighted scoring over ~2,500 grid cells across East Java (2.2 km resolution):

| Component | Weight | Source |
|-----------|--------|--------|
| Volcanic proximity (sweet spot 5–15 km) | 30% | Volcano coordinates |
| Candi cluster proximity | 25% | E065 candi spatial analysis |
| Discovery gap (low site density = high potential) | 20% | ADV-3 survey intensity |
| Terrain suitability | 15% | E005/E013 settlement model |
| Burial depth estimate | 10% | E075 Pyle exponential model |

Composite scores ranked, top 20 extracted, clustered into priority zones.

## Data

- Input: E005, E013, E065, E070, E075, ADV-3 (E069)
- Output: `results/all_targets_scored.csv` (full grid), `results/top20_targets.csv`, `results/e080_results.json`

## Results

**20 priority targets in 6 zones:**

| Zone | Region | Distance to Volcano | Predicted Burial | Nearest Candi |
|------|--------|---------------------|------------------|---------------|
| 1–3 | Kelud | 5–8 km | 5–8 m | Candi Gambar Wetan |
| 4–6 | Arjuno-Welirang | 5–8 km | 5–8 m | Candi Sumberawan, Candi Jawi |

**Top target:** -7.98, 112.36 (score 0.855) — 8 km from Kelud, 5 m predicted burial.

**Recommended survey methods by depth:**
- <1 m: Surface survey + test pits
- 1–3 m: Systematic trenching + GPR
- 3–5 m: Mechanical augering + GPR + remote sensing
- \>5 m: Deep augering + seismic + satellite analysis

**Cost estimate:** Phase 1 remote sensing ~$50–100, Phase 3 GPR ~$2,000–5,000.

## Conclusion

**SUCCESS.** The scoring system integrates 6+ upstream experiments into actionable fieldwork proposals. All top 20 targets cluster near Kelud and Arjuno-Welirang — volcanoes with high sedimentation rates AND nearby candi proving historical occupation. These are the strongest candidates for VOLCARCH's core prediction: buried sites at depth.

## Scripts

- `fieldwork_targeting.py` — Grid scoring, ranking, and zone clustering

## Relation to Other Experiments

- Builds on: E005, E013, E059, E065, E069, E070, E075
- Feeds into: P11 (methodology paper — validates that candi distributions predict buried sites)
- Next step: Fieldwork partnership (GPR/LiDAR) — see L3 backlog

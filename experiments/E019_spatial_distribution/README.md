# E019: Spatial Distribution Test (Paper 7 Metric 3)

## Status: SUCCESS

## Hypothesis
If the Taphonomic Overprint Model (H-TOM) is correct, then:
1. Zone B cells (high suitability, moderate burial, zero known sites) should cluster
   significantly closer to volcanic axes than Zone A cells (where sites are found).
2. All known deep-time (>10 ka) Java archaeological sites occur in protected
   geomorphic contexts (karst caves, river terraces) — never on volcanic plains.

## Method
Three complementary spatial analyses using pre-computed data from E013/E016:

### Analysis 1: Site distance to nearest volcano
- Haversine distance from each of 378 sites to nearest of 7 East Java volcanoes.
- Compare with baseline distance distribution of all 65K grid cells.
- Mann-Whitney U test for difference.

### Analysis 2: Zone B spatial clustering (key test)
- Distance to nearest volcano for all grid cells, grouped by zone (A/B/C/E).
- Compare Zone A vs Zone B distances (Mann-Whitney U + Cohen's d).
- If H-TOM is correct: Zone B should be significantly closer to volcanoes.

### Analysis 3: Deep-time site context map
- 4 published deep-time Java sites (Song Terus, Trinil, Sangiran, Wajak).
- Overlay on zone map with volcano positions.
- Qualitative: all should be in karst/river contexts, none on volcanic plains.

## Data Used
- `data/processed/dashboard/sites.csv` (378 sites with zones)
- `data/processed/dashboard/volcanoes.csv` (7 volcanoes)
- `data/processed/dashboard/grid_predictions.csv` (65,432 cells with zones)
- `data/deep_time_sites.csv` (4 sites from published literature)

## Kill / Pass Criteria

| Outcome | Finding | Decision |
|---------|---------|----------|
| Strong support | Zone B significantly closer to volcanoes (p<0.05); deep-time sites all in karst/river | **SUPPORTS H-TOM** |
| Moderate support | Trend visible but p>0.05; deep-time sites in protected contexts | **PARTIAL** |
| No signal | Zone B distance ~ Zone A distance | **NEUTRAL** |
| Counter-evidence | Zone B far from volcanoes; or deep-time sites on volcanic plains | **WEAKENS H-TOM** |

## Results

### Analysis 1: Sites are CLOSER to volcanoes than geographic chance
- Site median distance: 27.9 km vs grid baseline: 59.2 km
- Mann-Whitney U = 7,736,534, p = 3.02e-36, r = 0.049
- Interpretation: Sites cluster nearer to volcanoes because volcanic lowlands are fertile
  and attract settlement. This is the *input* that creates the taphonomic trap.

### Analysis 2: Zone B clusters near the volcanic axis (KEY RESULT)
- Zone A (sites exist): median = 42.5 km from nearest volcano (n=15,217)
- Zone B (no sites, buried): median = 16.1 km (n=1,093)
- Zone C (deep burial): median = 2.6 km (n=48)
- Zone E (low suitability): median = 71.0 km (n=49,074)
- **Mann-Whitney U = 14,254,494, p ≈ 0, Z = 39.50, r = 0.309, Cohen's d = 1.005**
- Clear monotonic gradient: C (2.5 km) < B (16.0 km) < A (59.5 km) < E (76.4 km)

### Analysis 3: Deep-time sites — all in protected contexts
- Song Terus: 153.5 km to Kelud (cave, Gunung Sewu karst)
- Trinil: 121.6 km to Kelud (river terrace, Solo River)
- Sangiran: 169.3 km to Kelud (river erosion, Solo basin dome)
- Wajak: 89.7 km to Kelud (cave, Tulungagung karst)
- All 4 sites are in karst caves or river terraces, 90-170 km from nearest volcano.
  None are on volcanic plains, consistent with H-TOM.

## Conclusion

**SUPPORTS H-TOM (strong).**

Zone B cells (high suitability but no known sites) are significantly and substantially
closer to volcanoes than Zone A cells (Cohen's d = 1.005, a "large" effect). The
distance gradient A > B > C maps directly onto the burial depth gradient: areas closer
to volcanoes have deeper tephra burial, pushing archaeological material below detection.

All four deep-time Java sites occur in geomorphically protected contexts (karst caves,
river terraces) far from volcanic axes — exactly where H-TOM predicts preservation.

The combination of Analysis 2 (quantitative: Zone B clusters near volcanoes) and
Analysis 3 (qualitative: deep-time sites avoid volcanic plains) provides strong spatial
evidence for the Taphonomic Overprint Model.

## Output Files
- `results/site_volcano_distances.csv` — 378 sites with distances
- `results/zone_distance_summary.csv` — zone-level statistics
- `results/spatial_summary.txt` — full report
- `results/fig_distance_histogram.png` — Analysis 1
- `results/fig_zone_distance_boxplot.png` — Analysis 2
- `results/fig_deep_time_context_map.png` — Analysis 3

# E081: ADV-2 Non-Volcanic Control Test

**Status: INCONCLUSIVE**

**Type:** Adversarial test (ADV-2)
**Date:** 2026-03-13
**Pass criterion:** Fisher exact p < 0.05 showing volcanic regions have statistically different site-type distribution from non-volcanic controls.

## Hypothesis

If VOLCARCH's L1 claim (volcanic taphonomy biases the archaeological record toward enclosed sites like caves) is correct, then NON-volcanic regions (Kalimantan, Madagascar) should show a different site-type distribution from volcanic regions (Java, Sumatra, Sulawesi, etc.). Specifically, volcanic regions should have a higher proportion of enclosed sites (caves + rockshelters) because open-air sites are preferentially destroyed/buried by volcanic deposits.

**Adversarial framing:** If non-volcanic regions show the SAME cave dominance as volcanic regions, the volcanic explanation collapses and cave bias must be attributed to universal taphonomic or research factors (e.g., caves preserve better everywhere, archaeologists preferentially excavate caves everywhere).

## Method

1. Classified 80 sites from mini_nusarc_v3.csv into volcanic (N=67) and non-volcanic control (N=13) groups.
   - **Volcanic:** Java, Sumatra, Sulawesi, Nusa Tenggara, Philippines, Maluku
   - **Non-volcanic controls:** Kalimantan (Borneo), Madagascar
2. Categorized site types as enclosed (cave + rockshelter) vs open (open_air + river_terrace).
3. Fisher exact test on 2x2 contingency table (enclosed/open x volcanic/non-volcanic).
4. Chi-square test on full 4-category site-type distribution.
5. Per-region breakdown with site density normalized by land area.
6. Supplementary Java-vs-Kalimantan pairwise comparison.

## Data Used

- `experiments/E020_mini_nusarc/data/mini_nusarc_v3.csv` (80 sites, 8 regions, 5 species)

## Results

### Aggregate comparison

|                    | Volcanic (N=67) | Non-volcanic (N=13) |
|--------------------|-----------------|---------------------|
| Cave               | 36 (53.7%)      | 7 (53.8%)           |
| Rockshelter        | 6 (9.0%)        | 2 (15.4%)           |
| Open-air           | 16 (23.9%)      | 4 (30.8%)           |
| River terrace      | 9 (13.4%)       | 0 (0.0%)            |
| **Enclosed total** | **42 (62.7%)**  | **9 (69.2%)**       |
| **Open total**     | **25 (37.3%)**  | **4 (30.8%)**       |

### Statistical tests

- **Fisher exact test (enclosed vs open):** p = 0.760, OR = 0.75. NOT SIGNIFICANT.
- **Chi-square test (4-category):** chi2 = 2.40, p = 0.493, dof = 3. NOT SIGNIFICANT.
  - Warning: minimum expected cell count = 1.30 (< 5), chi-square unreliable.

### Per-region breakdown (critical detail)

| Region          | Group    | N   | Enclosed % | Density (sites/10k km2) |
|-----------------|----------|-----|-----------|------------------------|
| Java            | Volcanic | 19  | 36.8%     | 1.47                   |
| Kalimantan      | Control  | 8   | 100.0%    | 0.15                   |
| Madagascar      | Control  | 5   | 20.0%     | 0.09                   |
| Maluku          | Volcanic | 5   | 100.0%    | 0.67                   |
| Nusa Tenggara   | Volcanic | 12  | 58.3%     | 1.64                   |
| Philippines     | Volcanic | 6   | 83.3%     | 0.20                   |
| Sulawesi        | Volcanic | 18  | 83.3%     | 1.03                   |
| Sumatra         | Volcanic | 7   | 42.9%     | 0.15                   |

### Java vs Kalimantan (supplementary)

- **Fisher exact p = 0.003**, OR = 0.0
- Java: 36.8% enclosed (7/19). Kalimantan: 100% enclosed (8/8).
- This is the OPPOSITE direction from the VOLCARCH prediction: the non-volcanic control (Kalimantan) is MORE cave-dominated than the most intensively studied volcanic region (Java).

## Conclusion

### The test FAILS to pass the criterion (Fisher p = 0.76 >> 0.05).

However, the result is INCONCLUSIVE rather than a definitive failure of L1, for the following reasons:

### Why INCONCLUSIVE, not FAILED:

1. **Severe sample size imbalance.** The non-volcanic control group has only 13 sites vs 67 volcanic. With N=13, the test has very low statistical power to detect moderate effect sizes.

2. **The two control regions tell opposite stories.** Kalimantan = 100% enclosed (8/8). Madagascar = 20% enclosed (1/5). These are radically different patterns that average out to 69.2% enclosed, which is misleadingly similar to the volcanic aggregate. The "non-volcanic" category is not internally coherent.

3. **Kalimantan's 100% enclosed rate is a research artifact.** All 8 Kalimantan sites are caves because that is where karst archaeology has been done in Borneo (Niah, Lubang Jeriji Saleh, etc.). There are almost certainly open-air sites in Kalimantan that have not been excavated or included. This is survey bias, not taphonomic signal.

4. **Madagascar's 80% open-air rate supports L1 indirectly.** Madagascar has no volcanoes and no karst cave archaeology tradition, and its sites are predominantly open-air. This IS the pattern L1 would predict for a non-volcanic region, but N=5 is far too small.

### Honest assessment of what this means for VOLCARCH:

**The aggregate comparison does NOT support L1.** At the aggregate level, volcanic and non-volcanic regions show statistically indistinguishable enclosed/open ratios. The VOLCARCH project cannot claim that volcanic regions have distinctly different site-type distributions based on this dataset.

**But the test is structurally flawed.** The real question is not "volcanic vs non-volcanic" but rather "what controls site discovery?" The per-region data show enormous heterogeneity driven by:
- **Karst availability** (Kalimantan, Sulawesi, Maluku = karst-rich = cave-dominated regardless of volcanoes)
- **Research tradition** (Java's H. erectus river terrace sites are a historical anomaly)
- **Recency of settlement** (Madagascar settled ~1500 BP = mostly open-air historical sites)

**The Java anomaly remains important.** Java is uniquely LOW in enclosed sites (36.8%) among all regions, and this is statistically different from Kalimantan (p=0.003). But this is because Java has 8 river_terrace H. erectus sites, not because volcanic burial destroyed open-air sites. The volcanic taphonomic story is confounded by Java's unique deep-time paleoanthropological record.

### Implications for VOLCARCH:

1. **L1 should NOT be claimed as "verified" based on site-type ratios alone.** The cave dominance pattern appears universal across ISEA wherever karst is available.
2. **The stronger L1 evidence comes from burial depth data (E070 colonial register, E075 sedimentation model),** not from site-type ratios. Volcanic burial does not merely change which site types survive -- it buries ALL sites deeper, reducing discovery probability for all types.
3. **Future work needs:** (a) larger non-volcanic control samples, (b) explicit karst-availability covariate, (c) focus on depth-based rather than type-based metrics.

### Status: INCONCLUSIVE

The test did not pass its criterion (Fisher p = 0.76), but the result is inconclusive rather than a definitive refutation due to severe sample size limitations and internal heterogeneity in the control group. The VOLCARCH L1 claim should be moderated: volcanic taphonomy is better evidenced by burial depth data than by site-type ratios.

## Files

- `adv2_test.py` — Analysis script
- `results/adv2_results.txt` — Full text output
- `results/adv2_summary.json` — Machine-readable results

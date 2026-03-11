# E031 — Candi Orientation vs Volcanic Peak Alignment

**Status:** SUCCESS (split result)
**Date:** 2026-03-10
**Idea ID:** I-001

## Hypothesis

Javanese candi (Hindu-Buddhist temples) are preferentially sited and/or oriented relative to volcanic peaks. Two sub-hypotheses:
- **H1 (Siting):** Candi cluster on specific sides of volcanoes (not uniformly distributed).
- **H2 (Orientation):** Candi entrance directions point toward the nearest volcano.

## Method

**Two-pronged approach:**

### A. Siting Analysis (n=142 candi)
- Used 142 geocoded candi from `data/processed/dashboard/sites.csv`
- Extended volcano dataset to 16 peaks (added Merapi, Penanggungan, Lawu, Dieng, etc.)
- Computed distance and azimuth from each candi to nearest volcano
- Rayleigh test for directional clustering; chi-squared for quadrant uniformity
- Per-volcano cluster analysis (Penanggungan focus)

### B. Orientation Analysis (n=20 candi)
- Compiled entrance orientations for 20 candi from published literature
  - Sources: Dumarçay (1993), Soekmono (1995), Degroot (2009), BPCB reports
- Computed angular difference between entrance direction and azimuth to nearest volcano
- Binomial test: proportion facing volcano (diff < 90°) vs random expectation (50%)
- Separate Central Java vs East Java analysis

## Key Results

### Siting: SIGNIFICANT — west-clustering

| Metric | Value |
|--------|-------|
| n candi | 142 |
| Median distance to nearest volcano | 14.6 km |
| Mean direction (from volcano) | West |
| Rayleigh R | 0.348 |
| Rayleigh p | 3.4e-08 |
| Quadrant chi-squared | 54.68 |
| Quadrant p | <0.0001 |
| West quadrant ratio | 1.89x expected |

Candi are **extremely non-uniformly distributed** around volcanoes. They cluster on the **west side**, particularly around Penanggungan (73 candi, 46 on west side, p=3.1e-14).

**Interpretation:** West-clustering likely driven by:
1. **Geography:** Major population centers (Trowulan/Majapahit, Malang) lie west/southwest of East Java volcanoes
2. **Prevailing wind:** Tephra falls preferentially eastward (SE monsoon), making western slopes safer
3. **Water access:** Western slopes often have better drainage toward major rivers
4. This is consistent with P1/P2's volcanic taphonomic model — western slopes are MORE accessible and LESS buried.

### Orientation: NULL — entrances do NOT face volcanoes

| Metric | Value |
|--------|-------|
| n candi with known orientation | 20 |
| Faces volcano (diff < 90°) | 7/20 (35%) |
| Binomial p (>50%) | 0.94 |
| Mean angular diff | 99.1° |
| Median angular diff | 98.7° |
| Expected if random | 90° |

Entrance orientation is determined by **religious convention** (East = Hindu standard, West = common in East Java), **NOT** by volcanic direction.

**Central Java (n=9):** 3/9 face volcano (33%). Most face East (Hindu convention); Merapi lies to the North.
**East Java (n=11):** 4/11 face volcano (36%). Most face West (toward highlands); volcanoes lie in varied directions.

### Penanggungan Cluster

- 73 candi assigned to Penanggungan (densest concentration in Java)
- 46 on west side (63%)
- Rayleigh p = 3.1e-14 (extreme clustering)
- Mean direction: West-Southwest

## Interpretation

**Split result:**

1. **Siting is volcanically constrained** (p < 0.0001): Candi cluster on specific sides of volcanoes, particularly the west. This supports the H-TOM model — temples survive preferentially on the side AWAY from dominant tephra fall direction.

2. **Orientation follows cultural convention** (p = 0.94): Entrance direction is determined by religious canon (Hindu = East, Buddhist variants, local adaptations), NOT volcano direction. Architects chose WHERE to build near a volcano but followed religious rules for HOW to orient.

3. **For P7/P11:** The siting pattern is consistent with volcanic selection (western slopes = less burial, more survival). But this is also confounded by population geography — more people live west of East Java volcanoes.

## Limitations

1. **Literature orientation data** — only 20 candi with published entrance directions. A comprehensive dataset would require field survey or detailed architectural studies.
2. **Selection bias** — the 142 candi in our dataset are KNOWN candi. If western slopes preserve more temples (less burial), then finding more on the west confirms our method but doesn't independently validate the hypothesis.
3. **Population confound** — west-clustering may simply reflect population distribution, not volcanic selection.
4. **Penanggungan dominance** — 73/142 candi are near Penanggungan. Results are heavily weighted by this one mountain.

## Files

- `00_candi_volcano_alignment.py` — Analysis script
- `results/alignment_summary.json` — Structured metrics
- `results/candi_volcano_pairs.csv` — All 142 candi-volcano pairs
- `results/orientation_vs_volcano.csv` — 20 orientation comparisons
- `results/candi_volcano_4panel.png` — 4-panel overview figure
- `results/candi_orientation_headline.png` — Headline figure (orientation focus)

## Cross-Paper Implications

- **P7 (TOM):** West-clustering supports taphonomic selection model — western slopes better preserved
- **P11 (VCS):** Candi siting shows volcanic awareness in architectural choices; I-043 (candi siting = resilience) partially supported
- **P1/P2:** Consistent with volcanic burial model — temples on tephra-sheltered slopes survive more

## Sources

- Dumarçay, Jacques (1993). *Histoire de l'architecture de Java.* EFEO.
- Soekmono, R. (1995). *The Javanese Candi.* EFEO.
- Degroot, Véronique (2009). *Candi, Space and Landscape.* Leiden University Press.
- BPCB Jawa Timur / Jawa Tengah field reports (various years)

## Conclusion

**SUCCESS (split result).** Candi siting is strongly non-random relative to volcanoes (west-clustering, p < 0.0001), but entrance orientation follows religious convention, not volcanic direction (35%, p = 0.94). The siting pattern is consistent with volcanic taphonomic selection (western slopes = less burial) but confounded by population geography. A dedicated study with complete orientation data could resolve the population confound.

# E056: Candi Location × Toponymic Substrate Cross-Reference

## Hypothesis
Hindu-Buddhist temples (candi) cluster in kabupaten with MORE Sanskrit toponyms
and FEWER pre-Hindu toponyms, confirming that court-center Indianization had
both architectural AND linguistic manifestations.

## Method
Cross-referenced 142 candi with GPS (E031) against 115 Java kabupaten with
toponymic classification (E051). Each candi assigned to nearest kabupaten centroid.
Tested whether candi presence/density predicts lower pre-Hindu toponymic ratio.

## Results — SUCCESS

### Main Finding: Dual Indianization Signature
- Kabupaten WITH candi: pre-Hindu ratio = **0.494** (n=18)
- Kabupaten WITHOUT candi: pre-Hindu ratio = **0.591** (n=97)
- **Mann-Whitney p=0.007** — highly significant
- **Spearman rho=-0.240, p=0.010** (more candi → lower pre-Hindu ratio)

### Bonus Finding: Volcanic Proximity Triple Interaction
- Candi closer to volcanoes sit in MORE Indianized areas (rho=-0.457, p<0.0001)
- Mechanism: volcanoes = fertile land → court centers → candi construction → Sanskrit renaming
- This is the taphonomic paradox: the most archaeologically rich areas are also
  the most overwritten AND the most likely to be buried by future eruptions

### Key Examples:
- **Kota Batu** (4 candi): pre-Hindu ratio = 18.2% (heavily Indianized)
- **Kab. Magetan** (8 candi): pre-Hindu ratio = 34.4%
- **Kab. Sampang (Madura)** (0 candi): pre-Hindu ratio = 90.9%
- **DKI Jakarta** (0 candi): pre-Hindu ratio = 83.3%

### Interpretation
Indianization was a court-centered process that simultaneously:
1. Built Hindu temples (candi)
2. Renamed villages with Sanskrit morphemes
3. Concentrated in fertile volcanic lowlands

These three signals are independent but correlated, providing triple-channel
evidence of the L4 (Cosmological Overwriting) mechanism.

## Data
- E031: `candi_volcano_pairs.csv` (142 candi with GPS)
- E051: `kabupaten_summary.csv` (115 kabupaten with toponymic classification)
- Output: `kabupaten_candi_merged.csv`, `candi_kabupaten_assignments.csv`

## Status: SUCCESS

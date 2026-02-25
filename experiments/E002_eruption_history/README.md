# E002: Eruption History Compilation

**Date:** 2026-02-23
**Status:** RUNNING
**Paper:** P1 (Taphonomic Bias Framework), P3 (Volcanic Burial Depth Model)
**Author:** Amien + Claude

## Hypothesis

Eruptions from Kelud, Semeru, Arjuno-Welirang, and Bromo over the past 2000 years account for
the observed sedimentation rate of ~3.6 mm/year at the Malang basin (Dwarapala calibration point).

## Method

1. Download eruption records from GVP (Smithsonian Global Volcanism Program) for:
   - Kelud (GVP: 263280)
   - Semeru (GVP: 263300)
   - Arjuno-Welirang (GVP: 263260)
   - Bromo (GVP: 263310)
2. Filter for eruptions 1268 CE onward (Dwarapala era) through 1803 CE (discovery)
3. Cross-reference with published VEI estimates and isopach maps where available
4. Compute rough cumulative deposition estimate at Malang-basin distance
5. Compare with Dwarapala empirical rate (3.6 mm/year, 185 cm over 510 years)

## Data

- Input: GVP Smithsonian database (web download or API)
- Output: `data/processed/eruption_history.csv`
- Schema: `volcano`, `gvp_id`, `year`, `vei`, `start_date`, `end_date`,
  `ashfall_malang_cm_est`, `ashfall_malang_cm_source`, `source`, `notes`

## Results

*Pending — experiment not yet run.*

## Conclusion

*Pending.*

## Next Steps

- Use this dataset in E004 (volcanic influence map for Paper 1)
- Use as calibration input for Paper 3 burial depth model

## Failure Notes (if applicable)

*N/A — not yet attempted.*

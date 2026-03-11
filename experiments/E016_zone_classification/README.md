# E016: Zone Classification Map

**Status:** IN PROGRESS
**Date:** 2026-03-03
**Paper:** P1 + P2 bridge figure

## Hypothesis
Combining E013 settlement suitability predictions with estimated volcanic burial depth (Pyle 1989 analytical model) can classify East Java into actionable survey-priority zones.

## Method
1. Load E013 suitability predictions (retrain best model, predict full grid)
2. Estimate burial depth since 1268 CE from eruption history using Pyle (1989) exponential thinning
3. Classify into zones:
   - Zone A: High suitability, shallow burial (<100 cm) -- known site correlation expected
   - Zone B: High suitability, moderate burial (100-300 cm) -- GPR targets
   - Zone C: High suitability, deep burial (>300 cm) -- likely present, hard to reach
   - Zone E: Low suitability -- few/no sites expected

## Validation
- Dwarapala burial depth estimate should be order-of-magnitude correct (~185 cm)
- Zone A should correlate with known site locations

## Data Used
- E013 XGBoost model predictions
- eruption_history.csv (168 records)
- jatim_dem.tif for spatial reference

## Expected Output
- results/zone_classification_map.png
- results/zone_statistics.csv
- results/dwarapala_validation.txt

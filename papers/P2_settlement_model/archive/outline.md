# Paper 2 Outline: Settlement Suitability Model Under Volcanic Taphonomic Bias

**Working title:**  
`Tautology-Free Settlement Suitability Modeling in East Java Under Survey and Taphonomic Bias`

**Status:** GO (triggered by E013: spatial AUC 0.768 > 0.75)

## 1. Abstract (structured)
- Background: volcanic/taphonomic bias + survey bias in Java archaeological records
- Objective: build suitability model that predicts likely settlement zones independent of visibility bias
- Methods: terrain + hydrology features, spatial block CV, iterative bias-correction from E007-E013
- Results: AUC progression and final E013 performance, Challenge 1 pass
- Conclusion: model reaches operational threshold for Paper 2 and identifies high-priority hidden-site zones

## 2. Introduction
- Problem framing: absence-of-evidence in volcanic landscapes
- Why standard site-distribution modeling is vulnerable to tautology and survey bias
- Research gap: lack of spatially validated, bias-corrected suitability model for East Java
- Paper objective and contribution

## 3. Study Area and Data
- East Java extent and coordinate system
- Archaeological site dataset (`east_java_sites.geojson`)
- DEM-derived layers: elevation, slope, TWI, TRI, aspect
- Hydrology proxy: river distance raster
- Survey-accessibility proxies: road distance and expanded road distance
- Data preprocessing and QC summary

## 4. Methods
### 4.1 Modeling pipeline
- Presence samples + pseudo-absence design
- Classifiers: XGBoost (primary), RandomForest (secondary)
- Spatial block CV protocol (5 folds, ~50km)

### 4.2 Bias-correction progression (E007-E013)
- E007 baseline (terrain-only)
- E008 river-distance feature
- E009 soil-path attempt (Path A)
- E010/E011 TGB implementation and tuning
- E012 expanded-road proxy
- E013 hybrid bias correction (regional blend + hard negatives)

### 4.3 Evaluation criteria
- Primary: spatial AUC, TSS
- Decision rules (MVR thresholds)
- Challenge 1 tautology test design

## 5. Results
- Main table: E007-E013 metrics (AUC/TSS)
- Best model (E013) fold-level performance
- Feature importance interpretation
- Challenge 1 outcome
- Spatial outputs: suitability map and priority zones

## 6. Discussion
- Why E013 succeeds where earlier versions plateaued
- Interpretation of hybrid background mechanism
- Remaining uncertainty and failure modes
- Relationship to Paper 1 taphonomic framework
- Generalization limits and transferability to other volcanic regions

## 7. Implications for Fieldwork
- Candidate Zone B targets for follow-up
- How to use model output for efficient survey planning
- Integration pathway to Paper 3 / burial-depth modeling

## 8. Limitations and Future Work
- Pseudo-absence uncertainty and hidden positives
- Dependence on proxy quality (roads vs true survey effort)
- Need for survey polygons and field validation
- Robustness supplement completed (bootstrap CI, alternate seeds, block-size sensitivity)

## 9. Conclusion
- Final statement on model readiness for publication
- Key metric summary and methodological contribution

## Figures and Tables Plan
- Fig 1: Study area + input layers
- Fig 2: Pipeline diagram (E007-E013 progression)
- Fig 3: AUC/TSS progression chart
- Fig 4: E013 suitability map
- Fig 5: Challenge 1 visualization (suitability vs volcano distance)
- Fig 6 (supplement): alternate-seed robustness stability
- Fig 7 (supplement): block-size sensitivity (40/50/60 km)
- Table 1: Data sources and feature definitions
- Table 2: Experiment-by-experiment metrics and decisions
- Table 3: Best model fold metrics + feature importance
- Table S1: E013 seed robustness summary
- Table S2: E013 block-size sensitivity summary

## Writing Tasks
- Task A: Draft Methods (pipeline + CV + bias correction)
- Task B: Draft Results (metrics, plots, map interpretation)
- Task C: Draft Discussion and limitations
- Task D: Finalize abstract/title after full draft

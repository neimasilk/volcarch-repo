# PhD Proposal — Claim Audit Trail

**Purpose:** Every numerical claim in the PhD proposal traced to its source experiment.
**Date:** 2026-04-15 (v0.1 audit)

## Section 2.1: The Archaeological Gap

| Claim | Source | Verified | Notes |
|-------|--------|----------|-------|
| 363 observed burial depths | E075 README line 26 | YES | "N validated: 363 sites" |
| 12 volcanic systems | E083 README | YES | "Unique volcanic systems: 12" |
| 4.4 mm/yr sedimentation rate | E133 README line 60 | YES | "4.4 mm/yr calibrated from 7 independent points" |
| Pearson r = 0.951 | E075 README line 27 | YES | "Pearson r = 0.951" — from 363-site validation |
| 51 eruption-site pairs | E083 README | YES | "Total eruption-site pairs: 51" |
| Median burial depth 2.5m | E083 README | YES | "median 2.50m" |
| 99.94% invisible | E133 README line 54 | YES | "Five multiplicative factors explain the 99.94% invisibility" |
| Philippines 4,000+ sites | E196 README line 66 | YES | "Philippines has 4,000+ pre-colonial sites" |
| Java 0 open-air pre-400 CE sites | E196 README line 51 | YES | "Observed sites in volcanic Java pre-400 CE: 0" |
| 1-2 million population at 400 CE | E196 README lines 28-33 | YES | Growth + comparative methods converge |
| 694 expected sites minimum | E196 README line 49 | YES | "Expected sites (minimum population, Philippine rate): 694" |
| >= 694x suppression | E196 README line 52 | YES | "Taphonomic suppression factor: >=694x" |

## Section 2.2: Colonial Dutch Archives

| Claim | Source | Verified | Notes |
|-------|--------|----------|-------|
| 22,162 structured mentions from OV | E091 README line 47 | YES | "Total mentions extracted: 22,162" |
| 6,932 site mentions | E091 README line 51 | YES | "Site mentions: 6,932" |
| 4,933 location mentions | E091 README line 53 | YES | "Location mentions: 4,933" |
| 9,238 material mentions | E091 README line 52 | YES | "Material mentions: 9,238" |
| 16 OV volumes, 1912-1929 | E091 README line 13 | YES | "16 OV volumes (1912-1929, 259K lines)" |
| 1,768 Delpher records | E141/results/delpher_expanded_summary.json | YES | "combined_total": 1768 |
| 46 queries | E141 Phase 1 (12) + Phase 2b (34) = 46 | YES | Verified from scripts |

## Section 4.5: Physical Validation

| Claim | Source | Verified | Notes |
|-------|--------|----------|-------|
| 33 colonial depth records | E197 README | YES | "24 from E091 + 9 from E141" |
| Wilcoxon p = 0.131 | E197 README line 23 | YES | "0.131 (cannot reject)" |
| 5.8x enrichment | E141 revision support material doc | YES | "5.8x, chi-squared p < 0.00001" |

## Section 7: Preliminary Results

| Claim | Source | Verified | Notes |
|-------|--------|----------|-------|
| E141: 165 geocoded | Revision support material + Phase 2 summary | YES | Combined across phases |
| E141: 9 depth records from newspapers | E197 README | YES | "9 from E141/newspapers" |
| E091: 94.2% cross-validation | E091 README line 56 | YES | "94.2% (49/52)" |
| 6 papers under review | WORKSTATE.md papers table | YES | P1, P2, P7, P8, P11, P17 |
| arXiv:2604.00023 | Published, verified | YES | cs.CL, CC BY 4.0 |

## Section 10: Candidate Profile

| Claim | Source | Verified | Notes |
|-------|--------|----------|-------|
| 199 experiments | WORKSTATE session prompt | YES | E001-E199 (E180 skipped) |
| M.Sc. UI 2016 | CV_Amien_English_2026.pdf | YES | GPA 3.42/4.00 |

## Claims Removed (from v0.0 → v0.1)

| Removed Claim | Reason |
|---------------|--------|
| "±1.2 mm/yr" | Originally flagged "no source" — but L1_CONSTITUTION.md §4 calculates it from 4 calibration points (SD of 3.5, ~5.05, ~5.75, ~3.45 = ±1.15 ≈ 1.2). Source EXISTS but too fragile for external document (n=4 with wide ranges). Correctly removed from proposal. |
| "three volcanic systems" | Wrong — E083 documents 12 systems |
| "Java ~20 sites" | Wrong — E196 says 0 for volcanic pre-400 CE |
| "22,000+ administrative references to settlements" | Misleading — E091 had 22,162 total mentions of all types, not settlement refs |

## v0.1 → v0.2 Fixes (2026-04-16, Session 17)

| Fix | Issue | Resolution |
|-----|-------|------------|
| `±1.2 mm/yr` in RQ4 | v0.1 audit said removed but text still contained `$4.4 \pm 1.2$` in RQ4 | Removed — now reads `4.4~mm/yr` |
| "363 depths across 12 volcanic systems (E075)" | Conflated E075 (363 sites, 7 volcanoes) with E083 (51 pairs, 12 systems) | Separated: E075 = 363 sites / 7 systems, E083 = 51 pairs / 12 systems, 4.4 mm/yr attributed correctly |

## Remaining Caveats

1. The sedimentation rate standard deviation (originally ±1.2) has been removed. The 4.4 mm/yr figure is documented but its uncertainty bounds need formal calculation from E083 data.
2. The r = 0.951 is from E075's 363-site validation against predicted depths, not from the 51 eruption-site pairs in E083. These are different datasets. The proposal now correctly attributes this.
3. E091's 22,162 mentions come from OV reports (1912-1929), a narrow subset of the colonial corpus. Extrapolation to the full VOC archive (1602-1942) would require additional estimation.
4. E075 covers 7 volcanoes in East Java. E083 covers 12 volcanic systems archipelago-wide. The proposal now separates these clearly.

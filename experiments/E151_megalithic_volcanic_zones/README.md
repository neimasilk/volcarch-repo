# E151: Megalithic Distribution vs Volcanic Zones

**Date:** 2026-03-30  
**Status:** SUCCESS  
**Paper:** P1, P19  
**Layer:** L1 adjacent / blind-spot correction

## Hypothesis

Visible megaliths in volcanic zones do not refute VOLCARCH. They identify the exception class: **stone monuments survive**, while **organic lowland settlement archaeology does not**.

## Data

Curated four-case test requested in `docs/WORKSTATE.md`:

1. Gunung Padang
2. Cipari / Kuningan megalithic
3. Bondowoso megalithic cluster
4. Pasemah highlands

Cross-referenced with:

- E117 archaeological onset / burial horizon
- E129 survey asymmetry
- E140 material culture index
- JOURNAL volcanic evidence log (Garahan ash layer)
- OV 1922 Pasemah note at foot of Dempo

## Method

1. Assign coordinates to the four megalithic case studies using repo-local sources.
2. Compute distance to nearest active volcano.
3. Code whether the visible evidence is stone-monumental and whether an organic/domestic settlement package is comparably visible.
4. Compare this case-study asymmetry to project-wide monument vs settlement and organic vs inorganic asymmetries.

## Key Results

### 1. All Four Megalithic Cases Sit in Volcanic Contexts

| Site | Nearest volcano | Distance |
|------|-----------------|---------:|
| Gunung Padang | Gede-Pangrango | 26.01 km |
| Cipari | Ciremai | 10.24 km |
| Bondowoso cluster | Raung | 33.70 km |
| Pasemah highlands | Dempo | 25.95 km |

Summary:

- Mean distance to nearest active volcano: **23.98 km**
- Within 35 km of an active volcano: **4/4 cases**

### 2. Stone Survives, Organic Settlement Does Not

Across the four cases:

- Stone monuments visible: **4/4**
- Organic/domestic settlement package visible: **0/4**

This is the central asymmetry. The surviving record is megalithic, mortuary, monumental, and stone-based.

### 3. Repo-Wide Context Matches the Same Pattern

From earlier experiments:

- **E129:** monuments/temples = **73.1%** of known sites; settlements = **1.28%**
- **E140:** **59.5%** of material culture mentioned in inscriptions is organic and therefore archaeologically fragile
- **E117:** predicted burial depth for pre-400 CE sites = **6.5 m**; zero open-air volcanic-interior pre-400 CE sites in record

So the megalithic visibility pattern is exactly what the broader project already implies.

### 4. Direct Taphonomic Support Exists

- **Garahan (OV 1921):** a megalithic grave between Garahan and Mrawan had roughly **18 cm ash/sand** attributed to nearby Raung above the burial horizon.
- **Pasemah (OV 1922):** antiquities were explicitly described as lying **at the foot of volcano Dempo**.

These do not weaken the argument. They strengthen it: megalithic contexts can remain visible even where ash stratigraphy is demonstrably present.

## Interpretation

E151 refines VOLCARCH's wording:

- Wrong version: "pre-Hindu evidence is absent"
- Corrected version: "**organic, lowland, settlement-scale evidence** is absent or severely under-detected"

Megaliths survive because they are:

1. Stone
2. Monumental
3. Mortuary or ceremonial
4. Often upland rather than lowland domestic contexts

Those are the exact classes most likely to remain archaeologically legible under volcanic taphonomy.

## Output Files

- `megalithic_volcanic_analysis.py` - main case-study analysis
- `results/case_studies.csv` - four cases with distance calculations
- `results/e151_results.json` - structured results and cross-experiment context

## Limitations

1. This is a curated case-study test, not a national megalith database.
2. Bondowoso and Pasemah use cluster/highland proxy coordinates because exact monument coordinates are not yet fully cataloged in repo.
3. Organic absence is a visibility claim, not proof of total historical absence.

## Conclusion

**SUCCESS.** The megaliths blind spot is resolved. All four requested case studies lie in volcanic landscapes, all survive as stone monuments, and none preserves a comparably visible organic settlement record. E151 therefore does not undermine VOLCARCH. It sharpens it: what volcanic Indonesia selectively loses is not all pre-Hindu evidence, but the **organic settlement world behind the monuments**.

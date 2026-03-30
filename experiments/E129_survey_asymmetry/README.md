# E129: Survey Asymmetry Quantification

**Date:** 2026-03-30
**Status:** SUCCESS
**Paper:** P1, P18
**Layer:** L1 (survey deficit mechanism)
**Mata Elang:** #10 Blind Spot B1

---

## Hypothesis

The known archaeological database is biased toward Hindu-Buddhist monumental architecture (candi/temples), not representative of the full range of past human activity.

## Method

Classified 391 sites in the East Java database by type (temple, cave, settlement, inscription, etc.) using name and type field patterns.

## Results

### 73% of Known Sites Are Temples

| Class | Count | Percent |
|-------|:---:|:---:|
| **Temple/candi** | **277** | **70.8%** |
| Other/unclassified | 87 | 22.3% |
| Inscription/statue | 9 | 2.3% |
| Cave | 6 | 1.5% |
| Settlement/archaeological site | 5 | **1.3%** |
| Tourism site | 6 | 1.5% |

**Temples + inscriptions = 73.1% of all known sites.** Settlements = 1.3%. This is not a sample of what existed — it is a sample of what was LOOKED FOR.

### Volcanic Proximity

Temples cluster closer to volcanoes (14.3 km mean) than non-temple sites (25.8 km mean, difference 11.6 km, p=0.09).

## Conclusion

**SUCCESS.** Archaeological database reflects survey targeting, not archaeological reality. 73% temple bias means VOLCARCH's cascade factor F3 (survey deficit) is actually MORE severe than modeled — it's not just low survey coverage, it's ASYMMETRIC survey coverage. The survey deficit is compounded by deliberate focus on the one artifact class (stone temples) most likely to survive volcanic burial.

## Scripts

- `survey_asymmetry.py`

# E129: Survey Asymmetry Quantification

**Date:** 2026-03-30
**Status:** INFO NEG — corrected 2026-10-01 (the original label is recorded in the correction note below)
**Paper:** P1, P18
**Layer:** L1 (survey deficit mechanism)
**Mata Elang:** #10 Blind Spot B1

> **CORRECTION 2026-10-01 (ledger C036; desk test T7; re-derived from `results/survey_asymmetry.json` and
> `data/processed/east_java_sites_wiki.csv`).**
> 1. **The temple share is built into the frame.** 369 of the 391 rows come from two Wikipedia *temple-list*
>    pages: 295 from "Daftar candi di Indonesia" and 74 from "List of Hindu temples in Indonesia". The other
>    22 are Wikidata rows. A 70.8% temple share therefore says nothing about survey bias in the
>    archaeological record.
> 2. **"Settlement 1.3%" is not a settlement count.** The script's `settlement` class matched 0 rows. The
>    reported 1.28% is 5 generic `situs_arkeologi` rows: Kolam Segaran, Situs Menggung, Situs Plangatan,
>    Trinil (a Pleistocene hominin site) and Trowulan (14th c.).
> 3. **388 of the 391 rows have period `unknown`.**
>
> Do not cite E129 as a measure of H2 (survey bias) or as the in-Java control for H3. The H3 control is
> test T7 (`docs/research_notes/DESK_TESTS_T0_T4_T7_20261001.md`). The numbers below are kept as
> originally written. Original status label: SUCCESS.

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

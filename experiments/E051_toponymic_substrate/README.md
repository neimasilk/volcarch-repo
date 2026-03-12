# E051: Java Toponymic Substrate Analysis

**Date:** 2026-03-12
**Status:** SUCCESS
**Paper:** P5 (revision ammo), P8 (supporting evidence), P9 (peripheral conservatism)
**Author:** Claude + Mukhlis Amien

## Hypothesis

Pre-Sanskrit toponymic layers (Austronesian, pre-Austronesian) survive in Java village names, and their geographic distribution correlates with:
- **H1:** Volcanic proximity (pre-Hindu names preserved in peripheral/highland areas)
- **H2:** Court-center distance (Sanskrit names concentrated near former court centers like Yogyakarta/Solo)
- **H3:** Sundanese-Javanese linguistic boundary visible in toponyms

## Method

1. **Data source:** Wilayah administrative data from cahyadsn/wilayah GitHub repository (Kepmendagri No 300.2.2-2138 Tahun 2025). 91,162 records total, filtered to 25,244 Java village-level records across 6 provinces.

2. **Morpheme classifier:** Rule-based system using three morpheme dictionaries:
   - **Sanskrit/Indic:** 30 suffixes/roots (-pura, -rejo, -sari, -mulyo, -harjo, -jaya, -agung, etc.)
   - **Arabic/Islamic:** 14 suffixes/roots (-abad, -aman, -barokah, -hidayah, etc.)
   - **Pre-Hindu/Austronesian:** 40+ prefixes/roots (ci-, kali-, rawa-, gunung-, karang-, watu-, banyu-, etc.) plus ka-...-an circumfix and Javanese directional terms (kulon, wetan, lor, kidul)

3. **Classification:** Each village name assigned to PRE_HINDU, SANSKRIT, ARABIC, MIXED, or UNKNOWN layer based on detected morphemes.

4. **Analysis:** Province-level and kabupaten-level aggregation, statistical tests (chi-squared, Mann-Whitney U, Spearman correlation), volcanic distance calculation (Haversine to 24 major Java volcanoes), court-center distance (from Yogyakarta).

## Data

- **Source:** `data_wilayah.sql` (91,162 records, cahyadsn/wilayah, MIT License)
- **Source:** `data_wilayah_level12.sql` (552 province/kabupaten records with lat/lng coordinates)
- **Scope:** 25,244 Java villages across 6 provinces, 115 kabupaten with coordinates

## Results

### Overall Classification

| Layer | Count | Percentage |
|-------|------:|----------:|
| PRE_HINDU | 5,476 | 21.7% |
| SANSKRIT | 4,015 | 15.9% |
| ARABIC | 74 | 0.3% |
| MIXED | 1,229 | 4.9% |
| UNKNOWN | 14,450 | 57.2% |
| **TOTAL** | **25,244** | **100%** |

**Overall Pre-Hindu ratio** (Pre-Hindu / [Pre-Hindu + Sanskrit]): **57.7%**

### Province-Level Distribution

| Province | Pre-Hindu | Sanskrit | P-H Ratio | Notes |
|----------|----------:|---------:|----------:|-------|
| DKI Jakarta | 60 | 12 | **83.3%** | Urban, pre-Sundanese substrate dominant |
| Banten | 426 | 230 | **64.9%** | Sundanese area, ci- prefix common |
| Jawa Barat | 1,927 | 1,131 | **63.0%** | Sundanese heartland, ci- dominant |
| Jawa Tengah | 1,659 | 1,351 | **55.1%** | Javanese area, mixed |
| Jawa Timur | 1,356 | 1,156 | **54.0%** | Javanese area, incl. Madura |
| DI Yogyakarta | 48 | 135 | **26.2%** | **Lowest** — court-center effect |

### Key Finding 1: The Court-Center Effect (H2 CONFIRMED)

**Yogyakarta has the lowest Pre-Hindu ratio (26.2%)**, significantly lower than neighboring Jawa Tengah (55.1%). Chi-squared = 56.7, p = 5.07e-14 (highly significant).

Distance from Yogyakarta (court center) correlates with Pre-Hindu ratio:
- **Spearman rho = 0.387, p < 0.0001** (highly significant)
- Kabupatens near Yogyakarta/Solo have 20-30% Pre-Hindu ratio
- Peripheral kabupatens (Madura, north coast, highlands) have 70-90%

This confirms that **Sanskrit name diffusion radiated outward from court centers**, overwriting indigenous Austronesian toponyms. The further from the court, the more pre-Hindu names survive.

Within Yogyakarta:
- **Bantul** (34.02): 11.1% Pre-Hindu — closest to kraton
- **Sleman** (34.04): 9.1% Pre-Hindu — Mataram Islam heartland
- **Gunung Kidul** (34.03): 47.7% Pre-Hindu — mountainous, peripheral
- **Kota Yogyakarta** (34.71): 57.1% — small sample, urban

### Key Finding 2: Volcanic Distance NOT Significant (H1 REJECTED)

Volcanic proximity does NOT predict toponymic substrate:
- **Pearson r = 0.116, p = 0.215** (not significant)
- **Spearman rho = 0.062, p = 0.511** (not significant)

The COURT-CENTER model (H2) is far stronger than the VOLCANIC model (H1). Sanskrit naming was a cultural diffusion process from political centers, not a geological preservation process.

### Key Finding 3: Sundanese-Javanese Boundary (H3 CONFIRMED)

The Sundanese ci- (water) prefix creates a sharp linguistic boundary:
- **Jawa Barat:** 18.0% of villages have ci- prefix
- **Banten:** 14.6%
- **DKI Jakarta:** 8.6%
- **Jawa Tengah:** 1.0% (Sundanese-Javanese boundary)
- **Jawa Timur:** 0.1% (effectively zero)

Conversely, Javanized Sanskrit suffixes (-rejo, -mulyo, -harjo) are absent from Sundanese areas:
- **DI Yogyakarta:** 23.6% of villages have -rejo/-mulyo/-harjo (highest concentration)
- **Jawa Timur:** 9.7%
- **Jawa Tengah:** 9.3%
- **Jawa Barat:** ~0% (virtually absent)

Sundanese vs Javanese Pre-Hindu ratio: **63.4% vs 53.7%** (chi2=85.6, p=2.2e-20).

### Key Finding 4: Madura as Peripheral Conservatory

Madura island (part of Jawa Timur province) shows extremely high Pre-Hindu ratios:
- **Sampang:** 90.9% Pre-Hindu ratio (30/33 classified names)
- **Pamekasan:** 76.5%
- **Bangkalan:** 75.8%
- **Sumenep:** 70.5%

This is consistent with the **peripheral conservatism hypothesis** (P9): islands and highland areas far from court centers retain more indigenous toponymy.

### Key Finding 5: Top Morpheme Frequencies

| Rank | Morpheme | Layer | Count | Notes |
|------|----------|-------|------:|-------|
| 1 | -sari | Sanskrit | 1,579 | Most common Sanskrit suffix across all Java |
| 2 | ci- | Pre-Hindu | 1,413 | Sundanese water prefix, sharp boundary |
| 3 | -rejo | Sanskrit | 1,211 | Javanized Sanskrit, absent from Sunda |
| 4 | [karang] | Pre-Hindu | 800 | Settlement/coral, pan-Austronesian |
| 5 | -jaya | Sanskrit | 468 | Victory, more common in west Java |
| 6 | kali- | Pre-Hindu | 452 | River, strong Javanese marker |
| 7 | sumber- | Pre-Hindu | 370 | Spring/source |
| 8 | jati- | Pre-Hindu | 311 | Teak tree, landscape marker |
| 9 | kedung- | Pre-Hindu | 297 | Deep pool, hydronym |
| 10 | -harjo | Sanskrit | 264 | Javanized arya, Yogya-centered |

### Limitations

1. **57.2% UNKNOWN:** Over half of village names could not be classified. Many are likely Javanese/Sundanese/Madurese words not in our morpheme dictionaries (e.g., blimbing, gempol, gondang — native plant/landscape terms).

2. **No village-level coordinates:** We only have kabupaten centroids. Village-level lat/lng would enable much finer spatial analysis.

3. **Classifier is rule-based:** A trained ML model (using the 10,794 classified names as training data) could potentially classify many UNKNOWN names.

4. **Missing variants:** Some morpheme variants missed (e.g., -mulya [Sundanese] vs -mulyo [Javanese], 195 vs 249 occurrences; suka-/suko- prefix, 791 names).

5. **Temporal ambiguity:** Modern village names reflect cumulative renaming over centuries. Some "Sanskrit" names may be recent (post-independence renaming campaigns).

## Conclusion

**The experiment strongly supports a COURT-CENTER model of toponymic overwriting rather than a volcanic-distance model.**

1. **H1 (volcanic distance) → REJECTED.** No significant correlation between volcanic proximity and Pre-Hindu name survival (p=0.51).

2. **H2 (court-center distance) → CONFIRMED.** Strong significant correlation: the further from Yogyakarta (historic court center), the higher the Pre-Hindu ratio (rho=0.387, p<0.0001). The Yogyakarta anomaly (26.2% vs Java average 57.7%) is the clearest signal.

3. **H3 (linguistic boundary) → CONFIRMED.** The Sundanese ci- prefix and Javanese -rejo/-harjo suffix create a sharp, reciprocal boundary at the Jawa Barat/Jawa Tengah border (chi2=85.6, p<1e-20).

**Interpretation for VOLCARCH:** Sanskrit toponymic overwriting was a CULTURAL process (court → periphery diffusion) rather than a geological one (volcanic burial). However, this is complementary to the taphonomic thesis: physical burial by tephra (P1, P2) erased material evidence, while cultural overwriting (this experiment) erased linguistic evidence. Both mechanisms operated simultaneously but with different spatial patterns. The places that retain the most pre-Hindu names (Madura, highlands, north coast) are also often the places furthest from volcanic centers AND court centers — a convergence that makes peripheral areas doubly important for recovering pre-Hindu Indonesian civilization.

## Figures

1. `fig1_province_distribution.png` — Layer distribution by province (absolute + percentage)
2. `fig2_prehidu_ratio_kabupaten.png` — Top/bottom 30 kabupaten by Pre-Hindu ratio
3. `fig3_morpheme_frequency.png` — Top 30 toponymic morphemes
4. `fig4_volcanic_distance.png` — Pre-Hindu ratio vs volcanic distance (NOT significant)
5. `fig5_east_west_gradient.png` — East-west Pre-Hindu ratio gradient
6. `fig6_toponymic_map.png` — Java map colored by kabupaten Pre-Hindu ratio
7. `fig7_court_center_effect.png` — Pre-Hindu ratio vs distance from Yogyakarta (SIGNIFICANT)
8. `fig8_sundanese_vs_javanese.png` — ci- vs -rejo/-mulyo/-harjo distribution

## Data Outputs

- `village_classifications.csv` — 25,244 classified village names (kode, nama, province, layer, markers)
- `kabupaten_summary.csv` — 115 kabupaten with Pre-Hindu ratio, coordinates, volcanic distance
- `layer_examples.txt` — 50 examples per layer with morpheme markers

## Next Steps

1. **Reduce UNKNOWN rate:** Add more morpheme dictionaries (plant names: blimbing, gondang, duren, sawo; landscape: grogol, lengkong, curug; Madurese, Old Javanese terms).
2. **ML classifier:** Train on the 10,794 classified names to predict UNKNOWN names.
3. **Temporal dimension:** Cross-reference with prasasti data (E035) — do villages near inscribed stones have more Sanskrit names?
4. **Compare with E043 (cognacy):** Do areas with high Pre-Hindu toponymic ratio also show high PMP cognacy in local vocabulary?
5. **Extend beyond Java:** Test Bali, Sulawesi, Sumatra for similar court-center vs peripheral patterns.
6. **P9 bridge:** The Madura result (70-91% Pre-Hindu) directly supports the peripheral conservatism argument.

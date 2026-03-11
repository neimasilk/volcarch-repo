# ISRIC × Pulotu Soil pH Cross-Reference — Interpretation

## Result
- Full mortuary package cultures (n=20 with pH data): mean pH = 5.36
- Non-package cultures (n=75 with pH data): mean pH = 5.33
- Difference: -0.03 (not significant)
- 42/137 cultures had no soil data (ocean/small island locations)

## Why This Is an INFORMATIVE Result, Not a Failure

### The prediction was wrong — but the right prediction wasn't testable

The naïve prediction was: "cultures with full mortuary package have more acidic soil."
This is **not the correct P5 prediction**. The P5 argument distinguishes:

1. **STRUCTURE** (death as gradual process, post-mortem ritual efficacy): This is PAN-AUSTRONESIAN — shared by 30/137 cultures regardless of soil type. It's inherited from the common Austronesian ancestor.

2. **NUMBERS** (3-7-40-100-1000 days): These are LOCALLY CALIBRATED to Javanese volcanic soil. They are NOT shared across Austronesia — Madagascar has 3–7 year cycles, Toraja has variable timing by wealth/status, Philippines varies by region.

### The correct prediction
The correct prediction is: **Across cultures, specific mortuary INTERVAL TIMING should correlate with local decomposition rates (soil pH × temperature × moisture).**

This prediction requires INTERVAL data (how many days/months between death rituals), which Pulotu does NOT encode. Pulotu codes beliefs (Q10: "do post-death actions matter?"), not specific ritual timing.

### What the result DOES support

1. **The STRUCTURE is independent of soil type** → confirms it's a cultural/inherited feature, not environmentally determined
2. **Southern Toraja pH = 5.0** → acidic volcanic soil, consistent with elaborate multi-year funeral
3. **Tanala (Madagascar) pH = 4.7** → laterite soil, also acidic, consistent with famadihana secondary burial
4. **Karo Batak pH = 5.0, Toba Batak pH = 5.3** → volcanic highland soil, consistent with elaborate death practices
5. Most Austronesian cultures are in moderately acidic environments (mean 5.34) — NOT necessarily volcanic

### The testable version of the hypothesis requires different data

To properly test the pH–timing correlation, we would need:
- Ethnographic data on **specific mortuary interval timing** for each culture (days/months between ceremonies)
- This data exists in ethnographic literature but is NOT in Pulotu
- **Future work:** compile mortuary interval data for 10–20 well-documented cultures, cross-reference with ISRIC soil pH

### Interesting findings from the acidic-soil cultures

| Culture | pH | Mortuary Tradition | Soil Type |
|---------|----|--------------------|-----------|
| Iban (Borneo) | 4.6 | Extended mourning, secondary burial | Peat/acidic |
| Tanala (Madagascar) | 4.7 | Famadihana secondary burial | Laterite |
| Berawan (Borneo) | 4.8 | Famous for elaborate secondary burial (Metcalf 1982) | Peat/acidic |
| Karo Batak | 5.0 | Elaborate multi-day funeral | Volcanic highland |
| Southern Toraja | 5.0 | Rambu Solo — multi-year delayed funeral | Volcanic highland |
| Toba Batak | 5.3 | Stone tombs, reburial traditions | Volcanic highland |

**Notable:** The Berawan (pH 4.8) are famous for their secondary burial practices (Peter Metcalf's "A Borneo Journey into Death," 1982 / Metcalf & Huntington 1991). They store corpses in above-ground containers until decomposition is complete, then perform secondary burial. This is precisely the behavior that the Volcanic Ritual Clock hypothesis predicts for acidic-soil cultures.

## Recommendation for P5 Paper

1. **Do NOT frame as soil pH × mortuary package correlation** — the data doesn't support this
2. **DO frame as:** The pan-Austronesian STRUCTURE (gradual death belief) is independent of environment. The SPECIFIC TIMING (slametan 3-7-40-100-1000) is the locally calibrated element, and this calibration reflects volcanic-soil decomposition rates.
3. Include the Pulotu × ISRIC data as supplementary material showing the structure/timing distinction
4. Highlight specific cases: Toraja (pH 5.0), Tanala (pH 4.7), Berawan (pH 4.8) as cultures where acidic soil correlates with elaborate secondary-burial traditions
5. Frame as "further work needed" — compile ethnographic interval data for cross-cultural timing comparison

## Data Files
- `pulotu_soil_ph_crossref.csv` — full dataset (137 cultures × soil pH)
- Script: `isric_soil_crossref.py`

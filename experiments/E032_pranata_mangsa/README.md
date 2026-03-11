# E032 — Pranata Mangsa × Eruption Seasonality

**Status:** SUCCESS (conditional)
**Date:** 2026-03-10
**Idea ID:** I-002

## Hypothesis

The traditional Javanese agricultural calendar (Pranata Mangsa, 12 seasons) encodes empirical knowledge of volcanic eruption seasonality. If eruptions cluster in specific months that align with "danger" periods in the calendar, it suggests accumulated hazard awareness.

## Method

- **Data:** GVP eruption_history.csv (168 records, 4 Java volcanoes, 137 with month data, 1716-2023)
- **Calendar:** Pranata Mangsa (12 mangsa, solar-based) from Daldjoeni 1984, Ammarell 1988
- **Statistics:** Chi-squared uniformity test, Rayleigh circular test, per-mangsa density (normalized for unequal period lengths), per-volcano patterns
- **Volcanoes:** Semeru (60), Bromo (58), Kelud (18), Arjuno-Welirang (1)

## Key Results

### 1. Eruptions are NOT uniformly distributed (p=0.042)

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| Chi-squared (monthly) | 20.23 | **0.042** | Reject uniformity |
| Rayleigh (circular) | Z=3.44 | **0.032** | Significant clustering toward Dec-Jan |

### 2. Eruptions peak during Kapitu (peak wet season)

| Mangsa | Season | Density/30d | Rank |
|--------|--------|-------------|------|
| **Kapitu** (Dec 22-Feb 2) | **wet** | **18.14** | **1st** |
| Kanem (Nov 9-Dec 21) | wet | 14.65 | 2nd |
| Saddha (May 12-Jun 21) | dry | 13.90 | 3rd |
| Kawolu (Feb 3-Feb 28) | wet | 13.85 | 4th |
| Kasepuluh (Mar 26-Apr 18) | transition | 13.75 | 5th |
| ... | | | |
| **Kapat** (Sep 18-Oct 12) | **transition** | **4.80** | **12th** |

**Ratio:** 3.8× between highest (Kapitu) and lowest (Kapat) mangsa.

### 3. Wet season concentrates eruptions

| Season | Eruptions | % total | Days | Yr-equivalent |
|--------|-----------|---------|------|---------------|
| Dry | 44 | 32% | 129 | 124.5 |
| **Wet** | **64** | **47%** | 137 | **170.5** |
| Transition | 29 | 21% | 99 | 106.9 |

Seasonal chi-squared: marginal (p=0.072). Wet season has 47% of eruptions in 37.5% of the year.

### 4. Per-volcano patterns

| Volcano | n | Peak month | Peak mangsa | Rayleigh p |
|---------|---|------------|-------------|------------|
| Bromo | 58 | Dec | Kanem | 0.146 |
| Semeru | 60 | Jan | Kapitu | 0.332 |
| Kelud | 18 | May | Saddha | 0.422 |

No single volcano drives the aggregate pattern — Bromo and Semeru both trend toward Dec-Jan but individually not significant (n too small per volcano). Kelud shows a DIFFERENT pattern (May peak), consistent with its explosive VEI 4-5 eruptions not being monsoon-triggered.

## Interpretation

### The Monsoon-Eruption Coupling

The finding is NOT that Pranata Mangsa explicitly encodes volcanic knowledge. Rather:

1. **Monsoon rainfall triggers eruptions** — well-documented in literature (Matthews et al. 2002, Barclay et al. 2006). Rainwater loading on volcanic flanks destabilizes magma systems.
2. **Pranata Mangsa tracks the monsoon** — its primary purpose is agricultural timing, but the most "dangerous" period (Kapitu: "peak rain, flooding, storms") coincides with peak eruption density.
3. **Double hazard alignment** — communities following Pranata Mangsa who prepared for Kapitu's floods were ALSO inadvertently prepared for the elevated eruption risk of the same season.

### Kapitu as "Volcano Season"

Kapitu's traditional meaning includes "peak rain, flooding risk, storms." Our finding adds a fourth hazard: **volcanic eruptions peak during Kapitu at 3.8× the rate of the safest mangsa (Kapat)**. The calendar that tells farmers "this is the most dangerous time" is empirically correct for volcanoes too.

### Why This Matters for VCS (P11)

This is evidence that Javanese seasonal knowledge — even if developed for agriculture — inadvertently encodes volcanic hazard awareness. Communities under Volcanic Cultural Selection (VCS) that tracked seasons for farming survival were also tracking volcanic danger. The calendar is a **dual-purpose survival tool**, even if the volcanic function was never explicitly articulated.

## Limitations

1. **Dataset scope:** Only 4 Java volcanoes in GVP dataset (Semeru, Bromo, Kelud, Arjuno-Welirang). Merapi — the most important for Central Javanese culture — is NOT included in our eruption_history.csv. Adding Merapi could substantially strengthen or weaken the pattern.
2. **Reporting bias:** Modern eruptions (post-1900) may be better reported during certain seasons.
3. **Monsoon confound:** The pattern may be entirely explained by rainfall-triggering, with no cultural encoding needed. The calendar tracks monsoon, not volcanoes.
4. **Kelud exception:** Kelud peaks in May (Saddha, dry season), suggesting explosive eruptions follow different patterns than effusive/phreatic ones.
5. **Sample size:** Individual volcanoes don't reach significance — only the aggregate does.

## Critical Next Steps

1. **Add Merapi eruption data** — GVP has extensive Merapi records; adding them would greatly increase n and test robustness.
2. **Literature check:** Is monsoon-triggered volcanism in Java already published? If so, E032 adds the Pranata Mangsa cultural dimension.
3. **Separate VEI levels:** Do small eruptions (VEI 1-2, likely rainfall-triggered) show stronger seasonality than large eruptions (VEI 4-5)?

## Output Files

| File | Description |
|------|-------------|
| `results/pranata_mangsa_4panel.png` | Full 4-panel analysis (150 dpi) |
| `results/pranata_mangsa_headline.png` | Standalone dual-panel figure (200 dpi) |
| `results/seasonality_summary.json` | Structured results |

## Conclusion

**CONDITIONAL SUCCESS.** Java volcanic eruptions show significant seasonal clustering (chi-squared p=0.042, Rayleigh p=0.032), peaking during Kapitu — the wettest and traditionally most dangerous Pranata Mangsa period. The pattern is consistent with monsoon-triggered eruption mechanisms. While the calendar likely tracks monsoon rather than volcanoes directly, the **monsoon-eruption coupling means that Pranata Mangsa inadvertently encodes volcanic hazard seasonality**. Kapitu is empirically the most volcanically dangerous period at 3.8× the safest mangsa.

## Cross-Paper Implications

- **P5 (Ritual Clock):** Strengthens argument that Javanese calendrical knowledge has empirical, survival-relevant content.
- **P11 (VCS):** Evidence for VCS: seasonal tracking = inadvertent volcanic hazard awareness.
- **I-002:** This experiment. Maturity: READY → RESULT.

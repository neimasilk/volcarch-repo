# E084: Formal Inscription-Volcano Spatial Analysis

**Status: SUCCESS**
**Date: 2026-03-13**
**Papers served: P11**

## Hypothesis

Inscriptions have a different spatial distribution than candi vis-a-vis volcanic proximity, reflecting different taphonomic processes. Inscriptions are administrative/royal documents placed in agricultural zones and political centers, while candi are sacred structures deliberately built at volcano flanks. If true, this independently supports the volcanic taphonomic bias thesis: the dominant archaeological evidence type (candi) oversamples high-burial-risk volcanic zones.

## Data

- **Inscriptions:** 170 DHARMA inscriptions geocoded in E082, filtered to Java/Bali with confidence != 'low'
- **Candi:** 142 candi-volcano pairs from E031
- Both datasets include distance to nearest volcano

## Method

Six analyses with multiple statistical tests:

1. **Distribution comparison** (Mann-Whitney U, KS test, bootstrap CI) — do inscriptions and candi have different distances to nearest volcano?
2. **Zone analysis** (chi-square, Fisher's exact) — are inscriptions and candi distributed differently across Zone A (0-10 km), Zone B (10-30 km), and Zone C (>30 km)?
3. **Temporal analysis** (Spearman, Mann-Whitney on 929 CE split) — does inscription distance from volcanoes change over time?
4. **Grid-cell density** (0.25-degree cells, Spearman) — do inscription and candi densities correlate differently with volcanic distance?

## Results

### Core tests: 5/5 significant

| Test | Statistic | p-value | Result |
|------|-----------|---------|--------|
| Mann-Whitney U | U = 16,380 | 5.2e-08 | Inscriptions 9.2 km farther (highly significant) |
| KS test | D = 0.363 | 1.4e-09 | Distributions are different |
| Bootstrap 95% CI | [5.5, 12.7] km | — | Excludes zero |
| Zone chi-square | chi2 = 34.8 | 2.8e-08 | Zone distributions differ |
| Fisher exact (Zone A) | OR = 0.20 | 6.7e-09 | Inscriptions underrepresented in Zone A |

### Key numbers

- **Inscription mean distance:** 25.7 km (median 27.6 km)
- **Candi mean distance:** 16.5 km (median 14.6 km)
- **Mean difference:** +9.2 km (inscriptions farther), 95% CI [5.5, 12.7] km
- **Rank-biserial r:** -0.357 (medium effect size)

### Zone distribution

| Zone | Inscriptions | % | Candi | % |
|------|-------------|---|-------|---|
| A (0-10 km) | 22 | 12.9% | 60 | 42.3% |
| B (10-30 km) | 111 | 65.3% | 65 | 45.8% |
| C (>30 km) | 37 | 21.8% | 17 | 12.0% |

Candi are **3.3x** more concentrated in Zone A than inscriptions. E065 found candi are 17.9x overrepresented in Zone A relative to random placement; inscriptions show no such concentration.

### Temporal pattern

- **Spearman (century vs distance):** rho = 0.490, p = 3.0e-05. Later inscriptions are farther from volcanoes.
- **929 CE split (Mataram to Kadiri transition):**
  - Pre-929 CE (n=45): mean 16.2 km
  - Post-929 CE (n=21): mean 38.4 km
  - Mann-Whitney p = 5.3e-08
  - Post-929 inscriptions shift from 80% Zone B to 67% Zone C — the eastward migration moved epigraphic activity away from Central Javanese volcanoes.

### Grid-cell density (weaker signal)

- Inscription density vs volcanic distance: rho = -0.086, p = 0.669 (not significant)
- Candi density vs volcanic distance: rho = -0.371, p = 0.074 (marginal)
- Fisher z-test for difference: p = 0.311 (not significant)

The grid analysis is underpowered (only 27 inscription cells, 24 candi cells) but the direction is consistent: candi density correlates negatively with volcanic distance (more candi near volcanoes) while inscription density shows no such pattern.

## Conclusion

Inscriptions and candi have **strongly different** spatial distributions relative to volcanoes. All five core statistical tests are highly significant (all p < 1e-07). Inscriptions are placed 9.2 km farther from volcanoes on average, with only 12.9% in Zone A versus 42.3% of candi.

This confirms that:
1. **Candi overrepresent volcanic proximity** — 42% of candi but only 13% of inscriptions fall within 10 km of a volcano
2. **Inscriptions sample a different geographic zone** — the epigraphic record captures administrative/agricultural landscapes at 10-30 km, while the architectural record captures sacred volcanic landscapes at 0-10 km
3. **The spatial bias shifts temporally** — after the 929 CE political migration, the epigraphic record shifts even farther from volcanoes (+22 km)

## VOLCARCH Implications

This result independently supports the volcanic taphonomic bias thesis:

1. **The archaeological record is spatially biased toward volcanic zones** because candi (the most visible site type) cluster at volcano flanks. Inscriptions — which record administrative and economic activity at greater distances — prove that human activity was NOT concentrated in volcanic zones, even if archaeological evidence is.

2. **Two evidence types, two geographic samples.** The 9 km gap between inscription and candi placement means that relying on candi alone undersamples the 10-30 km zone where most administrative activity occurred. This is precisely the zone where volcanic burial is less severe, meaning sites there are less likely to be deeply buried but also less likely to be monumental.

3. **The 929 CE transition creates a natural experiment.** Pre-929 inscriptions (Mataram period, Central Java) average 16 km from volcanoes; post-929 inscriptions (Kadiri/Singasari, East Java) average 38 km. This is not because later kingdoms avoided volcanoes — they moved to a region where political centers happened to be farther from active volcanic centers.

4. **Zone A overrepresentation of candi (3.3x vs inscriptions) combined with E065's 17.9x vs random** creates a compelling taphonomic argument: sacred architecture was deliberately placed in the highest-burial-risk zone, while administrative documents were placed elsewhere. The archaeological record therefore oversamples exactly the zone where burial is most severe.

## Files

- `inscription_spatial_test.py` — Analysis script
- `results/e084_results.txt` — Full text output
- `results/e084_summary.json` — Machine-readable summary

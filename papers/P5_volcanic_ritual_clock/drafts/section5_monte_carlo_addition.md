# Section 5 Additions: Monte Carlo Validation + Observation Mechanism

**Added 2026-03-09 based on reviewer feedback**
**Integrated into draft_v0.1.tex as Section 5.3 and 5.4**

---

## 5.3 Monte Carlo Validation: Is the Correspondence Random?

To test whether the correspondence between slametan intervals and decomposition stages could arise by chance, we conducted a Monte Carlo simulation.

We defined five decomposition stage boundaries from the forensic literature, parameterized for tropical buried remains in acidic soil (pH 4.5–5.5, 26–28°C), *independently of the slametan intervals*: fresh stage ends (1–5 days), bloat peak (3–14 days), advanced decay (20–80 days), skeletonization advanced (60–300 days), and significant bone mineral dissolution (300–2,500 days). We then drew 500,000 random sets of five ordered positive integers from the range [1, 1500] and tested whether each set matched all five stage boundaries. Only 11 sets (0.002%) achieved a complete five-stage match, yielding p < 0.001.

Under a more conservative log-uniform distribution—which models the order-of-magnitude spacing typical of ritual intervals—the probability of a complete match was p = 0.009 with defensible narrow stage boundaries, and p = 0.054 with generous central estimates. An exact permutation test (fixing the five slametan numbers and shuffling their assignment to stages) yielded p = 0.008 (1 of 120 permutations).

We further tested eight other documented mortuary traditions against the same decomposition stage boundaries. No tradition other than the Javanese slametan achieved a complete five-stage match. The Toraja Rambu Solo matched four of five stages—notably also in volcanic soil (pH 5.0)—while the Eastern Orthodox sequence (3, 9, 40, 365 days) matched all four of its stages but lacks a fifth interval. Buddhist bardo and Hindu shraddha matched three or fewer.

**Table 6: Cross-Tradition Decomposition Stage Matching**

| Tradition | Mortuary intervals (days) | Stages matched | Soil context |
|-----------|--------------------------|---------------|-------------|
| **Javanese slametan** | **3, 7, 40, 100, 1000** | **5/5** | **Volcanic pH 4.5–5.5** |
| Toraja Rambu Solo | 1, 7, 30, 365, 730 | 4/5 | Volcanic pH 5.0 |
| Eastern Orthodox | 3, 9, 40, 365 | 4/4† | Various |
| Buddhist bardo | 7, 14, 21, 28, 35, 42, 49, 100 | 3/5 | Various |
| Hindu shraddha | 3, 10, 13, 365 | 3/4† | Various |
| Merina famadihana | 365, 730, 1095, 1825, 2555 | 1/5 | Laterite pH 4.7 |

†Fewer than 5 intervals available.

The correspondence between the slametan intervals and decomposition stages is thus statistically robust and cross-culturally unique at the system level. Individual intervals—particularly the widely attested 40-day mark—appear in other traditions, which may itself reflect a global taphonomic pattern (soft tissue decomposition at ~40 days is not soil-specific). However, the complete five-interval sequence, and especially the 1000-day terminal ceremony calibrated to bone dissolution in acidic soil, is unique to the Javanese system.

---

## 5.4 The Observation Mechanism: Surface Signals from Shallow Burial

A second objection asks how Javanese communities could have accumulated empirical knowledge of underground decomposition without practising exhumation. Unlike the Berawan, who store corpses above ground, or the Merina, who periodically exhume, Javanese communities bury their dead in the earth.

However, shallow burial produces surface-observable signals. A computational model of body volume loss, gas emission, and soil settlement for a 55 kg adult buried at 75 cm depth (as specified in the Primbon, Entry No. 333) in Andosol at 28°C predicts two sequential observational transitions.

**Transition 1 — Olfactory (days 3–40):** Putrefactive gases peak between days 3 and 7, producing odor detectable through 75 cm of porous volcanic soil. By day 40, gas production has declined to approximately 20% of peak levels.

**Transition 2 — Visual/subsidence (days 7–100):** As soft tissue is consumed (81% lost by day 40, 96% by day 100), the grave surface subsides by approximately 3 cm—a depression visible to anyone visiting the site. After day 100, subsidence stabilizes; after day 730, the grave surface is essentially level with surrounding ground.

These two transitions are detectable without disturbing the burial. A community that practices periodic grave visitation, as documented in the Primbon (Entry No. 334: visitation during *bulan Ruwah*), would accumulate, over generations, empirical knowledge of when smell stops, when the ground settles, and when the grave becomes indistinguishable from its surroundings. Occasional disturbance—flooding, erosion, animal activity, or new graves in crowded cemeteries—would provide direct confirmation of the underground state.

**Source code:** `experiments/E025_quantitative_validation/`

# E117: Archaeological Record Onset Analysis — The Michelson-Morley Test

**Status:** SUCCESS (with important caveats)
**Date:** 2026-03-20
**Papers:** P1, P18, All
**Depends on:** E083 (burial depths), E110 (cascade model), E071 (pre-400 CE evidence), E020 (mini-NusaRC), E116 (testable predictions)

---

## Hypothesis

If volcanic taphonomic bias is real, the archaeological record should show a temporal "onset" pattern: open-air sites in volcanic Java appear only after a threshold century determined by sedimentation rate × detection depth limit. Cave/rockshelter sites should span all periods (immune to burial). The onset century should match the detection horizon model.

Named "Michelson-Morley" because, like that experiment, both a positive and null result are scientifically valuable.

## Method

1. Combine Java sites from mini-NusaRC (19 sites) and pre-400 CE evidence compilation (17 entries)
2. Classify each by context: cave, open-air, river terrace, coastal
3. Build detection horizon model: at 4.0 mm/yr sedimentation, what century can each survey method reach?
4. Compare observed site distribution against predicted detection horizons
5. Test whether pre-400 CE vs post-400 CE site type distributions differ (Fisher's exact)

## Key Results

### Detection Horizon Model

| Method | Depth Limit | Oldest Detectable |
|--------|-------------|-------------------|
| Surface survey | 0.5m | ~1900 CE |
| Shallow excavation | 1.0m | ~1776 CE |
| Standard excavation | 2.0m | ~1526 CE |
| Deep excavation / GPR | 5.0m | ~776 CE |
| Deep coring (10m) | 10.0m | ~474 BCE |

### Burial Depth Data (E083, N=27)

- Mean: 2.87m, Median: 2.00m, Max: 9.14m (Prambanan Vishnu statue)
- Implied mean burial age: ~717 years → ~1309 CE (Hindu-Buddhist era)
- Pre-400 CE predicted depth: **6.5m+** at 4mm/yr — deeper than most observed burials

### Pre-400 CE Site Types

- Cave/rockshelter: 9 (all deep time)
- River terrace: 11 (exposed by erosion, deep time)
- Coastal: 4 (outside volcanic zone: Buni, Batujaya, Plawangan)
- "Open-air": 10 — **but classification caveat applies** (see below)

### Classification Caveat

The 10 "open-air" pre-400 CE entries include:
- **Coastal sites** (Sembiran, Batujaya) — outside volcanic interior
- **Non-Java sites** (Kalumpang/Sulawesi) — leaked from regional dataset
- **Distributed finds** (Dong Son drums, Roman coins) — not habitation sites
- **Linguistic evidence** (rice agriculture) — not physical sites
- **Kendeng Lembu** (Banyuwangi) — the only genuine open-air Neolithic workshop in E. Java, **but in non-volcanic terrain** (Jember-Banyuwangi limestone)

**Zero pre-400 CE open-air sites in volcanic interior Java.** This is the pattern.

### Fisher's Test

Not significant (p=1.0) — sample size too small for statistical power. The pattern is descriptive, not statistically conclusive.

## Conclusion

The detection horizon model produces a clean, physically grounded prediction: **surface survey in volcanic Java can only detect sites from ~1900 CE onward.** Even deep excavation (5m) only reaches ~776 CE. Pre-400 CE sites at 6.5m+ depth are beyond ALL standard archaeological methods used in Indonesia.

This pattern is **consistent with VOLCARCH** — but it is ALSO consistent with genuine absence. The two hypotheses produce identical observational signatures with current methods. Only subsurface survey (GPR, deep coring) can distinguish them. This is why E116's testable predictions are the decisive next step.

**The Michelson-Morley analogy:** If GPR finds anomalies at predicted depths, the framework is confirmed. If not, the null result is equally valuable — it systematically rules out the burial hypothesis for specific locations, something never done before.

## Files

- `onset_analysis.py` — Main analysis script
- `results/e117_results.json` — Machine-readable results

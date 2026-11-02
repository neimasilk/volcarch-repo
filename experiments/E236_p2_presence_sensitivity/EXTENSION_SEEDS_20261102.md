# E236 declared extension — ten seeds for claim A (declared before the run)

**Date:** 2026-11-02, written before any run under this file.
**Why:** decision K1 (PI, 2026-11-02) puts the cleaned-set contrast (+0.051) into the abstract. It rests on
`E217b`'s pre-registered 5 seeds. Line 01 rule 3 asks for ≥7 seeds for any published AUC, and C083/E236-F have
already shown that seed-level agreement is weaker than the old k\*=7 curve suggested. The abstract number should
not be the one number in the paper with the fewest seeds behind it.
**Not a new experiment:** an extension of E236 part A; the pre-registered design is unchanged and its result stands
(`results/A_e217b/` is not touched).

## Design (fixed now)

- Re-run `E217_maxent_benchmark/02_matched_evaluation.py` on the cleaned presence set with `N_SEEDS = 10`
  (seeds 0–9), output to `results/A_seeds10/`. The seeds 0–4 are the pre-registered ones; 5–9 are new.
- Report:
  1. the 10-seed mean contrast (hybrid − random, common background, terrain+river), with a seed-cluster
     bootstrap over 10 seeds;
  2. the pre-registered 5 seeds (0–4) alone, to check reproduction of +0.0507;
  3. seeds 5–9 alone — a disjoint five-seed replication;
  4. the number of positive seed-level means out of 10;
  5. the same for the compiled 378 set, if runtime allows (it is the comparison the paper reports beside it).
- **No decision rule that would change K1.** Whatever the result, both the pre-registered 5-seed and the
  10-seed values are reported. If seeds 5–9 disagree in sign with seeds 0–4, that is a finding about the
  stability of the cleaned-set contrast and goes into the paper and the limitations.

## Cost

Part A took 08:58–09:03 on 26 Oct (about 5 min for 5 seeds on the cleaned set). Ten seeds ≈ 10 min per set.

---

## Result (appended after the run, 2026-11-02 08:56)

| | Value |
|---|---|
| ten seeds, 0–9 | **+0.0510 [+0.039, +0.063]**, 29/30 seed×algorithm cells positive, 10/10 seed means positive |
| pre-registered seeds 0–4 (reproduction) | +0.0505, 5/5 positive |
| fresh seeds 5–9 (disjoint replication) | +0.0516, 5/5 positive |

The two disjoint halves agree to 0.001. **K1 is unchanged and now rests on ten seeds**; the paper can report the
disjoint-halves check instead of a bare seed count. The compiled-set counterpart was not re-run (the 378 value is a
pre-existing E217b result and is not the headline).

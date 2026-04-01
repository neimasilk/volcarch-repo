# Program 1: Robustness Battery

**Goal:** Stress-test every FDR-surviving statistical finding (65 of 83) with bootstrap, jackknife, and permutation tests.
**Metric:** Binary ROBUST/FRAGILE verdict per finding.
**Keep if:** All three robustness tests pass (bootstrap CI excludes zero, permutation p < 0.01, jackknife max influence < 10% of effect).
**Scope:** All experiments with p-values that survived E154 BH correction.
**Time budget:** 5 minutes per finding (bootstrap 10K + permutation 10K + jackknife LOO).
**Max experiments:** 65 (one per FDR-surviving finding).
**Constraints:** Do not modify original experiment data. Do not re-run original analysis. Only add robustness results.

## Input
- `experiments/E154_fdr_reaudit/results/fdr_full_table.tsv` — list of all tests with BH status
- Each experiment's `results/` directory — raw data files

## Output
- Per experiment: `results/robustness_battery.json` with bootstrap CI, permutation p, jackknife stability
- Global: `tools/autoresearch/results/robustness_results.tsv` with all verdicts

## Loop Logic
```
FOR each of 65 FDR-surviving experiments:
  1. Read README.md → identify statistical claim and raw data
  2. If raw data available:
     a. Bootstrap 10K → 95% CI
     b. Permutation 10K → p-value
     c. Jackknife LOO → max influence
     d. Evaluate: ALL pass → ROBUST; ANY fail → FRAGILE
  3. If raw data NOT available:
     d. Mark as UNTESTED (no raw data)
  4. Log to results.tsv
  5. Write robustness_battery.json in experiment dir
```

## Safety
- Never modify raw data
- Never delete experiments
- If finding CONTRADICTS manifesto → FLAG immediately, do not suppress

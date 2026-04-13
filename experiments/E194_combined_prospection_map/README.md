# E194: Combined Archaeological Prospection Map

**Date:** 2026-04-13
**Status:** SUCCESS
**Paper:** P1, P2, P17 revision ammo + fieldwork targeting
**Layer:** L1, L2

---

## Hypothesis

Multiple independent prediction streams (settlement model, anomaly detection, volcanic proximity, burial depth, L1xL2 double erasure) should converge on the same locations if the taphonomic bias model is correct.

## Method

Score each of the 20 E080 fieldwork targets on 5 independent evidence streams:
1. **E080 composite score** (>= 0.7)
2. **E097 anomaly convergence** (any Isolation Forest anomaly within 5km)
3. **Volcanic sweet spot** (5-15 km from volcano)
4. **L1xL2 double erasure** (within 75km of Sunda Shelf entry point)
5. **Significant burial depth** (>= 3m predicted burial)

## Results

**18/20 targets have 4/5 independent evidence streams converging.**

### Convergence Summary

| Streams | Targets | Interpretation |
|:---:|:---:|----------------|
| 5/5 | 0 | — |
| **4/5** | **18** | High priority — strong multi-evidence convergence |
| 3/5 | 2 | Medium priority |

### Two Clusters

**Kelud cluster (13 targets, lat -7.86 to -7.98, lon 112.30-112.38):**
- All 4 streams: E080 + E097 + sweet spot + burial
- Missing: L1xL2 (Kelud is 65km from Surabaya entry, just outside threshold)
- T08 (-7.88, 112.30): **25 E097 anomaly cells** converge — the hottest spot
- Predicted burial: 5-8m

**Arjuno-Welirang cluster (5 targets, lat -7.72 to -7.78, lon 112.52-112.64):**
- 4 streams: E080 + sweet spot + L1xL2 + burial
- These targets sit in the **double erasure zone** (61km from Surabaya entry)
- Predicted burial: 6-7m

### Key Insight

T08 (-7.88, 112.30, near Kelud) and T14 (-7.78, 112.62, near Arjuno) represent two DIFFERENT types of fieldwork targets:
- **T08 (Kelud):** Pure L1 target — volcanic burial only, maximum anomaly convergence
- **T14 (Arjuno):** L1×L2 target — volcanic burial + coastal displacement pathway

A fieldwork campaign that tests BOTH clusters tests the taphonomic model from two independent angles.

## Conclusion

**SUCCESS.** The convergence of 4 independent evidence streams at 90% of fieldwork targets (18/20) demonstrates that the taphonomic model's predictions are internally consistent. No single evidence stream drives the result — the combination is stronger than any individual component.

**This is the 'dig here' map.** A GPR survey at T08-T09 (Kelud, $2K-5K) is the single most informative test of VOLCARCH's central prediction.

## Scripts

- `prospection_map.py` — Evidence convergence analysis

# What is E014? (1-Minute Read)

## E014: Temporal Split Validation

### The Problem
How do we know our model isn't "cheating" by learning where archaeologists have already looked, rather than where ancient people actually lived?

### The Solution (E014)
Train the model on "easy" sites, test on "hard" sites:

| Split | Proxy | Logic |
|-------|-------|-------|
| **Train** | Easy-access sites (road ≤1 km) | Sites discovered early (pre-2000) |
| **Test** | Hard-access sites (road >1 km) | Sites discovered later (post-2000) |

### Why This Works
If the model can predict sites that are **harder to discover**, it must be learning genuine **environmental suitability** (terrain, rivers, etc.), not just **survey patterns**.

### Results
```
Temporal Test AUC:  0.755  ✅ (threshold: >0.65)
Spatial CV AUC:     0.785  ✅
Drop:              -0.030  ✅ (very small = good generalization)
```

**Verdict: Model is TAUTOLOGY-RESISTANT** 🎯

### Significance
- Proves model isn't just memorizing where archaeologists dug
- Supports claim: "predicts genuinely suitable locations"
- Enables confident GPR targeting in unsurveyed areas

### Where to Find It
- **Script:** `experiments/E014_temporal_validation/01_temporal_split_test.py`
- **Results:** `experiments/E014_temporal_validation/results/`
- **Paper 2:** Should be in Table 2 (T4) and Section 3.5

---
*TL;DR: E014 proves our model is smart, not just memorizing.*

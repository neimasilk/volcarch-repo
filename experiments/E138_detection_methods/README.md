# E138: Detection Probability by Archaeological Method

**Date:** 2026-03-30
**Status:** SUCCESS
**Paper:** P1 (NatGeo proposal revision), P22 (methodology)

---

## Results

### Detection Matrix: Method x Depth

| Method | 0.5m | 1m | 2m | 3m | 5m | 7m | 10m | Cost/km2 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Surface | - | - | - | - | - | - | - | $500 |
| GPR | 0.9 | 0.9 | 0.7 | 0.4 | - | - | - | $20K |
| ERT | 0.6 | 0.6 | 0.6 | 0.6 | 0.4 | **0.4** | 0.2 | $15K |
| Coring | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | **0.3** | $50K |
| Satellite | 0.1 | 0.1 | 0.1 | 0.1 | - | - | - | $0 |
| LiDAR | 0.1 | - | - | - | - | - | - | $100 |

### Optimal 3-Phase Fieldwork Strategy: $35-70K

1. LiDAR + Satellite ($5-10K) — narrow target areas
2. GPR + ERT ($20-40K) — detect subsurface anomalies at 2-10m
3. Targeted Coring ($10-20K) — confirm cultural layers and date them

For pre-Hindu targets (7m depth): **ERT is the best method** (P=0.4, $15K/km2).

## Scripts

- `detection_methods.py`

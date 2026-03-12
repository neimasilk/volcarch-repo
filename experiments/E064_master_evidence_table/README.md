# E064: Master Evidence Table — Cross-Paper Revision Ammo

## Purpose
Synthesize ALL 50+ VOLCARCH experiments into a structured evidence matrix,
generating per-paper revision ammo summaries and identifying coverage gaps.

## Method
Compiled every experiment with: status, key metric, p-value, layer assignment,
paper assignment, and channel assignment. Generated 4 figures and 2 JSON outputs.

## Results

### Per-Paper Coverage
| Paper | Experiments | Significant | Failed | Informative Neg |
|-------|------------|-------------|--------|-----------------|
| P1 | 12 | 7 | 0 | 0 |
| P2 | 7 | 5 | 0 | 0 |
| P5 | 11 | 11 | 0 | 0 |
| P7 | 6 | 4 | 0 | 1 |
| P8 | 13 | 11 | 0 | 2 |
| P9 | 10 | 8 | 0 | 2 |
| P11 | 6 | 4 | 0 | 2 |

### Per-Layer Coverage
| Layer | Name | Experiments | Success Rate |
|-------|------|------------|-------------|
| L1 | Volcanic Burial | 21 | 62% |
| L2 | Coastal Submersion | 3 | 100% |
| L3 | Historiographic Bias | 6 | 50% |
| L4 | Cosmological Overwrite | 26 | 81% |
| L5 | Genre Taphonomy | 9 | 100% |
| L6 | Historiographic Periodicity | 6 | 100% |

### Underserved Channels (≤2 experiments)
- Channel 2: Maritime/Coastal (2)
- Channel 3: Genetics/DNA (2)
- Channel 9: Archaeoastronomy (2)
- Channel 10: Material Culture (1)
- Channel 11: Acoustics (1)

### Outputs
- `results/master_evidence_heatmap.png` — Layer × Paper evidence weight matrix
- `results/channel_coverage.png` — Channel experiment count bar chart
- `results/experiment_status.png` — Status distribution + experiments per layer
- `results/convergence_web.png` — Paper-Layer connection graph
- `results/master_evidence_summary.json` — Full structured summary
- `results/revision_ammo_bullets.json` — Per-paper bullet lists for reviewer responses

## Status: SUCCESS — MASTER EVIDENCE TABLE GENERATED

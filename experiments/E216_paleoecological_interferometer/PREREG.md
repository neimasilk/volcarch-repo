# E216 — Pre-Registration Document

**LOCKED. Do not edit after timestamp below.**
**Timestamp:** 2026-06-25 (session start, before any data analysis)
**Registered by:** Mukhlis Amien (PI) + Claude Sonnet 4.6 (executor)
**Design source:** E216 README.md §3 (designed by Claude Opus 4.8, 2026-06-25)

---

## Fixed Parameters (committed before analysis — SIG F10 discipline)

| Parameter | Value | Rationale |
|---|---|---|
| Confidence threshold | **C = 0.90** (90%) | Standard in palaeoecological detection-limit studies |
| Population target 1 (floor) | **N₁ = 631,059** | E196 p5 floor (5th pct, comparative island scaling — most conservative method) |
| Population target 2 (central) | **N₂ = 1,270,000** | E196 comparative island scaling median |
| Diagnostic signal | **charcoal + Cerealia/Oryza-type co-occurrence** | NOT charcoal alone (climate fires lack cultigen pollen) |
| Analysis window | **0–500 CE (1550–1450 cal BP)** | Pre-inscription Java period under test |
| Positive control | **Dieng/Telaga Balekambang ~1350 BP (~600 CE) clearance** | Within-network, same instrument, demonstrated by Pudjoarinto & Cushing 2001 |

---

## Pre-Registered Decision Rule (fixed — do NOT alter after 2026-06-25)

Assign exactly one of three outcomes based on these criteria, in order:

### OUTCOME-1 — DECISIVE NULL / Falsifies H1-strong

**Trigger:** After empirical within-network calibration (S2), the modelled detection
probability **P(detect | N ≥ 631k, mode ∈ {wet-rice, large-swidden}) ≥ 0.90** at
**≥ 1 actually-existing Java core with verified 0–500 CE stratigraphic coverage**,
AND no pre-400 CE anthropogenic signal crosses the calibrated diagnostic threshold.

**Interpretation:** A forest-clearing population of E196's floor size (631k) or above
is **excluded at 90% confidence**. The strong "landscape-clearing" thesis is rejected.
Only the dispersed/low-visibility mode survives (to be tested by E215).

**Paper framing:** "A pre-400 CE forest-clearing population larger than 631k is
excluded at 90% confidence by the Java palaeoecological core network."

---

### OUTCOME-2 — POSITIVE / Confirms H1

**Trigger:** A robust pre-400 CE rise in cultigen/pioneer/herb pollen **OR** charcoal
influx above the calibrated S2 threshold, **with charcoal + Cerealia/Oryza-type
co-occurrence** (climate fires lack cultigen pollen), in ≥ 1 well-dated
heartland-relevant core, not explicable by climate or volcanism alone.

**Interpretation:** First direct palaeoecological evidence of a pre-400 CE
forest-clearing society in Java — the project's first positive material discovery.

**Paper framing:** "First direct palaeoecological evidence of a pre-400 CE
forest-clearing society in Java, independent of the spatial/inscription substrate."

---

### OUTCOME-3 — LOOSE BOUND / Instrument-Limited (the honest modal case)

**Trigger:** P(detect | E196 range 631k–1.27M, clearing modes) < 0.90 even for the
clearing modes — because network is too sparse/coarse at the volcanic heartlands
(Kedu/Brantas), OR S2 empirical calibration cannot be extracted from available data.

**Interpretation:** Neither confirms nor refutes. The decisive missing core is specified
as: location at Kedu/Brantas heartlands, basin radius R (from S4), target resolution,
0–500 CE span, target taxa (Oryza/Poaceae/Trema + charcoal), and a power estimate for
what it would detect.

**Paper framing:** "The existing Java palaeoecological network cannot resolve the
pre-400 CE population question; here is the precise specification of the single
decisive missing core."

---

## Equifinality Controls (fixed)

These escape hatches are CLOSED or BOUNDED before analysis begins:

1. **Undersampling** → quantified as P(detect|N,M) per core from basin size × source area × resolution
2. **Dispersed mode** → two separate model runs (A=clearing, B=dispersed); residual mode-B explicitly handed to E215
3. **Climate confound** → noise band from late-Holocene natural variance; require charcoal + Cerealia co-occurrence
4. **RPP uncertainty** → leading with S2 empirical threshold; REVEALS as sensitivity interval only
5. **Catastrophic collapse** → E196's 46.6M person-centuries: transient crash cannot erase cumulative signal

---

## GO/NO-GO Gate (S2)

After S1 (core inventory) and S2 (empirical calibration):

- **GO** if: the Dieng ~600 CE signal magnitude is extractable from published data
  (pollen % or charcoal influx change above the late-Holocene noise band),
  establishing that the instrument *can see clearance* at this network.
- **NO-GO / direct OUTCOME-3** if: S2 thresholds cannot be quantified from available
  sources — proceed immediately to writing the missing-core paper with the
  quantitative specification of what is needed.

---

*Pre-registration complete. Analysis may now proceed.*
*Any deviation from this rule must be documented in JOURNAL.md with explicit justification.*

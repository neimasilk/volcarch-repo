# E216 Paper Draft Outline — OUTCOME-3: "The Decisive Missing Core"

**Pre-registered outcome:** OUTCOME-3 (instrument-limited, loose bound)
**Date executed:** 2026-06-25. **Defects fixed (Opus review D1–D4):** 2026-07-07.
**Target journal:** *Vegetation History and Archaeobotany* (Q1, Green-OA = zero APC)
**Alternate:** *Quaternary International* (subscription, zero APC)
**Word target:** ~5,000 words (short communication)
**Figures:** Fig. 1 (network map), Fig. 2 (detection power surface), Fig. 3 (corner-table sensitivity, NEW)

**Status note:** this outline was rewritten 2026-07-07 to incorporate 4 defects filed by
Claude Opus 4.8's cross-model review (`OPUS_REVIEW_20260625.md`) and fixed in
`code/e216_detection_function.py` + `code/e216_sensitivity_sweep.py`. All numbers below are
re-derived from the corrected code (`results/OUTCOME.json`, `results/missing_core_spec.json`,
`results/sensitivity_summary.json` — regenerated 2026-07-07). **This draft is NOT yet
submission-ready**: it still needs a palynologist co-author (SIG G2/G10) and a cross-model
review (G9) before any journal sees it.

---

## ABSTRACT (draft — note the caveat is now load-bearing, not buried)

The pre-400 CE human footprint in volcanic Java remains one of the largest gaps in
Southeast Asian archaeological science. We apply a pre-registered palaeoecological
interferometer — analogous to the Michelson-Morley experiment — to determine whether
the existing Java dated pollen-and-charcoal core network is capable of detecting a
pre-inscription-era farming population of E196's estimated 631,000–1.27 million people.

Using a simplified REVEALS source-area forward model, we find the existing seven-core
network cannot resolve the question: one core (J6, marine Solo) has a Relevant Source
Area of Pollen (RSAP) that *geometrically* reaches the Kedu/Brantas inscription
heartland, but catchment dilution over its ~400 km marine drainage suppresses the
expected signal roughly three orders of magnitude below the detection threshold — **zero
of seven cores can *resolve* heartland clearing, at any parameter setting we tested**
(a sensitivity sweep across published RPP, threshold, and local-weighting ranges gives
P(network detect) = 0.000 uniformly). Coverage is not resolution.

Instrument sensitivity itself rests on a qualitative positive control (Dieng ~600 CE,
Rawa Danau ~AD 1770): the raw pollen count series behind both are inaccessible
(paywalled), so the 15–20 pp NAP-rise detection threshold is a literature import, not a
value we re-derived from primary data. This independently satisfies the pre-registered
NO-GO branch, so OUTCOME-3 (instrument-limited, loose bound) is supported by two
separate reasons, not one.

We specify the single decisive missing core: a lowland lake or swamp within 20 km of
Kedu (~7.5°S, 110°E) or the Brantas headwaters (~7.8°S, 112°E), AMS-dated at ≤50-yr
resolution across 0–2000 CE, targeting Oryza-type/Poaceae/Trema pollen plus
microcharcoal co-occurrence, at an estimated cost of USD 8,000–15,000. **This core
would settle the question at E196's central population estimate (1.27M) under either a
uniform or spatially-clustered clearing assumption, and at the floor estimate (631k) if
clearing was clustered — but NOT at floor population with spatially uniform clearing**
(expected NAP rise 12.6 pp, below the 17.5 pp threshold; this corner fails to detect in
~85% of the parameter grid we swept). That residual — a small, dispersed, low-intensity
population — is structurally invisible to pollen at any core and is handed to the
companion phytolith/starch channel (E215).

---

## §1. Introduction

Context: the pre-400 CE archaeological gap in volcanic Java. Multi-method population
estimate (E196) gives 631k–1.27M people — yet zero open-air sites in volcanic interior.
The taphonomic suppression hypothesis (Amien et al., VOLCARCH series) argues burial and
erosion explain the absence.

The palaeoecological test: pollen and charcoal do NOT care about tephra burial (they
sink in lake sediment regardless of what happens on land). If a substantial farming
population existed, they should have left a land-clearance signal. E214 (Amien 2026b)
found the existing record LEANS AGAINST a large pre-400 CE population but left open:
(a) network undersampling and (b) low-clearance settlement mode.

This paper quantifies those escape hatches as exclusion bounds, designing the test as a
three-outcome pre-registered instrument (Michelson-Morley framing):
  OUTCOME-1: existing network excludes large-clearing population at C=90%
  OUTCOME-2: pre-400 CE cultigen/charcoal signal detected
  OUTCOME-3: network is instrument-limited → specify the decisive missing core

**Framing decision (2026-07-07 rewrite):** this is deliberately NOT framed as "does the
civilization exist" (an unfalsifiable equifinality trap — see the fate of P7). It is
framed as "here is exactly the measurement that would decide it, and here is what we
know without that measurement." This is the difference between a confirmation brick and
a falsification instrument.

---

## §2. Data and Methods

### 2.1 Core network (S1)
Seven dated Java pollen/charcoal records (Table 1; Fig. 1). Source: E214 systematic
literature review (Amien 2026b). Independent cross-check: 2025 GRL maritime-continent
soil-erosion synthesis (Ruan et al. 2025, doi:10.1029/2025GL114695) confirms fire/
erosion signal ~3,500 BP in East Java marine core (molecular markers; not pollen) —
consistent with, but not diagnostic of, pre-400 CE human activity (wrong proxy for our
pre-registered charcoal+Cerealia diagnostic).

### 2.2 Positive controls and the NO-GO branch (S2 calibration) — REVISED 2026-07-07
Two within-network positive controls were intended to establish instrument sensitivity:
  J1 Dieng/Telaga Balekambang: "substantial nearly continuous clearance" from
     ~1,350 BP (~600 CE), tied to Hindu-Javanese centre (Pudjoarinto & Cushing 2001).
  J2 Rawa Danau: food crops appear in last ~400 yr, clearance ~AD 1770 (Yulianto et al. 2005).

**Honest status (this is the corrected framing — do not soften it):** the raw pollen
count series behind both records are behind publisher paywalls (HTTP 403 at time of
analysis) and were never extracted. PREREG.md's S2 GO/NO-GO gate states explicitly: GO
only if the Dieng signal magnitude is extractable from primary data; **NO-GO — proceed
directly to OUTCOME-3 — if it is not.** It is not. This run therefore hit the
pre-registered NO-GO branch. The 15–20 pp NAP-rise detection threshold used throughout
is a **literature-consensus import**, not a blind re-derivation (fails SIG G1 in the
strict sense; this is disclosed, not hidden). Practical consequence: OUTCOME-3 is
supported by two independent reasons — (a) the heartland resolution gap (§3.2, which
holds regardless of the exact threshold value, see the sensitivity sweep in §3.2) and
(b) this calibration NO-GO. Report both; do not lean on (a) alone as if (b) were solved.

### 2.3 Population → cleared-area coupling (S3, E196)
E196 Monte Carlo (Amien 2026c): N_floor = 631,059 (p5, comparative island scaling),
N_central = 1,270,000 (median). Land-use coefficients: arable_frac = 0.65–0.80,
cultivation_frac = 0.10–0.40 (Mode A: wet-rice/large-swidden), 0.01–0.05 (Mode B:
dispersed forest-garden). Total cultivated area: 4,166–20,512 km² (N_floor, Mode A).

### 2.4 Forward model (S4, simplified REVEALS)
For each core: RSAP parameterised by archive type and lake radius (Sugita 2007).
Expected NAP rise = α × RPP_NAP × f / (RPP_NAP × f + (1–f))
where f = cleared fraction of RSAP. Detection threshold = 17.5 pp NAP rise above
background (mid of the imported 15–20 pp range; see §2.2 on its provenance).

**Parameter sensitivity (NEW, Defect 2 fix):** rather than evaluating this forward model
at a single MID parameter setting, we swept RPP_NAP ∈ {2.0, 3.0, 4.0} (Sugita 2007
tropical range), threshold ∈ {0.15, 0.175, 0.20}, and α ∈ {0.4, 0.55, 0.7} (the
README-stated local-RSAP weighting range) — 27 grid points per (population × mode)
combination. Full results: `results/sensitivity_network_detection.csv`,
`results/sensitivity_summary.json`.

### 2.5 Two-mode separation (S6)
Mode A (landscape-clearing wet-rice/large-swidden) and Mode B (dispersed forest-garden)
run through identical pipeline. Mode B residual (pollen cannot constrain at any plausible N)
→ explicitly handed to E215 (phytolith/starch channel) as the precisely-defined residual.

---

## §3. Results

### 3.1 Core network coverage — geometric coverage vs resolution (Table 1, Fig. 1) — REVISED 2026-07-07
Seven cores cover the Holocene; six have verified 0–500 CE coverage. Of these, **one
core (J6, marine Solo) geometrically overlaps the Kedu/Brantas heartland** — its ~400 km
marine RSAP reaches Brantas (144.6 km distant). The remaining six cores (J1–J5, J7) sit
55–450 km from the heartland, entirely outside their (much smaller, terrestrial) RSAPs.

**This is NOT the same as saying no core "covers" the heartland — J6 does, geometrically.
The correct and more precise statement is that coverage and resolution are different
questions.** J6's enormous catchment dilutes any heartland clearing signal to
~0.1–0.3% expected NAP rise — three orders of magnitude below the 17.5 pp threshold.
**Zero of seven cores can resolve heartland clearing.** (Earlier drafts and session logs
conflated these two statements; this is the corrected wording — see `OUTCOME.json`
`coverage_vs_resolution_note`.)

### 3.2 Detection probabilities — robust across the parameter sweep (Table 2, Fig. 2) — REVISED 2026-07-07
P(detect | N_floor=631k, Mode A) = 0.000 for ALL seven existing cores, at the MID
parameter setting. Critically, this null is **not an artifact of one parameter choice**:
the full 27-point sweep over RPP_NAP × threshold × α gives P(network detect) = 0.000 at
**every single grid point**, for both population levels (floor/central) and both modes
(A/B) (`sensitivity_summary.json`, `network_detection_sensitivity`). The heartland
resolution gap is a **structural geometry problem** (no core combines heartland
proximity with a non-diluting archive type), not a parameter-tuning artifact. This is
the paper's most robust finding.

### 3.3 Outcome assignment (S8)
Per PREREG.md: P(detect) < 0.90 for Mode A at N_floor → **OUTCOME-3**, independently
reinforced by the §2.2 NO-GO branch.

### 3.4 Positive control status — downgraded from "confirmed" to "qualitative import" — REVISED 2026-07-07
The instrument's *design* is sound (the geometric RSAP-vs-distance logic is
independently verifiable), but its *calibration* rests on qualitative literature
descriptions, not extracted raw data (§2.2). State plainly: **we did not verify
instrument sensitivity from primary data.** The null result in the 0–500 CE window
should be read as "the network cannot test this," not "the network tested this and
found nothing" — those are different claims and only the first is supported.

### 3.5 Confound controls (S7)
Climate confound: natural variance ~5–8 pp (from Bandung Basin LGM grass signal,
classified climatic); threshold of 17.5 pp = ~2.5σ above noise (subject to the same
literature-import caveat as §2.2/§2.4). GRL 2025 molecular fire signal (~3,500 BP):
informative but uses wrong proxy (brGDGTs/levoglucosan ≠ charcoal+Cerealia); does not
trigger OUTCOME-2 per pre-registration. Solo ~2950 BP signal: run as worked sensitivity
case; not counted without charcoal+Oryza confirmation.

### 3.6 Mode B separation (S6)
P(detect | Mode B) ≈ 0 at all N, at all swept parameters. A dispersed forest-garden
population of any size in E196's range is invisible to the pollen record at any existing
core AND at a hypothetical Kedu core. This mode is handed to E215 as the
precisely-defined residual: the maximum population this channel cannot exclude under
Mode B is effectively unbounded (pollen is structurally blind to dispersed
non-clearing settlement).

---

## §4. Discussion

### 4.1 What OUTCOME-3 means (and does not mean)
Not "we found nothing." Rather: "the existing network cannot test the hypothesis at the
heartland, for two independent reasons (resolution gap + uncalibrated threshold)." The
instrument's logic works; its current inputs do not let it decide.

### 4.2 The decisive missing core — now reported as a caveated corner table, not one number (S8b, Table 3) — REVISED 2026-07-07
Location: within 20 km of Kedu Plain (~7.5°S, 110.0°E) or Brantas headwaters (~7.8°S, 112.0°E).
Archive: closed lowland lake or ox-bow swamp (NOT highland, NOT marine).

**The original single-point claim "P(detect)=1.0" (Opus review Defect 4) hid a failing
corner. The honest statement, from the full 2×2 corner table
(`results/missing_core_corner_table.csv`) plus its extended parameter sweep
(`results/sensitivity_missing_core_corners.csv`), is:**

| Population | Clearing pattern | NAP rise | Detects (>17.5pp)? | Robust across parameter sweep? |
|---|---|---|---|---|
| floor (631k) | uniform | 12.6 pp | **NO** | fails in 85.2% of 27-pt grid |
| floor (631k) | clustered 4× | 34.5 pp | yes | detects in 100% of grid |
| central (1.27M) | uniform | 21.9 pp | yes | detects in 70.4% of grid |
| central (1.27M) | clustered 4× | 48.8 pp | yes | detects in 100% of grid |

A core at Kedu/Brantas would settle the question at central population under either
clustering assumption, and at floor population if clearing was spatially clustered. **It
would NOT settle the question at floor population with spatially uniform clearing** —
that specific corner stays open and is passed, like Mode B, to the dispersed-settlement
framing (E215). This caveat is in the abstract, not only here (SIG G8 compliance).

Cost: USD 8,000–15,000 (one vibrocore + ~20 AMS dates). This is a funded, tractable
coring campaign — suitable for a PhD chapter or targeted supplementary field component
to an existing Quaternary project in Java.

### 4.3 Mode A vs Mode B implications
If the missing core returns:
  - OUTCOME-1 (no pre-400 CE signal, at central population or clustered floor
    population): large-clearing population excluded in those regimes; the uniform-floor
    corner and Mode B (dispersed) remain untested by pollen; E215 becomes primary.
  - OUTCOME-2 (signal present): first direct palaeoecological evidence of pre-400 CE
    Java farming — non-circular, independent of the spatial/inscription substrate.

### 4.4 Relationship to companion channels
Molecular markers (GRL 2025): fire/erosion ~3,500 BP consistent with pre-400 CE
human activity but cannot be attributed to farming specifically. Supports the plausibility
of Mode A occupation; does not decide the question.
Phytolith/starch (E215): the ONLY existing method that can detect Mode B (dispersed,
non-clearing) AND the floor-uniform corner pollen cannot resolve; requires sediment
matrix from sites or lake cores; currently VOID for prehistoric Java (no published study).

---

## §5. Conclusion

We designed and executed a pre-registered palaeoecological interferometer for the
pre-400 CE Java population hypothesis. The existing seven-core network cannot resolve
the question at the inscription heartland: one core geometrically reaches it but cannot
resolve its signal through catchment dilution, and this null is robust across the full
swept parameter space (P(network detect)=0.000 at all 27 grid points tested).
Instrument calibration itself rests on a qualitative, non-re-derived literature import,
independently triggering the pre-registered NO-GO branch.

The paper's contribution is: (1) a demonstration that "coverage" and "resolution" are
distinct and must be reported separately in detection-power palaeoecology; (2) a
precise, parameter-swept specification of the single missing core that would decide the
question, honestly caveated at the corner where even that core would fail (floor
population, uniform clearing); and (3) an explicit two-mode separation routing the
dispersed-population and floor-uniform residual to phytolith analysis (E215). This is
the first formally power-analysed, pre-registered palaeoecological assessment of the
Java pre-inscription gap, and the project's first flagship instrument structurally
designed to be capable of disconfirming its own founding hypothesis.

---

## Tables

**Table 1** — Java dated palaeoecological core network: coordinates, archive type,
RSAP radius, distance to Kedu/Brantas heartland, 0–500 CE coverage, positive control status.
[→ results/core_coverage_table.csv]

**Table 2** — Detection probability per core at N_floor, N_central (Mode A and B):
NAP rise, P(detect), positive control flag.
[→ results/detection_probability_table.csv]

**Table 3** — Decisive missing-core specification, corner table (population × clustering).
[→ results/missing_core_spec.json, results/missing_core_corner_table.csv]

**Table 4 (NEW)** — Sensitivity sweep summary: P(detect) intervals and detecting-fraction
of parameter grid, network-level and missing-core-corner level.
[→ results/sensitivity_summary.json, results/sensitivity_network_detection.csv,
results/sensitivity_missing_core_corners.csv]

## Figures

**Fig. 1** — Java map: core locations + RSAP circles + heartland gap annotation.
[→ figures/fig1_network_rsap_map.png]

**Fig. 2** — Detection power surface: P(detect) vs population N for existing network
(Mode A, B) vs hypothetical Kedu core.
[→ figures/fig2_detection_power.png]

**Fig. 3 (NEW, not yet rendered)** — Corner-table sensitivity: bar/heatmap of
fraction-of-grid-detecting for each (population × clustering) corner, visually showing
the floor+uniform corner's fragility (14.8% detecting) vs the other three corners
(70.4–100%). Script: extend `code/e216_figure.py` or add `code/e216_figure_sensitivity.py`.

---

## SIG pre-submission checklist (S9) — UPDATED 2026-07-07

| Gate | Status |
|------|--------|
| G1 re-derivation | Numbers re-derived from corrected code (2026-07-07); positive-control threshold remains a labeled literature import, not re-derived from raw data (disclosed in §2.2, not hidden) |
| G2 domain-sanity | Quaternary palynologist review still required before submission |
| G3 canonical data | E214 core inventory used; NOT the P7 7-volcano file |
| G4 circularity | Pollen independent of Pyle/inscription substrate; non-circular |
| G5 equifinality | 5 escape hatches closed/bounded in §2 and §3.5 |
| G6 counter-evidence | E214 counter-evidence built in; Solo ~2950 BP worked case; GRL 2025 noted |
| G7 reproducibility | Code + data + REVEALS params + sensitivity sweep → Zenodo (PENDING Pak Amien upload, see SUBMISSION_CHECKLIST.md) |
| G8 overstatement | Pre-registered 3-outcome; bounds as intervals; conservative-corner caveat now in the abstract, not buried |
| G9 cross-model | Cross-model skeptical review still needed before submission |
| G10 human review | Quaternary palynologist review still required (critical — domain expertise gap) |

**Next steps for Pak Amien (human-gated — see `SUBMISSION_CHECKLIST.md`):**
1. Review this outline → approve or redirect
2. Find a Quaternary palynologist collaborator (Indonesian or international) for G2/G10
3. Write §1-§2 prose (≤2,000 words) in English from this outline
4. Zenodo deposit of code + E214 data summary + sensitivity sweep outputs (G7)
5. Submit after G9 (cross-model review) + G10 (domain review)

---

*E216 designed 2026-06-25 (Claude Opus 4.8 ultracode). Executed 2026-06-25 (Claude Sonnet 4.6).
Pre-registered before analysis. OUTCOME-3 assigned per pre-registered rule.
4 Opus-reviewed defects fixed 2026-07-07 (Fable strategic plan WS-A, executed by Sonnet 5):
coverage-vs-resolution distinction (D1), parameter sensitivity sweep replacing point
estimates (D2), positive-control status downgraded to labeled literature import + NO-GO
branch disclosed (D3), missing-core corner table replacing single overclaimed p=1.0 (D4).*

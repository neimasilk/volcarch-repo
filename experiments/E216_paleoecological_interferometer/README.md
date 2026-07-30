# E216 — The Paleo-Ecological Interferometer

**A pre-registered, empirically-calibrated detection-power test of the pre-400 CE "volcanic civilization" hypothesis in Java.**

**STATUS: `HARDENED — OUTCOME-3 (instrument-limited loose bound). 4 Opus-review defects fixed. Paper draft outline updated. Still NOT submission-ready (needs co-author + G9).`**
**Date designed:** 2026-06-25 (Claude Opus 4.8 ultracode)
**Date executed:** 2026-06-25 (Claude Sonnet 4.6, at PI request)
**Date hardened:** 2026-07-07 (Fable strategic plan WS-A → Sonnet 5 execution; see `docs/research_notes/FABLE_STRATEGIC_PLAN_20260707.md`)
**Outcome:** OUTCOME-3 per pre-registered rule, for TWO independent reasons: (1) heartland resolution gap — one core (J6) geometrically covers the heartland but cannot resolve it through catchment dilution; zero cores resolve it, robustly across a full parameter sweep; (2) the positive-control calibration hit the pre-registered NO-GO branch (raw data paywalled, threshold is a literature import, not re-derived). Coverage ≠ resolution — see `OPUS_REVIEW_20260625.md` and the fix log below.
**4 defects fixed 2026-07-07 (see `OPUS_REVIEW_20260625.md` for the original findings):**
- D1 (coverage≠resolution): `OUTCOME.json` now reports `n_cores_covering_heartland` (geometric, =1) separately from `n_cores_resolving_heartland` (=0). No more self-contradiction.
- D2 (deterministic P(detect)): `code/e216_sensitivity_sweep.py` sweeps RPP_NAP × threshold × α (27-point grid) — network-level null is P=0.000 at every grid point (robust, not parameter-dependent). See `results/sensitivity_summary.json`.
- D3 (overstated positive control): `positive_control_status` downgraded from "CONFIRMED" to "QUALITATIVE ONLY... NOT re-derived"; explicit `go_no_go_branch` field discloses the PREREG NO-GO trigger.
- D4 (hidden failing corner): `compute_missing_core_spec()` now reports the full population×clustering corner table instead of one hardcoded CONCENTRATION_FACTOR=4.0 headline. The conservative corner (floor population + uniform clearing) does NOT detect (12.6pp < 17.5pp threshold) and fails in 85% of its own parameter sweep — this caveat is now in the abstract (`results/PAPER_DRAFT_OUTLINE.md`), not buried in a constant.
**Next action (PI, human-gated — see `SUBMISSION_CHECKLIST.md`):** (1) Review updated paper draft outline → `results/PAPER_DRAFT_OUTLINE.md`. (2) Find palynologist co-author (G10). (3) Zenodo code deposit (G7). (4) Cross-model review (G9) before submit.

---

## 0. TL;DR — what this is and why it exists

The PI's framing (verbatim intent): *"This is a long, interlinked experiment. Build ONE additional paper idea. Like Michelson–Morley, which 'failed' to find the ether — I have no problem 'failing' to find a pre-400 CE civilization in Nusantara, **as long as it is proven definitively and falsifiably.**"*

E216 is that instrument. It is the **honest successor to the dead P7 paper**. Where P7 asked *"do deep-time sites avoid volcanoes by distance?"* (a confirmation test that collapsed into a volcano-inventory artifact, distances inflated 2.4–4.0×), E216 asks **"is the pre-400 CE absence of evidence REAL, or merely taphonomic?"** — and is built to answer it **either way as a defended number**.

The core move is to convert E214's qualitative *"the pollen record leans against a large pre-400 CE population"* into a **quantified, symmetric exclusion bound**: *"a forest-clearing population larger than N\* is excluded at confidence C; a population below N′ is genuinely invisible to this channel — and here is exactly the missing core that would close the remaining gap."*

**Why the null is decisive here (and was NOT in E214 / the radiocarbon-SPD alternative):** the same Java cores that show **no** pre-400 CE clearance **do** record the post-600 CE Hindu-Javanese clearance (Dieng ~600 CE; Rawa Danau ~AD 1770). That demonstrated late signal is a **within-network positive control**: it proves the interferometer is *sensitive*. A null then means *"the ether is not there,"* not *"we had no interferometer."* This is the property the radiocarbon-SPD design lacks (Java has ~1 dated 14C event in 0–500 CE; p3k14c excludes Island SE Asia → the SPD cannot run and would only "lean"). That is why E216 is the chosen design.

---

## 1. Relation to prior work (do not re-derive — build on these)

| Prior | What it gives E216 | File |
|---|---|---|
| **P7** (REJECTED Antiquity AQY-2026-0104) | The dead confirmation test E216 replaces. E216 must NOT touch the contaminated spatial/inscription/Pyle-burial substrate. | `papers/P7_TOM/` |
| **E214** (palynology, LEANS AGAINST) | The assembled Java core inventory + the two open escape hatches (undersampling; no-pop-vs-dispersed-pop) E216 closes/bounds. | `experiments/E214_palynology_anthropogenic_signal/README.md` + `findings_agent_20260608.md` |
| **E215** (phytolith VOID) | The companion channel. E216 hands the residual **dispersed-mode** population it *cannot* exclude to E215 as the precisely-defined next test. | `experiments/E215_phytolith_starch_gap/README.md` |
| **E196** (ML population, ~1.3–1.7M @400 CE; floor 631k) | The **falsifiable target** AND the population→cleared-area coupling coefficients (arable_frac 0.65–0.80; cultivation_frac 0.10–0.40; JAVA_AREA 129,000 km²). | `experiments/E196_population_estimation/population_model.py` + `results/e196_results.json` |
| **E020 / Mini-NusaRC v3** | Dating-completeness **sanity layer ONLY** (verified inadequate for a demographic SPD → NOT used as a population instrument). | `experiments/E020_mini_nusarc/data/mini_nusarc_v3.csv` |
| **I-146** (selective survival, bronze drums) | A from-below non-zero lower bound (excludes "never settled"), NOT a clearing proxy. | IDEA_REGISTRY |
| **SUBMISSION_INTEGRITY_GATE** | E216 is designed to pass G1–G10, especially G5 (equifinality) and G6 (counter-evidence built into the instrument). | `docs/SUBMISSION_INTEGRITY_GATE.md` |

**Idea IDs:** I-147 (the E216 detection-function framework), I-148 (forward-simulation power curve + decisive-missing-core spec), I-149 (two-mode separation: clearing vs dispersed). See IDEA_REGISTRY.

---

## 2. Core question and hypotheses

**Core question.** For each land-use mode (wet-rice / large-swidden / dispersed forest-garden–arboriculture), what is the **largest** pre-400 CE Javanese farming population whose pollen-and-charcoal footprint would have **escaped detection** by the existing dated lake/swamp/marine core network — i.e. what population size is **excluded** at confidence C, what population remains genuinely invisible to this channel, and is the observed near-total pre-400 CE silence what we would expect if E196's ~1–2 million people had lived there?

- **H1 (strong "volcanic civilization", testable):** a substantial pre-400 CE Javanese farming population (E196 central ~1.3–1.7M; floor 631k) produced **landscape-scale forest clearance + anthropogenic burning** that the existing core network would have recorded as a rise in herb/Poaceae/Cerealia-type/Trema pollen and charcoal influx **before ~1550 cal BP**.
- **H0 (null):** no such pre-400 CE signal exists.

The contribution is to pair H0 with a **detection function**, so the null reads: *"a forest-clearing population above N\* would have been recorded with probability ≥ C and was not; therefore such a population is excluded at confidence C, under explicitly stated source-area / productivity / coverage assumptions."*

**Honest prior (state this in the paper).** After E214, H1 *in its landscape-clearing form* is already largely doubted. The **live** hypothesis is the **dispersed low-clearance mode**, which this channel can only **bound**, not decide. Say so up front — this is the difference between an integrity-passing instrument and the project's old confirmation architecture.

---

## 3. PRE-REGISTERED decision rule (commit BEFORE any analysis — this is step S0)

Write this verbatim into `experiments/E216_paleoecological_interferometer/PREREG.md`, timestamp it in `docs/JOURNAL.md`, and do not edit it after analysis begins (SIG F10 discipline). Fixed parameters: **C = 90%**, target = **E196 range 631k–2M** (use 631k floor and 1.27M central as the two reference points), diagnostic = **charcoal + Cerealia/Oryza-type co-occurrence** (NOT charcoal alone).

> **OUTCOME-1 — DECISIVE NULL / FALSIFIES H1-strong.**
> IF, after empirical within-network calibration (S2), the modelled detection probability `P(detect | N ≥ 631k, mode ∈ {wet-rice, large-swidden}) ≥ 0.90` at **≥ 1 actually-existing core with verified 0–500 CE coverage**, AND no pre-400 CE anthropogenic signal crosses the calibrated threshold,
> THEN a forest-clearing population of E196's size is **excluded at 90%**; the strong thesis is rejected and only the dispersed/low-visibility mode survives (handed to E215).
>
> **OUTCOME-2 — POSITIVE / CONFIRMS H1.**
> A robust pre-400 CE rise in cultigen/herb/pioneer pollen **OR** charcoal influx above the calibrated threshold, **with charcoal + Cerealia/Oryza co-occurrence** (climate fires lack cultigen pollen), in ≥ 1 well-dated heartland-relevant core not explicable by climate or volcanism → the project's **first positive material discovery**.
>
> **OUTCOME-3 — LOOSE BOUND / INSTRUMENT-LIMITED (the honest modal outcome — see §8).**
> IF `P(detect | E196 range) < 0.90` even for the clearing modes (network too sparse/coarse at the heartlands),
> THEN neither confirm nor refute; the **headline becomes the quantified specification of the single decisive missing core** (location at Kedu/Brantas, basin radius, resolution, 0–500 CE span, target taxa).

All three outcomes are defined, reportable, and **publishable**. The result cannot be spun because the rule is fixed in advance.

---

## 4. Method (S0–S9) — executable steps

> Stack: Python 3.10+ (Anaconda); R for `rcarbon`/age-depth if needed; a REVEALS/LOVE implementation (DISQOVER-style, or a transparent reimplementation of Sugita 2007). All free. Deliverables go in `experiments/E216_paleoecological_interferometer/{data,results,figures}/` and code at the experiment root.

**S0 — Pre-registration.** Commit `PREREG.md` (the §3 rule) + timestamp in JOURNAL. Do this FIRST.

**S1 — Core inventory + 0–500 CE coverage table + independence audit.**
Finalize the Java dated-core network from E214 (Dieng/Telaga Balekambang; Rawa Danau; Teluk Banten; Bandung Basin; Situ Bayongbong; Solo marine core; Song Gupuh) and harvest additional records from **Neotoma Paleoecology Database** + **Indo-Pacific Pollen Database (IPPD)** + the **2011 *Quaternary International* Java synthesis** + the **2025 *GRL* Maritime-Continent soil-erosion synthesis (doi:10.1029/2025GL114695)**. For each record capture: coordinates, basin radius, depositional environment, temporal resolution, dating control (n 14C dates), and **verified 0–500 CE coverage (yes/no)**. Output `results/core_coverage_table.csv`. Cores lacking 0–500 CE coverage are **excluded from the exclusion bound** (they cannot constrain it) but **kept for the missing-core spec** (S8).

**S2 — EMPIRICAL CALIBRATION FIRST (the load-bearing step — do not skip or invert).**
From each core's published diagram/counts, extract the **magnitude** of the demonstrated *late* positive signal — the herb/Poaceae/Cerealia-type % rise and charcoal-influx step the core actually records at its known clearance event (e.g. Dieng ~600 CE; Rawa Danau ~AD 1770). This observed effect size **is** the empirical detection threshold **and** the within-network positive control proving the instrument is sensitive. Calibrated threshold = the **smallest such excursion the cores demonstrably resolve above their own late-Holocene natural-variance band**. Output `results/empirical_thresholds.csv`. *If this cannot be extracted from available data, fall back early to the honest missing-core paper (§9) rather than overbuilding.*

**S3 — Population → cleared-area coupling (use E196; do not invent coefficients).**
Read E196's carrying-capacity coefficients from `experiments/E196_population_estimation/population_model.py` (arable_frac 0.65–0.80; cultivation_frac 0.10–0.40; JAVA_AREA 129,000 km²). Convert a candidate population `N` and land-use mode `M` into a cleared/cultivated area and spatial configuration `L`: wet-rice & large-swidden → contiguous cleared area; dispersed forest-garden → diffuse low-clearance mosaic. Propagate E196's Monte-Carlo uncertainty.

**S4 — Forward model (REVEALS as a SENSITIVITY layer, not the backbone).**
For each `(N, M, L)`, compute expected pollen % / charcoal influx at each core via a relevant-source-area-of-pollen model (Sugita 2007 REVEALS/LOVE), parameterized **across the published range** of tropical Relative Pollen Productivity (RPP) + fall-speed values for Poaceae/Cerealia/Oryza/Trema. Report the bound as an **interval over the RPP range**, anchored to the S2 empirical threshold so it stays data-tied even where absolute productivity is uncertain.

**S5 — Detection function + power surface (graft from the radiocarbon design's best idea).**
For each core: `P(predicted signal crosses S2 threshold | N, M, L)` given basin source area, sampling resolution, count precision. Network-level `P(N,M,L) = P(≥1 core detects)`. Produce the **2-D exclusion surface over (population size × land-use mode)** and a **spatial coverage map** of where in Java a population could hide below detection. Make the power curve a **co-headline**, not a buried robustness check.

**S6 — Two-mode separation (graft from the genomics design's discipline).**
Run **model A (landscape-clearing wet-rice/large-swidden)** and **model B (dispersed forest-garden/arboriculture)** through the *identical* pipeline. Report explicitly: this channel excludes mode A up to `N_A` and mode B only up to `N_B` (expect `N_B` to be unusably high). The residual mode-B population it **cannot** see is the precisely-defined target handed to **E215** phytoliths.

**S7 — Confound controls.**
Set the natural-variance noise band from each core's own late-Holocene climate-driven fluctuations + the Bandung LGM grass signal (already flagged climatic). Require the anthropogenic threshold to **exceed** it. Treat **charcoal + cultigen co-occurrence** as diagnostic (climate fires lack Cerealia/Oryza). Run the Solo marine ~2950 cal BP ambiguous signal as a **worked sensitivity case** — do NOT suppress it (SIG G6).

**S8 — Apply the pre-registered rule.**
Assign OUTCOME-1 / -2 / -3 per §3. If OUTCOME-3, compute the **decisive missing-core spec**: location at Kedu/Brantas heartlands, basin radius, resolution, 0–500 CE span, target taxa (Oryza/Poaceae/Trema + charcoal).

**S9 — Integrity + write-up.**
Self-run SIG G1–G9 (§10). Deposit code + extracted data series + REVEALS parameters on Zenodo (satisfies G7). Draft the paper for the §9 outcome that fired.

---

## 5. Data sources (named, real, with accessibility)

1. **Java dated-core network — already assembled in E214** (`experiments/E214_palynology_anthropogenic_signal/findings_agent_20260608.md` + `README.md`). Real DOIs incl. **Pudjoarinto & Cushing 2001** *RPP* 116:13–45 (Dieng — the within-network positive control); **Yulianto et al. 2005** *Tropics* 14:271 (Rawa Danau, AD 1770 positive); **van der Kaars & van den Bergh 2004** *JQS* 19:229; **van der Kaars & Dam 1995** *P³* 117:55; **Stuijts 1993** *MQRSEA* 12; **Poliakova/Zonneveld et al. 2017** *RPP* 244 (Solo marine — the ~2950 BP loophole; open PhD PDF at ediss.uni-goettingen.de); **"Rain forest in Java through the Quaternary" 2011** *Quat. Int.* (S1040618211003326). *Accessible now via in-repo summary; raw counts partly behind 403 paywall — see §10.*
2. **Neotoma Paleoecology Database** + **Indo-Pacific Pollen Database (IPPD)** / **Global Charcoal Database** for digitized counts. *Open APIs; coverage of these specific Java cores UNVERIFIED — check in S1.*
3. **Comparator early-clearance records (method-works controls):** Maloney 1980 *Nature* 287:324 (Toba/Sumatra ~7500 BP); Anshari et al. 2001/2004 (Sentarum/Kalimantan ~3000 BP); Hunt & Rushworth 2005 (Niah ~6000 cal BP). *E214 already cites.*
4. **E196 population model** with per-capita land coefficients (`experiments/E196_population_estimation/population_model.py`, `results/e196_results.json`): comparative-island median 1.27M (p5 = 631,059), growth-back-projection median 1.68M, density 4.9–30.1/km². *In-repo.*
5. **Method software (all free):** a REVEALS/LOVE implementation with published tropical RPP + fall-speed values; `rcarbon` (Crema & Bevan 2021) + OxCal + IntCal20/SHCal20 for age-depth and Surovell et al. 2009 *JAS* 36:1715 taphonomic weighting of the dating layer; Python.
6. **2025 *GRL* synthesis** "Late Holocene Human Impact on Tropical Soil Erosion in the Maritime Continent" (doi:10.1029/2025GL114695) — independent cross-check on landscape-disturbance timing.
7. **p3k14c** (Bird et al. 2022 *Sci Data* 9:27) — **ONLY** as a dating-completeness cross-check, with the documented caveat that it **EXCLUDES Island SE Asia** (use it to *show* the radiocarbon channel is provably blind in-window, NOT as a demographic instrument).

---

## 6. Power analysis (the heart of the instrument)

Computed in two coupled layers, with the **empirical layer leading** so the number is data-tied:

1. **Empirical detection threshold (S2):** for each core, the minimum pollen-% / charcoal-influx excursion the core *demonstrably resolves* = the measured magnitude of its own post-600 CE clearance signal above its late-Holocene natural-variance band. A real observed effect size, not a guess — the within-network positive control proving the instrument can see clearance.
2. **Forward detection limit (S4–S5):** for each core, the minimum cleared/secondary-vegetation area within its relevant source area (RSAP, from REVEALS across the published tropical RPP range) needed to push a diagnostic pollen sum or charcoal influx past the S2 threshold; map that area to a population via E196's cultivation coefficients and mode `M`. Network `P(N,M,L) = P(≥1 core detects)`, reported as an **interval over RPP uncertainty** so a wide interval cannot silently sink the null.

The **exclusion bound N\*** = the smallest `N` at which `P ≥ 0.90` for the clearing modes. If RPP uncertainty makes the interval straddle E196's range, that triggers **OUTCOME-3** and the headline becomes the missing-core spec — power-analysis honesty rather than overclaim (SIG G5/G8).

---

## 7. Equifinality closures (the escape hatches — and how each is bounded)

This section is the reason E216 beats E214. Every major escape hatch is closed or **quantitatively bounded**.

| # | Escape hatch | Closure |
|---|---|---|
| 1 | **Undersampling** ("the cores just missed it" — E214's chief escape) | Becomes the number `P(detect|N,M,L)` per core from basin size × source area × resolution × count precision. High P + absent signal ⇒ undersampling **rejected at C**. Low P ⇒ bound honestly loose ⇒ deliverable is the required missing core. *Brutal residual:* a heartland-proximal high-resolution 0–500 CE core may not exist, so OUTCOME-3 is the modal result — acknowledged as a **designed headline**, not hidden. |
| 2 | **Dispersed / low-clearance society** (E214/E215 reframe) | Bounded by running the forward model per mode (S6): clearing modes are excludable at given `N`; forest-garden/arboriculture gives a weak signature, so the paper states the **maximum population this channel cannot exclude** under that mode and hands that residual **explicitly to E215**. The refuge becomes a stated upper bound, not an unfalsifiable move. |
| 3 | **Climate / volcanic confound** | Bounded: noise band from the cores' own late-Holocene climate fluctuations; require **charcoal + Cerealia/Oryza co-occurrence** (climate fires lack cultigen pollen); Solo ~2950 BP run as a worked sensitivity case. |
| 4 | **RPP / source-area parameter uncertainty** (REVEALS not calibrated for closed tropical canopy) | Bounded by leading with the S2 empirical within-network threshold and propagating the full published RPP range as an interval; if parameters dominate, **that** is reported as the finding. |
| 5 | **Catastrophic late collapse erased the signal** | Closed: the model integrates the full ~2400-yr window (E196's 46.6M person-centuries); a transient late crash cannot remove a cumulative multi-century clearance/charcoal footprint. |

**The one hatch that stays genuinely open** — a truly tiny sub-detection forager-arboriculturalist population — is reported as the **irreducible blind spot** and is precisely the **E215 target**, NOT denied.

---

## 8. Honest expected outcome (read this before getting excited)

The **modal outcome is OUTCOME-3 (the loose bound), not a decisive null.** E214 already showed no heartland-proximal high-resolution 0–500 CE lowland core exists, so `P(detect|E196)` may be low at the relevant locations. **This is fine and was designed for:** a quantified *"here is exactly the single core that would settle it"* is itself the project's first honest such deliverable, and it is fundable (PhD / targeted coring).

The mode E216 can *decisively* exclude (large forest-clearing) is the thesis the project already abandoned after E214; the **live** dispersed mode is out of reach of pollen. State this explicitly as a contribution (it bounds the abandoned strong claim AND defines the live one's test) — do **not** overclaim that it touches the live hypothesis. This honesty is what makes it Michelson-Morley rather than another confirmation brick.

---

## 9. The two (three) papers this produces — and the venue

- **NULL paper (OUTCOME-1):** *"A pre-400 CE forest-clearing population larger than N\* is excluded at 90% confidence given the existing Java core network."* A genuine Michelson-Morley null: falsifies the strong landscape-clearing thesis, constrains E196 from the ecology side, and is the project's **first flagship structurally designed to disconfirm L1** (cures the confirmation-architecture disease, ME#17 R1).
- **MISSING-CORE paper (OUTCOME-3, the modal case):** *"The Java palaeoecological network cannot decide the pre-400 CE population question; here is the single decisive missing core (Kedu/Brantas, basin radius R, resolution dt, 0–500 CE span, target taxa)."* A fundable, citable research-agenda result.
- **POSITIVE paper (OUTCOME-2):** *"First direct palaeoecological evidence of a pre-400 CE forest-clearing society in Java."* The project's first positive material discovery; non-circular; independent of the contaminated substrate.

**Target venue (absolute zero-APC honored):**
- **Primary:** *Vegetation History and Archaeobotany* (Springer; Scopus Q1; subscription / Green-OA self-archiving = **zero APC**) — natural home for REVEALS detection-limit work.
- **Alternatives (zero-APC):** *Quaternary International* (where the 2011 Java synthesis lives; subscription, no APC); Diamond-OA **JCAA** or *Internet Archaeology* for a methods-and-code framing; *Journal of Archaeological Method and Theory* (Q1) for the detection-theory framing.
- **AVOID** *Open Quaternary* (NOT Diamond; GBP 1,040 APC) unless a waiver is granted.
- Deposit a **Zenodo preprint of code + data regardless** (satisfies SIG G7).

---

## 10. Risks (carry these into the write-up)

1. **Modal outcome is the loose bound, not a decisive null** — mitigated by pre-registering OUTCOME-3 as a designed headline (§8).
2. **RPP / source-area parameters poorly constrained for closed tropical canopy** — lead with the S2 empirical threshold; relegate REVEALS to a sensitivity layer; report the bound as an interval; if parameter-limited, report THAT.
3. **Raw-data access** — count/charcoal series partly behind 403 paywalls; hand-digitizing from published diagrams is a G7 reproducibility risk → author data-requests / palynologist co-author; flag every digitized series transparently.
4. **Strawman-null risk** — the mode it can decisively exclude (large clearing) is the already-abandoned thesis → state this as a contribution, do not overclaim.
5. **Positive-control transfer** — the demonstrated late positive (highland/kingdom-scale clearance) is a different effect-type/geometry from a hypothetical dispersed lowland signal → report the control's effect-type and geometry explicitly; restrict decisive-null claims to clearing modes whose signature matches the control.

---

## 11. SIG (Submission Integrity Gate) compliance — self-check before any submission

| Gate | How E216 satisfies it |
|---|---|
| G1 re-derivation | All headline numbers re-derived blind from extracted series + code (Zenodo). |
| G2 domain-sanity | Quaternary-palynologist review (G10) hardens RPP choices. |
| G3 canonical data | Uses canonical inventories; does NOT reuse the P7 7-volcano file. |
| G4 circularity | Pollen is independent of the Pyle/E117 burial physics and of the spatial/inscription substrate that sank P7. |
| G5 equifinality | The whole instrument is a quantified answer to equifinality (§7); claims restricted to what the detection function supports. |
| G6 counter-evidence | E214's counter-evidence is built INTO the instrument; Solo ~2950 BP run as a worked case, not suppressed. |
| G7 reproducibility | Code + data + REVEALS parameters on Zenodo; digitized series flagged. |
| G8 overstatement | Pre-registered three-outcome rule; bound reported as interval; modal loose-bound stated honestly. |
| G9 cross-model | Run a cross-model skeptical review on the draft before submission. |
| G10 human review | **Required before flagship status** — a Quaternary palynologist / geoarchaeologist must review. |

---

## 12. Execution checklist for the future (Sonnet) session

```
[ ] S0  Write PREREG.md (the §3 rule, C=90%, target 631k/1.27M, diagnostic=charcoal+Cerealia/Oryza). Timestamp in JOURNAL. DO THIS FIRST.
[ ] S1  Build results/core_coverage_table.csv from E214 deliverables + Neotoma/IPPD API. Mark 0–500 CE coverage per core.
[ ] S2  Extract empirical thresholds (results/empirical_thresholds.csv) = magnitude of each core's demonstrated late clearance signal (Dieng ~600 CE etc.). 
        >>> GO/NO-GO: if thresholds are not extractable, switch to the missing-core paper (§9) and stop overbuilding. <<<
[ ] S3  Read E196 coefficients from population_model.py; implement N×mode → cleared-area coupling.
[ ] S4  Implement/obtain REVEALS forward model; sweep published tropical RPP range; output pollen%/charcoal per core per (N,M).
[ ] S5  Compute per-core detection function + network power surface + spatial coverage map (figures/).
[ ] S6  Run model A (clearing) vs model B (dispersed) through identical pipeline; write the separation statement; define the E215 residual.
[ ] S7  Confound controls (noise band; charcoal+cultigen co-occurrence; Solo ~2950 BP worked case).
[ ] S8  Apply pre-registered rule → OUTCOME 1/2/3. If 3, compute decisive-missing-core spec.
[ ] S9  SIG G1–G9 self-run; Zenodo deposit (code+data+params); draft the paper for the outcome that fired.
```

**First action (and ONLY the first action — confirm with PI before proceeding past S2):** create `PREREG.md`, then do S1+S2. If the within-network positive control (e.g. the Dieng ~600 CE excursion magnitude) is real and resolvable from available data, the instrument's sensitivity is established and the whole design is viable. If not, fall back early to the honest missing-core-specification paper.

---

*Designed 2026-06-25 under the PI's Michelson–Morley mandate ("I am fine failing to find a pre-400 CE civilization, as long as it is proven definitively and falsifiably"). Design produced by a 9-agent ultracode workflow (4 candidate channels → adversarial escape-hatch stress-test → synthesis); the radiocarbon-SPD, archaeogenetic-Ne, and convergent-detectability candidates were considered and rejected for the reasons recorded in `docs/JOURNAL.md` (2026-06-25). Execution intentionally deferred to a cheaper model and gated behind the ME#19 forcing function.*

# E216 — Opus Review of the Sonnet Execution (2026-06-25)

**Reviewer:** Claude Opus 4.8 (design author, reviewing the Sonnet 4.6 execution)
**Verdict:** OUTCOME-3 is **correctly assigned** and the experiment is honest Track-B work.
The pre-registration was respected, the geometry is right, and the "missing core"
framing is the correct modal conclusion. **But there are 4 defects that must be fixed
before this is ever shown to an external judge.** None is fatal to the conclusion; all
are fatal to a clean submission. Do NOT rush this to a journal — it needs a palynologist
co-author (G2/G10) regardless, so there is time to fix these properly.

This review is filed, not silently applied. Pak Amien / Monday session decides.

---

## What is solid (keep)

- **Pre-registration discipline held.** PREREG.md was locked before analysis; the decision
  rule was applied as written. No outcome-shopping. This is exactly the SIG-F10 behaviour we want.
- **The core geometric insight is correct and is the real contribution:** detection is governed
  by *location relative to RSAP*, not lake size. A small lake at the heartland beats a giant
  marine catchment. That is a genuine, publishable methodological point.
- **Mode B residual correctly handed to E215.** No double-counting.
- **GRL 2025 (Ruan) correctly excluded** from triggering OUTCOME-2 (wrong proxy: brGDGT/levoglucosan,
  not charcoal+Cerealia). Good pre-registration hygiene.

---

## Defect 1 — `OUTCOME.json` contradicts itself on heartland coverage (MUST FIX)

`results/OUTCOME.json` says, from the **same run**:
- `stats.n_cores_covering_heartland: 1` (core **J6**, Solo marine)
- `key_finding`: "**No** existing Java palaeoecological core has its RSAP covering the
  Kedu/Brantas inscription heartland."

Both cannot be true as worded. The detection table confirms **J6 `heartland_in_rsap = True`**
(its 400 km marine RSAP geometrically reaches Brantas, 144.6 km away). The reconciliation is real
but unstated: J6 **geometrically reaches** the heartland yet **cannot resolve** it, because the
huge catchment dilutes the cleared-area fraction to ~0 (expected NAP rise 0.0015–0.003 pp).

The Sonnet summary's phrasing "semua 7 inti terlalu jauh / tidak ada inti yang RSAP-nya mencakup
heartland" is **factually wrong for J6** and a reviewer will catch it. Reword everywhere to the
honest two-tier statement:

> 1 core's RSAP geometrically reaches the heartland (J6, marine Solo), but catchment dilution
> drives its expected signal three orders of magnitude below threshold; **0 cores can *resolve*
> heartland clearing.** Coverage ≠ resolution.

## Defect 2 — "P(detect)" is effectively deterministic; parameter uncertainty is NOT propagated (MUST FIX)

Every `p_detect` in the table is exactly `0.0`; the missing-core spec reports `1.0`. The only
stochasticity in `detect_prob()` is pollen count noise (n=300 grains), which is negligible against
the gap between expected signal and threshold (z ≈ −77 → 0; z ≈ +6 → 1). So **"P=0.000" is not a
calibrated probability — it is a hard step function** on whether the *midpoint* expected signal
clears 17.5 pp.

The real uncertainty lives in the **parameters**, all of which are run at MID only for the outcome:
RPP_NAP (2–4), background NAP (4–10%), threshold (15–20 pp), arable/cultivation fractions, and the
clustering factor. PREREG equifinality-control #4 promised "RPP uncertainty → REVEALS as a
**sensitivity interval**." **That interval was never computed.** Deliver it: sweep RPP_NAP,
threshold, and clustering and report P(detect) as a band, not a point. Until then, drop the word
"probability" and say "expected signal relative to threshold."

## Defect 3 — Positive control is qualitative; this was actually the GO/NO-GO **NO-GO** branch (MUST FLAG)

PREREG S2 GO/NO-GO: GO only if the Dieng signal magnitude is **extractable from published data**;
otherwise "**NO-GO / direct OUTCOME-3**." The raw pollen data (Pudjoarinto 2001; Yulianto 2005) was
**paywalled (403)** and never extracted. So the 15–20 pp threshold is **borrowed literature consensus,
not re-derived** — and the run in fact hit the explicit NO-GO branch.

That is fine and consistent with OUTCOME-3 — but `OUTCOME.json` calling the control "**CONFIRMED**"
overstates it. Honest wording: *"instrument sensitivity is assumed from the original authors'
qualitative interpretation of Dieng ~600 CE clearance; we did not re-derive it from raw data."*
For SIG G1 (re-derive every headline number blind), the threshold is an **import**, not a derivation —
label it as such. Two independent reasons point to OUTCOME-3 (heartland geometry gap **and**
calibration not extractable); say both.

## Defect 4 — The "decisive missing core" headline hides a conservative corner where it ALSO fails (MOST IMPORTANT)

`missing_core_spec.json` reports `p_detect_if_core_at_kedu_floor: 1.0` and a NAP rise of 34.5 pp.
That number depends on a **hardcoded, uncited `CONCENTRATION_FACTOR = 4.0`** (heartland assumed 4×
Java-average clearing density). Re-running the spec's own forward model across the honest corners:

| Population | Clustering | Heartland density | NAP rise | Detect (>17.5 pp)? |
|---|---|---|---|---|
| floor (631k) | uniform (1×) | 9.0% | **12.6 pp** | **NO** |
| floor | 4× (assumed) | 36.0% | 34.5 pp | yes |
| central (1.27M) | uniform | 18.1% | 21.9 pp | yes |
| central | 4× | 72.5% | 48.8 pp | yes |

**At the conservative corner — floor population + uniform clearing — even a perfectly co-located
core would NOT detect (12.6 pp < 17.5 pp threshold).** The `why_decisive` field even admits the
range "~9–36%," and 9% is below threshold — but the JSON still asserts `p_detect = 1.0`. The
"decisive" claim is real only for clustered clearing OR central population, not at the floor.

This is the one finding that changes the paper's spine. The honest deliverable is **not** "a core at
Kedu would settle it" but: **"a core at Kedu would settle it *unless* the population is at the floor
**and** clearing was spatially diffuse, in which case the question stays open and the residual passes
to the dispersed-mode channel (E215)."** That caveat must be in the abstract, not buried in a constant.

---

## Strategic note (the part that matters most this week)

E216 is **Track B (curiosity, untimed)** and it is fine work. It does **not** discharge the
**ME#19 forcing function**, which is binding and overdue: the project's constraint is *non-exposure*,
not rigor. Producing a second internal artifact (design + execution) while the Verberne reply has
hung 14+ days is precisely the procrastination pattern ME#19 named. **Monday's priority is not
polishing E216 — it is Pak Amien sending the three external items** (Verberne v4 reply → Zenodo D1/D2
→ Lamqaddam). E216's fixes can wait for the palynologist co-author it needs anyway.

**Recommended status for E216:** SUCCESS (as a Track-B specification study), **NOT submission-ready.**
Fix Defects 1–4, recruit co-author (G2/G10), Zenodo deposit (G7), cross-model gate (G9) — *after*
the forcing function is discharged.

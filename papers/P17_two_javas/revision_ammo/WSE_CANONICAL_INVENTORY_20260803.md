# WS-E — P17 headline numbers re-derived on the canonical volcano inventory

**Date:** 2026-08-03 · **Line:** 02 taphonomy (WS-E sweep) · **Paper:** P17 *Two Javas*,
**ArchCalc submission 365, currently under review, double-blind.**
**Script:** `verify_p17_numbers.py` · **Machine-readable:** `p17_inventory_comparison.csv`
**Precedent:** `papers/P11_volcanic_informedness/revision_ammo/CANONICAL_INVENTORY_CORRECTIONS_20260610.md`

---

## Verdict first

**The paper's core finding survives, and one of its numbers gets stronger.** But **the methods
statement does not describe the computation that produced the published numbers**, and two published
figures cannot be reproduced from any single volcano list. Both need correcting with the editor.

| | |
|---|---|
| Spatial segregation candi vs inscriptions | ✅ **SURVIVES** — canonical medians 14.5 vs 27.6 km, Mann-Whitney **p = 1.5 × 10⁻⁷** |
| 13 km median gap | ✅ **SURVIVES** — 13.1 km on the canonical inventory (published 13.0) |
| Court-zone concentration 1.86× | ⚠ **CHANGES — and strengthens to 2.70×** (p < 10⁻⁴) |
| "nearest of 10 major Java volcanoes" (methods) | ❌ **DOES NOT MATCH** the published numbers |
| n = 176 inscriptions | ❌ **174** under the paper's own Java filter |

---

## 1. Why this sweep exists

P17 computes every headline number from *distance to the nearest volcano*, against a hand-picked list
of **10** centres (`draft_v0.3_archcalc.tex` l.110, l.169). The canonical inventory
`data/processed/dashboard/volcanoes_java_full.csv` holds **30**.

That is the same defect class that ended P7 at *Antiquity*: there, a 7-volcano subset produced
"deep-time sites 90–170 km from volcanoes" when the true figure was 33–53 km, and the reviewers were
right. P17's subset is at least **stated and justified** in the text ("responsible for the majority of
Holocene eruptions affecting the cultural heartland"), which P7's was not — but a stated rationale does
not make the number robust, and a live submission with an unchecked inventory is an open exposure.

E104's clean rebuild (2026-06-08) re-derived the two medians and found they held. It did **not**
re-derive the zone distribution, the Fisher odds ratio, or the Mann-Whitney statistic, and the
distribution block in `e104_court_zone.json` still carries `candi: 0` from the original,
non-reproducible run. This sweep closes that gap.

## 2. Results

Both arms use the paper's own Java filter (lat −8.9…−5.8, lon 105.0…114.8) and the same source files
(`E031_candi_orientation/results/candi_volcano_pairs.csv`,
`E082_inscription_georeferencing/results/geocoded_inscriptions.csv`).

| Quantity | Published | Re-derived on the **stated 10** | Re-derived on the **canonical 30** |
|---|---|---|---|
| candi n | 142 | 142 | 142 |
| inscriptions n | **176** | **174** | **174** |
| candi median distance | 14.6 km | 15.4 km | **14.5 km** |
| inscription median distance | 27.6 km | 28.2 km | **27.6 km** |
| median gap | 13.0 km | 12.7 km | **13.1 km** |
| Mann-Whitney | U = 8081, p < 10⁻⁶ | U = 8366, p = 7.4 × 10⁻⁷ | **U = 8125, p = 1.5 × 10⁻⁷** |
| candi peak zone | 0–10 km, 42.3% | 0–10 km, 42.3% | **0–10 km, 45.1%** |
| inscription peak zone | 20–30 km, 39.2% | 20–30 km, 38.5% | **20–30 km, 40.2%** |
| court concentration (Fisher) | 1.86×, p = 0.012 | 1.76×, p = 0.026 | **2.70×, p < 10⁻⁴** |

Zone counts on the canonical inventory:

| Zone | candi | inscriptions |
|---|---|---|
| 0–10 km | 64 | 22 |
| 10–20 km | 25 | 50 |
| 20–30 km | 36 | 70 |
| 30–40 km | 8 | 26 |
| 40–60 km | 5 | 5 |
| > 60 km | 4 | 1 |

## 3. The four findings

**F1 — the core claim is robust.** Restoring the 20 omitted centres moves the candi median by −0.9 km
and the inscription median by −0.6 km relative to the stated-10 computation, leaves the gap essentially
unchanged (12.7 → 13.1 km), and makes the segregation test *more* significant, not less. The Two Javas
structure is not an artefact of volcano selection. **This is the most important line in this document
and it should be stated in the paper**, not just held in the repo — a reviewer who repeats the analysis
on the full inventory will get a stronger result than the one published, which is a good position to be
in only if we got there first.

**F2 — the court-zone concentration must be restated: 1.86× → 2.70×.** This is not a rounding change.
On the canonical inventory, 26 inscriptions fall in the 30–40 km band versus 8 candi (against 5 and 5
on the stated-10 list): restoring the omitted centres moves a substantial number of inscriptions into
the court zone. The published 1.86× **understates** the paper's own effect. Correct it upward, and say
why.

**F3 — the methods statement does not describe the computation. ⚠ This is the serious one.** Neither
published median is reproducible from the stated 10-volcano list: the manuscript reports 14.6 / 27.6 km
and the stated list yields 15.4 / 28.2 km. The canonical 30 yields 14.5 / 27.6 km — much closer, but
the candi median is still 0.1 km off and U differs (8081 vs 8125). The E104 rebuild note records the
likely cause: the original analysis used **a 9-volcano list for candi and a 15-volcano list for
inscriptions** — two different rulers for the two groups being compared. If so, the published
comparison was never like-for-like, and the sentence *"computed as the haversine distance from each
site to the nearest of 10 major Java volcanoes"* describes an analysis that was not run.

The finding survives this — that is what the canonical re-derivation establishes. The methods sentence
does not.

**F4 — n = 174, not 176.** Two inscription records in the published count fall outside the paper's own
Java filter. Trivial in effect, but it is a number in the abstract and a reader can check it.

## 4. What this requires

**Editor-facing (PI action — P17 is live at ArchCalc, submission 365):**
A short correction note to the editor listing F2, F3 and F4, with the corrected numbers and the
statement that the core finding strengthens. Drafting this is worthwhile *now*: a correction offered
before review completes reads as diligence; the same correction after acceptance is an erratum, and
after a reviewer finds it, it is the P7 scenario again.

**Manuscript-facing (whenever the file is next opened):**
1. §Methods: replace the 10-volcano sentence with the canonical 30-centre inventory and cite the file.
2. Abstract and §Results: 14.6 → **14.5 km**; 13 → **13.1 km**; 1.86× → **2.70×**; n = 176 → **174**.
3. Add one sentence reporting that the finding was checked against a hand-picked subset and the full
   inventory and holds under both — this is a robustness result, and it is free.

**Repo-facing:**
- `experiments/E104_court_zone_hypothesis/results/e104_court_zone.json` still carries `candi: 0` in its
  distribution block. Regenerate it from this script's output so the file stops contradicting its own
  README.
- The canonical inventory spells Sindoro as **"Sundoro"** (Smithsonian GVP form). Any name-matching
  code needs the alias or it silently drops a Central Java volcano — this sweep hit exactly that and it
  cost one volcano until caught. Worth a line in `data/sources.md`.

---

*Re-derived 2026-08-03 from raw coordinates. Reproduce with
`python papers/P17_two_javas/revision_ammo/verify_p17_numbers.py`.*

# Draft — correction note to the ArchCalc editor, submission 365 (P17 *Two Javas*)

**Status: DRAFT, NOT SENT. PI action.** Written 2026-08-03 by Claude Code.
**Evidence:** `papers/P17_two_javas/revision_ammo/WSE_CANONICAL_INVENTORY_20260803.md`
(reproduce with `verify_p17_numbers.py`).

---

## Why send this, and why now

P17 is under review. Three published numbers are wrong and one methods sentence describes an analysis
that was not run. **The core finding is unaffected — it gets stronger** — which is exactly why sending
is cheap: this is a note that improves the paper.

The alternative timelines are worse in every case. If a reviewer recomputes on the full volcano
inventory, they find numbers that do not match ours and a methods sentence that does not reproduce;
that is the P7/*Antiquity* scenario, and P7 was rejected for precisely this. If nobody checks and the
paper is accepted, the correction becomes an erratum on a published article.

**Do not send until:** (a) Pak Amien has read the WS-E report, and (b) the corrected numbers have been
re-derived once more from the script. Both take under an hour.

## Sending notes

- Submission **365**, portal `submission.archcalc.cnr.it/submission/365`. Review is **double-blind**:
  send through the portal's editor channel, and do **not** attach anything containing author names,
  the repository URL, or institutional identifiers. The text below contains none.
- Keep it short. This is a notification, not a revision — the editor decides what happens next.
- Do **not** bundle any other request (no APC question, no status enquiry). One subject per message.

---

## TEXT — copy from here

Dear Editors,

I am writing about submission 365, currently under review, to report corrections to three reported
values and one methods statement, which I found in an internal audit of my own analysis code.

The paper computes every distance-based result against a hand-selected list of ten volcanic centres.
I have since re-derived all of these results against the complete inventory of thirty Javanese
volcanic centres. **The paper's central finding is unchanged and slightly strengthened**, but four
items in the submitted text need correcting:

1. **Methods.** The submitted text states that volcanic distance was computed to the nearest of ten
   major Java volcanoes. That sentence does not describe the computation that produced the reported
   numbers: on inspection, the two compared groups were measured against two different volcano lists.
   In the corrected analysis both groups are measured against the same complete thirty-centre
   inventory.

2. **Median distances.** Candi 14.5 km (reported 14.6 km); inscriptions 27.6 km (unchanged). The
   median gap becomes 13.1 km (reported 13 km). The Mann-Whitney test on the corrected distances gives
   U = 8125, p = 1.5 × 10⁻⁷ — the segregation is more strongly supported than in the submitted version,
   not less.

3. **Court-zone concentration.** The submitted text reports that inscriptions are 1.86 times more
   concentrated in the court zone than candi (p = 0.012). On the complete inventory this is
   **2.70 times (p < 10⁻⁴)**. The submitted figure understates the effect: restoring the omitted
   centres moves a substantial number of inscriptions into the 30–40 km band.

4. **Sample size.** The inscription sample is 174, not 176; two records fall outside the study area
   filter the paper itself defines.

I would be glad to supply the corrected analysis script and the comparison table, and to provide a
revised manuscript with these values at whatever point in the process you consider appropriate. I
recognise that the decision on how to handle a correction during review is yours; I would rather report
it now than let it stand.

With apologies for the additional work, and thanks for your time,

[name / ORCID / affiliation as the portal requires]

## — end of text —

---

## If the editors ask for the corrected manuscript immediately

Everything needed is in the repository and the edit is mechanical — four numbers plus one methods
sentence, listed in `WSE_CANONICAL_INVENTORY_20260803.md` §4. Add one sentence to §Results reporting
that the analysis was run on both a hand-picked subset and the full inventory and holds under both.
That converts the correction into a robustness statement, which is what it actually is.

## What NOT to say

- Do not describe this as a "minor" correction. A methods sentence that does not match the computation
  is not minor, and understating it invites the editor to discover that for themselves.
- Do not mention P7, *Antiquity*, or the wider audit programme. The editors are assessing this paper.
- Do not promise a timeline the PI cannot keep.

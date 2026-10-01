# DRAFT — notice to the Archeologia e Calcolatori editor: P17's central result does not hold

**Status:** DRAFT 2026-10-01, **not sent**. This is a PI decision with consequences: it concerns a paper
under double-blind review. **Channel:** the ArchCalc portal's editor/discussion channel for submission
**365** (`submission.archcalc.cnr.it/submission/365`), the same channel as the 11 Aug correction (queryId
162). Double-blind: attach nothing that names the author.
**Evidence:** ledger **C032**; `experiments/E082_inscription_georeferencing/README.md` (CORRECTION
2026-10-01); `robustness_geocoding_20261001.py` → `results/canonical30/robustness_geocoding_20261001.txt`.
Found by an Opus verifier on 2026-10-01 and re-derived independently the same day.

## The decision the PI has to make

| Option | What it means | When it fits |
|---|---|---|
| **A. Withdraw** | Tell the editor the central result is an artefact and withdraw the submission. | Cleanest. The paper's title claim ("spatial segregation") does not survive. |
| **B. Notify and offer a revised paper** | Tell the editor now, and ask whether a substantially revised manuscript would be considered. It would report the artefact: how geocoding placeholders and one monument's captions can manufacture a "two geographies" pattern. Otherwise withdraw. | Only if the PI wants to write that methods paper. P2 took the same route successfully: it became a paper about its own artefact. |
| ~~C. Wait for the reviews~~ | — | **Not acceptable** under the Submission Integrity Gate: a known-false central claim cannot be left in review. |

Either way, **the editor should hear about this soon.** The 11 Aug correction note told them the finding
"survives and strengthens"; this notice supersedes it.

---

## Text (Option A; the bracketed line gives Option B)

> Dear Editor,
>
> We are writing about submission 365, which is currently under review. While re-auditing our data, we
> found that the paper's central result does not hold. The paper reports that Hindu-Buddhist temples and
> inscriptions occupy different distance zones relative to Java's volcanoes. That contrast is an artefact
> of how the inscriptions were georeferenced. Of the 175 inscription records, 50 are relief captions from
> a single monument placed at one point, and 42 are placed at region-level centroids ("East Java",
> "Central Java"), which happen to fall in the distance band the paper describes. When only inscriptions
> with specific findspots are used (83 records), the difference in median distance to the nearest volcano
> falls from 13.1 km to between 0.2 and 2.3 km. It is no longer statistically distinguishable (p between
> 0.07 and 0.85, depending on whether duplicate temple coordinates are removed). This also supersedes the
> correction we sent on 11 August.
>
> Because the central claim does not survive, we wish to withdraw the submission. [Option B instead: We
> would be grateful to know whether you would consider a substantially revised manuscript that reports
> this artefact and its consequences for spatial inference from epigraphic corpora. If not, we will
> withdraw.] We apologise for the time this has taken from you and the reviewers, and we are grateful that
> it can be corrected before publication.
>
> With regards,
> The author

# P2 / JCAA #280 — reply to the editor's abstract request — ✅ SENT 2026-10-02 10:09 WIB (Gmail reply in the editor's thread; verified in Sent)

**Prepared:** 2026-10-01 (Claude, autonomous session; PI away). **Not sent.** Sending is the PI's call.
**Why:** on **31 Aug 2026** the handling editor (Dr César González-Pérez) corrected the title in the record
but **did not replace the abstract**: *"I find the new abstract to focus on the reviewing process that
your submission is going through rather than the work itself and its contributions. Please rewrite and
advise as necessary."* It has been unanswered for 31 days, while round-2 reviewers read a record whose
abstract still states withdrawn findings. The same day the paper was *"sent for a new round of peer
review"*.

**Verification:** two independent Opus verifiers (claim fidelity vs `review_package_20260727/10_SET_KLAIM_TERKOREKSI.md`;
editorial fit vs the editor's objection), workflow `p2-abstract-verify` 2026-10-01. Both returned
SEND_WITH_EDITS; every fix below is applied. Numbers: only 378 presences (manuscript l.189) and +0.042
(claim A4, +0.0424, 12/12 positive). No banned wording. JCAA's abstract limit was not found in the repo:
the 11 Aug abstract was 216 words; this one is ~240. **Check the portal field limit before pasting**; if it
is tighter, cut the clause after the em dash in sentence 6 first, then the "(availability domain)" gloss.

---

## How to send (pick one)

1. **Reply by email** in the thread *"RE: Revised files uploaded — and a request to update the title and
   abstract in the record"* (from cesar.gonzalez-perez@incipit.csic.es, 31 Aug, 3:50 PM). That is the
   channel he used.
2. Or post the same text in the portal's Review Discussion for #280.

---

## Email text

> Dear Dr González-Pérez,
>
> Thank you for correcting the title, and for your comment on the abstract. You are right that it
> described the review process rather than the work. Below is a replacement abstract that states the
> study and its contributions only; the findings are unchanged. We would be grateful if it could replace
> the abstract in the submission record. We will use the same text as the abstract of the manuscript
> file in its next version, so that the record and the manuscript match.
>
> **An Evaluation Artefact in Presence–Background Archaeological Modelling: Evidence from East Java and a
> Corrected Comparison Protocol**
>
> Presence–background (presence-only) predictive models in archaeology are commonly cross-validated
> against held-out negatives drawn from the same background used to fit them. We show that this practice
> can manufacture an apparent performance gain that does not survive a fair comparison. Using a
> settlement-suitability model for East Java, Indonesia (378 site presences; XGBoost, Random Forest and
> Maximum Entropy, under fixed spatial-block cross-validation), we compare background (pseudo-absence)
> designs in two ways: each scored against its own background, and all scored against a common evaluation
> background. Scored on their own backgrounds, designs intended to be increasingly realistic produce a
> ladder of AUC gains; held to a common background, redesigning the background contributes approximately
> nothing, whereas adding a single hydrological feature contributes +0.042 AUC. Simulations with known
> ground truth and the real data show why: along the background-design dial we swept, AUC computed on a
> design's own background has no interior optimum, so it gives no signal of where to stop, and across that
> dial own-background and common-background AUC on the real data move in opposite directions. A gain of
> this kind is easily mistaken for a substantive finding — that realistic background design improves
> prediction — and archaeological interpretations built on it inherit the error. We propose a corrected
> comparison protocol: hold the evaluation background fixed across designs, declare the background
> (availability domain) over which every metric is computed, and build priority maps from seed ensembles,
> because single-seed maps are unstable. The contribution is methodological: the diagnosis and the
> protocol, not a new site-prediction claim.
>
> With thanks,
>
> Mukhlis Amien, on behalf of both authors

---

## After sending

- Record the date in `lines/01_spatial/STATE.md` and `docs/WORKSTATE.md` §4 (P2 item closed).
- At the next revision, port this abstract into `submission_jcaa_v0.2.tex` lines 90–100 (write the dashes
  as `--` / `---`). The verifiers also flagged that the manuscript abstract, Fig. 15 caption (l.420) and
  Conclusions (l.565) say "true performance … on the real data" where only common-background performance
  is measured (claim C2). Fix those in the same pass.

# Draft: public correction for the P1 preprint on Zenodo (PI publishes)

**Record:** 10.5281/zenodo.19081502 — *Multi-Site Calibration of Volcanic Sedimentation Rates and
Implications for Archaeological Visibility in Java, Indonesia* (published 2026-03-18, file
`submission_v1.0.pdf`, the only version). **Status:** DRAFT 2026-10-01 (revised the same day), not posted.
**Why:** ledger C023 (inverted E069 reading) and C024 (burial rates computed from assumed burial-start
dates). **The methodology reviewer (WF2) rates this a mandatory action, not an optional one:** it is a public
record carrying two claims the project now knows to be wrong. There is no gatekeeper; it takes ~15 minutes.
**How to post (PI):** Zenodo → the record → *Edit* → paste the notice at the top of the description (enough
today). A corrected PDF can follow after the P1 audit (C024). The current manuscript `submission_v5.0.tex`
(→ *Archaeological Research in Asia*) no longer contains the E069 passage but still reports the 4.4 mm/yr mean.
**Quotes below are verbatim from the published PDF** (extracted 2026-10-01; line numbers are the PDF's own).

---

## Notice text (paste at the top of the record description)

> **Correction (1 October 2026).**
>
> **1. Survey-intensity regression (PDF lines 400–406).** The preprint states that "after controlling for
> three independent survey intensity proxies … volcanic proximity remained a significant negative predictor
> of site density (quasi-Poisson likelihood ratio test: β = -0.477, p = 0.0015 …, n = 703 grid cells). The
> volcanic site deficit is not reducible to differential survey effort; a residual burial signal persists."
> The coefficient in that model is on *distance* to the nearest volcano. A negative value therefore means
> that recorded sites become **more** numerous closer to volcanoes once the survey proxies are controlled,
> not fewer. The passage states the opposite of what the analysis shows; it provides no evidence of a
> survey-independent deficit or of a residual burial signal. The site inventory used (an
> OpenStreetMap/Wikipedia compilation with almost no period information, which may include modern
> monuments) is also unsuitable for that test. We withdraw the passage.
>
> **2. "Java-wide taphonomic baseline" (abstract, lines 17–22).** The abstract derives "a mean
> sedimentation rate of 4.4 ± 1.2 mm/yr" from "four sites with independently documented construction
> dates", and concludes that "burial is not a local anomaly but a Java-wide taphonomic baseline". The rates
> divide burial depth by time since construction, which assumes that burial began at construction. At
> Sambisari the temple surface stayed exposed for centuries before rapid burial. Four sites cannot
> establish a Java-wide baseline either, because burial is episodic and depends on landform. The 4.4 mm/yr
> figure should be read as a preliminary anchor (n = 4), not a validated baseline, and the depth
> projections built on it as illustrative. A revised version will re-examine the calibration, including
> the attribution of the Dwarapala anchor to the Kelud system.
>
> Details: https://github.com/neimasilk/volcarch-repo — `experiments/E069_adversarial_comparanda/adv3_survey_intensity/README.md`
> (section "CORRECTION 2026-10-01") and `docs/CRITIQUE_LEDGER.md` (C023, C024).

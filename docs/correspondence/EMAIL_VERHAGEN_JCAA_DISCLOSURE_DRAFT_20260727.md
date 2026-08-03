# Draft email — JCAA #280: self-disclosure + extension request

> ⛔ **SUPERSEDED 2026-08-03 — JANGAN KIRIM DRAFT INI.**
> Pengganti: **`EMAIL_VERHAGEN_EXTENSION_REQUEST_20260803.md`** (permintaan perpanjangan pendek;
> koreksi klaim dipindahkan ke *Response to Reviewers*, tempat yang semestinya).
>
> **Kenapa basi:** draft ini ditulis **pagi 27 Jul**, sebelum babak 2 (E222/E223). Angka-angkanya —
> termasuk *"mean −0.014 AUC, positive in 3 of 15"* dan bingkai *"the manuscript's main finding does
> not hold"* — sudah dikoreksi sendiri di `papers/P2_settlement_model/review_package_20260727/09_REVIEW_ATAS_BABAK2.md`
> §7 sebagai **over-generalisasi dari satu dataset**. `08_HANDOFF_BABAK2.md` juga sudah menandai draft
> ini butuh diperbarui ("isinya berubah materiil setelah babak 2").
>
> Disimpan sebagai jejak audit, bukan untuk dipakai.

**To:** Dr Philip Verhagen <journal@caa-international.org>, cc j.w.h.p.verhagen@vu.nl
**Subject:** Re: Editor Decision, JCAA #280 — revision scope query and extension request
**Status:** ⛔ **ON HOLD — DO NOT SEND (PI instruction, 2026-07-27).** Deadline is 2026-08-20; there is time
to understand the phenomenon properly before telling the editor the central claim is dead. Revisit only
after E218/E219 report (see `experiments/E218_evaluation_artefact/DESIGN.md`). If those confirm and explain
the artefact, this draft is close to send-ready; if they qualify it, the email needs rewriting anyway.
**Why send this early:** the editor should hear about a self-identified error from us, before the revision
arrives, not discover a changed conclusion in the resubmitted file. Sending early also makes the extension
request a reasonable consequence of the finding rather than a scheduling complaint.
**Deliberately excluded:** the APC waiver. Mixing an integrity disclosure with a fee request weakens both.
Raise the waiver separately, at submission.

---

Dear Dr Verhagen,

Thank you for the decision on manuscript #280 and for forwarding two genuinely useful reviews. I am
writing before submitting the revision because addressing one of the reviewers' requests has produced a
result that changes the manuscript's central conclusion, and I would rather raise that with you now than
present it as a surprise in the resubmitted file.

Reviewer 1 asked, as an essential point, why the study does not benchmark against Maximum Entropy methods.
I built that benchmark: MaxEnt, XGBoost and Random Forest across the same three pseudo-absence designs
reported in the manuscript, under identical spatial block cross-validation folds. The reimplementation
reproduces the submitted results closely, including the paper's published headline (seed-averaged AUC
0.750 against the reported 0.751) and the unexplained realised hard-negative fraction of 0.62 that the
Methods section flags.

The benchmark also made it necessary to score each background design on a common evaluation set rather
than on the negatives it had selected for itself. Under that comparison, the manuscript's main finding
does not hold. The reported improvement across the experimental ladder (AUC 0.659 to 0.751) is almost
entirely attributable to the evaluation background changing along with the training background: designs
that draw negatives further from the presences in environmental space produce an easier test set and a
higher AUC without any gain in transfer. Holding the evaluation background fixed, the effect of
background redesign is approximately zero (mean −0.014 AUC, positive in 3 of 15 paired comparisons across
five seeds and three algorithms), while adding a single hydrological covariate gives +0.042 AUC in 60 of
60 paired comparisons. This is the standard caution about AUC comparability across background samples
(Lobo et al. 2008) — which the manuscript cites and then does not apply to its own experimental sequence.

I therefore cannot submit a revision that defends the claim as published. My proposal is to reframe the
paper around the corrected result: that the apparent benefit of pseudo-absence redesign in archaeological
presence-background modelling can be an artefact of evaluation design, and that background designs must
be compared on a held-fixed evaluation set. The empirical material is the same, the MaxEnt benchmark
Reviewer 1 asked for is now central rather than supplementary, and the sharper research question meets
Reviewer 2's main objection. I believe this is a more useful contribution to JCAA readers than the
original, but it is a different conclusion, and that is your call rather than mine.

Two questions, then:

1. Would you prefer this as a revision of #280, or as a withdrawal and fresh submission? I am content
   either way; the reframed manuscript will disclose the correction and its history explicitly.
2. May I have an extension to **30 September 2026**? The rewrite involves a new framing, the additional
   comparative analysis, and revised figures, and I would rather deliver it properly than quickly.

I am happy to send the benchmark code and results now if that would help you judge the proposal. My
thanks to both reviewers — Reviewer 1's insistence on engaging the MaxEnt literature is what surfaced
this, and the paper is better for it.

With kind regards,

Mukhlis Amien
Lab Data Sains, Universitas Bhinneka Nusantara
Malang, Indonesia
ORCID: 0000-0002-1848-167X
amien@ubhinus.ac.id

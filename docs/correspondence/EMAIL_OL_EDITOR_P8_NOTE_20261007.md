# Note to the editor of *Oceanic Linguistics* before the P8 revision is written (decision D7) — text for sending

**Status:** drafted 2026-10-07 by Claude on the PI's instruction of the same morning (G16 waived for P8, ledger C067);
**not yet sent** — shown to the PI first. Facts from `EMAIL_OL_EDITOR_P8_NOTE_DRAFT_20261006.md` (A1–A10); decisions as
of 2026-10-07 (D1–D6, D8, D9 carried out; D4 = option (a)).
**Channel:** reply to the decision e-mail of 2026-10-05 (sender oceanicl@uhpress.org, subject "Decision Letter for
OL-03-2026-11"), from the PI's university address, with `tools/mail/gmail_compose_send.py` (new message to that address,
subject below). The portal's "Send Manuscript Correspondence" form (CAPTCHA) is the alternative if the e-mail bounces.
**Public repo:** no reviewer text is quoted; no portal link.

---

**To:** oceanicl@uhpress.org
**Subject:** Re: Decision Letter for OL-03-2026-11 — a note from the authors before we revise

Dear Professor Adelaar,

Thank you for the decision on our manuscript OL-03-2026-11 and for the two reports. We will revise along the lines of both reports and give particular attention to the readability of Section 3, as the editors ask.

Before we rewrite, there is something we should tell you. In preparing the revision we re-derived every number in the manuscript from the raw ABVD files (an audit of 85 statements, followed by the robustness section). The stored result files reproduce from the data, but the manuscript text contains errors of our own: some figures and two directional statements are wrong, one test does not test the hypothesis it is attached to, one agreement measure was computed inside the training data, and several descriptions do not match what was computed. Some of these came to light through the reviewers' questions (the semantic input inside a "phonological" fingerprint, the spelling of the glottal stop, the meaning of the label and of "false positive"); others go beyond what the reports raise.

In brief: two steps described in the Methods were not carried out as described (the Proto-Austronesian cross-check compared no forms; the loanword filter matched five unrelated words); a few printed figures were constants written into the code rather than values computed from the data; and the "two-method consensus" compared a model with the label it had been trained on, so the 266 "high-confidence" forms and the kappa of 0.61 are withdrawn (out of sample the agreement is 0.31).

The abstract will change accordingly. The candidate set is 438 forms under one definition, 32.3 percent of the corpus (the 26.5 percent belonged to a smaller second set, and the Tolaki list the first reviewer asks us to release therefore has 134 items, not 114). The classifier uses inputs from the written form and from the meaning, so "exclusively phonological" is withdrawn; the reported 0.76 also used the identity of the source list, and without it the AUC is 0.73, or 0.67 from the written form alone. The statement about fewer canonical prefixes is withdrawn: the data do not show it, and at equal word length no difference can be separated from length. The claim that the result does not depend on orthography or word length is narrowed. The negative result rested on an inappropriate test; a permutation test finds that, among uncoded forms, same-meaning forms are slightly more alike across lists than different-meaning forms, and we will report that and no more. The geographic pattern over sixteen further languages is not supported and is withdrawn.

What the paper now claims is more modest: a ranking aid of moderate strength rather than a detector of substrate vocabulary, a description of what the absence of a cognate-set number means for Makasar and for Tolaki, with the counts the first reviewer asked for, and a released list of the forms for specialist inspection. Four figures become one; the consensus, clustering, ablation and sixteen-language sections are removed, as is the section on Javanese script; the title will change. One cited reference (Ross 2005) is given with the wrong venue and will be corrected.

May these corrections be handled within the revision you have invited, each listed in a separate section of our response to the reviewers, or would you prefer another procedure, such as a further look by the reviewers? We will follow whichever you prefer. Two practical questions: are there any author-side charges (page, figure or other) — we would not take a paid open-access option; and should the revised manuscript remain anonymised, is a tracked-changes version wanted beside the clean copy, and under which file type should the response letter be uploaded?

The re-derivation and the revision analyses were carried out with AI assistance, as the manuscript's AI declaration already states; the revised declaration will say so for the text as well.

With best regards,

Mukhlis Amien (corresponding author), on behalf of both authors
Universitas Bhinneka Nusantara

---

## After sending

Record the date here, in `docs/WORKSTATE.md` §1 and in the line STATE. The editor's answer stays in the mailbox (paraphrase
in the repo); the answer to the charges question closes G15 ("e-mail editor, date").

# MASTERPIECE — The One Paper, Slow-Cooked

**Status:** ACTIVE INCUBATION
**Created:** 2026-04-22 (Session 21, after 3-AI critical review convergence)
**Authorship gate:** 4-AI consensus (Claude + DeepSeek + Gemini Pro + ChatGPT Go)

---

## What this folder is

This is the **single masterpiece paper** — the one Pak Amien identified as the real object of the VOLCARCH project. It is not a velocity paper. It is not on a submission clock. It is the landmark paper the project exists to produce.

Everything else in `papers/` (P0, P2, P7, P8, P11, P16, P17, P18, P19, D1, D2) operates on a normal academic cycle: draft → submit → revise → publish. This folder does not.

## The rules

### Rule 1 — There is only ONE masterpiece

No P-numbered counterparts. No "MASTERPIECE v2" spin-offs. One paper at a time in incubation. If a new idea wants to be the masterpiece, the current candidate must first be retired or graduated.

### Rule 2 — The 4-AI consensus gate

Before submission to any journal, the paper must be reviewed by **all four** of the following AI skeptical-reviewer panels and receive a verdict of "this is masterpiece level" (or equivalent — e.g., "accept with minor revision", "genuinely paradigm-relevant", "top 1% of submissions in this field") from each:

1. **Claude** (via the author's direct collaboration session)
2. **DeepSeek** (via `tools/cross_model_review.py` API call)
3. **Gemini 3 Pro** (via Playwright MCP on personal account `neimasilk@gmail.com`)
4. **ChatGPT Go** (via Playwright MCP on `cprastiasih@gmail.com`)

If **any** of the four returns "REJECT" or "MAJOR REVISION" as its overall verdict, the paper is not ready. Return to incubation. No exceptions — not "3 out of 4 is good enough," not "the one that rejected is wrong," not "peer review will agree with the 3." **Four-of-four, or hold.**

### Rule 3 — Santai dalam waktu, serius dalam standar ilmiah

Incubation timeline: as long as it takes. 2 years is acceptable. 3 years is acceptable. The purpose is not to publish fast; the purpose is to publish *once*, correctly, at a level that shifts the field.

### Rule 4 — External (non-AI) validation required before final submission

After the 4-AI gate passes, the paper still requires:
- Pak Amien deep review
- Domain co-author (archaeology, not AI-only collaboration) — **non-negotiable per DeepSeek & Gemini & ChatGPT convergence**
- External statistics/methodology reviewer (budget $50-200, Fiverr or academic freelance)
- Optional: one Indonesian archaeologist second read (KITLV, BRIN, Castillo, or PhD-track connection once established)

### Rule 5 — Incubation ≠ silence

During incubation, other papers still get submitted normally. The masterpiece being slow-cooked does not pause the pipeline of P0, P2, P5, P7, P8, P9, P11, P16–P19, or future diamond-hunt papers (P23 satellite, P24 InSAR, etc.). The masterpiece is a parallel track, not a blocker.

### Rule 6 — Material may be moved in and out during incubation

The masterpiece is allowed to cannibalize from other VOLCARCH papers as they mature, and allowed to export material to other VOLCARCH papers when sub-arguments turn out to be publishable on their own. This is working-as-intended, not scope creep.

---

## What's in here now

This folder is the new home for what was `papers/P1_taphonomic_framework/` — the paper that went through v2 → v3 → v4 → v5 and received rejections from:
- Asian Perspectives (desk reject, 2026-03)
- EGQSJ (desk reject 2026-04-16, "structure/wording, not content")
- Internal cross-model critical review (DeepSeek + Gemini: REJECT on v3, softer REJECT on v4, MAJOR REVISION on v5)

The v5.0 "challenger pivot" and the full version history will live in `papers/P1_taphonomic_framework/` as a historical archive. **Active masterpiece incubation happens here in `papers/MASTERPIECE/`**, starting from a clean document.

### Incubation strategy (initial, to be revised)

The masterpiece is NOT simply a polished version of P1 v5.0. The 3-AI review convergence (2026-04-22) revealed that:

1. **Evidence foundation is thin** (all three AIs) — proxy-stack without physical anchor.
2. **Research Statement v4.3 is rhetoric, not theory** (ChatGPT) — needs formal specification.
3. **Multi-channel convergence = posterior stacking on correlated likelihoods** (ChatGPT) — convergence is not evidence when channels share hidden variables.
4. **Pre-Hindu material culture assumption unchecked** (Gemini) — if bamboo/wood/earthwork, no remote-sensing signature possible.
5. **Diamond-hunt pivot = rearranging deck chairs** (all three) unless calibrated on known period first (Gemini's "Engine not Discovery" pivot).
6. **Optimization target drift** (ChatGPT) — 207 experiments optimize for analyzable outputs, not decision-changing ones.
7. **Echo chamber at the paper level AND the review level** (all three) — AI-only review mimics peer review without finality.

A masterpiece that addresses all seven structurally will be very different from P1 v5.0. It may take the form of:

- A methodological-contribution paper that validates an AI detection engine on *known* historical periods (Majapahit, Mataram, VOC-era) before aiming at the pre-Hindu target — then uses the validated engine to make specific *testable predictions* about the pre-Hindu record that future fieldwork can verify.
- A demographic-anchor paper that establishes the 1–2M Java 400 CE population via formally specified Bayesian demographic modeling with rigorous sensitivity analysis and explicit engagement with alternative explanations.
- A "what I claim / what I do not claim" paper that clearly scopes the VOLCARCH framework, lists every load-bearing assumption, and provides a detailed pre-registered falsification protocol with field-verification plan.
- Something else entirely that emerges from the incubation.

No commitment to a specific form yet. That is the point.

---

## How incubation works

### Phase 0 — Fallow period (2026-04-22 to ~2026-06-01)

**No writing.** Active reading, note-taking, external-literature absorption only. Goal: arrive at a masterpiece-level *question* before attempting a masterpiece-level *answer*.

Specifically during this phase:
- Read Wolters 1999 in full
- Read Lombard (3 vols) in full
- Read Bloembergen & Eickhoff 2020 in full
- Read Pollock 2006 in full
- Read Ali 2011 in full
- Read 5 papers cited in each of DeepSeek, Gemini, ChatGPT reviews that we have not yet engaged
- Do not open a LaTeX file

### Phase 1 — Question selection (~2026-06-01 onward)

Select the *one question* the masterpiece answers. Write a one-paragraph statement of that question. Run it past all 4 AIs for pre-commitment sanity check ("is this question worth a 2-year incubation?"). Only then begin drafting.

### Phase 2 — Skeleton (~after Phase 1 clears)

A 5-page skeleton with formal theory, predictions, and falsification criteria. Again, 4-AI gate: do not proceed without all four signalling "yes, this structure is masterpiece-worthy."

### Phase 3 — Draft (as long as it takes)

Slow drafting. Iterate with external readers (Pak Amien, co-author, domain experts). Run 4-AI gate once every 6 months or on major structural change.

### Phase 4 — Pre-submission (4-AI final gate)

All four AIs return "accept" or "minor revision" verdicts. External reviewer(s) concur. Then and only then, submit to target journal.

### Phase 5 — Post-submission

Revise with actual reviewer feedback. Publish.

---

## What this folder is NOT

- Not a workspace for in-progress paper drafts during the Phase 0 fallow period
- Not a dumping ground for rejected paper iterations (those go to `papers/P1_taphonomic_framework/` archive)
- Not a venue for "quick polish" of existing work
- Not a substitute for the other papers in the pipeline

---

*"Michelangelo worked slowly after securing patronage. You are pre-patronage."* — ChatGPT Go, 2026-04-22
*The masterpiece exists. It is not finished. It is not late. It is being made correctly, or not at all.*

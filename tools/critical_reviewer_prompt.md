# Skeptical Reviewer Prompt — Cross-Model Echo Chamber Breaker

**Purpose:** Run an independent, critical peer-review simulation on VOLCARCH papers before submission. Addresses Mata Elang #13/#14/#15 critique of the closed Claude-Claude review loop.

**How to use:**
1. Copy prompt below.
2. Paste into a DIFFERENT model instance — DeepSeek R1, GPT-5, Gemini 2.5, or an open-weight model via Ollama. Do NOT use the same Claude session that drafted the paper.
3. Attach the paper (PDF or full LaTeX source).
4. Collect the review. Treat it as the paper's first external peer review even though it is not.
5. Address each critique in a response document before submitting to the journal.

**Budget:** DeepSeek R1 via API ≈ $0.50–2 per review. Claude 3.5 Sonnet via API ≈ $2–5. Ollama/local ≈ free (RTX 4080 runs 70B parameter models via 4-bit quantization).

---

## The Prompt

```
You are an unusually honest peer reviewer for the Journal of Anthropological Archaeology (or similar Q1 venue). Your job is to write the referee report that the paper DESERVES — not the one that is polite.

You have been asked to review the attached manuscript. The paper makes claims about an "invisible civilization" of pre-Hindu Nusantara that has been archaeologically erased. The authors are computational researchers based in Indonesia, and the paper relies heavily on computational methods rather than fieldwork.

Your background, for this review:
- You are a senior Southeast Asian archaeologist with 25 years of field experience on Java.
- You have supervised excavations at Trowulan, Penanggungan, and Kedulan.
- You are skeptical of computational archaeology that claims discoveries without touching dirt.
- You respect good methodology even when you disagree with conclusions.
- You believe the archaeological record should constrain theory, not the other way around.

Your task is to produce a rigorous referee report. Structure:

1. SUMMARY (3-5 sentences). What does the paper claim? What is its central contribution?

2. OVERALL ASSESSMENT (one sentence). Major revision / minor revision / reject.

3. MAJOR CONCERNS (3-7 items). These are concerns that must be addressed before publication. For each:
   - State the concern specifically (cite page/section numbers if possible).
   - Explain why it matters.
   - Suggest how the authors could address it OR explain why it cannot be addressed without new data.

4. METHODOLOGICAL CONCERNS. Specifically examine:
   - Are statistical methods appropriate? (power, multiple comparisons, assumptions)
   - Is the evidence multi-channel or is there hidden dependence?
   - Are the comparisons (Philippines, Japan, Thailand) controlled for confounders?
   - Is the falsification criterion actually falsifying?
   - Are the computational methods transparent and reproducible?

5. SPECIFIC CLAIMS TO CHALLENGE. For each, state whether you find the claim: (a) well-supported, (b) plausible but under-supported, (c) overreach, (d) unfalsifiable.

6. WHAT THE PAPER DOES WELL (2-3 items). Be honest even if critical overall. What is genuinely novel or useful?

7. WHAT THE PAPER SHOULD BE, IF NOT THIS. If you think the paper is overreaching, what would a more modest version look like that you would accept?

8. ONE QUESTION FOR THE AUTHORS. What is the single most important question whose answer would change your assessment?

DO NOT:
- Be diplomatic for its own sake. Diplomacy hides disagreement.
- Assume the authors have good intentions. Assume they have interesting results and imperfect methods.
- Cite literature you haven't read. If a reference is in the paper and you are uncertain about it, say "I would need to check [Ref X] before accepting this claim."
- Reward complexity. Simple valid arguments beat complex marginal ones.
- Accept "multiple lines of evidence" as a shield. Ask whether the lines are truly independent.

DO:
- Cite specific passages by section and paragraph.
- Distinguish "the evidence doesn't support this" from "I disagree with this."
- Be willing to recommend acceptance of parts you believe. Be willing to recommend rejection of parts you don't.
- Imagine this paper being used by future researchers. Does it help or harm the field?

Language: English, academic register, no informal phrasing. Aim for 1,500-2,500 words. Use numbered lists where they clarify; use prose for substantive argument.

At the end, produce a one-paragraph "editor's summary" that the Associate Editor could paste into a decision email. Start this paragraph with "Summary for Editor:".
```

---

## Specific Instruction Addenda for P1-core

When reviewing P1-core (calibration paper), add to the prompt:

```
SPECIAL FOCUS FOR THIS REVIEW: The paper claims a mean volcanic sedimentation rate of 4.4 ± 1.2 mm/yr based on four calibration sites (Dwarapala, Sambisari, Kedulan, Kimpulan). Assess specifically:

- Is four sites sufficient for the claim of "Java-wide" applicability?
- Are the depth measurements from BPCB reports reliable?
- Is the Dwarapala inference (half-buried → 185 cm over 535 years) actually justified? Could the photograph be a site-specific pedestal arrangement?
- Does the linear extrapolation to pre-400 CE depth (4-10 m) account for soil compaction?
- Is "cumulative sedimentation" distinguished from "local alluvial aggradation" in the projections?
- Do the authors adequately acknowledge that volcanic sediment thickness varies strongly with wind direction and distance from vent?
- Is the 51-pair eruption-site dataset genuinely independent from the 4-site calibration?
```

## Specific Instruction Addenda for P0

When reviewing P0 (synthesis masterpiece), add to the prompt:

```
SPECIAL FOCUS FOR THIS REVIEW: The paper claims five independent evidence channels converge on the existence of an archaeologically invisible pre-Hindu civilization of 1-2 million people in volcanic Java. Assess specifically:

- Is "invisible civilization" a meaningful archaeological category, or is this language sensationalizing the absence of evidence?
- Are the five channels truly independent, or does the same underlying assumption (volcanism is a plausible taphonomic agent) carry across all channels?
- The "selective survival" argument (bronze drums exist, settlements do not) — is this evidence of a filter, or an ad-hoc rescue of an argument that cannot be directly verified?
- The Philippines comparison uses ~275-340 pre-400 CE sites. Is this figure defensible? Is the Java "0 sites" figure defensible?
- The wayang argument (Semar as pre-Hindu deity demoted to servant) — is this historical linguistics, ethnography, or speculation?
- The genomic reinterpretation (no aDNA from Java = taphonomic erasure) — does this control for sampling bias in published paleogenomic studies?
- Does the paper overreach in proposing a 6-layer framework that cannot be falsified?
- Would a more modest paper — "Five Archaeological Filters in Volcanic Tropical Settings: Implications for the Java Record" — be more defensible?
```

---

## Post-Review Integration Protocol

After receiving the critical review:

1. Copy the review into `papers/P[0|1]/external_reviews/critical_[model]_[date].md`
2. For each MAJOR CONCERN, write a response document: `response_to_concern_[N].md`
3. Classify each concern:
   - **ACCEPT** — the concern is valid; revise the paper to address.
   - **PARTIALLY ACCEPT** — the concern has merit but the full critique overstates it; revise with qualification.
   - **REJECT WITH ARGUMENT** — the concern is based on a misreading; respond in rebuttal/cover letter.
   - **DEFER** — the concern requires data we do not have; acknowledge as limitation.
4. Revise the paper before submission to the target journal.
5. **Save the critical review with the paper.** If a real peer review objects to something the critical review anticipated, you will have a record of having addressed it.

---

## Cross-Model Triangulation

Maximum robustness: run the skeptical reviewer prompt across **three different models** and compare:

- **Claude (different instance, cold context):** trained on similar corpus to drafting model; expected to find the most sophisticated concerns.
- **DeepSeek R1:** different training lineage (Chinese corpus emphasis); expected to find methodology concerns Claude might miss.
- **GPT-4o or GPT-5:** OpenAI lineage; expected to find plausibility/narrative concerns.

Concerns flagged by 2 or 3 models are near-certain real concerns. Concerns flagged by only one model require judgment.

---

## What This Does NOT Replace

A simulated critical review is NOT the same as:
- An actual peer review by a named domain expert.
- A response from a collaborator who has run the code.
- Field verification of a predicted site.
- Replication by an independent research group.

It is a **cheap pre-commit check** — the echo-chamber equivalent of running unit tests before deploying code. It catches obvious mistakes. It does not validate the science.

---

*Created 2026-04-20 per Mata Elang #15 recommendation. Update with usage notes each time deployed.*

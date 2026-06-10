# Cross-Model Convergence Analysis — Session 19 (2026-04-21)

**Context:** L1 §9 Stop Criterion #6 specifies: *"If two or more independent skeptical cross-model reviews (different training corpus, different prompt instance) converge on the same methodological flaw that cannot be addressed by revision, the corresponding claim must be withdrawn."*

**Two reviews completed:**
- **DeepSeek** (`deepseek-chat`, Chinese-trained corpus) on P1 + P0
- **Gemini 2.5 Flash** (Google-trained corpus) on P1 + P0

**Total budget spent this session:** $0.004 (DeepSeek) + $0.009 (Gemini) = **$0.013 of $3.30**. Remaining: $3.287.

---

## P1-core v3.0 — Convergent concerns

| # | Concern | DeepSeek | Gemini | Consensus |
|---|---|:---:|:---:|:---:|
| 1 | Dwarapala is colonial anecdote, not geoarchaeological measurement | ✅ | ✅ | **TRIGGERED** |
| 2 | Monument-to-settlement transferability unproven | ✅ | ✅ | **TRIGGERED** |
| 3 | n=4 "convergence" statistically meaningless | ✅ | ✅ | **TRIGGERED** |
| 4 | Linear extrapolation ignores compaction in Table 2 | ✅ | ✅ | **TRIGGERED** |
| 5 | Spatial analysis autocorrelation invalidates p-values | ✅ | ✅ | **TRIGGERED** |
| 6 | Reframe as methodology/research proposal, remove "invisible civilization" | ✅ | ✅ | **TRIGGERED** |

### DeepSeek-only concerns (not replicated by Gemini)
- Delete §3.7/§4.4 spatial analysis entirely (Gemini suggests fix, not delete)

### Gemini-only concerns (not articulated by DeepSeek)
- 51-pair dataset lacks transparency (needs supplementary table)
- Circular reasoning with "invisible civilization" companion paper
- Spatial heterogeneity of volcanic activity not modelled

---

## P0 draft v0.1 — Convergent concerns

| # | Concern | DeepSeek | Gemini | Consensus |
|---|---|:---:|:---:|:---:|
| 1 | Foundational premise is circular / non-sequitur | ✅ | ✅ | **TRIGGERED** |
| 2 | Demographic modelling is speculation, not evidence | ✅ | ✅ | **TRIGGERED** |
| 3 | Five channels not truly independent (even P0 admits it) | ✅ | ✅ | **TRIGGERED** |
| 4 | Unfalsifiable in practice (escape hatches) | ✅ | ✅ | **TRIGGERED** |
| 5 | Dismisses / strawmans archaeological practice | ✅ | ✅ | **TRIGGERED** |
| 6 | "Civilization" terminology overreach | (implicit) | ✅ | **TRIGGERED** |
| 7 | Zero direct archaeological evidence for central claim | ✅ | ✅ | **TRIGGERED** |
| 8 | Reframe to methodology + Channel 1 only | ✅ | ✅ | **TRIGGERED** |

### DeepSeek-only concerns
- "Manufactured gap" framing (gap is model artifact, not empirical finding)

### Gemini-only concerns
- Undetailed channels (wayang, Semar, PAN *surat) are speculative cultural interpretation, unfalsifiable
- Uncritical reliance on unreviewed P1-core paper (circular dependency between P0 and P1)

---

## L1 §9 Stop Criterion #6 — STATUS

**TRIGGERED for both P1-core v3.0 and P0 draft v0.1.**

Both independent cross-model skeptical reviews converge on the same class of methodological flaws. This is a formal stop-criterion trigger per the updated L1 §9 (session 19).

### What the criterion requires

Per criterion text: *"the corresponding claim must be withdrawn."*

### What "the corresponding claim" means per paper

**P1-core:** The claim that the 4-site calibration is a reliable quantitative baseline for Java-wide burial projections. Both models reject the calibration as overreach.

**P0:** The claim of a 1-2M person "invisible civilization" requiring multi-channel taphonomic explanation. Both models reject this as unsupported by direct evidence.

### What remains defensible per both models

- **P1-core:** The observation that volcanic sedimentation is substantial, creates a detection horizon, and has implications for archaeological survey planning. The numerical specifics are under-supported; the qualitative insight is valid.

- **P0:** Channel 1 methodology (detection horizon from sedimentation) + the legitimate taphonomic/survey-bias question. The historical/demographic overlay is not defensible without direct evidence.

---

## Pivot criterion also TRIGGERED

L1 §9 new pivot criterion (added session 19): *"If cross-model critical review recommends reframing the paper as 'critical review + research proposal' rather than empirical finding, adopt that reframe for that specific paper before submission."*

**Both models recommended identical reframes:**
- P1-core → methodology/research-proposal framing
- P0 → Channel 1 + methodology only

This is **triggered**. Paper-level pivot is now mandated by our own stop criterion.

---

## Implications for Session 20 priorities

### Must do (per our own stop criterion)

1. **P1-core v3.0 → P1-core v4.0 pivot:** Rewrite to remove "calibration" framing, re-present as "detection horizon hypothesis + research proposal for rigorous geoarchaeological testing." Keep data and analysis; change epistemic status of claims.

2. **P0 reframe decision:** Either (a) reduce to Channel 1 + methodology ("Archaeological Invisibility in Volcanic Tropical Landscapes: A Framework"), or (b) withdraw grand synthesis until direct evidence arrives. Option (c) "proceed with current" is now incompatible with triggered stop criterion #6.

3. **Update WORKSTATE** to reflect that JASREP submission of P1-core v3.0 is now blocked by criterion #6. Revised target: P1-core v4.0 (with methodological reframe) ready in 2-3 weeks.

### Nice to do

4. Run Gemini 2.5 Pro (paid tier or wait for free quota reset) on P1-core to test whether even more capable model surfaces additional concerns.
5. Run Claude Sonnet via Anthropic API on both papers (if Pak Amien has Anthropic API access) — this tests whether Claude-different-instance produces the same echo-chamber pattern or genuinely independent critique.

---

## Meta-finding on validation

**Budget total for this validation infrastructure:** $0.013 (1.3 cents).

**Value produced:** Converged methodological critique from two independent models. The "echo chamber" concern that ME#14/#15 raised as RISK is now concretely demonstrated via convergent external signal. We have paid $0.013 to avoid a third JASREP rejection cycle (2-3 months each).

**For Pak Amien's consideration:** If we can run 3-4 more reviews across models (Claude, GPT, other) at this cost level, the validation infrastructure becomes a standard pre-submission step rather than a one-off. Total cost: <$0.10 per paper × N cross-model review = fraction of one Fiverr stats review.

---

*Cross-model convergence analysis produced 2026-04-21 Session 19. Both P1-core and P0 stop-criterion #6 triggered. Pivot mandated.*

# P0 v0.4 — Submission Pre-Flight Checklist

**Target journal:** Journal of Anthropological Archaeology (Elsevier, Q1, CiteScore ~3.5, IF ~2.3)
**Submission route:** Subscription (ZERO APC) — NOT gold OA
**Submission portal:** https://www.editorialmanager.com/yjaar/
**Status:** v0.4 is 48pp / ~10.5K words, compile clean. Ready for pre-flight audit before Pak Amien final review and submission.

---

## Pre-submission audit (must pass before upload)

### Content

- [ ] **Title** finalised. Current: "The Invisible Civilisation: Six Independent Lines of Evidence for an Archaeologically Erased Pre-Hindu Nusantara." Check with Pak Amien — consider whether "Six Independent Lines" is too strong given ChatGPT's "correlated likelihoods" critique. Possible softening: "Six Converging Lines of Evidence" or "A Multi-Channel Synthesis."
- [ ] **Abstract ≤ 250 words.** Current abstract in draft_v0.4.tex — recount and trim if needed.
- [ ] **Keywords** (typically 4-6). Current: "archaeological taphonomy; multi-channel convergence; pre-Hindu Nusantara; invisible civilisations; selective survival; volcanic tropical archaeology; Indonesian prehistory" (7 — trim to 5-6).
- [ ] **Deep-interior claim qualifier** present in §8 Limitations (already added v0.3).
- [ ] **"Civilisation" vocabulary discipline** maintained throughout per SLR Fase D synthesis guidance (reserve "civilisation" for contexts with evidenced organisational complexity; default "substrate community" elsewhere).
- [ ] **Ye-tiao 132 CE embassy** paragraph present in §2.2 (verified present v0.3+).
- [ ] **Channel 6 split 6A-bronze / 6B-glass** with honest chronology framing (verified v0.3+).
- [ ] **E208 corpus-scale attenuation** honestly reported in §3.3 — both 189-term and 5019-synset scales, neither dismissed.

### Elsevier required declarations (before References)

- [ ] **CRediT author statement** using exact Elsevier taxonomy:
  - Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Resources, Data curation, Writing – original draft, Writing – review & editing, Visualization, Supervision, Project administration, Funding acquisition
  - For P0 (Amien + Gunawan): suggest Amien = Conceptualization, Methodology, Software, Formal analysis, Investigation, Writing – original draft; Gunawan = Software, Data curation, Writing – review & editing
  - Confirm exact split with co-author

- [ ] **Funding statement:** "This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors." (unless otherwise applicable)

- [ ] **Declaration of competing interests:** standard template ("The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.")

- [ ] **Declaration of AI use (Elsevier required since 2024):** 
  > "During the preparation of this work the author(s) used [AI tool name, e.g., Claude (Anthropic), DeepSeek, Gemini] in order to [purpose, e.g., brainstorm argumentative structure, critique drafts via skeptical-reviewer prompts, draft sections subsequently revised by the author]. After using these tool(s), the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the publication."
  - Place BEFORE References section, as its own labelled paragraph.
  - Be specific about which AI and what role. Do NOT list AI as author.
  - Template available at `docs/AI_DISCLOSURE_TEMPLATE.md` (verify still current).

### Format & structure

- [ ] **Manuscript structure** matches standard Elsevier research article: Title, Authors, Affiliations, Abstract, Keywords, Introduction, [Main sections], Conclusions, Declarations, References. Current v0.4 complies.
- [ ] **References** in Harvard (author-date) style. `apalike.bst` compiles clean; JAA specifically uses a close variant. Check sample JAA 2024-2025 paper for exact format match.
- [ ] **Figures** — v0.4 currently has no figures. **Add at minimum:**
  - Fig 1: Dwarapala 1803 vs present (hook image; may need to source archival photo + permissions)
  - Fig 2: Map of six evidence channels' geographic loci
  - Fig 3: Selective survival matrix (Table 1 content could be visualised)
  - Fig 4: Six-filter framework
  - JAA accepts figures inline; submit as separate files at submission.
- [ ] **Tables** — v0.4 has Table 1 (channels), Table 2 (survival matrix), Table 3 (six filters). All self-contained.
- [ ] **Line numbering** enabled (`lineno` package in v0.4 — verified).
- [ ] **Double spacing** for submission (`setspace` package with `\doublespacing` — verified).
- [ ] **Page numbering** present.
- [ ] **PDF compile** clean. v0.4 last compiled 2026-04-22 with 0 errors / 0 bib warnings — re-verify before upload.

### Companion materials

- [ ] **Cover letter** drafted. Suggested structure:
  - Addressed to: Editor-in-Chief, Journal of Anthropological Archaeology
  - Para 1: Title + abstract summary in 2-3 sentences
  - Para 2: Significance — why this paper matters to JAA readership (multi-channel synthesis of a structural taphonomic problem in a Southeast Asian context; extends the theoretical conversation on invisible civilisations beyond single-region cases)
  - Para 3: Declaration that manuscript is not under consideration elsewhere + not previously published
  - Para 4: Acknowledge AI assistance disclosed in manuscript; note co-author approval; note manuscript is output of a multi-year single-investigator computational program with specific known limitations (per §8)
  - Para 5: Suggested reviewers (3-4 names, if P0 has them — verify)
  - Para 6: Thanks

- [ ] **Highlights** (if JAA requires; check current requirements — some Elsevier journals require 3-5 highlights ≤85 chars each):
  - "Six independent proxy channels converge on a pre-400 CE Javanese substrate."
  - "Cumulative volcanic burial places pre-Hindu horizon at 4–10 m in Java basins."
  - "Jatim glass and Pejeng bronze document invisibility of workshops in their own territory."
  - "Framework proposes 8 pre-registered falsifiable predictions for future fieldwork."
  - "Method: multi-channel proxy analysis under compound taphonomic filters."
  *(adjust counts to ≤85 chars each — approximate)*

- [ ] **Graphical abstract** (JAA may request): one-page visual summary. Optional.

- [ ] **Data availability statement:** if code/data released, state repository (Zenodo / GitHub); if not, state reason (ongoing research, privacy) + contact method.

- [ ] **Conflict of interest form** — download from Elsevier portal, sign, upload.

### External reviewer suggestions (3-5 names)

Candidates to consider (verify current affiliation + contact before listing):

- Peter Bellwood (ANU) — Austronesian prehistory, ISEA archaeology comparanda
- Pierre-Yves Manguin (EFEO) — early Southeast Asian maritime polities
- Philip Verhagen (VU Amsterdam) — computational archaeology, already JCAA editor
- John Miksic (NUS) — Javanese temple archaeology, familiar with Liangan/Sambisari contexts
- Daud Ali (UPenn) — early Indonesian inscriptions, author of our cited reference
- Dougald O'Reilly (ANU) — Southeast Asian archaeology, Angkor context comparanda

Avoid as conflict: Verberne (active PhD application), Blanke (active PhD application).

---

## Pak Amien's final review (REQUIRES-DEEP-REVIEW)

Items requiring Pak Amien's eyes before submission:

1. **Read draft_v0.4.pdf end-to-end** (48pp). Reserved reading time: 2-3 days.
2. **Confirm title** (current vs ChatGPT-softened alternative).
3. **Confirm abstract** and trim if needed.
4. **Approve CRediT split** between Amien and Gunawan.
5. **Confirm suggested reviewer list** — add or remove names based on current contact state.
6. **Approve AI disclosure wording** — specific models named (Claude Opus 4.7; DeepSeek via API; Gemini 3 Pro via browser for critical review; ChatGPT Go via browser for critical review).
7. **Final go/no-go decision** — submit as-is, or send to external reviewer first (budget $50-200 Fiverr/academic freelance as per ME#16 recommendation).

---

## What this checklist does NOT guarantee

This checklist ensures the manuscript is *procedurally* ready for submission. It does not:

- Validate that peer reviewers will accept the argument (they may not — we've had 5 editorial rejections on related work + 3 cross-model REJECTs on P0 v0.2/v0.3/v0.4).
- Protect against desk rejection on scope grounds. JAA accepts synthesis papers, but editor discretion applies.
- Shortcut the need for a domain co-author (flagged non-negotiable by DeepSeek + Gemini + ChatGPT independently).

**Submission strategy per ChatGPT meta-finding:** "credible partiality delivered on time" — submit imperfect but real work, let actual peer reviewers do their part of the critique loop, rather than further internal AI iteration.

---

## Suggested timeline

- **Day 1 (2026-04-23):** Pak Amien opens draft_v0.4.pdf, skim read.
- **Day 2–4:** Deep read with this checklist in hand.
- **Day 5:** Resolve decisions (title, reviewers, CRediT split, external reviewer budget).
- **Day 6–7:** Claude implements any revisions Pak Amien requests.
- **Day 8:** Cover letter drafted by Claude, Pak Amien edits for voice.
- **Day 9:** Highlights + graphical abstract (if needed) drafted.
- **Day 10:** Submit.

Alternative if external reviewer engaged:
- **Day 1–5:** Same as above.
- **Day 6:** Send to Fiverr/academic freelancer for methodology + writing review ($50-200, 1 week turnaround).
- **Day 13:** Receive external review, revise.
- **Day 14–15:** Submit.

---

*Checklist produced 2026-04-22 Session 21. Update as JAA author guidelines refinements surface.*

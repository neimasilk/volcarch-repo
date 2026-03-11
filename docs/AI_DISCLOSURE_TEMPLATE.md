# AI Disclosure Template — VOLCARCH Papers

**Purpose:** Standardized AI disclosure statement for all VOLCARCH papers. Adapt per paper; keep the core framing consistent.

**Philosophy:** Full transparency, framed as methodological strength. Not "AI wrote it" but "AI-augmented single researcher operates at research-group scale."

---

## Full Version (for papers with substantial AI involvement)

Use this for papers where AI was involved in experiments, analysis, and writing (P1, P2, P5, P7, P8, P9, and all future papers).

### LaTeX

```latex
\subsection*{AI Disclosure}

This research employed an AI-augmented workflow using Claude (Anthropic) for
literature synthesis, statistical scripting, cross-database comparison,
iterative experimental design, and manuscript drafting assistance.
The AI-augmented approach enabled a single interdisciplinary researcher to
execute [NUMBER] experiments across [DOMAINS] within [TIMEFRAME]---a throughput
that would typically require a multi-person research group.
The research hypotheses, domain interpretation, ethical decisions, and final
scholarly judgments were made by the human author(s).
All AI-generated code, statistical results, and manuscript text were reviewed,
validated, and edited by the author(s).
```

### Plain text (for submission forms)

```
This research used an AI-augmented workflow (Claude, Anthropic) for literature
synthesis, statistical scripting, cross-database comparison, and manuscript
drafting. The AI approach enabled a single researcher to execute [N] experiments
across [domains]---throughput typically requiring a multi-person team. Research
hypotheses, domain interpretation, and scholarly judgment were made by the human
author(s). All outputs were reviewed and validated by the author(s).
```

---

## Paper-Specific Adaptations

### P1 — Taphonomic Bias Framework (Asian Perspectives)
**Status:** SUBMITTED — add during revision if requested.

```latex
\subsection*{AI Disclosure}

This research employed an AI-augmented workflow using Claude (Anthropic) for
literature synthesis across archaeological, geological, and volcanological
databases, geocoding of 666 archaeological sites, statistical analysis
(sedimentation rate calibration, spatial correlation), and manuscript drafting
assistance. The AI-augmented approach enabled a single researcher to compile
and cross-reference multi-disciplinary datasets spanning four volcanic systems
and six centuries of eruption records. The research hypotheses (volcanic
taphonomic bias as systematic invisibility factor), domain interpretation
(Dwarapala case study, sedimentation rate calibration), and final scholarly
judgments were made by the human author. All AI-generated code, statistical
results, and text were reviewed, validated, and edited by the author.
```

### P2 — Settlement Suitability Model (JCAA)
**Status:** SUBMITTED with minimal disclosure. Expand during revision.

Current (submitted):
> AI-assisted tools (Claude, Anthropic) were used during manuscript preparation
> for code development assistance, data analysis scripting, and figure generation.

Revision version (stronger):
```latex
\subsection*{AI Disclosure}

This research employed an AI-augmented workflow using Claude (Anthropic) for
literature synthesis, XGBoost model development and iteration (seven experimental
configurations, E007--E013), spatial cross-validation implementation, SHAP
analysis, tautology test suite design, temporal validation (E014), figure
generation, and manuscript drafting assistance. The AI-augmented approach enabled
a single researcher to execute a seven-experiment iterative pipeline spanning
model development, bias correction, and multi-test validation---throughput
typically requiring a computational archaeology team. The research hypotheses
(taphonomic bias correction via pseudo-absence design, tautology elimination),
domain interpretation (Zone B/C survey prioritisation), and final scholarly
judgments were made by the human author(s). All AI-generated code, statistical
results, and text were reviewed, validated, and edited by the author(s).
```

### P5 — The Volcanic Ritual Clock (BKI)
**Status:** SUBMITTED — add during revision if requested.

```latex
\subsection*{AI Disclosure}

This research employed an AI-augmented workflow using Claude (Anthropic) for
cross-database synthesis (DHARMA prasasti corpus, Pulotu ethnographic database,
GVP eruption records, primbon manuscript extraction), statistical analysis,
and manuscript drafting assistance. The AI-augmented approach enabled a single
researcher to systematically screen 268 inscriptions for pre-Indic ritual
elements, cross-reference 137 Austronesian cultures for mortuary practices,
and extract decomposition-related passages from a 261-page Javanese primbon
manuscript. The research hypotheses (slametan as taphonomic calendar,
1000-day decomposition mapping), domain interpretation (volcanic soil chemistry
as mechanism, pre-Hindu ritual persistence), and final scholarly judgments were
made by the human author. All AI-generated analysis and text were reviewed,
validated, and edited by the author.
```

### P7 — Temporal Overlay Matrix (Antiquity)
**Status:** SUBMITTED — add during revision if requested.

```latex
\subsection*{AI Disclosure}

This research employed an AI-augmented workflow using Claude (Anthropic) for
spatial statistical analysis, dataset compilation (Mini-NusaRC, 48 deep-time
sites), figure generation, and manuscript drafting assistance. The research
hypotheses, domain interpretation, and scholarly judgments were made by the
human author. All outputs were reviewed and validated by the author.
```

### P8 — Phonological Fossils (Oceanic Linguistics)
**Status:** DRAFT — disclosure already added (expanded version).

See `papers/P8_linguistic_fossils/draft_v0.1.tex`, AI Disclosure section.

---

## Key Talking Points (for cover letters, reviewer responses)

1. **Scale argument:** "The AI-augmented workflow enabled execution of [N] experiments in [timeframe], covering [databases/languages/sites]. This scale of cross-disciplinary synthesis would typically require a multi-person research group."

2. **Intellectual ownership:** "All research hypotheses, domain interpretations, and scholarly judgments were made by the human author(s). The AI served as a computational research assistant, not as an intellectual contributor."

3. **Reproducibility:** "The AI-augmented workflow enhances rather than diminishes reproducibility: all code is version-controlled, all experiments are documented with hypotheses, methods, and results, and the AI's role is explicitly documented at each stage."

4. **Failure transparency:** "The AI-augmented approach also enabled rapid identification of negative results (e.g., [failed experiment]), which were documented rather than suppressed---a practice facilitated by the low marginal cost of each experimental iteration."

5. **First-mover framing:** "We believe transparent documentation of AI-augmented research workflows will become standard practice. By disclosing our methodology fully, we aim to contribute to emerging best practices in AI-assisted scholarship."

---

## Journal Policies (as of 2026-03)

| Journal | AI Policy | Source |
|---------|-----------|--------|
| JCAA (Ubiquity Press) | Requires AI disclosure | Submission guidelines |
| Asian Perspectives (UH Press) | No explicit policy yet | Check before revision |
| BKI (Brill) | Requires disclosure if AI used | Brill author guidelines |
| Antiquity (Cambridge UP) | Requires disclosure | Cambridge UP AI policy |
| Oceanic Linguistics (UH Press) | No explicit policy yet | Check before submission |

**All major publishers now require disclosure. Full transparency is both ethical and pragmatic.**

---

*Last updated: 2026-03-11*

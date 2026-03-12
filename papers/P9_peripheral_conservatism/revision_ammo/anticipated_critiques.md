# P9 Revision Ammo: Anticipated Critiques & Pre-Computed Responses

**Paper:** "Peripheral Conservatism in Western Austronesian Linguistic and Ritual Systems: Evidence from Java, Bali, and Madagascar"
**Journal:** JSEAS (Journal of Southeast Asian Studies, NUS Press)
**Authors:** Mukhlis Amien + Go Frendi Gunawan
**Prepared:** 2026-03-12

---

## Critique 1: "The cognacy comparison methodology is too simplistic — comparing ABVD forms against PMP is not real historical linguistics"

**Anticipated from:** Comparative linguist reviewer.

**Response:**
"We acknowledge that our cognacy comparison uses an automated binary matching approach (E043) rather than the systematic sound correspondence method standard in historical linguistics. This is explicitly noted as a limitation in §4.1.

However, our approach is defensible for two reasons:

1. **We are measuring RELATIVE differences, not absolute cognacy rates.** The exact cognacy percentage matters less than the Balinese > Javanese gradient. Whether the 'true' cognacy rates are 40.3% and 33.0% or some other values, the DIRECTION of the difference is robust: Balinese preserves more PMP forms than Javanese across all 5 semantic domains tested.

2. **The IPA and syllable robustness tests (E041, E042)** demonstrate that the underlying ML fingerprint is phonological, not orthographic. CV AUC changes by <0.01 under IPA approximation and syllable-count alternatives. The signal is robust to methodological variation.

If the reviewer requests formal sound correspondence analysis, we can collaborate with a linguist to validate the top-20 most diagnostic cognate pairs. This would strengthen the paper without changing its conclusions."

**Supporting data:** `experiments/E043_krama_alus_cognacy/README.md`, `experiments/E041_ipa_validation/`, `experiments/E042_syllable_validation/`

---

## Critique 2: "Tengger has LOWER cognacy than Javanese — this contradicts your peripheral conservatism hypothesis"

**Anticipated from:** Any reviewer who reads carefully. This is our strongest potential vulnerability.

**Response:**
"This is correct, and we address it in §4.2 as a key finding that REFINES our hypothesis:

**Peripheral conservatism operates at LARGE SCALE (Bali, Madagascar), not at small-isolate scale (Tengger).**

Small isolated communities like Tengger (population ~50,000, highland enclave) experience DRIFT, not conservation. With fewer speakers, less literacy, and less contact, random lexical replacement accelerates. Tengger's 27.7% PMP cognacy vs Javanese 33.0% (McNemar p=0.015, LOWER) is consistent with drift in small populations — the same mechanism well-documented in genetics (founder effect, genetic drift).

Large peripheral regions (Bali population ~4.3 million, Madagascar ~29 million) have enough speakers and internal complexity to CONSERVE forms that the core (Java) replaces under pressure from cosmopolitan contact, Sanskrit borrowing, and standardization.

This split result (H1 supported for large peripheries, H2 rejected for small isolates) is MORE interesting than a simple confirmation — it reveals the mechanism: **conservation requires critical mass.**"

**Supporting data:** `experiments/E043_krama_alus_cognacy/results/`

---

## Critique 3: "The Malagasy comparison is too distant — 8,000 km and ~800 years of independent development"

**Anticipated from:** Reviewer questioning comparability.

**Response:**
"Distance and time are features, not bugs. Malagasy culture departed Nusantara ~1200 CE and has been evolving independently for ~800 years in a completely different ecological context (non-volcanic, Bantu contact, later French colonization). If Malagasy STILL preserves Austronesian cultural and linguistic features at rates comparable to or higher than central Java, this is extremely strong evidence for peripheral conservatism.

Malagasy PMP cognacy: 40.8% — higher than both Javanese (33.0%) and Tengger (27.7%). This rate approximates a ~1200 CE Nusantaran baseline, before the full impact of Islamization-era lexical replacement in Java.

The botanical evidence (E044) independently supports this: both Malagasy and Javanese burial practices use Canarium (dammar/ramy) as sacred aromatic, a pan-Austronesian link that survived 8,000 km of separation. The Plumeria (kamboja) 'tradition' at Javanese graves is actually NEW WORLD (introduced ~1560) — a recent substitution that Malagasy practice did not undergo."

**Supporting data:** `experiments/E044_malagasy_burial_botany/README.md`

---

## Critique 4: "The paper tries to do too much — linguistics, botany, ritual, archaeology"

**Anticipated from:** Discipline-focused reviewer who prefers narrower scope.

**Response:**
"We understand this concern. The breadth is intentional and reflects the paper's central argument: peripheral conservatism is a MULTI-DOMAIN phenomenon, not limited to language or ritual alone. Demonstrating it in only one domain would invite the objection that it is domain-specific rather than structural.

Our evidence spans four domains:
1. **Linguistic:** PMP cognacy gradient (E043)
2. **Botanical:** Canarium aromatic link (E044)
3. **Ritual:** Famadihana ≈ double burial; ala masina ≈ hutan keramat
4. **Temporal:** Indianization wave pattern (E033)

Each domain reinforces the others. If any single domain is questioned, the others provide independent support. This consilience IS the argument.

However, if the reviewer requires disciplinary focus, we can restructure: linguistic evidence in main text, botanical and ritual evidence in supplementary material. This preserves the multi-domain argument while addressing formatting concerns."

---

## Critique 5: "No original fieldwork — all secondary data"

**Anticipated from:** Empirical researcher.

**Response:**
"Correct. This paper is explicitly a *computational synthesis* that identifies patterns across existing datasets (ABVD, DHARMA, Pulotu, ethnobotanical literature). We believe this is a legitimate and increasingly important mode of research, particularly for interdisciplinary questions that cross traditional disciplinary boundaries.

Our contribution is the FRAMEWORK (peripheral conservatism as taphonomic mechanism) and the EVIDENCE SYNTHESIS (4 domains, 9 experiments). The framework generates specific testable predictions that require fieldwork to confirm:

1. Systematic wordlist collection in Bali Aga communities (predicted: higher PMP cognacy than lowland Balinese)
2. Ethnobotanical survey of burial aromatics across the Nusantaran-Malagasy range
3. Documentation of variant slametan/famadihana timings and their correlation with local soil chemistry

We explicitly identify these as future directions (§6) and welcome collaboration with field researchers."

---

## Critique 6: "The AI disclosure is concerning — how much of this was AI-generated?"

**Anticipated from:** Reviewer or editor unfamiliar with AI-augmented research.

**Response:**
"We include full AI disclosure as a matter of principle and first-mover transparency. The key distinction:

- **AI-generated:** Code for experiments, statistical analysis scripts, initial literature searches, formatting. These are TOOLS.
- **Human-generated:** Research questions, hypothesis formulation, interpretation of results, theoretical framing, all substantive intellectual content.

The parallel: a researcher using SPSS or R for statistics discloses the software but is not considered to have 'AI-generated' their analysis. Claude Code functions as a computational research assistant — more capable than SPSS but in the same epistemological category.

Every experiment was reviewed by the human author, failed experiments were documented (E029, E038, E039), and inconvenient results were preserved honestly. The AI disclosure actually STRENGTHENS the paper: it demonstrates that a single researcher with AI tools can operate at research-group scale while maintaining intellectual ownership."

---

## Critique 7: "The 'Indianization as wave' argument (E033) is not novel"

**Anticipated from:** Historian familiar with de-Indianization literature.

**Response:**
"We agree that the qualitative argument (Indianization was not permanent) is established. Our contribution is the QUANTIFICATION: rho = -0.211, p = 0.030 on the Indic ratio decline across 268 dated inscriptions. This is, to our knowledge, the first time the 'wave' pattern has been demonstrated statistically rather than argued from selected exemplars.

Furthermore, our data shows that pre-Indic terms don't merely persist — they INCREASE in relative frequency (E030: rho = +0.502). This is a stronger claim than 'Indianization faded' — it says 'the substrate actively reasserted.'

The peripheral conservatism framework explains WHY: peripheries (Bali, Madagascar) preserved pre-Indianization forms, while the core (Java) underwent replacement and then partial reversion. This is a new interpretive framework for a known phenomenon."

**Supporting data:** `experiments/E033_indianization_curve/README.md`, `experiments/E030_prasasti_temporal_nlp/README.md`

---

## Additional Revision Resources

### Cross-AI Review Already Completed

P9 underwent 3 rounds of external review (ChatGPT + Gemini) before submission. 16 criticisms were identified and addressed. See `papers/P9_peripheral_conservatism/REVIEW_TRIAGE.md` for the full triage record.

### Experiments Available for Extended Analysis

| Experiment | What it adds | Execution time |
|-----------|-------------|----------------|
| New: Osing comparison (I-026) | Second "peripheral" test using East Javanese isolate | ~1 day |
| New: Bali Aga wordlist (I-027) | Highland Balinese predicted to have higher PMP cognacy | Needs ABVD data |
| E034 (Panji-Malagasy) | Negative result but informative — can cite in revision | Complete |

### Cross-Paper Reinforcement

- **P5 → P9:** P5's Indianization curve (E033) independently supports peripheral conservatism thesis. Different data (prasasti content vs cognacy rates), same conclusion.
- **P8 → P9:** P8's substrate detection (AUC 0.760) provides the ML fingerprint for the same pre-Austronesian layer that P9 argues peripheries conserve. If P8 is accepted, P9's linguistic substrate claims gain empirical backing.
- **P1 → P9:** P1's taphonomic bias framework explains WHY peripheral evidence was overlooked — the volcanic landscape buries central Javanese evidence, making the periphery appear culturally "lagging" rather than "conserving."

---

*Prepared 2026-03-12. Use when reviewer comments arrive.*

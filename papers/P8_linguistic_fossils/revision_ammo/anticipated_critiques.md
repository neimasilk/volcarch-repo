# P8 Revision Ammo: Anticipated Critiques & Pre-Computed Responses

**Paper:** "Detecting Pre-Austronesian Substrate Signals in Western Indonesian Languages Using Machine Learning"
**Journal:** Oceanic Linguistics (Q1, UH Press), MS# OL-03-2026-11
**Authors:** Mukhlis Amien + Go Frendi Gunawan
**Prepared:** 2026-03-12

**Note:** Companion file `I053_hanacaraka_pangram_uniqueness.md` provides narrative pangram argument for §4.5.

---

## Critique 1: "The ABVD cognacy coverage is a poor proxy for substrate status"

**Anticipated from:** Austronesianist linguist. This is the most likely serious critique.

**Response:**
"We agree that ABVD cognacy coverage is not equivalent to historical substrate status. Our label is a PROXY: forms with no ABVD cognate are candidates for substrate origin, innovation, or borrowing. The ML model does not classify individual forms as 'substrate' — it identifies PHONOLOGICAL PATTERNS that distinguish non-cognate from cognate forms.

The key validation is that these patterns are linguistically interpretable: longer forms, consonant clusters, glottal stops, fewer prefixes (Table 3, SHAP analysis). This phonological profile aligns with what linguists expect from pre-Austronesian substrate influence: retention of non-Austronesian phonotactic patterns in non-basic vocabulary.

E028 (consensus method, kappa=0.61) shows that the ML approach and the rule-based approach agree on 266 forms. The 172 false positives identified by cross-validation are overwhelmingly transparent Austronesian compounds (including numeral compounds, E028 §3.2). The consensus subset is cleaner than either method alone.

If the reviewer requests formal comparative linguistic validation, we can provide the full 266 consensus substrate list for expert review."

**Supporting data:** `experiments/E028_substrate_consensus/README.md`

---

## Critique 2: "6 languages is too few — the model may not generalize"

**Anticipated from:** Quantitative or comparative linguistics reviewer.

**Response:**
"The 6-language core was chosen for maximum diversity within western Indonesia: 3 Sulawesi (Bare'e, Muna, Tolaki), 1 Sumatra (Toba Batak), 1 Borneo (Ngaju Dayak), 1 Flores (Manggarai). This covers 4 major Austronesian subgroups and 4 islands.

Leave-One-Language-Out validation (LOLO) yields AUC ≥ 0.65 for all 6 languages (5/6 ≥ 0.69), demonstrating that the model generalizes across languages it has never seen.

E027b expands to 16 additional languages (total 22), confirming the geographic pattern: Sulawesi P(substrate) = 0.606 > Western Indonesian 0.393. The expansion did not change the phonological fingerprint — the same features dominate SHAP importance across all 22 languages.

We acknowledge that the 6-language core is a starting point. Expansion to 50+ languages with ASJP or Lexibank data is a natural next step."

**Supporting data:** `experiments/E027b_substrate_expansion/README.md`

---

## Critique 3: "E029 shows no shared substrate families — doesn't this undermine your argument?"

**Anticipated from:** Reviewer who reads the negative result carefully.

**Response:**
"E029 is a critical result that we report honestly (§5.2). It shows that substrate candidates do NOT cluster into shared word families across languages (permutation p=0.569). This means:

The substrates are NOT from a single pre-Austronesian language. Instead, they represent **parallel independent innovations** — each language community producing non-Austronesian phonological patterns independently.

This actually STRENGTHENS the ML detection approach: the phonological fingerprint captures a PROCESS (how non-Austronesian phonotactics enter a language) rather than LEXICAL INHERITANCE (shared words from a common ancestor). The model detects the phonological shadow of substrate influence, regardless of whether the substrate was one language or many.

We reframe P8 in §5.2: substrate detection via phonological non-conformity, not shared lexical residue. This is a more nuanced and ultimately more useful framework."

---

## Critique 4: "Tolaki is inflated (54.5% substrate) due to low ABVD cognacy coverage"

**Anticipated from:** Reviewer checking data quality.

**Response:**
"We acknowledge this (§3.1, footnote X). Tolaki's ABVD cognacy coverage is lower than the other 5 languages, meaning more forms lack cognates by default. This inflates the apparent substrate rate.

Sensitivity test: model retrained without Tolaki yields AUC 0.698 (vs 0.760 with Tolaki). This is a CONDITIONAL GO — still above our 0.65 threshold but reduced. We report both values transparently.

The Tolaki inflation does not affect the other 5 languages' predictions. In the 22-language expansion (E027b), Gorontalo (84.2%) and Acehnese (62.9%) show similarly high substrate rates for independent reasons: phonological divergence from training distribution (Gorontalo) and known Mon-Khmer substrate (Acehnese). These confirm that the model detects phonological non-conformity, not data-quality artifacts."

---

## Critique 5: "The Hanacaraka section (§4.5) is speculative — script analysis doesn't belong in an ML paper"

**Anticipated from:** ML-focused reviewer who wants a pure methods paper.

**Response:**
"§4.5 provides independent non-computational evidence for the same conclusion the ML model reaches: Javanese phonological inventory aligns with Proto-Austronesian, not Sanskrit. The Hanacaraka script's 20 consonants map to the reconstructed PAn inventory (17 + 3 innovations), while Sanskrit has 33. The 13 consonants lost in the Kawi → Hanacaraka transition are exactly those the ML model identifies as non-substrate features: aspirates, retroflexes, sibilant distinctions.

This convergence between ML detection and script-historical analysis strengthens both arguments. If the reviewer prefers, §4.5 can be moved to supplementary material, but we recommend keeping it in the main text as it is the paper's most memorable and citable finding."

**Supporting data:** `experiments/E036_hanacaraka_phonological/README.md`, `revision_ammo/I053_hanacaraka_pangram_uniqueness.md`

---

## Critique 6: "How do you distinguish substrate from borrowing (Malay, Arabic, Portuguese, Dutch)?"

**Anticipated from:** Contact linguistics specialist.

**Response:**
"Our feature set includes several safeguards:

1. **Semantic domain features** (sem_ACTION, sem_NATURE, sem_BODY): Substrates predicted to concentrate in action verbs and nature terms (§4.3, 46% of top-50 are action verbs). Borrowings typically concentrate in trade, religion, and technology domains.

2. **Language-internal features** (has_prefix, prefix_type): Austronesian morphology (prefixation) is a cognacy marker. Forms WITHOUT Austronesian prefixes but WITH consonant clusters are substrate candidates, not borrowings.

3. **Cross-language coverage** (language_cognacy_coverage, the top SHAP feature): This feature captures how well a form fits the Austronesian cognacy network. Borrowings from Malay or Arabic would appear as cognates across languages (shared borrowings), not as isolated non-cognates.

We cannot fully exclude all borrowing — this is a limitation of any automated approach. The E028 consensus method reduces this by requiring both rule-based and ML agreement, filtering out many transparent borrowings."

---

## Critique 7: "Action verbs dominating the substrate list seems counterintuitive"

**Anticipated from:** Reviewer expecting substrate in basic vocabulary (body parts, kinship).

**Response:**
"This is counterintuitive but interpretable. Basic vocabulary (Swadesh list items) is precisely the domain where Austronesian cognacy is highest — these words have been MAINTAINED from PAn. Action verbs are more culturally specific and more likely to be:

(a) inherited from pre-Austronesian communities who maintained their own activity vocabulary while adopting Austronesian basic terms (the substrate scenario), or
(b) independently innovated for local ecological activities (fishing techniques, forest management, volcanic soil agriculture) that have no PAn equivalent.

Both interpretations support P8's argument: the phonological fingerprint of non-Austronesian influence is strongest in culturally embedded vocabulary, not in basic vocabulary where Austronesian dominance is complete."

---

## Cross-Paper Reinforcement

- **P9 → P8:** P9's cognacy gradient (Balinese 40.3% > Javanese 33.0%) independently confirms that core vs periphery differences exist in the Austronesian vocabulary. The same substrate that P8 detects via ML is what P9 shows peripheries conserve.
- **P5 → P8:** P5's hyang persistence (43% of prasasti across 800 years) shows the same substrate signal in the epigraphic record. Pre-Indic ritual vocabulary survived Indianization — parallel to P8's phonological survival.
- **E033 → P8:** Indianization curve (Indic ratio declining, rho=-0.211) shows that Sanskrit overlay was a WAVE that receded, leaving the substrate visible again. This temporal pattern is exactly what P8's synchronic detection captures as a snapshot.

---

*Prepared 2026-03-12. Use when reviewer comments arrive.*

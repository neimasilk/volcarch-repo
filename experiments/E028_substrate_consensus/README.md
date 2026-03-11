# E028: Cross-Method Substrate Consensus Analysis

**Status:** SUCCESS

## Hypothesis

Forms identified as substrate by **both** E022 (rule-based ABVD subtraction) and E027 (ML phonological classifier) represent higher-confidence pre-Austronesian remnants than forms flagged by only one method. Cross-method agreement (Cohen's kappa) should be at least "fair" (>0.2) if both methods detect the same underlying signal.

## Method

1. **E022 (rule-based):** Subtracted ABVD cognates from ABVD wordlists for 6 Sulawesi languages. Forms without cognate matches are labeled "residual" (potential substrate). N=438 residuals out of 1,357 total forms.

2. **E027 (ML-based):** XGBoost Model B trained on phonological features (form length, vowel ratio, glottal stops, nasal clusters, consonant clusters, prefix patterns, etc.) to classify forms as Austronesian vs. substrate. Binary threshold at P(substrate) >= 0.5.

3. **Consensus:** Retrained E027 Model B on the full features matrix (1,357 forms) to obtain P(substrate) for every form, then cross-tabulated with E022 binary labels into four quadrants:
   - **CS (Consensus Substrate):** E022=residual AND ML P>=0.5
   - **CA (Consensus Austronesian):** E022=cognate AND ML P<0.5
   - **RO (Rule-only):** E022=residual BUT ML P<0.5
   - **MO (ML-only):** E022=cognate BUT ML P>=0.5

## Data

- E022 residuals: `experiments/E022_linguistic_subtraction/results/poc_residuals_detail.csv` (395 rows, 6 languages)
- E027 features: `experiments/E027_ml_substrate_detection/data/features_matrix.csv` (1,357 forms, 23 features)
- E027 ranking: `experiments/E027_ml_substrate_detection/results/substrate_ranking.csv` (438 rows)

## Results

### Agreement Statistics

| Metric | Value |
|--------|-------|
| Cohen's kappa | 0.6105 (substantial) |
| Pearson r(P_substrate, E022) | 0.7004 (p < 1e-200) |
| Spearman rho | 0.6631 (p < 1e-172) |

### Quadrant Distribution

| Quadrant | N | % | Mean P(substrate) | Interpretation |
|----------|---|---|-------------------|----------------|
| CS (Consensus Substrate) | 266 | 19.6% | 0.766 | High-confidence substrate |
| CA (Consensus Austronesian) | 878 | 64.7% | 0.173 | High-confidence Austronesian |
| RO (Rule-only) | 172 | 12.7% | 0.329 | E022 residuals the ML finds Austronesian-like |
| MO (ML-only) | 41 | 3.0% | 0.621 | Cognates the ML finds substrate-like |

60.7% of E022 residuals are confirmed by the ML classifier (266/438).

### Consensus Substrates by Language

| Language | CS | Total | % |
|----------|-----|-------|---|
| Tolaki | 121 | 209 | 57.9% |
| Makassar | 48 | 217 | 22.1% |
| Bugis | 36 | 242 | 14.9% |
| Wolio | 36 | 254 | 14.2% |
| Toraja-Sadan | 15 | 216 | 6.9% |
| Muna | 10 | 219 | 4.6% |

Tolaki dominates CS, consistent with E027's finding that Tolaki has the most phonologically distinct substrate signal.

### Semantic Domain Distribution (CS)

| Domain | N | % |
|--------|---|---|
| ACTION | 117 | 44.0% |
| GRAMMAR | 40 | 15.0% |
| QUALITY | 38 | 14.3% |
| NATURE | 20 | 7.5% |
| NUMBER | 20 | 7.5% |
| OTHER | 19 | 7.1% |
| BODY | 12 | 4.5% |

Action verbs dominate the consensus substrate, suggesting the pre-Austronesian language's verb vocabulary was partially retained even as Austronesian lexicon displaced most nouns.

### Cross-Language Consensus (4+/6 languages)

Five concepts appear as Consensus Substrate in 4 or more languages:

| Concept | Languages | Mean P(substrate) | Domain |
|---------|-----------|-------------------|--------|
| One Hundred | 4 | 0.829 | NUMBER |
| Fifty | 4 | 0.815 | NUMBER |
| Twenty | 4 | 0.723 | NUMBER |
| to stand | 4 | 0.710 | ACTION |
| to hit | 4 | 0.642 | ACTION |

Notably, three of five are numeral compound forms (Twenty, Fifty, One Hundred), which may reflect substrate numeral morphology rather than substrate lexical roots.

### Disagreement Analysis

**ML-only (MO) vs Consensus Austronesian (CA):**
MO forms are phonologically distinct from CA: longer (+1.5 chars), lower vowel ratio (-0.07), more glottals (+0.16), more nasal clusters (+0.19), many more consonant clusters (+0.43). These are plausibly missed substrates that happened to receive ABVD cognacy codes despite substrate-like phonology. Top candidates include Tolaki *lumangu* "to swim" (P=0.88) and *i lalo* "in, inside" (P=0.85).

**Rule-only (RO) vs Consensus Substrate (CS):**
RO forms are shorter (-1.2 chars), have fewer glottals (-0.22), and fewer consonant clusters (-0.21) than CS. They look phonologically Austronesian despite lacking ABVD cognacy matches. Many are likely E022 false positives: short forms (e.g., Muna *towu* "back", P=0.04; *riwu* "One Thousand", P=0.06) that simply had no ABVD match but are regular Austronesian vocabulary.

## Outputs

- `results/consensus_summary.json` -- Full agreement statistics and quadrant analysis
- `results/consensus_substrates.csv` -- 266 CS candidates ranked by P(substrate)
- `results/cross_language_consensus.csv` -- 5 concepts in CS for 4+/6 languages
- `results/quadrant_comparison.png` -- 4-panel visualization
- `results/verdict.json` -- Experiment verdict

## Verdict

**SUCCESS.** Substantial inter-method agreement (kappa = 0.611). The 266 Consensus Substrate forms represent a high-confidence core set of potential pre-Austronesian remnants, confirmed independently by both rule-based and ML methods. The 172 Rule-only forms (39.3% of E022 residuals) are likely false positives from the simple subtraction approach. The 41 ML-only forms warrant further investigation as potential missed substrates.

## Implications for P8

- The CS core set (266 forms) should replace the raw E022 residual list (438 forms) as the primary substrate candidate inventory for P8.
- The Tolaki dominance in CS (45.5% of all CS forms) requires careful interpretation: is Tolaki genuinely more substrate-rich, or is the ML biased by Tolaki's distinctive phonology? E027's LOLO analysis partially addresses this.
- The numeral-heavy cross-language consensus (3/5 concepts are numerals) suggests numeral compounding morphology may be a substrate signature worth investigating further.
- The 41 MO forms are actionable leads for expanding the substrate inventory beyond ABVD-based detection.

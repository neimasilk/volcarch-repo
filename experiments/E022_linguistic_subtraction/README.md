# E022: Linguistic Subtraction POC

## Hypothesis
Pre-Austronesian substrate vocabulary can be detected in Sulawesi languages by subtracting known loanword layers (Sanskrit, Arabic, Malay) and identified Austronesian cognates from basic vocabulary lists.

## Method
1. Loaded ABVD (Austronesian Basic Vocabulary Database) for 6 Sulawesi languages: Muna, Bugis, Makassar, Wolio, Toraja-Sa'dan, Tolaki
2. Tagged words with PAn cognacy codes (from ABVD cognate sets)
3. Tagged Sanskrit, Arabic, and Malay trade loanwords using known wordlists
4. Residual = words with NO cognacy code AND no loanword match

## Data
- ABVD CLDF dataset (lexibank/abvd, CC-BY 4.0)
- 210 Swadesh concepts per language, ~210-254 forms each

## Results

| Language | Total | Has Cognacy | Residual | % Residual |
|----------|-------|------------|----------|------------|
| Muna | 219 | 185 (84%) | 31 | 14.2% |
| Bugis | 242 | 180 (74%) | 55 | 22.7% |
| Makassar | 217 | 137 (63%) | 71 | 32.7% |
| Wolio | 254 | 171 (67%) | 73 | 28.7% |
| Toraja-Sa'dan | 216 | 171 (79%) | 40 | 18.5% |
| Tolaki | 209 | 75 (36%) | 125 | 59.8% |

**Average residual: 29.4%**

### Cross-Language Intersection (strongest substrate candidates)
Tier 1 (5-6 languages): 6 concepts — "to hit", "to see", "rope", "One Thousand", "to say", "to hold"
Tier 2 (4 languages): 18 concepts
Tier 3 (3 languages): 35 concepts

## Conclusion
**STATUS: SUCCESS — GO for full pipeline**

Substantial non-cognate, non-loanword residual detected across all 6 languages (14-60%, mean 29.4%). Six concepts are residual in 5+ of 6 languages — these are the strongest substrate candidates.

## Enhanced Analysis (enhanced_subtraction.py)

Added PAn cross-check layer: 15 known Proto-Austronesian reconstructions that ABVD missed cognacy codes for. Also fixed loan field case sensitivity bug.

### Enhanced Results
| Language | POC residual% | Enhanced residual% | PAn rescued |
|----------|--------------|-------------------|-------------|
| Muna | 28.2% | 11.9% | 7 |
| Bugis | 22.3% | 20.2% | 12 |
| Makassar | 22.1% | 30.9% | 12 |
| Wolio | 25.3% | 26.8% | 15 |
| Toraja-Sa'dan | 18.4% | 14.8% | 13 |
| Tolaki | 59.8% | 54.5% | 16 |

**Enhanced average: 26.5%** (vs POC 29.3%)

### Tier 1 Substrate Candidates (5-6 languages)
8 concepts remain after PAn cross-check: "if", "to bite", "to tie up", "to cut/hack", "grass", "to throw", "they", "big". These are the strongest pre-Austronesian or South Sulawesi substrate candidates.

## Caveats
1. Tolaki still inflated (54.5%) due to low cognacy coverage (36% in ABVD)
2. LingPy SCA scorer requires IPA tokenization — ABVD orthographic forms caused error
3. Enhanced PAn list is manual (15 entries) — should be expanded with Blust ACD
4. Full pipeline needs: IPA conversion, LingPy alignment, expert validation

## Next Steps
- [x] Install LingPy — DONE
- [x] PAn cross-check for known reconstructions — DONE (75 rescues)
- [ ] Convert ABVD forms to IPA for LingPy scorer
- [ ] Download de Casparis 1997 PDF for Sanskrit layer
- [ ] Access WOLD database for comparative loanword typology
- [ ] Expand beyond Swadesh-210 using van den Berg 1996 Muna dictionary
- [ ] Semantic field analysis of residual words (body parts? nature? kinship?)

## Paper
Links to: Paper 8 (Linguistic Fossils), draft at `docs/drafts/VOLCARCH_Paper8_LinguisticFossils_DRAFT.docx.pdf`

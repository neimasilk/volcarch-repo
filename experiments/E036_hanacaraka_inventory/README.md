# E036 — Hanacaraka Phonological Inventory Mapping

**Status:** SUCCESS
**Date:** 2026-03-10
**Idea ID:** I-006

## Hypothesis

The reduction from Sanskrit's 33 consonants to Hanacaraka's 20 consonants reveals which phonemes were NOT native to pre-Sanskrit Javanese. This "phonological fossil record" should align with Proto-Austronesian (PAn) rather than Sanskrit, and connect to E027's substrate phonological fingerprint.

## Method

1. Mapped all 33 Sanskrit/Devanagari consonants to their Hanacaraka equivalents
2. Classified each as RETAINED (20) or MERGED (13)
3. Categorized losses by phonological feature type
4. Compared Hanacaraka inventory with PAn reconstruction (Blust 2009)
5. Cross-referenced with E027 substrate fingerprint and modern Javanese

## Key Results

### The 33→20 Reduction

| Category | Lost | Phonemes |
|----------|------|----------|
| Aspiration | 8 | kha, gha, cha, jha, pha, bha, ttha, ddha |
| Retroflex | 5 | tta, ttha, dda, ddha, nna |
| Sibilant distinction | 2 | sha (palatal), ssa (retroflex) |
| **Total merged** | **13** | |

*Note: ttha and ddha count in both aspiration and retroflex categories.*

### Alignment with Proto-Austronesian

| Feature | PAn (~3000 BCE) | Hanacaraka (~800 CE) | Sanskrit |
|---------|-----------------|---------------------|----------|
| Aspiration contrast | NO | NO (except tha/dha) | YES (8 pairs) |
| Retroflex series | NO | NO | YES (6 phonemes) |
| Multiple sibilants | NO | NO (only /s/) | YES (3) |
| Palatal nasal /ny/ | NO | YES | YES |
| Glottal stop /ʔ/ | YES | No symbol | No symbol |
| Total consonants | ~17 | 20 | 33 |

**Hanacaraka (20) is much closer to PAn (17) than to Sanskrit (33).**

### Two Paradoxes

1. **tha/dha paradox:** All 8 aspiration contrasts were dropped EXCEPT dental aspirates tha and dha. Why? Possible explanations: (a) pre-Sanskrit Javanese had these from a substrate language, (b) they were phonemically important for common words. The fact that tha/dha are now LOST in modern standard Javanese confirms they were archaic.

2. **Glottal stop paradox:** Modern Javanese has phonemic /ʔ/, E027 shows substrate words have MORE glottal stops, but neither Hanacaraka nor Devanagari has a symbol for it. This proves /ʔ/ existed BEFORE writing was adopted — it's a pre-script feature that couldn't be represented because the script source language (Sanskrit) didn't have it.

### Modern Javanese Developments

- Retroflexes RE-EMERGED (/ʈ/, /ɖ/) — independently, not from Sanskrit
- Glottal stop still has no dedicated symbol (written with pangkon)
- tha/dha distinction lost — the preserved feature disappeared
- Aspiration still absent in native words (only in loanwords)

## Interpretation

**Hanacaraka is NOT a "reduced Sanskrit script." It is a JAVANESE phonology encoded in Indic-style notation.** The 13 "lost" phonemes were never native to Javanese — they represent Sanskrit sounds that Old Javanese speakers couldn't distinguish or didn't need.

This is the script-level equivalent of E027's finding: substrate words have a distinctive phonological profile (glottal stops, consonant clusters, no aspiration) that doesn't match Sanskrit phonology. The script itself is a phonological fossil.

## Limitations

1. **Analytical, not empirical** — this is a systematic comparison, not a statistical test. The mapping is well-established in historical linguistics.
2. **tha/dha interpretation speculative** — the "substrate" explanation is one of several possibilities. Could also be: (a) high frequency of Sanskrit loanwords with these sounds, (b) scribal tradition preserving distinctions no longer spoken.
3. **PAn reconstruction debated** — the exact PAn consonant inventory varies by scholar (Blust vs Wolff vs Ross). Used Blust 2009 as standard.
4. **Modern Javanese retroflexes** — whether these are truly independent re-emergence or areal influence from Dravidian/other contact is debated.

## Files

- `00_hanacaraka_phonology.py` — Analysis script
- `results/phonology_summary.json` — Structured findings
- `results/consonant_mapping.csv` — Full 33-consonant mapping table
- `results/hanacaraka_mapping.png` — Visualization (consonant matrix + inventory comparison)

## Cross-Paper Implications

- **P8 (Linguistic Fossils):** Hanacaraka confirms Austronesian phonological core beneath Sanskrit overlay. Connects to E027 substrate fingerprint (glottal stops, no aspiration).
- **P12 (Script Archaeology):** The tha/dha paradox and glottal stop paradox are testable hypotheses for a dedicated script analysis paper.
- **I-053 (Pangram narrative):** Hanacaraka's narrative pangram ("hana caraka...") is unique among SE Asian scripts — worth investigating.

## Conclusion

**SUCCESS.** The 33→20 reduction is systematic: aspiration (8), retroflexes (5), and sibilant distinctions (2) were lost because pre-Sanskrit Javanese didn't have them. The Hanacaraka inventory (20) aligns with PAn (17), not Sanskrit (33). Two paradoxes — dental aspirates retained, glottal stop unwritten — point to pre-Indic phonological features that deserve further investigation in P8 and P12.

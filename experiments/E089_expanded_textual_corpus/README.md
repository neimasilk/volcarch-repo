# E089: Expanded Textual Corpus

**Status:** SUCCESS
**Date:** 2026-03-16
**Layer:** L3 + cross-cutting
**Paper:** P16 (draft)

---

## Hypothesis

A systematically expanded corpus of ancient textual references to Nusantara — with actual passage text, not just summaries — is required for meaningful NLP analysis. E088's 27 references were too thin.

## Method

Systematic mining of ALL known ancient textual traditions referencing Nusantara:
- Added Tamil/Sangam literature (3 refs — gap identified in E088)
- Expanded Chinese dynastic histories (4 → 8 refs)
- Expanded Indian Pali canon (3 → 6 refs)
- Added Sanskrit epics and trade manuals (3 → 4 refs)
- Added Arab geographers (3 → 5 refs)
- Added more chemical/archaeobotanical evidence (4 → 8 refs)
- Added Nusantaran inscriptions for baseline (2 → 5 refs)
- Added linguistic evidence (2 → 4 refs)
- Added Greek sources (2 → 5 refs)

Every entry includes **actual translated passage text** for downstream NLP.

## Results

| Metric | E088 | E089 | Change |
|--------|------|------|--------|
| Total references | 27 | 50 | +85% |
| Traditions | 9 | 10 | +Tamil |
| Entities | 73 | 143 | +96% |
| Pre-400 CE | 18 (67%) | 32 (64%) | Balanced |
| Independence groups | 7 | 8 | +Tamil |
| With passage text | partial | 50/50 (100%) | Full coverage |
| CONSENSUS refs | 11 | 23 (46%) | Higher quality |

## v3 Expansion (Senter v2)

Additional systematic mining:
- Chinese dynastic histories: +15 refs (Liangshu, Songshu, Xin Tangshu, Jiu Tangshu, Zhufanzhi, Yijing, Daoyi Zhilue, Ma Huan, Lingwai Daida, Taiping Yulan)
- European medieval: +6 refs (Marco Polo, Odoric, Conti, Ibn Battuta, Tomé Pires) — **NEW tradition**
- Arab/Persian: +7 refs (Buzurg additional, al-Idrisi, Ibn Khurdadhbih, Abu Zayd, al-Masudi, Hudud al-Alam) — **NEW Persian tradition**
- Indian: +5 refs (Mahaniddesa, Milindapanha, Brhatsamhita, Raghuvamsa, Kathasaritsagara, Arthashastra, Niddesa)
- Tamil: +3 refs (Purananuru, Silappadikaram, Manimekalai)
- Nusantaran inscriptions: +5 refs (Kedukan Bukit, Nalanda copper plate, Laguna, Kota Kapur, Calcutta stone)
- Greek/Roman: +4 refs (Ptolemy additional, Pliny additional, Cosmas, Strabo)
- Chemical: +3 refs (Berenike cloves, Mantai camphor, Uluburun tin)
- Linguistic: +2 refs (Malagasy-Maanyan, Sanskrit loanwords)

| Metric | v2 | v3 | Change |
|--------|------|------|--------|
| Total references | 50 | 106 | +112% |
| Traditions | 10 | 12 | +European, +Persian |
| Entities | 143 | 346 | +142% |
| Pre-400 CE | 32 (64%) | 47 (44%) | More balanced temporally |
| Independence groups | 8 | 14 | +6 new groups |
| CONSENSUS refs | 23 (46%) | 60 (57%) | Higher quality |
| HIGH relevance | 38 (76%) | 84 (79%) | Maintained |

### Key v3 additions for VOLCARCH
- **ARB-012**: Arab eyewitness account of Javanese volcanic eruption ("rivers of mud" = lahars)
- **ARB-008**: al-Idrisi notes "mountains that sometimes emit fire" on Zabaj
- **CHN-019**: Ma Huan's eyewitness of Java volcano ("stones that roll down destroy houses")
- **EUR-006**: Tomé Pires observes half-buried candi in Java (first European)
- **NUS-010**: Srivijaya curse mentions "let the earth shake" (seismic awareness)

## v4 Expansion (Session 4)

Additional systematic mining of underrepresented traditions:
- Chinese: +12 (Sui Shu, Song Shi, Yuan Shi, Zhufanzhi, Daoyi Zhilue, Xingcha Shenglan)
- Arab: +10 (Akhbar al-Sin, Ibn Rustah, al-Maqdisi, Ibn Battuta detail, al-Dimashqi)
- European: +10 (Varthema, Barbosa, Pigafetta, Serrao, Linschoten, de Houtman)
- Nusantaran: +8 (Tanjore inscription, Ligor, Watu Kura, Kakawin Ramayana, Nagarakretagama)
- Indian: +5 (Mudrarakshasa, Divyavadana, Kathasaritsagara, Vayu Purana)
- Persian: +4 (Hudud al-Alam additional, Gardizi, Mustawfi)
- Roman: +4 (Pomponius Mela, Ammianus Marcellinus, Marcus Aurelius embassy, Pliny)
- Tamil: +3 (Pattinappalai, Maduraikkanji, Silappadikaram)

| Metric | v3 | v4 | Change |
|--------|------|------|--------|
| Total references | 106 | 162 | +53% |
| Traditions | 12 | 12 | Deepened, not widened |
| Entities | 346 | 551 | +59% |
| Pre-400 CE | 47 (44%) | 57 (35%) | More post-classical coverage |
| Independence groups | 14 | 15 | +1 |
| CONSENSUS refs | 60 (57%) | 98 (60%) | Higher quality |
| HIGH relevance | 84 (79%) | 134 (83%) | Improved |

### Key v4 additions for VOLCARCH
- **EUR-007**: Varthema (1505) observes "temples half buried in earth" in Java interior
- **EUR-011/012/013**: Pigafetta documents Ternate volcano, Magellan voyage spice trade
- **ARB-014**: Akhbar al-Sin records "fire-breathing islands" in Zabaj seas
- **ARB-019**: al-Dimashqi describes "fire mountains" and "stone temples" in Java
- **PER-004**: Mustawfi notes Java volcanoes and kris metallurgy
- **NUS-014/015/016**: Kakawin Ramayana describes volcanic landscapes as sacred fertility sources

## Output Files

| File | Description |
|------|-------------|
| `results/nusantara_corpus_v2.csv` | v2 corpus (50 refs) |
| `results/nusantara_corpus_v2.json` | v2 corpus with entities |
| `results/nusantara_corpus_v3.json` | v3 corpus (106 refs) |
| `results/nusantara_corpus_v3.csv` | v3 corpus tabular |
| `results/nusantara_corpus_v4.json` | **v4 corpus (162 refs)** |
| `results/nusantara_corpus_v4.csv` | v4 corpus tabular |
| `results/passages_for_nlp_v4.json` | v4 passages for E090 |
| `results/e089_v4_summary.json` | v4 statistics |
| `results/passages_for_nlp_v3.json` | v3 passages (legacy) |
| `results/e089_v3_summary.json` | v3 statistics |
| `results/passages_for_nlp.json` | v2 passages (legacy) |
| `results/e089_summary.json` | v2 statistics |

## Conclusion

**SUCCESS.** Corpus tripled from original (50 → 162) with higher quality and broader coverage. 12 traditions, 15 independence groups, 551 entities. 12 VOLCARCH-relevant entries with direct volcanic/burial references. BERTopic minimum (200) not yet met — need 38 more entries for optimal topic modeling. E090 can run EXP 1/2/5 on v4 corpus.

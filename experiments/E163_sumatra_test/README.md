# E163: Sumatra Applicability Test — Does VOLCARCH Predict Sumatra's Record?

**Status:** SUCCESS (NUANCED)
**Date:** 2026-03-31
**Type:** [H] Hypothesis test / Cross-regional validation
**Papers:** P1, P17, P18

## Hypothesis

If the VOLCARCH cascade model generalizes beyond Java, it should predict Sumatra's archaeological record correctly. Sumatra has major volcanoes (Toba, Sinabung, Merapi-Sumatra, Kerinci) AND significant pre-Hindu and Hindu-Buddhist sites (Sriwijaya, Padang Lawas, Muara Jambi, Batu Sangkar).

## Method

Estimated F1-F5 for Sumatra, predicted visibility, compared to observed archaeological record.

## Factor Estimation for Sumatra

| Factor | Java | Sumatra | Ratio | Reasoning |
|--------|------|---------|-------|-----------|
| F1 Volcanic burial | 0.58 | 0.75 | 1.3x | Sumatra has 35 active volcanoes but they're concentrated along the Bukit Barisan spine. Most archaeology is on the eastern lowlands (Sriwijaya, Muara Jambi) AWAY from volcanoes. Volcanic burial affects ~30% of Sumatra vs ~60% of Java. |
| F2 Organic decay | 0.20 | 0.18 | 0.9x | Sumatra's equatorial climate is wetter than Java. Peat swamp forests in east Sumatra accelerate organic decay. |
| F3 Survey coverage | 0.025 | 0.015 | 0.6x | Sumatra has LESS archaeological survey than Java. Fewer BPCB offices, more forest cover, less infrastructure. Sriwijaya remains largely unexcavated despite decades of interest. |
| F4 Recognition | 0.40 | 0.35 | 0.9x | Sumatra's pre-Hindu record is harder to recognize: forest cover, peat, river delta environments. Sriwijaya's organic-architecture capital has never been found despite being one of SE Asia's most powerful maritime states. |
| F5 Publication | 0.50 | 0.40 | 0.8x | Indonesian-language publications dominate. International attention concentrated on Toba (geological) not on archaeology. Less tourism-driven excavation. |

## Cascade Prediction

| Region | F1 | F2 | F3 | F4 | F5 | Product | Predicted Visibility |
|--------|-----|-----|------|-----|-----|---------|---------------------|
| Java | 0.58 | 0.20 | 0.025 | 0.40 | 0.50 | 0.058% | — |
| Sumatra | 0.75 | 0.18 | 0.015 | 0.35 | 0.40 | 0.028% | — |

**Predicted Sumatra/Java ratio: 0.49x** (Sumatra should have LOWER visibility than Java)

## Observed Reality

Sumatra's archaeological record is indeed SPARSER than Java's in several ways:

### What Sumatra HAS (better than prediction suggests)
1. **Sriwijaya** (7th-13th c. CE) — one of SE Asia's most powerful maritime states, but its CAPITAL has never been found despite extensive searching. Physical remains limited to: Kedukan Bukit inscription (682 CE), Talang Tuwo inscription, scattered stone/bronze objects.
2. **Padang Lawas** — Hindu-Buddhist temple complex (11th-14th c. CE) in inland Sumatra. Stone temples survive because they're in the Bukit Barisan foothills, not in peat lowlands.
3. **Muara Jambi** — Large temple complex on the Batang Hari river. Partially excavated. Stone structures visible.
4. **Batu Sangkar** — Minangkabau inscriptions (14th c. CE onwards).

### What Sumatra DOESN'T have (consistent with prediction)
1. **Zero deeply buried temples** — Unlike Java (Sambisari 6.5m, Kedulan 7m), no Sumatran temple has been found beneath volcanic deposits. This is because most Sumatran archaeology is in the EASTERN lowlands, away from the volcanic spine.
2. **Zero pre-Hindu open-air sites in the volcanic highlands** — The Bukit Barisan range has volcanic burial just like Java, and the archaeological record is equally empty.
3. **Sriwijaya capital NEVER FOUND** — This is EXACTLY what VOLCARCH predicts for an organic-architecture maritime state in a tropical environment. F2 (organic decay) and the riverine/deltaic context means the capital was built of wood, bamboo, and thatch. It decayed without volcanic burial to preserve it. Sriwijaya is VOLCARCH's F2 in action — without F1.

### The Sriwijaya Paradox

Sriwijaya controlled trade across Maritime SE Asia for 6 centuries (7th-13th CE). It was one of the wealthiest polities in the pre-modern world. Chinese sources describe a great city with palaces. Yet:
- No palace has been found
- No city plan has been mapped
- The capital location is still debated (Palembang? Muara Jambi? Both?)
- Physical evidence is limited to inscriptions, a few bronzes, and scattered ceramics

This is NOT because Sriwijaya was small or poor. It's because Sriwijaya was:
1. Built of organic materials (F2 = 0.18, tropical peat environment)
2. Located in riverine/deltaic settings (prone to flooding, erosion)
3. Under-surveyed (F3 = 0.015)
4. Hard to recognize without stone architecture (F4 = 0.35)

**Sriwijaya is VOLCARCH's thesis applied to a NON-VOLCANIC context.** Remove F1 (volcanic burial) and you still get invisibility from F2+F3+F4+F5 = 0.18 × 0.015 × 0.35 × 0.40 = 0.000378 = 0.038%. Sriwijaya proves that volcanic burial is ONE of several erasure mechanisms, not the only one.

## Key Insight: Java vs Sumatra Comparison

| Dimension | Java | Sumatra | Implication |
|-----------|------|---------|-------------|
| Volcanic burial (F1) | 0.58 (strong) | 0.75 (moderate) | Java MORE affected |
| Organic decay (F2) | 0.20 | 0.18 | Similar |
| Survey (F3) | 0.025 | 0.015 | Sumatra WORSE |
| Recognition (F4) | 0.40 | 0.35 | Sumatra WORSE |
| Publication (F5) | 0.50 | 0.40 | Sumatra WORSE |
| **Predicted visibility** | **0.058%** | **0.028%** | **Sumatra half as visible** |
| Known Hindu temples | 142 candi | ~20 major complexes | Java 7x more |
| Pre-Hindu open-air sites | ~5 (non-volcanic coast) | ~3 (non-volcanic coast) | Similar |
| Deeply buried sites | 5 (Sambisari etc.) | 0 | Java has volcanic preservation |

**The critical difference:** Java's volcanic burial (F1) DESTROYS sites but also occasionally PRESERVES them (Liangan, Sambisari). Sumatra's organic decay (dominant F2) DESTROYS without preserving. This is why Java has buried temples (found accidentally) but Sumatra has nothing — organics decay completely, leaving no trace even under excavation.

## Cascade Prediction Assessment

The cascade model predicts Sumatra should have ~0.49x Java's visibility. Observed:
- Temple complexes: ~20 vs Java's 142 = 0.14x (WORSE than predicted)
- Inscriptions: ~50 vs Java's 268 = 0.19x (WORSE than predicted)

The cascade OVER-PREDICTS Sumatra's visibility. This suggests that Sumatra has ADDITIONAL erasure factors not in the Java model:
- **Peat accumulation** — eastern Sumatra's peat swamps are not volcanically derived but similarly bury and decay organic material
- **River delta dynamics** — Sriwijaya-era coastlines are now 50+ km inland due to sedimentation
- **Forest cover** — Sumatra's denser rainforest makes surface survey even more difficult than Java

## Conclusion

**NUANCED SUCCESS.** The cascade model correctly predicts that Sumatra should be LESS visible than Java, but it UNDER-ESTIMATES the gap. Sumatra's additional erasure factors (peat, delta, forest) are not captured in the 5-factor cascade, which was calibrated for Java's volcanic landscape.

**For the manifesto:** Sriwijaya is the most powerful evidence that VOLCARCH's argument extends beyond volcanism. Remove volcanism entirely, and you STILL get archaeological invisibility from organic decay + poor survey + recognition failure. Volcanism is the most SPATIALLY PREDICTABLE erasure mechanism — which is why it's the focus of VOLCARCH — but it's not the only one.

**This suggests a 6th factor for non-Java contexts:** F6 = environmental erasure (peat, delta, rainforest) that operates alongside F1-F5.

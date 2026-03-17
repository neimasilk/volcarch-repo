# E111: Script Diffusion Timeline — Is Java's 650-Year Gap Anomalous?

**Date:** 2026-03-17
**Paper:** P18 (Invisible Civilization)
**Status:** SUCCESS — Java's lag is NORMAL (57th percentile)

## Hypothesis

The 3,500-year gap between Sumerian writing (3100 BCE) and Javanese inscriptions (400 CE) suggests civilizational "lateness." But the relevant comparison is not invention but ADOPTION: how fast does writing spread from source to recipient?

## Method

Compiled 23 global writing diffusion events (source → recipient with adoption lag). Computed Java's percentile rank. Also modeled organic writing survival probability in tropical conditions.

## Results

- **Java's lag (660 yr from Brahmi):** 57th percentile globally. NORMAL.
- **Brahmi-derived mean:** 508 years. Java is 152 years slower — within 1 SD.
- **Comparison:** Myanmar 760 yr, Thailand 810 yr, Sriwijaya 943 yr — all slower than Java.
- **Organic survival model:** Lontar P(survive 1626 yr) = 3.6×10⁻³. Bamboo = 1.6×10⁻¹⁰.

## Key Finding: The Three Gaps

1. **Gap 1 (Sumeria→Java, 3500 yr):** MISLEADING — invention vs adoption
2. **Gap 2 (Brahmi→Java, 660 yr):** NORMAL — 57th percentile
3. **Gap 3 (organic writing):** Possibly 0 — organic media don't survive in tropics

## Files

| File | Description |
|---|---|
| `script_diffusion.py` | Analysis script |
| `results/e111_results.json` | Full results |

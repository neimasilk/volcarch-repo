# E093 x E070 Cross-Reference Report

**Date:** 2026-03-17
**Method:** Programmatic site-name matching (direct, alias, fuzzy) + regional/volcanic system overlap + depth data classification

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| E070 entries | 52 (colonial register) |
| E093 entries | 65 (literature database) |
| Site-level matches | 5 (across 5 publications) |
| Regional/volcanic overlaps | 22 publications |
| Publications with burial depth data | 30 |
| Already in E070 (OV sources) | 3 |
| **Potentially new depth data** | **27 publications** |

---

## Key Findings

### 1. Trowulan Cluster (4 publications)

Trowulan/Majapahit is the strongest intersection point:
- **Satari 1977** — Early excavation, stratigraphic depths
- **Soekatno 1985** — Stratigraphy in Arjuno-Welirang volcanic zone
- **Pojoh 2007** — Only published GPR survey in volcanic Java
- **Wibisono 2014** — Updated excavation context

E070 has 5 Trowulan entries (depths: 0.75-4.28m). These 4 publications may contain **additional depth measurements** from different excavation trenches/sectors.

### 2. High-Priority New Depth Sources (not in E070)

| Publication | Site/Zone | Why Important |
|-------------|-----------|---------------|
| **Lukas 2012** | Kimpulan (~2m, Merapi) | Post-colonial discovery. Independent depth. |
| **Rangkuti 2008** | Lumajang (Semeru) | Zone B survey. New zone for E070. |
| **Rangkuti 2000** | Arjuno slopes | Deep burial zone. Not in colonial register. |
| **Abbas 2012** | Malang (multiple) | Modern survey overlapping OV 1926 zone. |
| **Susetyo 2013** | Kumitir (Arjuno-Welirang) | Near Trowulan, recent excavation. |
| **Inagurasi 2010** | Kediri (Kelud zone) | Overlaps with E070's Kelud entries. |
| **Mulyaningsih 2006** | Merapi deposits | Geological method = independent validation. |

### 3. Volcanic System Coverage

| Volcanic System | E070 sites | E093 publications | Gap |
|-----------------|-----------|-------------------|-----|
| Merapi | 7 | 10 | Good overlap. Extract more depth data from volcanology papers. |
| Kelud | 8 | 4 | OV well-covered. Modern surveys (Inagurasi 2010) add new data. |
| Arjuno-Welirang | 11 | 4 | Trowulan dominant. Rangkuti 2000 adds slope sites. |
| Semeru | 2 | 1 | **UNDERREPRESENTED.** Rangkuti 2008 is only modern survey. |
| Sindoro/Sumbing | 1 | 3 | Liyangan well-documented. Multiple publications to mine. |
| Dieng | 2 | 0 | **GAP.** No E093 publications with depth data for Dieng. |

### 4. Validation Chain Potential

```
E093 literature (27 new depth pubs)
  --> Extract 15-25 new depth measurements
    --> Expand E070 register from 52 to ~70+ entries
      --> Recalibrate E075 sedimentation model
        --> Strengthen P1 revision evidence
```

---

## Actionable Recommendations

### Tier 1: Immediate (open access, extractable now)
1. **Rangkuti 2008** (Berkala Arkeologi) — Lumajang/Semeru depths
2. **Rangkuti 2000** (Berkala Arkeologi) — Arjuno slope depths
3. **Lukas 2012** (Archipel) — Kimpulan 2m depth measurement
4. **Riyanto 2014** (Amerta) — Liyangan full stratigraphy

### Tier 2: Require careful extraction
5. **Newhall 2000** (JVGR) — Archaeological layer depths in Merapi stratigraphic columns
6. **Thouret 2000** (BV) — Kelud lahar deposit thicknesses at known distances
7. **Mulyaningsih 2006** (IJG) — Deposit thickness maps over Mataram territory

### Tier 3: Supplementary
8. **Abbas 2012** (Berkala Arkeologi) — Modern Malang survey depths
9. **Susetyo 2013** (Amerta) — Kumitir excavation depths
10. **Pojoh 2007** (BIPPA) — GPR penetration depth in andosols

---

## Script Location
`experiments/E093_indonesian_lit_mining/cross_reference_e070.py`

## JSON Output
`experiments/E093_indonesian_lit_mining/results/cross_reference_e070.json`

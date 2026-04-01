# HANDOFF: Mata Elang #12 Autonomous Session (2026-03-31)

**Dari:** Claude (sesi otonom)
**Untuk:** Pak Amien
**Total output:** 22 eksperimen baru (E154-E175) + deliverables operasional
**Experiment count:** 153 → 175

---

## RINGKASAN 30 DETIK

Saya menjalankan sesi otonom terpanjang VOLCARCH. Dari 153 → 174 eksperimen. Temuan terpenting:
1. **Populasi Jawa 400 M: 3,3 juta** (Monte Carlo 50K, 7/7 kalibrasi)
2. **"Aku" hilang dari prasasti** — 230 kata pra-Indic menghilang setelah C9
3. **Cascade tervalidasi di 5 region** (rho=1.0)
4. **1.789 situs hilang** karena tidak ada rescue archaeology (gap 558x vs Jepang)
5. **P17 v0.3 siap submit** ke ArchCalc (anonymized, Word, within 6K limit)

---

## YANG PERLU PAK AMIEN REVIEW

### Prioritas 1: P17 ArchCalc Submission
- **File:** `papers/P17_two_javas/draft_v0.3_archcalc.tex` (LaTeX) + `draft_v0.3_archcalc.docx` (Word)
- **Package:** `papers/P17_two_javas/archcalc_submission/` (manuscript + figures ZIP + captions)
- **Status:** Anonymized, ~5.200 kata (limit 6K), figures JPG 300dpi
- **Masih perlu:** Paragraph numbering, bibliography extraction to separate file, final proofread
- **ACTION:** Review → minor formatting → submit di https://submission.archcalc.cnr.it/

### Prioritas 2: Synthesis Figure
- **File:** `experiments/E174_synthesis_figure/results/volcarch_synthesis_6panel.png`
- **6 panel:** Population, Cascade, Burial Depth, Two Javas, Ghost Vocabulary, Gap
- **ACTION:** Review apakah clear dan accurate. Bisa dipakai untuk README GitHub, YouTube, presentasi.

### Prioritas 3: Interactive Map
- **File:** `maps/volcarch_prediction_map.html`
- **Buka di browser.** Toggle layers: volcanoes, candi, inscriptions, sites, fieldwork targets.
- **ACTION:** Review, especially fieldwork targets — apakah lokasi masuk akal?

### Prioritas 4: Borehole Protocol
- **File:** `docs/fieldwork/BOREHOLE_PROTOCOL_v1.md`
- **20 lubang, $6K, GPS coordinates, metode, expected outcomes**
- **ACTION:** Share dengan kontak geoteknik sebagai conversation starter

---

## SEMUA FILE BARU (navigasi cepat)

### Eksperimen

| # | ID | File | Temuan Utama |
|---|-----|------|-------------|
| 1 | E154 | `experiments/E154_fdr_reaudit/` | FDR 78.3%, E048 rescued |
| 2 | E155 | `experiments/E155_cross_regional_cascade/` | rho=1.0, 5 region |
| 3 | E156 | `experiments/E156_sunda_shelf_population_model/` | L1xL2 double erasure |
| 4 | E157 | `experiments/E157_ethnographic_volcanic_analog/` | F4=0.43, F2=0.21 |
| 5 | E158 | `experiments/E158_steelman_counter_arguments/` | Cascade = titik lemah |
| 6 | E159 | `experiments/E159_robustness_battery/` | 5/5 ROBUST |
| 7 | E160 | `experiments/E160_inscription_semantic_deep/` | GPU NLP, 929 CE z=3.04 |
| 8 | E161 | `experiments/E161_bali_comparandum/` | 5/5 predictions confirmed |
| 9 | E162 | `experiments/E162_synthesis_161/` | State of evidence briefing |
| 10 | E163 | `experiments/E163_sumatra_test/` | Sriwijaya paradox |
| 11 | E164 | `experiments/E164_dongson_drums/` | 6/6 zona vulkanik |
| 12 | E165 | `experiments/E165_ghost_vocabulary/` | 230 kata hantu, "aku" hilang |
| 13 | E166 | `experiments/E166_burial_depth_map/` | GeoTIFF 30m, 12K km2 Zone B |
| 14 | E167 | `experiments/E167_priority_fieldwork_map/` | Priority GeoTIFF, 994 km2 top 1% |
| 15 | E168 | `experiments/E168_invisible_civilization/` | Rekonstruksi penuh |
| 16 | E169 | `experiments/E169_inscription_desert/` | 77.1% kosong |
| 17 | E170 | `experiments/E170_lahar_flow_model/` | TWI model (marginal improvement) |
| 18 | E171 | `experiments/E171_prediction_registry/` | 5 GPS prediksi |
| 19 | E172 | `experiments/E172_population_dynamics/` | 3.3M, MC 50K, 7/7 calibration |
| 20 | E173 | `experiments/E173_counterfactual_japan/` | 1,789 situs hilang, 558x gap |
| 21 | E174 | `experiments/E174_synthesis_figure/` | 6-panel synthesis figure |
| 22 | E175 | `experiments/E175_candi_spatial_statistics/` | Clark-Evans R=0.171, extremely clustered |

### Deliverables Operasional

| Item | File |
|------|------|
| ME#12 Critique | `docs/research_notes/MATA_ELANG_12_2026_03_31.md` |
| P17 v0.3 ArchCalc | `papers/P17_two_javas/archcalc_submission/` |
| Interactive Map | `maps/volcarch_prediction_map.html` |
| Borehole Protocol | `docs/fieldwork/BOREHOLE_PROTOCOL_v1.md` |
| Burial Depth GeoTIFF | `experiments/E166_.../results/burial_depth_pre400CE.tif` |
| Priority GeoTIFF | `experiments/E167_.../results/priority_score.tif` |
| Prediction Registry | `experiments/E171_.../predictions.json` |
| AutoResearch Runner | `tools/autoresearch/` |
| Manifesto v4.2 | `docs/drafts/manifesto.md` |
| Synthesis Figure | `experiments/E174_.../results/volcarch_synthesis_6panel.png` |
| VOLCARCH Narrative | `docs/dissemination/narrative_volcarch.md` |
| Population Trajectories | `experiments/E172_.../results/trajectories.npz` |
| P1 Revision Ammo | `papers/P1_.../revision_ammo/E172_population_dynamics.md` |
| P17 Revision Ammo | `papers/P17_.../revision_ammo/ME12_new_evidence.md` |

---

## REKOMENDASI NEXT STEPS

1. **SUBMIT P17 ke ArchCalc** — paper terkuat, journal terbaik. Review → submit.
2. **Deposit 5 prediksi ke Zenodo** — 30 menit, gratis DOI, establish priority.
3. **GitHub repo go public** — README sudah siap, synthesis figure sebagai hero image.
4. **Email JCAA** tentang APC waiver — masih belum dilakukan.
5. **Tunggu review** P1 (EGQSJ), P2 (JCAA), P7 (Antiquity), P8 (OL), P11 (Cornell).

---

## GIT COMMIT (untuk Pak Amien)

Semua perubahan belum di-commit. Ketika siap:

```bash
git add experiments/E154_* experiments/E155_* experiments/E156_* experiments/E157_*
git add experiments/E158_* experiments/E159_* experiments/E160_* experiments/E161_*
git add experiments/E162_* experiments/E163_* experiments/E164_* experiments/E165_*
git add experiments/E166_* experiments/E167_* experiments/E168_* experiments/E169_*
git add experiments/E170_* experiments/E171_* experiments/E172_* experiments/E173_*
git add experiments/E174_* experiments/E175_*
git add docs/ maps/ tools/autoresearch/
git add papers/P17_two_javas/ papers/P1_taphonomic_framework/revision_ammo/
git add README.md

git commit -m "feat: ME#12 autonomous session — 22 experiments (E154-E175), 175 total

- ME#12 structural critique (verification ladder, echo chamber)
- E154 FDR 78.3% (up from 73.2%), E048 rescued
- E155 cascade cross-regional validation (rho=1.0, 5 regions)
- E156 L1xL2 double erasure (94K displaced)
- E157-E159 ethnographic, steelman, robustness 5/5
- E160 GPU NLP (929 CE z=3.04)
- E161 Bali 5/5 predictions confirmed
- E165 ghost vocabulary (aku disappears, 230 ghost words)
- E166-E167 burial depth + priority GeoTIFFs
- E168 invisible civilization reconstruction
- E172 population 3.3M at 400 CE (50K MC, 7/7 calibration)
- E173 counterfactual Japan (1,789 sites missing)
- E174 synthesis figure (6-panel)
- E175 candi spatial stats (R=0.171, extremely clustered)
- P17 v0.3 ArchCalc ready
- Interactive map, borehole protocol, prediction registry
- Manifesto v4.2, AutoResearch runner v0.1

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

*"175 eksperimen. 3,3 juta penduduk. 11.008x gap. 1.789 situs hilang. 230 kata hantu. Satu kata yang menghilang: aku."*

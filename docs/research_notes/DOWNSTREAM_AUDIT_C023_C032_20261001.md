# Audit hilir C023/C032 — 11 eksperimen yang mewarisi cacat (2026-10-01 malam)

**Asal:** workflow `downstream-audit-c023-c032`, dengan 3 pembaca Sonnet yang masing-masing memegang satu
kelompok eksperimen. Run pertama dihentikan saat jatah habis dan dilanjutkan setelah jatah pulih.

**Definisi cacat:**
- D1 = tanda E069 terbalik (C023);
- D2 = artefak geocoding E082 (C032/C034);
- D3 = inventori situs yang belum divalidasi (C029);
- D4 = bingkai E129 (C036);
- D5 = sel laut (C035).

**Verifikasi orkestrator** (diturunkan ulang dari file):
- **E109 (D5):** dari 592 sel gabungan E075×E069, **210 sel <10% darat dengan 0 situs** (199 tanpa darat
  sama sekali), memakai aturan DEM>0 yang sama dengan cek darat E069.
- **E159:** JSON hasilnya sendiri berbunyi E069 ROBUST, E031 ROBUST, **E051 FRAGILE**, E084 ROBUST, **E065
  ERROR**. Jadi "5/5 ROBUST" di README salah oleh keluarannya sendiri.
- **E154:** `fdr_reaudit.py` baris 26 berisi "Other survivors (estimated from typical VOLCARCH results)",
  yaitu nilai p yang diketik tangan.
- **E080:** skor maksimum 0,855 dimiliki **29 sel** dari 4.600, jadi "20 target teratas" hanya urutan pindai.
- **Paparan kiriman:**
  - Naskah P2 v0.2 dan surat jawaban reviewer yang terkirim ke JCAA **tidak** memuat E069/E109; yang memuatnya
    hanya `revision_ammo`.
  - **Docx P17 yang terkirim memuat** hasil relokasi 929 M (E105) dan kalimat ketahanan (E159). Draf surat
    integritas P17 sudah ditambah satu kalimat.

**Hal lain** adalah laporan agen. Bukti `file:line` dan replikasinya ada di bawah. Belum semua diturunkan
ulang orkestrator; yang sudah tercantum di atas.

| Eksperimen | Vonis | D1 | D2 | D3 | D4 | D5 |
|---|---|---|---|---|---|---|

| E109 | AFFECTED_HEADLINE | PARTIAL | NONE | HEADLINE | NONE | HEADLINE |
| E120 | AFFECTED_MINOR | PARTIAL | NONE | NONE | NONE | NONE |
| E136 | AFFECTED_HEADLINE | HEADLINE | NONE | PARTIAL | HEADLINE | PARTIAL |
| E154 | AFFECTED_HEADLINE | PARTIAL | HEADLINE | PARTIAL | PARTIAL | PARTIAL |
| E159 | AFFECTED_HEADLINE | PARTIAL | HEADLINE | HEADLINE | NONE | PARTIAL |
| E073 | AFFECTED_HEADLINE | PARTIAL | NONE | HEADLINE | NONE | PARTIAL |
| E085 | AFFECTED_MINOR | PARTIAL | NONE | NONE | NONE | NONE |
| E080 | AFFECTED_HEADLINE | PARTIAL | NONE | HEADLINE | NONE | PARTIAL |
| E105 | AFFECTED_HEADLINE | NONE | HEADLINE | NONE | NONE | NONE |
| E158 | AFFECTED_HEADLINE | HEADLINE | HEADLINE | PARTIAL | PARTIAL | PARTIAL |
| E098 | AFFECTED_MINOR | PARTIAL | NONE | NONE | NONE | PARTIAL |

---


## E109 — Forward Simulation — Archaeological Record Under Burial Hypothesis

**Vonis:** AFFECTED_HEADLINE  
**Status README sebelum koreksi:** Status MIXED (README:5); no correction note present; docs/experiment_index.json:1414 carries MIXED.

**Klaim utama:** README.md:60 '**Estimated hidden: 824 sites (detection rate 30.2%)**' (README.md:58-61: observed 357, estimated total 1,181, 22.5% of cells >200 cm); README.md:24,34 'SURPRISE: Site Density INCREASES with Burial Depth ... (rho = 1.0)'; README.md:73 'The 824 estimated hidden sites ... are hidden by SURVEY ACCESS, not burial depth, in this model. E069's nested approach is needed to isolate the burial effect.' Status README.md:5 'MIXED'.

**Masukan yang benar-benar dibaca skrip:** forward_simulation.py:39-41 E075 results/burial_grid_sample.csv (2,838 cells; Pyle-model depth, a model output, C026); :44-46 E069 adv3_survey_intensity/results/adv3_cell_data.csv (703 rectangular 0.1-degree cells: site_count, road_dist, volcano_dist); :49 dashboard/sites.csv (loaded, only len() printed at :89). Inner merge on rounded lat/lon (:92-98) = 592 cells; total_observed = sum of E069 site_count (:117-118) = 357, which E069 counts from data/processed/east_java_sites.geojson (adv3_survey_intensity.py:54,127-134) on a grid with no land mask (:258 comment 'valid = has road distance = on land'). MLE :247-254, hidden :306-310, Japan scenario :363-365. A read-only replication in the scratchpad (nothing written to the repo) reproduces 592 cells, 357 sites, lambda0 1.99478, tau inf, rho 181, total 1,181, hidden 824 and 22.5% exactly.

**Ketergantungan pada cacat:**

- **D1 — PARTIAL:** The script never uses E069's coefficient, but README.md:54 'E069 (ADV-3) resolved this confound using nested model comparison and found p=0.0015 for volcanic proximity AFTER controlling survey' and :73 'E069's nested approach is needed to isolate the burial effect' rest on the inverted reading (C023: beta on distance <0 = surplus near volcanoes). E109's own direction (density rises with modelled depth) is the same surplus signal, so its direction agrees with the corrected E069; only the cross-reference is wrong. papers/P2_settlement_model/revision_ammo/E109_survey_confound.md:15,25 repeats the inverted claim as reviewer-response text.
- **D2 — NONE:** No inscription, E082, E062 or geocoded file is read anywhere in forward_simulation.py (:39-49).
- **D3 — HEADLINE:** The 357 observed sites are E069's site_count from the unvalidated geojson. My recount reproduces E069's per-cell counts exactly (375 in grid, 357 in merged cells); of the 375: 154 temple-named, 121 type 'monument', 83 modern-name hits ('Tugu Bambu Runcing', 'Monumen Perjuangan Polri', 'Patung Karapan Sapi'). Dropping modern-name plus monument/kuil-typed features: observed 357 -> 245, estimated hidden 824 -> 457 (total 1,181 -> 702); modern-name only: observed 284, hidden 544. The direction survives (Q1..Q4 densities 0.020, 0.162, 0.088, 1.385).
- **D4 — NONE:** No E129 figure is used. Composition note only: 269 of 375 in-grid features are temple-named or monument/kuil typed (71.7%), the same temple-heavy frame as C036, so 'hidden sites' means hidden recorded monuments, not settlements.
- **D5 — HEADLINE:** Estimated total = lambda0 x number of cells (:294, :306-310), so sea cells count as hidden sites. 210 of the 592 merged cells have <10% land (199 have none; DEM>0 rule of E069 direction_check_land_20261001.py) and hold 0 of the 357 sites, yet under the published parameters they carry 413.9 of the 824.2 hidden sites (50.2%). Land-only refit (>=10% land, 382 cells): lambda0 1.869, rho 232 m, total 714, hidden 357 (detection 50.0%); >=50% land (344 cells): hidden 271. Japan-level 846 (2.4x) -> 667 (1.9x); '>200 cm' share 22.5% -> 33.0%. Quartile direction unchanged (0.156, 0.389, 0.768, 2.417). Land mask plus cleaned inventory together: hidden 181-217.

**Dikutip di:**

- papers/P2_settlement_model/revision_ammo/E109_survey_confound.md:11,15,19,25 (824 hidden / 30.2%; 'E069 ... isolates a residual volcanic effect'; ready-to-paste JCAA reviewer text)
- docs/research_notes/OBJECTIVE_ANSWER_20261001.md:231,247,333 (already treats E109 as non-diagnostic and drops it from 'against H1'; consistent)
- docs/research_notes/CRITIQUE_SYSTEM_DESIGN_20260811.md:71 (E109/E086 cited for survey-deficit leverage)
- docs/drafts/T2_DEEP_CHANCE_FINDS_DESIGN_DRAFT_20261001.md:27 (E069, E109 named as survey-biased site-count analyses; consistent)
- docs/experiment_index.json:1414 (status MIXED)

**Koreksi yang disarankan:** Set README status to INFO NEG / NON-DIAGNOSTIC and add a correction block: strike '824 hidden / 30.2% / 1,181 total / Japan 846 (2.4x)' because on land cells the same script gives 357 hidden (50.0%, total 714, Japan 667) and on a cleaned inventory 181-457, with 413.9 of the 824 in cells at least 90% sea; keep only the direction (density rises with modelled depth, robust to D3/D5, same signal as the corrected E069), delete README:54/73 saying E069 'resolved'/'isolates' a burial effect (C023), and note that trend_p = 0.0 (e109_results.json:8) is invalid for n=4 quartile points (minimum exact p is 0.083). Rewrite or withdraw papers/P2_settlement_model/revision_ammo/E109_survey_confound.md lines 11-25 (never paste into the JCAA response), add the 357/181-457 figures to the E109 row of OBJECTIVE_ANSWER_20261001.md:231, and update the docs/experiment_index.json status.


## E120 — Cascade Stress Test — Systematic Adversarial Probing

**Vonis:** AFFECTED_MINOR  
**Status README sebelum koreksi:** Status SUCCESS (README:4); no correction note present; docs/experiment_index.json:1572 carries SUCCESS.

**Klaim utama:** README.md:65 '**F3 (survey coverage) is the ONLY structurally necessary factor.** Remove any other single factor and the model still holds within 10x of observed. Remove F3 and the model overshoots by 75x.' README.md:67 'the archaeological gap is primarily a survey deficit, not a burial effect. VOLCARCH's contribution is not that volcanism IS the main cause — it's that volcanism makes the gap **spatially predictable**'. Status README.md:4 'SUCCESS'; README.md:71 'The cascade is robust under systematic adversarial probing.'

**Masukan yang benar-benar dibaca skrip:** No data file is read: cascade_stress_test.py:15-19 imports only numpy/json/csv/itertools/pathlib; the only file operations are writes (:91, :362). All inputs are hard-coded: FACTORS :26-32 (best 0.58/0.20/0.025/0.40/0.50 'from E110'), OBSERVED_VISIBILITY = 0.00031059 (:34, '0.031% from E108'), DEMOGRAPHIC_GAP = 3220 (:35). Provenance: E110 visibility_cascade.py:57-154 (hand-set values; F1 0.58 = 0.40*0.10+0.60*0.90 at :73; F3 0.025 from an assumed '~5% of Java surveyed ... ~50% post-colonial' at :108-111; evidence tags cite E075/E083/E069/E086 at :61-66,:103-106) and 0.031% = 3/9,659 (visibility_cascade.py:178-181, 'observed_sites = 3  # generous'; E108 is an assumption model whose script reads no data). Every E120 output is deterministic arithmetic on those constants.

**Ketergantungan pada cacat:**

- **D1 — PARTIAL:** README.md:60 'it requires survey coverage = 10%, which contradicts ADV-3 (E069, p=0.0015)' (p=0.0015 is the volcano-distance LR test with inverted sign, C023, and says nothing about survey coverage); README.md:71 'independently supported by ADV-3 (E069) and the Japan comparison (E086)' is circular because E110 chose F3 citing E069 and E086 (visibility_cascade.py:103-105); README.md:67 'volcanism makes the gap spatially predictable' leans on the inverted E069/E109; README.md:82 'Supports: E086, E069, E109'. The arithmetic itself (F3 window 0.119-0.133, 74.7x removal overshoot, 35.5x all-high) is unaffected.
- **D2 — NONE:** No inscription or geocoded data anywhere in cascade_stress_test.py or E110's factor definitions.
- **D3 — NONE:** F3 = 0.025 is hand-set from an assumed 5% coverage (visibility_cascade.py:108-111), not computed from the site inventory; E069's road-distance coefficient is only cited as evidence (:105), never used numerically.
- **D4 — NONE:** No E129 figure is cited or used in README or script (grep E129 empty).
- **D5 — NONE:** No grid or raster. E110's F1 evidence bullet 'E075: 32.3% of East Java >1m burial' (visibility_cascade.py:63) is a sea-inclusive grid share (C035), but F1 = 0.58 is hand-weighted (:73) and E120's own removal test holds without F1 (3.2x, stress_test_summary.json:44-49), so no E120 conclusion depends on it.

**Dikutip di:**

- docs/research_notes/MATA_ELANG_11_2026_03_30.md:143 (L1 'DIDUKUNG KUAT' lists 'E120 (F3 structurally necessary)')
- docs/research_notes/MATA_ELANG_10_2026_03_30.md:47 ('F3 (survey) satu-satunya faktor structurally necessary. Model robust.')
- docs/research_notes/MATA_ELANG_12_2026_03_31.md:25,60 (E120 as internal-consistency robustness 'DONE')
- docs/research_notes/OBJECTIVE_ANSWER_20261001.md:334 (cascade E110/E120 already excluded for H2)
- papers/P0_invisible_civilization/SKELETON_v0.1.md:365 (claim restated without ID: 'identifying survey coverage (40x leverage) as dominant'; this is E110's leverage)
- docs/experiment_index.json:1572 (status SUCCESS)

**Koreksi yang disarankan:** Re-label status SUCCESS to INFO (arithmetic on E110's hand-set factors; no data read, so it neither tests nor supports the thesis) and add a correction note deleting 'contradicts ADV-3 (E069, p=0.0015)', 'independently supported by ADV-3/E086' and 'spatially predictable' support (README:60, 67, 71, 82): that p is the volcano-distance test with inverted sign (C023), is not about survey coverage, and the support is circular since E110 picked F3 = 0.025 citing E069 and E086. The isolation and removal numbers stand only as arithmetic on E110's inputs (the 'observed' 0.031% is 3/9,659, with '3' set 'generous'); papers/P0_invisible_civilization/SKELETON_v0.1.md:365 may keep its 'diagnostic' wording but must not cite E120/E069 as support, and docs/experiment_index.json:1572 should follow.


## E136 — Bayesian Integration of All VOLCARCH Evidence

**Vonis:** AFFECTED_HEADLINE  
**Status README sebelum koreksi:** Status SUCCESS (ILLUSTRATIVE) (README:4); an illustrative-only caveat exists (README:44-58) but there is no note on the E069 or E129 lines; docs/experiment_index.json:1842 carries SUCCESS.

**Klaim utama:** README.md:20 '### Composite Bayes Factor: 72,000,000,000 : 1'; README.md:35 '### Posterior: ~100%'; README.md:48-49 'What IS valid: The QUALITATIVE conclusion: 10 independent evidence lines all point in the same direction'; README.md:68 '10 independent evidence lines that all converge'; status README.md:4 'SUCCESS (ILLUSTRATIVE — see caveats)'; README.md:46 'The composite number (72 billion) should NEVER be cited in a paper as evidence.'

**Masukan yang benar-benar dibaca skrip:** No data file is read: bayesian_integration.py:12-14 imports numpy/json/pathlib; ten hand-typed Bayes factors in evidence_lines (:39-117) are multiplied at :125-127 with prior 0.10 (:25); the only file operation is the JSON write (:203). Each BF names an upstream experiment: E108 (assumption model, reads no data), E122, E127, E126, E131, E135 (all hard-coded, scripts only write), E083+E128 (:63-70; colonial register / E091 OV mentions), E085 (:71-78; E027 features_matrix), E069 (:87-93; east_java_sites.geojson), E129 (:94-101; east_java_sites_wiki.csv).

**Ketergantungan pada cacat:**

- **D1 — HEADLINE:** bayesian_integration.py:88-91 'E069: ADV-3 volcanic signal survives survey control (p=0.0015)', bf 10, 'After controlling for survey intensity, volcanic proximity still predicts site absence. P(this|thesis) ~0.9. P(this|no thesis) ~0.1'; README.md:30 'E069: ADV-3 PASSED | 10:1'. Corrected (C023): beta on distance <0 = MORE sites near volcanoes, also on land-only cells (beta -0.734, p 0.0012; E069 direction_check_land_20261001.txt). The burial thesis predicts the opposite sign, so this BF is at most 1, not 10; setting it to 1 alone divides the composite by 10 and breaks 'all point in the same direction'.
- **D2 — NONE:** No BF line reads E082, E062, E105 or geocoded inscriptions: E085 reads E027 features_matrix (adv4_substrate_noise.py:64); E108/E122/E126/E127/E131/E135 scripts read no data; E083/E128 read the colonial register and E091 OV depth mentions.
- **D3 — PARTIAL:** The E069 line rests on east_java_sites.geojson (adv3_survey_intensity.py:54; 662/666 periods unknown, 78 modern-monument names, C029) and the E129 line on east_java_sites_wiki.csv (survey_asymmetry.py:23). Both lines are already void by D1/D4, so D3 adds no independent change.
- **D4 — HEADLINE:** bayesian_integration.py:94-99 'E129: 73% temple survey bias', bf 5, 'Archaeological database is 73% temples — exactly the class that survives burial ... P(this|no thesis) ~0.2'; README.md:31. E129 README correction (lines 8-20, C036): 369 of 391 rows come from two Wikipedia temple-list pages, so the temple share is built into the frame (P(this|no thesis) = 1, BF = 1); '73%' is also temples plus inscriptions (73.1%), the temple share alone being 70.8%. Fixing it divides the composite by 5.
- **D5 — PARTIAL:** Only through the E069 line: 225 of E069's 703 cells have <10% land (E069 direction_check_land_20261001.txt:1). On land cells beta stays negative (-0.734, p=0.0012), so D5 does not rescue the line; it remains void by D1.

**Dikutip di:**

- docs/research_notes/MATA_ELANG_11_2026_03_30.md:68-71,164 ('BF=72 billion | E136 | Estimated, not computed'; 'Never cite BF=72B in a paper')
- docs/research_notes/MATA_ELANG_17_2026_06_08.md:93 ('sudah illustrative only — jangan dihidupkan lagi')
- docs/research_notes/MATA_ELANG_18_PORTFOLIO_2026_06_08.md:88 ('DOWNGRADE->illustrative: cascade E110/E115, E136, E137, E132')
- docs/experiment_index.json:1842 (status SUCCESS)

**Koreksi yang disarankan:** Mark the README RETIRED / ILLUSTRATIVE-ONLY with a correction note: the E069 line (BF 10, script :88-91) is inverted (C023: surplus near volcanoes, so BF at most 1), the E129 line (BF 5, :94-99) is tautological (C036: temple share built into the frame, BF = 1), and E083+E128 (BF 15, :63-70) are not independent (C025: 13 of 16 depth values shared), so '10 independent evidence lines all point in the same direction' (README:48-49, 68) is false (at most 7 remain, and they are correlated, F9) and the composite drops from 7.2e10 to 1.44e9 (9.6e7 without E083+E128) from numbers that are typed anyway. No manuscript or live doc cites the composite (grep of papers/ and docs/), so only docs/experiment_index.json:1842 needs the status change plus a ledger pointer.


## E154 — Comprehensive FDR Re-Audit at 153 Experiments

**Vonis:** AFFECTED_HEADLINE  
**Status README sebelum koreksi:** Status SUCCESS (README:3); no correction note present; docs/experiment_index.json:2115 carries SUCCESS with key_metric 'p<10'.

**Klaim utama:** README.md:26 '| Survive BH | 30 (73.2%) | 65 (78.3%) | **+5.1pp** |'; README.md:28 'Cathedral (p<10^-4) | 10 | 13'; README.md:39 'New cathedral findings: E152a (post-929 longitude shift, p=3.89x10^-12), E084 (inscription-volcano MW, p=5.2x10^-8), E085'; README.md:60 'The project's statistical foundation is STRONGER at 153 experiments than at 90.'; README.md:69 'The core claims (volcanic burial, cosmological overwrite, genre taphonomy, court-center model, post-929 shift) all rest on cathedral findings that survive any reasonable correction.'

**Masukan yang benar-benar dibaca skrip:** No data file is read: fdr_reaudit.py imports only numpy/pathlib and hard-types 83 (experiment, test, p) tuples at :13-133, applies a per-row BH rule at :153-158 (agrees with the step-up rule here), and writes the TSV at :243-247. P-values are copied from upstream READMEs, so every D-dependency is inherited row by row. 21 rows (:27-47) sit under the comment '# Other survivors (estimated from typical VOLCARCH results)' (:26); E110 is entered with '# model fit, not p-value per se' (:89); E109's typed 0.001 (:88) matches no E109 output (README 'p < 0.000001', JSON trend_p 0.0); E068's results/ folder is empty and its README lists only 15 of the 41 p-values. Read-only BH recounts were run from the committed TSV (scratchpad).

**Ketergantungan pada cacat:**

- **D1 — PARTIAL:** Row E069 (:68, rank 51, p=0.0015) is a valid two-sided p but its sign is a surplus near volcanoes. Same-sign rows: E004 (:23, rank 12, cathedral; E004 README:58-62 'rho = -0.991 ... MVR NOT MET'), E005 (:24, rank 14; README:64-69 'This is still the OPPOSITE pattern from H1's simple prediction'), E109 (:88; 'density INCREASES with depth'), E145 (:113; rho +0.908, INFO NEG). BH counts are unchanged because BH is sign-blind, but the 'volcanic burial' clause of README:69 has no supporting row: all four spatial tests show a surplus near volcanoes, and the only spatial row in the cathedral tier is E004.
- **D2 — HEADLINE:** Rows built on E082 coordinates or the 48 one-word Borobudur captions that E030/E062 date by default (year_ce = 750; 48 of 55 century-8 rows, 48 of 130 pre-929 rows): E084 (rank 6; C032: p 5.2e-8 -> 0.068 precise findspots, 0.27 de-dup candi, 0.85 East Java only), E121e (rank 22; bootstrap/permutation on the same file, robustness_wave2.py:140-171), E152a (rank 4), E152b (29), E152c (27), E152d (13); also E102, E104, E105 (C034: 9/43 vs 58%), E149d, E152e (not re-derived). My read-only replication of post929_analysis.py reproduces the typed p-values exactly (3.89e-12, 6.68e-4, 1.36e-4, 2.48e-5). Then E152a -> 1.09e-8 (captions out) -> 1.3e-4 (+ region placeholders out) -> 0.012 (one row per coordinate x period, n=15 vs 10); E152b -> 0.255 on that cut; E152c -> 0.091 and E152d -> 0.229 with captions removed. BH recount with those six rows re-derived: 65/83 (78.3%) -> 59/83 (71.1%), cathedral 13 -> 10, E048 loses its 'RESCUED' status (README:35); two of the three 'new cathedral findings' (README:39) are gone or downgraded. (E145's rho is rank-insensitive to the captions; E147 stays significant, rho 0.593 -> 0.278.)
- **D3 — PARTIAL:** At least 19 rows (18 of the 65 survivors) read the unvalidated inventory or the 142-row candi list (103 unique coordinates, C029/C031): geojson directly E004, E005, E069, E100, E153; wiki csv E129; E109 via E069 cell data; E031 candi list: E031 (rank 5), E121b, E065a/b, E066/E066b, E056, E084, E121e, E104; dashboard sites.csv: E019; E121a (E004 replication). Their p-values are extreme enough that BH survival is probably unchanged, but the island-wide reading is not (C031: the 73 candi other than Penanggungan give R=0.136, p=0.26); E153 passes the modern-feature robustness (C029).
- **D4 — PARTIAL:** Row E129 (:107, rank 71, p=0.09, 'Temple vs non-temple distance t-test') sits in the non-significant bucket and is computed on the temple-list frame (E129 README correction: 369 of 391 rows from two Wikipedia temple lists). Dropping it gives 65/82 = 79.3%; the 'survey bias' reading is unavailable (C036).
- **D5 — PARTIAL:** E004 (rank 12) and its replication E121a (rank 52) divide by a 'Study area: 192,048 km2 (Jawa Timur polygon from OSM)' (E004 README:46), larger than the entire jatim_dem.tif rectangle (about 104,850 km2 from its bounds), so it includes sea; the two bins at 150-200 and 200+ km hold 0 sites over 99,247 km2. E069 (225 of 703 cells <10% land) and E109 (210 of 592) also use sea-inclusive grids. E005's 25-km grid was not checked.

**Dikutip di:**

- docs/EVAL.md:173 ('E154 re-audit: 65/83 survive BH, 78.3%')
- docs/research_notes/MATA_ELANG_12_2026_03_31.md:313-315 (origin of the audit)
- docs/research_notes/MATA_ELANG_13_2026_04_09.md:161-172,370 ('Cathedral findings (survive everything)' list incl. ADV-3 p=0.0015 and E084 p=5.2e-8; count 13 -> 10)
- docs/NEXT_SESSION_BRIEF.md:124-131 (deprecated; cathedral table with E084 and ADV-3 marked 'Clean')
- docs/COMPANION_REPOS.md:43 (E053 as an FDR casualty; unaffected)
- docs/experiment_index.json:2115 (status SUCCESS, key_metric 'p<10')

**Koreksi yang disarankan:** Add a correction note that E154 is a sign-blind table of typed p-values: (1) delete README:60 'foundation STRONGER' and the 'volcanic burial' clause of README:69, because all four spatial rows (E004, E005, E069, E109) are significant in the direction opposite to H1; (2) flag E084/E121e/E152a-d as D2-dependent, since with those six rows re-derived the recount is 59/83 (71.1%) with 10 cathedral rows (E004, E031, E051, E057a/b, E065a/b, E066/b, E085), not 65/83 and 13, and E048 is no longer 'rescued'; (3) disclose that 21 rows are 'estimated from typical VOLCARCH results' (fdr_reaudit.py:26) and E108/E109/E110 have no computed p (traceable rows only: 34/59, 57.6%), and drop the E129 row. Change docs/EVAL.md:173 (remove '65/83, 78.3%' or caveat it), add ledger pointers to the ME#13 cathedral list (:161-172) and NEXT_SESSION_BRIEF.md:124-131, and update docs/experiment_index.json:2115.


## E159 — Robustness Battery for Cathedral Findings

**Vonis:** AFFECTED_HEADLINE  
**Status README sebelum koreksi:** No correction banner (0 hits for CORRECTION / 2026-10-01 / C0xx). The README table (:15-21) disagrees with its own saved results/robustness_results.json: E051 is FRAGILE (rho=0.062, perm p=0.4975, JSON:31-43) and E065_zone_a is ERROR ('scipy.stats has no attribute binom_test', JSON:57-60), i.e. 3 ROBUST / 1 FRAGILE / 1 ERROR, not 5/5. README's E051 rho=0.387 with CI [0.22,0.53] is produced by no code in this folder (rho=0.387 exists only in E051's own README:65; kabupaten_summary.csv has no court-distance column), and README's E065 CI [10.7x,16.8x] does not reproduce (the script's own bootstrap gives [11.3x,15.8x]). README, script and JSON are all stamped 2026-03-31 10:22-10:23.

**Klaim utama:** experiments/E159_robustness_battery/README.md:23 '**5/5 ROBUST.** All cathedral findings survive bootstrap, permutation, and jackknife testing.'; :75 'The project's statistical foundations are solid.'; :77 'These robustness tests can be cited as evidence of careful validation.' (status :3 'SUCCESS (with one important discovery)')

**Masukan yang benar-benar dibaca skrip:** robustness_battery.py reads PRE-canonical outputs, not the canonical30 re-runs: :46 E069 results/adv3_cell_data.csv (703 cells; site counts from data/processed/east_java_sites.geojson per adv3_survey_intensity.py:54,218; 7-volcano distances :59-60); :158 and :347 E031 results/candi_volcano_pairs.csv (142 rows = 103 unique coordinates, 16-volcano inventory); :264 E051 results/kabupaten_summary.csv (115 kabupaten; its only distance column is dist_volcano_km); :346 E082 results/geocoded_inscriptions.csv (20-volcano inventory). Zone A expected fraction is a hand-typed 0.038 (:433-436).

**Ketergantungan pada cacat:**

- **D1 — PARTIAL:** robustness_battery.py:46,57 reads adv3_cell_data.csv and correlates `volcano_dist` (a DISTANCE) with `site_count`; saved partial rho = -0.1308 (robustness_results.json:6) is negative = MORE sites near volcanoes (C023 reading; direction_check_20261001.txt: 3.18 sites/cell within 10 km vs 0.10 at 50-100 km). README:17 and script docstring :8 present it as 'Volcanic signal survives survey control' among the 'cathedral findings', and the ROBUST rule (:132, two-sided) passes a surplus exactly like a deficit. The numbers do not change on correction; what they support does.
- **D2 — HEADLINE:** robustness_battery.py:346 reads E082/results/geocoded_inscriptions.csv (pre-canonical). Of 174 Java rows, 50 Borobudur relief captions share ONE coordinate and 42 sit on region placeholders (Mataram Central Java 20, East Java 18, Central Java 4); 41 distinct coordinates in all. Re-running E159's own Test 4 on the 82 precise findspots (read-only): gap 12.9 -> 4.3 km, Mann-Whitney p 4.3e-8 -> 0.031 (README:20,64 '13.0 km gap ... ROBUST'; E159's own rule :398 needs perm p<0.001). Ledger C032 / E082 re-run on the canonical file: 13.1 -> 2.3 km (p=0.068); 0.2 km (p=0.27) with de-duplicated candi; East Java only 0.2 km (p=0.85). The E084 row does not survive, so '5/5' cannot stand.
- **D3 — HEADLINE:** Tests 2, 4 and 5 read the 142-row candi file (:158,:347): 103 unique coordinates (39 duplicate rows), 73 rows nearest Penanggungan (69 on canonical-30); bootstrap and jackknife treat the 142 rows as independent. Duplicates alone leave R-bar significant (0.348 -> 0.323, p 2e-8 -> 1.5e-5), but README:55 'Candi genuinely cluster west of volcanoes' fails once the one-mountain cluster is separated: on canonical-30 the 73 other candi give R=0.136, p=0.26 (ledger C031; reproduced here). Test 1's counts come from east_java_sites.geojson (666 features, 662 period 'unknown'); removing monument-type or modern-named features leaves partial rho -0.144 to -0.162 (p<=1.3e-4), so that row is robust to D3.
- **D4 — NONE:** No reference to E129, '70.8%' or a settlement class in robustness_battery.py or README.md (grep for E129|70.8|settlement returns nothing).
- **D5 — PARTIAL:** Test 1 uses all 703 E069 cells (:46); 225 have <10% land. Partial Spearman on the 478 land cells (read-only re-run): rho=-0.170 (p=0.0002), and -0.197 (p=1.5e-5) with canonical-30 distances, so the sign survives and strengthens. Test 5's expected fraction is a hand-typed 0.038 (:433-436: 7 volcano disks over whole-Java 129,000 km2) applied to candi that are all East Java (lon 111.09-114.37); a land-masked East Java baseline with canonical-30 volcanoes (my recomputation on E069's 0.1-degree frame) gives about 3.5x at 15 km (3.1x de-duplicated), not 13.5x (README:70). That row is also ERROR in the saved JSON (:57-60).

**Dikutip di:**

- papers/P17_two_javas/draft_v0.3_archcalc.tex:407 - 'All key statistical findings reported in Sections 4.1--4.5 survived bootstrap (10,000 resamples), permutation (10,000 shuffles), and leave-one-out jackknife robustness tests.' The same sentence is in the submitted papers/P17_two_javas/archcalc_submission/P17_manuscript.docx and P17_manuscript_formatted.docx (ArchCalc #365)
- papers/P17_two_javas/revision_ammo/ANTICIPATED_REVIEWER_QUESTIONS.md:54 - 'Bootstrap resampling (10,000 iterations) confirms all five core findings are robust to sample perturbation (E159).'
- papers/P17_two_javas/revision_ammo/ME12_new_evidence.md:22-33 - 'Both E084 and E031 are ROBUST' and the proposed Methods sentence; :80 'One sentence on robustness testing (E159)'
- docs/VOLCARCH_STORY.md:220 - E159 (bootstrap 10K) listed under 'Robustness/validasi'
- docs/research_notes/CRITIQUE_SYSTEM_DESIGN_20260811.md:148 - E159 cited as the pattern for the T5 robustness battery (pattern, not the claim)
- docs/research_notes/CRITIQUE_SYSTEM_DESIGN_20260813.md:200 - 'Pola ada (E121/E159)' (pattern)
- docs/experiment_index.json:2193-2198 - status SUCCESS, key_metric 'p=0.51; rho=-0.131' (mixes E051's volcano-distance p with E069's rho)

**Koreksi yang disarankan:** README banner: retract '5/5 ROBUST' and set status to INFO NEG/REVISIT - the saved JSON has E069 ROBUST, E031 ROBUST, E051 FRAGILE, E084 ROBUST, E065 ERROR; E084 collapses on precise findspots (13.1 -> 2.3 km, p=0.068; 0.2 km, p=0.27 with de-duplicated candi); E069's negative rho is a surplus near volcanoes (D1); E031's west-clustering is the Penanggungan cluster (non-Penanggungan p=0.26 on canonical-30); the README's E051 (0.387) and E065 (CI) rows have no saved output; delete 'can be cited as evidence of careful validation'. Citing documents to change: the 'survived bootstrap, permutation, jackknife' sentence in P17 draft_v0.3_archcalc.tex:407 and in the submitted docx (add it to the pending ArchCalc #365 integrity decision; the notice draft does not mention it), P17 ANTICIPATED_REVIEWER_QUESTIONS.md:54 and ME12_new_evidence.md:22-33,80 (do not reuse), docs/VOLCARCH_STORY.md:220, docs/experiment_index.json:2193-2198, and experiments/E162_synthesis_161/README.md:26-33,70, which tags E066, E051, E031, E084, E065 and E069 'E159: ROBUST' (E159 never tested E066).


## E073 — Spatial vs Linguistic Evidence Meta-Test

**Vonis:** AFFECTED_HEADLINE  
**Status README sebelum koreksi:** No correction banner; 'STATUS: SUCCESS' (:43). README:33 'Fisher's combined p < 1e-30' is wrong: chi2=149.2, df=10 gives 5.4e-27 (the script's 1-cdf underflows to 0.0). docs/experiment_index.json:854-857 has no status.

**Klaim utama:** experiments/E073_spatial_vs_linguistic/README.md:46-48 'ALL spatial tests detect volcanic informedness / NO linguistic test detects volcanic informedness / The two domains are perfectly separated (r = 1.0)'; :39-40 'Mann-Whitney U = 0.0, p = 0.008 (one-tailed)'; :32-33 'Tests significant 5/5 vs 0/4; Fisher's combined p < 1e-30' (:43 'STATUS: SUCCESS')

**Masukan yang benar-benar dibaca skrip:** No file is read. spatial_vs_linguistic_meta.py:26-129 hand-types nine p-values: E065 zone overrepresentation p=1e-6 'conservative bound' (:30) and Rayleigh p=3.4e-8 (:41), both from E031's 142-row candi file (E065 analyze.py:38); E066 equinox p=4.9e-14 (:52) and McNemar p=0.0016 (:63), from E031 orientation_vs_volcano.csv (n=20, E066 analyze.py:30-31); ADV-3 p=0.0015, beta=-0.477, n=703 (:71-80; E069 adv3_cell_data.csv / east_java_sites.geojson); linguistic E029 p=0.569, E038 p=0.68, E067 p=0.146 and 0.734 (:84-129). The script only writes (:335-348).

**Ketergantungan pada cacat:**

- **D1 — PARTIAL:** spatial_vs_linguistic_meta.py:71-80 (and results/evidence_table.csv:6) hand-types ADV-3 as p=0.0015, effect_size=-0.477 (a coefficient on DISTANCE), supports_thesis=True, notes 'Fewer sites near volcanoes even after controlling road/BPCB/university distance' - the inverted C023 reading. Corrected: a SURPLUS near volcanoes (canonical-30 p=2.9e-7, adv3_canonical30_results.json; land-only p=0.0012). The p-value is two-sided, so the vote count, Fisher and Mann-Whitney numbers do not move, but the row's direction label is wrong and a surplus is non-specific (recorded sites are about 71% temples per the corrected E069 README).
- **D2 — NONE:** No inscription, E082 or geocoded_inscriptions reference in spatial_vs_linguistic_meta.py or README.md (grep).
- **D3 — HEADLINE:** Rows 1-2 are E065 on E031's 142-row candi file (E065 analyze.py:38; 103 unique coordinates, 39 duplicate rows, 73 rows around Penanggungan, E065 README:34). Duplicates alone leave the Rayleigh row significant (de-duplicated p=1.5e-5), but on canonical-30 the non-Penanggungan candi give R=0.136, p=0.26 (ledger C031; reproduced). Then that row's p exceeds the smallest linguistic p (0.146): README:46 'ALL spatial tests' becomes 4/5 and README:39-40 'U=0.0 ... r=1.0 (perfect separation)' becomes U=1, r=0.9, one-tailed p=0.016 (my recomputation). Row 5 rests on east_java_sites.geojson (666 OSM/Wikipedia features, 662 period 'unknown') but is robust to removing monument/modern features (rho -0.134 to -0.162, p<=4e-4).
- **D4 — NONE:** No E129, '70.8%' or settlement-class reference in the script or README (grep). F9 note: the corrected E069 README says recorded sites are about 71% temples (E129), so rows 1, 2, 5 and E066's 20 candi (a subset of the same 142) are one temple channel, not independent tests for Fisher's method.
- **D5 — PARTIAL:** Row 1: E065's 'expected 3.4' is 142 x (10/R_max)^2 with R_max = 65.1 km, a single disk with no land mask or volcano inventory (E065 analyze.py:271-277; reproduced 3.35). A land-masked East Java baseline with canonical-30 volcanoes (my recomputation) gives about 6.0x at 10 km (5.1x de-duplicated), not 17.9x (script:31). Row 5: 225 of E069's 703 cells have <10% land; the land-only refit keeps sign and significance (direction_check_land_20261001.txt: NB2 p=1.1e-7, quasi-Poisson p=0.0012).

**Dikutip di:**

- docs/IDEA_REGISTRY.md:173 - I-114 'p=0.008, r=1.0. Volcanic influence is spatial-behavioral, not lexical.'
- docs/IDEA_REGISTRY.md:172 - I-113 (E067) 'VI = behavioral, not lexical' (the framing E073 'tests')
- docs/TRIGGER_MAP.md:236 - 'P11 Discussion: volcanic informedness = BEHAVIORAL (architecture, calendar), NOT LEXICAL'
- docs/research_notes/MATA_ELANG_14_2026_04_16.md:22 - E067 'reframed as behavioral, not lexical'
- docs/research_notes/MATA_ELANG_15_2026_04_20.md:142 - E067 'behavioral, not lexical'
- docs/experiment_index.json:854-857 - index entry (no status or key_metric)

**Koreksi yang disarankan:** Add a README banner and move status from SUCCESS to INFO NEG: 'ALL spatial tests detect volcanic informedness' and 'perfect separation' are not supported - ADV-3 is a surplus near volcanoes (canonical p=2.9e-7, not 'fewer sites'), the Rayleigh row is the Penanggungan cluster (non-Penanggungan p=0.26 on canonical-30, giving U=1, r=0.9, p=0.016), E065's 17.9x is about 6x on a land-masked baseline, and the two E066 rows test astronomical orientation from hand-coded cardinal labels (a 4-category null gives p=0.0013 for 17/20, not 4.9e-14), while all nine p-values are hand-typed and Fisher-combined although rows 1-5 share one candi/temple channel (F9). Retire 'p=0.008, r=1.0' (docs/IDEA_REGISTRY.md:173) and the 'BEHAVIORAL, NOT LEXICAL' lines (docs/TRIGGER_MAP.md:236, IDEA_REGISTRY.md:172; the E067 null stands, but a null does not establish a behavioural alternative); no manuscript in papers/ cites E073, so no paper edit is needed.


## E085 — ADV-4 Substrate Noise Permutation Test

**Vonis:** AFFECTED_MINOR  
**Status README sebelum koreksi:** No correction banner. README:108 still reads '| ADV-3 (Survey intensity) | L1 | **PASSED** (p=0.0015) |'. docs/experiment_index.json:980-985 key_metric 'AUC: 0.7599; p=0.760' (p=0.760 is ADV-2's Fisher p from README:107, mis-parsed). README:88 calls the non-Austronesian class 'Sanskrit-influenced vocabulary' although README:31 and the script (:82) define it as residual (non-cognate) forms.

**Klaim utama:** experiments/E085_adv4_substrate_noise/README.md:3 '**Status: SUCCESS - VOLCARCH L4 SUPPORTED**'; :82-83 '**The substrate detection is NOT noise.** The observed AUC of 0.762 is: 11.1 standard deviations above the permuted null (p = 0.0000)'

**Masukan yang benar-benar dibaca skrip:** adv4_substrate_noise.py:40,64 reads only experiments/E027_ml_substrate_detection/data/features_matrix.csv (1,357 rows x 23 columns: Sulawesi lexical forms, labels 919 Austronesian / 438 residual from E022 per E027 README:15). No coordinates, distance, site or inscription columns (checked). Results: results/adv4_summary.json (observed RF AUC 0.7618, permuted mean 0.5001 sd 0.0237, z=11.05, 0/1000 permutations >= observed; no-lcov AUC 0.759; lcov alone 0.6803).

**Ketergantungan pada cacat:**

- **D1 — PARTIAL:** README.md:108 '| ADV-3 (Survey intensity) | L1 | **PASSED** (p=0.0015) |' restates the inverted ADV-3 verdict (C023: surplus near volcanoes, INFO NEG). The scorecard line is a cross-reference; E085's own ADV-4 result does not use ADV-3.
- **D2 — NONE:** adv4_substrate_noise.py:40,64 reads only E027 features_matrix.csv, which has no coordinate or inscription fields.
- **D3 — NONE:** No site inventory (east_java_sites.geojson / east_java_sites_wiki.csv / candi list) is read; the input is lexical data only.
- **D4 — NONE:** No E129 or temple/settlement reference in the script or README (grep).
- **D5 — NONE:** No grid or raster is used; the unit is a lexical form, not a map cell.

**Dikutip di:**

- docs/L2_STRATEGY.md:178 - 'ADV-4 Substrate noise | E085 | L4 | PASSED - p=0.0000, z=11.05'; the row above it (:177) still says ADV-3 'PASSED'
- docs/NEXT_SESSION_BRIEF.md:76 - ADV-4 PASSED (p=0.0000, z=11.05); :75 ADV-3 'PASSED' uncorrected
- docs/VOLCARCH_STORY.md:248 - 'ADV-4 ... PASSED (z = 11,05)'; its ADV-3 row (:247) is already corrected
- docs/research_notes/MATA_ELANG_13_2026_04_09.md:168 - 'ADV-4: Substrate signal is real | E085, z=11.05' (:167 carries ADV-3 'survives survey control')
- docs/research_notes/MATA_ELANG_12_2026_03_31.md:59 - E085 counted among 'Genuine hypothesis tests ... E069, E085, E108'
- docs/experiment_index.json:980-985 - key_metric 'AUC: 0.7599; p=0.760'

**Koreksi yang disarankan:** Add a README note correcting line 108 (ADV-3 'PASSED' is inverted: surplus near volcanoes, INFO NEG, C023) and restating the scope of the ADV-4 result: the permutation test shows the features discriminate E022's residual labels (AUC 0.762 vs 0.50), not that the labels are substrate; report p<0.001 (0/1000 permutations) instead of 'p = 0.0000' and fix the 'Sanskrit-influenced' wording at line 88. Fix the sibling ADV-3 'PASSED' rows next to E085 in docs/L2_STRATEGY.md:177 and docs/NEXT_SESSION_BRIEF.md:75 (and the stale index key_metric at docs/experiment_index.json:980-985); the ADV-4 claim itself needs no D1-D5 change in docs/VOLCARCH_STORY.md:248.


## E080 — Fieldwork Targeting - Priority Zones

**Vonis:** AFFECTED_HEADLINE  
**Status README sebelum koreksi:** No correction banner; status SUCCESS (:4). README:16 says '~2,500 grid cells' while results/e080_results.json:4 has n_candidates=4600. docs/fieldwork/BOREHOLE_PROTOCOL_v1.md, built on E080 targets, was already RETIRED on 2026-10-01 (it names E080/E166 and C035).

**Klaim utama:** experiments/E080_fieldwork_targets/README.md:54 'All top 20 targets cluster near Kelud and Arjuno-Welirang - volcanoes with high sedimentation rates AND nearby candi proving historical occupation. These are the strongest candidates for VOLCARCH's core prediction: buried sites at depth.'; :42 'Top target: -7.98, 112.36 (score 0.855) - 8 km from Kelud, 5 m predicted burial.'

**Masukan yang benar-benar dibaca skrip:** fieldwork_targeting.py reads NO data file: DATA_DIR (:38) and load_csv (:42) are never called; COLONIAL_DEPTHS (:91-96) is never used. Everything is hand-typed: 7 eastern volcanoes (:62-70), 14 candi 'abbreviated list' (:72-88), terrain = latitude proxy abs(lat+7.8) (:153), burial = 165*15*exp(-d/5)/100 'Simplified' (:166), step-function weights (:112-176). README:30 and :62 list E005, E013, E065, E070, E075 and ADV-3 as inputs; none is read. Ranking: stable sort (:228) of 4,600 cells in which 29 tie at exactly 0.855 (all 20 rows of top20_targets.csv are 0.855), so the 'top 20' are the first 20 tied cells in scan order (latitude ascending), dropping 7 Penanggungan and 2 Arjuno-Welirang ties. generate_infographic.py:21-23,52 reads top20_targets.csv, E097 top50_anomaly_cells.csv and data/processed/east_java_sites.geojson (display layer only).

**Ketergantungan pada cacat:**

- **D1 — PARTIAL:** Nominal only. README.md:22 '| Discovery gap (low site density = high potential) | 20% | ADV-3 survey intensity |' and fieldwork_targeting.py:11 '3. Survey gap (ADV-3) - areas with volcanic signal after survey control' cite the inverted ADV-3 (C023: MORE recorded sites near volcanoes). The code never reads ADV-3: gap_score (:140-147) is a step function of distance to 14 hand-typed candi. After correction ADV-3 gives no support to the premise that near-volcano low-density cells hide buried sites.
- **D2 — NONE:** No inscription, E082 or geocoded_inscriptions reference in any E080 script or README (grep).
- **D3 — HEADLINE:** Class-level, not file-level: fieldwork_targeting.py does not read the geojson or wiki CSV (only generate_infographic.py:23,52 does, display only). But the headline coordinates depend on an equally unvalidated inventory: KNOWN_CANDI (:72-88, 14 hand-typed entries) whose coordinates differ from E031's by up to 48.9 km (Sawentar), 12.8 km (Jawi, Kidal), 11.8 (Sumberawan), 10.2 (Penataran); candi_score + gap_score = 45% of the composite. Re-running the script's own formulas (scratchpad, read-only) with E031's de-duplicated candi (103-104 unique coordinates) leaves only 7 of the 20 cells at the maximum score (13 fall to 0.795). The volc_score rationale (:113 '17.9x overrepresentation (E065)') comes from the duplicated 142-row list (E065 README:20).
- **D4 — NONE:** No E129, '70.8%' or settlement-class reference in fieldwork_targeting.py or README.md (grep).
- **D5 — PARTIAL:** Rectangular grid lat -8.3..-7.4, lon 111.5..113.5 with no land mask (fieldwork_targeting.py:215-222): 543 of 4,600 cells (11.8%) are sea or outside DEM land (jatim_dem.tif sampled after UTM 49S reprojection). None is among the 100 best-scored cells and all 20 targets are on land (434-2,260 m), so the target list is unchanged; n_candidates and any cell-share statistic include sea.

**Dikutip di:**

- papers/P1_taphonomic_framework/revision_ammo/E141_COLONIAL_DATA_VALIDATION.md:28 - '23% of geocoded colonial finds fall within 25km of VOLCARCH's computationally-predicted fieldwork candidates (E080). Random expectation: 4%. Enrichment: 5.8x, chi-squared p < 0.00001.'
- papers/P17_two_javas/revision_ammo/E141_COLONIAL_DATA_VALIDATION.md:28 - same text
- papers/P17_two_javas/OUTLINE_v0.1.md:111 - 'Zone B/C targets (E080, E097) are in VOLCANO JAVA - the buried sacred landscape'
- docs/fieldwork/BOREHOLE_PROTOCOL_v1.md:8,54-67 - targets 'come from E080/E166' (already RETIRED 2026-10-01)
- docs/research_notes/OBJECTIVE_ANSWER_20261001.md:272 - already lists the E080/E166-derived protocol as no longer usable
- docs/TRIGGER_MAP.md:199 - 'E097 SUCCESS ... 65% overlap with E080 top 20 targets (13/20 within 5km)'
- docs/L1_CONSTITUTION.md:126 - fallback 'keep E110 cascade + E080/E097 fieldwork targeting'
- docs/buku/OUTLINE_v0.1.md:92 - 'Bab 9 ... 20 koordinat GPS target (E080)' (six more mentions in docs/: dissemination/youtube_ep2_outline.md:146, plans/senter_v3_handoff.md:13, AUTORESEARCH_CONCEPT.md:165-175, research_notes/MATA_ELANG_11_2026_03_30.md:111, L3_EXECUTION.md:81, research_notes/SATELLITE_ARCHAEOLOGY_FRONTIER.md:49,80,108)

**Koreksi yang disarankan:** README banner and status SUCCESS -> INFO NEG/REVISIT: the 20 'targets' are the first 20 of 29 cells tied at exactly 0.855 in grid-scan order, built from hand-typed inputs (7 volcanoes; 14 candi that disagree with E031 by up to 49 km; a latitude 'terrain' proxy, with 7 targets at 1,050-2,260 m on Arjuno-Welirang; an assumed 165x15 cm x exp(-d/5) burial formula), not from the E005/E013/E069/E075 outputs the README lists, so the ADV-3 attribution (inverted, D1) and '5-8 m predicted burial' are not evidence, and swapping in E031's de-duplicated candi leaves only 7 of 20 at the maximum. Annotate so that nothing cites E080 as a prediction or validation: the E141 'enrichment 5.8x' in P1 and P17 revision_ammo (E141_COLONIAL_DATA_VALIDATION.md:28), docs/TRIGGER_MAP.md:199 (E097 '65% overlap'), docs/L1_CONSTITUTION.md:126, docs/buku/OUTLINE_v0.1.md:92 and P17 OUTLINE_v0.1.md:111; BOREHOLE_PROTOCOL_v1.md is already retired.


## E105 — BERTopic Topics x Geographic Distribution (zone x topic, 929 CE relocation)

**Vonis:** AFFECTED_HEADLINE  
**Status README sebelum koreksi:** No correction banner. README.md:3 still 'Status: SUCCESS (descriptive, not statistically strong)'; file unchanged since its first commit 7d0b4a7 (2026-03-17). RESULTS_CANONICAL30_20260813.md:23 ('arah temuan tidak berubah') is now contradicted. The E082 README banner (:3-22) and ledger C034 name E105, but E105's own folder carries nothing; docs/EXPERIMENT_INDEX.md:167 still says SUCCESS. README header lists P5, P7, P9 but none of them cites E105; the real citers are P11 and P17.

**Klaim utama:** experiments/E105_topic_geography/README.md:32 'Sanskrit-dominant inscriptions are MASSIVELY concentrated in the court zone (56/78 = 72%)'; :53-56 'Pre-929: Court zone dominates (58/101 = 57%), almost entirely Sanskrit (53/58 = 91%) ... Post-929: Periphery dominates (19/36 = 53%), mostly mixed/indigenous (17/19 = 89%) ... it changed WHERE they're written'; :74 'Completes the Two Javas model'. Canonical-30 restatement, RESULTS_CANONICAL30_20260813.md:16-21: 58.0% (58/100), 91.4% (53/58), 48.4% (15/31), 86.7% (13/15), n=131; its :23 says 'arah temuan tidak berubah'.

**Masukan yang benar-benar dibaca skrip:** e105_rerun_canonical30.py:24 reads E062_temporal_synthesis/results/joined_dated_inscriptions.csv (filename, year_ce, pre_indic_ratio; :30-41); :25 reads E082_inscription_georeferencing/results/canonical30/geocoded_inscriptions_canonical30.csv (volcano_dist_km_c30; :45-52). Upstream: year_ce is a title regex with century midpoint (E030_prasasti_temporal_nlp/00_temporal_analysis.py:115-120 'return century * 100 - 50  # 8th century -> 750'); coordinates are E082's hard-coded known-location table (E082 README:30) whose own caveat is 'regional centroids ... +/- 20 km' (README:92), wider than the 15 km zone bands. No site inventory, E069, E129, grid or raster is read. The original March script is not in the repo (git: commit 7d0b4a7 added only README.md and results/e105_results.json, which holds topic-distance means and no zone table), so the March tables are unreproducible; README:54 divides by 36 while its own post-929 table (:45-50) sums to 31.

**Ketergantungan pada cacat:**

- **D1 — NONE:** No mention of E069/ADV-3 or 'deficit near volcanoes' in README.md (all 77 lines) or e105_rerun_canonical30.py (all 131 lines).
- **D2 — HEADLINE:** e105_rerun_canonical30.py:25 reads the E082 canonical file. A read-only re-join of the same two files (scratchpad, nothing written) reproduces n=131 (100 pre / 31 post) and shows: 48 records are Borobudur relief captions at ONE point (-7.61,110.2; 28.2 km, so 'Court'), all year_ce=750.0 from E030's century midpoint (E082's own date_ce is empty for 50 of the 51 rows there); 22 sit on 'Mataram Central Java'/'Central Java' placeholders (12.5 km, so 'Volcano'; 21 of the 38 pre-929 volcano-zone rows); 17 on 'East Java' placeholders (16 land in 'Periphery'). Only 44 of 131 are precise findspots. Pre-929 court share 58/100 -> 10/52 (19%) without captions (ledger C034's 9/43 uses E082's own date_ce); Sanskrit share of pre-929 court 53/58 -> 5/10; all-era Sanskrit-in-court (README:32, 72%) -> 8/27 (30%); post-929 periphery 15/31 -> 2/16 once placeholders go (13 of the 15 are 'East Java' rows). Precise-only: court share 10/28 (36%) pre vs 11/16 (69%) post, periphery 1/28 vs 2/16, so no court-to-periphery relocation is visible (n too small to test).
- **D3 — NONE:** Script reads no east_java_sites.geojson, east_java_sites_wiki.csv or candi list. Only 2 of 131 joined rows are 'candi_match' (E082 borrows E031 coordinates): immaterial. README:64-66 'Volcano Java (0-15km): Candi' is interpretive (from E065/E031), not computed here.
- **D4 — NONE:** No E129 or temple-share input anywhere in the folder.
- **D5 — NONE:** Point records are binned by distance to the nearest volcano (script:64-71); no grid, raster or area null, so no sea cells enter. Latent only: README:34 calls the volcano zone and periphery 'underrepresented' with no land-area denominator.

**Dikutip di:**

- papers/P17_two_javas/draft_v0.3_archcalc.tex:73 and :98-99 (ArchCalc #365 source; 'Post-929 relocation (E105) 91% Sanskrit -> 89% indigenous', the March numbers)
- papers/P17_two_javas/archcalc_submission/fix_tables.py:67 (builds the same row into the submitted .docx)
- papers/P17_two_javas/OUTLINE_v0.1.md:148 ('Sanskrit 72% court, post-929->periphery')
- papers/P11_volcanic_informedness/draft_v0.8_spafa.tex:202-203 ('58 per cent ... 91 per cent ... 48 per cent (n=31) ... 87 per cent'; same text in draft_v0.7_spafa.tex:203 and draft_v0.6_spafa.tex:191)
- papers/P11_volcanic_informedness/SIG_signoff.md:74 (NO-GO to-do 're-derive E031/E065/E105/E153')
- papers/P11_volcanic_informedness/external_reviews/G9_CROSS_MODEL_20260813.md:16-17,33 (origin of the canonical re-run)
- docs/research_notes/OBJECTIVE_ANSWER_20261001.md:235 (already flags it: 'hanya 9/43 (21%)', C034)
- docs/CRITIQUE_LEDGER.md:92 (C034)

**Koreksi yang disarankan:** Put a CORRECTION 2026-10-01 banner on E105's README (status SUCCESS -> INFO NEG, 'not supported'): the 72% court-zone Sanskrit concentration and the 929 CE court-to-periphery relocation are produced by 48 Borobudur relief captions (one coordinate, dated 750 CE by E030's century midpoint) and by region placeholders; without captions the pre-929 court share is 10/52 (19%), and with precise findspots only periphery is 1/28 pre vs 2/16 post (C032/C034); mark RESULTS_CANONICAL30_20260813.md:23 as superseded. Change the citers: P17 draft_v0.3_archcalc.tex:73,98-99 and archcalc_submission/fix_tables.py:67 (submitted text still carries the March 91% -> 89%; fold into the pending C032 integrity decision), P11 draft_v0.6-v0.8 section on 929 CE (drop; P11 is already NO-GO), P17 OUTLINE_v0.1.md:148. Also E154 fdr_reaudit.py:82 enters E105 at p=0.001, a value no E105 output contains (only Kruskal-Wallis p=0.580, README:60); regenerate the experiment index after the status change.


## E158 — Steelman Counter-Arguments for Cathedral Findings

**Vonis:** AFFECTED_HEADLINE  
**Status README sebelum koreksi:** No correction banner. README.md:3 'Status: SUCCESS'. Ledger C023 lists E158 as a pending downstream audit; the generated index rows still show SUCCESS with key_metric 'AUC=0.762; p=0.0015'.

**Klaim utama:** experiments/E158_steelman_counter_arguments/README.md:134 'VOLCARCH's strongest claims are the cathedral findings (E066, E051, E084, E085, E069) that survive any statistical correction and have clear, simple interpretations.'; :126 table row 'E069 Survey control p=0.0015 | MODERATE-STRONG | MEDIUM'; :136 'Recommendation for P17: Lead with cathedral findings (E084 post-929 shift, E105 Two Javas pattern).' The one finding it ranks weakest is the E110 cascade (:132).

**Masukan yang benar-benar dibaca skrip:** None: README-only (no script, no data, no results/). Every number is quoted from other experiments: E069 'quasi-Poisson beta=-0.477, p=0.0015' (:64), E129 '73% temple bias' (:74), E031 'Rayleigh p=3.4e-8' and E065 '17.9x' (:112-113), E066 '85% of Java's 142 candi' (:103), E084/E105 (:134,136), E108 3,220x (:28), E110 (:45-58), E085 z=11.05 (:83). Checked at source: E084 reads E082 geocoded_inscriptions.csv and E031 candi_volcano_pairs.csv (E084_inscription_volcano_spatial/inscription_spatial_test.py:51,63); E069 reads east_java_sites.geojson on a 0.1-degree rectangle (adv3_survey_intensity_canonical30.py:38,72); E066's 85% is 17/20 candi (E066 README:14,24), not 142.

**Ketergantungan pada cacat:**

- **D1 — HEADLINE:** README:64 and :126 carry E069's 'beta=-0.477, p=0.0015' as a surviving volcanic signal and :134 lists E069 among claims that 'survive any statistical correction'. The coefficient is on distance (E069 adv3_survey_intensity/README.md:3-22; ledger C023), so beta<0 means MORE recorded sites near volcanoes (a surplus); canonical-30 beta=-0.831 and the land-only refit (-0.584 / -0.734, direction_check_land_20261001.txt) keep the sign. The steelman never raises the sign, so the 'MEDIUM' vulnerability rating and E069's cathedral status do not hold. docs/NEXT_SESSION_BRIEF.md:132 repeats 'Clean - survives survey intensity control'.
- **D2 — HEADLINE:** Not through Findings 1-5 (none reads E082) but through the overall assessment: :134 lists E084 as surviving and :136 recommends leading P17 with 'E084 post-929 shift, E105 Two Javas pattern'. E084 reads E082's geocoded_inscriptions.csv (inscription_spatial_test.py:51) and E105 reads the canonical E082 file; both rest on 175 inscriptions at 42 coordinates (C032), where the candi-inscription gap falls from 13.1 km to 0.2-2.3 km (p 0.07-0.85) with precise findspots, and E105's 929 CE split is the Borobudur-caption artefact (C034). The P17 recommendation reverses.
- **D3 — PARTIAL:** Finding 3's p=0.0015 sits on E069's inventory east_java_sites.geojson (666 OSM/Wikipedia features, 662 period 'unknown'; adv3_survey_intensity_canonical30.py:38; C029). Finding 5's rebuttal (:112 'siting clusters WEST of volcanoes (Rayleigh p=3.4e-8)') uses the 142-row candi file, which has 103 unique coordinates (verified in E031 candi_volcano_pairs.csv) and is a Penanggungan effect (other 73 candi p=0.26, C031). E084 (in the :134 list) also reads that candi file (script:63). E066's 85% itself (n=20 orientation records) is untouched.
- **D4 — PARTIAL:** README:74 'E129 (73% temple bias) shows that what Indonesia surveys is primarily stone temples ... explains the survey deficit': the E129 share is tautological (369 of 391 rows from two temple-list pages; 'settlement 1.3%' = 5 generic rows; E129 README:9-24, C036), so this rebuttal bullet falls; the E086 Japan bullet (:73) is untouched.
- **D5 — PARTIAL:** Finding 3 inherits E069's rectangular 0.1-degree grid: 225 of 703 cells have <10% land (direction_check_land_20261001.txt; C035), though the land-only refit keeps the negative sign. Rebuttal :113 cites E065's '17.9x', whose area null is a full disc, expected = n * pi*10^2 / (pi*max(distance)^2) (E065_candi_elevation_analysis/analyze.py:269-277), with no land mask.

**Dikutip di:**

- docs/CRITIQUE_LEDGER.md:81 (C023 'Sisa' list: E158 awaits downstream audit)
- docs/HANDOFF_20261001.md:130,162 (audit queue)
- docs/experiment_index.json:2187 and docs/EXPERIMENT_INDEX.md:220 (key_metric 'AUC=0.762; p=0.0015')
- docs/NEXT_SESSION_BRIEF.md:124-132 (same cathedral claims without the E158 ID: ':130 E084 ... Clean - genuinely novel', ':132 ADV-3 volcanic signal 0.0015 Clean - survives survey intensity control')
- docs/research_notes/MATA_ELANG_12_2026_03_31.md:50 (origin of the steelman requirement, not a citation of the result)
- lines/06_thesis/CLAUDE.md:75 (outside the papers/docs grep: cites E158 as 'adversarial self-attack'); no file under papers/ cites E158

**Koreksi yang disarankan:** Add a correction banner and withdraw Finding 3's rating and the Overall Assessment lines :134 and :136: E069's beta=-0.477 is on distance, i.e. a surplus near volcanoes (C023), and E084/E105 rest on the E082 geocoding artefact (C032/C034), so neither E069 nor E084 can be a 'cathedral finding that survives any statistical correction', and P17 should not lead with E084/E105. Also flag the Finding 3 rebuttal via E129 (C036), the Finding 5 siting rebuttal (C031, Penanggungan) and the mis-scale at :103 (E066's 85% is 17/20 candi, not 142); Finding 2 (E110 as weakest flank) and Finding 4 are not touched by D1-D5. Update lines/06_thesis/CLAUDE.md:75 and docs/NEXT_SESSION_BRIEF.md:130-132, and regenerate the experiment index; no paper cites E158.


## E098 — Systematic Literature Database - Sedimentation, Burial, and GPR Feasibility

**Vonis:** AFFECTED_MINOR  
**Status README sebelum koreksi:** No correction banner. README.md:3 'Status: SUCCESS'; meta_analysis.md:176 uncorrected. Minor G1 gaps: README:20 says 69 sedimentation entries but :119 says '66-entry database'; README:58-60 global stats (n=30, mean 5.14 m) do not reproduce from the 29-row CSV (26 numeric rows, mean 5.29 m), whereas the Indonesian n=9 / 3.57 m does.

**Klaim utama:** experiments/E098_lit_database/README.md:88 'Three independent approaches converge on ~3.4-3.6 m mean burial depth for Java's volcanic archaeological sites.'; :92 'Meta-analysis confirms and quantifies VOLCARCH's central claims.'; :76-78 'GPR penetrates 1.5-2.5 m in Java's andosols - far short of the 3.5 m mean burial depth'; :70 'Java's volcanic sedimentation rates (0.5-2.8 cm/yr ...) guarantee multi-meter burial'.

**Masukan yang benar-benar dibaca skrip:** No script. Hand-compiled literature CSVs in results/ (volcanic_sedimentation_rates.csv 69 rows, buried_sites_volcanic.csv 29, gpr_tropical_volcanic.csv 20; each row has a 'reference' column) plus results/meta_analysis.md. The Indonesian figure (n=9, mean 3.57 m, median 3.0 m) reproduces from the CSV. Other experiments enter by citation only: E083 3.41 m n=24 (README:84), E070 register mean 3.2 m n=32 (README:86, labelled 'E075 model prediction'), E075 grid shares 32.3% / 12.8% and ~10,000 km2 (meta_analysis.md:40,164,191-192,225), E069 p=0.0015 (meta_analysis.md:176). Circularity inside the database, against README:100 'measured, not modeled': 3 rate rows are inferred from VOLCARCH burial depths (volcanic_sedimentation_rates.csv:21,67,68) and 2 burial rows cite 'VOLCARCH E070/E083' (buried_sites_volcanic.csv:13,16).

**Ketergantungan pada cacat:**

- **D1 — PARTIAL:** results/meta_analysis.md:176 '4. Low survey intensity (E069: p=0.0015 for survey bias)' is one of five factors in 'Java is the worst combination' (:171-179), echoed uncited at README:74. E069's p is the likelihood-ratio p of a distance coefficient, not a measure of survey bias, and its sign shows MORE recorded sites near volcanoes after survey control (C023). The headline numbers (rates, depths, GPR) do not use it.
- **D2 — NONE:** No inscription or geocoded-inscription input in README.md, results/*.csv or meta_analysis.md (no match for 'inscription' or 'E082').
- **D3 — NONE:** Reads neither east_java_sites.geojson nor east_java_sites_wiki.csv nor a candi list; the three CSVs are literature compilations with per-row references. The E083/E070 depths it cites come from the colonial register (E070 README:211-225), not the OSM/Wikipedia inventory.
- **D4 — NONE:** No E129 or temple-share use (no match for 'E129' or 'temple bias').
- **D5 — PARTIAL:** The quoted E075 shares are fractions of a rectangular, unmasked grid: lat -8.8..-7.2, lon 110.5..114.8, 2,838 cells (E075 sedimentation_burial_model.py:190-197; README:10; no land mask in the script; 7 volcanoes, README:7). My read-only check against data/processed/dem/jatim_dem.tif: 929 of the 2,442 in-DEM cells (about 38%) have <10% land, so '32.3% / 12.8% of East Java cells' (meta_analysis.md:40,192) are rectangle shares (land-only about 47.7% / 23.9%). The ~10,000 km2 (:164,225) is the 0-2025 CE window; E075's own 400-1500 CE window gives 1.5% (>1 m) and 0.2% (>3 m), i.e. 6 cells (E075 README:22).

**Dikutip di:**

- docs/plans/senter_v3_handoff.md:73 ('GPR can't reach mean burial depth - E098 meta-analysis ... mean burial is 3.41m')
- docs/plans/senter_v3_handoff.md:75 ('Three independent methods converge on ~3.4-3.6m - E075 model, E083 field measurements, E098 global literature')
- docs/plans/senter_v3_handoff.md:16,31 (E098 summary '69 sed. rates + 29 buried sites + 20 GPR surveys')
- docs/TRIGGER_MAP.md:207-209 ('E098 SUCCESS ... GPR <= 2.5m in Java')
- docs/research_notes/DESK_TESTS_T0_T4_T7_20261001.md:530 (Liyangan 'about 3 m in E098' vs about 10 m in OBJECTIVE_ANSWER)
- docs/CRITIQUE_LEDGER.md:81 and docs/HANDOFF_20261001.md:130 (audit pointers only)
- lines/05_archival_nlp/CLAUDE.md:66 (outside the papers/docs grep); no file under papers/ cites E098 by ID

**Koreksi yang disarankan:** Keep the databases; add a dated note to README/meta_analysis.md that line 176 ('E069: p=0.0015 for survey bias') is withdrawn (C023: beta on distance < 0 means more recorded sites near volcanoes; it measures no survey bias) and that the E075 shares at :40,164,191-192,225 are fractions of an unmasked 2,838-cell rectangle for the 0-2025 CE window (400-1500 CE: 1.5% / 0.2%), not East Java land shares. Adjacent, outside D1-D5: README:88 'three independent approaches' is not independent (E083 draws on the E070 register, E070 README:225; the 3.2 m row is E070's own mean mislabelled 'E075 model prediction'; two E098 depths cite E070/E083), so call it a correlated convergence (F9, cf. C025/C026). Fix the citers docs/plans/senter_v3_handoff.md:73,75 and docs/TRIGGER_MAP.md:209 that repeat the 'independent / converge' wording; no paper cites E098 by ID.

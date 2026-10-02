# E226 / T7 settlement register: search notes (agent log)

Compiled 2026-10-02 by a delegated Claude agent following `register/PROMPT.md` (kept verbatim in this folder).
Output: `t7_settlement_register_candidates.csv` (22 rows; 1 YES, 9 UNCLEAR, 12 NO).
Extra columns appended after the requested ones: `access_mode` (how the source was read) and `region_tier`
(primary / buffer). The 16 requested columns are in the requested order.

## 1. Blindness record (binding)

- NOT opened: anything under `experiments/E226_t7_detection_calibration/frame/` or `/results/`; the DHARMA corpus
  (`experiments/E023_ritual_screening/data/dharma/`); any inscription-derived village list.
- I only listed the E226 directory (names of `DESIGN.md`, `frame/`, `register/`, `results/`, `scripts/`); I did
  not open `DESIGN.md`.
- No search used an inscription village name. Sites were found by site type + region only (queries below).
- Incidental exposure to inscription names inside archaeological papers, disclosed for the record, none used for
  selection: the Liangan monograph (read from `data/raw/literature_cache/t0_liangan/`, a pre-existing repo copy)
  mentions the Rukam inscription at Parakan; Kusen 1995 (Ratu Boko background paper, scanned for habitation
  keywords) cites a Mantyasih inscription passage; Putri et al. 2024 mentions toponym data (Kusen 1990);
  Christie 1991 discusses sima charters at a general level. No site was added or ranked because of these.
- Other files already sitting in the session scratchpad (`wilayah.sql`, `b0.py`, `index_backup_20261002` etc.)
  belong to another process and were not read or modified; my working files are in `scratchpad/reg/`.

## 2. How `qualifies` and `opened_source` were assigned

- YES = excavated/recorded open-air habitation deposit (house or domestic features and/or domestic assemblage)
  AND a 8th-10th c. CE date by radiocarbon or diagnostic ceramics AND the source itself treats it as habitation.
- UNCLEAR = at least one of those is missing or only inferred (sherds only, news-only, function disputed,
  date not isolated, secondary source only). The `reason` column says which.
- NO = temple on its own, survey-only/undated features, hoard, pilgrim use, or outside the date window.
- `opened_source`: YES = I downloaded and read the document text (`access_mode` full_text) or fetched the web page
  and read the page-fetch extract (`access_mode` fetch_extract, mostly news pages; the fetch tool returns an
  extractive summary of the page, not raw text, so treat fetch_extract rows as one notch weaker). NO = I only saw
  search-result snippets or a secondary citation.
- Coordinates: only Liyangan (source states S7 15 07.0, E110 01 37.4) and Ratu Boko (Wikipedia site-level
  coordinate) are filled; everything else is blank because no source I read gave one.

## 3. Search log (all 2026-10-02)

Web search (general engine) queries, Indonesian and English, in the order run:
1. situs Liyangan permukiman Mataram Kuno ekskavasi rumah abad IX Temanggung
2. Liyangan archaeological site Temanggung settlement excavation Mataram house granary radiocarbon
3. ekskavasi situs permukiman Mataram Kuno abad IX sisa hunian Sleman Klaten Berkala Arkeologi gerabah umpak
4. Berkala Arkeologi "situs permukiman" Hindu-Buddha Jawa Tengah ekskavasi lantai rumah tungku arang pertanggalan radiokarbon
5. settlement archaeology Central Java ninth century domestic habitation excavation Kedu plain Mataram open-air site
6. Ratu Boko ekskavasi permukiman hunian keramik Berkala Arkeologi
7. Berkala Arkeologi Liangan permukiman Mataram Kuno hasil ekskavasi rumah kayu pelataran
8. "Liangan" site Data Variability, Chronology, and Spatial Aspect Berkala Arkeologi
9. Permukiman Mataram Kuno ditemukan di Klaten Kropakan Jatinom BRIN (and two follow-ups on Kropakan journal articles)
10. Candi Kedulan Sambisari Kimpulan ekskavasi permukiman sekitar candi tertimbun lahar Merapi (and a Kedulan "indikasi permukiman" follow-up)
11. situs Wonoboyo Klaten permukiman ekskavasi gerabah
12. Dieng plateau archaeological settlement habitation excavation Banjarnegara Wonosobo 8th century
13. Wisseman Christie settlement Central Java ninth century excavated habitation
14. ekskavasi situs permukiman kuno Magelang temuan sisa hunian abad IX-X (Borobudur surroundings)
15. situs Kalibening / Tegalsari / Sengi / Gunung Wukir permukiman Kedu
16. Gunung Kidul situs permukiman Hindu-Buddha ekskavasi Wonosari
17. Kulon Progo Purworejo situs permukiman Mataram Kuno ekskavasi
18. Gunung Wingko Bantul situs ekskavasi
19. Balong Bayen Purwomartani Kalasan ekskavasi 2018
20. Temanggung situs permukiman kuno selain Liyangan
21. Kendal / Semarang / Kebumen / Purworejo situs permukiman Hindu-Buddha abad VIII-X
22. Bototumpang Kendal situs ekskavasi
23. BRIN / Balai Arkeologi ekskavasi 2024-2025 permukiman Mataram Kuno Sleman Yogyakarta
24. Jurnal Borobudur / Brojonalan Wanurejo permukiman biksu
25. Degroot "Candi, space and landscape" settlement; Miksic/Tjahjono/Degroot ninth-century settlement evidence
26. Reconstruction of the c. 8th-9th century eruption of Mt. Sindoro ... Liyangan (abstract only, see s.5)
27. Naditira Widya / Amerta / Kalpataru ekskavasi situs hunian masa klasik; Forum Arkeologi situs permukiman
28. cagarbudaya.kemdikbud.go.id situs permukiman masa klasik; jogjacagar.jogjaprov.go.id situs permukiman kuno

Journal archive crawl (to avoid relying on the weak OJS search box): complete issue archives on
`ejournal.brin.go.id` for Berkala Arkeologi (84 issues, 828 article titles), Amerta (58 issues, 467 titles),
Kalpataru (24 issues, 129 titles) and Naditira Widya (`/nw`, 19 issues, 153 titles). Titles were filtered by keyword
(permukiman, hunian, ekskavasi, Mataram, klasik, Merapi, and every target kabupaten/site name). Candidate
articles were downloaded as PDF and read: Berkala Arkeologi ids 4480, 4416, 4535, 5257, 5298, 5296, 5302, 5304,
4767, 5177, 4548, 5377, 4198, 4546, 4592, 5133 (Tempursari: East Java, out of region, dropped), 5418 (not read beyond
abstract). Also read: the Liangan 2014 monograph and Noerwidi 2017 / Riyanto 2015 from the repo literature cache,
Dieng 2010 report (NUS ePress HTML sections), Husein et al. 2010 (Kedulan georadar), Masyhudi 2005 (Sambisari),
Hidayatullah et al. 2020 (Kimpulan), Wisseman Christie 1991 (`Indonesia` 52).

Data-base note: `berkalaarkeologi.kemdikbud.go.id` does not resolve from this machine; the same articles are on
`ejournal.brin.go.id/berkalaarkeologi` (DOIs in the CSV use the 10.30883 prefix).

## 4. What was inaccessible or not done

- Google Scholar, UGM/UI thesis repositories, and the SRN Cagar Budaya site records: not directly queryable here;
  general-engine searches returned no settlement-type records from SRN for the target kabupaten.
- BPCB/BPK pages on `kebudayaan.kemdikbud.go.id` (DNS failure from this machine); BKB `Jurnal Konservasi Cagar
  Budaya` article for Brojonalan not located (news only).
- Suhendro et al. 2026, J. Volcanol. Geotherm. Res. (Liyangan eruption reconstruction, charcoal 14C): ScienceDirect
  returned 403; only an abstract-level search summary was seen. It probably holds lab-coded 14C for the burial
  event and is the best next read for Liyangan dating. Not used in any row.
- Primary excavation reports cited but not online: Asmar & Bronson 1973 (Ratu Boko), Kantor SPSP Jateng & UGM
  1990-1991 and Tim Penelitian Wonoboyo 1991, Yuwono 2003b/2008 (Plaosan hydrology/settlement), Nitihaminoto
  2001/2005 (Gunung Wingko), Balai Arkeologi DIY 2018 and BRIN 2023 reports (Balong Bayen, Kropakan).
- Tribun Jogja (Bototumpang) page: HTTP 403. Degroot 2009 (Leiden PDF) located but not read.
- Not searched at all: Wonosobo/Kulon Progo/Purworejo/Kebumen village-level sources beyond the generic queries
  (they returned nothing of settlement type); Kota Yogyakarta (urban; modern overbuild).

## 5. Confidence

- High: Liyangan is a documented, excavated, ceramically and radiocarbon dated 8th-10th c. habitation site with
  post-holes, burned house remains, domestic pottery and food remains, buried under 4-10 m of volcanic deposits
  (quoted ranges differ by source). Caveat: calibrated ages lack lab codes/sigmas in what I could read; one of the
  excavators' own reading is that a stilt-house row may be ritual support.
- Moderate: the 9 UNCLEAR statuses, each driven by missing features, missing absolute dates, or news-only reporting.
- Low: completeness. The register is the set of sites an English/Indonesian web search and a title-level crawl of
  four BRIN journals could surface; the grey literature (Balai Arkeologi laporan penelitian, theses) was not
  reachable. Absence from the register means "not found", not "not recorded".
- Context (Wisseman Christie 1991, `Indonesia` 52): non-religious sites in the Merapi/Perahu uplands were already
  described as "nearly impossible" to locate; Putri et al. 2024 (citing Mundardjito 1993) say commoner
  village-level settlement had not been studied. The thin register is consistent with that, but that consistency
  is not evidence about burial vs survey vs character (see E226 frame, not opened).

## 6. Three biggest gaps

1. A reference set of N = 1. Only Liyangan meets the full definition; the next-best candidates (Kropakan,
   Balong Bayen, Brojonalan, Nepen) are news-only, and the older excavations (Wonoboyo, Borobudur 2012, Ratu Boko)
   rest on ceramics and secondary summaries. Pulling the primary BRIN/Balai Arkeologi reports for Kropakan, Balong
   Bayen and Brojonalan, and the 1990-91 Wonoboyo reports, is the most direct way to turn UNCLEAR rows into YES/NO.
2. No survey-effort or discovery-mode information. Most recorded deposits were found by sand mining, clay digging
   or construction, not by systematic survey, and no source gives the area searched; this is what a detection
   calibration needs and it is absent. Gunung Kidul, Kulon Progo, Purworejo, Kebumen and Kota Yogyakarta contribute
   no 8th-10th c. settlement rows.
3. Absolute dating. Liyangan's radiocarbon ages are reported only as calendar years; Kropakan, Balong Bayen,
   Brojonalan, Borobudur 2012, Wonoboyo's culture layer and Ratu Boko have no radiocarbon in the sources read.
   The Dieng 2010 dates (NZA36576, NZA36404) are the only lab-coded 8th-9th c. dates I found, and they belong to a
   deposit the excavators call pilgrim use.

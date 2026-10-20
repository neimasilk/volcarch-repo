# NusaTemu — sources and first query list (v0.1)

## Source tiers

| Tier | Source | Period | Access | Notes |
|---|---|---|---|---|
| N1 | National online news: Kompas, detik (incl. detikJatim/Jateng), Antara, Tempo, CNN Indonesia, Liputan6, Republika | 2000s–2026 | site search, search engines; GDELT document index for discovery of URLs | retellings are common (the Lasem 2018 find reappeared in 2024) |
| N2 | Regional news: Radar (Jawa Pos group), Tribun regional, Suara Merdeka, local portals | 2010s–2026 | as N1 | more finds, less verification |
| H | Heritage offices: BPK Wilayah (press releases), the national heritage registry (Sistem Registrasi Nasional Cagar Budaya), museum reports | 2010s–2026 | web | authority for period claims |
| J | Journals: *Berkala Arkeologi*, *Naditira Widya*, *Amerta*, *Kalpataru*, *Walennae* (BRIN) | 1980s–2026 | OAI-PMH frame already built in **E225** (1,676 records) | many reports begin with "temuan warga" |
| C | Colonial: *Oudheidkundig Verslag* (OV), *Notulen Bataviaasch Genootschap*, Delpher newspapers | 1860s–1940s | OV OCR corpus in **E091**; Delpher API (terms and GDPR limits to check) | Phase 3 |

## First query list (Indonesian; for Phase 1, frozen in its DESIGN)

Object × verb pairs, each with and without a province name:
- "warga temukan" + {koin kuno, uang kuno, uang kepeng, arca, prasasti, guci, keramik kuno, batu bata kuno, struktur
  bata, manik-manik, kerangka manusia, fosil}
- "ditemukan saat" + {menggali pondasi, menggali sumur, menggali pasir, membajak sawah, menanam, proyek jalan, proyek
  tol, proyek bendungan, normalisasi sungai, penggalian MRT}
- "temuan benda diduga cagar budaya"; "BPK survei temuan warga"; "BPCB tinjau temuan"; "harta karun ditemukan petani"

**Exclusion terms to test in the pilot:** "replika", "pameran", "museum koleksi lama" (exhibition news), "game"
(treasure in games).

## Already in the repo

- Kemendagri desa/kecamatan codes: `data/raw/gazetteer/kemendagri_wilayah_2025_cahyadsn.sql`
- ADM2 boundaries: `data/raw/gazetteer/geoBoundaries-IDN-ADM2_simplified.geojson`
- OV OCR + extraction: `experiments/E091_ov_nlp_mining/`
- BRIN journals frame + verbatim-quote guard: `experiments/E225_gray_literature_mining/`
- Mail and web tooling: `tools/mail/` (not needed here)

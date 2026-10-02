# E226 identification tier I1 - notes (2026-10-02)

Output: `frame/identifications_I1.csv` (17 rows, 13 distinct frame keys). Built by one agent in roughly two hours,
blind to `register/`. **Result in one line: published identifications are scarce; I1 is thin, and is not a usable G on its own.**

## Coverage

| Scope | Names with any I1 row | Share |
|---|---|---|
| Primary frame (primary=True, 170 names) | **11** | 6.5% |
| ... excluding the one tier-C-only name (`poh`) | 10 | 5.9% |
| ... with at least one `stated` (not only `hedged`) row | 6 (mantyasih, vadumpoh, kunim, hinapit, hijo, tajigunum) | 3.5% |
| ... with a named desa/kelurahan | 3 (mantyasih, vadumpoh, kunim; all Magelang cluster, all from one author) | 1.8% |
| Rest of frame (96 non-primary names) | 2 (tlam, salinsinan; both outside Kedu/Prambanan) | 2.1% |
| All 266 frame names | 13 | 4.9% |

Primary names with an I1 row: mantyasih, vadumpoh, poh (tier C), kunim, glamglam, hinapit, hijo, tajigunum, pamramuan,
pandamuan, rukam. The ~70 witness villages of the Poh charter and the ~50 of Lintakan (the bulk of the primary frame)
have **no** I1: I found no published identification for any of them except Hinapit, Hijo, the Pangramwan/Paṇḍamuan pair
and (weakly) Glam-glam.

Source tiers used in the `note` column: **[A]** epigrapher/editor text read directly (here: editors' commentary in the
DHARMA XML, which itself quotes Stutterheim, Nastiti, Soekmono, Sidomulyo); **[B]** an epigrapher/archaeologist's claim
that I could only read as relayed by a web page; **[C]** non-specialist source (local-heritage blogs).
`opened_source` means the cited **original work** was opened: it is `NO` for 14 of 17 rows (Atmodjo, Soekarto, Muchtar,
Stutterheim, Nastiti, Soekmono, Sidomulyo, Damais were all read only through a relay or through the DHARMA commentary that
quotes them; the relay is named at the start of each `note` as "[read via: ...]"). It is `YES` only for `poh` (the heritage
blogs are the cited sources), the Matesih row (IDENK entry) and the Budaya Indonesia Rukam row (the portal entry itself).

## The three most useful sources

1. **Atmodjo (Soekarto K.) 1988, "Sekitar Masalah Hari Jadi Magelang"** - read via a verbatim first-person repost
   (https://sejarahjawaid.blogspot.com/2020/08/prasasti-poh.html) and the Kemendikbud portal
   (https://budaya-indonesia.org/Asal-Usul-Magelang-1). Gives Mantyasih = Meteseh, Wadung Poh = Dumpoh, Kuning Kagunturan
   = Kembang Kuning + Guntur (Rejosari, Bandongan), Glam/Glangglang ~ Magelang. The only source that yields desa-level
   identifications for named frame villages. Caveat: the 1988 study underpins the Magelang anniversary (Perda 6/1989).
2. **DHARMA editors' commentary (XML in `experiments/E023_ritual_screening/data/dharma/xml/`)** - the only primary-quality
   text I could open. Telang II (Stutterheim 1934 vs Nastiti 2015 on Teleng), Tihang (Soekmono vs Hadi Sidomulyo on
   Salingsingan), Mantyasih III (Damais 1970 on paṇḍamuan/paṅramuan), and Hampran (Griffiths: Kusen's modern-map
   identifications of Poh "do not seem particularly convincing"). Useful mostly as a calibration of how contested these are.
3. **Soekarto's Pangramwan/Prambanan argument as relayed by Kompas.com 19 Jun 2025**
   (https://www.kompas.com/stori/read/2025/06/19/210000079/asal-usul-nama-candi-prambanan-dan-dua-versi-cerita-pembangunannya):
   Cepit (Hinapit), Taji (Tajigunung), Ijo (Hijo), Kelurak (Kalumwarak). Secondary and phonetic, but it is the only
   published pointer for the Prambanan half of the frame; the underlying systematic source is **Kusen 1990/91** (below).

## What I searched

- **DHARMA XML, all 269 editions**: commentary, translation, apparatus, bibliography, `<placeName>`/`<geogName>` (only
  Canggu, Padingding, Tirah carry placeName, none in the frame). Scanned for modern-place vocabulary (desa, dusun, kecamatan,
  kabupaten, sekarang, identified, kabupaten names) and, separately, for every frame spelling inside commentary text
  (128 hits, read). Result: the commentary is almost entirely intra-corpus cross-reference; modern identifications occur
  only in Telang II, Tihang, Barahasrama (Serayu), Tulang Er (findspots) and the Hampran remark on Kusen.
- **IDENK metadata xlsx (Part1 and Rest)**: findspot columns (province/kabupaten/kecamatan/desa/dukuh) for the frame
  inscriptions. These are **inscription findspots, not village identifications**, and were deliberately NOT written as I1 rows
  (examples: Rukam plates at Desa Petarongan, Parakan; Tihang at Srumbung; Palepangan at Borobudur; Rumwiga at Srimulyo/Piyungan
  (Machi Suhadi says "desa Payak, Kec. Srimulyo"); Samalagi at Kretek, Bantul; Kasugihan/Luitan at Kesugihan, Cilacap;
  Wanua Tengah III at Dusun Dunglo, Gandulan, Kaloran; Pananggaran/Sumundul at Kedulan, Tirtomartani, Kalasan; Kandangan
  "Gunung Kidul division"). They remain available to I2 (findspot kabupaten) and to the PI as hints.
- **Open journals (ejournal.brin.go.id, full text downloaded and grepped for every frame name)**: Berkala Arkeologi
  "Penetapan sima ... prasasti-prasasti Balitung" (5112), "Prasasti Rumwiga" (Suhadi, 5470), "Prasasti Tulang Er" (Santosa,
  14(2):186-190), Wanua Tengah III papers (5246, 5220, 4621), Poh Sanskrit names (4257), "Desa-desa kuna pantai selatan Jawa"
  (4723), "Emas dan tanah" (5269), AMERTA "Kalang" (5549). They give findspot tables and charter content, **no new village
  identifications**. Not readable: Balitung putra daerah (4657) and AMERTA 15235 (image-only PDFs, no text layer), AMERTA 3410,
  Naditira 5670 (no download).
- **Web**: id.wikipedia pages of ~25 inscriptions (findspots only), Kemendikbud "Budaya Indonesia" entries, Temanggung and
  Magelang government/heritage pages, Tirto, Kompas, ANTARA, Magelang heritage blogs.

## Not recorded as I1 (considered and rejected, so nobody re-derives them)

- Findspots listed above (not identifications).
- **Puluwatu** "now Puluwatu, Kec. Ngaglik / Kalurahan Purwobinangun, Pakem, Sleman" (Budaya Indonesia / id.wikipedia, no
  scholar named): refers to the *watak* Puluwatu in the Panggumulan charter, a different referent from the frame key `puluvatu`
  (village in the Guntur charter).
- **Tegalrukem/Ngrukem** (Tirto, journalist) for Rukam; **Limwung = Klimbungan, Dusun Kauman, Desa Karanggedong, Ngadirejo**
  (Sugeng Riyanto, archaeologist, oral, "baru sebatas dugaan"; Limwung is a sacred building, not a frame village).
- **Liyangan as "the village destroyed in the Rukam charter"** (Nastiti 1982; Wisseman Christie Register; Budaya Indonesia):
  not an identification of Rukam; Muchtar 2014 explicitly separates the two.
- **Kdu = Desa Kedu (Kec. Kedu)**, **Wunut = Dusun Wunut, Desa Wonotirto (Kec. Bulu)** (Atmodjo 1988; Resiyani 2010): not frame
  keys (they occur in `rāma i X` witness contexts outside the village-noun frame, threat 7 in DESIGN). Same for Pikatan (Desa
  Mudal, Temanggung, per Temanggung government pages) and Kalumwarak = Kelurak.
- **Gilikan = Pelikan/Plikon?** - Atmodjo: "Entahlah". **Guntur** (frame key from the Guntur and Humanding inscriptions) not
  linked to Atmodjo's Kagunturan: different occurrence, not shown to be the same place.
- **Wanua Tengah**: no explicit identification found; Temanggung government pages only give the Dunglo/Gandulan findspot.

## Inaccessible (could not be opened) - these would raise coverage most

- **Kusen 1990/91**, "Identifikasi Toponim dalam Prasasti Jawa Kuno Abad IX-X dari Prambanan dan Sekitarnya dengan Toponim
  Masa Kini" (UGM Fakultas Sastra research report; presented at the Trowulan analytic meeting, Nov 1991). The systematic
  I1 source for the Prambanan half. Not online. Ask UGM Archaeology (Dr Kusen's papers) or the BPK Wilayah X library.
- **Atmodjo 1991**, "Toponim Beberapa Nama Desa dalam Prasasti Salingsingan" (paper, epigraphy workshop 9-10 Nov 1991) and
  the original **Atmodjo 1988** paper.
- **Resiyani 2010** (UGM undergraduate thesis, "Toponim masa kini berasal dari sumber prasasti abad IX-X ... Temanggung";
  ResearchGate returned HTTP 403) and her book *Jejak Kata di Bumi Temanggung* (Dewa Publishing, commercial). Systematic for Kedu.
- **Boechari 2012**, *Melacak Sejarah Kuno Indonesia lewat Prasasti*; **Sarkar, CIJ**; **Damais 1970** (as cited in
  the DHARMA commentary, pp. 367-368); **Wisseman Christie, Register of the Inscriptions of Java**; **Nastiti 2003 and 2015**; **Darmosoetopo**;
  **Stutterheim 1927/1934/1940**; **Muchtar 2014** (Liangan volume). Not online or paywalled; only seen as citations.
- Hosts that did not respond from this environment: archive.org (HTTP 503), dharmalekha.info (connection refused),
  surakartadaily.com (DNS), researchgate.net (403), tirto.id via WebFetch (403; reached through curl).

## Methodological flags for the PI (carry into the write-up)

1. **Key collapse.** The frame key merges homonyms across charters (Poh, Hijo, Taji, Kunim, Mantyasih, Paṇḍamuan ...). An
   I1 row applies to one occurrence (stated in each `note`); nothing licenses assigning it to every occurrence. `poh` and
   `hijo` are generic toponyms (mango, green) - Griffiths warns of exactly this.
2. **Precision.** Most I1 rows are kampung/dusun- or area-level (Dumpoh, Meteseh, Cepit, Ijo, "Prambanan"). The E226 desa
   gazetteer carries desa/kelurahan only, so rho = 1 km matching needs a hamlet gazetteer (BIG/OSM) for these.
3. **Motivated and chained identifications.** The Magelang cluster comes from one author, produced for a civic anniversary; the
   Prambanan pairs are phonetic matches reported secondhand; `pandamuan` is a chain of two steps that I assembled.
4. **Circularity (Rukam).** Identification of Rukam is bound up with a hypothesis about an excavated settlement site; do not
   feed it into T7 matching.
5. **I1 is too thin to define G.** With 3 desa-level names out of 170, the bounds S/|V| <= d <= (S+|V\G|)/|V| in DESIGN s.3
   are dominated by I3 unless I2 or a domain expert (see below) supplies identifications. Suggest the PI ask Titi Surti Nastiti,
   Arlo Griffiths or the UGM group for the Kusen and Resiyani lists - one email likely beats further web search.

## Blindness and hygiene log

- `register/` was not opened; I did not search for archaeological sites near any candidate place; the register compiler's
  scratchpad files (`reg/`, `a*.txt`, `b*.py` etc. in the session scratchpad) were not opened.
- Two incidents, disclosed: (i) a mis-targeted search result for the Resiyani thesis returned a Berkala Arkeologi paper on
  the Liyangan settlement (Tanudirjo et al. 2019); I read its abstract and the first lines of the introduction, then deleted
  the file unused. (ii) The Rukam identification literature (Tirto, Budaya Indonesia) discusses Liyangan; I read it only for the
  identification question and flagged the circularity above.
- No identification was invented from name similarity by me. The only assembled step is the `pandamuan` chain, labelled as such.
- Local working copies (downloaded PDFs, scripts) are in the session scratchpad, not in the repo; the CSV cites URLs and
  DHARMA XML paths.

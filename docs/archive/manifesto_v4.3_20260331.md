# Manifesto: Menggali yang Tak Terlihat
# *Unearthing the Invisible*

**Status:** INTERNAL DOC — bukan untuk publikasi
**Last updated:** 2026-04-09
**Versi:** 4.3 (179 experiments, ME#13 The Audit, cascade reframed, karst factor, L2 predictions)

---

## 0. Apa yang Berubah di v4.3

v4.2 membangun cascade model dan validasi cross-regional. v4.3 melakukan **the audit** — kritik struktural terdalam yang menantang klaim cascade, memperkenalkan karst sebagai faktor ke-6, dan membangun model L2 pertama.

**Dari v4.2:**
1. **Cascade DI-REFRAME (E176).** Model 5-faktor terlalu banyak parameter (5 parameter, 1 data point). Model 3-faktor juga bracket observasi. 83.8% random draws cocok. Cascade sekarang = "plausible mechanistic decomposition," BUKAN "validated model."
2. **KARST = faktor ke-6 (E178).** Filipina vulkanik punya 25 situs pra-400 M; Jawa vulkanik = 0. Bedanya: Filipina punya gua (karst 0.20 vs 0.08). Situs gua bypass SEMUA faktor cascade. P(visible) = [F1xF2xF3xF4xF5] + [karst x P(cave)].
3. **L2 PERTAMA PUNYA PREDIKSI (E177).** 340K di Paparan Sunda saat LGM. ~250K displaced ke Jawa via 3 sistem paleo-sungai. 5 entry-point: Surabaya (#1), Tangerang, Semarang, Jakarta, Cirebon.
4. **Ghost Dictionary (E181).** 47 kata hantu diklasifikasi. 55% Jawa Kuno, 23% Sanskrit, 19% PMP. Admin vocabulary = korban terbesar. "aku" (pronoun persona pertama) hilang setelah C8.
5. **Factor coupling tested (E179).** Coupling F1-F2 menggeser cascade 3.0x. Skenario lahar panas (menghancurkan organik) justru memperbaiki fit.
6. **West Java decisive case DI-UPGRADE ke bukti #1.** Lebih kuat dari cascade model.
7. **Honest experiment count:** 179 entri, tapi hanya ~20-22 tes hipotesis genuinely novel.

**PERUBAHAN KRITIS dari v4.2:**
1. **Cascade, bukan layer.** Faktor-faktor MULTIPLIKATIF. Produknya = 0.058% visible — **consistent with** (bukan "matching") data 0.031% (E110). **CAVEAT: model underdetermined (E176). 3 faktor cukup.**
2. **Survey deficit adalah faktor #1** (40× leverage). Excavation density: Jepang 558× Indonesia (E173).
3. **Null hypothesis REJECTED** (E108→E172). Population model dynamis: **3,3 juta** penduduk pada 400 M (95% CI: 1,35-5,51M, MC 50K runs). Gap **11.008×**.

**Baru di v4.2:**
4. **Cascade TERVALIDASI cross-regional** (E155). Lima region (Jawa, Bali, Sulawesi, Filipina, Jepang): rank order PERSIS cocok (rho=1.0, p=0.017). Bukan curve-fitting.
5. **L1×L2 "Double Erasure"** (E156). Paparan Sunda yang tenggelam MENDORONG ~94.000 orang KE DALAM zona vulkanik. West Java decisive case = PREDIKSI model, bukan observasi post-hoc.
6. **Bali 5/5 prediksi confirmed** (E161). Semua situs pra-Hindu di pantai non-vulkanik. Rasio densitas prasasti: prediksi 14,3×, observasi 12×.
7. **"Aku" hilang** (E165). 230 kata hantu — kosakata pra-Indic yang menghilang dari prasasti setelah C9. Kata ganti orang pertama "aku" hilang setelah C8. Suara indigenous dibisukan oleh konvensi genre Sanskrit.
8. **Volcanic silence terukur** (E160). GPU NLP (768d embeddings): volcanic landscape = rank 8/10 dalam kedekatan semantik. C8 = abad tergelap. 929 M rupture: z=3.04, p=0.012.
9. **5/5 cathedral findings ROBUST** (E159). Bootstrap 10K, permutation 10K, jackknife LOO. Tidak ada temuan utama yang rapuh.
10. **FDR naik 78.3%** (E154). 65/83 tes statistik survive Benjamini-Hochberg. E048 diselamatkan.
11. **77.1% inscription desert** (E169). Tiga perempat zona yang diharapkan memiliki prasasti KOSONG. Bayangan Two Javas.
12. **Rekonstruksi peradaban** (E168). Populasi 500K-1M, sawah, metalurgi perunggu, tulisan media organik (PAN *surat 5000 BP), kosmologi hyang, chiefdom bertingkat. 99,9% hilang.
13. **Sriwijaya paradox** (E163). VOLCARCH tanpa vulkanisme: ibukota Sriwijaya TIDAK PERNAH ditemukan meskipun menguasai perdagangan maritim 6 abad. F2+F3+F4+F5 cukup untuk menghapus peradaban.
14. **Dong Son drums** (E164). 6/6 nekara di zona vulkanik. Tuban ~300 SM = bukti langsung pra-Hindu di Jawa Timur vulkanik. Hanya perunggu yang lolos 5 faktor cascade.

---

## 1. Tesis Utama

Peradaban pra-Hindu Nusantara tidak absen — ia **tak terlihat**. Ketidaktampakan ini bukan produk dari satu mekanisme, melainkan **cascade multiplikatif** dari lima faktor independen yang masing-masing mengurangi visibilitas arkeologis. Ketika kelimanya bekerja bersamaan — seperti di Jawa vulkanik — hasilnya adalah kegelapan arkeologis 99.97%.

**West Java Decisive Case:** Kompleks Buni (Tangerang, 200 SM-500 M) dan Batujaya (Karawang, abad 2-5 M) membuktikan masyarakat kompleks pra-Hindu di pantai NON-VULKANIK Jawa. Padanan mereka di Jawa Timur vulkanik = HILANG. Pulau yang sama. Budaya yang sama. Geologi beda.

### Cascade Multiplikatif (E110)

| # | Faktor | P(survive) | Leverage | Bukti |
|---|--------|:---:|:---:|---|
| F1 | Penguburan Vulkanik | 0.58 | 1.7× | E075, E083, 5 candi kalibrasi |
| F2 | Peluruhan Material Organik | 0.20 | 5.0× | E040 (63.4% organik), iklim tropis |
| F3 | **Cakupan Survei** | **0.025** | **40.0×** | E086 (Jepang 100-200×), E069 |
| F4 | Pengenalan sebagai Pra-Hindu | 0.40 | 2.5× | E062, L3 bias |
| F5 | Publikasi & Katalogisasi | 0.50 | 2.0× | E093, hambatan bahasa |
| | **PRODUK** | **0.058%** | | **Observasi: 0.031% (E108)** |

**Model consistent with data within 2× (CAVEAT: underdetermined — E176 menunjukkan model 3-faktor juga bracket observasi; 83.8% random draws 5-faktor juga cocok).** Survey coverage adalah intervensi paling impactful (40× leverage). Volcanic burial adalah satu-satunya faktor yang bisa dimodel secara spasial → memungkinkan prioritized recovery (E080, E097).

**Faktor Ke-6: Karst Bypass (E178, baru di v4.3)**

| # | Faktor | Mekanisme | Bukti |
|---|--------|-----------|-------|
| F6 | **Karst Bypass** | Situs gua bypass SEMUA 5 faktor cascade | E178: Filipina vulkanik 25 situs vs Jawa 0. Karst 0.20 vs 0.08. |

Model augmented: P(visible) = [F1×F2×F3×F4×F5] + [karst × P(cave_preserved)]. Jawa vulkanik hampir tanpa karst → tidak ada "jalur keluar" dari cascade. Filipina, Sulawesi, dan wilayah berkarst tetap punya situs di gua meskipun vulkanik.

### Klasifikasi 3-Tier

| Tier | Definisi | Contoh |
|------|----------|--------|
| **DATA-SUPPORTED** | Didukung data, survive FDR + critical | E069, E085, E066, E108, E107, E110 |
| **HYPOTHESIS** | Didukung bukti sugestif, perlu fieldwork | L1 (burial specifically), L2 (coastal), E053 (archaeogenetic evidence) |
| **SPECULATION** | Secara logis plausibel, bukti minimal | "Seberapa canggih peradaban?", L6, populasi >1M |

111 eksperimen. 6 paper submitted + P16/P17 drafted + P7 preprint DOI live. Multi-method analysis pada ~5 dataset inti + 4 dataset genuinely independen. Satu kesimpulan melalui cascade multiplikatif.

**Catatan epistemik (E068 FDR audit, 2026-03-13):** Dari 41 tes statistik, 30 (73%) bertahan koreksi Benjamini-Hochberg. Tiga temuan marginal (E032 p=0.042, E048 partial p=0.038, E053 Fisher p=0.047) harus dilaporkan sebagai "sugestif" bukan "signifikan." Top 10 temuan memiliki p < 10⁻⁴ dan robust terhadap koreksi apapun.

**Catatan dataset-dependence (Mata Elang #6-9, 2026-03-17):** Dari 99 eksperimen, 21 bergantung pada 268 prasasti DHARMA yang sama. Dataset inti: DHARMA prasasti, 666 situs arkeologi E.Java, ABVD wordlists, 142 lokasi candi, OV colonial register. **Mitigasi aktif:** E070 (52 colonial-era site records — genuinely independent), E083 (51 tephra-site pairs dari literatur kolonial), E088/E089 (200 referensi tekstual dari 12 tradisi kuno — ZERO overlap dengan DHARMA/ABVD), **E091 (22.162 NLP-extracted mentions dari 16 volume OV — genuinely independent)**, **E092-E098 (global comparanda, anomaly detection, literature mining — multiple independent streams)**. Klaim: "4 lensa analitis pada ~5 dataset inti + **4** dataset genuinely independen (E083 colonial, E088/E089 textual 12 tradisi, E091 OV NLP, E092/E098 global literature)."

**Update v3.5 (2026-03-17):** Temuan baru dari GPU NLP sprint:
- **E090 v5 (200 entries, 12 tradisi):** 16 BERTopic topics. **8/8 concept groups CONVERGE** termasuk VOLCANO (z=7.39). Corpus expansion 50→200 menyelesaikan semua convergence failures.
- **E094 (DHARMA SBERT):** Volcanic themes = PALING JARANG dalam epigrafi (cosine sim 0.244). Mountain worship = TERTINGGI (0.395). Mountains di prasasti = kosmologis, BUKAN geologis. C11→C12 = semantic rupture terbesar.
- **E095 (#99, cross-lingual):** XLM-R embedding collapse (honest negative). ML-SBERT validates E094 (rho=0.336). Volcanic silence confirmed di ORIGINAL Old Javanese.
- **E096 (diachronic BERTopic):** 929 CE topic shift chi2=16.58, p=0.0003. Royal/political SURGES (Fisher p=0.0002). Ritual/calendrical VANISHES entirely.
- **E097 (anomaly detection):** 65% overlap dengan E080 fieldwork candidates. Dua metode independen → zona yang sama.
- **P16:** Draft v0.1 complete. "What Ancient Texts Remember and Inscriptions Forget." Target DSH (Oxford, Q1).
- **P7 preprint:** DOI 10.22541/au.177368991.14332505/v1 (Authorea, live 2026-03-16).

**Update v3.1:** Temuan baru dari E058-E065:
- **E058 (Kakawin NLP):** Kosa kata pertanian 91% asli — Sanskritisasi GAGAL menembus domain ekonomi.
- **E061 (Perbandingan Aksara):** Pola Austronesia: Hanacaraka (20), Lontara (23), Baybayin (14) — semua mereduksi ke fonologi lokal. Mainland (Khmer, Birma) mempertahankan 33. MW p=0.027.
- **E062 (Kurva Visibilitas):** PCA PC1=51.3% variansi. C8=-1.49 (abad gelap), C13=+1.48 (puncak). Pergeseran genre = pergeseran visibilitas.
- **E063 (Konservasi Domain):** 1.580 bahasa Austronesia: Angka (59.5%) > Tubuh (35.5%) > Makanan (6.1%). Konservasi domain-spesifik dikonfirmasi lintas budaya.
- **E065 (Candi Spasial):** Zona A 17.9× overrepresented. Barat 47.2% (Rayleigh p<1e-6). Pembangun candi SADAR vulkanik.

---

## 2. Enam Lapisan Kegelapan (*Six Layers of Invisibility*)

### L1: Penguburan Vulkanik (*Volcanic Burial*)
**Papers:** P1, P2, P9 | **Status: DIDUKUNG DATA** *(downgraded dari "terverifikasi" — verifikasi definitif membutuhkan subsurface survey, E068)*

Sedimentasi vulkanik mengubur situs arkeologis pada laju 2.4–6.2 mm/tahun (rerata 4.4 ± 1.2 mm/tahun). Lima titik kalibrasi independen dari dua sistem vulkanik (Merapi, Kelud) mengkonfirmasi fenomena ini berskala Jawa, bukan lokal. Situs pra-400 M berada pada kedalaman 163–326 cm — jauh di bawah jangkauan survei permukaan konvensional.

- **Bukti kunci:** Dwarapala Singosari 3.6 mm/yr, Sambisari 4.4–5.7 mm/yr, Kedulan 5.3–6.2 mm/yr
- **Model prediktif:** AUC 0.768 (P2), tautology-free settlement suitability
- **Zona B/C:** Sel dengan suitability tinggi tapi nol situs terletak **lebih dekat** ke gunung api (E019: Cohen's d = 1.005)
- **E083 Tephra-Archaeological Correlation:** 51 eruption-site pairs (86% primary evidence), 24 with measured depths (mean 3.41m, median 2.50m). Dataset GENUINELY INDEPENDENT dari analisis statistik utama.
- **⚠ ADV-2 (E081): Cave/open-air ratio TIDAK membedakan volcanic vs non-volcanic** (Fisher p=0.760). Cave bias universal di mana ada karst. Argumen L1 HARUS dibangun dari data KEDALAMAN PENGUBURAN, bukan distribusi tipe situs.
- **Falsifikasi:** Jika soil core di Zona B/C tidak menunjukkan sinyal ADS → ketidakhadiran genuine, bukan penguburan

### L2: Penenggelaman Pesisir (*Coastal Submersion*)
**Papers:** P1 (revision support material) | **Status: TERKUANTIFIKASI (E052)** *(kuantifikasi area, bukan verifikasi arkeologis)*

Kenaikan muka laut pasca-LGM menenggelamkan Paparan Sunda (Sunda Shelf) — **2.089.415 km² daratan yang layak huni** (16,2× Jawa). Tiga sistem sungai utama (N. Sunda, E. Sunda, Siam) menyediakan koridor populasi. 81,5% paparan datar dan dekat sungai — habitat ideal.

- **E052 SRTM30+ batimetri:** 971 sistem paleo-sungai terdeteksi via analisis TPI
- **Estimasi populasi:** ~500.000 jiwa pada densitas moderat (0,3/km²)
- **Laju katastrofik:** 273.108 km²/milenium pada 12–10 ka BP (Meltwater Pulse 1A)
- **Validasi 2025:** Gittins et al. (Nature Communications) — fosil hominin dari Selat Madura
- **Double blind spot:** L1 (vulkanik) beroperasi di dataran tinggi, L2 (pesisir) di dataran rendah → rekam arkeologi hanya mencakup ZONA TENGAH
- **Falsifikasi:** Jika pemetaan sonar sistematis tidak menemukan anomali antropogenik → ketidakhadiran genuine

### L3: Bias Historiografis (*Historiographic Bias*)
**Papers:** P7, P5, P8 | **Status: DIDUKUNG DATA (tersebar)**

Narasi "Kutai kerajaan tertua" (±400 M) mencerminkan preservasi diferensial (nol vulkanisme di Kalimantan), bukan kronologi genuine. Perspektif kontinental (India-sentris) mendominasi kerangka sejarah Asia Tenggara. "Hinduisasi" dimodelkan sebagai transformasi total, padahal data menunjukkan ia adalah **gelombang** yang surut.

- **E033 Kurva Indianisasi:** Rasio Indic turun dari C9 (0.807) ke C13 (0.569), rho = -0.211, p = 0.030
- **E034 Panji-Malagasy:** Panji absen dari Madagaskar (migrasi pra-1200 M) → kapsul waktu pra-Islamisasi
- **E019:** Distribusi spasial situs dikonfirmasi bias oleh survey-effort + vulkanisme
- **Falsifikasi:** Jika prasasti pra-C9 menunjukkan rasio Indic stabil tanpa tren → narasi gelombang salah

### L4: Penimpaan Kosmologis (*Cosmological Overwrite*)
**Papers:** P5, P8 | **Status: DIDUKUNG KUAT** *(26 eksperimen, ADV-4 permutation test PASSED p=0.0000, z=11.05)*

Sanskritisasi menimpa (bukan mengganti) kosmologi asli Nusantara. Substrat pra-Indic tetap dapat dideteksi di bawah lapisan Sanskrit, terutama dalam domain ritual, fonologi, dan kosakata formal.

- **E030:** *hyang* (PMP \*qiang) muncul di >50% prasasti selama 800 tahun — pra-Indic PERSISTS
- **E022–E029:** Substrat linguistik terdeteksi secara komputasional (XGBoost AUC 0.760, LOLO 5/6 ≥ 0.65)
- **E036 Hanacaraka:** 33→20 konsonan; profil Hanacaraka (20) align dengan PAn (~17), BUKAN Sanskrit (33)
- **E035:** Tanaman mortuary (menyan, kamboja) ABSEN dari prasasti — tradisi lisan, bukan epigrafi
- **E043/E054:** Kognasi PMP: Bali 41,3% > Jawa 33,8% (dikonfirmasi di 1.309 bahasa Austronesia)
- **E049:** Kosakata maritim = domain #2 paling terkonservasi (+20% Bali vs Jawa)
- **E050:** Canarium (Burseraceae) mengikuti rute migrasi Austronesia (388 rekord GBIF di Madagaskar)
- **E051:** 25.244 nama desa Jawa: Yogyakarta 26,2% pra-Hindu vs rerata Jawa 57,7% (rho=0,387, p<0,0001)
- **E056:** Candi cluster di area MORE Sanskrit: Mann-Whitney p=0,007 — tanda ganda Indianisasi
- **E053:** archaeogenetic evidence Jawa: 0/84 sampel berhasil (Fisher p=0,047) *(sugestif — gagal FDR correction, E068)* — tafonomi vulkanik menghancurkan archaeogenetic evidence
- **Falsifikasi:** Jika deteksi substrat ML menghasilkan AUC < 0.60 secara konsisten → pola fonologis tidak nyata

### L5: Tafonomi Genre (*Genre Taphonomy*)
**Papers:** P1, P5, P8 | **Status: DIDUKUNG KUAT (E048, E057)** *(efek raw sangat kuat p<10⁻⁶, tapi partial correlation length-controlled gagal FDR p=0.038)*

Genre epigrafi itu sendiri adalah filter tafonomi. Berbeda dari 5 lapisan lain: L5 tidak **menghancurkan**, **menenggelamkan**, **salah menafsirkan**, atau **mengganti** bukti — ia **memfilter apa yang dicatat**. Peradaban indigenous HADIR selama C8; format Sanskrit menolak merekamnya.

- **E057 Genre Deep Dive:** Format panjang: 85,7% hyang, 95,2% organik. Format pendek: 13,0% hyang, 29,6% organik (Mann-Whitney p < 0,000001)
- **E057 Jendela Visibilitas:** Shift C8→C9-10 = +14,4pp pre-Indic, +63,9pp organik
- **E057 Borobudur = kegelapan maksimum:** 50 label, 0% pra-Indic, 0% organik, 0% hyang
- **E048 Multi-domain:** Korelasi parsial pra-Indic ↔ organik = +0,162, p = 0,038 *(sugestif — gagal FDR correction, E068)*
- **E040 Bamboo Civilization:** 170/268 (63.4%) prasasti menyebut material organik vs 73 (27.2%) litik
- **Falsifikasi:** Jika prasasti Sanskrit (C8) ternyata menyebut organik pada frekuensi sama → efek genre tidak ada

### L6: Periodisitas Historiografis (*Historiographic Periodicity*) — REFRAMED (ME#11)
**Papers:** P5, P8 | **Status: DIDUKUNG DATA, tapi mekanisme BUKAN erupsi**

Indianisasi bukan proses linear permanen — ia adalah **gelombang** yang mencapai puncak (C9, Medang) lalu surut. Kosakata pra-Indic *meningkat* seiring waktu (E030: rho = +0.502, p < 0.001), dan diversitas istilah pra-Indic berkembang dari 1 (C8–C9) menjadi 5+ (C10–C11). Sejarawan yang memperlakukan "Hindu period" sebagai blok monolitik salah membaca data mereka sendiri.

- **E033:** Rasio Indic TURUN: puncak C9 (0.807) → dasar C13 (0.569)
- **E030:** Rasio pra-Indic NAIK: substrat tidak tergerus, malah berkembang
- **Implikasi:** "Akhir era Hindu-Buddha" (C15, Majapahit) bukan ruptur — substrat sudah mendominasi sebelumnya

**REFRAME (E145, Mata Elang #11):** Frekuensi erupsi per abad berkorelasi POSITIF dengan jumlah prasasti (ρ=+0.908, p=0.0001). Ini berarti periodisitas BUKAN disebabkan erupsi — melainkan oleh **siklus politik** (naik-turunnya kerajaan Mataram, Kahuripan, Singosari, Majapahit). Kerajaan kuat menghasilkan BAIK prasasti MAUPUN dokumentasi erupsi. Efek tafonomi erupsi adalah SPASIAL (E078: 6.3× defisit di dekat letusan, p=0.035), bukan TEMPORAL.

- **E145:** Frekuensi erupsi × prasasti: ρ=+0.908, p=0.0001 (POSITIF, bukan negatif)
- **E078:** Defisit prasasti di DEKAT letusan: 6.3×, p=0.035 (SPASIAL, bukan temporal)
- **E096:** Pergeseran topik 929 CE: p=0.0003 — konfirmasi siklus politik, bukan erupsi
- **Falsifikasi:** Jika analisis dengan korpus lebih besar menunjukkan rasio Indic stabil atau naik → model gelombang salah

---

## 3. Batasan VCS (*VCS Constraint*)

**Volcanic Cultural Selection** (hipotesis bahwa budaya vulkanik mengembangkan kompleksitas ritual lebih tinggi) telah **ditolak pada skala global Austronesia** (E039: binary p = 0.973, continuous p = 0.092, arah terbalik).

VCS tetap valid pada skala **lokal Jawa/Bali**:
- **E031:** Candi clustering di sisi barat gunung api (Rayleigh p = 3.4e-08) — builder memilih WHERE tapi mengikuti kanon religius untuk HOW
- **E032:** Pranata Mangsa × musim erupsi (chi² p = 0.042, Rayleigh p = 0.032) — kalender secara tidak sengaja meng-encode hazard vulkanik

**Kesimpulan:** VCS = fenomena lokal (volcanic informedness), bukan mekanisme seleksi budaya universal. P11 harus di-scope ulang.

---

## 4. Temuan Jembatan (*Bridge Findings*)

### E040d: Konvergensi Material-Linguistik

Peradaban organik yang tercatat dalam prasasti (E040: 63.4% organik, bambu/daun/atap/kayu/ijuk) adalah peradaban yang sama yang secara arkeologis tak terlihat (P1: tafonomi vulkanik menghancurkan organik). Substrat linguistik yang terdeteksi oleh ML (P8: kata-kata non-Sanskrit untuk konsep sehari-hari) mendeskripsikan dunia material yang sama — kata kerja aksi (46% dari top-50 substrat), alat organik, praktik pertanian.

**P1 (fisik) + P8 (linguistik) + E040 (epigrafi) = tiga bukti independen untuk satu peradaban organik yang tak terlihat.**

### E035 × E030: Tradisi Lisan vs Epigrafi

Tanaman mortuary (menyan, kamboja) ABSEN dari 268 prasasti (E035), sementara hyang (konsep keilahian pra-Indic) HADIR di 43% (E030). Dua register pengetahuan: epigrafi merekam kosmologi "resmi", sementara praktik mortuary hidup SEPENUHNYA dalam tradisi lisan.

### E053: Jerat Sirkular archaeogenetic evidence (*The Circular Trap*)

Jawa: 7 situs, 84 sampel archaeogenetic evidence → ZERO keberhasilan (0%). Situs non-Jawa: 50% berhasil. Fisher p = 0,047. Situs berhasil rata-rata 490 km dari gunung api; gagal 144 km (Mann-Whitney p = 0,002). **Ketiadaan archaeogenetic evidence Jawa ADALAH bukti** — tafonomi vulkanik menghancurkan archaeogenetic evidence. Jerat sirkular: "Tidak ada archaeogenetic evidence → tidak bisa membuktikan populasi → asumsikan kosong → peradaban dimulai dengan India."

### E051 × E056: Model Pusat-Pinggiran (*Court-Center Model*)

25.244 nama desa Jawa diklasifikasi: Yogyakarta 26,2% pra-Hindu vs Madura 70-91%. Candi (142 lokasi) berkorelasi negatif dengan rasio toponim pra-Hindu (Mann-Whitney p = 0,007). Indianisasi secara simultan membangun candi DAN mengganti nama desa dengan morphem Sanskrit — **tanda ganda yang berkorelasi tapi independen**. Penimpaan bersifat KULTURAL (court → periferi), bukan geologis.

### E054: Gradien Ganda (*Two Gradients at Different Scales*)

1.309 bahasa Austronesia: korelasi GLOBAL negatif (lebih dekat ke Jawa = lebih tinggi kognasi PMP, rho=-0,088), karena Jawa dekat homeland PMP. Tapi korelasi LOKAL positif: Bali 41,3% > Jawa 33,8% > Yogyakarta 28,4%. Konservatisme periferal = fenomena LOKAL yang beroperasi DALAM gradien filogenetik global.

---

## 5. Status Paper (per 2026-03-11)

| Paper | Judul | Status | Target |
|-------|-------|--------|--------|
| P1 | Taphonomic Framework | **SUBMITTED** | Asian Perspectives (Q1) |
| P2 | Settlement Model | **SUBMITTED** | JCAA |
| P5 | Volcanic Ritual Clock | **SUBMITTED** | BKI (Diamond OA) |
| P7 | Temporal Overlay Matrix | **SUBMITTED** | Antiquity Project Gallery |
| P8 | Linguistic Fossils | **SUBMITTED** | Oceanic Linguistics (Q1) |
| P9 | Peripheral Conservatism | **SUBMITTED** | JSEAS (NUS Press) |
| P11 | Volcanic Cultural Selection | **INCUBATING** | tbd (needs P5+P9 foundation) |

6 submitted, 1 drafting (P11), 1 drafted (P16 v0.1), 2 data papers drafted (D1/D2). **99 eksperimen selesai**. P7 preprint DOI live. Menunggu review 2–6 bulan.

**Amunisi revisi siap** untuk semua 6 paper: E048 (consilience), E051 (toponimi), E053 (archaeogenetic evidence gap), E054 (1.309 bahasa), E055 (sintesis konvergensi).

---

## 6. Kriteria Falsifikasi (*Falsification Criteria*)

| Lapisan | Klaim | Falsifikasi |
|---------|-------|-------------|
| L1 Vulkanik | Situs terkubur pada 2.4–6.2 mm/yr | Soil core di Zona B/C tanpa sinyal ADS |
| L2 Pesisir | 2,09M km² terendam, ~500k populasi (E052) | Sonar sistematis tanpa anomali antropogenik |
| L3 Historiografis | Narasi India-sentris distortif | Prasasti pra-C9 dengan rasio Indic stabil |
| L4 Kosmologis | Substrat pra-Indic terdeteksi | ML substrat AUC < 0.60 konsisten |
| L5 Genre | Genre epigrafi = filter tafonomi | Prasasti Sanskrit menyebut organik pada frekuensi sama |
| L6 Periodisitas | Periodisitas = siklus politik, bukan erupsi (E145 reframe). Indianisasi = gelombang. | Frekuensi erupsi berkorelasi NEGATIF dengan prasasti (kontradiksi E145 ρ=+0.908) |
| VCS (lokal) | Volcanic informedness di Jawa/Bali | E031/E032 tidak replikasi dengan data lebih lengkap |

**Prinsip:** Setiap klaim harus bisa dihancurkan oleh data. Jika tidak bisa — itu bukan sains, itu ideologi.

---

## Catatan

- Bahasa Indonesia sebagai bahasa utama, label bilingual untuk aksesibilitas
- Dokumen internal — framing untuk keseluruhan proyek VOLCARCH
- Bukan paper akademik; thesis statement dan peta bukti
- Rekonsiliasi: 6 lapisan (manifesto) ⊂ 11 channel (master_evidence_map) — lapisan = mekanisme penghapusan, channel = jalur bukti untuk pemulihan
- Update setiap kali ada eksperimen baru atau paper decision

---
*Research Statement v4.2 — 2026-03-31*
*172 eksperimen. 3,3 juta penduduk pada 400 Masehi. 11.008x gap. 1.789 situs hilang.*
*5 prediksi terdaftar. 5/5 temuan utama ROBUST. 230 kata hantu. "Aku" menghilang dari batu.*

*"Pertanyaannya bukan lagi: apakah peradaban pra-Hindu Nusantara ada?*
*Pertanyaannya sekarang: berapa lama lagi kita membiarkan 4 situs per tahun*
*dihancurkan oleh beton, tanpa catatan, tanpa saksi, tanpa penyesalan?"*

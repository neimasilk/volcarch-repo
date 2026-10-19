# Audit hilir C050 — E087, E107, E130, kartu skor ADV-5, jawaban obyektif baris 180 (2026-10-19)

**Asal:** ledger C050 (dibuka 2026-10-05, terbuka sejak itu; disebut di handoff 6, 7, 8, 12 dan 15 Okt sebagai tugas orbit).
Dikerjakan pengendali sendiri, tanpa subagen. Semua angka di bawah diturunkan ulang dari berkas, kecuali yang ditandai
"dibaca, tidak diturunkan ulang". Naskah P8 R1 yang disubmit (`P8_revision_v0.4.md`) **tidak** mengutip E087, E107, E130,
"Mon-Khmer" atau "438 substrate words"; angka 438 di sana adalah hitungan bentuk tanpa kode, dengan nama yang benar.
Pracetak arXiv v1 masih memuat "438 substrate candidates" dan kalimat Acehnese–Mon-Khmer; v2 baru sesudah keputusan jurnal.

**Cacat yang dicari:**
- **K1** — "tanpa kode di ABVD" dibaca sebagai substrat (C044, C055, C056).
- **K2** — masukan tingkat bahasa, atau masukan bentuk yang membawa identitas daftar (C047, C049, C053).
- **K3** — perbandingan antarbahasa dibaca sebagai tanda substrat (baru, E107).
- **K4** — label ikut menjadi masukan; prediksi in-sample (baru, E130).
- **K5** — rumusan lama hasil negatif P8 (C048, C058).

| Butir | Vonis | Cacat | Ledger |
|---|---|---|---|
| E087 | **BERTAHAN, dipersempit** — diagnosisnya ("artefak dokumentasi") kini bacaan yang berlaku | K1, K2 | C050 |
| E107 | **VONIS DITARIK** — data E107 sendiri berbicara melawannya | K1, K3 | **C074** |
| E130 | **METODE TIDAK SAH** — 438 "kata substrat" = 438 bentuk tanpa kode, persis | K1, K4 | **C075** |
| Kartu skor ADV-5 | "RESOLVED by E107; E027 upgraded" → **kembali GREY ZONE**; E107 keluar dari daftar "katedral" | — | C074 |
| `OBJECTIVE_ANSWER_20261001.md` baris 180 | rumusan diganti | K5 | C058 |

---

## 1. E087 — kontrol negatif detektor substrat (2026-03-16)

**Yang diperiksa:** README, `negative_control_pure_phon.py` (masukan dan skema CV).

- Labelnya label 438: bentuk tanpa nomor himpunan kognat = "residual" (438 dari 1.357, 32,3 %). K1: nama "substrat"
  tidak sah. Namun README sendiri sudah menyimpulkan bahwa detektor **tidak bisa** membedakan substrat dari bentuk yang
  tidak dikodekan.
- Lengan "Pure Phonology" **bukan fonologi saja**. Ia memuat dummy huruf awal (`init_*`), masukan makna (`sem_*`) dan
  `is_core_vocab` (skrip baris 272–283); yang dibuang hanya masukan tingkat bahasa. CV-nya `StratifiedKFold` acak,
  sehingga bentuk dari daftar yang sama ada di lipatan latih dan uji. Menurut C053, masukan bentuk membawa identitas
  daftar. Karena itu AUC 0,727 (dan 0,761 / 0,766) tidak sebanding dengan angka *out-of-fold* P8 R1 (E228 S7: bentuk
  saja 0,672). **Jangan dikutip sebagai pengganti angka R1.**
- Uji permutasinya (z = 10, p < 0,001) hanya menunjukkan bahwa masukan meramal label lebih baik dari kebetulan. Itu
  termasuk efek daftar asal; ia tidak menunjukkan substrat.
- Tabel pembanding di `negative_control_pure_phon.py` baris 427–446 mengetik tangan nilai dua lengan lain ("values from
  previous runs"). Nilai itu ada di JSON masing-masing; selisih tidak diperiksa (dibaca, tidak diturunkan ulang).
- **Yang bertahan:** temuan C5 (Iban + Malay, 0,713 tanpa harapan substrat) dan diagnosis README, yaitu label
  mencampur "tidak terdokumentasi" dengan "substrat". Diagnosis ini sejalan dengan semua koreksi P8 sesudahnya (C054,
  C056, C063) dan kini juga dengan audit E107 (§2). Bagian "Honest framing for P8" di README digantikan oleh rumusan
  R1: "a ranking aid, not a detector of substrate vocabulary".
- "Phonological fingerprint (longer forms, more consonant clusters, more glottal stops)" hanya berlaku sebagai deskripsi.
  Panjang: ya. Tanda glotal bertahan pada jumlah huruf yang sama, dengan catatan Tolaki (C066). Gugus konsonan
  diturunkan (C064).

**Tindakan:** catatan koreksi di README; status tetap CONDITIONAL.

## 2. E107 — "ADV-5 re-examination: Mon-Khmer substrate" (2026-03-17)

**Yang diperiksa:** README, `adv5_reexamination.py` (semua 5 uji dan aturan skor), `results/e107_results.json`.
**Cek audit (post hoc, aturan baca ditulis sebelum dijalankan):** `audit_c050_within_list_20261019.py` →
`results/audit_c050_within_list_20261019.json`.

**Cacat desain (K3).** Kelima kriteria skor membandingkan bentuk tanpa kode C5 dengan bentuk tanpa kode **Sulawesi**
(skrip baris 435–481). Akibatnya, setiap bahasa yang fonotaktiknya berbeda dari Sulawesi mendapat skor "Mon-Khmer".
Alternatif "artefak dokumentasi" dirumuskan sebagai "profil C5 sama dengan Sulawesi", padahal artefak dokumentasi
meramalkan bentuk tanpa kode mirip **bahasanya sendiri**. Ini lawan tanding yang dibuat-buat.

**Data E107 sendiri:**
- 67 dari 73 bentuk tanpa kode C5 adalah Iban; Malay hanya punya **6**. Jadi "C5 tanpa kode" praktis = "bentuk Iban".
- Satu-satunya uji di dalam daftar dalam skrip itu adalah Fisher bentuk-MK, C5 tanpa kode vs C5 berkode. Hasilnya
  **OR 0,79, p = 0,41** (tersimpan di JSON). Uji ini tidak ikut diskor dan tidak disebut README.
- Ramalan "lebih banyak gugus konsonan" dinyatakan di docstring, tetapi arahnya terbalik (C5 0,38 vs Sulawesi 0,48;
  p = 0,14). Ramalan itu tidak ada di tabel README, yang kemudian berbunyi "All six MK predictions confirmed".
- Daftar "known MK loans" memuat *bulan*, *tanah* dan *padi*. Ketiganya lazim direkonstruksi sebagai warisan PMP
  (\*bulan, \*taneq, \*pajay) — **tanda domain, butuh ahli**. Di data, *bulan*, *tanah* dan *kutu* justru **berkode**
  sebagai kognat Austronesia; irisan dengan bentuk tanpa kode kosong.
- Rujukan "Iban has well-documented Mon-Khmer (Aslian) substrate (Adelaar 1985, 1992, 2005; Blust 2010)" tidak
  diverifikasi (C060-jenis; butuh ahli).

**Cek di dalam daftar, Iban tanpa kode (67) vs Iban berkode (208), dengan pemuat dan fungsi ciri E107 tanpa perubahan:**

| Ciri | ramalan E107 | tanpa kode | berkode | p (MWU) | terpenuhi? |
|---|---|---|---|---|---|
| suku kata | lebih sedikit | 2,00 | 1,96 | 0,55 | tidak |
| berakhir vokal | lebih sedikit | 0,239 | 0,332 | 0,15 | tidak (n.s.) |
| untai mirip prefiks | lebih sedikit | 0,134 | 0,159 | 0,63 | tidak |
| gugus konsonan | lebih banyak | 0,358 | 0,120 | 0,0004 | **ya** |
| bentuk "MK" (1 suku, atau 2 suku ≤ 5 huruf) | lebih banyak | 0,701 | **0,861** | 0,003 | **terbalik** |
| panjang (huruf) | lebih pendek | 5,28 | 4,75 | 0,003 | terbalik |

- **1 dari 4 ramalan terpenuhi → menurut aturan yang ditulis sebelum dijalankan, vonis E107 tidak berdasar pada datanya
  sendiri.**
- Bentuk Iban yang **berkode** bahkan lebih "Mon-Khmer" menurut kriteria E107 daripada bentuk Sulawesi tanpa kode:
  bentuk-MK 0,861 vs 0,395, berakhir konsonan 0,668 vs 0,215, untai prefiks 0,159 vs 0,370, suku kata 1,96 vs 2,57.
- Jadi kriteria itu mengukur **bahasa** (fonotaktik Melayik), bukan ketiadaan kode.
- Deskripsi tanpa uji: bentuk Iban tanpa kode lebih panjang, lebih banyak gugus konsonan, lebih banyak tanda glotal
  (0,209 vs 0,096; apostrof ortografis ikut terhitung). Arahnya sama dengan bentuk Sulawesi tanpa kode. Itu sesuai dengan
  diagnosis E087 (bentuk panjang/kompleks lebih jarang dikodekan), tetapi **tidak menguji** sebab apa pun.

**Tindakan:**
- Status README → INFO NEG, dengan catatan koreksi.
- "E087 ADV-5 reclassified → partial positive control", "E027 UPGRADED" dan "L4 UPGRADED" **ditarik**.
- ADV-5 kembali GREY ZONE (vonis E087).

## 3. E130 — "438 substrate words" (2026-03-30)

**Yang diperiksa:** `substrate_interpretability.py`, `results/substrate_interpretability.json`,
`results/substrate_words.json`, `E027/data/features_matrix.csv`.

- **Label menjadi masukan (K4).** Masukannya memuat `n_cognate_sets` dan `max_cognate_set_size`. Keduanya bernilai 0
  **tepat** bila label = 0: tabulasi silang 438/0 dan 0/919. Masukan `concept_residual_rate` juga diturunkan dari label.
  Ini himpunan masukan "Model A" E027 (AUC 1,000, sudah dilaporkan melingkar di E027), minus dua masukan tingkat bahasa.
- RF 200 pohon dilatih pada seluruh data, lalu meramal data yang sama (baris 46 dan 57).
- Hasilnya: 438 bentuk dengan "prob_substrate > 0,8" = **himpunan yang persis sama** dengan 438 bentuk tanpa kode
  (dicek per bahasa–makna–bentuk; probabilitas terendah 0,975); 919 "kognat" = 919 bentuk berkode.
- Penggolongnya tidak menambah apa pun. **"438 kata substrat" adalah label tanpa kode dengan nama lain.**
- Tabel ranah dan tabel fonologinya **hanya** deskripsi bentuk tanpa kode vs berkode.
  - Makna tindakan dan tanda glotal bertahan pada jumlah huruf yang sama (C057, C066).
  - Gugus konsonan dan nasal tidak terpisahkan dari panjang (C057, C064).
- "Tolaki 64,1 %, Muna 15,5 % → differential Austronesian penetration": angka Muna adalah efek daftar Muna kedua (Wuna)
  di ABVD; tanpa itu 35,6 % (C056). Tafsirnya ditarik.
- Seluruh bagian "What substrate words reveal about pre-Indic life" (kekerabatan, bilangan, "INVISIBLE CIVILIZATION")
  dan kaitannya dengan E040 **tidak berdasar**:
  - "tanpa kode di ABVD" ≠ "bukan Austronesia" (C055);
  - numeralia majemuk (C060);
  - F9: menghitung kanal yang berkorelasi.

**Tindakan:** status README → FAILED (metode tidak sah), dengan catatan koreksi.

## 4. Kartu skor ADV-5 dan baris 180 jawaban obyektif

- Memori `project_evidence_scorecard.md`:
  - baris ADV-5 → GREY ZONE (E087), "resolution" E107 ditarik (C074);
  - E107 dikeluarkan dari daftar "Cathedral";
  - L4 "strongly supported (E130 438 substrate words …)" → E130 tidak bisa menopang L4 (C075).
- `OBJECTIVE_ANSWER_20261001.md` baris 180: "temuan negatif P8 (tidak ada substrat bersama yang koheren)" diganti dengan
  "P8: di antara bentuk tanpa kode, kemiripan lintas daftar kecil; uji tidak memeriksa kelas berkode" (C058).

## 5. Lingkar kedua — yang mengutip butir-butir di atas (dibaca di baris kutipannya saja, tidak diaudit ulang)

| Berkas | Kutipan | Tindakan |
|---|---|---|
| `lines/04_language_text/CLAUDE.md` l.54–55 | "E027 UPGRADED after E107 resolved ADV-5 → Mon-Khmer"; "E130 (438 substrate words)" | dikoreksi |
| `lines/02_taphonomy/CLAUDE.md` l.75; `STATE.md` inbox l.116 | "resolved by E107 (Mon-Khmer substrate)" | dikoreksi |
| `lines/04_language_text/STATE.md` l.238 | tugas "pastikan upgrade E027 tersebar" | ditutup: upgrade ditarik |
| `lines/06_thesis/CLAUDE.md` l.69 | L4 "supported (E130, E134)" | E130 dicoret; E134 tidak diaudit di sini |
| E112, E133, E150, E158, E168, E203 (README) | "438 substrate words" / "E107 … 6/6 Mon-Khmer" sebagai dasar | satu baris penjaga di atas README; isi tidak diaudit |
| E154 `fdr_reaudit.py` l.86 | p E107 = 0,001 diketik tangan | sudah AFFECTED_HEADLINE (audit 1 Okt); dicatat |
| `docs/VOLCARCH_STORY.md` l.144, l.249 | "6 dari 6 bahasa Mon-Khmer diklasifikasi dengan benar"; "ADV-5 PASSED" | salah bahkan tentang E107 (E107 tidak menggolongkan bahasa Mon-Khmer mana pun); dikoreksi |
| P8 `revision_ammo/E107_adv5_resolved.md`, `ADV5_negative_control.md` | bahan jawaban reviewer lama | banner DO NOT USE (seperti C051) |
| P18 `OUTLINE_v0.1.md` (HOLD) l.17, 38, 85 | memakai "438 words" sebagai masukan; "E107 validates substrate detection" | baris penjaga; P18 tidak boleh dibuka tanpa membaca catatan ini |
| `docs/bibliography/_SYNTHESIS_for_P0.md` l.117; draf P0 v0.1–v0.4 (tex) | "438 substrate words" | penanda di sintesis; draf tex historis tidak disentuh |
| `docs/TRIGGER_MAP.md` l.171–175, `NEXT_SESSION_BRIEF.md`, catatan Mata Elang | GREY ZONE (E087) | benar apa adanya; tidak disentuh |
| E027 README l.15, 185, 195 | "438 candidate substrate" | sumbernya; diurus revisi P8 (E227–E231) |

## 6. Arti untuk proyek

- Kanal linguistik (L4) kehilangan dua penopang. E107 adalah satu-satunya "kontrol positif" detektor, dan E130 adalah
  satu-satunya katalog "kata substrat". Keduanya tidak sah.
- Yang tersisa sama dengan yang dikatakan P8 R1: penggolong itu alat pemeringkat bentuk tanpa kode, bukan detektor
  substrat.
- Ini tidak menyentuh naskah yang sedang di-review. R1 sudah tidak memakai klaim-klaim itu.
- Bila reviewer putaran 2 bertanya soal kontrol negatif, jawabannya E087 apa adanya (GREY ZONE), bukan E107.
- Pelajaran (C075): `features_matrix.csv` E027 menyimpan empat masukan "distribusi kognat" (`n_cognate_sets`,
  `max_cognate_set_size`, `concept_residual_rate`, `concept_cross_lang_count`). E027 sudah tahu bahayanya: "Model A"
  dengan keempatnya mencapai AUC 1,000 dan dilaporkan sebagai model melingkar. Model utama P8 ("Model B") dan R1 tidak
  memakainya (E227 C08–C09). **E130 melatih ulang dengan himpunan masukan Model A tanpa menyebutnya.** Setiap pemakaian
  matriks itu wajib membuang keempatnya.

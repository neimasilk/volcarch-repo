# Sesi 4 — Results §3.10–3.11 — lembar fakta untuk PI (11 Nov 2026)

**Aturan lembar ini (C093, `AGENDA_PI_20261103.md` §5):** hanya butir dan angka, tanpa kalimat siap-salin. Bapak menulis
dengan kata-kata Bapak sendiri, dalam bahasa Indonesia. Saya menerjemahkan tanpa menambah isi; setiap perubahan di luar
terjemahan saya sampaikan dulu. Pertanyaan soal metode/angka boleh di chat — penjelasannya bukan untuk disalin.

**Angka:** dicek ulang 11 Nov dari berkas sumber (kolom "Sumber" di setiap tabel). Sumber utama:
`experiments/E235_elevation_vs_volcanic_control/` (`results/e235_outcome.json`, `README.md`, `DESIGN.md`,
`AUDIT_20261020.md`), `experiments/E236_p2_presence_sensitivity/results/E_e235/e235_outcome.json` (set bersih),
`experiments/E222_synthetic_ground_truth/results/e222{c,d}_outcome.json` (desain kuota),
`data/processed/dashboard/volcanoes_java_full.csv`.

**Yang berubah sejak sesi 3:** §3.10–3.11 di v0.3 dan v0.4 **identik** (diff = 0 baris). Jadi **tidak ada** tambahan
sesi 6 (K1) di bagian ini; kalimat set bersih E235 dari sesi 6 ada di **§3.3** (v0.4 baris 469–470), bukan di sini.

**Panjang (aturan PI 9 Nov: plafon = panjang v0.3, metode `count_subsection_words.py`):**

| Subseksi | Plafon v0.3 | v0.4 sekarang | Keterangan |
|---|---|---|---|
| §3.10 | **582** | **582** | tepat di plafon → tulisan baru harus sepanjang yang diganti |
| §3.11 | **86** | **86** | tepat di plafon |

Sesi ini **tidak menambah kata**. Kalau ada yang perlu ditambah (F25), ada yang harus dipotong di bagian yang sama.

---

## 0. Peta authorship §3.10–3.11 (dasar sesi ini)

"PI" = tulisan Bapak, diterjemahkan Claude 21–22 Okt → cukup dibaca ulang. "Claude" = disisipkan Claude di sesi 5–6 →
**Bapak tulis ulang**. Nomor baris = `submission_jcaa_v0.4.tex`.

| # | Baris | Isi (ringkas) | Penulis | Tindakan |
|---|---|---|---|---|
| 1 | 592 | pertanyaan analisis: vulkanik atau ketinggian | PI (passage 2 ¶1) | baca ulang |
| 2 | 593 | `volcano_dist` bukan fitur pelatihan → pola lewat variabel medan | PI | baca ulang |
| 3 | 594 | dekat ≤ 20 km, jauh ≥ 40 km, 30 pusat kanonik | PI | baca ulang |
| 4 | 595 | pita 200 m + CEM slope/TRI/TWI | PI | baca ulang |
| 5 | 596 | peta ensemble 10 seed tersimpan, tanpa latih ulang | PI | baca ulang |
| 6 | 597 | bootstrap blok 1.000, δ = 0,05, pra-registrasi | PI | baca ulang (pra-registrasi **sah** — F27) |
| 7 | 599 | definisi label SPECIFIC / NOT SPECIFIC / INCONCLUSIVE | PI (passage 3a) | baca ulang |
| 8 | 601 | di atas 1.000 m: 3.435 km², 97,5 % dekat, 0,4 % jauh | PI (angka dibulatkan Claude, sudah diberitahukan) | baca ulang |
| 9 | 602 | daratan jauh 15 km² (1.000–1.200 m), nol di atas 1.200 m | PI | baca ulang |
| 10 | 603 | di wilayah studi keduanya tak terpisahkan di atas ± 1.000 m | PI | baca ulang |
| 11 | 604 | karena itu tidak diatribusikan ke salah satu | PI | baca ulang |
| 12 | 605 | batas bergeser 800–1.200 m menurut kriteria luas minimum | PI | baca ulang |
| 13 | 607 | di bawah 1.000 m: +0,049 (SK +0,024…+0,077), +0,049, +0,060 (SK +0,030…+0,091) | PI | baca ulang (+ "XGBoost" — F25) |
| 14 | 608 | hybrid SPECIFIC; ringkasan lintas desain INCONCLUSIVE | PI | baca ulang |
| **15** | **609** | **dua definisi sensitivitas pra-registrasi + varian post hoc + "tidak ada arah yang diklaim"** | **Claude** (R6a sesi 5; P12 sesi 6) | **Bapak tulis ulang** |
| 16 | 610 | di upland (400–1.000 m) daratan pembanding jauh = Pegunungan Selatan, busur Eosen–Miosen, karst (`smyth2008`) | PI; Claude menambah "In the uplands (400–1,000 m)," (R6b) + sitasi | baca ulang |
| 17 | 611 | fringe MaxEnt (Section 3.6) median 1.107 m → tidak bisa ditafsirkan sebagai salah satu efek | PI; Claude mengganti "Figure 5" → "Section 3.6" (R6c) | baca ulang |
| 18 | 612 | daya uji terbatas → tidak ada kesimpulan tegas soal perbedaan antardesain | PI; Claude memperjelas objeknya ("near vs far land") | baca ulang |
| 19 | 626 | permukaan kesesuaian tidak bisa membedakan mekanisme | PI (passage 8E) | baca ulang |
| **20** | **627** | **kesesuaian rendah dekat gunung api tidak bisa dibaca sebagai bukti permukiman terkubur** | **Claude** (H20, sesi 5) | **Bapak tulis ulang** |
| **21** | **628** | **desain kuota: gagal di kedua dunia (angka World C/D)** | **teks v0.2 era Claude**; Bapak 22 Okt: "hasil dan angka kuota tetap" | **Bapak putuskan: tulis ulang / pakai dengan angka yang sama** (F26) |

Jadi yang **baru** hanya tiga kalimat (15, 20, 21) + keputusan angka; sisanya baca ulang. Perkiraan **± 150 kata baru**
(baris 609: 79 kata · 627: 17 · 628: 51), sepanjang yang diganti.

---

## 1. §3.10 "Volcanic setting or elevation?" — fakta

**Fungsi bagian:** menjawab pertanyaan R2 (round 2): apakah model menunjuk **lingkungan vulkanik** atau sekadar
**dataran tinggi**. Hasilnya: di atas ± 1.000 m keduanya tidak bisa dipisahkan **di wilayah studi ini**; di bawah 1.000 m
tidak ada yang tegas.

### A. ¶1 — desain (bahan baris 592–599)

| Fakta | Nilai | Sumber |
|---|---|---|
| Peta yang dipakai | ensemble **10 seed** × 3 desain × 3 algoritma, **588.535 sel**, disambung persis ke frame E217 | E221 README; E235 README |
| "Dekat" (V) | ≤ 20 km dari **pusat vulkanik Kuarter** kanonik; **30 pusat**, 17 di antaranya di luar bingkai peta 111–115° BT | `volcanoes_java_full.csv` (30 baris, bujur 106,74–114,37) |
| "Jauh" (N) | ≥ 40 km |
| Pencocokan | dalam tiap pita 200 m, CEM pada slope × TRI × TWI (bin tersil) | DESIGN §4 |
| Pita dipakai | syarat ≥ 25 km² **dan** ≥ 2 blok spasial di masing-masing lengan | DESIGN §4 |
| Ketidakpastian | bootstrap blok spasial 1.000 ulangan, seed 235 | DESIGN §4 |
| Ambang | δ = 0,05 kesesuaian; 5 poin persen untuk fringe | DESIGN §5 |
| Pra-registrasi | `DESIGN.md` dibekukan di commit `3ac41cc`, **di-push 20 Okt sebelum skripnya ada** | `e235_outcome.json` `design_commit` |
| Label (passage 3a) | SPECIFIC: SK tak mencakup nol **dan** \|Δ\| ≥ δ · NOT SPECIFIC: seluruh SK di dalam (−δ, +δ) · INCONCLUSIVE: sisanya | DESIGN §5 |
| Ringkasan untuk naskah | vonis yang dimiliki **≥ 2 dari 3 desain**; kalau tidak ada → "mixed" | DESIGN §5 |

### B. ¶2 — tumpang tindih di dataran tinggi, M1 (baris 601–605)

| Fakta | Nilai | Sumber |
|---|---|---|
| E\* (batas ketinggian) | **1.000 m** (pita terpakai 0–200 … 800–1.000 m) | `e235_outcome.json` M1 |
| Daratan di atas E\* | 3.434,52 km² → dibulatkan **3.435** | idem |
| Bagian dekat (V) | 0,97479 → **97,5 %** | idem |
| Bagian jauh (N) | 0,00435 → **0,4 %** | idem |
| Daratan jauh per pita 0–200…800–1.000 m | 19.742 · 2.701 · 778 · 435 · 185 km² | README M1 |
| Daratan jauh 1.000–1.200 m | **14,95 km²** (4 blok) → "sekitar 15 km²" | `e235_bands.csv` |
| Di atas 1.200 m | **nol** | idem |
| Pergeseran E\* | 1.200 m kalau luas minimum ≤ 14,9 km² · 1.000 m untuk 15–185 km² · **800 m** kalau ≥ 190 km² | AUDIT §4 |
| Set bersih (161 situs) | M1 **identik** (3.434,52 km²; 97,5 %; 0,4 %) — tidak perlu diulang di sini; sudah di §3.3 | E236 `E_e235/e235_outcome.json` |

### C. ¶3 — perbandingan di bawah 1.000 m, M2 (baris 607–612)

Angka naskah = **XGBoost** (algoritma utama, Metode baris 224), 53.927 sel tercocokkan.

| Desain | Δ̄ (V − N) | SK 95 % | Vonis δ = 0,05 |
|---|---|---|---|
| random | **+0,0491** | +0,0237 … +0,0768 | INCONCLUSIVE |
| TGB | **+0,0489** | +0,0240 … +0,0762 | INCONCLUSIVE |
| hybrid | **+0,0599** | +0,0298 … +0,0914 | **SPECIFIC** |
| ringkasan (2 dari 3) | — | — | **INCONCLUSIVE** |

Sumber: `E235/results/e235_outcome.json` (xgboost). Algoritma lain: RandomForest INCONCLUSIVE (utama), NOT SPECIFIC di
kedua alternatif; MaxEnt INCONCLUSIVE di ketiga definisi. **Hybrid SPECIFIC hanya pada XGBoost.**

**Dua definisi sensitivitas pra-registrasi** (keduanya juga memindahkan E\* ke 1.200 m dan mengubah jauh 40 → 30 km):

| Definisi | Δ̄ XGB random | SK 95 % | Vonis |
|---|---|---|---|
| V ≤ 10 / N ≥ 30 km | **−0,002** (−0,0023) | −0,0471 … +0,0458 | NOT SPECIFIC |
| V ≤ 15 / N ≥ 30 km | **+0,016** (+0,0164) | −0,0155 … +0,0514 | INCONCLUSIVE |

Sumber: `e235_outcome.json` `sens_10_30`, `sens_15_30`. Set bersih: labelnya **bertukar** (10/30 INCONCLUSIVE,
15/30 NOT SPECIFIC) — sudah di §3.3, jangan diulang di sini.

**Dekomposisi post hoc** (AUDIT §4; XGB, random / TGB / hybrid). Ini penting untuk kalimat 15: definisi 10/30
**mengubah tiga hal sekaligus**, bukan hanya lebar zona.

| Varian | Δ̄ random / TGB / hybrid |
|---|---|
| hanya V diperketat (≤ 10, N ≥ 40) | +0,029 / +0,032 / +0,035 |
| V ≤ 15, N ≥ 40 | +0,041 / +0,042 / +0,050 |
| hanya N diperlonggar (V ≤ 20, N ≥ 30) | +0,029 / +0,029 / +0,039 |
| kedua perubahan, pita 0–1.000 m saja | **+0,005** / +0,008 / +0,010 |
| run 10/30 tersimpan (**termasuk** pita 1.000–1.200 m, Δ −0,18) | **−0,002** |
| N diperketat (≥ 50) | **+0,056** / +0,056 / **+0,067** (ketiganya SPECIFIC) |
| jarak ke sungai ikut dicocokkan | +0,028 |
| sungai + 5 bin per variabel | +0,010 |

- Rentang naskah "+0,005 sampai +0,07" = ujung bawah "kedua perubahan" (+0,005) dan ujung atas N ≥ 50 (+0,067).
  Ringkasan README membulatkannya "+0,01 sampai +0,07". Keduanya sah; sebut sumbernya.
- Titik +0,05 **duduk di ambang**: vonis pra-registrasi tetap INCONCLUSIVE dan tidak ada arah yang diklaim.

**Pegunungan Selatan** (baris 610): 95 % sel jauh di 400–1.000 m ada di selatan 7,9° LS dan 87 % di barat 112,2° BT —
Pegunungan Selatan barat (Pacitan–Ponorogo), busur vulkanik Eosen–Miosen berkarst (Smyth, Hall & Nichols 2008,
GSA SP 436, doi 10.1130/2008.2436(10)). Jadi "sebagian besar" sah; menyebutnya "non-vulkanik" tidak.
Catatan audit: membuang sel jauh pesisir utara memotong Δ̄ separuh, membuang Pegunungan Selatan menaikkannya.

**Fringe MaxEnt** (baris 611): median ketinggian **1.107 m** (XGBoost 491 m) — di atas batas pemisahan → tidak bisa
ditafsirkan sebagai efek ketinggian atau efek vulkanik. Sumber: E221 README Part B; edit sesi 5 P6.

**Daya uji M3** (baris 612): vonis pra-registrasi NOT SPECIFIC, **tetapi tes tidak mungkin keluar SPECIFIC** — di bawah
1.000 m fringe hanya 2,5 % (dekat) dan 1,4 % (jauh), sedangkan ambangnya 5 poin persen. Dalam istilah relatif fringe
1,5–1,8× lebih sering dekat gunung api dan SK-nya tak mencakup nol. Karena itu kalimat baris 612 **tidak boleh** menjadi
"perbedaan antardesain tidak bergantung pada lingkungan vulkanik".

---

## 2. §3.11 "What a suitability surface cannot show, and a quota design that failed"

| Fakta | Nilai | Sumber |
|---|---|---|
| Desain kuota diuji di rezim asalnya (World C: survei tidak merata antarkawasan) | AUC_true kuota − random **−0,2457**; Jaccard **−0,4688**; **0/30** sel positif | `e222c_outcome.json` (10 dunia) |
| Di rezim paling menguntungkannya (World D: kebenaran seimbang antarkawasan) | **−0,2027**; **−0,2826**; **0/30** | `e222d_outcome.json` |
| TGB di dua dunia itu (pembanding, sudah di §3.6) | C −0,0010 (17/30) · D +0,0022 (22/30) | idem |

Isi yang perlu tersampaikan: (i) permukaan kesesuaian tidak membedakan mekanisme ketidakteramatan; (ii) karena itu
kesesuaian rendah dekat gunung api bukan bukti permukiman terkubur; (iii) desain kuota gagal di rezim asalnya **dan**
di rezim paling menguntungkan — dilaporkan tanpa diperhalus. Angka kuota **tidak berubah** (keputusan Bapak 22 Okt);
yang dipertanyakan hanya kalimatnya (F26).

---

## 3. Flags — Bapak putuskan (default saya di kanan)

| # | Hal | Fakta | Default |
|---|---|---|---|
| **F24** | Kalimat 15 (baris 609): definisi 10/30 bukan hanya lebar zona | Definisi 10/30 juga memindahkan E\* ke 1.200 m dan mengubah jauh 40 → 30 km; pita 1.000–1.200 m itulah yang menjatuhkan Δ̄ ke −0,002 (dekomposisi di §1C). "Hanya definisi zona" akan salah | Tulis seadanya: dua definisi pra-registrasi memberi **−0,002 (NOT SPECIFIC)** dan **+0,016 (INCONCLUSIVE)**; varian post hoc +0,005…+0,07; **tidak ada arah yang diklaim**. Jangan sebut sebab penurunan |
| **F25** | Baris 607–608 menyebut angka tanpa algoritma | Semua angka di ¶3 = XGBoost; hybrid SPECIFIC hanya pada XGBoost (RF & MaxEnt INCONCLUSIVE). Caption Gambar 7B sudah menyebut XGBoost, teks belum | Tambah "(XGBoost, algoritma utama)" ≈ 4 kata saat menulis ulang kalimat 15 — **tanpa** menambah total (potong dari tempat lain di §3.10) |
| **F26** | Kalimat 21 (baris 628, desain kuota) bukan tulisan Bapak: teks v0.2 era Claude | Angka sudah dicek ulang (tabel §2). Aturan editor: penulis menulis sendiri | **Bapak tulis ulang** (51 kata), angka seperti di tabel §2. Kalau Bapak lebih suka memakai kalimat lama, saya catat sebagai "dipakai dengan pengecekan" — tapi itu teks Claude |
| **F27** | Baris 597: "pra-registrasi" | **Sah** untuk E235: `DESIGN.md` dibekukan `3ac41cc`, di-push sebelum skrip ada. Aturan "jangan tulis pra-registrasi" (C082) berlaku untuk E219/E221/E223, bukan E235 | Pertahankan |
| **F28** | Baris 610: klausa "In the uplands (400–1.000 m)," ditambahkan Claude (R6b) | Tanpa klausa itu, "daratan pembanding jauh" berlaku juga di pita rendah, padahal mayoritas Pegunungan Selatan ada di 400–1.000 m | Pertahankan klausa |
| **F29** | Baris 611: "Figure 5" → "Section 3.6" (R6c) | Gambar 5 punya dua panel; rujukan ke section lebih tepat | Pertahankan |
| **F30** (opsional) | §3.10 tidak menyebut set bersih | §3.3 baris 469–470 sudah memberi angka set bersih + vonis + pertukaran label, dengan rujukan ke Section 3.10 | Tidak diulang di sini (plafon 582/582) |

**Bekas sesi 1–3 yang masih terbuka** (jangan dikerjakan di sesi ini, hanya pengingat): F12 dan C094 → Metode;
F15 → Limitations; Gambar 2 belum dirujuk di teks; caption Gambar 4–6 masih kata v0.3; label ¶2 §3.9 vs penutupnya
(opsional, PI).

---

## 4. Jangan ditulis (sudah terbukti salah atau terlalu kuat)

- "perbedaan antardesain tidak bergantung pada lingkungan vulkanik" — M3 tidak punya daya uji (baris 612 sudah benar).
- "definisi 10/30 = hanya zona lebih sempit" — ia juga menambah pita 1.000–1.200 m (F24).
- "suitability lebih tinggi dekat gunung api **pada semua algoritma**" — hybrid SPECIFIC hanya XGBoost.
- "dataran tinggi **non-vulkanik**" — Pegunungan Selatan busur Eosen–Miosen; sebut "jauh dari pusat Kuarter".
- "+0,05 signifikan" tanpa "duduk di ambang / vonis ringkasan INCONCLUSIVE".
- "kuota gagal **karena** …" — tidak ada analisis sebab; cukup hasilnya gagal di dua rezim.
- "pra-registrasi" untuk E219/E221/E223 (C082). E235 **boleh** (F27).
- Angka tanpa pembulatan naskah: 3.434,5 → **3.435**; 97,48 → **97,5**; 0,435 → **0,4**; 14,95 → **15**; 0,0491 → **0,049**.

## 5. Sumber yang dipakai (11 Nov)

- `experiments/E235_elevation_vs_volcanic_control/results/e235_outcome.json` — M1, M2, sensitivitas, M3, M4
- `.../results/e235_bands.csv` — luas per pita, termasuk 14,95 km² (1.000–1.200 m)
- `.../README.md` — M2 tabel, M3 kekuatan uji, M4, audit
- `.../AUDIT_20261020.md` — dekomposisi post hoc, pergeseran E\*, Pegunungan Selatan
- `.../DESIGN.md` §5 — aturan vonis + aturan ringkasan ≥ 2 dari 3
- `experiments/E236_p2_presence_sensitivity/results/E_e235/e235_outcome.json` — set bersih (M1 identik; label sensitivitas bertukar)
- `experiments/E222_synthetic_ground_truth/results/e222c_outcome.json`, `e222d_outcome.json` — desain kuota
- `data/processed/dashboard/volcanoes_java_full.csv` — 30 pusat kanonik

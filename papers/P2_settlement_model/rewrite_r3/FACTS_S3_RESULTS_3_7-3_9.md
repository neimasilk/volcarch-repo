# Sesi 3 — Results §3.7–3.9 — lembar fakta untuk PI (10 Nov 2026)

**Aturan lembar ini (C093, `AGENDA_PI_20261103.md` §5):** hanya butir dan angka, tanpa kalimat siap-salin. Bapak menulis
dengan kata-kata Bapak sendiri, dalam bahasa Indonesia. Saya menerjemahkan tanpa menambah isi; setiap perubahan di luar
terjemahan saya sampaikan dulu. Pertanyaan soal metode/angka boleh di chat — penjelasannya bukan untuk disalin.

**Angka:** dicek ulang 10 Nov dari berkas sumber (lihat kolom sumber). `revision_ammo/verify_headline_numbers.py` 60/64
(4 = K5/K6/K7/G1c yang memang ditarik — dua di antaranya ada di sesi ini: **K7** dan **G1c**).

**Panjang (aturan PI 9 Nov: plafon = panjang v0.3, metode `count_subsection_words.py`):**

| Subseksi | Plafon v0.3 | Yang sudah tulisan Bapak (boleh dipakai lagi setelah dibaca ulang) | Yang Bapak tulis baru |
|---|---|---|---|
| §3.7 | **42** | — | semua (≤ 42) |
| §3.8 | **297** | ¶2 = passage 1 (21 Okt), 137 kata | ¶1 (≤ 160) |
| §3.9 | **189** | ¶2 tiga kalimat pertama = passage 8D (22 Okt), 29 kata | ¶1 (≤ 93) + sisa ¶2 (≤ 67) |

Total tulisan baru ≤ ± 362 kata Inggris (± 410–430 kata Indonesia). Results §3.1–3.6 sudah +351 dari v0.3 → kalau bisa,
sesi ini **di bawah** plafon.

**Kenapa paragraf lama "boleh dipakai lagi" hanya sebagian (aturan 6 di `PI_PROSE_LOG_R3.md`):** passage 1 dan 8D adalah
tulisan Bapak, diterjemahkan Claude. Tapi di **8D**, kalimat terakhir Bapak diganti Claude di sesi 5 (edit H19,
`apply_session5_edits.py`) — bagian itu harus Bapak tulis ulang. Passage 1 hanya diubah satu angka (5.7 → 5.65, sudah
Bapak setujui 21 Okt).

---

## §3.7 Interval kepercayaan dan batas deteksi ("Confidence intervals and a detection floor") — ≤ 42 kata

**Fungsi bagian:** menyatakan apa yang **tidak bisa** disingkirkan oleh data ini — efek kecil di bawah batas tertentu
tidak terdeteksi, bukan tidak ada. §3.2 (baris 429) dan Limitations (baris 672) merujuk ke sini untuk "+0.03".

| Fakta | Angka | Sumber |
|---|---|---|
| Batas atas CI bootstrap blok presensi (29 ulangan), hybrid − random, common background | MaxEnt +0.0082 · XGBoost +0.0253 · RF +0.0256 | `E223_statistical_robustness/results/e223b_bootstrap_summary.csv` (G1 #53) |
| Batas bawah CI yang sama | −0.0589 · −0.0241 · −0.0173 | idem |
| Lebar CI | 0.067 · 0.049 · 0.043 | idem (dihitung) |
| Batas deteksi yang dinyatakan | efek **< ± +0.03 tidak bisa disingkirkan** pada n = 378 | `E223/README.md` B ("honest power statement"); G1 catatan #53: "batas yang dinyatakan, bukan hasil" |
| Yang tegas ditolak | +0.092 (12/12 sel CI kesetaraan; 3/3 bootstrap) | `e223a_equivalence_ci.csv`, G1 #49 |

- **Sudah Bapak tulis di §3.2** (jangan diulang): 12/12 CI kesetaraan, 29 ulangan bootstrap, batas atas +0.008/+0.025/+0.026.
- **Set bersih (161 situs):** batasnya lebih besar dan **tidak diukur** (interval +0.051 [+0.039, +0.063] adalah atas
  seed, bukan atas situs). Sudah di Limitations — di sini boleh tidak disebut.
- ⚑ **Opsi O1 (struktur, Bapak putuskan):** §3.7 hanya dua kalimat; isinya bisa juga ditaruh di akhir §3.2. Saran saya:
  **tetap subseksi sendiri**, karena §3.2 dan Limitations merujuk ke `Section 3.7` — memindahkannya berarti mengubah dua
  rujukan dan urutan nomor §3.8–3.11.

---

## §3.8 Peta prioritas berubah dengan seed; merata-ratakan seed mengurangi ini — ¶1 ≤ 160 kata (¶2 = passage 1)

**Fungsi ¶1:** angka AUC bisa artefak, tapi yang dipakai di lapangan adalah **peta**; seberapa banyak peta berubah hanya
karena seed, dan berapa banyak ensemble menolong — diukur dengan cara yang adil.

**A. Pergantian sel desil teratas antar-seed (E221):**

| Fakta | Angka | Sumber |
|---|---|---|
| Pergantian (1 − Jaccard) antar pasangan seed, sel 10 % teratas | **28.1 % – 47.4 %** (rata-rata per algoritma × desain; 405 pasangan) | `E221_seed_ensemble_stability/results/e221_turnover_pairs.csv` (G1 #58) |
| Paling stabil / paling tidak stabil | RF-TGB 28.1 % · XGBoost-hybrid 47.4 % | idem |
| Konsekuensi | satu peta satu-seed bukan dasar yang baik untuk keputusan survei | — |

**B. Berapa seed yang cukup — dua cara mengukur:**

| Cara | Angka | Sumber | Catatan |
|---|---|---|---|
| Ensemble k seed vs ensemble 10 seed (yang **berisi** seed itu sendiri) | Jaccard > 0.90 setelah **4–7 seed** (k* = 2–5 di 0.85; 7–9 di 0.95) | `E223/results/e223d_kstar_thresholds.csv`; `E221/results/e221_stabilisation_curve.csv` | **menyanjung**: pembanding berbagi seed (C083) |
| Dua ensemble yang **tidak berbagi seed** (yang adil) | 1 seed: **0.55–0.70** · 5 seed: XGBoost **0.76**, MaxEnt **0.85**, RF **0.86** | `E236_p2_presence_sensitivity/results/F_disjoint_full.csv` (rata-rata tiga desain) | ukuran stabilitas yang adil |
| Set bersih, 5 seed | 0.70–0.77 | `F_disjoint_clean.csv` | **sudah di §3.3** — cukup rujukan |

- Protokol yang direkomendasikan naskah: **minimal 7 seed sebagai lantai** + **laporkan kesepakatan antar ensemble yang tidak
  berbagi seed** (aturan line 01 #3; C083). Bukan "7 seed membuat peta stabil".
- **Gambar 6** (`fig:stabilisation`) hanya dirujuk di ¶1 ini → rujukannya harus tetap ada.

**C. ¶2 = passage 1 Bapak (21 Okt) — mohon dibaca ulang dan jawab: dipakai / diubah:**

> Ensemble ini menunjukkan perbedaan yang bermanfaat antara robust core (wilayah inti yang konsisten) dan contingent
> fringe (wilayah pinggiran yang bergantung pada desain latar). Robust core didefinisikan sebagai sel yang masuk dalam 10%
> peringkat teratas pada ketiga desain latar (random, TGB, dan hybrid), dengan setiap peta merupakan rata-rata dari 10
> seed; sedangkan contingent fringe adalah sel yang masuk dalam 10% peringkat teratas pada tepat satu desain. Kepadatan
> situs yang teramati pada wilayah inti lebih tinggi dibandingkan wilayah pinggiran, yaitu 1,93× untuk RandomForest (30,7
> dibandingkan 15,9 situs per 1.000 km²), 4,34× untuk XGBoost (40,8 dibandingkan 9,4), dan 5,62× untuk MaxEnt (31,7
> dibandingkan 5,7). Perbedaan ini menunjukkan konsistensi spasial, bukan validasi independen, karena situs yang digunakan
> untuk menghitung kepadatan tersebut juga digunakan untuk melatih model.

(Di naskah: 5,7 → **5.65**, nilai sumber; dengan 5,7 pembaca menghitung 5,56×. Sudah Bapak setujui 21 Okt.)

Angka dicek ulang 10 Nov (`E221/results/e221_priority_sets.csv`): inti 169 / 141 / 100 situs, pinggiran **15 / 15 / 16
situs** (XGBoost / RF / MaxEnt); kepadatan 40.8/9.4, 30.7/15.9, 31.7/5.65 → 4.34×, 1.93×, 5.62×.

- ⚑ **F16 (opsional):** di §3.3 Bapak menulis untuk set bersih bahwa pinggiran "hanya 5, 15 dan 5 situs, jadi rasionya
  menunjukkan arah, bukan ukuran yang tepat". Di set 378, pinggirannya juga kecil (**15, 15, 16 situs**). Menyebutnya di
  ¶2 membuat kedua set sejajar (± 12 kata). Bapak putuskan.

---

## §3.9 Koreksi dua diagnostik vulkanik ("Corrections to two volcanic diagnostics") — ¶1 ≤ 93 · sisa ¶2 ≤ 67

**Fungsi bagian:** dua diagnostik pipeline asli diperbaiki: (1) daftar gunung api yang dipakai kode; (2) satu korelasi
yang tidak stabil antar-seed. **Test 1** sudah dijelaskan di Metode (baris 274): korelasi kesesuaian prediksi dengan jarak
ke gunung api (dan ke jalan, sungai) — lulus kalau |ρ| < 0.5 (model tidak sekadar belajar jarak gunung api).

**A. ¶1 — inventaris gunung api:**

| Fakta | Angka | Sumber |
|---|---|---|
| Pusat vulkanik yang ditanam di kode lama | **7** (Kelud, Arjuno-Welirang, Bromo-Tengger, Semeru, Lamongan, Raung, Ijen) | file lama `data/processed/dashboard/volcanoes.csv` (cacat file 7 gunung api, line 02 / P7) |
| Inventaris kanonik | 30 pusat di Jawa; **13** di dalam batas naskah 111–115° BT | idem; `E219/results/e219_outcome.json` `int1_n_canonical_in_bounds` = 13 |
| Enam tambahan | Lawu, Wilis, Kawi-Butak, Penanggungan, Iyang-Argapura, Baluran | idem |
| Dua tambahan di dalam konsentrasi situs Malang–Mojokerto | Penanggungan (74 rekaman ≤ 15 km), Kawi-Butak (56 rekaman ≤ 25 km) | dicek kasar 10 Nov pada `data/processed/east_java_sites.geojson` (666 rekaman, bukan 378 presensi model) |
| `volcano_dist` fitur pelatihan? | **tidak** → AUC model tidak terpengaruh; yang terpengaruh diagnostik Test 1 | Metode §2 |
| ρ Test 1, inventaris kanonik (13) | **−0.281** | `e219_outcome.json` (G1 #62) |
| ρ Test 1, inventaris lama (7), diulang 5 seed | **−0.243** | idem (G1 #61) |
| Vonis | \|ρ\| < 0.5 di keduanya → tes tetap lulus | — |

**B. ¶2 — korelasi yang tidak konsisten antar-seed (passage 8D Bapak, 22 Okt):**

Tiga kalimat pertama adalah tulisan Bapak (di naskah sudah tanpa "pada versi sebelumnya", disetujui 22 Okt):

> Korelasi yang tidak konsisten antar-seed. [Pada versi sebelumnya,] korelasi Spearman sebesar −0,163 dilaporkan
> berdasarkan satu instans model. Namun, pengulangan analisis dengan lima seed pada inventaris tujuh gunung api yang sama
> menghasilkan korelasi sebesar −0,243 (E219).

Kalimat terakhir Bapak waktu itu menyebut perbedaan itu menunjukkan hasil berubah **antar-seed**. Di sesi 5 Claude
menggantinya (H19) karena fakta berikut — **bagian ini yang Bapak tulis ulang:**

| Fakta | Sumber |
|---|---|
| Pengulangan E219 tidak hanya mengganti seed: ia juga memakai **sampling background lattice** (bukan sampling koordinat kontinu pipeline asli; Metode, `sec:refutation`) | `E219`; naskah baris 292 |
| Jadi selisih −0.163 → −0.243 bisa dari seed **atau** dari cara sampling — **tidak bisa dipisahkan** di sini | — |
| Yang tetap aman: satu instans model bukan dasar yang baik untuk diagnostik ini | G1c (`review_package_20260727/10_SET_KLAIM_TERKOREKSI.md`) |
| (kalimat v0.2, opsional) nilai koreksi di bagian ini memakai ensemble seed; tangga E007–E013 di **Tabel 4** adalah pipeline asli, direproduksi sebagai objek yang diperiksa | — |

- ⚑ **F17:** kalau kalimat terakhir tentang Tabel 4 dibuang, Tabel 4 hanya dirujuk dari caption Gambar 2 — dan Gambar 2 sendiri
  belum dirujuk di teks (sisa sesi 1). Salah satunya perlu rujukan dari teks sebelum *freeze*. Bapak bisa menaruh rujukan
  Tabel 4 di sini, atau nanti di §3.1.

---

## Jangan ditulis (sudah terbukti salah atau terlalu kuat)

- **K7:** "kepadatan inti 2–5.6× pinggiran" — yang benar **1.93× / 4.34× / 5.62×** (atau "1.9–5.6×"); membulatkan 1.93 ke 2
  adalah kesalahan sejenis K3.
- **C083:** "7 seed membuat peta stabil" / "k* = 7 menjamin stabil" — kurva k-vs-10 menyanjung; 7 seed = **lantai**.
- **G1c:** −0.163 sebagai nilai yang bisa direproduksi — tidak bisa (5 seed: −0.243).
- "pra-registrasi" untuk E219, E221, E223 (desain tertulis, di-commit bersama hasil; Metode baris 289). Hanya E224 yang sah.
- "+0.03 adalah hasil" — itu batas yang dinyatakan, bukan temuan.
- "inti terbukti/tervalidasi" — kepadatan dihitung dari situs yang juga melatih model (konsistensi, bukan validasi).
- "selalu" / "tidak pernah" tanpa pecahan.

## Untuk pembaca (R1) — cukup tersirat

- Naskah berdiri sendiri: tanpa "reviewer", "revisi", "versi sebelumnya", "originally reported"; "Section", bukan "§".
- Istilah: *seed ensemble* dan *Jaccard overlap* sudah ada di Tabel 1 (Key terms) — tidak perlu didefinisikan ulang.
  *Robust core* / *contingent fringe* **tidak** ada di Tabel 1; definisinya ada di ¶2 (passage 1).

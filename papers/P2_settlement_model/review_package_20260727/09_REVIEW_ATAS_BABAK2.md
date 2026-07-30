# Dokumen 9 — Review Menyeluruh atas Paket Babak 2 (dokumen 07 + E222/E223)

**Tanggal:** 2026-07-27 | **Sifat:** verifikasi independen + review bermusuhan atas review Go Frendi
**Metode:** setiap angka yang dikutip dokumen 07 dicocokkan ke file hasil mentahnya; lalu klaim
struktural diuji dengan analisis ulang sendiri, bukan dibaca ulang.

**Putusan singkat:** kerja babak 2 **kuat dan sebagian besar terverifikasi** — 100% angka yang saya cek
cocok persis dengan `results/`. Tapi **dua klaim headline tidak didukung datanya sendiri**, dan keduanya
ada di R1 — syarat naskah yang jadi tumpuan seluruh putusan. Keduanya harus dikoreksi sebelum apa pun
masuk naskah. Ini bukan minta eksperimen baru; ini koreksi klaim, disiplin yang sama yang sedang
kita terapkan ke naskah asli.

---

## 1. Yang saya verifikasi — semuanya cocok

| Klaim dokumen 07 | File | Verifikasi |
|---|---|---|
| Tabel utama: random 0.847/0.828, tgb 0.840/0.827, hybrid(0.0) 0.737/0.541, hybrid(1.0) 0.890/0.617 | `e222_runs.csv` | **cocok persis** |
| Biaya kebenaran median +0.194, 100% positif | `e222_outcome.json` (0.1937, 1.0) | **cocok** |
| P1 gagal: Spearman pooled +0.44 < 0.5 | `e222_outcome.json` (0.4395, `P1_supported: false`) | **cocok** |
| P3 TGB null: −0.010, 47% positif | `e222_outcome.json` (−0.010, 0.4667) | **cocok** |
| P4 Boyce borderline: median +0.50/+0.54 | `e222_outcome.json` (0.5009, 0.5429) | **cocok** |
| World C: −0.246 / −0.469, 0/30 | `e222c_outcome.json` (−0.2457, −0.4688, 0.0) | **cocok** |
| World D: −0.203 / −0.283, 0/30 | `e222d_outcome.json` (−0.2027, −0.2826, 0.0) | **cocok** |
| E223-A: 12/12 tolak +0.092; MaxEnt uniform CI −0.039…−0.028 | `e223a_equivalence_ci.csv` (−0.0389…−0.0279) | **cocok** |
| E223-B: n=29, batas atas +0.008…+0.026 | `e223b_bootstrap_summary.csv` (+0.0082/+0.0253/+0.0256) | **cocok** |
| E223-C: −0.020…−0.022 semua beta, 1/10 positif | `e223c_beta_summary.csv` (−0.0198…−0.0217, 0.1) | **cocok** |

**Desain E222 sehat.** Saya periksa `01_synthetic_truth.py`: bias survei simulasi memakai
`clip(exp(-road/12000), 0.03, 1)` — **fungsi yang persis sama** dengan yang dipakai `base.draw_tgb`.
Artinya TGB diberi kondisi yang teorinya (Phillips dkk. 2009) butuhkan. Itu tes yang adil, bahkan murah
hati. Presence evaluasi disampel ∝ λ **tanpa** filter survei = sampel tak-bias yang sah. Pipeline memakai
kode E217 yang sama. Ini konstruksi yang benar.

**Praktik yang patut dipuji, bukan basa-basi:** P1 dinyatakan **gagal** padahal itu klaim kita sendiri;
dua fork terdaftar (World C dan World D) jatuh ke cabang NO dan dilaporkan apa adanya; kesalahan referensi
first-pass E221 dikoreksi dan jejaknya dipertahankan. Itu perilaku yang benar.

---

## 2. MAYOR-1 — Patologi seleksi LENYAP di grid yang benar-benar dipakai paper

**Ini temuan terpenting review saya.**

Dokumen 07 menulis: *"Aturan seleksi naskah memilih hybrid(1.0) di 60/60 kasus... padahal kebenaran bilang
random lebih baik +0.21 AUC."*

Tapi grid E013 di naskah adalah `hard_frac ∈ {0.0, 0.15, 0.30}` (naskah §2.4, terverifikasi di
`submission_jcaa_v0.1.tex`). **hard_frac 0.7 dan 1.0 tidak pernah ada di sapuan naskah.** Saya jalankan
ulang seleksinya dengan dan tanpa batas itu:

| Kumpulan kandidat | Dipilih AUC laporan | Dipilih kebenaran | Biaya median | Seleksi salah |
|---|---|---|---|---|
| Grid penuh E222 (sampai 1.0) | hybrid(1.0) 60/60 | random 33, tgb 27 | **+0.1937** | **60/60** |
| **Grid paper (≤0.30)** | **random 50, tgb 10** | random 33, tgb 27 | **+0.0000** | **0/60** |

Dan hal yang sama berlaku di data nyata (`e218b_summary.csv`):

| Batas dial | AUC laporan memilih | Biaya kebenaran |
|---|---|---|
| Grid paper (≤0.30) | hard_frac = 0.3 | **+0.0044** |
| Dial penuh (≤1.0) | hard_frac = 1.0 | **+0.0973** |

**Artinya:** angka headline +0.194 (sintetik) dan +0.094 (nyata) **seluruhnya bergantung pada memperluas
dial melewati apa yang naskah pernah pakai.** Di titik operasi naskah sendiri, biayanya +0.000 dan +0.004 —
20 sampai 40 kali lebih kecil, dan secara praktis nol.

**Konsekuensi untuk M1:** M1 **tidak tertutup** untuk klaim "seleksi naskah berjalan ke arah yang salah".
Di dunia sintetik, kriteria naskah yang dijalankan pada grid naskah justru memilih **random** — bukan
hybrid. Jadi dunia sintetik bahkan **tidak mereproduksi perilaku seleksi data nyata di titik operasi
naskah** (di data nyata kriteria memilih hybrid; di sintetik memilih random). Itu celah kalibrasi yang
membatasi transfer kesimpulan E222 kembali ke kasus nyata, dan harus diungkap.

**Yang MASIH bertahan, dan ini tetap berharga:** kriterianya **tidak punya optimum interior**. AUC laporan
naik monoton sampai ujung dial di kedua dunia. Naskah berhenti di 0.30 **hanya karena gridnya berhenti di
situ** — tidak ada apa pun di dalam kriteria yang menyuruh berhenti. Itu bahaya nyata, bisa dipublikasi,
dan jujur.

**Rumusan pengganti yang saya rekomendasikan:**
> *Kriteria seleksi berbasis AUC-background-sendiri tidak punya optimum interior: ia memberi imbalan pada
> desain yang makin ekstrem tanpa batas, sementara kebenaran tidak mengikuti. Naskah kami berhenti di
> hard_frac = 0.30 karena gridnya berhenti di situ, bukan karena kriterianya menyuruh berhenti. Biaya di
> titik itu kecil (+0.004 nyata, +0.000 sintetik); bahayanya bukan besaran kesalahan kami, melainkan bahwa
> kriteria itu tidak menyediakan pengaman apa pun terhadap kesalahan yang jauh lebih besar.*

Itu lebih lemah dari "60/60 memilih yang terburuk" — dan lebih tahan serangan.

---

## 3. MAYOR-2 — Faktor "~10×" di R1 tidak didukung data

R1 (syarat naskah #1, tumpuan seluruh putusan) berbunyi: *"dial desain menggerakkan angka laporan
**~10× lebih cepat** daripada kebenaran ke arah mana pun."*

Saya hitung dari `e222_runs.csv` dan `e218b_summary.csv`:

| Sumber | Δ auc_own | Δ kebenaran | Rasio |
|---|---|---|---|
| Sintetik, dial 0.0→1.0 | +0.1538 | +0.0764 | **2.01×** |
| Sintetik, slope per-run (median) | +0.1535 | +0.0726 | **2.12×** |
| Data nyata E218b, dial 0.0→1.0 | +0.1227 | −0.0973 | **1.26×** |

**Rasionya 2×, bukan 10× — meleset sekitar 5 kali lipat.** Angka ini ada di kalimat yang jadi klaim
mekanisme utama naskah v0.2. Kalau masuk naskah apa adanya, seorang reviewer yang menghitung sendiri akan
menemukannya, dan kredibilitas seluruh paket koreksi-diri ini rusak — persis pada paper yang isinya
tentang melaporkan angka secara jujur.

**Perbaikan:** ganti dengan besaran terukur — *"angka laporan bergerak sekitar dua kali lebih cepat
daripada kebenaran di dunia sintetik (2.0×; slope per-run 2.1×), dan di data nyata bergerak ke arah
berlawanan dari kebenaran"*. Yang berlawanan-arah itu justru poin yang lebih kuat dan tidak butuh
pembesaran.

---

## 4. MODERAT-3 — "Angka laporan SELALU terinflasi" sebenarnya 95,3%

Dokumen 07 (R1): *"evaluasi di background sendiri **selalu** terinflasi secara struktural."*

Data: **343 dari 360 run** inflasi positif (95,3%); nilai minimum **−0.031**; median +0.187.

Arahnya sistematis dan kuat, tapi "selalu" itu kuantor absolut yang datanya tidak dukung. Ganti dengan
"sistematis (95% run; median +0.19)". Ini kelas kesalahan yang sama dengan yang sedang kita koreksi di
naskah asli — kuantor lebih kuat dari buktinya.

---

## 5. MODERAT-4 — Penjelasan mekanistik yang hilang untuk null TGB (ini justru menguatkan)

Dokumen 07 menjelaskan null TGB sebagai "satu rejim bias, daya n≈500". Itu benar tapi kurang tajam, dan
seorang reviewer SDM yang hafal Phillips dkk. (2009) akan **mengharapkan TGB bekerja** di simulasi ini —
karena simulasinya justru memberi TGB asumsi persisnya. Null yang tak terjelaskan akan dibaca sebagai bug.

**Diagnosis yang hilang:** `FEAT` = elevation, slope, twi, tri, aspect, river_dist. **`road_dist` bukan
fitur.** Model secara struktural **tidak bisa mengekspresikan** bias survei. Jadi bias itu tidak masuk
sebagai distorsi sistematis di ruang fitur — ia masuk sebagai label noise (sebagian sel ber-λ tinggi tidak
terobservasi). TGB dirancang membatalkan faktor s(x) di ruang fitur; kalau s(x) tidak bisa direpresentasi
di ruang fitur, **tidak ada yang perlu dibatalkan.**

Itu mengubah null yang mengejutkan menjadi null yang **terprediksi**, dan menghasilkan syarat yang bisa
diuji: *koreksi target-group hanya bisa membantu kalau variabel biasnya berkorelasi dengan ruang fitur
model.* Tulis ini di naskah dan null TGB berubah dari kelemahan jadi kontribusi. (Uji cepat yang
mengonfirmasinya: masukkan `road_dist` ke fitur, ulangi P3 — kalau TGB lalu menolong, diagnosisnya benar.)

---

## 6. Catatan minor

- **m-a.** `e222_outcome.json` menulis `"P4_boyce_tracks_truth": true`, sementara narasi dokumen 07
  menurunkan pangkat Boyce jadi sanity check. Narasinya lebih konservatif dari flag-nya sendiri — itu
  benar, tapi flag JSON-nya harus disesuaikan supaya jejak audit tidak berkontradiksi.
- **m-b.** `P5_B_misspecified`: map Jaccard tgb **0.4504 > random 0.4458** — TGB sedikit lebih baik di
  bawah misspecification. Kecil, tapi ini satu-satunya kondisi di mana TGB menang; dokumen 07 tidak
  menyebutnya. Sebutkan (menambah kredibilitas, dan sejalan dengan diagnosis §5).
- **m-c.** E223-A: 3 dari 12 sel (kolom eval hybrid) punya efek **positif** dengan CI mengecualikan nol
  (+0.007…+0.016). Itu justru tanda-tangan artefak dan konsisten; tapi kalimat "12/12 menolak" sebaiknya
  disertai catatan bahwa 3 sel itu positif-kecil, supaya tidak terbaca seolah semua sel negatif.
- **m-d.** World C/D: `tgb_vs_random_auc_true` = −0.001 (C) dan **+0.002 dengan 73% positif** (D). TGB
  netral, bukan merugikan. Konsisten dengan §5.

---

## 7. Koreksi atas klaim SAYA sendiri

Pagi ini saya melaporkan E218b sebagai: *"hard negative aktif menurunkan generalisasi"*. **Itu
over-generalisasi.** E222 menunjukkan tanda slope kebenaran itu kontingen terhadap rejim bias — di dunia
sintetik `auc_true` justru **naik** dengan hard_frac (0.541→0.617). Dokumen 07 benar menuntut reframing
ini, dan saya salah menyatakannya sekuat itu dari satu dataset.

Yang bertahan dari E218b: inflasi meningkat tajam dan sistematis dengan hard_frac, dan kriteria
laporan tidak punya optimum interior.

---

## 8. Putusan saya atas babak 2

**Setuju dengan arah dan sebagian besar isinya. Tidak setuju dengan dua klaim headline-nya.**

| Item | Status |
|---|---|
| Kualitas eksekusi E222/E223 | **kuat** — desain benar, pre-registrasi ada, angka terverifikasi 100% |
| Kejujuran pelaporan (P1 gagal, 2 fork jatuh NO) | **teladan** |
| M2, M3, M4, M6 tertutup | **setuju** |
| M5 (Boyce turun pangkat) | **setuju**, perbaiki flag JSON |
| M7, M8 | **setuju** sebagai residual risk terdeklarasi |
| **M1 tertutup** | **TIDAK SETUJU** — lenyap di grid naskah; perlu re-scope (§2) |
| **R1 apa adanya** | **TIDAK BOLEH MASUK NASKAH** — faktor 10× salah 5×; "selalu" → 95,3% (§3, §4) |

**Tiga perbaikan wajib, semuanya koreksi klaim, bukan eksperimen baru:**
1. **R1 ditulis ulang** dengan rasio terukur (2.0× sintetik; berlawanan arah di data nyata) dan kuantor
   yang benar (95% run, bukan "selalu").
2. **M1 di-re-scope** ke "kriteria tanpa optimum interior" + ungkap bahwa di titik operasi naskah biayanya
   +0.004/+0.000, **dan** ungkap celah kalibrasi sintetik-vs-nyata.
3. **Diagnosis ruang-fitur untuk null TGB** dimasukkan (§5) — mengubah kelemahan jadi kontribusi.

Setelah tiga itu, saya setuju dengan putusan dokumen 07: paket ini layak dibawa ke Q1. Sebelum tiga itu,
tidak — karena paper yang isinya tentang melaporkan angka secara jujur tidak boleh mengirimkan angka yang
tidak didukung datanya sendiri.

---

*Verifikasi dijalankan ulang dari `results/` mentah pada 2026-07-27. Perintah reproduksinya ada di
dokumen 3 §7; analisis re-seleksi di §2 dokumen ini dihitung langsung dari `e222_runs.csv` dan
`e218b_summary.csv`.*

# Dokumen 7 — Review Keras Q1 (Go Frendi, babak 2) atas Paket Revisi P2/JCAA #280

**Tanggal:** 2026-07-27 | **Sifat:** review internal standar Q1 — menyerang paket babak 1
(dokumen 05–06 + E220/E221) sebagaimana reviewer jurnal top akan menyerangnya
**Metode:** setiap kritik mayor harus (a) saya nyatakan dalam bentuk paling kuatnya, (b) ada eksperimen
pre-registered yang menutupnya, atau (c) diakui sebagai residual risk yang harus ditulis eksplisit di
naskah. Tidak ada kritik yang dijawab dengan kata-kata saja.
**Eksperimen penutup:** E222 (`experiments/E222_synthetic_ground_truth/`), E223
(`experiments/E223_statistical_robustness/`), keduanya dengan DESIGN.md terdaftar sebelum run.

---

## Kritik Mayor

### M1 — "Tidak ada ground truth di mana pun dalam rantai refutasi Anda." ★ paling mematikan

**Serangan.** Seluruh angka "generalisasi" E217–E220 diskor terhadap 378 presence yang sama-sama bias
survei. Common-background AUC mengukur diskriminasi *rekaman yang bias* terhadap availability — besaran
yang buta terhadap persis hal yang mau dikoreksi target-group background. Jadi: Anda membuktikan metrik
laporan rusak memakai penggaris yang tidak bisa melihat koreksi bias. "Seleksi berjalan mundur" bisa jadi
artefak dari penggaris-rusak yang mengukur penggaris-rusak. Tunjukkan patologinya terhadap ground truth,
atau klaim terkuat Anda lokal terhadap perbandingan yang sama-sama rusak.

**Penutup: E222 — dunia sintetik di lattice nyata.** Intensitas diketahui (4 driver terrain; surface B
menyembunyikan clay dari fitur = misspecification realistis), bias survei jalan diterapkan sengaja
(fungsi persis asumsi TGB — rasional diberi kesempatan terbaiknya), pipeline **kode yang sama** dengan
eksperimen nyata, dan setiap konfigurasi diskor terhadap (i) sampel presence tak-bias yang ditahan,
(ii) permukaan intensitas itu sendiri. 2 surface × 10 dunia × 6 konfigurasi × 3 algoritma.

**Hasil (dokumen apa adanya, termasuk yang tidak nyaman):**

| Besaran | random | tgb | hybrid(0.0) | hybrid(1.0) |
|---|---|---|---|---|
| auc_own ("angka laporan") | 0.847 | 0.840 | 0.737 | **0.890** |
| auc_true (kebenaran) | **0.828** | **0.827** | 0.541 | 0.617 |
| inflasi | +0.019 | +0.012 | +0.196 | +0.273 |

- Aturan seleksi naskah memilih hybrid(1.0) di **60/60 kasus** (laporan 0.890 > random 0.847), padahal
  kebenaran bilang random lebih baik **+0.21 AUC** (biaya median +0.194, 100% positif; biaya peta
  0.35–0.53 Jaccard). **Patologi seleksi tereplikasi terhadap ground truth, di kedua surface.** Ini
  menutup M1 untuk klaim inti.
- **Koreksi wajib terhadap klaim kita sendiri (P1 gagal sebagaimana terdaftar):** slope gabungan
  Spearman(hard_frac, inflasi) = +0.44 < ambang 0.5. Per-run: median +1.000, 100% > 0.5 — mekanismenya
  bulat, statistik pooled-nya terdilusi beda antar dunia (4 titik dial). Tapi lebih dalam: di dunia
  sintetik auc_true justru **naik** dengan hard_frac (0.54→0.62), sementara di data nyata auc_common
  **turun** (0.699→0.602). Artinya **tanda slope kebenaran itu kontingen terhadap rejim bias** —
  klaim "hard negatives selalu merusak" TIDAK digeneralisasi. Yang struktural dan selalu benar di kedua
  dunia: **angka laporan selalu terinflasi, dan dial menggerakkan angka laporan ~10× lebih cepat daripada
  kebenaran, ke arah mana pun.** Naskah wajib membingkai ulang klaim mekanismenya persis setajam ini
  (dari "slope negatif" menjadi "inkomparabilitas struktural + seleksi rusak"). Ini perubahan klaim
  ketiga yang saya tuntut sebagai syarat (R1 di §Putusan).
- Wawasan mekanistik baru (wajib masuk naskah): di dunia terkonsentrasi, **kuota regional hybrid
  menyuntikkan false negative** — negatif ditarik masuk ke klaster presence, dan sel-sel berhabitat baik
  dilatih sebagai negatif. Itu menjelaskan mengapa hybrid jauh lebih buruk di sintetik daripada di data
  nyata. Ini bukan bug dunia sintetik; ini diagnosis tentang *kapan* desain itu berbahaya.

### M2 — "'No reliable benefit' itu absence of evidence."

**Penutup: E223-A (equivalence).** 95% CI (hybrid−random, 20 seed): **12/12 sel algo×eval-background
menolak +0.092** (tangga terbit). Kolom uniform: MaxEnt −0.033 (CI −0.039…−0.028 — bahkan menolak 0),
XGB/RF ≈ −0.009. Klaim kini berbentuk positif: *manfaat sebesar yang diterbitkan ditolak*, bukan
"gagal menemukan manfaat".

### M3 — "Seed itu Monte Carlo, bukan replikasi. p-value Anda mengukur noise pipeline, bukan data."

**Penutup: E223-B (bootstrap blok spasial).** 30 replikasi resample blok presence (multiplisitas
dipertahankan), fit in-bag, skor OOB vs background uniform tetap. Semua algoritma: CI persentil 95%
menolak +0.092 (batas atas +0.008…+0.026). **Wajib ditulis jujur:** CI OOB lebar (±0.02–0.06); yang
tertolak tegas adalah tangga terbit; efek < ~+0.03 tak bisa disingkirkan pada n=378. Dan semua p-value di
naskah harus dilabeli unit replikasinya (R4 di §Putusan).

### M4 — "MaxEnt Anda satu konfigurasi. Regularisasi beda, kesimpulan beda?"

**Penutup: E223-C.** beta_multiplier 0.5–4.0 × 3 desain × 10 seed: hybrid−random = **−0.020…−0.022 di
semua beta** (1/10 seed positif). Tidak sensitif.

### M5 — "Narasi Boyce Anda kontradiksi internal."

**Serangan.** E218 memakai Boyce sebagai arbiter jujur; E220 P4 menunjukkan Boyce optimis di hard_frac
moderat. Jadi instrumen yang Anda pakai untuk menghukum instrumen orang lain pun menyimpang — dan Anda
tidak tahu apakah penyimpangannya ke arah kebenaran.

**Penutup: E222 P4 (Boyce diadili oleh kebenaran).** Boyce menempatkan random/tgb di atas semua hybrid
(benar, sesuai kebenaran), dan menghukum ujung dial (benar). Tapi titik optimumnya tidak terkalibrasi
kebenaran (puncak di hf=0.7; kebenaran memilih 1.0 *di antara hybrid*; kesepakatan per-run median
+0.50/+0.54 — **borderline**). Putusan: **Boyce turun pangkat** dari "metrik jujur" menjadi "sanity
check arah dengan mode gagal yang diketahui". Protokol primer = availability terdeklarasi yang dipatok
(R2 di §Putusan).

### M6 — "k* = 7 itu arbitrer."

**Penutup: E223-D.** k* = 2–5 (J≥0.85), 4–7 (J≥0.90), 7–9 (J≥0.95). Rekomendasi dinyatakan rentang:
**4–7 seed untuk stabilitas praktis; 7–9 untuk publikasi peta.** Sensitivitasnya kini tertera.

### M7 — "Satu wilayah, satu dataset. Generalitas?"

**Penutup parsial: E222** memberi generalitas level mekanisme (lattice nyata, arkeologi sintetik,
termasuk misspecification — hasil konsisten A vs B). **Residual risk, tulis eksplisit:** replikasi di
wilayah kedua dengan rekaman nyata tetap pekerjaan lanjutan; klaim eksternal naskah dibatasi pada
"mekanisme + satu kasus arkeologis".

### M8 — "Peta berbeda, bukan berarti lebih baik — lalu peta mana yang harus dipakai manajer cagar budaya?"

**Serangan.** Paket babak 1 menghukum metrik tapi tidak menjawab pertanyaan pengguna. "Berbeda dan
berkonsekuensi" bukan rekomendasi.

**Penutup tiga lapis:**
1. **E221:** produk yang dapat dipertahankan = **inti robust** (top-desil di ketiga desain; densitas
   situs 2–5,6× fringe) untuk alokasi survei; fringe contingent = hipotesis, bukan target.
2. **E222 P3 (fork jatuh ke cabang NO):** TGB — padahal di dunia sintetik ia model bias yang *benar* —
   **tidak** memulihkan kebenaran lebih baik dari random (ΔJaccard −0.010, 47% positif; per algoritma
   tidak ada yang meyakinkan). Dalam rejim yang diuji, rasional "TGB untuk peta, bukan skor" tidak
   terbukti di peta maupun skor. (Batas kuat: satu rejim bias, daya n≈500 — bukan bantahan universal
   Phillips 2009; tulis persis begitu.)
3. **World C + World D (kedua fork terdaftar jatuh ke cabang NO):** kuota diuji di dua rejim yang
   seharusnya memihaknya. World C (bias survei regional [1.0, 0.4, 0.15, 0.05], kebenaran
   terkonsentrasi): quota(0.0) − random = **−0.246 AUC_true, −0.469 Jaccard, 0/30 positif**. World D
   (kebenaran diseimbangkan antar wilayah sehingga konsentrasi rekaman murni dari survei — rejim
   paling ramah kuota yang bisa kami bangun): **−0.203 AUC_true, −0.283 Jaccard, 0/30 positif**.
   Mekanismenya sama dan kini bernama: **mencocokkan background ke distribusi rekaman (TGB via jalan,
   kuota via wilayah) memusatkan negatif di tempat presence berkumpul; setiap kali rekaman
   berklaster — karena survei atau karena kebenaran — itu menyuntikkan false negative persis di
   tempat model paling harus belajar.** Di empat rejim sintetik, tidak ada desain yang mengalahkan
   uniform pada kebenaran, sementara AUC laporan selalu memilih desain paling ekstrem.

## Kritik Minor

- **m1.** Definisi turnover: sudah diwajibkan sejak babak 1; E221 Bagian C melaporkan kedua definisi
  (1−J: 28–47%; bagian-terganti: 16–31%). Pilih satu di naskah, nyatakan.
- **m2.** Dosis-respons yang dikutip harus versi 20-seed (E220: +0.967/−0.689); versi 5-seed (E218b)
  tetap sebagai jejak penemuan.
- **m3.** Putusan first-pass E221 tentang "gap desain lenyap di ensemble" salah referensi; sudah
  digantikan `02_split_half_control.py` (kontrol 5+5 yang cocok: gap hybrid **bertahan** di ketiga
  algoritma). Jejak koreksinya terdokumentasi — pertahankan di supplementary.
- **m4.** Environment: warning numpy 1.x/2.x muncul dari numexpr/bottleneck opsional; hasil tidak
  terpengaruh (semua angka terverifikasi ulang), tapi pin `requirements_freeze.txt` di supplementary.
- **m5.** E223-B melewati 1 replikasi (OOB<50) — n=29, dinyatakan.
- **m6.** Setiap p-value di naskah dilabeli unit replikasinya (seed = Monte Carlo; bootstrap = data).

## Putusan Review

**Minor-to-moderate revision, conditional pass** — paket ini bertahan dari pembacaan bermusuhan saya,
dengan empat syarat perubahan naskah (bukan eksperimen baru):

- **R1 (dari M1).** Bingkai ulang klaim mekanisme: bukan "hard negatives menurunkan generalisasi"
  (kontingen rejim), melainkan "evaluasi di background sendiri selalu terinflasi secara struktural; dial
  desain menggerakkan angka laporan ~10× lebih cepat daripada kebenaran ke arah mana pun; karena itu
  seleksi model pada angka laporan rusak secara prinsip". E222 membuktikannya di dua rejim berlawanan
  tanda — itu justru kekuatannya.
- **R2 (dari M5).** Boyce turun pangkat menjadi sanity check; protokol primer = availability
  terdeklarasi + dipatok + aturan seleksi dinyatakan.
- **R3 (dari M8).** Jawaban "peta mana" = inti robust + fringe sebagai hipotesis + desain terdeklarasi;
  null TGB (P3) dan kegagalan kuota di keempat rejim sintetik (C: −0.246/−0.469; D: −0.203/−0.283,
  0/30) diungkap apa adanya, termasuk mekanisme kontaminasinya.
- **R4 (dari M3).** Unit replikasi dilabeli pada setiap inferensi; pernyataan daya bootstrap (tolak
  +0.092; tak singkirkan < +0.03) masuk Methods.

**Residual risks (tulis eksplisit, jangan ditambal kata-kata):** (1) dunia sintetik = 2–3 rejim bias di
1 lattice; rejim lain ada (bias nonstasioner, bias berkorelasi kovariat non-jalan). (2) Null TGB bisa
terbatas daya. (3) Boyce borderline. (4) Replikasi wilayah kedua = pekerjaan lanjutan.

Tidak ada satu pun dari delapan kritik mayor yang bertahan sebagai alasan menolak — semuanya tertutup
eksperimen atau terkonversi menjadi pengakuan terukur. **Paket ini, dengan R1–R4, layak Q1.**

---

*Ditandatangani secara analisis: Go Frendi (via Claude Code). Bukti: E222 (`results/e222_runs.csv`,
`e222_selection.csv`, `e222c_*`, `e222d_*`) dan E223 (`results/e223*`).*

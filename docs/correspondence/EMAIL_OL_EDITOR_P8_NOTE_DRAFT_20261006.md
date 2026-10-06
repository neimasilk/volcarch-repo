# Catatan untuk editor *Oceanic Linguistics* sebelum revisi P8 ditulis (keputusan D7) — daftar fakta + kerangka, **bukan** naskah surat

**Status:** belum dikirim. Disiapkan 2026-10-06 (line 04, P8, MS OL-03-2026-11); disusun ulang dua kali pada hari yang
sama setelah pemeriksaan independen (kelengkapan terhadap surat keputusan; dua pembacaan skeptis).
**Yang mengirim:** PI, sebagai balasan atas email keputusan 2026-10-05. ⚠ Alamat yang tercetak di *Instructions for
Contributors* (oceanicl@hawaii.edu) **berbeda** dari alamat pengirim email keputusan — PI memastikan mana yang sampai
ke editor (membalas langsung email keputusan adalah jalan yang paling aman). Rekan penulis (Go Frendi Gunawan) setuju dulu.
**Gerbang G16:** surat kepada editor adalah suara penulis. Berkas ini sengaja **tidak** memuat naskah surat berbahasa
Inggris; isinya daftar fakta yang harus ada dan kerangka paragraf dalam bahasa Indonesia. PI menulis suratnya sendiri.
(Versi pagi berkas ini memuat draf Inggris lengkap; itu dihapus setelah pembaca skeptis menunjukkan bahwa draf semacam
itu akan terbawa kata demi kata ke surat.)
**Prasyarat:** daftar di bawah mengikuti keputusan yang *direkomendasikan* (D1–D6, D9;
`papers/P8_linguistic_fossils/REVISION_WORKPLAN.md` §2, §10.5). **Kirim hanya setelah PI mengukuhkannya**; butir yang
keputusannya berubah ikut berubah atau hilang (mis. bila D4 = pilihan (c), butir kelima A4 tentang hasil lintas daftar
gugur). Tidak ada yang diumumkan di sini yang tidak akan dikerjakan revisi.
**Mengapa catatan ini perlu:** revisi yang mengubah klaim dan abstrak tidak boleh tiba tanpa pemberitahuan;
reviewer 1 menyatakan sisi komputasinya di luar bidangnya, dan komentar reviewer 2 berupa permintaan penjelasan —
jadi besar kemungkinan editor tidak punya jalan lain untuk tahu (ini pertimbangan penulis, **bukan** untuk ditulis
ke editor); dan kebijakan biaya jurnal
tidak tertulis di mana pun (G15).
**Repo publik:** tidak ada teks reviewer dan tidak ada tautan portal di berkas ini.

---

## A. Fakta yang harus ada (urut)

1. Terima kasih; penulis akan merevisi menurut kedua laporan dan permintaan editor tentang keterbacaan.
2. **Asal temuan, apa adanya:** sebelum merevisi, penulis memeriksa ulang angka-angka naskah dari berkas ABVD mentah
   (audit 85 pernyataan, lalu bagian uji ketahanan; `experiments/E227_*`, `E228_*`, `E230_*`). Sebagian temuan dipicu
   oleh pertanyaan reviewer (ciri semantik dalam "phonological fingerprint"; ejaan hentian glotal; arti label dan
   "false positive"); sebagian melampaui yang mereka angkat. **Jangan** menulis bahwa reviewer tidak mengangkatnya.
3. **Apa yang salah, apa adanya:** berkas hasil yang tersimpan tereproduksi dari data mentah, tetapi naskahnya memuat
   beberapa angka dan dua arah pernyataan yang keliru, satu uji yang tidak menguji hipotesisnya, satu ukuran
   kesepakatan yang dihitung di dalam data latih, dan beberapa uraian yang tidak sesuai dengan yang dihitung (audit
   85 pernyataan: 47 cocok, 13 dengan catatan, 9 berbeda nilai, 11 tidak seperti diuraikan, 3 tidak berdasar, 2 terbalik
   arah; `experiments/E227_p8_g1_blind_rederivation/results/audit_summary.json`). Jangan
   menulis bahwa "hanya uraiannya" yang salah. Sifatnya, bukan hanya besarnya:
   - **dua langkah yang diuraikan di Metode tidak dikerjakan seperti diuraikan**: "pemeriksaan silang
     Proto-Austronesia" tidak membandingkan satu bentuk pun (semua bentuk tanpa kode untuk 15 makna dibuang), dan
     daftar pinjaman hanya cocok dengan lima kata yang tak berkaitan;
   - **beberapa angka tercetak adalah konstanta yang ditulis tangan di kode**, bukan hasil hitung dari data (variabel
     "cakupan kognat"; empat angka "asli" di Tabel 5);
   - **"kesepakatan dua metode"** bukan sekadar lebih rendah: metode kedua dilatih pada label metode pertama lalu
     dibandingkan dengan label itu; himpunan "konsensus" 266 bentuk ditarik (di luar data latih κ = 0,31, bukan 0,61).
4. **Yang berubah di abstrak:**
   - jumlah: satu definisi (tanpa penetapan himpunan kognat di ABVD) → 438 bentuk = 32,3 persen; 26,5 persen di abstrak
     milik himpunan kedua yang lebih kecil; daftar yang diminta reviewer 1 untuk dirilis karena itu memuat 134 bentuk
     Tolaki, bukan 114 seperti di Tabel 1 naskah terkirim;
   - penggolong: masukannya ciri bentuk tertulis **dan** makna, jadi "exclusively phonological" salah; model 0,76 juga
     diberi tahu dari daftar mana sebuah bentuk berasal; tanpa itu AUC 0,73, dan dari bentuk tertulis saja 0,67;
   - pernyataan "lebih sedikit prefiks" **ditarik**: data tidak menunjukkannya — menurut frekuensi mentah arahnya
     justru sebaliknya, dan pada panjang kata yang sama tidak ada beda yang bisa dipisahkan (E231 P3, P11). Katakan itu
     dalam satu klausa (editor akan bertanya ke mana arah datanya), tetapi **jangan** mengumumkan "lebih banyak
     prefiks" sebagai temuan;
   - klaim "tidak bergantung pada ejaan dan panjang kata" dipersempit: hasil sedikit bergantung pada cara hentian glotal
     ditulis, dan ukuran kata memang dipakai penggolong;
   - hasil negatif: uji di balik "tidak ada bentuk bersama" bukan uji yang tepat. Uji permutasi menemukan bahwa, **di
     antara bentuk tanpa kode**, bentuk semakna sedikit lebih mirip lintas daftar daripada bentuk tak semakna. Rumusan
     penggantinya ditentukan PI setelah membaca `REVISION_WORKPLAN.md` §10.7 (kosakata bersama yang dikenali ABVD
     ada di kelas berkode per definisi) — jangan mengumumkan rumusan yang lebih kuat dari itu;
   - enam belas bahasa tambahan: pola geografis tidak didukung; klaimnya ditarik.
5. **Apa yang kini diklaim naskah, dan apa yang tidak lagi:** bukan lagi "deteksi kosakata non-arus-utama dengan
   pembelajaran mesin"; yang tersisa adalah alat **pemeringkat** yang sedang-sedang saja (sebagai penggolong ya/tidak
   nyaris sama dengan jawaban trivial), uraian tentang apa arti "tanpa kode" untuk Makasar dan Tolaki, dan daftar
   bentuk yang dirilis untuk diperiksa ahli. Judul mungkin berubah (keputusan PI).
6. **Ukurannya, sebatas yang pasti:** gambar dari empat menjadi satu; bagian konsensus, pengelompokan, *ablation* dan
   ekspansi geografis hilang; §4.5 (aksara Jawa) dibuang *(hanya bila D6 dikukuhkan)*. **Jangan** menjanjikan "lebih
   pendek" sebelum naskah barunya ada.
7. **Yang ditambahkan:** untuk pertanyaan reviewer 1 tentang Makasar dan Tolaki ada hitungan dari data. Tolaki: 71
   persen bentuk tanpa kode punya padanan semakna di dalam Bungku–Tolaki (10,5 persen karena kebetulan; saringan
   mekanis, bukan etimologi; tidak berkata apa-apa tentang asal-usul). Makasar: ⚠ **kalimatnya ditahan** sampai PI
   membaca sumber angka 38 persen (`REVISION_WORKPLAN.md` §10.7) — yang pasti hanya bahwa Makasar punya lebih banyak
   makna tanpa kode daripada Bugis dan Sa'dan Toraja, dan bahwa hanya sedikit dari bentuk-bentuk itu punya padanan mekanis
   (2 dari 75 dengan bentuk PMP; 10 dari 80 dengan bentuk Bugis atau Sa'dan Toraja).
   Jangan menulis "konsisten dengan angka terbitan", "tidak diwarisi 60,7 persen", "belum dikodekan", atau "di
   antara makna berkode ketiganya sama" sebagai argumen. Daftar bentuk dirilis dengan DOI.
8. **Satu rujukan** yang disitir di naskah (`ross2005`) tertulis sebagai artikel jurnal ini padahal bab buku — PI
   memutuskan: disebut di sini, atau cukup di surat tanggapan.
9. **Pertanyaan kepada editor:** apakah koreksi ini boleh ditangani di dalam revisi yang diminta, masing-masing
   didaftar di bagian tersendiri surat tanggapan — atau editor menghendaki prosedur lain (misalnya dilihat lagi oleh
   reviewer)? Penulis mengikuti mana pun.
10. **Pertanyaan praktis:** adakah biaya bagi penulis (halaman, gambar berwarna, atau lain)? Penulis tidak mengambil
    opsi akses terbuka berbayar.

## B. Kerangka paragraf (isi tiap paragraf; kalimatnya milik PI)

| Paragraf | Isi |
|---|---|
| 1 | terima kasih; akan merevisi menurut laporan; keterbacaan Bagian 3 |
| 2 | sebelum menulis ulang, ada yang harus disampaikan: pemeriksaan ulang angka; berkas hasil tersimpan tereproduksi, tetapi naskah memuat angka, arah dan uji yang keliru serta uraian yang tidak sesuai; sebagian dipicu pertanyaan reviewer, sebagian lebih jauh; ini kesalahan penulis |
| 3 | sifat kesalahannya (butir A3) dalam dua–tiga kalimat |
| 4 | apa yang berubah di abstrak (butir A4) — daftar pendek |
| 5 | apa yang kini diklaim dan tidak lagi diklaim (A5); apa yang hilang (A6, hanya yang pasti) |
| 6 | apa yang ditambahkan (A7) |
| 7 | pertanyaan prosedur (A9) dan biaya (A10) |

## C. Butir pilihan — keputusan PI

- **Waktu.** Surat keputusan tidak menyebut tenggat. Menyebut tanggal = komitmen (G14 butuh jeda 14 hari setelah draf utuh).
- **Surat penerimaan resmi.** Surat keputusan menyebut surat penerimaan bersyarat sudah tersedia sekarang; tidak
  dikatakan apakah datang tanpa diminta. Bila PI membutuhkannya untuk institusi: catatan ini dulu, permintaan sesudah
  editor menjawab — bukan sebaliknya, dan bukan dalam napas yang sama dengan daftar koreksi. *(Rekomendasi: tidak diminta
  di catatan ini.)*
- **Format.** Halaman jurnal: versi akhir harus Word, LaTeX tidak didukung (`papers/P8_linguistic_fossils/VENUE.md`).
  Tidak perlu ditanyakan; itu keputusan D8.
- **Bantuan AI.** Pemeriksaan ulang angka dan analisis revisi dikerjakan dengan bantuan AI (naskah terkirim sudah
  memuat pernyataan AI). Apakah itu disebut di catatan ini atau cukup di pernyataan AI naskah revisi: keputusan PI.
- **Pracetak.** Versi arXiv (2604.00023) memuat angka-angka yang akan dikoreksi. Apakah editor diberi tahu sekarang
  bahwa pracetak akan diperbarui setelah keputusan jurnal: keputusan PI.

## D. Setelah dikirim

Catat tanggalnya di sini dan di `docs/WORKSTATE.md` §1; balasan editor tetap di kotak surat (di repo hanya parafrase);
jawaban atas A10 menutup gerbang G15 — bukti dicatat sebagai "email editor, tanggal".

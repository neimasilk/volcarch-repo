# P11 → SPAFA Journal — Submission Prep (2026-08-11)

**Status:** SIAP-KIRIM secara konten (SIG CONDITIONAL GO). Portal submission = PI.
**Draf:** `draft_v0.6_spafa.tex` → `draft_v0.6_spafa.pdf` (14 halaman, kompilasi bersih).
**Riwayat:** tolak 2× (Cornell *Indonesia* — scope; *Archipel* — editorial). Temuan inti selamat
(candi–pemukiman gap 6.78 km, 80.6% <10 km, p<1e-6, kebal inventori). Retarget: **SPAFA Journal**.

---

## 1. Yang SUDAH dikerjakan sesi ini

| Item | Status |
|---|---|
| Angka kanonik diterapkan ke teks (baris 116/121/139/171/172): 298°·p=1.2e-9·47.9%·9.2%·45.1%·19.1×·6.1 km·p=2.8e-7 | ✅ |
| Kalimat inventori 30-puncak ditambahkan (§Volcanic Landscape) | ✅ |
| **Re-derivasi buta (G1)** dari `canonical30/`: semua angka cocok persis (297.9°, p=1.217e-9, W47.9%, E9.2%) | ✅ |
| **Figur fig1+fig2 diregenerasi** dari canonical30 (sebelumnya hardcode 279°/47.2%/3.5%/3.4e-8 — inkonsisten dengan teks; pasti tertangkap reviewer) | ✅ |
| Toolchain `generate_figures.py` diperbaiki (fig4 boxplot API matplotlib; semua 5 figur ter-regenerasi) | ✅ |
| Kompilasi final 14 halaman, 0 undefined reference | ✅ |
| SIG sign-off jujur | ✅ `SIG_signoff.md` |

## 2. SIG sign-off ringkas

**CONDITIONAL GO** — 5 syarat: (1) re-derive E069 survey-control pada inventori kanonik · (2) G9
cross-model pada v0.6 · (3) cek format SPAFA · (4) soften G8 baris ~104 + verifikasi baris ~266 ·
(5) konversi format + submit = PI. Detail: `SIG_signoff.md`.

## 3. Yang BELUM (tindakan tersisa, urut prioritas)

1. **Cek format SPAFA Journal** — muat pedoman penulis situs SPAFA (SEAMEO SPAFA, Bangkok): panjang,
   gaya kutipan, format gambar, Word/LaTeX. ⚠ JANGAN submit sebelum ini. (Belum diverifikasi sesi ini —
   tidak boleh di-fabrikasi.)
2. **Re-derive E069** (`β=−0.477, p=0.0015`, baris ~259) dari data kanonik — cek apakah dependen
   inventori gunung; bila ya, hitung ulang.
3. **G9 cross-model** — jalankan `tools/critical_reviewer_prompt.md` pada v0.6 di DeepSeek; masukkan
   hasilnya ke `external_reviews/`.
4. **G8** — lunakkan baris ~104; verifikasi angka baris ~266 (391 situs, 277 candi, 5 pemukiman).
5. **Konversi + submit** (portal SPAFA = PI) — ikuti checklist format di item 1.

## 4. Cover letter (SPAFA)

```markdown
# Cover Letter — SPAFA Journal

**To:** The Editors, SPAFA Journal (SEAMEO Regional Centre for Archaeology and Fine Arts)
**From:** Mukhlis Amien, Universitas Bhinneka Nusantara, Malang, Indonesia
**Re:** Manuscript submission — "Temples Without Villages: Candi and the Hidden Settlement Geography of Volcanic Java"

Dear Editors,

We submit for your consideration the attached manuscript, "Temples Without Villages: Candi and the
Hidden Settlement Geography of Volcanic Java."

The paper argues that Java's surviving Hindu-Buddhist temples (candi) serve as spatial markers of a
settlement landscape that volcanic sedimentation has rendered invisible to archaeological survey.
Through analysis of 142 candi and 170 georeferenced inscriptions, we show that temples cluster on the
west-northwest flanks of the island's volcanoes — precisely the zones where burial by pyroclastic
deposits is most intense — while their orientation follows religious canon, demonstrating deliberate
landscape reading by their builders. Inscriptions sit on average ~6 km farther from volcanoes than
temples, marking a separate administrative geography. The 2008 discovery of the buried village of
Liangan validates this framework.

We believe this paper suits SPAFA Journal because it addresses a core Southeast Asian archaeological
question — the taphonomy of the region's volcanic landscapes and what it hides — with a method
(spatial statistics of standing monuments) directly relevant to heritage practice across the
archipelago.

The manuscript is approximately 3,200 words with 2 figures. It has not been published elsewhere and
is not under consideration at any other journal. All data and computational scripts are publicly
available (https://github.com/neimasilk/volcarch-repo).

**AI Disclosure:** AI tools (Claude, Anthropic) were used for spatial geocoding, statistical
computation, and manuscript drafting assistance. Research design, hypothesis formation, data
interpretation, and all conclusions are the authors' work.

Thank you for considering this submission.

Sincerely,
Mukhlis Amien
Lab Data Sains, Universitas Bhinneka Nusantara
Jl. Bunga Andong Selatan No.73, Malang 65141, Indonesia
amien@ubhinus.ac.id | ORCID: 0000-0002-1848-167X
```

*(Simpan ke `cover_letter_spafa.md`; perbarui jumlah kata saat format final.)*

## 5. Catatan strategis

- Temuan inti P11 **kebal inventori** (koreksi menguatkan) — ini argumen utama yang dipakai di
  abstract/cover: stabilitas terhadap koreksi kanonik adalah kekuatan, bukan kelemahan.
- Jika SPAFA menolak cepat (editorial), urutan cadangan tetap: Wacana (jika isu terbuka cocok) →
  PCI Archaeology. Jangan membuka line baru.

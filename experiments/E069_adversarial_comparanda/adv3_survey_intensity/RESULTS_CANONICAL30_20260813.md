# ADV-3 — Re-derivasi Kanonik 30 Puncak (WS-E / SIG G1) — 2026-08-13

**Konteks:** syarat 1 dari 5 menuju submit P11 → SPAFA (`SIG_signoff.md` 2026-08-11):
angka survey-control E069 (`β=-0.477, p=0.0015`) belum pernah di-re-derive pada inventori
kanonik 30 gunung.

**Metode:** `adv3_survey_intensity_canonical30.py` — identik dengan script Maret 2026 KECUALI
sumber gunung: hardcode 7 gunung Jawa Timur → `data/processed/dashboard/volcanoes_java_full.csv`
(30 puncak, Java-wide). Situs, grid 0.1°, proksi survei, model Poisson, uji LR quasi — semua sama.

**Hasil (buta, dari file hasil mentah):**

| Metrik | 7 gunung (2026-03) | 30 gunung kanonik (2026-08-13) |
|---|---|---|
| β volcano_dist (terstandardisasi) | −0.477 | **−0.831** |
| LR statistic | 35.64 | **92.28** |
| p (quasi-likelihood, terkoreksi overdispersi φ≈3.5) | 0.00154 | **2.86×10⁻⁷** |
| Δ pseudo-R² | 0.0161 | **0.0418** |
| Perbaikan AIC | 33.6 | **90.3** |
| Situs / sel valid | 666 / 703 | 666 / 703 (identik) |
| Verdict | VOLCARCH SUPPORTED | **VOLCARCH SUPPORTED** |

**Kesimpulan:** temuan **selamat dan menguat** — pola yang sama dengan P17 (WS-E 2026-08-03).
Dengan inventori penuh, sinyal vulkanik pada defisit situs justru lebih tajam. Naskah P11 baris
~259 + footnote diperbarui ke angka kanonik (β = −0.831, p = 2.9×10⁻⁷), dengan angka lama
disebut jujur sebagai pembanding.

**Berkas:** `results/canonical30/adv3_canonical30_results.json` ·
script `adv3_survey_intensity_canonical30.py`.

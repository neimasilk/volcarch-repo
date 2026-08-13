# E105 — Re-derivasi Kanonik 30 Puncak (WS-E / SIG G1) — 2026-08-13

**Konteks:** G9 cross-model (P11→SPAFA) menemukan bahwa persentase seksi 929 M naskah
(57/91/53/89%) dihitung 2026-03-17 pada inventori gunung lama — pola yang sama yang membunuh P17.
**Fix:** re-run klasifikasi zone × topik dengan jarak kanonik 30 puncak.

**Metode:** `e105_rerun_canonical30.py` — join `E062/results/joined_dated_inscriptions.csv`
(per-prasasti pre-Indic ratio, kunci filename DHARMA) ×
`E082/results/canonical30/geocoded_inscriptions_canonical30.csv` (jarak kanonik `volcano_dist_km_c30`).
Aturan klasifikasi identik dengan E105 asli: Sanskrit <0.05 · Mixed 0.05–0.20 · Indigenous >0.20;
zone Volcano <15 km · Court 15–30 km · Periphery >30 km; split era 929 M.

**Hasil:**

| Angka | 7-gunung (2026-03) | Kanonik 30 (2026-08-13) |
|---|---|---|
| Pre-929 di court zone | 57% | **58.0%** (58/100) |
| — dari itu Sanskrit-dominant | 91% | **91.4%** (53/58) |
| Post-929 di periphery | 53% | **48.4%** (15/31) |
| — dari itu mixed/indigenous | 89% | **86.7%** (13/15) |
| n join (dated+geocoded+ratio) | 137 | **131** (pre 100 · post 31) |

**Kesimpulan:** arah temuan **tidak berubah** — pergeseran 929 M tetap dramatis; angka kanonik
dipakai naskah P11 v0.6 (58/91/48/87, dengan footnote re-derivasi). Naskah menyebut n=100/31 untuk
transparansi G1. Berkas: `results/e105_results_canonical30.json`.

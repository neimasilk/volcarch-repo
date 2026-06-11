# P11 — Koreksi Inventori Kanonik untuk Tahap Revisi (2026-06-10)

**Status naskah:** SUBMITTED Archipel 2026-04-08 (v0.5/v0.6), under review. **JANGAN kirim koreksi sekarang** — terapkan saat editor meminta revisi. Jika review kembali ACCEPT tanpa revisi, kirim erratum singkat ke editor SEBELUM produksi.

**Konteks:** Penolakan P7/Antiquity mengekspos inventori gunung api terpotong (7 gunung Jawa Timur). Inventori kanonik sekarang `data/processed/dashboard/volcanoes_java_full.csv` (30 gunung). Semua angka P11 yang bergantung jarak-ke-gunung telah di-re-derive (E031/E082 canonical re-runs, JOURNAL 2026-06-10). **Semua temuan P11 SELAMAT — dua malah menguat, satu mengecil tapi tetap signifikan.**

## Koreksi per lokasi (referensi baris = draft_v0.5_archipel.tex)

### 1. Baris ~116 (azimuthal/western flanks) — MENGUAT
| Angka | Lama (16 gunung) | Kanonik (30) |
|---|---|---|
| Mean bearing | 279° (west) | **298° (WNW)** |
| Candi di flank barat | 47.2% | **47.9%** |
| Kuadran timur | "fewer than 4%" (3.5%) | **9.2%** — ganti frasa, mis. "under 10%" |
| Rayleigh p | 3.4×10⁻⁸ | **1.2×10⁻⁹** |
Provenance lama: E031 pairs (16 gunung) — direproduksi persis 2026-06-10. Kanonik: `E031/results/canonical30/`.

### 2. Baris ~139 (Zone A overrepresentation) — MENGUAT
| Angka | Lama | Kanonik |
|---|---|---|
| Candi di Zone A (<10 km) | 42.3% | **45.1%** (64/142) |
| Overrepresentasi vs land area | 17.9× | **19.1×** |
Provenance: E065 (model cincin-konsentris, land-share Zone A 2.36%, max dist 65.1 km — tidak berubah). Direproduksi + re-derive 2026-06-10 (JOURNAL). Catatan jujur bila reviewer tanya: model land-area-nya kasar (disk konsentris tunggal); arah temuan tidak sensitif terhadap itu.

### 3. Baris ~171 + abstrak (gap candi-vs-inskripsi) — MENGECIL, TETAP SIGNIFIKAN
| Angka | Lama (20 gunung, ada Krakatau) | Kanonik (30 + Agung/Batur, tanpa Krakatau) |
|---|---|---|
| Mean gap | 9.2 km | **6.1 km** |
| Bootstrap 95% CI | 5.5–12.7 | **3.2–9.1** |
| Mann-Whitney p | 5.2×10⁻⁸ | **2.8×10⁻⁷** |
Provenance: E082. Kanonik: `E082/results/canonical30/e082_results_canonical30.json`. Median tidak berubah (candi 14.5 vs inskripsi 27.6 km — konsisten dengan P17).

### 4. Abstrak — ganti keempat angka sekaligus
"Rayleigh p=3.4×10⁻⁸" → 1.2×10⁻⁹; "17.9×" → 19.1×; "9.2 km" → 6.1 km; "p=5.2×10⁻⁸" → 2.8×10⁻⁷.

### 5. Metode — tambah satu kalimat
Sebutkan inventori: "Volcano locations follow a 30-peak inventory of active and recently active Java volcanoes (Smithsonian GVP-derived), replacing the partial inventory used in earlier project work." (sesuaikan wording sumber inventori bila perlu)

## Yang TIDAK perlu dikoreksi (diverifikasi 2026-06-10)
- **Baris ~110 / E153 Test 1:** "81% within 10 km, mean 6.8 km" — verifikasi ulang memberi 80.6% / 6.78 km, **cocok**. (Catatan WORKSTATE 2026-06-10 pagi yang menduga mismatch 9.2-km-vs-E153 adalah salah atribusi: 9.2 km berasal dari E082, bukan E153.)
- **Orientasi equinox (85%, p=4.9×10⁻¹⁴):** independen dari inventori gunung (entrance vs equinox).
- Klaim laju sedimentasi, Liangan, mekanisme taphonomic: tidak bergantung inventori.

## Test 3 (jika reviewer minta) 
Leg sekunder volcano-distance di E153 masih memakai list non-kanonik 14 gunung — bila direvisi besar, repoint ke kanonik 30; tidak mempengaruhi headline Test 1.

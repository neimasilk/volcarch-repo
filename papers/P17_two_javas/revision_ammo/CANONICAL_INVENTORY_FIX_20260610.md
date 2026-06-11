# P17 — Methods/Repro Fix Inventori Kanonik untuk Revisi ArchCalc (2026-06-10)

**Status naskah:** SUBMITTED ArchCalc ID 365 (2026-04-09), review mulai ~akhir 2026. **Terapkan saat revisi diminta** — jangan kontak editor sekarang (double-blind, antrian 2027).

**Masalah:** naskah memakai "10 major Java volcanoes" (Methods, baris ~169) untuk semua jarak — inventori non-kanonik. Headline SUDAH diverifikasi selamat dengan inventori kanonik 30 (`E104/rebuild_clean_full_inventory.py`, 2026-06-08: 14.5 vs 27.6 km, MW p=1.5e-7). Re-run E031/E082 (2026-06-10) memberi tabel pengganti lengkap di bawah.

## Penggantian angka (sumber: E031+E082 `results/canonical30/`, JOURNAL 2026-06-10)

| Lokasi | Lama (10-gunung) | Kanonik (30) |
|---|---|---|
| Median candi (abstrak + §hasil) | 14.6 km | **14.5 km** |
| Median inskripsi | 27.6 km | 27.6 km (tetap) |
| Mann-Whitney | U=8081, p<1e-6 | **U=8267, p=2.8e-7** (E104 region-matched: p=9.9e-4) |
| Candi 0–10 km | 60 (42.3%) | **64 (45.1%)** |
| Candi 10–20 / 20–30 / 30–40 / >40 | (lihat tabel lama) | **25 (17.6%) / 36 (25.4%) / 8 (5.6%) / 9 (6.3%)** |
| Inskripsi 0–10 km | 22 (12.5%) | **23 (13.1%)** |
| Inskripsi 10–20 / 20–30 / 30–40 / >40 | — | **50 (28.6%) / 70 (40.0%) / 26 (14.9%) / 6 (3.4%)** |
| Fisher volcano(<20) vs court(20–40) | 1.86×, p=0.012 | **konsentrasi 1.72×, OR=2.66, p<1e-4** (menguat signifikansinya) |
| n inskripsi | 176 | 175 (Java/Bali box, E082 kanonik) |

**Kesimpulan tidak berubah:** segregasi sacred-vs-administrative bertahan, beberapa uji malah menguat. Gap median tetap 13 km.

## Perubahan teks Methods
1. Ganti "the nearest of 10 major Java volcanoes" → "the nearest of 30 active and recently active Java volcanoes (canonical inventory, Smithsonian GVP-derived; `volcanoes_java_full.csv` in the project repository)".
2. Baris ~110 "45 active volcanoes \citep{GVP2024}" — klaim literatur deskriptif, boleh tetap, tapi tambah satu kalimat yang menjelaskan subset analisis: "Of these, the 30 peaks with Holocene activity on Java proper form the distance-analysis inventory."
3. Regenerasi Figure (distribusi jarak) dari data kanonik — `generate_figures.py` perlu repoint ke `canonical30/` CSVs.
4. Transparansi: tambahkan 1–2 kalimat di Limitations bahwa angka v1 memakai inventori 10-gunung dan revisi memakai 30-gunung dengan hasil yang stabil (reviewer menghargai ini; konsisten dengan SUBMISSION_INTEGRITY_GATE).

## Jejak verifikasi
- `experiments/E104_court_zone_hypothesis/results/e104_court_zone.json` (clean_rebuild_2026_06_08) — verifikasi headline.
- `experiments/E031_candi_orientation/results/canonical30/` + README §RE-RUN — sisi candi.
- `experiments/E082_inscription_georeferencing/results/canonical30/` + README §RE-RUN — sisi inskripsi (termasuk catatan Krakatau dibuang).
- JOURNAL 2026-06-10 (entri purge integritas).

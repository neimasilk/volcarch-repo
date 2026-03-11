# Draft Papers Pipeline

Folder ini berisi draft paper yang belum memasuki fase eksperimen aktif. Semua draft berstatus **INCUBATION** kecuali dinyatakan lain.

Filosofi: *"Santai dalam waktu, serius dalam metode."*

## Standar

- **Format:** Markdown only. PDF originals disimpan di `archive/` sebagai referensi.
- **Naming:** `PNN_short_name.md` (e.g., `P10_archaeological_biosignatures.md`)
- **Gate:** Sebelum eksekusi, setiap paper harus melewati: hipotesis testable + data accessible + metodologi executable.
- Paper yang sudah aktif/submitted ada di `papers/P[N]_*/`, BUKAN di sini.

## Katalog

### Papers Aktif (di `papers/`, bukan di sini)

| # | Judul | Status | Lokasi |
|---|-------|--------|--------|
| P1 | Taphonomic Framework | Draft complete, single-author submission imminent | `papers/P1_taphonomic_framework/` |
| P2 | Settlement Suitability Model | Draft complete, single-author submission imminent | `papers/P2_settlement_model/` |
| P5 | The Volcanic Ritual Clock | **SUBMITTED** (BKI) | `papers/P5_volcanic_ritual_clock/` |
| P7 | Temporal Overlay Matrix | **SUBMITTED** (Antiquity) | `papers/P7_TOM/` |
| P8 | Linguistic Fossils | Draft v0.1 complete, needs red-team | `papers/P8_linguistic_fossils/` |
| P14 | Pararaton Volcanic Collapse | Draft v0.1, pivoting to research note | `papers/P14_pararaton_collapse/` |

### Draft Papers (di folder ini)

| # | File | Judul | Maturity | Data? | Prioritas |
|---|------|-------|----------|-------|-----------|
| P7-full | `P07_temporal_overlay_matrix.md` | TOM Full-Length (canonical v3.1) | Full draft | PARTIAL | REF (short version submitted) |
| P9 | `P09_peripheral_substrate.md` | Peripheral Conservatism | Partial draft | PARTIAL | MEDIUM |
| P9-alt | `P09alt_borehole_archaeology.md` | Borehole Archaeology | Raw idea | PARTIAL | MEDIUM |
| P11 | `P11_volcanic_cultural_selection.md` | Volcanic Cultural Selection (VCS) | Partial draft | PARTIAL | MEDIUM |

### Killed / Dissolved (Mata Elang #3, 2026-03-10)

| # | File | Reason |
|---|------|--------|
| P4 | `P04_estuarine_hybrid.md` | Stub, no data, no path to execution |
| P6 | `P06_linguistic_phylogenetics.md` | Depends on P8 + linguist, too speculative for 2026 |
| P10 | `P10_archaeological_biosignatures.md` | Requires fieldwork, no partner |
| P12 | `P12_computational_mythology.md` | Requires corpus construction, no corpus |
| P15 | `P15_terminology_without_structure.md` | **Dissolved** into `papers/P5_volcanic_ritual_clock/revision_ammo/` — salami-slicing risk if separate |
| P-cst | `Pcst_coastal_taphonomy.md` | Stub, no data, no method |

**Ide-ide dari killed papers TIDAK hilang.** Semuanya sudah dipindahkan ke `docs/IDEA_REGISTRY.md` dengan ID unik:
- P4 → I-045 (estuarine resilience), I-054 (Surabaya-Venice), I-055 (Mongol 1293)
- P6 → I-022 (KawiKupas), I-023 (Kawi clustering)
- P10 → I-052 (tephrochronology), ADS framework ideas (I-082, I-083)
- P12 → I-004 (Panji-Malagasy), I-009 (carangan wayang), I-021 (myth classifier), I-085 (La Galigo), I-086 (Batara Kala)
- P-coastal → I-029 (Sunda Shelf bathymetry), I-080 (Pertamina sonar)

Lihat `docs/TRIGGER_MAP.md` untuk kondisi yang akan meng-unblock ide-ide ini.

### Dokumen Lain

| File | Deskripsi |
|------|-----------|
| `manifesto.md` | Grand narrative — 4 lapisan kegelapan (internal, bukan publikasi) |
| `parking_lot_vcs_colonial.md` | Raw ideas: VCS + colonial resistance + population estimates |
| `working_note_ancient_dna.md` | Working note: aDNA preservation paradox di Java |

### Strategy (di `docs/`)

| File | Deskripsi |
|------|-----------|
| `docs/master_attack_map.md` | 11 evidential channels × paper mapping (consilience framework) |

## Konflik Nomor (Resolved)

| Nomor | Konflik | Resolusi |
|-------|---------|----------|
| P9 | Peripheral Substrate vs Borehole Archaeology | **P9 = Peripheral Substrate** (konsisten dgn memory). Borehole = `P09alt` |
| P14 | Pararaton Collapse vs VCS Colonial (parking lot) | **P14 = Pararaton**. VCS Colonial → `parking_lot` |
| P15 | ~~Terminology Without Structure~~ | **DISSOLVED** into P5 revision ammo (2026-03-10) |

## Urutan Eksekusi yang Disarankan (post Mata Elang #3)

1. **P9** (Peripheral Substrate) — partial draft, connects P5+P8 findings
2. **P11** (VCS) — needs P5+P9 as foundation

*P14 moved to `papers/P14_pararaton_collapse/` (pivoting to research note). P8 moved to `papers/P8_linguistic_fossils/` (draft complete). P4, P6, P10, P12, P15, P-coastal killed/dissolved.*

## Archive

`archive/` berisi PDF original (docx→pdf export) dari draft awal. Disimpan sebagai referensi historis. Konten kanonik ada di file markdown.

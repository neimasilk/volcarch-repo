# Panduan Upload Zenodo — D1 (CARJ) + D2 (Mini-NusaRC)

**Tujuan:** dua DOI sitable hari ini. Ini mengakhiri status "0 output eksternal" (ME#19 forcing function) tanpa biaya dan tanpa risiko artefak — kedua dataset sudah diverifikasi ulang blind hari ini (lihat bagian Verifikasi di bawah).

**Waktu:** ~15 menit per dataset. Butuh akun Zenodo (login bisa via ORCID).

> **STATUS: ✅ EXECUTED 2026-08-11.** D1 (CARJ) published → `10.5281/zenodo.21882007`; D2
> (Mini-NusaRC) → `10.5281/zenodo.21882247`. Keduanya via sesi portal (login ORCID). Panduan ini
> disimpan sebagai rekam jejak / resep untuk deposit berikutnya.

---

## Persiapan (sekali saja)

1. Buka https://zenodo.org → **Log in** → pilih **Sign in with ORCID** (0000-0002-1848-167X).
2. Setelah login: tombol **New upload** (kanan atas).

---

## Upload 1 — D1: Colonial Archaeological Register of Java (CARJ) v1.0

**File yang di-upload** (dari `papers/D1_colonial_register/zenodo_upload/`) — upload sebagai 3 file lepas (JANGAN zip-nya, supaya CSV bisa di-preview di Zenodo):
- `colonial_site_register_v1.0.csv`
- `README.md`
- `REGISTER_NOTES.md`

**Metadata (copy-paste):**

| Field | Isi |
|---|---|
| Resource type | **Dataset** |
| Title | `The Colonial Archaeological Register of Java (CARJ): Site Observations from Dutch Oudheidkundig Verslag Reports, 1912-1929` |
| Publication date | 2026-06-10 |
| Creators | `Amien, Mukhlis` — Universitas Bhinneka Nusantara — ORCID 0000-0002-1848-167X |
| Description | salin bagian **Description + Key Statistics** dari `README.md` |
| License | **Creative Commons Attribution 4.0 International (CC BY 4.0)** |
| Keywords | `archaeology; Java; Indonesia; colonial archives; Oudheidkundig Verslag; burial depth; volcanic taphonomy; georeferenced dataset` |
| Version | `1.0` |
| Language | English |

**JANGAN diisi/diklaim:** afiliasi jurnal apa pun. Dataset ini BELUM pernah disubmit ke JOAD — README lama menulis "[submitted]", itu sudah saya koreksi di README paket.

Klik **Publish** → catat DOI yang muncul (format `10.5281/zenodo.XXXXXXXX`).

---

## Upload 2 — D2: Mini-NusaRC v3

**File** (dari `papers/D2_mini_nusarc/zenodo_upload/`) — 2 file lepas:
- `mini_nusarc_v3.csv`
- `README.md`

**Metadata:**

| Field | Isi |
|---|---|
| Resource type | **Dataset** |
| Title | `Mini-NusaRC: A Georeferenced Archaeological Site Database for Island Southeast Asia and Madagascar (1,200-1,600,000 BP)` |
| Publication date | 2026-06-10 |
| Creators | `Amien, Mukhlis` — Universitas Bhinneka Nusantara — ORCID 0000-0002-1848-167X |
| Description | salin **Description + Key Statistics** dari `README.md` |
| License | **CC BY 4.0** |
| Keywords | `archaeology; Island Southeast Asia; Indonesia; Madagascar; hominin sites; radiocarbon dating; georeferenced dataset; site discovery bias` |
| Version | `3` |
| Language | English |

Klik **Publish** → catat DOI.

---

## Setelah publish (saya yang kerjakan sesi berikut — tinggal tempel 2 DOI di chat)

1. Tulis DOI ke `MEMORY.md` + `WORKSTATE.md` + `JOURNAL.md`.
2. Update sitasi di README paket (ganti "DOI to be assigned" → DOI riil) via fitur **New version** TIDAK perlu — cukup catat di repo.
3. DOI masuk CV + bahan PhD track (Verberne/Lamqaddam): dua dataset terbuka tersitasi = sinyal eksekusi yang reliable.

---

## Verifikasi data (sudah dilakukan, 2026-06-10 — mini-G1 blind recompute)

| Klaim README | Hasil recompute | Status |
|---|---|---|
| D1: 52 records, 21 fields | 52, 21 | ✓ |
| D1: coords 43 (83%) | 43 | ✓ |
| D1: depth n=32, range 0.60-9.14, mean 2.88 | 32 nonzero (dari 34 terisi; 2 surface=0 m), range/mean exact | ✓ (README dikoreksi: jelaskan 34 vs 32) |
| D1: **median 2.00** | **1.75** (median 2.00 hanya berlaku subset measured-only n=27) | ✗→ DIKOREKSI di README+NOTES paket |
| D2: 80 records, 17 fields, 8 regions | 80, 17, 8 | ✓ |
| Sitasi "JOAD [submitted]" | D1/D2 tidak pernah disubmit ke JOAD | ✗→ DIKOREKSI (sitasi Zenodo + "data paper in preparation") |

Sumber kanonik tidak diubah — koreksi hanya di salinan `zenodo_upload/`. Koreksi median juga perlu di-backport ke `experiments/E070_.../REGISTER_NOTES.md` (belum dilakukan, menunggu konfirmasi).

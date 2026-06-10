# Panduan Pendaftaran HKI — VOC-ArchNLP v1.0.0

*Hak Cipta Program Komputer ke DJKI*

---

## Jenis HKI yang Didaftarkan

**Hak Cipta — Program Komputer** (bukan Paten)

- Pendaftaran Hak Cipta untuk program komputer memberikan perlindungan hukum dan kepastian hak tanpa perlu membuktikan kebaruan (novelty) seperti paten.
- Nilai KUM: ±20–25 poin untuk keperluan kenaikan jabatan akademik (Lektor → Lektor Kepala).
- Biaya: Rp 400.000 (perseorangan, online via e-hakcipta.dgip.go.id).

---

## Portal Pendaftaran

**e-hakcipta.dgip.go.id** (Sistem Online DJKI)

---

## Dokumen yang Dibutuhkan

| No | Dokumen | Status | File |
|---|---|---|---|
| 1 | KTP pencipta (Mukhlis Amien) | Perlu scan | — |
| 2 | Deskripsi program (Bahasa Indonesia) | **SIAP** | `docs/HKI/DESKRIPSI_PROGRAM.md` |
| 3 | Manual pengguna | **SIAP** | `docs/HKI/MANUAL_PENGGUNA.md` |
| 4 | Arsitektur sistem | **SIAP** | `docs/HKI/ARSITEKTUR_SISTEM.md` |
| 5 | Sampel kode sumber (≤50 halaman) | Perlu cetak | Lihat di bawah |
| 6 | Surat pernyataan keaslian (bermaterai) | Perlu tanda tangan | Template di bawah |
| 7 | Surat kuasa (jika melalui LPM Ubhinus) | Opsional | — |

---

## Sampel Kode Sumber (Item 5)

DJKI mensyaratkan sampel kode sumber program. Sertakan 4 berkas berikut (±50 halaman A4, font Courier 10pt):

1. `tools/voc_archnlp/__init__.py` (metadata package)
2. `tools/voc_archnlp/extractor.py` (komponen utama baru)
3. `tools/voc_archnlp/pipeline.py` (orkestrasi)
4. `tools/voc_archnlp/cli.py` (antarmuka CLI)

Tambahkan header pada setiap halaman: **"VOC-ArchNLP v1.0.0 — Hak Cipta Mukhlis Amien, 2026"**

---

## Template Surat Pernyataan Keaslian (Item 6)

```
SURAT PERNYATAAN KEASLIAN CIPTAAN

Yang bertanda tangan di bawah ini:
Nama           : Mukhlis Amien, S.Kom., M.Cs.
NIK            : [NIK Anda]
Pekerjaan      : Dosen
Instansi       : Universitas Bhinneka Nusantara (Ubhinus), Malang
Alamat         : [Alamat Anda]

Menyatakan bahwa Ciptaan berupa Program Komputer dengan judul:

"VOC-ArchNLP: Sistem Penambangan Arsip Kolonial Belanda untuk
Data Arkeologi Indonesia, Versi 1.0.0"

adalah karya asli ciptaan saya sendiri, bukan merupakan jiplakan,
tiruan, atau saduran dari ciptaan pihak lain tanpa izin.
Pernyataan ini saya buat dengan sesungguhnya dan apabila di kemudian
hari terbukti tidak benar, saya bersedia menanggung segala akibat
hukum yang timbul.

Malang, [TANGGAL]

Materai Rp 10.000

[Tanda Tangan]

Mukhlis Amien, S.Kom., M.Cs.
```

---

## Informasi yang Diisi dalam Formulir Online

| Field | Isian |
|---|---|
| Jenis Ciptaan | Program Komputer |
| Judul | VOC-ArchNLP: Sistem Penambangan Arsip Kolonial Belanda untuk Data Arkeologi Indonesia |
| Sub-jenis | Perangkat Lunak (Software) |
| Tahun selesai dibuat | 2026 |
| Tahun pertama kali diumumkan | 2026 |
| Negara pertama kali diumumkan | Indonesia |
| Pencipta | Mukhlis Amien |
| Pemegang Hak Cipta | Mukhlis Amien (perseorangan) ATAU Universitas Bhinneka Nusantara (jika via lembaga) |
| Uraian singkat | Sistem NLP untuk mengekstraksi data arkeologi dari arsip koloni Belanda (VOC, 1602–1799) |

---

## Opsi Pendaftaran: Perseorangan vs. Lembaga

### Opsi A: Perseorangan (lebih cepat)
- Pemegang: Mukhlis Amien
- Biaya: Rp 400.000
- Proses: ±1–3 bulan

### Opsi B: Via LPM/LPPM Ubhinus (direkomendasikan untuk KUM)
- Pemegang: Universitas Bhinneka Nusantara
- Pencipta tetap: Mukhlis Amien
- Nilai KUM lebih tinggi karena tercatat sebagai luaran penelitian institusional
- Hubungi: LPPM Ubhinus untuk formulir internal

---

## Catatan Penting

1. **Simpan bukti tanggal pembuatan:** Git log (`git log --oneline`) membuktikan tanggal pembuatan kode. Sertakan sebagai lampiran.
2. **DJKI hanya mengakui hak cipta atas ekspresi (kode), bukan ide:** Program ini asli karena kombinasi 4 komponennya unik; tidak perlu membuktikan kebaruan secara teknis.
3. **Hak cipta otomatis berlaku sejak dibuat**, pendaftaran DJKI hanya memberikan kepastian hukum dan dokumentasi.
4. **Zenodo deposit** (gratis, DOI permanen) bisa menjadi bukti timestamp tambahan: deposit ke zenodo.org sebelum mendaftar ke DJKI.

---

*Disiapkan oleh Claude Code untuk VOLCARCH/Ubhinus — 23 April 2026*

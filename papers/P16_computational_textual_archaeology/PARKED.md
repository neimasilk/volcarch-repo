# P16 — PARKED (2026-06-10, keputusan Pak Amien)

**Status: PARKED — jangan submit, jangan rewrite, sampai syarat unpark terpenuhi.**
Keputusan: "untuk arah P16 parkir dulu, catat semua" (Pak Amien, 2026-06-10, sesi lanjutan review menyeluruh). Ini = Opsi 3 dari tiga opsi yang diajukan (lihat di bawah).

## Mengapa diparkir

Pilar sentral paper — "cross-tradition convergence: 12 tradisi tekstual independen konvergen pada tema yang sama di ruang semantik" — **REFUTED oleh tes yang benar**, bukan sekadar dikritik.

Kronologi falsifikasi:
1. **2026-06-08 (G9 R0):** DeepSeek review pada `submission_wacana_v1.0.tex` → REJECT. W1: tes konvergensi v6 sirkular. R1 revision dikerjakan (tes "tradition-controlled" `e090_v6_tradition_controlled.py`, klaim 8/8 groups survive).
2. **2026-06-10 (G9 R1 re-run):** DeepSeek pada draft R1 → tetap REJECT. W1 (FATAL): v6 masih sirkular — membandingkan within-group similarity vs baseline **whole-corpus**, padahal passage masuk grup justru karena berbagi keyword → z positif nyaris terjamin. Fix yang dipreskripsikan: **label-shuffle within-group**.
3. **2026-06-10 (E090 v7):** Saya implementasikan tes preskripsi DeepSeek (`experiments/E090_transformer_textual_nlp/e090_v7_label_shuffle.py`): null = shuffle label tradisi DALAM tiap concept group; koherensi topikal dipegang konstan. **Hasil: konvergensi 0/8 grup** (v6 klaim 8/8). Semua z negatif besar (−5.8 s.d. −14.1): pasangan cross-tradition justru KURANG mirip daripada chance relabeling. Korroborasi internal: S_within > S_cross di semua 8 grup (mis. VOLCANO 0.422 vs 0.326).

Kesimpulan: kluster topikal disatukan oleh **homogenitas within-tradition** (gaya/genre per tradisi), bukan konvergensi lintas-tradisi. Ini artefak seleksi keyword, bukan sinyal.

**W2 (FATAL kedua):** klaim diachronic 929 CE bersandar n=46; max centroid drift adalah C11→C12, bukan 929. Preskripsi: HAPUS (bukan dilunakkan) jika paper dihidupkan lagi.

## Apa yang MATI vs apa yang SELAMAT

| Klaim | Status | Catatan |
|---|---|---|
| Cross-tradition convergence di ruang semantik (Finding 1, fig3) | **MATI** | E090 v7: 0/8, z −5.8 s.d. −14.1 |
| Klaim diachronic 929 CE | **MATI** (n=46, drift bukan di 929) | hapus total jika unpark |
| Distributional attestation: tema teratest di 11–12/12 tradisi (VOLCANO/MARITIME/METAL 12/12, JAVA 11/12) | **SELAMAT** (lebih lemah) | ini co-occurrence count, BUKAN convergence |
| Asimetri genre inskripsi (genre-honest) | **SELAMAT** | tidak tergantung tes konvergensi |
| Kritik XLM-R / cross-lingual limitation | SELAMAT | temuan metodologis jujur |

## Tiga opsi yang diajukan (2026-06-10) dan keputusan

1. Reframe + downgrade → paper distributional-attestation + genre asymmetry, drop 929 CE. (tidak diambil sekarang)
2. Switch venue ke DHQ — TIDAK menyelamatkan W1 (refuted di venue manapun). (ditolak)
3. **PARKIR sampai ada desain konvergensi non-sirkular. ← DIAMBIL (Pak Amien, 2026-06-10)**

## Syarat UNPARK (salah satu)

1. **Desain konvergensi non-sirkular tersedia dan lolos:** unsupervised clustering (mis. topic model / clustering embedding TANPA keyword tagging) yang me-recover tema lintas-tradisi secara mandiri — "gold standard" DeepSeek, belum pernah dites. Jika tes ini positif → pilar konvergensi bisa dibangun ulang dengan jujur.
2. **Keputusan Pak Amien untuk reframe-downgrade** (Opsi 1): paper lebih kecil berbasis distributional attestation + genre asymmetry. Bisa kapan saja; tidak butuh eksperimen baru, butuh rewrite §4–§5 + drop fig3 + drop 929 CE.

Jika di-unpark via Opsi 1, wajib lewat SUBMISSION_INTEGRITY_GATE penuh lagi (G9 cross-model ulang pada draft baru).

## Peta file (semua utuh, tidak ada yang dihapus)

- Naskah: `submission_wacana_v1.0.tex` (R1, 18pp, compile clean) — JANGAN edit selama parkir; `draft_v0.1.tex` (versi DHQ-arah lama)
- Status kanonik: `CANONICAL.md` (diupdate menunjuk file ini), `SIG_signoff.md` (G1–G8 pass; G9 = REJECT → NO-GO)
- Review eksternal: `external_reviews/critical_deepseek_p16_wacana_20260608.md` (R0), `external_reviews/critical_deepseek_p16_wacana_R1_20260610.md` (R1 re-run)
- Falsifikasi: `experiments/E090_transformer_textual_nlp/e090_v7_label_shuffle.py` + `V7_LABEL_SHUFFLE_FINDING_20260610.md` (write-up lengkap)
- Tes yang terbantah: `e090_v6_tradition_controlled.py` (sirkular — JANGAN dipakai lagi sebagai bukti)
- Jejak jurnal: JOURNAL 2026-06-10 (entri pertama)
- Cover letter Wacana: `cover_letter_wacana.md` (obsolet selama parkir)

## Catatan integritas

Ini contoh gate bekerja sebagaimana mestinya: pilar yang terbantah tertangkap SEBELUM mencapai venue Scopus. Tidak ada angka P16 yang pernah dipublikasikan/disubmit — kerusakan eksternal: nol. Per `feedback_confirmation_architecture`: kritik struktural dijawab dengan TES, hasil tes negatif diterima apa adanya, paper diparkir — bukan di-reword.

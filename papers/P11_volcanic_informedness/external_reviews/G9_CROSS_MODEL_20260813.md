# G9 Cross-Model Adversarial Review — P11 draft_v0.6_spafa — 2026-08-13

**Metode:** subagent adversarial, diprogram menolak; sumber: E031 canonical30, E082, E105, E129,
E153, E069 canonical30, E065. Laporan lengkap di transkrip sesi; ringkasan temuan + disposisi:

## Verdict inti
- **Tidak ada fabrikasi.** Semua angka headline kanonik yang bisa di-re-derive dari JSON cocok dalam
  pembulatan jujur. ADV-3 kanonik benar-benar menguatkan klaim survey-control.
- **Klaim inti selamat** (klaster sisi barat, divergensi candi↔inskripsi, kontrol survei).
- **CONDITIONAL GO** dengan paket perbaikan presisi di bawah — terutama seksi 929 M.

## Temuan dengan disposisi (V×C)
| # | Temuan | V×C | Disposisi | Status |
|---|---|---|---|---|
| 1 | 12.9% prasasti Zone A vs E082 kanonik 13.1% (23/175) | 2×1 | FIX | OPEN |
| 2 | "91% Sanskrit-dominant" salah-skop (E105: 53/58 court-zone pre-929) | 2×1 | FIX | OPEN |
| 3 | Seksi 929 M (57/91/53/89%) dihitung pada inventori superseded, tak pernah di-re-derive kanonik | 2×1 | FIX | OPEN — re-derive E105 kanonik |
| 4 | Tiga n beredar: 170 (naskah) vs 175 (E082) vs 174 (P17 terkoreksi) | 2×1 | FIX | OPEN |
| 5 | Kedalaman Liangan 6–8 m (naskah) vs 4–6 m (E153) + internal 3–9 vs 3–7 m | 2×1 | FIX | OPEN |
| 6 | 71% / 70.8% / 73% — tiga render satu angka | 1×0 | REJECT-with-reason | Sudah konsisten pasca-fix G8 (badan 71% = 70.8% dibulatkan; footnote menjelaskan 73.1% komposit) |
| 7 | E129 277 vs E153 283 klasifikasi beda atas database sama | 2×0 | PARK | OPEN — catat satu baris di naskah bila dipakai keduanya |
| 8 | p=2.9e-7 di footnote 259 adalah p uji LR, bukan uji koefisien | 2×1 | FIX-CHEAP | OPEN — wording |
| 9 | "Leeward" salah arah secara klimatologis (monsun tenggara musim kering → sisi barat = downwind) + caption vs badan kontradiksi | 2×2 | FIX | OPEN — tulis ulang mekanisme angin |
| 10 | Kalimat "continuously visible or rediscovered during colonial-era earthworks" dipalsukan oleh Sambisari (1966) & Kimpulan (2009) milik paper sendiri | 2×2 | FIX | OPEN — tulis ulang kalimat (G8 item 4 setengah-jadi) |

## Keputusan
Perbaiki #1–#5 + #8–#10 di naskah sebelum submit; #6 REJECT (sudah konsisten); #7 PARK (catatan
bila perlu). Re-derive E105 pada inventori kanonik 30 — ini pola P17 yang sama (angka 929-adjacent
P17 salah karena alasan ini).

## STATUS EKSEKUSI — 2026-08-13 (malam, sesi yang sama)
✅ **Semua FIX diterapkan di v0.6 DAN v0.7**: #1 13.1% · #2 skop "of those court-zone" · #3
re-derivasi E105 kanonik (58.0/91.4/48.4/86.7, n=100/31; `e105_rerun_canonical30.py`) · #4 n=182/175
· #5 Liangan 4–6 m + 2–9 m harmonisasi · #8 wording LR · #9 mekanisme angin ditulis ulang (monsun
tenggara; caption "downwind") · #10 kalimat Sambisari/Kimpulan. #6 REJECT (konsisten). #7 PARK
(footnote 266 mencatat 70.8% vs 73.1% komposit — 108+277+6=391 tertutup).

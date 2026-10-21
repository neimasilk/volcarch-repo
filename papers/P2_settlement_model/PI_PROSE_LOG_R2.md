# P2 round-2 revision — log of the PI's prose (G16 and the JCAA AI policy)

**Purpose:** a record, passage by passage, that the revised prose of v0.3 is the PI's own.
- The PI writes in Indonesian or rough English.
- Claude translates faithfully and checks numbers and language, without adding content.
- Every change Claude makes beyond translation is listed with its reason.

This is the "language assistance" that JCAA lists as acceptable (`VENUE.md` A5). It is declared in the AI statement.

---

## Passage 1 — Section 3.7, robust core and contingent fringe (v0.2 lines 458–460; v0.3 lines 416–418) — 2026-10-21

**Fixes:** C078. The v0.2 definition ("cells selected by every seed") and its caveat ("n = 2 in the non-volcanic arm") were wrong.

**PI's text (as written):**

> Ensemble ini menunjukkan perbedaan yang bermanfaat antara robust core (wilayah inti yang konsisten) dan contingent
> fringe (wilayah pinggiran yang bergantung pada desain latar). Robust core didefinisikan sebagai sel yang masuk dalam 10%
> peringkat teratas pada ketiga desain latar (random, TGB, dan hybrid), dengan setiap peta merupakan rata-rata dari 10
> seed; sedangkan contingent fringe adalah sel yang masuk dalam 10% peringkat teratas pada tepat satu desain. Kepadatan
> situs yang teramati pada wilayah inti lebih tinggi dibandingkan wilayah pinggiran, yaitu 1,93× untuk RandomForest (30,7
> dibandingkan 15,9 situs per 1.000 km²), 4,34× untuk XGBoost (40,8 dibandingkan 9,4), dan 5,62× untuk MaxEnt (31,7
> dibandingkan 5,7). Perbedaan ini menunjukkan konsistensi spasial, bukan validasi independen, karena situs yang digunakan
> untuk menghitung kepadatan tersebut juga digunakan untuk melatih model.

**Inserted into v0.3 (translation):**

> This ensemble reveals a useful distinction between a *robust core* (the consistent core area) and a *contingent fringe*
> (the peripheral area that depends on the background design). The robust core is defined as the cells that fall in the
> top 10% under all three background designs (random, TGB and hybrid), each map being the average of 10 seeds; the
> contingent fringe is the cells that fall in the top 10% under exactly one design. Observed site density is higher in the
> core than in the fringe: 1.93× for RandomForest (30.7 vs 15.9 sites per 1,000 km²), 4.34× for XGBoost (40.8 vs 9.4)
> and 5.62× for MaxEnt (31.7 vs 5.65; Figure 5). This difference indicates spatial consistency, not independent
> validation, because the sites used to compute these densities were also used to train the models.

**Changes beyond translation:**
- **MaxEnt fringe 5.7 → 5.65.** The source value is 5.65 (`e221_outcome.json`). With 5.7 a reader computes 5.56×, not
  5.62×. Told to the PI before insertion.
- **Figure reference** (Figure 5) kept from v0.2.

**Numbers re-derived from source (2026-10-21):** `experiments/E221_seed_ensemble_stability/results/e221_outcome.json`.
- RF 30.741 / 15.896 = 1.934
- XGB 40.785 / 9.404 = 4.337
- MaxEnt 31.74 / 5.65 = 5.618

The definitions match E221 README (Method, Part B) and the Figure 5 legend.

---

## Passage 2 — new Section 3.9 "Volcanic setting or elevation?" + Figure 7 (E235) — 2026-10-21

**Replaces:** the v0.2 paragraph "The terrain-matched control" (E219 part C; v0.2 lines 491–493), which E235 supersedes.
The paragraph answers R2 of round 2.

**PI's text (as written; the English in v0.3 lines ~466–485 is its translation, apart from the changes listed below):**

> Judul: Lingkungan Vulkanik atau Ketinggian?
>
> Paragraf 1 — Pertanyaan dan metode
>
> Analisis ini menguji apakah pola kesesuaian yang teramati mencerminkan lingkungan vulkanik atau sekadar pengaruh
> ketinggian. Informasi jarak ke pusat vulkanik (volcano_dist) tidak digunakan sebagai fitur pelatihan model, sehingga
> setiap pola yang berkaitan dengan lingkungan vulkanik dalam prediksi model harus tercermin melalui variabel medan yang
> digunakan. Wilayah dekat didefinisikan sebagai area berjarak ≤ 20 km dari pusat vulkanik Kuarter, sedangkan wilayah
> jauh berjarak ≥ 40 km; analisis mencakup 30 pusat vulkanik kanonik, termasuk yang berada di luar bingkai peta.
> Perbandingan dilakukan pada setiap pita ketinggian 200 m dengan pencocokan tepat bertingkat (coarsened exact matching)
> berdasarkan kemiringan lereng (slope), indeks kekasaran medan (TRI), dan indeks kebasahan topografi (TWI). Analisis
> menggunakan peta ensemble dari 10 seed yang telah tersimpan tanpa melatih ulang model. Ketidakpastian diestimasi
> menggunakan bootstrap blok spasial sebanyak 1.000 ulangan, dengan ambang keputusan 0,05 yang ditetapkan sebelum
> analisis dan desain yang dipra-registrasi di repositori publik.
>
> Paragraf 2 — Keterbatasan pemisahan di dataran tinggi (Gambar E235A)
>
> Di atas ketinggian 1.000 m, wilayah studi mencakup sekitar 3.434,5 km² daratan, dengan 97,48% berada dalam jarak
> ≤ 20 km dari pusat vulkanik Kuarter dan hanya 0,435% berada pada jarak ≥ 40 km. Daratan yang jauh dari pusat vulkanik
> Kuarter hanya mencakup sekitar 15 km² pada ketinggian 1.000–1.200 m dan tidak ditemukan di atas 1.200 m. Kondisi ini
> menunjukkan bahwa, di wilayah studi ini, pengaruh ketinggian dan kedekatan dengan pusat vulkanik Kuarter tidak dapat
> dipisahkan secara memadai pada ketinggian di atas sekitar 1.000 m. Oleh karena itu, penelitian ini tidak
> mengatribusikan pola tersebut secara khusus kepada ketinggian maupun lingkungan vulkanik. Batas keterpisahan ini dapat
> bergeser antara 800 dan 1.200 m, bergantung pada kriteria luas minimum yang digunakan.
>
> Paragraf 3 — Perbandingan di bawah 1.000 m (Gambar E235B)
>
> Di bawah 1.000 m, selisih kesesuaian antara wilayah dekat dan jauh dari pusat vulkanik Kuarter adalah +0,049 untuk
> desain random (SK 95%: +0,024 hingga +0,077), +0,049 untuk TGB, dan +0,060 untuk hybrid (SK 95%: +0,030 hingga +0,091).
> Berdasarkan aturan pra-registrasi, hasil desain hybrid memenuhi kriteria SPECIFIC, tetapi ringkasan lintas desain tetap
> INCONCLUSIVE karena dua dari tiga desain tidak menghasilkan kesimpulan yang tegas menurut kriteria tersebut. Analisis
> tambahan post hoc menunjukkan bahwa selisih ini berkisar antara +0,01 dan +0,07, bergantung pada definisi zona dan
> prosedur pencocokan; ketika jarak ke sungai turut dicocokkan, selisihnya turun menjadi +0,028. Wilayah pembanding yang
> jauh sebagian besar mencakup Pegunungan Selatan, yaitu busur vulkanik tua berumur Eosen–Miosen yang memiliki bentang
> alam karst, sehingga wilayah tersebut tidak tepat disebut sebagai wilayah nonvulkanik. Selain itu, contingent fringe
> MaxEnt pada Gambar 5 memiliki median ketinggian 1.107 m, berada di atas batas pemisahan yang dapat diandalkan; karena
> itu, pola tersebut tidak dapat ditafsirkan secara khusus sebagai efek ketinggian maupun efek vulkanik. Daya uji yang
> terbatas juga tidak memungkinkan penarikan kesimpulan yang kuat mengenai perbedaan antardesain.
>
> Usulan caption Gambar E235
>
> Gambar E235. (A) Luas daratan pada setiap pita ketinggian 200 m untuk wilayah dekat (≤ 20 km) dan jauh (≥ 40 km) dari
> pusat vulkanik Kuarter. (B) Selisih kesesuaian antara wilayah dekat dan jauh setelah pencocokan berdasarkan
> ketinggian, slope, TRI, dan TWI, beserta selang kepercayaan 95%; pita abu-abu menunjukkan ambang keputusan ±0,05.

**Changes beyond translation (each told to the PI before insertion):**
1. **Rounding.** 3,434.5 km², 97.48 % and 0.435 % → 3,435 km², 97.5 % and 0.4 %, to match the text printed in Figure 7A.
   Source: `e235_outcome.json` (3,434.52; 0.97479; 0.00435).
2. **One sentence made unambiguous.** The PI wrote that limited power does not allow a conclusion about "perbedaan
   antardesain". Read literally, that could be taken for the paper's main finding, which is about differences between
   designs. The English says what the test was about: whether the disagreement between designs differs between near and
   far land (E235 M3).
3. **Citation added** for "Southern Mountains = Eocene–Miocene volcanic arc": Smyth, Hall & Nichols 2008, GSA Special
   Paper 436, pp. 199–222, doi 10.1130/2008.2436(10). Metadata and editors (Draut, Clift & Scholl) checked on Crossref on
   2026-10-21; entry `smyth2008` added to `references.bib`; `references_jcaa.tex` rebuilt (30 cited entries).
4. **"XGBoost" added to the caption.** Panel B shows XGBoost only (`build_v03_figures.py`, `fig_elevation_control`).
5. **Mechanical:** "Three corrections" → "Two corrections" at the head of Section 3.8, since the third moved here.

**Open (TODO-PI in the .tex):** one sentence defining the verdict labels SPECIFIC / NOT SPECIFIC / INCONCLUSIVE.

**Numbers re-derived from source (2026-10-21):** `experiments/E235_elevation_vs_volcanic_control/results/e235_outcome.json`.
- M1: E* 1,000 m; land above it 3,434.5 km²; V share 0.9748; N share 0.00435.
- M2 (XGBoost):
  - random +0.0491 (0.0237 to 0.0768);
  - TGB +0.0489;
  - hybrid +0.0599 (0.0298 to 0.0914), SPECIFIC;
  - summary INCONCLUSIVE.
- Post hoc range +0.01 to +0.07 and river matching +0.028: E235 README, §Audit.
- 15 km² at 1,000–1,200 m: E235 README, M1.
- 1,107 m: E221 README, Part B.

**Build:** v0.3 compiles clean (31 pp.). The new subsection is Section 3.9 and the figure is Figure 7. Figure 6 is the
seed stabilisation. Alt-text order changes accordingly.

---

## Passage 3 — verdict labels (Section 3.9) and one Limitations item — 2026-10-21

**PI's text (as written):**

> (a) Definisi label di §3.9
>
> Label hasil mengikuti aturan pra-registrasi: SPECIFIC jika SK 95% tidak mencakup nol dan besar selisih
> sekurang-kurangnya 0,05; NOT SPECIFIC jika seluruh SK 95% berada dalam rentang −0,05 hingga +0,05; dan INCONCLUSIVE
> untuk semua hasil yang tidak memenuhi kedua kriteria tersebut.
>
> (b) Butir Limitations, pengganti baris 547
>
> Di atas ketinggian sekitar 1.000 m (dengan batas yang bergeser antara 800 dan 1.200 m, bergantung pada luas minimum
> yang digunakan), ketinggian dan kedekatan dengan pusat vulkanik Kuarter tidak dapat dipisahkan secara memadai di
> wilayah studi; karena itu, pola prediksi model pada wilayah tersebut, termasuk high-elevation fringe pada Gambar 5,
> tidak dapat diatribusikan secara khusus kepada salah satu faktor (§3.9).

**Inserted:** (a) as the last sentence of the first paragraph of Section 3.9; (b) in place of the v0.2 item "n = 2 in
the non-volcanic arm" in the Limitations list.

**Changes beyond translation:** one clause of (b) is set in bold, matching the other Limitations items (formatting only).

**Check:** the rule matches `experiments/E235_elevation_vs_volcanic_control/DESIGN.md` (verdict table, lines 77–79).
v0.3 compiles clean. The last "n = 2" in the manuscript is gone.

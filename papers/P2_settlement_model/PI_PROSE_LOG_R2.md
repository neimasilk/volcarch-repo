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

---

## Passage 4 — research question in the first paragraph of the Introduction (v0.3 lines 68–69 → 68–70) — 2026-10-22

**Answers:** R1-2 (state the research question earlier and more plainly). Writing session 2, item 1 of `WRITING_SESSIONS_R2.md`.

**PI's text (as written):**

> Perubahan cara pemilihan titik latar dapat meningkatkan nilai AUC pada validasi silang, tetapi peningkatan ini belum
> tentu menunjukkan bahwa model menjadi lebih baik. Nilai AUC bisa meningkat karena perubahan desain latar membuat tugas
> evaluasi menjadi lebih mudah, bukan karena model memiliki kemampuan prediksi yang lebih baik. Penelitian ini bertujuan
> untuk membedakan kedua kemungkinan tersebut dengan menguji bagaimana desain latar dan latar evaluasi memengaruhi hasil
> evaluasi model, menggunakan pemodelan kehadiran–latar (presence–background) untuk sebaran spesies di Jawa Timur.

**Inserted into v0.3 (translation):**

> Changing how background points are selected can raise the cross-validated AUC, but such an increase does not
> necessarily mean that the model has become better. The AUC can rise because the change in background design makes the
> evaluation task easier, not because the model has better predictive ability. This study aims to distinguish between
> these two possibilities by testing how the background design and the evaluation background affect the outcome of model
> evaluation, using presence–background modelling of the distribution of archaeological sites in East Java.

**Changes beyond translation (both told to the PI before insertion; PI: "ok"):**
- **"sebaran spesies" → "the distribution of archaeological sites".** P2 models archaeological sites (settlement
  suitability, 378 presences; abstract), not species; the method comes from species distribution modelling, which 1.1
  (line 74) and line 115 already say. A literal translation would have told a reader the paper is about species.
- **Placement:** the two context sentences of the old opening paragraph (lines 66–67: preservation and discovery,
  volcanism, label noise) are kept; the PI's paragraph replaces lines 68–69. The old line 69 ("The answer we offer is
  uncomfortable, because it is the answer we had to give about our own model") is removed with them.

**Check:** no numbers in the passage. v0.3 compiles clean (32 pp., 0 errors, 0 unresolved citations; references come from
`references_jcaa.tex`, so bibtex's "no \bibstyle" message is expected). The new text reads correctly on PDF page 2.

---

## Passage 5 — Table 1 "Key terms", end of Section 1.1 (new; v0.3 lines ~86–113) — 2026-10-22

**Answers:** R1-2 and R1-3 (explain the statistics for readers who are not statisticians). Writing session 2, item 2.
Facts given to the PI first (each checked against code or manuscript); the PI wrote one definition per term.

**PI's text (as written):**

> 1. Presence–background model: Model yang dilatih menggunakan situs yang diketahui keberadaannya (presences) dan titik
> latar (background) yang status keberadaannya tidak diketahui, sehingga menghasilkan skor kesesuaian relatif, bukan
> probabilitas keberadaan situs.
> 2. Background design (desain latar): Aturan pemilihan titik latar, yaitu random yang mengambil sampel secara seragam di
> seluruh wilayah, TGB yang membobot pemilihan berdasarkan jarak ke jalan dengan skala penurunan eksponensial 12 km, dan
> hybrid yang menggabungkan kumpulan titik TGB, kuota per wilayah, serta titik hard negative dengan target proporsi 30%.
> 3. Hard negatives: Titik latar yang kondisi lingkungannya jauh berbeda dari rata-rata lingkungan situs, dengan jarak
> lingkungan 2–5 simpangan baku berdasarkan zdist; dalam makalah ini, istilah tersebut digunakan untuk titik yang berbeda
> secara lingkungan dari situs, bukan titik yang mirip dengan situs sebagaimana lazim dimaknai dalam pembelajaran mesin.
> 4. Evaluation background (latar evaluasi): Titik latar yang digunakan untuk menilai kinerja model pada data uji; own
> background menggunakan desain yang sama dengan desain pelatihan model, sedangkan common background menggunakan satu
> kumpulan titik tetap yang sama untuk semua desain, yaitu sampel seragam dalam penelitian ini.
> 5. Availability domain: Wilayah geografis asal titik latar atau titik evaluasi yang menentukan konteks pembandingan,
> sehingga nilai metrik evaluasi harus ditafsirkan relatif terhadap wilayah tersebut (Barve et al., 2011).
> 6. AUC (Area Under the ROC Curve): Ukuran peluang bahwa sebuah situs acak memperoleh skor lebih tinggi daripada sebuah
> titik latar acak, dengan nilai 0,5 setara tebakan acak dan nilai 1 menunjukkan pemisahan sempurna; nilainya bergantung
> pada titik latar yang digunakan untuk evaluasi.
> 7. TSS (True Skill Statistic): Ukuran kinerja klasifikasi yang dihitung sebagai sensitivitas ditambah spesifisitas
> dikurangi satu pada ambang klasifikasi tertentu.
> 8. Continuous Boyce index: Metrik evaluasi presence-only yang membandingkan frekuensi situs teramati dengan frekuensi
> yang diharapkan pada berbagai kelas kesesuaian relatif terhadap sampel seragam yang tetap, sehingga perhitungannya
> tidak bergantung pada desain latar pelatihan.
> 9. Spatial block cross-validation (CV): Metode validasi silang yang membagi wilayah menjadi lima lipatan berupa blok
> spasial berukuran sekitar 50 km (0,45°), untuk mengurangi kedekatan geografis antara situs pada data latih dan data uji.
> 10. Seed ensemble: Peta gabungan yang diperoleh dengan merata-ratakan prediksi dari beberapa seed acak; dalam penelitian
> ini, peta dengan sel-sel pada 10% peringkat teratas mencapai kestabilan mulai tujuh seed, dengan Jaccard overlap
> minimal 0,90 terhadap ensemble 10 seed.
> 11. Jaccard overlap: Ukuran kesamaan dua peta yang dihitung dengan membagi jumlah sel yang termasuk dalam 10% peringkat
> teratas pada kedua peta dengan jumlah sel yang termasuk dalam 10% teratas pada setidaknya salah satu peta.
> 12. Coarsened exact matching: Metode pencocokan yang membandingkan dua kelompok hanya dalam strata yang memiliki anggota
> dari kedua kelompok, dengan pembobotan berdasarkan jumlah anggota kelompok yang lebih kecil; dalam §3.9, metode ini
> diterapkan pada setiap pita ketinggian 200 m.
>
> Catatan istilah: Definisi hard negatives sengaja menegaskan bahwa makalah ini menggunakan istilah tersebut secara
> berbeda dari pengertian yang lazim dalam pembelajaran mesin. Ini penting agar pembaca tidak salah memahami arah
> pemilihan titik latar.

**Inserted:** the English text in `submission_jcaa_v0.3.tex` (`tab:keyterms`), a faithful translation apart from the
changes below.

**Changes beyond translation.** Items (b)–(d) were told to the PI before insertion. The PI answered "pikirkan baik2..
saya ngikut kamu", so the final wording of (a) was decided by Claude under that delegation and reported right after.
- **(a) No. 3, hard negatives.** Translated as "Background points whose environment lies at a standardised distance of
  2–5 from the mean site environment (z, in standard-deviation units across the model's features; Section 2.3). In this
  paper the term denotes points selected by their distance from the typical site environment, not, as usually in
  machine learning, points that resemble the sites."
  - Why: a check before insertion (`experiments/E217_maxent_benchmark/03_zdist_of_sites_check.py`) found that **40.7 %
    of the 378 sites themselves lie in the 2–5 band** (44.2 % have z ≥ 2; site median 1.92, uniform frame 1.99). The
    points are therefore not "jauh berbeda" from the sites. What is true is the selection rule.
  - "2–5 simpangan baku" became "a standardised distance of 2–5 … in standard-deviation units across the model's
    features", because z is one distance over six features, not 2–5 SD on each variable.
  - "zdist" is a code name the manuscript never uses; the manuscript calls it z (Section 2.3).
  - The contrast with machine learning, the PI's point, is kept.
  - Ledger **C080** records the same finding for the existing Methods sentences (v0.3 lines 249–250).
- **(b) No. 4: "in this study mainly a uniform sample".** Section 3.2 also scores on target-group and stratified common
  backgrounds (line 338 of v0.2 numbering), so "a uniform sample" alone would contradict it.
- **(c) No. 8: "so that its reference sample does not depend on the training background design".** The Boyce value
  still depends on the model's predictions, which depend on the training design; only the reference sample is fixed
  (E218 code: `availability_bg`, the uniform draw).
- **(d) "Catatan istilah" not inserted:** a note to Claude; its content is in no. 3.
- **(e) Mechanical:**
  - one callout sentence at the end of 1.1: "Table 1 defines the key terms used throughout the paper." Tables are cited
    in the text.
  - numbering dropped; English only (the Indonesian glosses dropped); left column ragged-right.
  - the TGB expansion was not added (defined at first use in 1.1).
- **Consequences:** the other tables renumber automatically (2–5). The supplement cites manuscript tables by number,
  so `build_supplement.py` (v0.3 only) now shifts them by one ("Table 2" → 3, "Table 3" → 4). The same edit applies
  R1's "Section" rule to three table notes that still printed "§" (missed on 21 Oct). The v0.2 build is unchanged.

**Facts checked:**
- TGB decay 12 km and the hybrid target 0.30: E217 code L72, L76.
- z band 2–5: E217 L75; manuscript Section 2.3.
- 0.45°, 5 folds: manuscript line 185 (old numbering).
- k* 4–7 for Jaccard ≥ 0.90, k ≥ 7 recommended: manuscript Section 3.7 and Figure caption.
- CEM weights = min(near, far) per stratum: E235 code.
- Boyce on the uniform availability sample: E218 code.
- barve2011 in `references_jcaa.tex`.

v0.3 compiles clean (33 pp., 0 errors); Table 1 checked on PDF page 4. Supplement v0.3 rebuilt and compiled (7 pp.).

---

## Passage 6 — "no interior optimum" explained at first use (Introduction, after v0.3 line 136) — 2026-10-22

**Answers:** R1-2/R1-3 (explain at first use). Writing session 2, item 3.

**PI's text (as written):**

> Kriteria pemilihan berupa AUC yang dihitung menggunakan latar milik desain itu sendiri (own background) tidak
> menghasilkan titik optimum di tengah rentang porsi hard negatives: pada simulasi sintetis, AUC terus meningkat dari
> 0,7367 menjadi 0,8904 ketika porsinya dinaikkan dari 0 menjadi 1, sedangkan pada data nyata AUC meningkat mulai porsi
> 0,1, dengan satu penurunan kecil sebesar 0,0071 antara porsi 0 dan 0,1, sehingga nilai maksimum selalu berada di ujung
> rentang yang diuji dan kriteria ini tidak memberikan petunjuk kapan penambahan hard negatives harus dihentikan.

**Inserted** as a new sentence right after the proposition at line 136 (same paragraph); text in the .tex under the
"passage 6" comment.

**Changes beyond translation** (translation-level; made under the PI's "saya ngikut kamu" of the same morning and
reported right after):
- "simulasi sintetis" → "the synthetic worlds", the paper's own term for the same thing (Section 3.3).
- The gloss "(own background)" dropped: the English sentence already says "a design's own background".

**Numbers re-derived from source:**
- Synthetic: `experiments/E222_synthetic_ground_truth/results/e222_runs.csv`, hybrid configs, mean `auc_own` by
  `hard_frac` over both surfaces and all learners: 0.0 → 0.7367, 0.3 → 0.7820, 0.7 → 0.8412, 1.0 → 0.8904 (rising).
- Real: `experiments/E218_evaluation_artefact/results/e218b_hardfrac_sweep.csv`, mean `auc_own`: 0.0 → 0.7208, 0.1 →
  0.7137 (−0.0071), then rising every step to 1.0 → 0.8435. The maximum is at 1.0 in both.

---

## Passage 7 — Methods, hard negatives and the realised 0.62 (v0.3 lines 249–250; ledger C080) — 2026-10-22

**Fixes:** C080. The old line 250 explained the realised 0.62 with a TGB property that E217's own numbers contradict.

**PI's text (as written):**

> Dalam penelitian ini, hard negatives didefinisikan sebagai titik dengan jarak lingkungan terstandar (zdist) sebesar
> 2–5 dari rata-rata lingkungan situs (Tabel 1); meskipun proporsi yang ditargetkan dalam desain hybrid adalah 30%,
> proporsi yang terealisasi mencapai 0,62. Hal ini terjadi karena 70% titik sisanya diambil dari kolam latar yang sekitar
> separuh anggotanya sudah memiliki zdist ≥ 2, bukan karena bias khusus dari TGB, yang memiliki proporsi 0,485,
> mendekati proporsi pada sampel acak sebesar 0,503.

**Inserted:** the two sentences replace the old line 250 ("Note that the realised proportion …"). Line 251 ("This
pool-composition effect bears on the absolute AUC values …") is kept.

**Changes beyond translation** (under the same delegation; reported right after):
- **Placement:** the first half of the old line 249 is kept — the hybrid design blends a TGB pool, a regional quota and
  hard negatives, sweeping `region_blend` ∈ {0.0, 0.3, 0.5, 0.7} and `hard_frac` ∈ {0.0, 0.15, 0.30}. Only its
  parenthesis "(cells environmentally dissimilar from presences, 2.0 ≤ z ≤ 5.0)" is removed, because the PI's sentence
  now defines the band. Replacing all of line 249 would have dropped the sweep grid from the Methods.
- "zdist" → "z" (twice), as in Table 1 and in the old line 249.
- "yang memiliki proporsi 0,485" → "whose proportion of such points (0.485)", so the English says proportion of what.
- "sampel acak" → "the random background", the name of the design the 0.503 belongs to.
- The PI's "30%" and "0,62" are kept as written (mixed percent and decimal).

**Numbers re-derived from source:**
- 0.62: E013 README (0.62) and E217 (`e217_raw_results.csv`, hybrid, mean of 30 runs: 0.623).
- TGB 0.485 and random 0.503: the same file.
- "About half": the realised 0.623 = 0.30 forced from the band + 0.70 × s, so s = (0.623 − 0.30)/0.70 = 0.461 for
  the remaining 70 %. The TGB background alone has 0.485 with z ≥ 2, slightly less once the z > 5 tail is excluded.
- `experiments/E217_maxent_benchmark/results/e217_zdist_sites_check.txt`.

v0.3 compiles clean (33 pp., 0 errors, 0 undefined); both passages read back from the PDF.

---

## Session 3, part 1 — 41 language edits approved as one list (R1-1, R1-4) — 2026-10-22

**Not the PI's prose; listed here because they change the text.** The PI approved the list in one word ("bagian 1,
ya"). They are deletions of review history, "published" → "original"/"reported" (P2 was never published) and the
idiom substitutions R1 asked for. No sentence gains new content. The exact old → new pairs are in
`papers/P2_settlement_model/apply_session3_language_edits.py` (ids H1–H17, P1–P7, I1–I17); the dry run confirmed that
each old string occurred exactly once, then `--apply` wrote them.

- **H (review history):**
  - deleted: "Following a reviewer's taxonomy,"; the sentence "We state this restraint up front because the earlier
    version …"; the paragraph "The honest shape of this revision …"; "added at a reviewer's request"; "(answers the
    Maximum Entropy request)"; the clause "the submitted version quoted these gaps against the best run …"; "correcting
    an earlier internal claim"; "none prompted by the reviewers" (with its TODO comment); "With the site-prediction
    claim withdrawn,"; the sentence "This restraint is now stated in the manuscript …";
  - reworded: "a reviewer of the previous version observed — correctly — that" → "because this family …";
    "originally reported as evidence" → "initially read as evidence"; "the honest answer to the question of what makes"
    → "what makes"; "in the reviewer's own terms" → "in the terms of Section 1.2"; "this revision" → "this study" (2);
    "the site-prediction claim we have withdrawn" → "site-level prediction, which this design does not support".
- **P ("published"):** "published pipeline/values/ladder gain" → "original pipeline/values", "reported ladder gain"
  (7 places).
- **I (idioms):**
  - "dial" → "the share of hard negatives (0 to 1)", "the hard-negative share", "the end of the range" (8 places;
    only the invisible label `fig:dial` remains);
  - "judged at home", "judging at home", "home-court" → "scored on its own background", "own-background inflation"
    (6);
  - "rung(s)" → "step(s)" (2);
  - "map Jaccard said no" → "it did not support that reading";
  - "home regime" → "the regime it was built for".
  - "ladder" is kept; the PI defines it at first use (session 3 block A).

v0.3 compiles clean (33 pp., 0 errors, 0 undefined). Left for the PI in session 3: blocks A–E (lines 132–135, 255–258,
266–267, 473, 513–516). Left for session 4: abstract, conclusion, acknowledgements, contributions, AI statement.

---

## Passage 8 — session 3, part 2: blocks A–E (history passages that needed new content) — 2026-10-22

**Answers:** R1-1. The PI wrote five blocks in Indonesian. The English is in the .tex under the comments "passage
8A"–"8E". Changes beyond translation were made under the PI's standing "saya ngikut kamu" (same morning) and reported
right after.

**PI's text (as written):**

> **A.** Setiap desain latar dibandingkan dengan Maximum Entropy (MaxEnt), metode pemodelan presence–background yang
> umum digunakan (Elith et al., 2006), pada lipatan validasi silang yang sama; perbandingan ini menjadi analisis penentu
> dalam penelitian. Analisis awal menunjukkan suatu ladder, yaitu urutan tujuh desain (E007–E013) yang masing-masing
> dilaporkan menghasilkan AUC lebih tinggi daripada desain sebelumnya, dengan rata-rata AUC antar-seed meningkat dari
> 0,659 menjadi 0,751. Namun, hasil pada Section 3.2 menunjukkan bahwa kenaikan tersebut merupakan artefak evaluasi,
> bukan bukti bahwa desain latar yang semakin realistis menghasilkan model yang lebih baik. Dengan demikian, bahan
> empiris yang sama kini mendukung kesimpulan yang lebih terbatas: penilaian kinerja model presence–background sangat
> bergantung pada latar yang digunakan untuk mengevaluasinya.
>
> **B.** Selain perannya dalam pemodelan dan pembentukan desain latar, jarak ke jalan (road_dist) juga digunakan sebagai
> proksi visibilitas dalam rangkaian diagnostik. Pada Test 1, korelasi antara kesesuaian prediksi dan jarak ke gunung
> api, jarak ke jalan, serta jarak ke situs terdekat diperiksa; hasil diagnostik hanya dianggap gagal jika nilai absolut
> korelasi Spearman melampaui 0,5. Test 3 mengevaluasi kinerja model pada kuartil jarak ke jalan yang paling jarang
> disurvei, sedangkan komponen terkait gunung api dari Test 1 dilaporkan pada Section 3.8. Karena variabel yang
> digunakan untuk membentuk latar juga dapat memengaruhi nilai metrik yang dilaporkan, peran road_dist dalam pembentukan
> latar dan diagnostik perlu dijelaskan di bagian metode ini bersama kovariat lainnya.
>
> **C.** E014 merupakan evaluasi holdout berbasis proksi aksesibilitas: model dilatih menggunakan situs yang berjarak
> ≤ 1 km dari jalan dan diuji menggunakan situs yang berjarak > 1 km, sehingga pemisahan data mencerminkan perbedaan
> aksesibilitas, bukan perbedaan waktu. Model ini menghasilkan AUC sebesar 0,755.
>
> **D.** Korelasi yang tidak konsisten antar-seed. Pada versi sebelumnya, korelasi Spearman sebesar −0,163 dilaporkan
> berdasarkan satu instans model. Namun, pengulangan analisis dengan lima seed pada inventaris tujuh gunung api yang sama
> menghasilkan korelasi sebesar −0,243 (E219). Perbedaan ini menunjukkan bahwa hasil diagnostik tersebut dapat berubah
> antar-seed, sehingga interpretasinya perlu mempertimbangkan ketidakstabilan dalam satu konfigurasi analisis dan
> pentingnya protokol seed ensemble.
>
> **E.** Judul: Apa yang Tidak Dapat Ditunjukkan oleh Permukaan Kesesuaian, dan Desain Kuota yang Gagal. Kalimat:
> Permukaan kesesuaian tidak dapat membedakan mekanisme yang menyebabkan suatu situs tidak teramati. Karena itu,
> interpretasi bahwa kesesuaian prediksi yang rendah di dekat gunung api merupakan bukti adanya permukiman yang terkubur
> tidak dapat dipertahankan, sebab perbandingan model yang mendasarinya tidak bertahan dalam evaluasi menggunakan latar
> bersama (Section 3.2). [Note to Claude: the quota results and numbers stay.]

**Changes beyond translation:**
- **A1 — factual, from Claude's own fact list.**
  - "masing-masing dilaporkan menghasilkan AUC lebih tinggi daripada desain sebelumnya" → "whose reported AUC rose
    from 0.659 (E007) to 0.751 (E013, mean over seeds), with a single dip at E009". Table 4: E008 0.695 → E009 0.664.
    The rise is not step by step. The wrong "each better than the last" came from the facts Claude gave the PI.
  - "rata-rata AUC antar-seed … dari 0,659": 0.659 is E007's run value, not a seed mean; only 0.751 is.
- **A2 — "tujuh desain" → "seven experiments".** E007–E009 share one background design and differ in features. In
  this paper "design" means background design (Table 1).
- **A3 — "kenaikan tersebut merupakan artefak" → "the part of this rise attributed to increasingly realistic background
  designs is an evaluation artefact".** Section 3.2: on a common background, the redesign contributes −0.014 and the
  river feature +0.042. The feature part of the rise is real, so "the rise is an artefact" said more than the
  results. The PI asked to avoid statements that are too strong.
- **A4 — placement.** The PI's last sentence takes the place of the opening of the old line 136. The italic proposition
  (background designs cannot be compared unless the evaluation background is held fixed; no interior optimum) is kept
  after a semicolon, because passage 6 explains it right after.
- **A5 — "metode … yang umum digunakan" → "a widely used … method";** citation kept as \citep{elith2006}.
- **B1 — the PI's first sentence is folded into item (iv) of the line that lists the four roles** ("and (iv) serves as
  a visibility proxy in the diagnostic suite"). That line already gives roles (i)–(iii). "Perannya dalam pemodelan"
  is dropped because `road_dist` is never a training feature, as the same line says.
- **C1 — C replaces the three old E014 lines.** The old clause "road accessibility as a proxy for discovery order
  (sites far from roads stand in for sites found late)" is kept inside the PI's sentence, because it gives the reason
  the holdout means anything. The parenthesis about the mislabelled file and the "Response to Reviewers" is cut, as
  agreed; the record stays in the repository. "accessibility-proxy holdout" keeps its italics.
- **D1 — "Pada versi sebelumnya … dilaporkan" → "A single model instance gives …".** "Previous version" is review
  history (R1-1), so it is dropped. "for the same test" ties the number to Test 1 above. The bold run-in heading
  translates the PI's title.
- **E1 — the old sentence "Low predicted suitability near a volcano is not evidence of a buried site, in the terms of
  Section 1.2 …" is removed.** It said the same as the PI's first sentence. Its cross-reference moves into that
  sentence: "(Section 1.2)".

**Checked:** Table 4 (E007 0.659, E008 0.695, E009 0.664, E013 0.768 best / 0.751 seed mean); Section 3.2 numbers
(−0.0142, +0.0424); ρ −0.163 (single instance) and −0.243 (five seeds, 7 centres): claim set G1c,
`review_package_20260727/10_SET_KLAIM_TERKOREKSI.md`. v0.3 compiles clean (33 pp., 0 errors, 0 undefined). A rescan
finds review-history wording only in the session-4 places (abstract, conclusion, acknowledgements, contributions, AI
statement) and in the invisible label `sec:withdrawal`.

---

## Session 4, part 1 — back-matter defaults approved in one word ("ya") — 2026-10-22

**Not the PI's prose.** Four standard or mechanical changes; the PI approved the list as proposed.

- **Keywords 7 → 6** (JCAA limit, `VENUE.md` A3): "target-group background" dropped. "pseudo-absence design" covers
  it, and the term is defined in Table 1 and the Introduction.
- **Authors' Contributions:** "Both authors approved the withdrawal of the central claim reported in this revision."
  → "Both authors read and approved the final manuscript." (R1-1: no review history.)
- **Data Availability, Code:** "including E217--E224 and the verification script" → "including E217--E224, E235 and
  the verification script" (E235 is the Section 3.9 control).
- **Supplement:** the factual captions of Tables S7–S8 are kept; only the "[S7/S8 -- caption for the PI]" markers are
  removed (`build_e235_tables.py`). The four-sentence supplement intro is kept as it stands (no review history; it
  points to Sections 2.6 and 3.9, checked against the .aux). Both rebuilt: supplement 7 pp.; manuscript 33 pp., 0
  errors, 0 undefined.

Left for the PI: abstract, conclusion, the "strong gains" sentence, acknowledgements, AI statement.

---

## Session 4, part 2 — abstract, conclusions, "strong gains", acknowledgements: **drafted by Claude** — 2026-10-22

**Not the PI's prose.** The PI asked Claude to write these items ("apa bisa kamu kerjakan semuanya? tapi seminim mungkin
AI like, jadi minimalkan penulisan seperti AI"). This departs from G16 for these items, at the PI's request. Claude's
condition, told to the PI with the drafts: plain style is fine, but the AI statement must say that Claude drafted them.
JCAA counts "turning bullet points into prose, generating a prose based on your notes" as acceptable and non-disclosure
as unacceptable (`VENUE.md` A5).

How it was done:
- The abstract and conclusions reuse the PI's own translated sentences wherever they fit: passage 4 (research
  question), 8A (the sequence of seven experiments, 0.659 → 0.751, "the part of the rise attributed to … background
  designs is an evaluation artefact"), 6 (no optimum inside the range), 8E (a suitability surface cannot separate the
  reasons a site is unobserved), 2 (above about 1,000 m elevation and volcanic proximity cannot be adequately separated).
- Plain wording, short sentences, no review history ("reviewer", "withdraw", "originally reported", "dial" are gone).
- The script with every old → new pair is kept in the session scratchpad (`session4_drafts.py`); the .tex carries a
  comment at each drafted block.

Changes:
- **Abstract** (283 words in the PDF; limit 300). Same text goes into the portal metadata at upload.
- **Conclusions**, two paragraphs. The cause of the artefact is stated as in Section 3.2: on a common background the
  design change adds nothing (−0.014), and the hybrid design scores higher only on its own background (3/3 there,
  0/3 on each other background). The old "each design was scored against the negatives it had selected for itself"
  was too broad: the TGB steps are not explained by own-background inflation (Section 3.2, line on 22/60 cells).
- **"Strong gains" sentence** (C076): now "Recent applied studies report fair to good discrimination when predicting
  site locations from environmental variables (AUC of about 0.7 to 0.9; Castiello & Tonini 2021; Wang, Shi & Oguchi
  2023), and Comer et al. (2023) report that the way samples are drawn for training and testing can itself change the
  measured performance." Facts: `REFERENCE_FIX_20261020.md` §2 (Castiello 0.71–0.72 "fair"; Wang 0.785–0.901; Comer:
  sampling design changes AUC, narrow reading). The em-dash aside before it became a plain sentence.
- **Acknowledgements:** "The authors thank the editor and the anonymous reviewers for comments that substantially
  improved this paper."
- **Two leftovers fixed:** Section 3.8 title "a correlation that does not reproduce" → "a correlation that is not
  consistent across seeds" (matches the PI's heading in 8D); Figure 5 caption "the dial ends at 1.0" → "the range ends
  at 1.0" (missed by the session-3 idiom list).

**Checked:** −0.0142 / +0.0424 (Section 3.2, line "On real data, redesigning…"); 343/360 (Section 3.2); k ≥ 7 and
Jaccard ≥ 0.90 (Section 3.7); 97.5 % / 1,000 m (Section 3.9); 0.659 / 0.751 (Table 4). Manuscript 33 pp., 0 errors,
0 undefined.

**Open:** the AI statement (PI decides; Claude's honest draft given in chat); a list of about ten remaining stylistic
phrases in Results/Discussion offered as word swaps.

---

## Session 5 — corrections after the independent adversarial read: **drafted by Claude** at the PI's "ok, perbaiki" — 2026-10-26

**Not the PI's prose.** On 26 Oct the PI answered the agenda (`AGENDA_PI_20261026.md`) with "ok, perbaiki". Claude took
that as approval of the recommended defaults: D1 (E236), A + C for the AI statement (the editor email is drafted, not
sent), and the T/S/R batches. Every change below was drafted by Claude. Each one is either a correction found by
the reader and verified at its source (ledger C082–C089) or an approved style or template fix. The new AI declaration
says that corrections were drafted with Claude. **The PI must read this list and the changed passages before the
freeze.**

### Part 1 — `apply_session5_edits.py` (66 edits, each old string matched exactly once; dry run first)

| ID | Where | What | Why |
|---|---|---|---|
| R5a–e | abstract, Section 2.1 (road_dist), Section 3.2 (×2), conclusions | Hybrid step = evaluation artefact; **TGB steps "not reproduced"** (TGB − random +0.003 even on its own background); "The rise is not real" → "the part attributed to background redesign does not survive a common evaluation background"; the road_dist sentence no longer claims it "drove the reported number" | C084 |
| R9a–d | abstract, Section 3.2, Fig. 3 caption, Discussion | 343/360 split by design: hybrid 240/240 (median +0.238); random 52/60 (+0.020); TGB 51/60 (+0.014); the random/TGB gap partly reflects observed vs true presences | C087 |
| R6a–c | Section 3.9 | Adds E235's **pre-registered sensitivity results** (−0.002 NOT SPECIFIC; +0.016 INCONCLUSIVE); post hoc range +0.005 to +0.07; "No direction is therefore claimed below 1,000 m"; the Southern Mountains limited to 400–1,000 m; the MaxEnt fringe no longer attributed to Figure 5 | C085 |
| R8 | Section 2.6 | "seven experiments … pre-registered" → eight; only E224 had its design committed before the run; E217 and E219 have no design file | C082 |
| H1–H20 | throughout | Review/version history and idioms out ("no longer the argument", "earlier prose", "MVR", "the decimation is ours and not the paper's", "manuscript's stop/grid/operating point", "hidden in a supplement", "aligned the audit flag", "unbraked", "dose-response"); E224 paragraph title; Section 3.8 heading → "Corrections to two volcanic diagnostics" (< 75 characters); **the PI's 8D sentence**: the −0.163 vs −0.243 difference may come from seeds or from the change of sampling (E219 used the lattice pipeline); **the PI's 8E sentence**: "Low predicted suitability near a volcano therefore cannot, by itself, be read as evidence of buried settlement" (the old "because" clause cited a comparison the paper never set up) | C086, C088, reader |
| P1–P8 | Sections 3.2, 3.3, 3.5, 3.7, Limitations, Fig. 5 caption | 29 of 30 bootstrap replicates used (one skipped, < 50 out-of-bag sites); World C/D values labelled as true AUC; 27/60 scoped to the two surfaces; "best run 0.768" → original run, highest of 20 further seeds 0.774; Fig. 5 caption gives XGBoost's fringe elevation (491 m) next to MaxEnt's (1,107 m); hybrid(1.0) and "truth-best/-worst" written out | C088 |
| C1–C3 | Section 2.4, Section 4.4 | Test 1 / Test 3 introduced; Test 3's result not used (it was scored on own backgrounds); BPCB spelled out | R11 |
| S1–S10 | throughout | The style list of the agenda; 12 em-dash asides replaced | agenda S |
| T1, T2 | whole file; back matter | 11 double-quoted spans → single quotes (JCAA); "The authors have no competing interests to declare." | VENUE A7, A4 |
| A1 + AI | back matter | Authors' Contributions (M.A. "directed and checked the analyses (implemented with AI assistance, as declared below)"); **new "Declaration of AI Use"** (option A text of the agenda, plus "a number of corrections were drafted with Claude") | C089 |
| ALT | after the references | "Figure Alt Text" list, seven items (approved 21 Oct) | VENUE A8 |
| R12 | preamble | `\usepackage[T1]{fontenc}` + `lmodern`: accented names now extract correctly | reader |

Also rebuilt, without changing the text:
- **Figures 2, 3, 4 and 6** (`build_v03_figures_r2.py`, new file names; the v0.2 figures are untouched). Plain labels;
  no "MVR/TSS target"; Figure 3's right panel split by design; Figure 6 adds the **disjoint-ensemble** curve (E236 F).
- **Supplement:** a v0.3-only wording map in `build_supplement.py` ("top rung", "the reviewer's … request",
  "published ladder gain", "dial", "manuscript" → "main text"). The v0.2 build stays byte-identical.

**Open for part 2 (after E236):**
- the presence-data description (C081);
- the seed-stability sentences: abstract, Table 1, Section 3.7, protocol, conclusions (C083);
- the densities;
- Section 3.9 with the cleaned-set E235;
- Table S9.

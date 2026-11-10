# PI prose log — P2 round 2, author rewrite (replacement for JCAA, due 30 Nov 2026)

**Why this log exists:** on 2 Nov 2026 the JCAA editor ruled that AI may not write substantial parts of the paper and
the authors write the text themselves (ledger C093; `AGENDA_PI_20261103.md`). The PI chose to rewrite the whole text.

**Rules for every entry (stricter than `PI_PROSE_LOG_R2.md`):**
1. Claude's fact sheet for a section is **bullets and numbers only**, each number with its source file. No Indonesian
   prose the PI could copy.
2. Claude may explain a method or a statistic in chat when the PI asks. Those explanations are not pasted into the text.
3. The PI writes the section in Indonesian, in his own words.
4. Claude translates faithfully and adds nothing. Any change beyond translation (rounding, a citation, a term) is
   **told to the PI before insertion**, with its reason, and recorded here.
5. Each entry records: date, section, the PI's Indonesian text, the English inserted into `submission_jcaa_v0.4.tex`, and
   every flagged change with the PI's answer.
6. Passages from `PI_PROSE_LOG_R2.md` (passages 1–8) may be kept only after the PI re-reads and accepts them; any
   sentence Claude inserted into them in sessions 5–6 is rewritten by the PI.

**Order and dates:** `AGENDA_PI_20261103.md` §5 (Results → Methods → Discussion → Introduction → Conclusions →
Abstract → response letter → captions and alt text; checkpoint 25 Nov).

---

## Session 1 — Results §3.1–3.3 — opened 2026-11-03

- **Fact sheet:** `rewrite_r3/FACTS_S1_RESULTS_3_1-3_3.md` (bullets and numbers only). Numbers re-checked the same day:
  `verify_k1_numbers.py` 39/39; `verify_headline_numbers.py` 60/64 (the 4 are the withdrawn K5/K6/K7/G1c); ladder table
  E007–E013 against each `model_results.txt`; E222 per-design counts (240/240, 52/60, 51/60, 343/360) from `e222_runs.csv`.
- **Writing sheet given to the PI:** `rewrite_r3/LEMBAR_TULIS_S1_20261103_v2.docx`.
- **File:** `submission_jcaa_v0.4.tex` created from v0.3 on 3 Nov; v0.3 kept unchanged as the uploaded file (10667).

### §3.1 — PI text received 2026-11-03 — **INSERTED** into v0.4 (lines 327–332)

**PI text (Indonesian, verbatim):**

> Rangkaian tujuh eksperimen (E007–E013) menunjukkan peningkatan AUC dari 0,659 pada desain background acak (E007)
> menjadi 0,751 pada desain hybrid (E013), berdasarkan rata-rata 20 seed. Peningkatan sebesar 0,092 ini terjadi melalui
> penambahan fitur lingkungan, penerapan Target-Group Background (TGB), hingga penggabungan beberapa strategi pemilihan
> background. Nilai AUC dan TSS rata-rata pada E013 masing-masing mencapai 0,751 ± 0,013 dan 0,465 ± 0,021. Sekilas,
> hasil ini menunjukkan bahwa desain background yang semakin realistis dapat meningkatkan performa model. Namun,
> interpretasi tersebut perlu diuji lebih lanjut karena evaluasi model nol pada Tabel 5 menggunakan background yang
> berbeda, sementara E013 dinilai menggunakan background-nya sendiri. Selain itu, DKNS menggunakan lokasi situs yang
> sebenarnya sehingga bukan pembanding yang sepenuhnya adil.

**English inserted (v0.4, lines 327–332):**

> The seven experiments (E007--E013) show the AUC rising from 0.659 under a random background (E007) to a 20-seed
> average of 0.751 under the hybrid design (E013). That rise of 0.092 runs through the addition of environmental
> features, the introduction of target-group background (TGB), and the combination of several background-selection
> strategies. For E013 the seed-averaged AUC and TSS are $0.751 \pm 0.013$ and $0.465 \pm 0.021$ respectively. At first
> sight, this supports the reading that increasingly realistic background designs improve model performance. That
> reading still needs testing: the null models in Table~\ref{tab:nullmodel} are evaluated on a different background, and
> E013 is scored on its own. DKNS also uses the actual site locations, so it is not a fully fair comparison.

**Flags raised before insertion, and what was applied:**

| # | Flag | Applied |
|---|---|---|
| **F1** | §3.1 paired the seed-averaged AUC 0.751 with the original run's TSS 0.507 ± 0.167 | **Resolved by the PI's text:** it pairs 0.751 ± 0.013 with the 20-seed TSS 0.465 ± 0.021 (option A). No further action. |
| **F2** | "berdasarkan rata-rata 20 seed" sat at the end of the sentence, so it could be read as applying to both 0.659 and 0.751; only E013 has a 20-seed average | Attached to 0.751 only: "to a 20-seed average of 0.751". PI did not ask to change it. |
| **F3** | "Target-Group Background (TGB)" repeats a definition already made in Section 2.1 (line 86) | Kept as the PI wrote it. Reversible to "TGB" if he prefers. |
| **F4** | The paragraph no longer refers to Table 4 (the progression table) or Figure 2, both of which sit in this subsection | **Not added** (content addition needs the PI's word). |
| **F5** | "perlu diuji lebih lanjut" does not say where the test is (§3.2) | **Not added.** Optional cross-reference. |

**Content dropped relative to v0.3 (informational — no action taken):** the original run's AUC 0.768, the highest of 20
seeds 0.774, the 20-seed range 0.729–0.774, the TSS 0.507 ± 0.167, and the sentence "the model beat the three null
models". The +0.083 worst-seed margin still appears in §3.2 (block F). All of these remain in Table 4. The PI's text
therefore reports fewer numbers than v0.3 did. If he wants any of them back, it is a one-line change.

### §3.2 — PI text received 2026-11-03 — **INSERTED** into v0.4 (after the 3 Nov "lakukan")

**PI text (Indonesian, verbatim):**

> Untuk mengetahui apakah peningkatan AUC dalam rangkaian eksperimen sebelumnya benar-benar mencerminkan peningkatan
> kemampuan model, dilakukan pengujian ulang dengan menggunakan background evaluasi yang sama untuk seluruh desain.
> Pengujian ini melibatkan tiga algoritma, yaitu XGBoost, Random Forest, dan Maximum Entropy (MaxEnt), dengan lima seed
> dan pembagian validasi silang berbasis blok spasial yang sama. Pendekatan ini memungkinkan perbandingan yang lebih adil
> karena setiap model dinilai berdasarkan kumpulan background yang identik, bukan berdasarkan background yang digunakan
> masing-masing desain. Hasil dekomposisi menunjukkan bahwa pada kumpulan 378 rekaman kompilasi dengan fitur lengkap,
> kontribusi perubahan desain background hanya sebesar −0,0142 AUC, dengan interval berbasis kelompok seed antara −0,021
> hingga −0,005. Hanya 3 dari 15 kombinasi seed dan algoritma yang menunjukkan kontribusi positif. Sebaliknya, ketika
> hanya menggunakan fitur terrain, kontribusi perubahan desain background mencapai +0,0054. Penambahan fitur jarak
> sungai memberikan pengaruh yang lebih konsisten, yaitu peningkatan AUC sebesar +0,0424, dengan hasil positif pada
> seluruh 12 perbandingan berpasangan. Temuan ini menunjukkan bahwa pada data kompilasi, peningkatan performa lebih
> banyak berkaitan dengan penambahan informasi lingkungan berupa jarak sungai daripada perubahan desain background.
>
> Pengujian berikutnya membandingkan desain hybrid dengan desain acak menggunakan beberapa jenis background evaluasi.
> Hasilnya memperlihatkan bahwa desain hybrid memperoleh AUC lebih tinggi daripada desain acak pada ketiga algoritma
> ketika keduanya dinilai menggunakan background hybrid. Namun, keunggulan tersebut tidak ditemukan ketika evaluasi
> menggunakan background uniform, TGB, maupun stratified. Pada ketiga jenis background tersebut, tidak satu pun dari tiga
> algoritma menunjukkan keunggulan hybrid terhadap desain acak. Hal ini memperlihatkan bahwa hasil perbandingan dapat
> berubah bergantung pada background yang digunakan untuk evaluasi. Ketika setiap desain dinilai menggunakan background
> miliknya sendiri, kontribusi redesign terhadap AUC rata-rata mencapai +0,0316, yang terdiri atas +0,0145 pada MaxEnt,
> +0,0431 pada XGBoost, dan +0,0373 pada Random Forest. Dengan demikian, penilaian menggunakan background masing-masing
> dapat memberikan gambaran yang berbeda dibandingkan penilaian menggunakan satu background bersama.
>
> Besarnya pengaruh tersebut terlihat lebih jelas pada pengujian khusus terhadap desain hybrid. Pada 378 rekaman
> kompilasi, perbedaan antara AUC yang diperoleh menggunakan own background dan uniform background bernilai positif pada
> seluruh 60 kombinasi seed dan algoritma. Rata-rata inflasinya mencapai +0,037 AUC, dengan rentang +0,005 hingga +0,084.
> Besaran ini berada pada orde yang sama dengan kenaikan tangga yang sebelumnya dilaporkan, yaitu +0,092. Akan tetapi,
> pengaruh tersebut tidak berlaku secara seragam untuk semua desain. Pada TGB, penilaian menggunakan own background justru
> menghasilkan rata-rata AUC 0,0054 lebih rendah dibandingkan uniform background, dengan hanya 22 dari 60 kombinasi yang
> menunjukkan nilai positif. Selain itu, perbandingan TGB dengan desain acak pada own background hanya menghasilkan
> selisih +0,003. Oleh karena itu, peningkatan pada langkah TGB dalam eksperimen E010–E012 tidak dapat dijelaskan sebagai
> akibat inflasi own background. Hasil tersebut tidak terulang dalam benchmark, tetapi tidak cukup bukti untuk menyebut
> langkah TGB sebagai artefak evaluasi. Temuan ini perlu dibedakan dari pengaruh yang secara langsung teridentifikasi
> pada desain hybrid.
>
> Untuk menguji apakah kecenderungan tersebut juga muncul ketika kebenaran hasil dapat diketahui, dilakukan eksperimen
> menggunakan data sintetis dengan permukaan intensitas yang telah ditentukan. Dalam pengujian ini, AUC yang dilaporkan
> berdasarkan own background dibandingkan dengan kemampuan model dalam memulihkan pola sebenarnya melalui common
> background. Hasilnya menunjukkan bahwa pada desain hybrid, AUC yang dilaporkan lebih tinggi daripada performa
> sebenarnya dalam seluruh 240 percobaan, dengan median selisih +0,238. Pada desain acak, kondisi tersebut terjadi pada
> 52 dari 60 percobaan dengan median selisih +0,020, sedangkan pada TGB terjadi pada 51 dari 60 percobaan dengan median
> +0,014. Secara keseluruhan, terdapat 343 dari 360 percobaan yang menunjukkan nilai AUC terlapor lebih tinggi daripada
> performa sebenarnya. Meskipun demikian, pada desain acak dan TGB, sebagian perbedaan juga berasal dari ketidaksamaan
> antara sampel presensi yang diamati dan keseluruhan presensi sebenarnya, bukan semata-mata akibat pemilihan background.
> Hasil eksperimen sintetis ini memperkuat bukti bahwa evaluasi menggunakan background milik desain sendiri, khususnya
> pada desain hybrid, dapat menghasilkan estimasi performa yang terlalu tinggi.
>
> Bukti tersebut kemudian diperiksa melalui pengujian statistik dan analisis sensitivitas. Interval kepercayaan
> ekuivalensi menolak kemungkinan bahwa kenaikan sebesar +0,092 benar-benar bertahan pada seluruh 12 kombinasi algoritma
> dan background evaluasi. Pengujian tambahan menggunakan bootstrap berbasis blok presensi dilakukan sebanyak 30
> replikasi. Satu replikasi tidak digunakan karena jumlah situs out-of-bag kurang dari 50, sehingga tersisa 29 replikasi
> untuk analisis. Batas atas interval kepercayaan yang diperoleh adalah +0,008 untuk MaxEnt, +0,025 untuk XGBoost, dan
> +0,026 untuk Random Forest. Ketiga batas tersebut berada jauh di bawah kenaikan +0,092 yang sebelumnya dilaporkan.
> Meskipun jumlah replikasi bootstrap relatif kecil, pengujian ini tetap memberikan bukti penolakan karena jarak antara
> batas interval dan nilai kenaikan yang diuji cukup lebar. Namun, hasil ini tidak berarti bahwa titik akhir interval
> telah diestimasi dengan presisi tinggi. Selain itu, perubahan parameter regularisasi MaxEnt pada rentang β = 0,5–4,0
> menghasilkan kontribusi antara −0,020 hingga −0,022, dengan hanya satu dari sepuluh percobaan bernilai positif. Hal ini
> menunjukkan bahwa temuan tersebut bukan sekadar akibat penggunaan model yang kurang teratur atau mengalami overfitting.
>
> Temuan ini juga mengubah cara membaca perbandingan model pada Tabel 5. Dalam perbandingan awal, E013 memiliki AUC
> rata-rata sebesar 0,751, sedangkan model nol dinilai menggunakan grid reguler yang mencakup seluruh sel wilayah
> penelitian. Karena kedua kelompok dinilai menggunakan background berbeda, margin keunggulan sebesar +0,105 terhadap
> model interpolasi spasial DKNS (AUC 0,646) berpotensi terlalu tinggi. Ketika E013 dinilai menggunakan common uniform
> background, AUC XGBoost turun menjadi 0,706. Model tersebut masih memiliki nilai lebih tinggi daripada DKNS, tetapi
> marginnya berkurang menjadi sekitar +0,06. Margin pada seed terburuk yang sebelumnya sebesar +0,083 juga menjadi lebih
> dekat dengan batas deteksi +0,03 yang dibahas pada Section 3.7. Perlu diperhatikan bahwa nilai 0,751 berasal dari
> pipeline awal, sedangkan 0,706 berasal dari benchmark berbasis lattice, sehingga keduanya tidak menggunakan prosedur
> sampling yang identik. Oleh karena itu, penyusutan tersebut sebaiknya dipahami sebagai gambaran besaran pengaruh,
> bukan sebagai selisih langsung dari dua pengukuran yang sepenuhnya setara. Secara keseluruhan, hasil penelitian masih
> mendukung bahwa fitur lingkungan mampu menangkap pola yang tidak hanya bergantung pada interpolasi lokasi situs
> terdekat. Namun, besarnya keunggulan model yang sebelumnya dilaporkan tidak dapat dipertahankan setelah background
> evaluasi diseragamkan. Dengan demikian, Tabel 5 lebih tepat dipandang sebagai batas atas keunggulan model terhadap
> model nol, bukan sebagai batas bawah yang dapat dianggap stabil.

**English translation, in six paragraphs (A–F):**

> A. To test whether the AUC rise in the preceding experiments reflects a real gain in model skill, every design was
> re-scored on the same evaluation background. The test uses three algorithms — XGBoost, Random Forest and Maximum
> Entropy (MaxEnt) — with five seeds and the same spatial-block cross-validation split. This allows a fairer comparison,
> because every model is scored on an identical set of background points rather than on the background its own design
> produced. On the 378 compiled records with the full feature set, the decomposition gives the background redesign a
> contribution of only −0.0142 AUC, with a seed-cluster interval of −0.021 to −0.005. Only 3 of the 15 seed × algorithm
> combinations are positive. With terrain features alone, the same redesign contributes +0.0054. Adding distance to
> rivers has a more consistent effect: +0.0424 AUC, positive in all 12 paired comparisons. On the compiled data, then,
> the gain tracks the added environmental information (distance to rivers) more than the change in background design.
>
> B. The next test compares the hybrid design with the random design under several kinds of evaluation background.
> Hybrid scores higher than random on all three algorithms when both are evaluated on the hybrid background. The
> advantage disappears when the evaluation uses the uniform, TGB or stratified background: on each of those three, not
> one of the three algorithms shows it. The comparison therefore depends on the background used to evaluate it. Scored
> on its own background instead, the redesign contributes +0.0316 on average: +0.0145 for MaxEnt, +0.0431 for XGBoost
> and +0.0373 for Random Forest. Scoring each design on its own background therefore paints a different picture from
> scoring all of them on one shared background.
>
> C. The size of that effect is clearest in the test aimed at the hybrid design. On the 378 compiled records, own
> background minus uniform background is positive in all 60 seed × algorithm combinations. The mean inflation is +0.037
> AUC, over a range of +0.005 to +0.084. That is the same order of magnitude as the ladder rise reported above, +0.092.
> The effect is not uniform across designs, however. For TGB, scoring on the own background instead gives an AUC 0.0054
> lower on average than the uniform background, with only 22 of 60 combinations positive. TGB against the random design
> on the own background differs by only +0.003. The rise at the TGB step (E010–E012) therefore cannot be explained as
> own-background inflation. That result does not recur in the benchmark, but that is not evidence enough to call the TGB
> step an evaluation artefact. It has to be kept distinct from the effect identified directly on the hybrid design.
>
> D. To test whether the same tendency appears where the truth is known, we ran an experiment on synthetic data with a
> prescribed intensity surface. Here the reported AUC — own background — is compared with how well the model recovers
> the true pattern when scored on the common background. For the hybrid design the reported AUC exceeds the true
> performance in all 240 runs, with a median gap of +0.238. For the random design it happens in 52 of 60 runs (median
> +0.020), and for TGB in 51 of 60 (median +0.014). Across both surfaces, 343 of 360 runs put the reported AUC above the
> true performance. In the random and TGB designs, however, part of the gap comes from the difference between the
> observed presence sample and the full true presences, not from the background choice alone. The synthetic experiment
> therefore strengthens the evidence that scoring a design on its own background — the hybrid design above all — can
> overstate performance.
>
> E. That evidence was then checked by statistical tests and a sensitivity analysis. Equivalence confidence intervals
> reject +0.092 in all 12 algorithm × evaluation-background combinations. A presence-block bootstrap was run for 30
> replicates; one was discarded for having fewer than 50 out-of-bag sites, leaving 29. The upper confidence bounds are
> +0.008 for MaxEnt, +0.025 for XGBoost and +0.026 for Random Forest, all well below the +0.092 rise. The bootstrap is
> small, but it still rejects the rise: the distance between the bounds and the value under test is wide. It does not
> mean the interval end-points are estimated with high precision. Varying MaxEnt regularisation over β = 0.5–4.0 gives
> contributions of −0.020 to −0.022, with one of ten runs positive. The finding is not an artefact of an
> under-regularised or overfitted model.
>
> F. The same finding changes how Table~\ref{tab:nullmodel} should be read. In the earlier comparison E013 has a
> seed-averaged AUC of 0.751, while the nulls are scored on a regular grid covering every cell of the study area. Since
> the two groups are scored on different backgrounds, the +0.105 margin over the spatial-interpolation null DKNS (AUC
> 0.646) is potentially too high. Scored on the common uniform background, the XGBoost design of E013 falls to 0.706. It
> stays above DKNS, but the margin shrinks to about +0.06. The worst-seed margin, +0.083, is likewise close to the
> detection floor of +0.03 discussed in Section~\ref{sec:ci}. The 0.751 comes from the original pipeline and the 0.706
> from the lattice-based benchmark, so the two do not share an identical sampling procedure; the shrinkage is best read
> as an order of magnitude, not as the direct difference between two fully comparable measurements. What survives is
> that environmental features capture structure beyond interpolation from the nearest site. What does not survive is the
> size of the advantage once the evaluation background is made uniform. Table~\ref{tab:nullmodel} is therefore an upper
> bound on the advantage over the nulls, not a lower bound that can be relied on.

**Checks run before translating (all passed):**

- Bootstrap upper bounds: `e223b_bootstrap_summary.csv` gives maxent +0.0082, xgboost +0.0253, randomforest +0.0256 →
  the PI's +0.008 / +0.025 / +0.026 assignment to MaxEnt / XGBoost / Random Forest is **correct**. All three exclude
  +0.092.
- β sweep: `e223c_beta_summary.csv` means −0.0198 to −0.0217 (PI: −0.020 to −0.022), `frac_positive` 0.1 in all five
  β values (PI: one of ten). Correct.
- E222: 240/240, 52/60, 51/60, 343/360, medians +0.238 / +0.020 / +0.014 — match `e222_runs.csv`.

**Flags waiting for the PI:**

| # | Flag | Claude's proposal |
|---|---|---|
| **G1** | `Section 3.7` is hard-coded; the manuscript uses `\ref` everywhere | Replace with `Section~\ref{sec:ci}` (same number). |
| **G2** | "yang sebelumnya dilaporkan" (twice) reads as revision talk; R1 asked for a self-contained paper | Drop the two words; "the ladder rise of +0.092" is enough. |
| **G3** | Length: 989 Indonesian words → 940 English, against a target of ±700 | See the budget note below. This is the largest single decision of the session. |
| **G4** | "Margin pada seed terburuk yang sebelumnya sebesar +0,083 juga menjadi lebih dekat dengan batas deteksi" — the +0.083 margin does not change; what is near the floor is the corrected margin | Reworded as "The worst-seed margin, +0.083, is likewise close to the detection floor". |

### §3.3 — PI text received 2026-11-03 — **INSERTED** into v0.4 (after the 3 Nov "lakukan")

**PI text (Indonesian, verbatim):**

> Data presensi yang digunakan dalam penelitian ini terdiri atas 378 rekaman yang berasal dari 335 koordinat berbeda.
> Data tersebut dikompilasi dari sumber publik dan bukan berasal dari satu katalog situs arkeologi yang telah
> diverifikasi secara menyeluruh. Permasalahan utama pada kumpulan data ini adalah adanya rekaman yang tidak benar-benar
> merepresentasikan situs arkeologi. Sebanyak 124 rekaman merupakan monumen, patung, atau tugu modern, sedangkan 13
> rekaman lainnya merupakan objek yang salah lokasi atau bukan situs arkeologi, seperti candi dari wilayah lain, objek
> wisata, dan institusi modern. Dengan demikian, sekitar sepertiga dari keseluruhan rekaman memiliki permasalahan yang
> dapat memengaruhi hasil pemodelan. Untuk menguji pengaruh kualitas data tersebut, seluruh analisis kemudian diulang
> menggunakan subset yang telah dibersihkan melalui eksperimen E236. Proses pembersihan dilakukan dengan menghapus 85
> rekaman yang koordinatnya diperoleh melalui geocoding nama dari daftar Wikipedia, 124 monumen modern, serta 8 candi
> dari wilayah lain, objek wisata, dan institusi modern. Setelah proses tersebut, tersisa 161 situs, yang terdiri atas
> 155 rekaman dari OpenStreetMap dan 6 dari Wikidata. Seluruh skrip analisis tetap dipertahankan tanpa perubahan, kecuali
> berkas data presensi yang digunakan. Dengan demikian, perbedaan hasil antara kedua kumpulan data dapat dievaluasi tanpa
> mengubah prosedur pemodelan.
>
> Berbeda dengan hasil pada 378 rekaman kompilasi, pengujian menggunakan 161 situs bersih menunjukkan bahwa perubahan
> desain background memberikan peningkatan performa yang positif ketika seluruh model dinilai menggunakan common
> evaluation background. Pada pengujian lima seed, peningkatan AUC mencapai +0,0507. Ketika jumlah seed diperluas
> menjadi sepuluh, hasilnya tetap konsisten pada +0,0510 dengan interval kepercayaan [+0,039, +0,063]. Dari 30 kombinasi
> seed dan algoritma, sebanyak 29 menunjukkan hasil positif. Peningkatan juga ditemukan pada masing-masing algoritma,
> yaitu +0,038 untuk XGBoost, +0,044 untuk Random Forest, dan +0,072 untuk MaxEnt. Konsistensi hasil semakin terlihat
> ketika dua kelompok ensemble yang terdiri atas lima seed tanpa seed yang sama menghasilkan peningkatan masing-masing
> +0,0505 dan +0,0516. Hasil ini menunjukkan bahwa pada data presensi yang telah dibersihkan, redesign background memang
> memberikan peningkatan yang dapat dipertahankan ketika evaluasi dilakukan menggunakan background yang sama. Namun,
> ketika masing-masing desain dinilai menggunakan own background, peningkatannya mencapai +0,098. Selisih sebesar +0,047
> antara kedua pendekatan tersebut menunjukkan bahwa inflasi evaluasi masih terjadi dan besarnya hampir sebanding dengan
> peningkatan performa yang sebenarnya.
>
> Beberapa temuan dari kumpulan data kompilasi tetap berlaku pada subset yang telah dibersihkan. Salah satunya adalah
> pengaruh own background terhadap hasil evaluasi. Pada data bersih, nilai AUC hybrid lebih tinggi ketika dinilai
> menggunakan own background dibandingkan uniform background pada 54 dari 60 kombinasi seed dan algoritma, dengan
> rata-rata perbedaan +0,041. Pada data kompilasi, hasil positif ditemukan pada 60 dari 60 kombinasi dengan rata-rata
> +0,037. Hal ini menunjukkan bahwa inflasi akibat penggunaan background sendiri tidak hanya muncul karena kualitas data
> presensi, tetapi juga berkaitan dengan mekanisme evaluasi. Meskipun demikian, terdapat perbedaan penting antara kedua
> kumpulan data. Pada subset bersih, desain hybrid mengungguli desain acak pada seluruh jenis background evaluasi, yaitu
> uniform, target-group, hybrid, dan stratified, untuk ketiga algoritma. Sebaliknya, pada data kompilasi, keunggulan
> hybrid hanya terlihat ketika dinilai menggunakan background miliknya sendiri. Sementara itu, langkah TGB tetap tidak
> menunjukkan peningkatan yang berarti pada kedua kumpulan data. Selisih TGB terhadap desain acak adalah −0,005 pada own
> background dan +0,001 pada common background. Dengan demikian, temuan mengenai tidak terulangnya peningkatan TGB tetap
> konsisten, sedangkan pengaruh hybrid terhadap performa model ternyata bergantung pada kualitas data presensi.
>
> Pengujian pada subset bersih juga memperlihatkan kelemahan kriteria pemilihan desain yang menggunakan AUC pada own
> background. Ketika proporsi hard negatives dinaikkan dari 0,0 menjadi 1,0, AUC meningkat dari 0,697 menjadi 0,866 dan
> mencapai nilai maksimum pada batas akhir rentang pengujian. Namun, hasil pada common background menunjukkan pola yang
> berbeda. Pada proporsi hard negatives sebesar 0,0, 0,3, dan 1,0, nilai AUC masing-masing adalah 0,680, 0,713, dan
> 0,656. Artinya, konfigurasi dengan AUC tertinggi berdasarkan own background justru menghasilkan nilai terendah ketika
> dinilai menggunakan common background. Dengan demikian, kriteria seleksi tersebut bukan hanya gagal menunjukkan kapan
> penambahan hard negatives harus dihentikan, tetapi dalam kumpulan data ini juga dapat mengarahkan pemilihan desain ke
> konfigurasi dengan performa evaluasi bersama yang lebih rendah.
>
> Ketidakstabilan juga terlihat pada peta prioritas yang dihasilkan model. Perbandingan antara robust core dan contingent
> fringe menghasilkan rasio densitas sebesar 2,54 untuk Random Forest, 8,51 untuk XGBoost, dan 5,30 untuk MaxEnt. Selain
> itu, perubahan anggota pada kelompok sepuluh persen sel dengan prioritas tertinggi mencapai 41,6–54,8% di antara
> pasangan seed. Ketika dua ensemble yang masing-masing terdiri atas lima seed tanpa seed yang sama dibandingkan, nilai
> Jaccard overlap berada pada rentang 0,70–0,77. Nilai ini sedikit lebih rendah dibandingkan data kompilasi yang
> mencapai 0,76–0,86. Perlu diperhatikan bahwa contingent fringe hanya mencakup 5 situs pada XGBoost, 15 situs pada
> Random Forest, dan 5 situs pada MaxEnt. Oleh karena itu, rasio densitas tersebut lebih tepat digunakan untuk
> menunjukkan arah perbedaan daripada sebagai ukuran besaran yang presisi.
>
> Terakhir, pengujian kontrol elevasi menunjukkan pola yang relatif serupa antara kedua kumpulan data. Pada wilayah
> dengan elevasi di bawah 1.000 meter, selisih kesesuaian antara lokasi yang dekat dan jauh dari pusat gunung berapi
> mencapai +0,046 untuk desain acak dan TGB, serta +0,060 untuk desain hybrid pada subset bersih. Pada data kompilasi,
> nilainya masing-masing adalah +0,049, +0,049, dan +0,060. Berdasarkan kriteria SPECIFIC yang telah ditetapkan
> sebelumnya, hanya desain hybrid yang memenuhi kriteria tersebut pada kedua kumpulan data. Namun, ringkasan hasil dari
> ketiga desain secara keseluruhan masih bersifat inconclusive. Selain itu, dua definisi sensitivitas yang digunakan
> menghasilkan perubahan label antara kedua kumpulan data. Temuan ini menunjukkan bahwa hasil kontrol elevasi tidak
> sepenuhnya berubah setelah pembersihan data, tetapi juga belum cukup kuat untuk menghasilkan kesimpulan umum mengenai
> pengaruh elevasi dan kedekatan terhadap pusat vulkanik. Secara keseluruhan, analisis pada subset bersih memperlihatkan
> bahwa redesign background dapat memberikan peningkatan performa yang nyata, tetapi besarnya tetap dapat
> dilebih-lebihkan oleh own-background evaluation. Kualitas data presensi, konsistensi seed, dan penggunaan background
> evaluasi yang sama karena itu tetap menjadi faktor penting dalam menilai hasil pemodelan.

**English translation, in six paragraphs:**

> A. The presence data used here consist of 378 records at 335 distinct coordinates. They are compiled from public
> sources, not taken from a fully verified archaeological site catalogue. The main problem with the set is that not
> every record represents an archaeological site. Of these, 124 records are modern monuments, statues or memorials, and
> a further 13 are mislocated or are not sites at all — temples from other regions, tourist attractions and modern
> institutions. About a third of the records therefore carry a defect that could affect the modelling results. To test
> the effect of data quality, every analysis was repeated on a cleaned subset in experiment E236. Cleaning removed 85
> records whose coordinates were geocoded from a name in a Wikipedia list, 124 modern monuments, and 8 temples from
> other regions, tourist objects and modern institutions. What remained were 161 sites: 155 records from OpenStreetMap
> and 6 from Wikidata. Every analysis script was kept unchanged apart from the presence file it reads, so differences
> between the two sets can be compared without altering the modelling procedure.
>
> B. Unlike the results on the 378 compiled records, the test on the 161 cleaned sites shows the background redesign
> adding performance when every model is scored on the common evaluation background. With five seeds the gain is
> +0.0507. Extending to ten seeds gives +0.0510 with a confidence interval of [+0.039, +0.063]. Of the 30 seed ×
> algorithm cells, 29 are positive. The gain holds for each algorithm separately: +0.038 for XGBoost, +0.044 for Random
> Forest and +0.072 for MaxEnt. Two ensembles of five seeds that share no seed give +0.0505 and +0.0516. On cleaned
> presences the redesign therefore delivers a gain that survives a common evaluation background. Scored on its own
> background, however, the same contrast reaches +0.098. The +0.047 difference between the two approaches shows that
> evaluation inflation persists, and that it is close in size to the gain itself.
>
> C. Several findings from the compiled set carry over to the cleaned subset. One is the effect of the own background.
> On the clean data, hybrid scores higher on its own background than on the uniform background in 54 of 60 seed ×
> algorithm cells, with a mean difference of +0.041; on the compiled data the same is true of 60 of 60 cells, with a mean
> of +0.037. Own-background inflation therefore does not depend on the quality of the presence data alone; it also
> arises from the evaluation design. There is an important difference between the two sets, however. On the clean
> subset the hybrid design beats the random design under every evaluation background — uniform, TGB, hybrid and
> stratified — for all three algorithms. On the compiled data the hybrid advantage appears only when it is scored on its
> own background. The TGB step shows no meaningful gain in either set: TGB minus random is −0.005 on the own background
> and +0.001 on the common background. The failure of the TGB rise to recur is therefore stable, while the hybrid
> advantage turns out to depend on the quality of the presence data.
>
> D. The cleaned subset also exposes a weakness in a selection criterion that uses AUC on the own background. Raising
> the hard-negative share from 0.0 to 1.0 lifts the own-background AUC from 0.697 to 0.866, with the maximum at the end
> of the tested range. On the common background the pattern is different: at shares of 0.0, 0.3 and 1.0 the AUC is
> 0.680, 0.713 and 0.656 respectively. The configuration with the highest own-background AUC therefore has the lowest
> common-background score. The criterion does not merely fail to say when adding hard negatives should stop; on this
> data set it would steer the choice towards the configuration with the weaker shared evaluation.
>
> E. Instability also appears in the priority maps. The density ratio between the robust core and the contingent fringe
> is 2.54 for Random Forest, 8.51 for XGBoost and 5.30 for MaxEnt. The membership of the top decile of cells turns over
> by 41.6–54.8% between seed pairs. Comparing two ensembles of five seeds that share no seed gives a Jaccard overlap of
> 0.70–0.77, slightly below the 0.76–0.86 of the compiled data. The contingent fringe contains only 5 sites for XGBoost,
> 15 for Random Forest and 5 for MaxEnt, so the density ratios indicate a direction rather than a precise size.
>
> F. Finally, the elevation control gives a broadly similar pattern in the two sets. Below 1,000 m, the suitability
> difference between near and far locations reaches +0.046 for the random and TGB designs and +0.060 for the hybrid
> design on the clean subset; on the compiled data the values are +0.049, +0.049 and +0.060. Against the pre-registered
> SPECIFIC criterion, only the hybrid design meets it in both sets. The summary across the three designs remains
> inconclusive, and the two sensitivity definitions exchange labels between the two sets. Cleaning the data therefore
> does not change the elevation control outright, but neither is the result strong enough to support a general
> conclusion about elevation and proximity to a volcanic centre. Taken together, the cleaned subset shows that the
> background redesign can deliver a real gain in performance, but that its size can still be overstated by own-background
> evaluation. Presence-data quality, seed stability and a common evaluation background therefore remain essential to
> judging a presence–background result.

**Flags waiting for the PI:**

| # | Flag | Claude's proposal |
|---|---|---|
| **H1** | **The numbers 13 and 8 do not reconcile.** Paragraph A says 13 records are mislocated or not sites; the cleaning sentence in the same paragraph removes 8 of that kind. The cleaning accounting is 85 + 124 + 8 = 217 → 161. | Use **8** in both places, or say what the other five were. Needs the PI's word: this is a number a reviewer can check. |
| **H2** | "melalui eksperimen E236" — the paper's other experiment names (E217–E224) are used bare, without "experiment" | Keep as written; flagged only for consistency. |
| **H3** | Length: 949 Indonesian words → 894 English, against a target of ±600 | See the budget note. |

**Numbers checked in §3.3:** 378/335 ✓ · 124 modern ✓ · 85 + 124 + 8 → 217 ✓ · 161 = 155 OSM + 6 Wikidata ✓ · +0.0507 ✓ ·
+0.0510 [+0.039, +0.063] ✓ · 29/30 ✓ · +0.038 / +0.044 / +0.072 ✓ · +0.0505 / +0.0516 ✓ · +0.098 ✓ · +0.047 ✓ ·
54/60 (+0.041) vs 60/60 (+0.037) ✓ · TGB −0.005 / +0.001 ✓ · 0.697 → 0.866 ✓ · 0.680 / 0.713 / 0.656 ✓ ·
2.54 / 8.51 / 5.30 ✓ · 41.6–54.8% ✓ · Jaccard 0.70–0.77 vs 0.76–0.86 ✓ · fringe 5 / 15 / 5 ✓ ·
+0.046 / +0.046 / +0.060 and +0.049 / +0.049 / +0.060 ✓ · SPECIFIC: hybrid only, summary inconclusive ✓

### Decisions taken on 3 Nov ("lakukan")

- **H1 → 8.** The manuscript now says "124 records are modern monuments … and 8 more are temples from other regions,
  tourist objects or modern institutions", and the cleaning sentence reuses those same two categories. The 13 is gone;
  the accounting is 85 + 124 + 8 = 217 → 161. **If the PI knows what the other five records were, this can be reopened.**
- **G1 → `Section~\ref{sec:ci}`**, replacing the hard-coded "Section 3.7".
- **G2 → "the ladder rise of $+0.092$"**, with "yang sebelumnya dilaporkan" dropped; the second occurrence in block F
  was already translated without it.
- **G4 →** reworded as "The worst-seed margin, $+0.083$, is likewise close to the detection floor".
- **Budget, option (a)+(b) → applied, but it saved only 87 words**, not the ~300 estimated. Sentences cut (all
  restatements, no number lost):
  - §3.2 block B: "The comparison therefore depends on the background used to evaluate it." and "Scoring each design on
    its own background therefore paints a different picture from scoring all of them on one shared background."
  - §3.3 block A: the two sentences that restated the cleaning accounting were merged into one; "so the two sets can be
    compared without altering the modelling procedure" kept as the single closing clause.
  - §3.3 block B: "On cleaned presences the redesign therefore delivers a gain that survives a common evaluation
    background."
  - §3.3 block F: "Presence-data quality, seed stability and a common evaluation background therefore remain essential…"
  - **§3.3 block C was kept in full** — its inference ("inflation does not depend on the presence data alone") is the
    argument of that paragraph, not a restatement.
- **Two cross-references added** (these were in v0.3 and would otherwise have been orphaned; reversible):
  - "(Figure~\ref{fig:artefact})" at the end of §3.2 block D;
  - "Table~S9 reports every quantity below for both sets." at the end of §3.3 block A.

### Word budget (session 1) — measured in v0.4, after the PI's "silahkan sesuaikan"

| | v0.3 | PI text (Indonesian) | v0.4 final | Δ |
|---|---|---|---|---|
| §3.1 | 79 | 111 | 117 | **+38** |
| §3.2 | 697 | 989 | 911 | **+214** |
| §3.3 | 603 | 949 | 614 | **+11** |
| **Total** | **1,379** | **2,049** | **1,642** | **+263** |

Manuscript now stands at about **11,497 of 11,500** — inside the limit by three words. That was achieved by compressing
§3.3 back to its own target (834 → 614); **no number and no claim was dropped**, and the compression is recorded in the
sentences that survive. §3.2 remains 214 words above its 700-word target, so it has no slack left: sessions 2–4 of the
Results must be written *shorter* than v0.3, not longer, or something else has to give.

Compression of §3.3 (second pass, PI authorised "silahkan sesuaikan"):

- the cleaning accounting became one sentence rather than two;
- "Own-background inflation therefore does not depend on the quality of the presence data alone; it also arises from the
  evaluation design" was folded into the closing sentence of that paragraph ("The hybrid advantage therefore depends on
  the quality of the presence data, while its inflation does not");
- the two "does not merely fail to stop" sentences were merged;
- the elevation paragraph's last two sentences were merged into two shorter ones.



---

## Session 2 — Results §3.4–3.6 — prepared 2026-11-09, PI writing 2026-11-10

- **Fact sheet:** `rewrite_r3/FACTS_S2_RESULTS_3_4-3_6.md`; **writing sheet:** `rewrite_r3/LEMBAR_TULIS_S2_20261109.docx`
  (numbers cross-checked against the fact sheet on 10 Nov: no number in the sheet that is not in the facts).
- **Caps (PI 9 Nov):** §3.4 ≤ 261 · §3.5 ≤ 92 · §3.6 ≤ 247 English words (count: `count_subsection_words.py` method).

### §3.4 — PI text received 2026-11-10 — translated, **NOT YET INSERTED** (flags open)

**PI text (Indonesian, verbatim):**

> ### 3.4 Kriteria seleksi tidak memiliki optimum di bagian dalam rentang
>
> Kriteria seleksi yang digunakan adalah AUC yang dihitung pada *background* milik desain itu sendiri. Dalam dunia
> sintetis, nilainya meningkat dari 0,7367 pada *hard-negative share* 0,0 menjadi 0,8904 pada 1,0. Pada data nyata, AUC
> bergerak dari 0,7208 menjadi 0,8435 sepanjang sebelas titik pengujian; terdapat satu penurunan sebesar 0,0071 antara
> 0,0 dan 0,1, kemudian nilainya terus meningkat. Dengan demikian, pada kedua jenis data, nilai maksimum berada di ujung
> rentang yang diuji, bukan di bagian dalamnya. Kriteria ini tidak memberikan petunjuk kapan penambahan *hard negatives*
> sebaiknya dihentikan.
>
> Dampak praktisnya masih kecil pada grid E013, yang hanya mencakup *share* 0,0, 0,15, dan 0,30. Biaya pemilihan desain
> pada grid tersebut adalah +0,0044 pada data nyata; pada 60 kasus sintetis, mediannya +0,0000, rata-ratanya +0,0012, dan
> nilai maksimumnya +0,0088. E013 berhenti pada 0,30 karena grid eksperimennya berakhir di sana, bukan karena kriteria
> seleksi menunjukkan bahwa titik tersebut merupakan tempat untuk berhenti. Risikonya menjadi jauh lebih besar jika
> rentang diperpanjang sampai 1,0: biaya mencapai +0,0973 pada data nyata dan median +0,1937 pada data sintetis, dengan
> biaya positif pada 60/60 kasus. Kriteria memilih hybrid(1.0) pada 60/60 kasus sintetis, tetapi desain itu bukan yang
> memiliki AUC sebenarnya terendah pada satu pun kasus (0/60); biayanya terhadap desain terbaik adalah +0,194. Jadi,
> masalahnya bukan bahwa kriteria selalu memilih desain terburuk, melainkan bahwa kriteria tersebut tidak menyediakan
> mekanisme untuk memberi tanda berhenti ketika penambahan *hard negatives* mulai merugikan.

**English (faithful translation, 264 words; heading and label kept from v0.4):**

> The selection criterion is the AUC computed on a design's own background. In the synthetic worlds it rises from
> 0.7367 at a hard-negative share of 0.0 to 0.8904 at 1.0. On the real data it moves from 0.7208 to 0.8435 across the
> eleven points tested, with one dip of 0.0071 between 0.0 and 0.1 and a continuous rise after it. In both kinds of
> data, then, the maximum lies at the end of the range tested, not inside it (Figure~\ref{fig:dial}). The criterion
> gives no indication of when to stop adding hard negatives.
>
> The practical consequence is still small on the E013 grid, which covers only the shares 0.0, 0.15 and 0.30. The cost
> of the design choice on that grid is +0.0044 on the real data; over 60 synthetic cases the median is +0.0000, the mean
> +0.0012 and the maximum +0.0088. E013 stopped at 0.30 because its grid ended there, not because the criterion marked
> that point as the place to stop. The risk becomes much larger if the range is extended to 1.0: the cost reaches
> +0.0973 on the real data and a median of +0.1937 on the synthetic data, positive in 60/60 cases. The criterion selects
> hybrid(1.0) in 60/60 synthetic cases, but in no case (0/60) is that the design with the lowest true AUC; its cost
> against the best design is +0.194. The problem, then, is not that the criterion always selects the worst design, but
> that it provides no mechanism to signal a stop when adding hard negatives begins to do harm.

**Numbers verified 10 Nov (all match):** `e222_runs.csv` (hybrid own 0.7367/0.7820/0.8412/0.8904; per case: selected
hybrid(1.0) 60/60 via `e222_selection.csv`; truth-worst hybrid(0.0) 50, hybrid(0.3) 8, hybrid(0.7) 2, hybrid(1.0) 0;
cost on the full grid median 0.1937, positive 60/60); `e218b_hardfrac_sweep.csv` (5-seed × 3-algorithm means: own
0.7208 → 0.8435, the only fall 0.0 → 0.1 = −0.0071; common 0.6991 → 0.6018; E013-grid cost 0.6991 − 0.6947 = 0.0044;
full-grid cost 0.6991 − 0.6018 = 0.0973). Translation tightened for length only ("used", "experimental", "selection"
before "criterion" in the second mention, "the addition of" → "adding"); no content changed.

**Flags raised before insertion (PI's answers to be recorded):**

| # | Flag | Proposal | PI |
|---|---|---|---|
| **M1** | Figure 4 (`fig:dial`) is cited only in this subsection; without a reference it would be uncited | Add "(Figure~\ref{fig:dial})" after "not inside it" (already in the English above) | open |
| **F4** | "Biaya" is no longer defined anywhere (v0.3 defined it here: against the configuration with the highest true AUC on the same grid) | The PI adds a short definition in the PI's own words. Facts: synthetic = true AUC of the best design on the same grid minus that of the design the criterion selects; real data = the same with common-background AUC (no truth) | open |
| **F5** | Last clause "ketika penambahan hard negatives mulai merugikan" holds on the real data (common AUC 0.6991 → 0.6947 over 0.0–0.3, then down to 0.6018) but **not on the synthetic data**: there the hybrid design's true AUC *rises* with the share (0.5407 → 0.6171); the cost comes from the criterion moving from random/TGB (true AUC 0.8278 / 0.8274) to hybrid(1.0), because hybrid(1.0)'s own-background AUC (0.8904) overtakes random's (0.8469) | **A (recommended):** end the sentence at "…memberi tanda berhenti" (−8 words). B: the PI limits the clause to the real data. C: the PI rewrites it | open |
| **F6** | +0.194 is the same quantity as the median +0.1937 a sentence earlier (rounded), so the paragraph gives the same number twice and the second is not marked as a median | Drop "biayanya terhadap desain terbaik adalah +0,194" (−8 words), especially if F4 defines cost against the best design | open |
| **F7** | "hybrid(1.0)" is no longer glossed (v0.3 did, in the passage being replaced) | Render it as "the hybrid design at a share of 1.0" (translation choice, +5 words) | open |

**Length:** 264 against the cap of 261. With the defaults (M1, F4 ≈ +12, F5-A −8, F6 −8, F7 +5) ≈ 265; the session cap
(600 for §3.4–3.6) can absorb it if §3.5 + §3.6 stay ≤ 335.

### §3.5 — PI text received 2026-11-10 — translated, **NOT YET INSERTED** (flags open)

**PI text (Indonesian, verbatim):**

> ### 3.5 AUC yang dilaporkan dan AUC sebenarnya bergerak berbeda
>
> Pada dunia sintetis, ketika *hard-negative share* meningkat dari 0,0 ke 1,0, AUC yang dilaporkan berdasarkan *own
> background* naik sebesar +0,1538, sedangkan AUC sebenarnya hanya naik +0,0764—sekitar setengahnya. Dengan kata lain,
> kenaikan AUC yang dilaporkan sekitar dua kali kenaikan AUC sebenarnya. Pada data nyata, kedua ukuran justru bergerak
> berlawanan: AUC yang dilaporkan naik +0,1227, sementara AUC pada *common background* turun −0,0973. Prediksi tertulis
> E222 bahwa inflasi AUC meningkat seiring *share* tidak terpenuhi: satu uji Spearman gabungan seluruh *runs*
> menghasilkan ρ = 0,4395, di bawah ambang 0,5.

**English (faithful translation, 104 words; cap 92):**

> **Reported and true AUC move differently**
>
> In the synthetic worlds, as the hard-negative share rises from 0.0 to 1.0, the reported (own-background) AUC rises by
> 0.1538 and the true AUC by only 0.0764, about half as much. In other words, the reported AUC rises about twice as
> much as the true AUC. On the real data the two move in opposite directions: the reported AUC rises by 0.1227, while
> the common-background AUC falls by 0.0973. E222's written prediction that AUC inflation increases with the share was
> not met: a single Spearman test pooled over all runs gives $\rho = 0.4395$, below the threshold of 0.5.

**Numbers verified 10 Nov (all match):** `e222_runs.csv`, hybrid means by share: own 0.7367 → 0.8904 (+0.1538 from
unrounded values), true 0.5407 → 0.6171 (+0.0764); endpoint ratio 2.01. `e218b_hardfrac_sweep.csv`: own +0.1227, common
−0.0973. Pooled Spearman(hard_frac, auc_own − auc_true) over the **240 hybrid runs** = 0.4395 (A 0.424, B 0.460);
`e222_outcome.json` `P1_supported: false`. `E222/DESIGN.md` P1: "Spearman > 0.5 across all runs" — F3 of the fact sheet
is resolved by the PI's text ("satu uji Spearman gabungan seluruh runs").

**Translation notes (no content change):** heading translated from the PI's heading (v0.3: "Reported and true
performance diverge"; label `sec:diverge` unchanged); "naik sebesar +0,1538" → "rises by 0.1538" and "turun −0,0973" →
"falls by 0.0973" (a signed number after "falls by" would reverse the meaning); "AUC yang dilaporkan berdasarkan own
background" → "the reported (own-background) AUC"; "AUC pada common background" → "the common-background AUC".

**Flags raised before insertion:**

| # | Flag | Proposal | PI |
|---|---|---|---|
| **F8** | "Dengan kata lain…" restates "sekitar setengahnya" (15 words); §3.5 is 12 over its cap | Drop the sentence → 89 words | open |
| **F9** | The pooled Spearman is over the **240 hybrid runs** (only hybrid has a hard-negative share); "all runs" can be read as all 360 | "pooled over all 240 hybrid runs" (+2) | open |
| **F10** | ρ is **positive** and mean inflation does rise (0.1538 − 0.0764 = 0.077); what failed is the strength threshold set in advance. "Not met" alone can read as "inflation does not rise", against sentence 1 | Add "positif, tetapi" before "di bawah ambang" (+2) | open |

With F8–F10 ≈ 93 words.

### §3.5 — PI text, second version, received 2026-11-10 — **supersedes the first**; translated, NOT YET INSERTED

**PI text (Indonesian, verbatim):**

> ### 3.5 Perbedaan antara AUC yang dilaporkan dan AUC sebenarnya
>
> Di dunia sintetis, kenaikan *hard-negative share* dari 0,0 ke 1,0 membuat AUC yang dihitung menggunakan *background*
> masing-masing desain naik sebesar +0,1538. Namun, AUC sebenarnya hanya naik +0,0764, atau kira-kira setengahnya.
> Artinya, kenaikan yang terlihat pada AUC yang dilaporkan sekitar dua kali lebih besar daripada kenaikan AUC
> sebenarnya. Pada data nyata, polanya justru berlawanan: AUC yang dilaporkan naik +0,1227, sedangkan AUC pada *common
> background* turun −0,0973. Prediksi tertulis E222 bahwa inflasi AUC akan meningkat seiring bertambahnya *share* juga
> tidak terpenuhi. Dalam satu uji Spearman yang menggabungkan seluruh *runs*, nilainya hanya ρ = 0,4395, di bawah
> ambang 0,5.

**English (faithful translation, 117 words; cap 92):**

> **The difference between reported and true AUC**
>
> In the synthetic worlds, raising the hard-negative share from 0.0 to 1.0 makes the AUC computed on each design's own
> background rise by 0.1538. The true AUC, however, rises by only 0.0764, or roughly half as much. The rise seen in the
> reported AUC is therefore about twice the rise in the true AUC. On the real data the pattern is the opposite: the
> reported AUC rises by 0.1227, while the common-background AUC falls by 0.0973. E222's written prediction that AUC
> inflation would increase with the share was also not met. In a single Spearman test pooled over all runs, the value
> is only $\rho = 0.4395$, below the threshold of 0.5.

Numbers unchanged from the first version (verified above). Same sign convention for "rises/falls by".

**Flags (first-version flags re-checked against this text):**

| # | Flag | Proposal | PI |
|---|---|---|---|
| **F8** | still applies: "Artinya…" restates "kira-kira setengahnya" (14 words); §3.5 is now 25 over its cap | drop it → ≈ 103 | open |
| **F9** | still applies: the Spearman is over the 240 hybrid runs, not all 360 | "seluruh 240 run hybrid" (+2) | open |
| **F10** | still applies, in a new form: "hanya ρ = 0,4395" stresses weakness but does not say ρ is positive; mean inflation does rise (+0.077), only the strength threshold failed | "positif, tetapi di bawah ambang 0,5" (+2) | open |
| **F11** | new: "juga tidak terpenuhi" — "juga" has no earlier failed prediction in this subsection to refer back to | drop "juga" (−1) | open |

### §3.6 — PI text received 2026-11-10 — translated, **NOT YET INSERTED** (flags open)

**PI text (Indonesian, verbatim):**

> ### 3.6 TGB tidak menunjukkan keunggulan yang konsisten
>
> Di empat rezim sintetis yang diuji, TGB tidak memberi keunggulan yang konsisten atas desain random. Pada gabungan
> World A dan B, selisih Jaccard untuk kesesuaian peta pada desil teratas adalah −0,010; TGB menghasilkan nilai lebih
> tinggi pada 28/60 pasangan (46,7%), masih di bawah ambang 60% yang ditetapkan. Hasil per rezim juga berbeda. Di World
> A, selisihnya −0,0247 dengan 30% pasangan positif. Di World B, TGB sedikit lebih baik, dengan Jaccard 0,4504 dibanding
> 0,4458 dan 63,3% pasangan positif. Untuk AUC sebenarnya, hasil World C adalah −0,0010 (17/30 pasangan positif; 56,7%),
> sedangkan World D +0,0022 (22/30; 73,3%). Hasilnya bergantung pada rezim yang diuji, bukan menunjukkan keunggulan TGB
> yang berlaku secara konsisten.
>
> E224 menguji apakah tidak dimasukkannya *road_dist* sebagai fitur model dapat menjelaskan hasil di World A. Eksperimen
> ini menggunakan kondisi World A yang sama, dengan satu perubahan: *road_dist* dimasukkan sebagai fitur pada kelompok
> uji, sementara kelompok kontrol tetap tanpa fitur tersebut. Kedua cabang ditetapkan sebelum eksperimen dijalankan,
> dengan total 120 *runs* atau 30 pasangan untuk setiap kelompok. Hasil metrik utamanya tidak mendukung penjelasan
> tersebut. Selisih Jaccard TGB−random adalah −0,0254 pada kontrol dan −0,0217 pada kelompok yang menggunakan
> *road_dist*; proporsi pasangan positif tetap 30% pada kedua kelompok. Karena itu, E224 berstatus FAILED. Pada metrik
> sekunder berupa AUC sebenarnya, proporsi positif naik dari 36,7% menjadi 60,0%, tetapi hasil ini tidak mengubah
> kesimpulan dari metrik utama. Perlu dicatat pula bahwa *road_dist* berkorelasi +0,49 dengan *river_dist* dan +0,31
> dengan elevasi, sehingga model kontrol mungkin sudah menangkap sebagian sinyal bias jalan.

**English (faithful translation, 281 words; cap 247):**

> **TGB shows no consistent advantage**
>
> Across the four synthetic regimes tested, TGB gives no consistent advantage over the random design. For Worlds A and
> B combined, the TGB − random difference in top-decile map agreement (Jaccard) is −0.010; TGB scores higher in 28/60
> pairs (46.7%), still below the specified threshold of 60%. The results also differ by regime. In World A the
> difference is −0.0247, with 30% of pairs positive. In World B TGB does slightly better, with a Jaccard of 0.4504
> against 0.4458 and 63.3% of pairs positive. For true AUC, World C gives −0.0010 (17/30 pairs positive; 56.7%) and
> World D +0.0022 (22/30; 73.3%). The outcome depends on the regime tested; it does not show a TGB advantage that holds
> consistently.
>
> E224 tested whether leaving road_dist out of the model features could explain the result in World A. The experiment
> used the same World A conditions with one change: road_dist was included as a feature in the test group, while the
> control group remained without it. Both decision branches were fixed before the experiment was run, with 120 runs in
> total, or 30 pairs per group. The result on the primary metric does not support the explanation. The TGB − random
> Jaccard difference is −0.0254 in the control and −0.0217 in the group with road_dist; the share of positive pairs
> stays at 30% in both. E224 is therefore recorded as failed. On the secondary metric, true AUC, the share of positive
> pairs rises from 36.7% to 60.0%, but this does not change the conclusion drawn from the primary metric. It should also
> be noted that road_dist correlates +0.49 with river_dist and +0.31 with elevation, so the control model may already
> capture part of the road-bias signal.

**Numbers verified 10 Nov (all match):** `e222_runs.csv` TGB − random Jaccard A+B −0.010, 28/60; A −0.0247, 30 %
(0.6898 vs 0.7145); B +0.0046, 63.3 % (0.4504 vs 0.4458); `e222c_runs.csv` true AUC −0.0010, 17/30; `e222d_runs.csv`
+0.0022, 22/30; `E224/results/e224_outcome.json`: no_road Jaccard −0.0254 (30 %), with_road −0.0217 (30 %); auc_true
36.7 % → 60.0 %; collinearity river_dist 0.4916, elevation 0.3142; 120 runs, 30 pairs per arm.

**Translation notes:** heading from the PI (v0.3: "The target-group-background null, reported as unexplained"; label
`sec:tgbnull` unchanged — Methods line 317 points here); "berstatus FAILED" → "recorded as failed" (lower case: the
capitals are the repo's status label); "ambang 60% yang ditetapkan" → "the specified threshold of 60%".

**Flags raised before insertion:**

| # | Flag | Proposal | PI |
|---|---|---|---|
| **F12** | Worlds A–D are described nowhere in the manuscript: Methods (line 312) says only "four synthetic worlds"; v0.3 characterised B, C, D in the passage now replaced; §3.10 describes C and D, but after this first use. Facts (`E222/README.md`, `DESIGN.md`): A = intensity fully observed, road-shaped survey bias (TGB is the correctly specified bias model); B = A + a real clay raster withheld from the features (misspecified); C = A + regionally uneven survey effort; D = region-balanced truth + regional survey effort; road survey bias in all four | **Carry to the Methods session** (E222 paragraph, PI's words) — no words added here | open |
| **F13** | **K4 rule:** the text never says the TGB null is *unexplained*; Methods line 318 says "the null is reported as unexplained" and points here | The PI adds one short sentence at the end of the second paragraph (≈ 6–8 words) | open |
| **F14** | "Kedua cabang" translated as "both decision branches" (the two pre-written outcomes, as in the fact sheet); if the PI meant the two arms (test/control), the sentence changes | PI confirms. For E224 "pre-registered" is accurate (design `d4f44af` before results `c826b40`) if the PI wants the word | open |
| **F15** | Dropped from v0.3 and absent elsewhere: "a clean test needs a survey-effort surface orthogonal to terrain, which is future work" | Carry to Limitations (Discussion session) — no words here | open |

**Session 2 length:** §3.4 264 + §3.5 117 + §3.6 281 = **662** against the session cap 600 (+62). With the flagged cuts
(F5-A −8, F6 −8, F8 −14, F4 +12, F7 +5, F9/F10 +4, F11 −1, F13 +7) ≈ **659**. Further trims would be the PI's choice
(e.g. "The results also differ by regime", the closing restatement of §3.6 ¶1). Context (JOURNAL 9 Nov): text +
references without tables/figures is 8,834 words for v0.4 — the 11,500 limit is not at risk by that count; the
per-subsection cap is the PI's own rule of 9 Nov.

### Session 2 — PI decisions (2026-11-10) and insertion — **§3.4–3.6 INSERTED** into v0.4 (lines 474–527)

**PI: "Default untuk semua catatan saya terima"**, plus two sentences of the PI's own:

| # | Decision |
|---|---|
| M1 | Figure 4 reference added |
| **F4** | PI's text: *"Biaya pemilihan dihitung sebagai selisih antara AUC desain terbaik dan AUC desain yang dipilih oleh kriteria pada grid yang sama. Untuk data sintetis, perhitungan menggunakan AUC sebenarnya, sedangkan untuk data nyata menggunakan AUC pada common background karena AUC sebenarnya tidak diketahui."* The PI stated the order: best minus selected. → "The selection cost is computed as the AUC of the best design minus the AUC of the design selected by the criterion, on the same grid. For the synthetic data the calculation uses the true AUC, whereas for the real data it uses the AUC on the common background, because the true AUC is not known." Placed before the first use of "cost"; the next sentence now reads "The selection cost on that grid…" (same term as the definition) |
| F5 | A — sentence ends at "…signal a stop." |
| F6 | "+0.194" clause dropped |
| F7 | "hybrid(1.0)" → "the hybrid design at a share of 1.0" |
| F8 | "Artinya…" sentence dropped |
| F9 | "all 240 hybrid runs" |
| F10 | "positive, but below the threshold of 0.5" |
| F11 | "juga" dropped |
| F12 | worlds A–D → **Methods session** (E222 paragraph) |
| **F13** | PI's text: *"Hasil nol TGB masih belum terjelaskan."* (end of ¶2; reason given by the PI: E224 tested one explanation only, so nothing may imply the cause has been found) → "The TGB null result remains unexplained." |
| F14 | "decision branches" confirmed (two pre-set outcomes, not the two arms) |
| F15 | orthogonal survey-effort surface → **Limitations** (Discussion session) |
| Length | PI accepts the overrun; savings to be found in Methods and Discussion without dropping what the reader needs |

**Compile:** clean, **38 pp.**, no undefined references. PDF text of the three subsections checked against the `.tex`.

**Length, measured after insertion** (`count_subsection_words.py` method, same method for both files): v0.3 §3.4–3.6
**609** (268 + 94 + 247) → v0.4 **697** (307 + 103 + 287) = **+88**. (The fact sheet's caps 261/92 were slightly lower
than this method gives for v0.3; the F4 definition came out at ≈ 45 English words, more than the ≈ 12 estimated.)
Results so far: session 1 +263, session 2 +88 → **+351 against v0.3** for §3.1–3.6.

**Carried forward:** F12 (Methods), F15 (Limitations), Figure 2 uncited in the running text (session 1 F4; fix before
freeze), Figure 4 caption still in v0.3 wording ("about twice as fast") — captions are a later session.

---

## Session 3 — Results §3.7–3.9 — prepared 2026-11-10 (PI: "silahkan lanjut")

- **Fact sheet:** `rewrite_r3/FACTS_S3_RESULTS_3_7-3_9.md`; **writing sheet:** `rewrite_r3/LEMBAR_TULIS_S3_20261110.docx`
  (`build_writing_sheet_s3.py`). Numbers re-checked 10 Nov: `e223b_bootstrap_summary.csv`, `e223a_equivalence_ci.csv`,
  `e223d_kstar_thresholds.csv`, `e221_turnover_pairs.csv` (1 − mean Jaccard per algorithm × design: 28.1–47.4 %),
  `e221_priority_sets.csv` (core/fringe sites 169/15, 141/15, 100/16; densities as in v0.4), `F_disjoint_full.csv` /
  `F_disjoint_clean.csv`, `e219_outcome.json` (−0.2811 canonical, 13 in bounds; −0.2435 legacy), the two volcano files.
- **Caps (v0.3 lengths):** §3.7 42 · §3.8 297 (¶2 = PI passage 1, 137) · §3.9 189 (¶2 opening = PI passage 8D, 29).
  New writing ≤ ≈ 362 English words.
- **Authorship of the existing text (rule 6):** §3.7, §3.8 ¶1, §3.9 ¶1 = Claude-drafted (v0.2 + session-5 edits) → PI
  rewrites. §3.8 ¶2 = PI passage 1 (one agreed number change) → PI re-reads. §3.9 ¶2: the first three sentences are PI
  passage 8D; **the rest was written by Claude in session 5 (H19, `apply_session5_edits.py`) and v0.2** → PI rewrites.
- **Options/flags on the sheet:** O1 (keep §3.7 separate — default), F16 (fringe site counts 15/15/16 on the 378 set),
  F17 (Table 4 / Figure 2 need a reference from the text).

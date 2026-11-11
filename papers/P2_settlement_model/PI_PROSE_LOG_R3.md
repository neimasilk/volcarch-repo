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

### §3.7 — PI text received 2026-11-10 — translated, **NOT YET INSERTED** (flags open)

**PI text (Indonesian, verbatim):**

> ### 3.7 Batas deteksi
>
> Pada 378 situs, data ini belum dapat menyingkirkan kemungkinan adanya efek dalam kisaran −0,03 hingga +0,03. Efek
> sekecil itu tidak terdeteksi pada ukuran sampel ini, tetapi bukan berarti efeknya tidak ada.

**English (faithful translation, 37 words; cap 42):**

> **Detection floor**
>
> With 378 sites, these data cannot yet rule out an effect in the range −0.03 to +0.03. An effect that small is not
> detected at this sample size, but that does not mean it is absent.

Heading translated from the PI's (v0.3: "Confidence intervals and a detection floor"; label `sec:ci` unchanged — §3.2 line 429
and Limitations line 672 point here). O1 answered implicitly: §3.7 stays a subsection of its own.

**Flags raised before insertion:**

| # | Flag | Proposal | PI |
|---|---|---|---|
| **F18** | The **lower** end "−0,03" is not what the data show. Bootstrap CIs (`e223b_bootstrap_summary.csv`, hybrid − random, common background): upper bounds +0.0082 / +0.0253 / +0.0256 → "a gain up to about +0.03 cannot be excluded" (E223 README); lower bounds **−0.0589 / −0.0241 / −0.0173** (MaxEnt / XGBoost / RF) — so on the negative side the excluded region differs by algorithm (MaxEnt cannot exclude a loss down to −0.059; RF excludes losses beyond −0.017) | **A (recommended):** one-sided — "a gain of up to about +0.03" (the floor is about detecting a gain). B: give both sides from the data per algorithm (≈ +15 words) | open |
| **F19** | "efek" does not say *of what*; the reader arrives from §3.6 (TGB). The floor is for the background redesign (hybrid − random, common background) | The PI adds who/what in a few words, e.g. which contrast | open |
| **M2** | No link to where the intervals are reported | Add "(Section~\ref{sec:artefact})" after the first sentence (mechanical) | open |

### §3.7 — PI text, second version, received 2026-11-10 — **supersedes the first; INSERTED** into v0.4 (lines 529–533)

**PI text (Indonesian, verbatim):**

> ### 3.7 Batas deteksi
>
> Pada 378 situs, data belum dapat menyingkirkan kenaikan hingga sekitar +0,03 pada efek desain background (*hybrid*
> dibanding *random*) yang diukur dengan *common background* (Section 3.2). Kenaikan sebesar itu belum terdeteksi pada
> ukuran sampel ini, tetapi bukan berarti efeknya tidak ada.

**English inserted:**

> **Detection floor**
>
> With 378 sites, the data cannot yet rule out a gain of up to about +0.03 in the effect of the background design
> (hybrid against random) measured on the common background (Section~\ref{sec:artefact}). A gain of that size is not yet
> detected at this sample size, but that does not mean the effect is absent.

**Flags:** F18 resolved by the PI's text (one-sided, option A); F19 resolved (hybrid vs random, common background); M2
resolved (the PI wrote "(Section 3.2)" → `\ref{sec:artefact}`). Number checked: bootstrap upper bounds +0.0082 /
+0.0253 / +0.0256 (`e223b_bootstrap_summary.csv`) → "up to about +0.03". No change beyond translation.

### §3.8 ¶1 — PI text received 2026-11-10 — translated, **NOT YET INSERTED** (flags open)

**PI text (Indonesian, verbatim):**

> ### 3.8 Peta prioritas berubah mengikuti seed
>
> Dalam praktik konservasi, peta prioritas lebih langsung digunakan untuk menentukan lokasi yang perlu diperiksa atau
> dilindungi daripada angka AUC saja. Namun, peta tersebut ternyata cukup sensitif terhadap seed. Pada 405 pasangan
> perbandingan, sekitar 28,1–47,4% sel dalam kelompok 10% teratas berganti hanya karena seed yang digunakan berbeda
> (Figure 6). RF dengan TGB relatif paling stabil (28,1%), sedangkan XGBoost dengan hybrid paling banyak berubah (47,4%).
> Menggabungkan beberapa seed memang membantu, tetapi jumlah yang dibutuhkan bergantung pada cara kesepakatan diukur.
> Perbandingan dengan ensemble 10 seed yang masih berbagi seed dengan ensemble pembanding memberi kesan bahwa 4–7 seed
> sudah cukup untuk mencapai Jaccard di atas 0,90. Ukuran yang lebih adil adalah membandingkan dua ensemble yang tidak
> berbagi seed: pada lima seed, Jaccard hanya 0,76 untuk XGBoost, 0,85 untuk MaxEnt, dan 0,86 untuk RF; dengan satu seed
> nilainya berkisar 0,55–0,70. Karena itu, saya menyarankan minimal tujuh seed sebagai batas bawah, disertai pelaporan
> kesepakatan antara dua ensemble independen. Tujuh seed bukan jaminan bahwa peta sudah stabil.

**English (faithful translation, 192 words; cap 160 for ¶1):**

> **Priority maps change with the seed**
>
> In conservation practice, it is the priority map, rather than the AUC figure alone, that is used more directly to
> decide which locations should be examined or protected. These maps, however, turn out to be fairly sensitive to the
> seed. Across 405 paired comparisons, about 28.1–47.4% of the cells in the top 10% change simply because a different
> seed is used (Figure 6). Random Forest with TGB is comparatively the most stable (28.1%), and XGBoost with hybrid
> changes the most (47.4%). Combining several seeds does help, but how many are needed depends on how agreement is
> measured. A comparison with a 10-seed ensemble that still shares seeds with the ensemble being compared gives the
> impression that 4–7 seeds are enough to reach a Jaccard above 0.90. A fairer measure is to compare two ensembles that
> share no seed: with five seeds, the Jaccard is only 0.76 for XGBoost, 0.85 for MaxEnt and 0.86 for Random Forest; with
> one seed it ranges from 0.55 to 0.70. We therefore recommend at least seven seeds as a floor, together with reporting
> the agreement between two independent ensembles. Seven seeds do not guarantee that a map is stable.

**Numbers verified 10 Nov:** `e221_turnover_pairs.csv` — 405 pairs (9 algorithm × design cells × 45); **means per cell**
28.1 % (RF-TGB) to 47.4 % (XGBoost-hybrid); **single pairs 16.6–52.4 %**; overall mean 35.1 %. `e223d_kstar_thresholds.csv`:
k* 4–7 at J ≥ 0.90. `F_disjoint_full.csv`: 1 seed 0.548–0.702; 5 seeds XGBoost 0.759, MaxEnt 0.849, RF 0.861.

**Translation notes:** heading from the PI (v0.3: "…; averaging seeds reduces this"; label `sec:prioritymaps` unchanged);
"RF" → "Random Forest" (the abbreviation is not defined in the text).

**Flags raised before insertion:**

| # | Flag | Proposal | PI |
|---|---|---|---|
| **F20** | "Pada 405 pasangan …, sekitar 28,1–47,4%" reads as the range over single pairs; it is the range of the **averages** over the 9 algorithm × design cells (45 pairs each). Single pairs range 16.6–52.4 % | Say "on average, depending on algorithm and design" (≈ +5 words), or give the single-pair range instead | open |
| **F21** | "dua ensemble independen": the two ensembles use the same data and design; only their seeds are disjoint (the previous sentence says so) | "two ensembles that share no seed" (+2 words) | open |
| **M3** | "(Figure 6)" sits after the turnover sentence, but Figure 6 shows agreement as seeds are averaged (the k-vs-10 and the disjoint curves), not the pairwise turnover | Move "(Figure 6)" to the end of the disjoint-ensemble sentence | open |
| **T1** | "saya menyarankan" — the paper has two authors and uses "we" throughout | "We recommend" (already in the English above) | open |

**§3.8 ¶1 resent 2026-11-10** with no visible change; treated as the same text — flags F20, F21, M3, T1 stay open.

### §3.8 ¶2 — PI decision 2026-11-10: passage 1 (21 Oct) kept, **F16 accepted** — **INSERTED** into v0.4

**PI's reason (paraphrase):** the fringe site counts are small, so the density ratios should not be read as precise.

**PI text (Indonesian, verbatim):**

> Ensemble ini menunjukkan perbedaan yang bermanfaat antara *robust core* (wilayah inti yang konsisten) dan *contingent
> fringe* (wilayah pinggiran yang bergantung pada desain latar). *Robust core* didefinisikan sebagai sel yang masuk dalam
> 10% peringkat teratas pada ketiga desain latar (random, TGB, dan hybrid), dengan setiap peta merupakan rata-rata dari 10
> seed; sedangkan *contingent fringe* adalah sel yang masuk dalam 10% peringkat teratas pada tepat satu desain. Kepadatan
> situs yang teramati pada wilayah inti lebih tinggi dibandingkan wilayah pinggiran, yaitu 1,93× untuk Random Forest
> (30,7 dibandingkan 15,9 situs per 1.000 km²), 4,34× untuk XGBoost (40,8 dibandingkan 9,4), dan 5,62× untuk MaxEnt (31,7
> dibandingkan 5,65). Namun, wilayah pinggiran hanya mencakup 15 situs untuk XGBoost, 15 untuk Random Forest, dan 16 untuk
> MaxEnt. Karena itu, rasio ini lebih tepat menunjukkan arah perbedaan daripada ukuran yang presisi. Perbedaan kepadatan
> tersebut menunjukkan konsistensi spasial, bukan validasi independen, karena situs yang digunakan untuk menghitung
> kepadatan juga digunakan untuk melatih model.

**English inserted (v0.4 §3.8 ¶2, 174 words):** the 21 Oct translation, with "RandomForest" → "Random Forest" (the PI's
spelling now), two new sentences — "However, the fringe contains only 15 sites for XGBoost, 15 for Random Forest and 16
for MaxEnt. These ratios are therefore better read as showing the direction of the difference than as a precise
measure." — and the last sentence opening "The density difference indicates …" (PI's "Perbedaan kepadatan tersebut").
"(Figure 5)" kept from the 21 Oct version (not in the PI's Indonesian; Figure 5 is cited here and in Section 4).

**Numbers checked:** `e221_priority_sets.csv` — fringe sites 15 (XGBoost), 15 (RF), 16 (MaxEnt); densities and ratios as
before. Compile clean, 38 pp., no undefined references.

**Length:** ¶2 137 → 174 (+37).

### §3.9 ¶1 — PI text received 2026-11-10 — translated, **NOT YET INSERTED** (flags open)

**PI text (Indonesian, verbatim):**

> ### 3.9 Koreksi pada dua diagnostik gunung api
>
> Kode analisis memuat tujuh pusat gunung api, sedangkan inventaris kanonik mencakup 13 pusat di dalam batas peta
> penelitian (111–115° BT). Koreksi inventaris ini tidak mengubah AUC model karena *volcano_dist* tidak digunakan sebagai
> fitur pelatihan. Namun, hasil Test 1 berubah karena tes ini mengukur korelasi antara kesesuaian spasial dan jarak ke
> gunung api. Dengan 13 pusat, nilai Spearman menjadi ρ = −0,281, dibandingkan −0,243 dengan tujuh pusat. Kedua nilai itu
> tetap memenuhi kriteria |ρ| < 0,5, sehingga Test 1 tetap lulus.

**English (faithful translation, 95 words; cap 93; heading kept "Corrections to two volcanic diagnostics"):**

> The analysis code contained seven volcanic centres, whereas the canonical inventory includes 13 centres within the
> bounds of the study area (111–115°E). This correction to the inventory does not change the model AUCs, because
> volcano_dist is not used as a training feature. The result of Test 1 does change, however, because this test measures
> the correlation between predicted suitability and distance to volcanoes. With 13 centres the Spearman value becomes
> ρ = −0.281, against −0.243 with seven centres. Both values still meet the criterion |ρ| < 0.5, so Test 1 still passes.

**Numbers verified:** `E219/results/e219_outcome.json` (−0.2811 canonical, 13 in bounds; −0.2435 legacy 7), both from the
same E219 re-run (5 seeds); `volcanoes.csv` (7) vs `volcanoes_java_full.csv` (13 in 111–115° E). Translation note:
"kesesuaian spasial" → "predicted suitability" (what Test 1 correlates, Methods line 274).

**Flags raised before insertion:**

| # | Flag | Proposal | PI |
|---|---|---|---|
| **F23** (ledger **C094**) | "Test 1 tetap lulus": the original code (`enhanced_tautology_tests.py` l. 64–65, 213–235) has **three** levels on the maximum \|ρ\| over three proxies — PASS ≤ 0.30, GREY_ZONE 0.30–0.50, FAIL > 0.50. The original Test 1 verdict was **GREY_ZONE** (road_dist \|ρ\| 0.307, single run; v0.1 l. 307; E219 README "GREY_ZONE verdict survives"). The volcano correlation alone (0.243 / 0.281) is below even 0.30. "Passes" is true only in the two-level reading of Methods l. 274 | **A (recommended):** "…tetap di bawah ambang gagal 0,5, sehingga vonis Test 1 tidak berubah" (same length). Methods (later session): state the three levels or the GREY_ZONE verdict — PI writes | open |
| **F22** (optional) | The six added centres and the two inside the Malang–Mojokerto site concentration (Penanggungan, Kawi-Butak) are no longer named; they explain *why* ρ moves | Leave out (the PI chose brevity) — default | open |
| **M4** | ¶2 opens with the PI's run-in heading "A correlation that is not consistent across seeds."; ¶1 now has none (v0.2's "The volcano inventory." was Claude's) | Add the label "The volcano inventory." to ¶1 for parallel structure (3 words), or drop both labels | open |

### §3.9 ¶2 — PI decision and closing, received 2026-11-10 — **INSERTED**

**PI:** the three sentences of passage 8D are kept ("Pada versi sebelumnya" stays removed), opened by the run-in label.
The closing names the sampling change explicitly so that the change in correlation is not attributed to the seed alone.

**PI text, closing (Indonesian, verbatim):**

> Selisih ini tidak bisa dikaitkan dengan seed saja. E219 juga mengubah cara sampling background: menggunakan lattice,
> bukan koordinat kontinu seperti pada pipeline asli. Karena keduanya berubah bersamaan, pengaruh seed dan sampling tidak
> dapat dipisahkan. Satu instans model karena itu tidak cukup menjadi dasar diagnostik ini. Nilai di sini memakai
> ensemble lima seed; Tabel 4 menyajikan rangkaian E007–E013 dari pipeline asli sebagai objek yang diperiksa.

**English inserted:**

> This difference cannot be attributed to the seed alone. E219 also changed the way the background is sampled, using a
> lattice rather than the continuous coordinates of the original pipeline. Because both changed together, the effects of
> seed and sampling cannot be separated. A single model instance is therefore not a sufficient basis for this
> diagnostic. The values here use a five-seed ensemble; Table~\ref{tab:progression} presents the E007--E013 series from
> the original pipeline as the object under examination.

**Checked:** E219 `01_map_divergence.py` l. 279–283 — both Test 1 correlations are computed on the five-seed ensemble map
(mean of XGBoost-hybrid seeds), so "five-seed ensemble" is accurate; the lattice is described in Methods
(`sec:refutation`, l. 292).

### Session 3 — PI decisions (2026-11-10) and insertion — **§3.7–3.9 INSERTED** into v0.4

| # | Decision |
|---|---|
| O1 | §3.7 stays a subsection of its own (implicit in the PI's text) |
| F18, F19, M2 | resolved by the PI's second version of §3.7 |
| F16 | accepted (fringe site counts 15/15/16 in §3.8 ¶2) |
| **F20** | accepted → "an average of 28.1–47.4% of the cells in the top 10%, depending on the combination of algorithm and design" |
| **F21** | accepted → "two ensembles that share no seed" |
| **M3** | "(Figure 6)" moved to the end of the disjoint-ensemble sentence |
| **T1** | "We recommend" |
| F22 | the six added centres are not named |
| **F23** (C094) | option A → "Both correlations remain below the failure threshold of 0.5, so the verdict of Test 1 does not change." ("passes" removed) |
| **M4** | run-in label "The volcano inventory." added to §3.9 ¶1 |
| **F17** | Table 4 referenced in the §3.9 ¶2 closing |

**Removed from v0.4 with these insertions (Claude-drafted, v0.2/session 5):** §3.7 two sentences; §3.8 ¶1; §3.9 opening
"Two diagnostics of the original pipeline are corrected here." and ¶1; §3.9 ¶2 H19 sentences and the v0.2 last sentence.

**Compile:** clean, **38 pp.**, no undefined references; PDF text checked.

**Length** (same method both files): §3.7 42 → 55 · §3.8 297 → 381 · §3.9 189 → 203 · **session 3: 528 → 639 (+111)**.
Results §3.1–3.9: **+462** against v0.3 (+263, +88, +111). The PI's rule: savings in Methods/Discussion.

**Open, for the PI (optional):** the run-in label "A correlation that is not consistent across seeds" now sits above a
closing that says the change *cannot be attributed to the seed alone*. The label is the PI's (22 Oct, kept today); a
label such as "a correlation that does not reproduce" would match the closing. Not changed.

---

## Session 4 — Results §3.10–3.11 — prepared 2026-11-11

- **Fact sheet:** `rewrite_r3/FACTS_S4_RESULTS_3_10-3_11.md`; **writing sheet:** `rewrite_r3/LEMBAR_TULIS_S4_20261111.docx`
  (`build_writing_sheet_s4.py`, same layout as session 3).
- **Authorship separated before anything else** (fact-sheet §0, sentence by sentence against v0.4 lines 592–628, R2
  passages 2 / 3a / 8E and the session-5/6 scripts):
  - **Claude sentences → the PI rewrites:** line 609 (the two pre-registered sensitivity definitions + post hoc range,
    R6a session 5 / P12 session 6), line 627 (H20), line 628 (v0.2 text, quota design — the PI's 22 Oct note kept the
    results and numbers, not the wording).
  - **The PI's sentences with Claude insertions → re-read:** line 610 ("In the uplands (400–1,000 m)," + `smyth2008`,
    R6b), line 611 (Figure 5 → Section 3.6, R6c), line 612 (object of the limited-power sentence made explicit).
  - Everything else in §3.10 (¶1–¶2, R2 passage 2; verdict labels, passage 3a) and §3.11 sentence 1 (8E) is the PI's.
- **What session 6 did *not* touch here:** §3.10–3.11 are **identical in v0.3 and v0.4** (diff = 0 lines). The K1
  clean-set sentence for E235 sits in §3.3 (v0.4 l. 469–470), which was re-verified today: clean set hybrid +0.0597
  [+0.0320, +0.0855] SPECIFIC, summary INCONCLUSIVE, and the 10/30 vs 15/30 labels do exchange between the sets.
- **Numbers re-derived from source (2026-11-11):** E235 `e235_outcome.json` (M1 3,434.52 km² / 0.97479 / 0.00435;
  M2 random +0.0491 [+0.0237, +0.0768], TGB +0.0489, hybrid +0.0599 [+0.0298, +0.0914] SPECIFIC, summary INCONCLUSIVE;
  sens 10/30 −0.0023 [−0.0471, +0.0458] NOT SPECIFIC, 15/30 +0.0164 [−0.0155, +0.0514] INCONCLUSIVE), `e235_bands.csv`
  (14.95 km² at 1,000–1,200 m, zero above), AUDIT §4 (post hoc decomposition: +0.005 / +0.067 extremes; E\* 800–1,200 m),
  E221 README (fringe median elevation 1,107 m), `volcanoes_java_full.csv` (30 centres, 17 outside 111–115°E),
  `e222c/d_outcome.json` (World C −0.2457 / −0.4688 / 0-of-30; World D −0.2027 / −0.2826 / 0-of-30).
- **Flags opened:** F24 (the 10/30 definition also moves E\* to 1,200 m and widens the far zone from 40 to 30 km, so the
  drop to −0.002 is not "a tighter volcanic zone"), F25 (¶3 numbers are XGBoost; the running text does not say so),
  F26 (quota sentence is v0.2 text), F27 ("pre-registered" is accurate for E235 — `3ac41cc` pushed before the script),
  F28, F29 (Claude's insertions in the PI's sentences), F30 (§3.3 already carries the clean-set figures; do not repeat
  at a 582/582 cap).
- **Length:** §3.10 582 and §3.11 86 in **both** v0.3 and v0.4 — exactly at the PI's cap; the new sentences must be the
  length of the ones they replace.
- **Status:** waiting for the PI's text (session 4). Nothing inserted yet; nothing committed.

### Session 4 — the PI's text and insertion (2026-11-11) — **§3.10 ¶3 and §3.11 paragraph INSERTED** into v0.4

**The PI's decisions on the flags:** F24 (report the result, not the cause of the drop) · F25 → solved by his own
"Pada XGBoost" · F26 (rewrite the quota sentence) · F27 (keep "pre-registered") · F28, F29 (keep) · F30 (do not repeat
the clean-set figures) · **F31** (name the random design) · **F32** (both mechanical trims accepted) · **F33** (no CI).

**The PI's text (§3.10 ¶3, written in Indonesian):**
> Pada XGBoost, dua definisi sensitivitas yang dipraregistrasikan menghasilkan selisih lebih kecil: −0,002 (NOT SPECIFIC)
> untuk V ≤ 10 km dan N ≥ 30 km, serta +0,016 (INCONCLUSIVE) untuk V ≤ 15 km dan N ≥ 30 km. Dalam analisis post hoc,
> selisih berkisar dari +0,005 hingga +0,067, bergantung pada definisi zona dan pencocokan. Pencocokan tambahan
> berdasarkan jarak ke sungai menurunkannya menjadi +0,028. Nilai +0,05 tepat berada pada ambang keputusan, sehingga hasil
> ini tidak memberi dasar untuk menyatakan arah efek di bawah 1.000 m.

**Inserted (v0.4, l. 611–612):**
> For the random design on XGBoost, the two pre-registered sensitivity definitions give smaller differences: $-0.002$
> (NOT SPECIFIC) for near land within 10~km and far land beyond 30~km, and $+0.016$ (INCONCLUSIVE) with 15 and 30~km.
> The post hoc variants give $+0.005$ to $+0.067$, depending on the zone definitions and the matching procedure;
> additional matching on river distance lowers it to $+0.028$. The $+0.05$ sits exactly on the decision threshold, so
> the result gives no basis for claiming a direction below 1,000~m.

**The PI's text (§3.11, two sentences) and what was inserted:**
> Rendahnya kesesuaian prediksi di dekat gunung api, jika dilihat sendiri, tidak dapat dijadikan bukti bahwa permukiman
> telah terkubur.
>
> Desain kuota juga gagal, baik pada World C, yaitu rezim ketimpangan survei antarkawasan yang menjadi alasan
> pengembangannya (selisih AUC sebenarnya −0,2457; Jaccard −0,4688; 0/30 sel positif), maupun pada World D, rezim
> kebenaran seimbang yang paling menguntungkannya (masing-masing −0,2027 dan −0,2826; 0/30 sel positif).

> Low predicted suitability near a volcano, on its own, cannot be taken as evidence that a settlement was buried.
> The quota design also fails, both on World C, the regime of uneven survey effort between regions that motivated its
> development (true AUC difference $-0.2457$; Jaccard $-0.4688$; 0/30 positive cells), and on World D, the balanced-truth
> regime most favourable to it ($-0.2027$ and $-0.2826$ respectively; 0/30).

**Changes beyond translation (told before insertion):**
1. **F31** — "For the random design" added before "on XGBoost" (the values are the random design's; TGB gives +0.0014 /
   +0.0181 and hybrid +0.0068 / +0.0257). The PI agreed in advance.
2. **Terminology:** his "V ≤ 10 km / N ≥ 30 km" → "near land within 10~km and far land beyond 30~km" (the section's own
   terms; the symbols V/N appear nowhere in the text).
3. **"true AUC", no hyphen**, matching §3.6 (l. 516) and §3.4; a hyphen broke across lines in the PDF.
4. **Ellipse:** the second "0/30" drops "positive cells" (identical to the first parenthesis, −2 words).
5. **"claiming a direction below 1,000 m"** drops "of the effect" (−3), matching the phrase already accepted in v0.3
   ("No direction is therefore claimed below 1,000 m").
6. **"additional matching on river distance"** — the PI's "pencocokan tambahan"; "as well" was replaced, −1 word.
7. **F32 trims applied** (the PI agreed): l. 599 "and INCONCLUSIVE for any result that meets neither criterion" →
   "and INCONCLUSIVE otherwise" (−6); l. 603 "This means that, in this study area," → "In this study area," (−3).

**Numbers re-verified before insertion (11 Nov):** −0.0023 [−0.0471, +0.0458] NOT SPECIFIC; +0.0164 [−0.0155, +0.0514]
INCONCLUSIVE; post hoc +0.005 … +0.067 (AUDIT §4); river matching +0.028; World C −0.2457 / −0.4688 / 0-of-30; World D
−0.2027 / −0.2826 / 0-of-30. All match the manuscript.

**Removed (Claude-drafted):** the old sensitivity/post hoc sentence (R6a/P12); H20; the v0.2 quota sentence.

**Length** (same method): §3.10 **582 → 580** (−2) · §3.11 **86 → 83** (−3). Session 4 adds no words.

**Compile:** clean, **38 pp.**, 0 undefined references; PDF text checked (pp. 12, 13).

**Flag left open — F34:** §3.11 says "0/30 **positive cells**" (the PI's "sel"), while §3.6 (l. 516) calls the same 30
units "**pairs** positive". Default: align §3.11 to "pairs". Awaiting the PI; one word.

**F34 — closed the same day:** §3.11 now reads "0/30 pairs positive", matching §3.6 (l. 516, "17/30 pairs positive");
terminal only, no number changed. Compile clean, 38 pp. Session 4 complete.

---

## Session 5 — Methods §2.1–2.6 — prepared 2026-11-12

- **Fact sheet:** `rewrite_r3/FACTS_S5_METHODS_2.md`; **writing sheet:** `rewrite_r3/LEMBAR_TULIS_S5_20261112.docx`
  (`build_writing_sheet_s5.py`). The PI's order of priorities (11 Nov): **F12 → C094 → word reduction**.
- **Authorship (§0 of the sheet):** Methods is ± 85 % Claude's text (v0.2 era + sessions 5–6), so this is a rewrite,
  not a re-read. The PI's own text inside Methods: **passage 7** (hard-negative paragraph, ↔ C080), most of **passage
  8B** (§2.4, with Claude's C1/C2 insertions) and all of **passage 8C** (§2.5, E014). Everything in §2.1, §2.2 and
  §2.6 is Claude's.
- **F12 (worlds A–D)** — facts from `E222/README.md` + `DESIGN.md`: A = intensity fully observed + road-shaped survey
  bias; B = A + clay withheld from features (misspecified); C = A + regional survey effort [1.0, 0.4, 0.15, 0.05]
  (region-0 share 43 %); D = region-balanced truth + regional effort (region-0 share 62.6 %); 10 worlds each; observed
  n ≈ 300–500; configs random/tgb/hybrid at hard_frac {0.0, 0.3, 0.7, 1.0}; truth = auc_true against an independent
  unbiased sample + uniform availability, map recovery by Jaccard/Spearman; **E013's grid (0.0/0.15/0.30) ≠ E222's**.
- **C094 (Test 1's three levels)** — verified in `enhanced_tautology_tests.py`: verdict on the **maximum |ρ| over three
  proxies** (volcano, road, nearest site); **PASS ≤ 0.30 · GREY_ZONE (0.30, 0.50] · FAIL > 0.50** (l. 62–66, 213–234);
  original verdict GREY_ZONE, driven by road distance |ρ| = 0.307. Test 3 (road-distance quartiles, 50/50 hold-out
  inside each quartile, background from the same design → own-background scoring) stays unused (l. 354–420).
- **§2.1 counts re-derived today** from `data/processed/east_java_sites.geojson` (383 rows in bounds → 378 after the
  covariate filter; 340 distinct coordinates in bounds → 335 in the 378): R1 85 (of 87), R2 124 (of 125), R3 8 (of 10),
  R4 0 → 161 = 155 OSM + 6 Wikidata. The counts are **exclusive** (R1 first); do not add them twice. Sources in bounds:
  OSM 280, Wikipedia id 71, Wikipedia en 16, Wikidata 16. 124 modern monuments + 13 misplaced are the C081 audit's
  composition counts, not the R1–R4 drop counts.
- **§2.2 parameters verified** in `E217/01_maxent_benchmark.py`: XGBoost 300/4/0.05 with balanced class weight; RF
  200/6/balanced; MaxEnt elapid (linear, hinge, product; β = 1.5; cloglog); 5-fold deterministic spatial blocks at
  0.45°; background 1:5 (n_background_target 1,890 for 378 presences). E014 AUC 0.755 confirmed.
- **Length:** §2.1 321 · §2.2 152 · §2.3 244 (§2.4 ± 120 · §2.5 58 · §2.6 ± 300); Methods 1,420 in both v0.3 and v0.4.
  F12 (+110) and C094 (+25) must be covered by cuts listed in the sheet.
- **Status:** waiting for the PI's first part of session 5. Nothing inserted; nothing committed.

### Session 5 — F12 inserted (2026-11-12) — **the four synthetic worlds, §2.6**

**The PI's text (second version, after F35–F37):**
> Untuk membandingkan desain latar ketika intensitas permukiman sebenarnya diketahui, kami menyimulasikan empat dunia
> pada kisi nyata Jawa Timur dengan pipeline yang sama. Setiap dunia terdiri atas 10 realisasi dengan sekitar 300–500
> presensi teramati. Dunia A menggunakan intensitas penuh dari elevasi, lereng, sungai, dan TWI, tetapi menerapkan bias
> survei yang mengikuti jalan. Dunia B menambahkan raster clay nyata yang ditahan dari fitur model, sehingga model
> mengalami salah spesifikasi. Dunia C menambahkan usaha survei yang tidak merata antarkawasan, dengan bobot 1,0; 0,4;
> 0,15; dan 0,05. Dunia D menggabungkan usaha survei regional dengan intensitas sebenarnya yang seimbang antarkawasan,
> sehingga menjadi rezim paling menguntungkan bagi desain kuota. Bias jalan tetap ada di keempat dunia. Desain random,
> TGB, dan hybrid diuji pada `hard_frac` 0,0; 0,3; 0,7; dan 1,0. Kebenaran dinilai dengan AUC terhadap sampel presensi
> independen yang tidak bias pada domain ketersediaan seragam yang tetap. Pemulihan peta dinilai menggunakan Jaccard
> pada desil teratas dan Spearman terhadap permukaan intensitas sebenarnya.

**Inserted** as ten sentences in §2.6 under the test label `\paragraph{E222 -- synthetic ground truth.}`, immediately
after the existing one-line description (that line is **kept for now**; see F38).

**Changes beyond translation (all agreed beforehand):**
1. **F35** — "domain ketersediaan seragam yang tetap" → "on a fixed, uniform availability domain" (his revision; it is
   how `auc_true` is computed in E222 and it follows line 01 rule 2: declare the estimand).
2. **F36** — "kami" → "we" (the manuscript is co-authored and uses "we" throughout).
3. **F37** — "rezim" → "regime", the term his own §3.11 text uses.
4. Format: `hard_frac` set as `\texttt{hard\_frac}`, as in §2.3; "desil teratas" → "top decile", as in §3.8.

**Flag opened — F38 (queued for the compaction pass):** the pre-existing E222 one-liner ("Four synthetic worlds of
known intensity surface let us compare reported AUC (own-background) against true recovery (common-background) where
ground truth is not in question") now repeats the purpose sentence of the new paragraph, and its "true recovery
(common-background)" is not how `auc_true` is actually computed (independent unbiased sample + uniform availability).
Default: **delete the one-liner** in the compaction pass (−26 words), so the run-in label sits directly above the PI's
paragraph, as in the E224 entry.

**Length:** §2.6 426 → **604** (+178); Methods **1,420 → 1,598**. Compile clean, **38 pp.**, 0 undefined.

**The PI's compaction principles (2026-11-12), binding for the pass after session 6:**
- final target: Methods ≤ **1,420** words;
- cut **repetition** first, never procedural information needed to reproduce or understand an experiment;
- **no number, definition or methodological limit may be lost** to meet the cap;
- **every cut is recorded** in this log so the change can be traced.

Candidate cuts measured on 12 Nov (will be re-measured after session 6): §2.1 sources+cleaning 198 → ~120 (−78);
§2.4 closing "This is also what makes the approach specifically archaeological…" 40 → 0 (−40); §2.6 lattice paragraph
166 → ~120 (−46); §2.1 "This separation is not a technicality…" 46 → ~25 (−21); §2.3 "Two paragraphs of
interpretation." 4 → 0 (−4); F38 one-liner −26. Total ≈ −215 against +178 (F12) and an expected +25 (C094).

### Session 5 — C094 inserted (2026-11-12) — **Test 1's three verdict levels, §2.4**

**The PI's text:**
> Test 1 memeriksa korelasi Spearman antara kesesuaian prediksi dan tiga proksi: jarak ke pusat gunung api, jalan,
> dan situs terdekat. Vonis ditentukan oleh nilai maksimum |ρ|: PASS jika ≤0,30, GREY_ZONE jika >0,30 hingga 0,50,
> dan FAIL jika >0,50. Pada pipeline asli, vonisnya GREY_ZONE karena korelasi dengan jarak ke jalan mencapai 0,307.
> Korelasi jarak ke gunung api (−0,243 dengan tujuh pusat dan −0,281 dengan 13 pusat) tetap di bawah 0,30. Test 3
> membagi wilayah menurut kuartil jarak-jalan dan melakukan uji tahan 50/50 pada setiap kuartil, tetapi hasilnya tidak
> digunakan karena prediksi dinilai pada background desainnya sendiri.

**Inserted** in §2.4 in place of the two sentences Claude had edited in session 5 (C1/C2). His explanatory closing
("Rumusan ini mengikuti aturan tiga tingkat dari kode…") was read as a note to Claude and **not** inserted.

**Checked against the code before insertion:** verdict on the maximum |ρ| over `volcano_dist_km`, `road_dist_m`,
`nearest_site_dist_m` (`enhanced_tautology_tests.py` l. 191–234); PASS ≤ 0.30, GREY_ZONE above 0.30 and not above 0.50,
FAIL above 0.50 (l. 62–66); the original verdict GREY_ZONE, road distance 0.307; volcano correlations −0.243 (7 centres)
and −0.281 (13 centres), both below 0.30; Test 3 = road-distance quartiles with a 50/50 hold-out inside each quartile
(l. 354–420).

**Changes beyond translation (agreed before insertion):** F40 — "The original pipeline's Test 1…" (the two tests belong
to the pipeline, not to this paper); F39 — "(Section 3.9)" after the volcano correlations, because §3.9 is where the
7 → 13 inventory correction is explained.

**Length:** the replaced text was 93 words; the new text is 133. Methods 1,598 → **1,638**. Compile clean, 38 pp.

### Session 5 — §2.1 inserted (2026-11-12) — data, cleaning, role separation

**The PI's text (three paragraphs):** records and composition (378; audit's 124 monuments + 13 misplaced; the raw
count is not a count of valid sites) → cleaning in order (R1 85 geocoded, R2 124 modern monuments, R3 8 other objects,
R4 no further rows → 217 removed, 161 = 155 OSM + 6 Wikidata; the ladder stays on 378, contrasts on both sets) → one
role per variable (terrain block + river from E008; clay/silt tested in E009 and dropped; road_dist = accessibility,
volcano_dist = taphonomic diagnostics; neither a feature; the ambiguity argument).

**Verified by full reproduction (first time end-to-end, 2026-11-12):** the geojson clipped to 111–115°E gives 383 rows;
sampling the E217 rasters at those points leaves **378** with valid covariates; R1 **85**, R2-exclusive **124**,
R3-exclusive **8**, R4 **0**; remainder **161** (155 OSM + 6 Wikidata). So the audit-composition counts (124, 13) and
the sequential drop counts (85, 124, 8) are both correct and must not be added together — the PI's paragraph keeps them
apart. The old "335 distinct coordinates" is **not restored**: recomputation gives 338 at metre rounding and 340 at five
decimals, so the figure depends on precision.

**Decisions:** F42 yes — the PI's own sentence "Table 2 summarises the role assigned to each variable." F41 no (the 259
E009 subset stays out). F44 no (the examples of misplaced records stay out).

**F43 — a Claude paragraph deleted rather than left in place:** with the subsection body replaced, the trailing
paragraph "This separation is not a technicality. It is central to this paper: the variable that defines the target-group
background (accessibility, through road_dist) is kept out of the features, so accessibility can affect the reported
numbers only through the background design, which is what Section 3 examines." could not stay (Claude's text under the
editor's rule) and its ambiguity argument is now in the PI's own closing sentences. If the PI wants the **mechanism**
clause ("accessibility can reach the reported numbers only through the background design") back in §2.1, he writes
~15 words; for now the point is carried by §2.4 and §1.3.

**Length:** §2.1 321 → **287** (−34). Methods **1,638 → 1,604**. Compile clean, 38 pp., 0 undefined references.

### Session 5 — §2.2 inserted (2026-11-12) — the modelling pipeline

**The PI's text** (one paragraph: 378/1,890 at 1:5; three algorithms with XGBoost primary and each model's settings;
the β sweep in §3.2; five deterministic spatial-block folds at 0.45°; AUC and TSS, TSS over kappa for prevalence;
the original pipeline's >0.75 target; the closing point that both metrics are computed against whatever background is
supplied, so a value cannot be read without naming the evaluation background).

**Checked before insertion (E217 `01_maxent_benchmark.py`, `e217_outcome.json`):** 378 presences, `n_background_target`
1,890, ratio 5; XGBoost 300/4/0.05 with `scale_pos_weight`; RandomForest 200/6/balanced; MaxEnt `elapid`/maxnet with
linear–hinge–product, β 1.5, cloglog; β swept 0.5–4.0 in §3.2 (`sec:artefact`); 5 deterministic spatial-block folds at
0.45°; AUC + TSS.

**Changes beyond translation (agreed):**
1. **F45** — the citations are back at their places, at zero word cost: `chen2016` (XGBoost), `breiman2001` (RandomForest),
   `phillips2006, elith2006` (MaxEnt), `allouche2006` (TSS over kappa), `lobo2008, guillera2015` (metrics vs background).
   Three of them (`chen2016`, `breiman2001`, `allouche2006`) appeared nowhere else, so without this the algorithms would
   be unattributed and the entries would leave the bibliography.
2. **F46** — "of the full set": the cleaned-set models train on 161 presences and 805 background points, so 378/1,890
   must be scoped to the full set.
3. "Random Forest" → **RandomForest**, as everywhere else in the manuscript.
4. The Claude topic sentence "We model binary presence/background suitability." was dropped with the old paragraph; the
   PI's opening sentence carries the model type through "presences and background points".

**Length:** §2.2 152 → **171** (+19). Methods **1,604 → 1,623**. Compile clean, 38 pp., 0 undefined, all citations resolve.

**Word space after session 5 (per subsection, v0.3 → v0.4):** §2.1 321 → 287 (−34) · §2.2 152 → 171 (+19) · §2.3 244 →
244 (not yet) · §2.4 219 → 259 (+40, C094) · §2.5 58 → 58 (the PI's own) · §2.6 426 → 604 (+178, F12) · **Methods 1,420
→ 1,623 (+203)**. Still to be written in session 6: §2.3 (244, of which the hard-negative paragraph ~72 is the PI's),
the two Claude sentences left in §2.4 (~103), the lattice paragraph and the seven experiment paragraphs in §2.6 (~426),
and a re-read of §2.5 (58).

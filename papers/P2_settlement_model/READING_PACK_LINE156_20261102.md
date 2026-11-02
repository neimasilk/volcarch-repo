# Reading pack: the three papers cited at line 156 of P2 v0.3 (for the PI, 2 Nov 2026)

**Why:** JCAA's AI policy lists "summarising papers not actually read by the author" as an unacceptable use. The
sentence below summarises three papers, and so far only Claude has read them (`REFERENCE_CONTENT_CHECK_20261020.md`,
ledger C089). The PI reads the passage of each paper that the sentence relies on (about 10 minutes) and answers
**"sudah baca, sesuai"** or names the mismatch.

**Sentence in the manuscript (v0.3 line 156):**
> Recent applied studies report fair to good discrimination when predicting site locations from environmental
> variables (AUC of about 0.7 to 0.9; Castiello & Tonini 2021; Wang, Shi & Oguchi 2023), and Comer et al. (2023)
> report that the way samples are drawn for training and testing can itself change the measured performance.

*Terjemahan:* Studi terapan terkini melaporkan daya pembeda yang cukup sampai baik saat memprediksi lokasi situs
dari variabel lingkungan (AUC sekitar 0,7 sampai 0,9; Castiello & Tonini 2021; Wang, Shi & Oguchi 2023), dan Comer
dkk. (2023) melaporkan bahwa cara sampel diambil untuk pelatihan dan pengujian dapat dengan sendirinya mengubah
kinerja yang terukur.

Passages fetched again on 2 Nov 2026 from the sources named. The quotes are verbatim.

---

## 1. Castiello & Tonini 2021 — JCAA 4(1): 110–125 (open access)

Link: https://journal.caa-international.org/articles/10.5334/jcaa.71 — Section 3 "Results and Discussion": the
paragraph on variable importance and validation (Figures 5–6), and the paragraph that begins "Dealing with
archeological remains".

> "Concerning the model validation and the estimation of its predictive performance (Figure 6), the AUC using all the
> variables is equal to 0.71 (blue line), while for the second model (i.e. considering only the six most important
> variables) it is slightly higher and equal to 0.72 (red line), which is considered as an acceptable discrimination
> according to the criteria of Hosmer and Lemeshow (2000)."

> "The final dataset of Roman settlements in the Canton of Zurich examined in the present study includes only 227
> cases over an area of 1729 km². The scarcity of observations can explain the model's fair performance, as evaluated
> with an AUC of 0.72."

*Terjemahan:* AUC dengan semua variabel 0,71; dengan enam variabel terpenting 0,72 — "daya pembeda yang dapat
diterima" menurut kriteria Hosmer & Lemeshow. Datanya hanya 227 situs Romawi di Kanton Zurich (1.729 km²);
kelangkaan data itu bisa menjelaskan kinerja yang "cukup" (*fair*).

Predictors (same paper, Section 1): "topographic indices derived from the digital elevation model (DEM); …
characteristics related to the soil …; strategic and water-related criteria" — i.e. environmental variables.

**What the sentence takes from it:** AUC ≈ 0.7, "fair", environmental predictors.

## 2. Wang, Shi & Oguchi 2023 — ISPRS Int. J. Geo-Inf. 12(6): 238 (open access)

Link: https://doi.org/10.3390/ijgi12060238 — Abstract, and Section 4.5 "Model Evaluation" with Table 5 (the means below were re-computed from the four folds printed there and match).

Abstract: "This study aims to investigate the impact of topographic and hydrological factors on archaeological sites
in the Japanese archipelago and Shaanxi Province, China … The models' outcomes were cross-checked using the
four-fold cross-validation method, and the models' performances were compared using the area under the receiver
operating characteristic curve (AUC) and Kvamme's Gain."

Section 4.5: "The AUC values for the test data indicate that the AM_FR model has the highest accuracy in both study
areas (see mean AUC values in Table 5)."

Table 5, mean **test** AUC over the four folds:

| | AM_FR | MaxEnt | FR |
|---|---|---|---|
| Japan | 0.901 | 0.872 | 0.856 |
| Shaanxi, China | 0.816 | 0.785 | 0.791 |

*Terjemahan:* Faktor topografi dan hidrologi; validasi silang empat lipatan; AUC uji rata-rata 0,785 sampai 0,901
tergantung model dan wilayah.

**What the sentence takes from it:** AUC ≈ 0.8–0.9 from topographic and hydrological variables. (Together with
Castiello & Tonini: "about 0.7 to 0.9".)

## 3. Comer et al. 2023 — J. Archaeol. Sci.: Reports 48: 103824 (paywalled)

Link: https://doi.org/10.1016/j.jasrep.2022.103824 — the Highlights and the Abstract are free on that page; the full
text is behind the paywall. **The sentence uses only what the abstract says**, so the abstract is the part to read.

Highlights: "Spatial autocorrelation influences measured performances of archaeological predictive models (APMs)" ·
"Sampling method choice can over- or understate model performance on intended task."

Abstract (part): "… we compare models that are identical except for the sampling methods that draw the data used to
train and test them. We find that models trained and tested with a random cell sampling strategy perform better than
do models that implement a kind of leave-one-out cross-validation (LOOCV) that focuses on discrete sites and non-site
areas. We argue that this difference in measured predictive performance is due to spatial autocorrelation … Different
sampling methods can also yield calculations of predictive ability that over- or understate a model's performance at
the task for which it was designed."

*Terjemahan:* Model yang identik kecuali cara pengambilan sampel latih/uji: sampel sel acak memberi kinerja lebih
tinggi daripada validasi silang tinggalkan-satu-situs; selisihnya karena autokorelasi spasial; cara sampling bisa
melebih-lebihkan atau mengecilkan kinerja terukur.

**What the sentence takes from it:** the way training and test samples are drawn changes the measured performance.

---

## The PI's answer

- [ ] Read 1 · [ ] Read 2 · [ ] Read 3 — and the sentence matches what they say: **yes / no (which part)**
- Date:

When answered "yes", ledger C089's item "the three summarised papers" closes, and `SIG_signoff.md` (G16 / AI policy)
records the date.

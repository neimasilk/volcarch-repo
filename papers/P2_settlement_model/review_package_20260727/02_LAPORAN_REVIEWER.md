# Dokumen 2 — Keputusan Editor + Laporan Reviewer JCAA #280

**Diterima:** 2026-07-23, 18:13
**Dari:** Dr Philip Verhagen <notification@mail.journal.caa-international.org>
**Naskah:** "Tautology-Free Settlement Suitability Modeling in East Java Under Survey and Taphonomic Bias"
**Keputusan:** **REVISIONS REQUESTED** — "These revisions may then undergo further peer review prior to acceptance."
**Tenggat:** 4 minggu → **2026-08-20**
**Cara unggah:** login akun jurnal → My Queue → View → Upload File (bagian Revisions); balasan ke reviewer
diunggah dengan komponen artikel **"Response to Reviewers"**.

Teks reviewer di bawah **verbatim, tidak disunting, tidak diterjemahkan** — supaya bisa diperiksa sendiri
tanpa lewat tafsiran saya. Anotasi saya diberi tanda `[catatan]` dan selalu terpisah dari teks aslinya.

---

## Ringkasan skor

| Aspek | Reviewer 1 | Reviewer 2 |
|---|---|---|
| **Rekomendasi** | **Resubmit for Review** | **Resubmit Elsewhere** |
| Relevance for the journal | Good | Fair |
| Originality | Fair | **Excellent** |
| Framing of the research question | Good | **Poor** |
| Research design and methods | Fair | Fair |
| Discussion and conclusions | Good | Good |
| Clarity and structure | Fair | Fair |
| Bibliography | Good | Excellent |
| Figures/tables | Good | Good |

`[catatan]` Editor meng-override rekomendasi R2. Karena revisi "may then undergo further peer review",
asumsi kerja: **R2 akan melihat revisinya.** R2 adalah gerbangnya, bukan R1.

---

## REVIEWER 1 — Resubmit for Review

### 2. Originality — *Fair*

> The manuscript is a strong example of method development for archaeological predictive modeling. In
> particular, the focus on survey bias and pseudo-absence design, combined with a formalised attempt to
> avoid "tautological" predictors (e.g., volcanic proximity), represents a thoughtful and potentially
> valuable contribution.
>
> However, the empirical finding is not entirely novel and is well established in adjacent fields such as
> ecological niche modeling. This has gained significant attention in archaeology too, and there are
> examples to relate to, something that is missing in the current version of the manuscript. The claim of
> "tautology-free" modeling is the most distinctive conceptual contribution, but this claim currently
> appears stronger than the evidence presented and requires clearer definition and more rigorous support.

`[catatan]` Kalimat "not entirely novel... well established in adjacent fields" adalah **risiko terbesar**
bagi arah revisi kita sekarang — lihat dokumen 3 §6.

### 3. Framing of the Research question — *Good*

> The research question about the possibility and limitations of settlement suitability modelling under
> survey and taphonomic bias is clearly stated and relevant. The manuscript situates this question within
> an appropriate interdisciplinary context, drawing on literature from machine learning, archaeology, and
> volcanology. The motivation is well articulated, particularly in highlighting the dual challenges of
> preservation bias and uneven survey effort. The framing around avoiding tautological predictors is also
> conceptually interesting.
>
> However, the current structure occasionally obscures the central research question with methodological
> complexity (e.g., multiple experimental iterations and validation layers). A more focused presentation
> would strengthen the overall argument. The claims regarding novelty should be moderated, and relation to
> other archaeological examples needs expanding.

### 4. Research design and methods — *Fair*

> The methodological framework is one of the strongest aspects of the paper, and there are several advanced
> best practices that are relevant and innovative in this context. However, as the main topic is around the
> precence-only modeling, the reason for not using, comparing to, or at least relating to the similar
> approach through the Maximum Entropy functions is problematic. Maxent is indirectly refereed to in
> pointing to (Phillips et al., 2006) for the experimental design. But why not use Maxent to evaluate the
> results? Or at the very minimum explain the reasoning for not using Maxent, and what this method does
> that Maxent does not.
>
> The archaeological side of the paper is under developed, and it would be relevant to describe the
> archaeological context more. It was for a long time unclear why "Accessibility proxies" with roads where
> included, and only at the end explained that this was for survey bias. This should be explained earlier
> and motivated more.

`[catatan]` **Permintaan MaxEnt inilah yang menghasilkan seluruh isi dokumen 3.** Kami menurutinya, dan
benchmark-nya menggugurkan klaim kita sendiri.

### 5. Discussion and conclusions — *Good*

> The conclusions reflect the results of the modeling experiments and are generally consistent with the
> analysis presented. The key takeaway—that pseudo-absence realism plays a dominant role in model
> transferability—is clearly stated. Some conclusions, particularly regarding "tautology-free" modeling,
> are presented more strongly than the supporting evidence allows. The archaeological side is also under
> developed and the study would benefit to be put in context of current and future heritage management
> efforts in East Java.

`[catatan]` "The key takeaway—that pseudo-absence realism plays a dominant role in model
transferability—is clearly stated." Reviewer membaca klaim kita persis sebagaimana dimaksud. **Klaim itulah
yang sekarang gugur.**

### 6. Clarity and structure of the writing — *Fair*

> The manuscript is generally well organized but is often dense and difficult to follow due to specialized
> terminology and a highly technical presentation. Several terms are introduced without sufficient
> explanation, such as "tautology suite", "conditional pass", "null-model ceiling". The experimental
> structure, involving multiple iterations and validation strategies, is also complex and could be
> streamlined to improve readability, especially in order to be accessible to readers outside a very narrow
> envelope of practitioners. The text is partly very dense and introduce concepts and abbreviations that
> needs explaining, and much jargon could be replaced with more intuitive language.
>
> Especially the abstract goes too much in dept with AUC values for different iterations, and would benefit
> from a more general introduction to the study, design and results, keeping the technical descriptions for
> the main text.

### 7. Bibliography — *Good*

> The manuscript includes much relevant literature across archaeological predictive modeling, machine
> learning, and volcanic processes. There are several both foundational and recent contributions mentioned.
> However, the integration of references into the argument could be improved. At times, citations are
> presented in a general or descriptive manner rather than being used to support specific claims or
> contrasts.

### 8. Figures, tables and other additional materials — *Good*

> Figures and tables are good and clear. It might be relevant to also have definitions of TRI and TWI also
> in the text, and not only in the figures.

### 9. Additional feedback

> This is a well-conceived and methodologically careful study that addresses important challenges in
> archaeological predictive modeling. The manuscript's strengths lie in its rigorous approach to spatial
> validation, bias-aware modeling, and interdisciplinary framing. For the topic of this paper, and with the
> introduction of the "tautology-free" concept, I find it essential to relate to Maxent. As I understand it
> address very similar issues (?).
>
> The text should also be streamlined and clearly explain the experimental design and the different steps.
> The output and the results also needs to be discussed more in an archaeological context.

---

## REVIEWER 2 — Resubmit Elsewhere

### 2. Originality — *Excellent*

> It is original.

### 3. Framing of the Research question — *Poor*

> The manuscript presents an interesting and technically ambitious attempt to model archaeological
> settlement suitability while addressing volcanic burial and unequal survey intensity. Nevertheless, the
> principal research question needs to be defined more precisely. At present, it is unclear whether the
> study aims to model general settlement suitability, predict unknown archaeological sites, identify sites
> potentially buried by volcanic deposits, correct for survey bias or combine all these objectives. These
> are related but different questions.
>
> The manuscript could distinguish between:
> - Areas that were environmentally unsuitable for human settlement.
> - Areas that have not been adequately surveyed (they are ommited and key in the archaeological background).
> - Areas where archaeological remains may have been buried by volcanic or sedimentary processes.
> - Areas where archaeological evidence may have been destroyed or displaced.
>
> The archaeological specificity of the model should also be demonstrated. Currently, archaeological sites
> function mainly as spatial observations. The authors should clarify what makes the approach specifically
> archaeological and what new understanding it provides of settlement patterns in East Java.
>
> A potentially stronger framework would begin by modelling the environmental and spatial characteristics of
> known settlements. The resulting expected settlement pattern could then be compared with areas where sites
> are absent, allowing the authors to test whether volcanic burial or unequal survey intensity explains that
> absence.

`[catatan]` "Currently, archaeological sites function mainly as spatial observations" — keberatan ini
**bertambah parah** kalau kita membuat paper lebih metodologis. Itulah alasan E219 dikerjakan.

### 4. Research design and methods — *Fair*

> The general analytical strategy appears coherent, and the reported values do not show any obvious internal
> inconsistency. The use of XGBoost, Random Forest, spatial block cross-validation, multiple seeds and
> different pseudo-absence strategies is potentially appropriate. However, the construction of the successive
> experiments is difficult to follow. It should be a way to reproduce the model in a clear way. Also for me
> is not clear which variables are included and excluded.
>
> Also is relevant to adress how the terrain-only model is constructed, how river distance is added, how clay
> and silt are incorporated, and how the final hybrid model differs from the preceding experiments.
>
> The variables also need to be separated according to their analytical role. Elevation, slope and hydrology
> may represent settlement suitability; distance to roads may represent modern accessibility and survey
> effort; soils and volcanic deposits may represent preservation, burial or sedimentary processes. These
> processes should not be combined without explaining how each contributes to the final interpretation. Even
> how they are classiffy and studied in case they need so.
>
> The role of elevation and slope requires particular attention. Mountainous and rugged terrain may receive
> low suitability scores because it was less favourable for settlement, independently of volcanic activity.
> The authors should compare volcanic areas with environmentally similar non-volcanic mountainous areas.
>
> The exclusion of volcanic-proximity variables during training is understandable as a form of tautology
> control. However, this does not by itself demonstrate that areas of low predicted suitability near
> volcanoes contain buried archaeological sites. Similar results could be produced indirectly by elevation,
> slope or correlated soil variables.
>
> For me the model would be better if two stage design experiment was prepare:
> - First, model the environmental and spatial characteristics of known archaeological settlements. (maybe
>   with a catchment analysis could also be considered to assess the landscape surrounding.
> - Second, identify suitable areas where sites are unexpectedly absent and assess whether volcanic deposits,
>   sedimentation, accessibility or survey intensity explain that discrepancy.
>
> Finaly, I thin that the model should be tested against high-elevation or environmentally marginal areas
> that are not associated with volcanic activity. If these areas receive predictions similar to those of
> volcanic zones, the model may be detecting general settlement constraints rather than a specifically
> volcanic or taphonomic effect. A comparison between environmentally similar volcanic and non-volcanic
> areas would help assess this issue.

`[catatan]` Permintaan "compare volcanic areas with environmentally similar non-volcanic mountainous areas"
**sudah dikerjakan** — E219 Bagian C, lihat dokumen 3 §5.

### 5. Discussion and conclusions — *Good*

> The discussion should distinguish more clearly between settlement suitability and archaeological
> visibility. A low suitability score does not necessarily indicate volcanic burial, just as a high
> suitability score does not demonstrate that an archaeological site exists.

### 6. Clarity and structure of the writing — *Fair*

> The manuscript is highly technical and difficult to follow for a reader whose principal field is
> archaeology rather than computational science. The technical detail should be retained, but the conceptual
> purpose of each procedure needs to be explained more clearly.

### 7. Bibliography — *Excellent*

> The bibliography appears appropriate, relevant and sufficiently up to date for the subject addressed.

### 8. Figures, tables and other additional materials — *Good*

> The figures require substantial improvement because they do not currently explain the modelling process or
> its archaeological implications with sufficient clarity.
>
> Figure 1 is too simple and does not adequately show how archaeological, environmental and computational
> data are integrated. Figure 1 and 4 should be revised because some labels extend beyond their boxes.
>
> Figure 5 should explain more clearly how variable importance is calculated and how it should be
> interpreted. The figure appears to show that elevation and slope are particularly influential, but it does
> not explain whether they represent settlement preference, terrain accessibility or volcanic landscape
> structure.

`[catatan]` Nomor gambar sudah dicocokkan ke `submission_jcaa_v0.1.aux`: **Gambar 1** = kerangka
interdisipliner, **Gambar 4** = progresi AUC/TSS, **Gambar 5** = feature importance E007–E013.

### 9. Additional feedback

> The article contains a potentially valuable methodological contribution, particularly in its attention to
> pseudo-absence realism and survey bias. However, the archaeological purpose must be made more explicit.
> The principal issue is whether the model identifies areas where archaeological sites may have been
> obscured by volcanic processes or simply areas where settlement was generally less likely. The current
> analysis does not distinguish these possibilities sufficiently.

---

## Triase 17 item reviewer

Triase lengkap per item (jenis respons, usaha, status setelah eksperimen baru) ada di
`../revision_ammo/JCAA_R1_RESPONSE_PLAN_20260727.md`.

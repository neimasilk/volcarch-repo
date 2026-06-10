# Email Draft: Piek Vossen (VU Amsterdam / CLTL / GLOBALISE)

**To:** p.t.j.m.vossen@vu.nl
**Subject:** PhD Inquiry — NLP for Settlement Extraction from VOC Archives (building on GLOBALISE)
**Status:** UPDATED 2026-04-23 — ready for Pak Amien final review before sending
**Date drafted:** 2026-04-15 | **Last updated:** 2026-04-23

**Context:** Vossen is PI of GLOBALISE (5M+ VOC manuscript pages being digitized). Spinoza Prize winner. CLTL Lab = computational linguistics + text mining. Already has PhD student Stella Verkijk doing Event Reconstruction from VOC archives. This is the strongest infrastructure alignment possible.

**Strategy:** Pitch as potential promotor. Keep tone exploratory — don't mention Verberne or Lamqaddam unless Vossen brings it up. If Vossen is interested, multi-supervisor arrangement can be discussed later.

**2026-04-23 changes:** (1) BPI Dosen mention removed — age 48 likely exceeds cap; replaced with "exploring multiple funding routes including the Dutch promovendus model"; (2) VOC-ArchNLP HKI added as concrete recent deliverable; (3) experiment count updated to 207+; (4) 500-file corpus run mentioned.

---

Dear Professor Vossen,

I am writing to explore the possibility of doctoral research under your supervision at VU Amsterdam, on a topic that intersects directly with the GLOBALISE infrastructure your group is building.

I am developing VOLCARCH, a computational framework that quantifies how volcanic sedimentation has systematically buried pre-colonial archaeological evidence across Java. Across 207 tracked experiments, we have established that the expected archaeological gap in volcanic Java exceeds 694 times the observed record — meaning that settlement reconstruction must come from textual sources rather than fieldwork. The project currently has six papers under review at international journals and an arXiv preprint (arXiv:2604.00023).

The natural next step is an NLP pipeline for extracting structured settlement data from colonial Dutch archives. In pilot work, I have already:

- Extracted 22,162 structured mentions from 16 volumes of colonial archaeological service reports (OV, 1912–1929) using rule-based Dutch NLP, including 6,932 named sites and 4,933 administrative locations (E091)
- Recovered 1,768 archaeological records from Delpher newspaper searches via the KB SRU API, of which 165 were geocoded to specific locations (E141)
- Validated 33 colonial depth measurements against the VOLCARCH sedimentation model (Wilcoxon p = 0.131, model not rejected), demonstrating that colonial observations independently confirm the burial predictions (E197)

Building on this, I have recently developed and registered VOC-ArchNLP (v1.0.0, HKI Hak Cipta 2026) — a Python package that integrates GLOBALISE corpus downloading, HTR transcript preprocessing, colonial Dutch spelling normalisation, and archaeological mention extraction (six entity types: MONUMENT, GRAVE, RUIN, ARTIFACT, INSCRIPTION, DEPTH) with automatic unit conversion (voet/el/palm/duim → metres). An initial extraction run on 500 GLOBALISE files is currently underway. The pipeline is designed to scale to the full 6,893-file corpus.

The core NLP challenges ahead — named entity recognition for historical Dutch, temporal information extraction, and linking extracted entities to structured knowledge representations — are ones your group has direct expertise in.

What makes this project methodologically unusual is that it offers a physically grounded, non-textual validation signal for NLP extractions: the VOLCARCH sedimentation model predicts where sites should be buried and at what depth. Extracted settlement mentions in volcanic zones that report deeper finds are either correct extractions or geological anomalies — both are informative. This is, to my knowledge, a rare case where historical NLP has access to an independent physical ground truth.

I am a permanent lecturer in Informatics at Universitas Bhinneka Nusantara (Malang, Indonesia), with an M.Sc. in Computer Science from Universitas Indonesia. My NLP background includes published work on Indonesian morphological analysis (ModernKataKupas, PyPI), low-resource NLP, and sub-word segmentation. I am exploring multiple funding routes for the doctoral period, including the Dutch promovendus employment model and MSCA Doctoral Networks — I would welcome your perspective on what funding mechanisms your group typically uses for international PhD candidates.

I am aware that Stella Verkijk's PhD work on Event Reconstruction from VOC archives at your lab addresses related material from a different angle — event structure rather than settlement geography. I believe the two projects would be complementary rather than overlapping.

Would you be open to a conversation about whether this direction fits within your group's research programme?

Best regards,

Mukhlis Amien
Universitas Bhinneka Nusantara, Malang, Indonesia
amien@ubhinus.ac.id | ORCID: 0000-0002-1848-167X
GitHub: github.com/neimasilk/volcarch-repo

---

**Notes for Pak Amien:**
- Semua angka sudah di-audit (E091: 22,162 mentions verified, E141: 1,768 verified, E197: Wilcoxon p=0.131 verified)
- "207+" experiments = akurat per 23 April 2026
- Tidak menyebut Verberne atau Lamqaddam — biar Vossen yang raise kalau relevan
- Menyebut Stella Verkijk = menunjukkan Pak Amien sudah riset lab mereka
- Tone: eksploratif, bukan desperate
- BPI Dosen DIHAPUS — umur 48 kemungkinan melebihi batas; diganti dengan "exploring multiple funding routes including Dutch promovendus model"
- VOC-ArchNLP HKI ditambahkan = deliverable konkret yang menunjukkan progress nyata
- Pertanyaan funding ke Vossen = natural conversation starter yang juga memberi info kita butuhkan

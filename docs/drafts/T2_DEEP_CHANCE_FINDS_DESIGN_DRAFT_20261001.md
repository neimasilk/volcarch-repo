# T2 — Register of chance finds at depth in volcanic Java: design draft (pre-registration candidate)

**Status:** DRAFT 2026-10-01 · becomes an experiment number (E226 or E227) only after PI approval of the
3-month direction (`docs/WORKSTATE.md` §4). Source: `docs/research_notes/OBJECTIVE_ANSWER_20261001.md` §5.
**Line:** 05_archival_nlp (primary; it is the PI's NLP strength) + 02_taphonomy. **Extends:** E225
(gray-literature mining, pre-registered 2026-08-13). **Cost:** ≤$10 API (DeepSeek key now in `.env`),
2–4 sessions. **Doubles as:** the overdue D6 standing falsification (dated 2026-09-15, missed).

---

> **Revised 2026-10-01 after the WF2 review (methodology + geomorphology + protohistory): the original kill
> rule below was tilted against H1 and is superseded.** Changes: (1) run **after T4-desk**, so zones are defined
> by the dated depth of the ±400 CE surface, not by an absolute >3 m threshold, which is shallower than the
> 9th-c. surface where H1 is strongest; (2) start from **colonial sources** (*Oudheidkundig Verslag*, Rapporten
> OC, Notulen/Tijdschrift Bataviaasch Genootschap 1900–1949), then gray literature (E225) and news; (3) unit =
> find event with dig type, find depth, dig bottom and context (in-situ surface vs dug feature vs reworked
> material); (4) **tiered dating** (A radiometric/inscription; B diagnostic by an approved list; C/D not counted),
> coded blind to depth and zone; (5) a **positive control for the reporting channel** (regions with known
> shallow prehistoric sites); without it a null is recorded as **NOT INFORMATIVE**; (6) T2 can separate H1b
> (recognisable material) from H6 only where digging reaches the ±400 CE surface, and **never H3 from H6**.

## 1. Why this test

Deep digging in Java happens every day without archaeologists: sand quarries (*galian C*), wells,
foundations, irrigation. **Liangan, Kedulan, Kimpulan, Sambisari and the Wonoboyo hoard were all found
that way.** These accidental "deep samples" are not aimed at monuments. That partly sidesteps the survey
bias (H2) that confounds every site-count analysis in the repo (E069, E109). If pre-400 occupation lies
buried in the volcanic interior (H1), some of this digging should hit it.

## 2. Questions (fixed before extraction)

- **Q1.** How many reported finds at depth (>2 m; and separately >3 m) come from volcanic-zone Java, and of
  what type and period?
- **Q2.** What fraction of the professionally attributed deep finds is **pre-400 CE**?
- **Q3.** Among professionally dated deep finds, does depth increase with age, and at what rate per setting
  (vent-proximal fan / floodplain / upland)? This gives independent rate data for C024.

## 3. Sources (public; local analysis only for restricted ones)

Indonesian online news 2000–2026 (national and regional outlets); BPCB/BPK and Balai Arkeologi reports
(repositori.kemdikbud.go.id; *Berita Penelitian Arkeologi*); colonial *Oudheidkundig Verslag* depths
already compiled (E083/E128/D1). Delpher may be analysed locally under its GDPR limits, but never
redistributed.

## 4. Extraction schema (LLM extraction, every field with the original quote)

`source, date, desa/kecamatan/kabupaten, lat, lon, geocode_precision, activity (sand_mining | well |
foundation | irrigation | other), depth_m_reported, find_type (structure | statue | pottery | metal | bone |
burial | charcoal | other), period_attributed, attributed_by (BPCB/archaeologist | journalist | resident),
dating_evidence (14C | typology | inscription | none), follow_up_excavation (y/n), quote, page_or_url`.
De-duplicate by place, date and find, since one find is often reported by several outlets.

## 5. Predictions and kill criteria (written before any mining)

| Outcome among **professionally attributed** finds at >3 m in volcanic zones | Reading |
|---|---|
| ≥3 attributed pre-400 | **H1 alive.** These locations become T4 targets (dated sections). |
| 0 attributed pre-400, with n ≥ 50 professionally attributed deep finds | **H1-as-general downgraded to "local only"** (the D6 standing falsification). H6 strengthens. |
| n < 50 | Inconclusive; report the counts as they are and widen the corpus. |

**Reporting-propensity control:** run the same extraction for non-volcanic coastal West/North Java (where
Buni-type pre-400 material is known). If pre-400 material is rarely *reported* even there, a volcanic-zone
null is weak evidence and must be stated as such. A null counts only if the control shows that such
material does get reported.

## 6. Known biases (state in any write-up)

News favours candi, statues and gold over sherds and burials. Journalists' period attributions are not
evidence (only `attributed_by = professional` enters Q2). Reported depths are approximate. Geocoding is
often district-level. All of these push toward **under-detecting** pre-400 finds, so a positive result is
stronger than a null.

## 7. Outputs

`results/t2_register.csv` (sourced rows) · `results/t2_scan_log.csv` (documents scanned, failures) · a short
data note; any pre-400 hits go straight into the T4 target list.

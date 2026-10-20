# NusaTemu — codebook v0.1 (draft; frozen as v1.0 at the end of the Phase 1 pilot)

One **record** = one source text about a find. One **event** = one find in the world, possibly reported by many records.
Every extracted value carries a **verbatim quote** from the source (≤ 30 words). A value with no quote stays empty.

## Record fields

| Field | Type | Rule |
|---|---|---|
| `record_id` | text | `NT-R-<yyyymmdd>-<n>` |
| `event_id` | text | assigned at the event-merge step (`NT-E-<n>`) |
| `source_type` | enum | `news` · `colonial_report` · `heritage_office` · `journal` · `other` |
| `source_outlet`, `source_url`, `source_date`, `language` | text | as published; date of publication, not of the find |
| `find_date` | date | as stated; with `find_date_precision` = day / month / year / unknown |
| `retelling` | bool | **true** if the text reports a find made earlier than its publication year and says or implies it is not new (example: a 2024 article retelling the 2018 Dasun, Lasem hoard) |
| `provinsi`, `kabupaten`, `kecamatan`, `desa` | text | as stated; codes from the Kemendagri list at geocoding |
| `location_precision` | enum | desa / kecamatan / kabupaten / province / unknown |
| `setting` | enum | see below |
| `object_class` | multi-enum | see below |
| `object_count`, `object_weight_kg` | number | as stated; "ribuan" → leave the number empty, set `count_text` |
| `period_claimed` | text | as stated (e.g. "Majapahit", "Dinasti Song", "zaman Belanda") |
| `period_claim_source` | enum | `heritage_office` (BPK/BPCB) · `archaeologist` · `museum` · `journalist_or_unattributed` · `finder` |
| `discovery_activity` | enum | see below (the core variable) |
| `finder_role` | enum | farmer · labourer/builder · miner (sand/gravel) · fisher · child/student · contractor/company · official · archaeologist (survey) · unknown. **Never a name.** |
| `depth_m` | number | only if stated; with `depth_text` |
| `response` | multi-enum | reported_to_authority · BPK/BPCB_survey · excavation · kept_by_finder · sold · museum · reburied/left_in_situ · unknown |
| `extraction_confidence` | enum | high / medium / low (the extractor's own rating; audited) |
| `quotes` | json | field → verbatim quote |

## Taxonomies

**`discovery_activity`** (one main value; `activity_secondary` optional):

| Code | Meaning |
|---|---|
| `house_construction` | digging foundations, septic tanks, wells for a house |
| `infrastructure` | roads, toll roads, railways/MRT/LRT, dams, bridges, drainage, airports |
| `sand_gravel_mining` | galian C, river sand, quarrying |
| `agriculture` | ploughing, planting, terracing, irrigation ditches |
| `natural_exposure` | river or coastal erosion, flood, landslide, drought (low water) |
| `fishing_dredging` | nets, dredging, diving (non-research) |
| `looting_illicit` | treasure hunting, metal detecting for sale |
| `research_survey` | planned survey or excavation (records kept to measure the share, not as chance finds) |
| `colonial_works` | (Phase 3) railway, plantation, irrigation, military works |
| `other`, `unknown` | — |

**`object_class`:** coins · ceramics/pottery · statue/arca · inscription · building/structure (bata, stone) ·
metal object (non-coin) · beads/glass · human remains · fossil/faunal · boat/wreck · other.

**`setting`:** field (sawah/ladang) · settlement/urban · river/riverbank · coast/beach · hill/slope · forest/plantation ·
sea · cave · unknown.

## Event merge (record → event)

Two records are the same event when **object class, kabupaten and find date** (within 1 year, or the same stated
date) **agree**, and either a named place below kabupaten agrees or one record cites the other. A disagreement on
count or depth does **not** split an event: keep both values and flag `conflict`. Retellings join the original event.
Their publication date never becomes the find date.

## Things the pilot must test (open questions)

- Can `discovery_activity` be coded reliably from news text? If agreement is < 0.80, merge categories.
- How often is depth stated at all?
- How often are period claims attributed to an authority (BPK) rather than to the journalist?

# T1 — Regional onset of local writing: design draft (pre-registration candidate)

**Status:** DRAFT 2026-10-01 · not yet an experiment number (becomes **E226** if the PI approves the 3-month
direction in `docs/WORKSTATE.md` §4). Source: `docs/research_notes/OBJECTIVE_ANSWER_20261001.md` §5.
**Line:** 04_language_text (primary) + 06_thesis. **Cost:** $0, 1–2 sessions, desk only.
> **Revised 2026-10-01 after the WF2 review:** T1 *documents* H0; it is **not a test** (the outcome is known,
> n≈10, predictors collinear), and DHARMA (already in the repo, E023) settles its key comparison: Tuk Mas
> (mid-7th c., volcanic interior) is no later than Kedukan Bukit (682, non-volcanic Sumatra), and Tarumanagara's
> corpus lies in volcanic Bogor. Add one discriminating variable: the **lag from first evidence of contact to the
> first local inscription** per region (Bali ≈900 yr at a coastal node; West Java Buni → Tarumanagara ≈400 yr).
> Output: a background table for an Indonesian-language synthesis, not a stand-alone international paper.
> It is no longer the first test to run — see `docs/research_notes/OBJECTIVE_ANSWER_20261001.md` §5/§7.

**Why this first (original reasoning):** it answers the PI's literal question ("why does history start ~400 CE?") with a
publishable, falsifiable comparison, and it is cheap. It will probably confirm the mainstream view (H0).
That is fine; the point is to close one question cleanly and in public.

---

## 1. Questions (fixed before data collection)

- **Q1.** Is ~400 CE an anomaly for Nusantara relative to Southeast Asia, or in step with the region?
- **Q2.** Within the region, does the onset of local writing track **volcanic burial exposure** of the
  setting (H1) or **position in the maritime trade network** (H0-trade)?
- **Q3 (survival check).** Where early inscriptions exist in volcanic settings, do they sit on
  burial-resistant spots (hilltops, river boulders) rather than on aggrading plains? A strong selection
  effect would show up here.

## 2. Units and variables

One row per polity or region with locally produced inscriptions before ~1000 CE: Champa (Vo Canh,
Dong Yen Chau), Funan, Chenla, Pyu, Dvaravati, Kedah/Bujang (peninsula), Kutai, Tarumanagara, Central
Java, East Java, South Sumatra (Srivijaya), Bali, the Philippines (Laguna Copperplate), plus explicit
nulls (Sulawesi, Lesser Sundas: none before 1000 CE).

| Variable | Definition | Source |
|---|---|---|
| `onset_min`, `onset_max` | earliest inscription date range: palaeographic window or absolute (Saka) date | Guy 2011; Manguin 2011; Vickery 2005; Griffiths; Zakharov 2012; DHARMA |
| `dating_basis` | palaeography / internal date / other | same |
| `volcanic_setting` | active volcano within 30 km of the earliest findspot (canonical inventory for Java; GVP elsewhere) | `data/processed/dashboard/volcanoes_java_full.csv`, GVP |
| `findspot_geomorph` | hilltop / river boulder / plain / coast | edition + maps |
| `trade_node` | 0 interior, 1 riverine port, 2 coastal node on a documented route (Hall 1985; Manguin 2011) | coded blind to `onset` where possible |

## 3. Predictions and kill criteria (written before coding)

- **H0-trade predicts:** onset ordered by `trade_node`; `volcanic_setting` adds nothing once `trade_node`
  is known; early volcanic-zone inscriptions on burial-resistant findspots.
- **H1 (burial as cause of a late start) predicts:** at equal `trade_node`, volcanic settings start later.
- **H1-as-onset-explanation is rejected if** at least one volcanic-setting polity at a given `trade_node`
  shows onset no later than the non-volcanic polities at the same `trade_node` (e.g. Tarumanagara near
  Salak/Gede against Kutai; Canggal on Gunung Wukir against Sumatra 683). The repo already suspects this
  outcome. Pre-registering it keeps the result honest either way.
- **No p-value theatre:** n≈12–15 with ±~100-year palaeographic windows. Report an ordered interval plot and
  the rank pattern; state that it is descriptive.

## 4. Outputs

`results/onset_table.csv` (every row sourced) · `results/onset_intervals.png` · a 2–3 page note: *"Why does
written history in Nusantara begin around 400 CE? A regional comparison"* → regional zero-APC venue (e.g.
*Berkala Arkeologi*, *Amerta*, *Wacana*). The note would also carry the "first inscriptions attest earlier
polities" point (Kundungga; the Tugu canal), which is the PI's question answered in public.

## 5. Known limits (state in the note)

Palaeographic dating carries roughly ±100 years. Some foreign-text identifications (Yediao, Iabadiou,
Yepoti) are contested and are excluded from `onset`. DHARMA includes Tugu (Tarumanagara, dated there to ca. 6th
c.) and Tuk Mas; the Kutai row comes from editions and secondary syntheses and needs a specialist's check
before publication (G10).

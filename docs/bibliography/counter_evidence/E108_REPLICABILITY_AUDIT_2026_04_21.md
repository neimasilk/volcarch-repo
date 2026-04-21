# E108 Replicability Audit — Session 19 (2026-04-21)

**Context:** ME#15 §7B recommended pre-registered replicability test on one high-stakes experiment. E108 demographic null model (3,220× gap) selected because it underpins the "invisible civilization" thesis across P1, P0, P17.

**Method:** Read README + script without consulting JSON results. Re-derive population scenarios and gap ratio from inputs. Compare.

---

## Replication

### Scenario A (minimal)

Script logic applies to Scenario A:
- Wet rice area: 15% × 0.3 = 4.5% of habitable
- Swidden area: 40% × 0.5 = 20% of habitable
- Others unchanged at table fractions
- All densities at `density_low`

Using `habitable_km2 = 60,000 + 35,000 + 14,000 + 5,000 = 114,000`:

| Mode | Area (km²) | Density | Pop |
|---|---:|---:|---:|
| Forest foraging | 11,400 | 0.05 | 570 |
| Coastal fishing | 5,700 | 1.0 | 5,700 |
| Swidden | 22,800 | 5.0 | 114,000 |
| Wet rice | 5,130 | 25.0 | 128,250 |
| Arboriculture | 34,200 | 10.0 | 342,000 |
| **Total** | | | **590,520** |

**README claims: 590,520. Match: EXACT.**

### Scenario B (moderate)

No area modifications; all densities at `density_best`:

| Mode | Area | Density | Pop |
|---|---:|---:|---:|
| Forest | 11,400 | 0.2 | 2,280 |
| Coastal | 5,700 | 2.5 | 14,250 |
| Swidden | 45,600 | 12.0 | 547,200 |
| Wet rice | 17,100 | 40.0 | 684,000 |
| Arboriculture | 34,200 | 20.0 | 684,000 |
| **Total** | | | **1,931,730** |

**README claims: 1,931,730. Match: EXACT.**

### Gap ratio 3,220×

Script: `pop_b / settlement_ratio_low / max(3, 1)` = `1,931,730 / 200 / 3` = `3,219.55` → `3,220×`

**README claims: >3,220×. Match: EXACT.**

---

## Findings

### Pass
1. **Core computation is fully replicable.** Running the arithmetic independently yields identical numbers to README.
2. **No parameter hunt.** The inputs (density tables, land fractions, comparanda) trace to named sources (Bellwood 2017, Kirch 2000, Bayliss-Smith 1980, Higham 2014). Whether these sources are correctly represented is a separate question (see caveats), but the downstream calculation is deterministic given the inputs.
3. **Comparanda are sourced:** Thailand Dvaravati 300-500K (Higham 2014), Vietnam Dong Son 500K-1M (Bellwood 2017), Philippines 200-500K (Junker 1999), PNG Highlands 1M (Golson 1977) — script has citations. **This resolves P0 Flag C for these specific numbers.**

### Documentation drift (minor)
1. **README table shows `density_low = 0.1` for forest foraging; script actually uses `0.05`.** Half the value. Not a computational error, but a table-code mismatch that confuses anyone reading only the README. **Fix: update README table to show 0.05.**
2. **Scenario A area modifications not shown in README table.** The script reduces wet rice area by ×0.3 and swidden by ×0.5 for Scenario A, but the README table presents fractions as constant across scenarios. **Fix: add Scenario A note to README explaining area modifications.**
3. **`density_low` value discrepancy documented:** README says `0.1`, script `0.05`. Doesn't affect the published 590,520 because the script uses 0.05. But creates audit confusion.

### Load-bearing claims traced
1. **3,220× gap:** = 1,931,730 ÷ 200 ÷ 3. Uses Scenario B population, MAX village size (200), MAX liberal site count (3). This is a *specific combination*, not the gap midpoint.
2. **P0 "1,000 to 7,000-fold" range:** = Scenario A ÷ 200 ÷ 3 (= 984, round to 1,000) and Scenario C ÷ 200 ÷ 3 (= 6,517, round to 7,000). **Derivable from same formula with scenario substitution.** But not shown explicitly in E108 output. Fix: add these rows to README gap section.
3. **P0 "500-fold under most conservative parameter combination":** Computationally: Scenario A (590K) ÷ village 200 ÷ liberal sites (~10) = 295; or ÷ village 300 ÷ 3 = 656. This claim is *approximately defensible* but needs explicit derivation. Fix: either drop to "1,000-fold conservative" (matches Scenario A / max village / max sites) or provide explicit MC sensitivity.

### Verdict

**E108 REPLICATES. No parameter hunt detected.** The downgrading in ME#15's concerns about "every claim at v0.1 reliability" is partially resolved for E108 specifically — its math is reproducible.

What remains open:
- **Input sources** (Bellwood ethnographic densities, Kirch analogues) have not been verified against their original publications in this audit. That would require library access to Bellwood 2017 and Kirch 2000 original texts.
- **Ethnographic analogues may be wrong** for 400 CE Java specifically. This is the Flag D concern (channels not truly independent; all use Bellwood/Reid literature). Not resolvable via replication.
- **"Settlement per 50-200 people"** rule of thumb is a Claude-specified ratio. Would benefit from domain-expert (Fiverr stats review or PhD supervisor) validation.

**Bottom line:** E108 math is solid. Downstream claims (3,220× gap, 1,000-7,000× range) derive cleanly from inputs. Documentation needs minor updates. No red flags for parameter hunt. This is a resolution of one ME#15 echo chamber concern, though not the meta-concern (which is about external validation, not internal replicability).

---

*E108 replicability audit produced 2026-04-21. Closes ME#15 §7B recommendation.*

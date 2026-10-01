# T7 — Detection calibration from villages named in inscriptions: design draft (pre-registration candidate)

**Status:** DRAFT 2026-10-01. It gets an experiment number (E226, or the next free one) only after the PI
approves the 3-month direction (`docs/WORKSTATE.md` §4).
**Source:** the desk scan in `docs/research_notes/DESK_TESTS_T0_T4_T7_20261001.md` (T7). That scan was a
proposal by the T7 desk agent, revised here by the orchestrator. Neither the PI nor a domain expert has
reviewed it yet.
**Line:** 05_archival_nlp (primary: NLP over DHARMA, the PI's strength), with 06_thesis.
**Cost:** $0, about 2–4 sessions.
**Why now:** T0 and T4-desk are blocked on data held by other people. T7 is the one decisive desk test that
can start immediately.

---

## 1. Question

The question is whether "no recorded pre-400 settlement in interior volcanic Java" is evidence of few
people (H6), or whether it is uninformative.

The test uses a population we know existed: the 8th–10th c. villages (*vanua*, *wanwa*, *thāni*) named in
*sīma* charters. It measures what share of them has a recorded settlement deposit. Call that share the
detection rate *d*. If even these villages are almost never found, a zero for pre-400 settlements says little.

**What T7 can and cannot show.** *d* measures **combined** detectability in that landscape:
- burial (H1a);
- survey effort (H2);
- light, organic architecture (H3).

It cannot tell these apart. A low *d* makes the H6 inference from absence unsafe. It does **not** prove H3 or
H1b. The 8th–10th c. window sits in the same burial zone that is being tested, and this must be stated in any
write-up.

## 2. What is already in hand (desk scan, 2026-10-01)

- **Denominator.** 111 dated 8th–10th c. inscriptions in the local DHARMA corpus, leaving out the 50 Borobudur
  captions. 69 of them name villages, with 440 village-noun tokens and roughly 150–200 village-name strings.
- **Where the names are.** Mostly Kedu (Magelang, Temanggung, Wonosobo, Purworejo) and the Prambanan–Mataram
  plain (Sleman, Bantul, Klaten). The Brantas has about 5 names, too few for a Brantas arm without the
  upstream corpus or the literature on Sindok-era charters.
- **Numerator: missing.** The repo holds no dated register of 8th–10th c. settlement sites.
  - E129 is **not** such a register: its frame is temple lists, and its "settlement 1.3%" is 5 generic rows
    (C036).
  - E153, D2 and E070 do not provide one either.

## 3. Frozen before the numerator is touched

1. **Village frame V_r.** A hand-curated list of village names from DHARMA edition text only (not translation
   or apparatus), date window 701–1000 CE, regions Kedu and Prambanan–Mataram.
   - Two coders. Extraction rules as in the desk scan: normalise diacritics; village nouns are
     vanua/vanva/thāni; *karaman* is counted separately; *desa* and *kuvu* are excluded.
   - Inscriptions dated only through the IDENK spreadsheet or a manual alias are flagged and get a
     sensitivity run without them.
2. **Identification tier** for each name:
   - **I1:** modern location published by an epigrapher (Boechari, Damais, Sarkar, Wisseman Christie, DHARMA
     metadata);
   - **I2:** a unique modern *desa* of the same name within the same *kabupaten*;
   - **I3:** ambiguous or not located.

   *d* is estimated on I1+I2. I3 enters only the bounds:
   S/|V| ≤ *d* ≤ (S + |V∖G|)/|V|.
3. **"Recorded settlement" defined.** An open-air habitation deposit (floors, post-holes, hearths, domestic
   assemblage) dated to the 8th–10th c. by radiocarbon or diagnostic ceramics. **Excluded:** temples,
   inscriptions, tombs, and find-spots of single objects or hoards.
4. **Register frozen before matching**, assembled blind to the village list.
   - Sources: Balai Arkeologi DIY/Jateng reports, *Berkala Arkeologi*, BPK Wilayah X lists, SRN Cagar
     Budaya.
   - **Completeness check:** known excavated 8th–10th c. settlements (Liyangan, plus any a domain expert
     adds) must appear in the register. If they do not, the register is incomplete and the run is
     **NOT INFORMATIVE**.
5. **Matching.** Radius ρ = 1 km from the modern *desa* centroid, with sensitivity runs at 0.5 and 2 km.
   - **Chance-match null:** the same number of random land points per *kabupaten*, giving *d*₀.
   - Report *d* − *d*₀. Prambanan is dense with recorded sites, so raw *d* is inflated there.
6. **Estimates.** *d̂* = S/|G| with a Jeffreys 95% interval. With zero pre-400 settlements recorded, the
   number of villages consistent at 95% is N₉₅ from the beta-binomial predictive (roughly 3/*d*).
7. **Decision rule.** N_ref and m are fixed by the PI with a domain expert **before** matching.
   - N_ref = |V_r|, read as "a pre-400 population as large as the 8th–10th c. one".
   - *d*_pre = m·*d*, with m ∈ {1, 0.5, 0.25}, because older sites are more buried and eroded.

   | Condition | Reading |
   |---|---|
   | N₉₅(m=0.25) ≥ N_ref | Zero pre-400 settlements is **uninformative**. "Nothing found" must not be read as "no people". |
   | N₉₅(m=1) < N_ref | A zero is inconsistent with a population the size of the 8th–10th c. one. **Supports H6**, conditional on comparable survey effort. |
   | otherwise | Ambiguous; report the numbers. |

8. **Falsifier for the H3 in-Java control.** If (S/|G|) − *d*₀ ≥ 0.25 with |G| ≥ 30 in Kedu/Prambanan, the
   8th–10th c. record is not village-blind there. The argument that "villages are invisible even when we know
   they existed" is then dead, and the write-up says so.

## 4. Threats to state in the write-up

1. *Sīma* villages were taxable, politically salient and close to the royal core. Inscriptions are found near
   temples, where survey effort is highest. Both push *d* **up** for ordinary villages, so the N₉₅ bound is
   anti-conservative.
2. Older sites are more buried, eroded and rebuilt, so *d*_pre ≤ *d*. That is the reason for m.
3. Failing to match a toponym is an identification failure, not a non-detection. This is why the I-tiers and
   bounds exist.
4. Villages may have moved since the 10th c. This is why the radius sensitivity runs exist.
5. The local corpus holds 268 encoded editions, while the IDENK metadata lists about 1,271 records.
6. Heritage registers are built around temples, so S may be near zero **because of how the registry was
   built**. Literature-based counts are therefore required. Without them the test measures registry bias.

## 5. Outputs

- `results/t7_village_frame.csv`: name, inscription, date, region, tier, citation for the identification.
- `results/t7_settlement_register.csv`: frozen before matching.
- `results/t7_match.csv`, and a short note giving *d*, *d*₀, the bounds, N₉₅ for each m, and the decision
  as pre-registered.

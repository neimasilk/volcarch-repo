# E234 — Design (pre-registration): E216's assumptions checked against published values (SIG G10 route S)

**Written:** 2026-10-19, before any source was consulted for comparison and before any agent ran.
**Frozen when:** this folder is committed and pushed. Nothing in §3–§7 changes after that; deviations go to
the README and `docs/JOURNAL.md`.
**Line:** 03_paleoenv · **Checks:** E216 · **Gates:** SIG G10 route S (steps 1 and 2), G2 domain sanity.

---

## 0. Why, and why not sooner

E216's `SUBMISSION_CHECKLIST.md` puts a palynologist co-author first, and that step "BLOCKS everything". Line 03
rule 2 calls the co-author a hard SIG requirement (G2/G10), because "this line makes claims about pollen detection
physics that no one here can peer-check". No co-author has been asked since July.

The PI's rule of 2026-10-19 is "AI penuh = fitur … kalau perlu ahli, pivot". SIG G10 now has a route S. E233 showed
the pattern: measure the work against expert judgement that is already published.

The co-author was meant to judge "whether the RPP ranges, RSAP radii, and detection-threshold assumptions used here
are defensible palynological practice" (checklist §1). Each of those has published values. E234 checks every
assumption E216 rests on against them. It then re-runs E216's own code with the published values.

**Timing.**
- G13 is full: P2 and P8 are in review, and P23 holds the one drafting slot. So the E216 manuscript cannot be drafted
  or sent before one of those is decided. E234 prepares for it; it does not jump the queue.
- K9 stop list (no new computational experiment for the burial thesis without partner data). E234 is an audit of
  existing work, not a new test of the thesis. Its only computation is E216's own code with corrected inputs.
  *(PI to confirm this reading.)*

**Line rule 2 is not changed by this design.** §9 proposes text; the PI decides after the results.

## 1. Question

Do the assumptions behind E216's OUTCOME-3 hold against what palynologists (and, for the land-use coupling,
ethnographers and agronomists) have published? Do E216's headline claims survive when its inputs take the published
values?

## 2. What E216 claims, and what carries it

**Headlines** (`E216 results/PAPER_DRAFT_OUTLINE.md`, abstract):
- **H1.** Zero of seven cores can resolve heartland clearing. P(network detect) = 0.000 at all 27 grid points, "at
  any parameter setting we tested".
- **H2.** OUTCOME-3 rests on two independent reasons:
  - (a) the resolution gap;
  - (b) the calibration cannot be extracted (PREREG S2 NO-GO).
- **H3.** The decisive missing core:
  - a lowland lake or swamp within 20 km of the heartland points;
  - 50-year resolution or better, with about 20 AMS dates;
  - Oryza-type, Poaceae and Trema pollen plus microcharcoal;
  - USD 8,000–15,000.
- **H4.** At the conservative corner (floor population, uniform clearing), even a co-located core does not detect:
  12.6 pp is below 17.5 pp, and the corner fails in 85 % of the grid.
- **H5.** The dispersed mode (B) is invisible to pollen at every core; it is handed to E215.

**Observations made while reading E216's code for this design.** No source was consulted for these. Each is a row of
the register (§3) and an item of the re-run (§4 phase 5).

- **O1 — spatial allocation.** The two halves of E216 place the clearing differently, and neither conserves the
  cleared area.
  - The network function (`code/e216_detection_function.py` L270–276) gives a core whose RSAP does not reach a
    heartland point only 1/100 of its uniform share of the island's clearing. No source is given for this. In effect,
    the whole population's clearing (about 11,600 km² at the floor population, 23,400 km² at the central one) sits at
    two points.
  - The corner table (L441–481) puts clearing at a density instead: the Java average (uniform) or 2–4 times it.
  - H1's P = 0.000 depends on the 1/100 factor. The sweep never varied it.
- **O2 — per-person land.** The coupling implies 0.66–3.25 ha of cleared land per person (mid 1.84 ha) at every
  population size. Wet rice and swidden (with fallow) share one value, though their land needs per person are usually
  described as an order of magnitude apart. That is to be checked (E01).
- **O3 — what was and was not swept.**
  - Swept: RPP_NAP, threshold and alpha.
  - Never swept: RSAP radii, core positions, the 1/100 factor, the J6 factors (0.35, 0.3), and per-person land.
  - NAP_BACKGROUND and RPP_ORYZA are defined but do not enter any result.
  - The code comment's RSAP ranges (L73–77) differ from the ranges in `CORES`.
- **O4 — the heartland points.**
  - The Kedu point (−7.50, 110.00) is labelled "Borobudur / Prambanan zone". By design-time arithmetic, with
    coordinates from model memory, it lies about 25 km from Borobudur and 61 km from Prambanan.
  - The Brantas point (−7.80, 112.00) is labelled "Brantas headwaters". It is about 2 km from Kediri, on the middle
    Brantas, and about 59 km from the river's source near Batu.
- **O5 — Plantago.** The Dieng positive control is recorded as a "substantial Plantago + herb rise" at about
  600 CE. If *Plantago* in Java is an introduced weed, the dating or reading of that control needs checking (P01).
- **O6 — J6 and the Brantas.** The code puts 35 % of the Brantas headwaters in the Solo catchment. The Brantas and the
  Solo are usually described as separate basins (S04).
- **O7 — the database harvest.** E216 PREREG S1 promised a harvest from Neotoma and IPPD. The core table comes from
  E214's ~20 web searches. No database query is recorded (I01, D04).

**Prediction recorded before the run (from O1).** It uses E216's own corner table and the same formula.
- Under uniform allocation, every terrestrial core's RSAP holds the Java-average cleared fraction: 9.0 % at the floor
  population and 18.1 % at the central one.
- At E216's mid parameters that gives a rise of 12.6 pp and 21.9 pp, against a 17.5 pp threshold.
- If the re-run confirms this, H1 is not true as stated. The existing network would detect a central-sized
  population spread over the island, as long as its cores resolve the window (I03).
- The result would then move *toward* E214's counter-evidence, not away from it. Line 03 asks that such a move get
  the hardest scrutiny, which is why the allocation is a pre-registered re-run here and not a sentence.

## 3. The claim register

`claims_register.csv` is frozen with this design: **41 rows** in nine groups (inventory, geography, source area,
signal model, detection, diagnostics, cited findings, population→land, missing core).
- **Tier 1 (15 rows)** can change H1–H5: I01 I02 I03 S01 S02 S03 M01 M02 M03 D01 D02 X01 P01 E01 K04.
- **Tier 2** rows are the descriptive accuracy a reviewer checks: coordinates, citations, cost.
- **Tier 3** rows are stated in E216 but do not enter any result. They are checked for the text only.

Rows from the domain read (§4 phase 1) are added as `R-A01…`, `R-B01…`, marked "added after freeze".

## 4. Procedure

Phases 1–3 are independent and run in parallel. Agent prompts are in `PROMPTS.md`, verbatim, written before the run.

**Phase 0 — freeze.** Commit and push this folder (DESIGN, register, PROMPTS, README) before any agent runs.

**Phase 1 — AI domain read (G10 route S step 2).**
- Two readers, blind to each other: **DR-A** on the main model (Opus) and **DR-B** on Sonnet. Each is prompted as a
  Quaternary palynologist reviewing for *Vegetation History and Archaeobotany*.
- **Inputs:** E216's `PAPER_DRAFT_OUTLINE.md`, `code/e216_detection_function.py`, `results/core_coverage_table.csv`
  and `results/sensitivity_summary.json`.
- **Withheld:** this design, the register and O1–O7. This lets us measure whether a specialist read finds them
  without help.
- **Rules:** no web search. Output is one row per objection: target claim, severity (fatal / major / minor), why, and
  what evidence would settle it.
- **Measure:** which of O1–O7 each reader raised.

**Phase 2 — inventory search (I01–I03, I08, K03, D04).**
- Two Sonnet searchers, **IS-1** and **IS-2**, blind to each other. Neither is shown E216's core list or E214.
- Each one:
  - queries the Neotoma API for sites with pollen or charcoal datasets in Java's bounding box (lat −8.9 to −5.8,
    lon 105.1 to 114.7);
  - runs the fixed query strings in `PROMPTS.md` (English and Indonesian), then follows reference lists of any review
    found;
  - reports every record found (not only 0–500 CE) with: coordinates, archive, proxies, dated span, a quote showing
    the 0–500 CE coverage or its absence, open raw data (yes/no, where), URL and evidence level.
- **Recall check:** the share of J1–J7 found by the two searchers together. Below 5 of 7, a third searcher (IS-3)
  runs with the queries extended before any verdict.
- The orchestrator merges and de-duplicates the records. Every record within 50 km of a heartland point (E216's or
  the corrected ones) is verified at evidence level F or A.

**Phase 3 — source extraction (all other rows).**
- Three Sonnet extractors, by group: **EX-1** (S, M, D), **EX-2** (X, P), **EX-3** (E, K, I04–I07).
- Each gets its rows, with the claim and E216's value, and reports what sources say: a value or statement, a verbatim
  quote of 300 characters or fewer, the citation, URL, evidence level, and the biome of the study site (tropical T,
  subtropical ST, other O).
- When nothing is found, the extractor writes `NO SOURCE FOUND` with the queries tried. **Extractors give no verdicts.**

**Phase 4 — guard and verdicts (orchestrator).**
- **Guard.** Every quote must be found in the saved text of the page it came from, after whitespace and NFC
  normalisation. Agents save fetched texts in the session scratchpad, outside the repo, because many are copyrighted.
  The repo keeps the URL, access date, SHA-256 of the saved text, and the quote. A quote that fails the guard is
  dropped, and its row loses that evidence.
- **Verdicts** follow §5, one row at a time, each with its reason.

**Phase 5 — re-run (`scripts/rerun.py`).** The script imports E216's code and overrides inputs. E216's own files are
not touched; they are the pre-registered run of record.
- **R0 — reproduce E216.** Assert that `OUTCOME.json` stats, the corner table and the sweep summary come out
  identical (a mini-G1).
- **R1 — allocation scenarios.**
  - E216-original (the 1/100 factor), reported as the reference.
  - **U:** uniform density over Java.
  - **C2, C4:** mass-conserving heartland clustering. Zones of 20 km radius around the two points hold 2× or 4× the
    Java average; the rest of the island holds what remains, uniformly. Zone radius 10 km and 40 km as sensitivity.
  - **H:** highland cores (J1, J5, above 1000 m) receive no clearing; the other cores get the uniform density. This
    is a conservative bound: there is no West Java DEM in the repo, so lowland density is not raised to conserve area.
- **R2 — parameter grid.** For every numeric Tier-1 parameter (RSAP per core, RPP, alpha, threshold), the grid spans
  the union of E216's range and the published range, at three points (low, mid, high). A parameter with no source
  takes E216's range × 0.5 to × 2.
- **R3 — land per person.** Mode A is split into A-sawah and A-swidden by the published per-person areas (E01). If E01
  has no source, E216's coupling is kept, and that is said.
- **R4 — inventory and geography.** Cores are added (I01), dropped when they have no dated 0–500 CE section (I03), or
  moved (I04).
  - **Main run:** E216's heartland points.
  - **Sensitivity:** corrected points — Borobudur and Prambanan for Kedu; Kediri for the Brantas.
- **Report:**
  - for each population × mode × scenario, the share of the grid where network P ≥ 0.90;
  - for each core, the smallest RSAP that would reach a heartland point, beside the largest published RSAP for its
    archive (the "K4-style" boundary that line 03's STATE asks for);
  - the corner table, recomputed;
  - E216's PREREG rule applied unchanged.

**Phase 6 — outputs** (§8).

## 5. Verdict rules

**Evidence level of a source:**
- **F** — full text read;
- **A** — abstract or publisher summary;
- **S** — search-result snippet;
- **2** — secondary (cited by another paper);
- **M** — model memory. M is never evidence; it may only point to a lead.

**Verdicts:**
- **AGREES** — the source states the same fact, or E216's range lies inside the published range.
- **PARTLY** — the ranges overlap, but E216's extends past the published one or misses part of it. The re-run grid
  takes the union.
- **DISAGREES** — the source states otherwise, or the ranges do not overlap. This is fixed in the re-run and in the
  text, never by rewording alone (line 03 rule 4).
- **NO SOURCE** — nothing found after the search protocol. The manuscript states it as an assumption, and the grid is
  widened (§4 R2).
- **INTERNAL** — checked against code, not literature (E02, S02 in part).

**Conditions on verdicts:**
- **Weak evidence.** A verdict resting only on level S or 2 is marked *weak*. A weak DISAGREES widens the grid but
  cannot trigger the kill switch (§6.1).
- **Biome.** Published numeric ranges for S01, M01 and M02 are built from tropical and subtropical studies. Values
  from other biomes are reported; they may widen a range but never narrow it.
- **Mechanical where possible.** Numeric rows are compared as ranges by script. A qualitative verdict carries the
  quote and a one-line reason, so the PI can audit it.

## 6. Decision rules (fixed now)

**6.1 Kill switch (inventory).**
- **Trigger:** a published pollen or charcoal record whose dated section covers 0–500 CE lies within 20 km of either
  E216 heartland point, or of Borobudur, Prambanan or Kediri. The evidence must be F or A.
- **Then:**
  - H3 ("the decisive missing core") is wrong as stated, and H1 must be redone with that record;
  - the E216 manuscript is not written in its present form;
  - a new experiment reads that record's published diagram against E216's pre-registered diagnostic.

**6.2 Per headline:**

| Headline | HOLDS when | Otherwise |
|---|---|---|
| H1 | network P < 0.90 at every grid point under **both** U and C4, not only under E216-original | CHANGES: restated as the share of the grid where the network detects, per population, mode and scenario. "At any parameter setting" and "0.000 uniformly" are withdrawn |
| H2 (a) | H1 holds | restated as in H1 |
| H2 (b) | no open raw counts for J1 or J2 (D04) | S2 must be run, as a new experiment, before the manuscript; (b) is not cited until then |
| H3 | X01, K01, K02 and K03 agree, and the kill switch is not triggered | CHANGES. For example, if Oryza cannot be separated from wild grasses by light microscopy, the target becomes "large Poaceae as a weak indicator + rice phytoliths + charcoal" |
| H4 | the recomputed conservative corner still fails in most of the grid | CHANGES with the new numbers |
| H5 | Mode B stays below threshold everywhere | CHANGES |

**6.3 Outcome label.** E216's PREREG rule is applied unchanged. With S2 at NO-GO, the rule sends E216 to OUTCOME-3
whatever the re-run gives. So E234 can change E216's **claims**, but not its label, unless D04 reopens S2.

**6.4 Route S status.**
- **Step 1 is done** when every register row, and every row the domain read added, has a verdict, an evidence level,
  and (unless NO SOURCE) a guarded quote and URL.
- **Step 2 is done** when every domain-read objection has a disposition:
  - fixed by the re-run;
  - claim downgraded;
  - rejected with a source;
  - open, stated in the manuscript.
- **Route S is INCOMPLETE** if **4 or more of the 15 Tier-1 rows** end NO SOURCE. More than a quarter of the
  load-bearing rows unsourced means the check is too thin to stand in for a reader. Then the abstract must name those
  rows as untested assumptions, and Claude recommends a route-H reader before submission. The PI decides.

## 7. Limits — what route S cannot do here (stated in the README and, later, in the manuscript)

- It cannot read raw counts behind paywalls. S2 stays NO-GO unless open data turn up (D04).
- It cannot judge site-specific taphonomy (hiatuses, reworking, old-carbon effects in the dates) beyond what the
  papers themselves report.
- It cannot know of unpublished cores, or theses that are not online.
- The domain read is an AI prompted as a palynologist, not a palynologist. It approximates the reviewer's tacit
  judgement, for example whether a VHA reviewer accepts a two-group simplification of REVEALS. It does not replace it.
- Published RSAP and RPP values come mostly from temperate Europe and subtropical Asia. Tropical values may be few or
  none; the biome rule (§5) keeps that visible.
- The re-run's allocation scenarios are themselves modelling choices. They are reported side by side; none is chosen
  after the fact.

## 8. Threats and controls

| Threat | Control |
|---|---|
| Searchers miss records (grey literature, Indonesian theses) | two blind searchers; recall of the 7 known cores as a sensitivity check (third searcher if below 5/7); Indonesian queries |
| Fabricated or altered quotes | verbatim guard against saved page text; SHA-256 recorded |
| Paywalls hide the decisive text | evidence levels; weak verdicts cannot trigger the kill switch |
| Orchestrator bias in verdicts | numeric rows compared by script; qualitative verdicts carry quote + reason; register published in full |
| The domain read is shaped by our own findings | readers do not see O1–O7 or the register; the overlap is measured |
| Re-run changes more than one thing at once | R0 reproduces E216 first; R1–R4 are reported one at a time and then combined |
| A result toward confirmation of the thesis gets less scrutiny | line 03 rule: a move toward the thesis gets the hardest look. O1 moves the other way; it is still pre-registered |

## 8a. Outputs

- `results/claims_verdicts.csv` — the register with verdict, evidence level, biome, quote, URL and SHA-256.
- `results/inventory.csv` — merged search results, with the recall check.
- `results/domain_read_objections.csv` — both readers' objections, dispositions, and the O1–O7 overlap.
- `results/rerun_*.csv`, `results/rerun_summary.json` — R0–R4.
- `results/headline_status.md` — H1–H5: HOLDS / CHANGES / WITHDRAWN, with the new wording as numbers (the PI writes
  the prose, G16).
- `results/G10_route_S_package.md`:
  - the claims-vs-sources table, ready for the manuscript's `SIG_signoff.md`;
  - the domain-read dispositions;
  - a draft of the AI/acknowledgement statement (G10 route S step 3).

## 9. What the PI decides after the results (proposed here, not applied)

- **Line 03 rule 2.** Proposed text: *"Domain review (G2/G10) is required, by route H (a palynologist reads — sought,
  never waited for) or route S (E234's package). This line's claims about pollen detection physics are checked
  against published values, and the manuscript says so."*
- **E216 `SUBMISSION_CHECKLIST.md` step 1.** Proposed: "Co-author secured (G2/G10) ← BLOCKS everything below"
  becomes "Domain review by route H or S (E234) ← blocks submission, not drafting".
- **Route H in parallel.** A reading request can still be sent, and starts G10's 14-day clock. The VEGAN follow-up
  of 23 Oct is about T5 and stays separate.

## 10. AI use

Every step is AI:
- DR-A on Opus and DR-B on Sonnet (domain readers);
- IS-1 and IS-2 (searchers) and EX-1 to EX-3 (extractors), all on Sonnet;
- the orchestrator (Claude) for verdicts, the re-run and the write-up.

The human judgement enters only through the published papers each row cites. This is stated in every output, as in
E233.

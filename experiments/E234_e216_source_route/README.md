# E234: E216's assumptions checked against published values (SIG G10 route S)

**Status:** PRE-REGISTERED (DESIGN.md, 2026-10-19) — not run; runs after the PI's word and after the design is pushed
**Line:** 03_paleoenv · **Checks:** E216 (paper: the "decisive missing core" manuscript, VHA) · **Gate:** SIG G10 route S, G2

**Hypothesis:** the assumptions E216's OUTCOME-3 rests on agree with published palynological, ethnographic and
agronomic values. This covers:
- the core inventory, RSAP, RPP, the local weight, the detection threshold;
- the rice-pollen diagnostic, and the reading of the cited records;
- the cleared land per person.

E216's headline claims (H1–H5) survive when its own code runs on those values.

**Why:** E216 has waited since July for a palynologist co-author, a gate its checklist calls "BLOCKS everything". PI
2026-10-19: an expert is never a blocker; pivot. E233 showed the pattern: measure against expert judgement that is
already published.

**Method:** `DESIGN.md` (frozen before any comparison).
- The claim register `claims_register.csv` has 41 rows, 15 of them Tier 1 (they can change a headline).
- Two blind AI domain reads, prompted as a palynologist reviewing for VHA (route S step 2).
- Two blind inventory searches: Neotoma plus fixed English and Indonesian queries, with a recall check on E216's
  seven cores.
- Source extraction with a verbatim guard; verdicts by fixed rules.
- A re-run of E216's own code under published ranges and mass-conserving land allocation.
- Decision rules, including a kill switch: a dated 0–500 CE record already exists at the heartland.
- Agent prompts are in `PROMPTS.md`.

**Found while designing (not yet run; DESIGN §2 O1–O7).**
- H1's "P = 0.000 at every grid point" rests on an uncited factor (L270–276). A core whose RSAP does not reach a
  heartland point gets 1/100 of its uniform share of the island's clearing. E216's own corner table instead assumes
  uniform clearing.
- Prediction recorded before the run: under uniform allocation, the terrestrial cores see the Java-average cleared
  fraction. At E216's mid parameters that is 21.9 pp at the central population against a 17.5 pp threshold, so the
  existing network would detect it.

**Data used:**
- E216 `code/`, `results/`, `PAPER_DRAFT_OUTLINE.md`;
- E196 `population_model.py`;
- published sources found in the run, with URL, access date and SHA-256 of the text quoted.

Full texts are kept outside the repo (copyright).

**Result:** — (not run)

**Conclusion:** —

**AI use:** every step is AI; the human judgement comes only from the published sources each row cites (DESIGN §10).

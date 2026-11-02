# K1 writing pack — decision taken 2026-11-02

> **STATUS 2026-11-02 (later): APPLIED.** The PI asked Claude to write the text ("kamu saja yg menuliskan"), so the
> six places below were written by Claude and applied by `apply_session6_k1.py` — see `PI_PROSE_LOG_R2.md`
> "Session 6". The manuscript (`submission_jcaa_v0.3.tex`) now carries them. This file stays as the record of what
> the text had to say and where the numbers came from; §2 is no longer a to-do list.

**Decision (PI, 2026-11-02):** K1 — report both presence sets; the central claim is what holds in both.
**Who writes:** the PI, in English. Claude supplies facts, checks every number, and inserts the text.
**Clock:** freeze ≤ 9 Nov; upload ≤ 16 Nov. Everything below is needed before the freeze.

This replaces the K row of `AGENDA_PI_20261026.md`. The other 26 Oct items (D, AI, R2, T, S, R4–R12) were
**applied in session 5** and are not reopened here, except the AI declaration clause at the end (§H).

---

## 0. The claim, in one paragraph

On the compiled records (n = 378) redesigning the background adds nothing once both designs are scored on a
common background (−0.014 AUC). On the cleaned sites (n = 161) it adds +0.051. So the size of the real gain
depends on the presence data. What holds on both sets: scoring a design on its own background overstates the
design gain by about +0.04–0.05 AUC, and only a common evaluation background shows how much real gain is left.
The selection criterion has no interior optimum in either set.

**Why this is not a weaker paper:** the diagnosis (the inflation) is the contribution, and it is now shown to be
robust to the presence data, which is the strongest form of the argument the paper can make. The real-data gain
becomes a data-dependent quantity, reported as such.

---

## 1. Facts you can use (all checked at source; do not re-state from memory)

Source of everything in this table: `experiments/E236_p2_presence_sensitivity/README.md`,
`results/summary_{full,clean}.json`, and the intervals from `ci_check.py` (2026-11-02).
The same numbers are laid out as **Table S9** in the supplement (built, compiles, `\label{tab:e236compare}`).

| Quantity | Compiled 378 | Cleaned 161 |
|---|---|---|
| Presences | 378 records, 335 distinct coordinates | 161 (155 OSM, 6 Wikidata) |
| **A.** Hybrid − random, **common** background | **−0.014** [−0.021, −0.005]\* (3/15 cells positive) | **+0.051** [+0.036, +0.067]\* (14/15 positive) |
| **A.** Same, **ten seeds** (declared extension, 2026-11-02) | not run | **+0.051 [+0.039, +0.063]**, 29/30 positive, 10/10 seed means positive; disjoint five-seed halves **+0.051 / +0.052**; on own backgrounds **+0.098**, so the inflation on this set is **+0.047** |
| **A.** Same, **own** backgrounds | +0.032 | +0.103 |
| **A.** River feature, common background | +0.042 (12/12) | +0.069 (12/12) |
| **A.** TGB − random, own / common | +0.003 / −0.001 | −0.005 / +0.001 |
| **B.** Hybrid own-background inflation | 60/60, mean +0.037 | 54/60, mean +0.041 |
| **B.** Hybrid beats random on uniform / TGB / hybrid / stratified | 0 / 0 / 3 / 0 | 3 / 3 / 3 / 3 |
| **B.** TGB inflation | −0.005 (22/60) | −0.004 (24/60) |
| **C.** Own-background criterion at hard-negative share 0.0 → 0.3 → 1.0 | 0.721 → 0.738 → 0.844 | 0.697 → 0.739 → 0.866 |
| **C.** Common background, same three settings | 0.699 → 0.695 → 0.602 | 0.680 → **0.713** → 0.656 |
| **D.** Density ratio robust/contingent (RF, XGB, MaxEnt) | 1.93, 4.34, 5.62 | 2.54, 8.51, 5.30 |
| **D.** Top-decile seed turnover | 28.1–47.4 % | 41.6–54.8 % |
| **F.** Disjoint-ensemble Jaccard, k = 1 → 5 (XGB, RF, MaxEnt) | 0.55→0.76, 0.70→0.86, 0.70→0.85 | 0.47→0.70, 0.52→0.73, 0.58→0.77 |
| **E.** Below 1,000 m, near − far, 20/40 km (random/TGB/hybrid) | +0.049/+0.049/+0.060 | +0.046/+0.046/+0.060 |

\* seed-cluster bootstrap, 5 seeds, 95 %. **It is an interval over seeds, not over sites.** With 161 sites the
site-level detection floor is larger than the ±0.03 measured at n = 378 (E223); say so, do not compute a new floor.

**What the cleaning removes** (needed for Methods): rows geocoded from their names; modern monuments; temples of
other regions, tourist objects and modern institutions; then one row per coordinate. Examples that a reader
recognises: Tugu Pahlawan, Monumen Kapal Selam, Patung Karapan Sapi, Monumen Nol Kilometer Surabaya; Borobudur
and Prambanan at East Java coordinates; Pura Besakih near Surabaya; the Tumpak Sewu waterfall. About 124 of the
378 are modern monuments and 13 are misplaced or not sites. **C029 flagged this file on 1 Oct; P2 was never checked.**

**What did not change:** E007–E013 remain the historical object and stay trained on the full file; the text says so.
The synthetic worlds (E222) are untouched and carry the mechanism.

**The ten-seed extension is done** (`EXTENSION_SEEDS_20261102.md`, declared before the run). The cleaned-set
contrast on seeds 0–9 is +0.051 [+0.039, +0.063], 29/30 cells positive, and the two disjoint halves (0–4 and 5–9)
agree at +0.0505 / +0.0516. So the abstract number is now ten seeds, and it satisfies line rule 3; you can also
write the disjoint-halves check instead of a bare seed count, which is the fair form of the stability claim (C083).
The pre-registered five-seed value is unchanged and is what Table S9's main row still reports.

---

## 2. Where to write — six places, in this order

Line numbers are those of `submission_jcaa_v0.3.tex` as of commit `f341313`. Each row quotes the opening words.

### A. Methods 2.1 — "Data, covariates, and the principle of role separation" (line 178)

**Now (line 181):** "Usable presences after covariate filtering were 378 (111--115°E); the soil-coverage subset
used once in E009 was 259."

**Must now also say:**
- what the 378 records are (compiled from OSM, Wikidata and Wikipedia, not a site catalogue);
- their composition: about 124 modern monuments, 13 misplaced or not sites, 335 distinct coordinates;
- the cleaning rule and the result (161 sites);
- that the record-level analyses (E007–E013) remain on the 378 and why (they are the object under examination);
- that the refutation contrasts (E217–E221, E235) are reported on both sets.

Keep it short — three to five sentences. No new table here; Table S9 carries the numbers.

### B. Section 3.2 — "The ladder is an evaluation artefact" (line 368)

**Now (line 371):** "The part of the rise attributed to background redesign does not survive a common evaluation
background." **Unconditional — false on the cleaned set.**

**Must now:** attach the −0.0142 / +0.0424 numbers to **the compiled records**, and end with a pointer to the new
subsection. The heading may stay: the ladder (E007–E013) is trained on the full file, so the claim that *its*
rise is an evaluation artefact is untouched. What flips is the step from the ladder to the general statement that
background redesign adds nothing — that step was measured on the compiled records and does not survive cleaning.
Do not over-correct: the ladder result stands, the generalisation does not.

The rest of 3.2 (inflation 60/60, TGB not reproduced, synthetic 240/240, the equivalence CIs) stays as it is —
that part is true on both sets, but say which set the 60/60 belongs to.

### C. NEW Results subsection, after 3.2 (becomes 3.3; everything after renumbers automatically)

**Must report,** in whatever order reads best:
1. the cleaning and the two sets (one sentence; Table S9);
2. **A:** +0.051 [0.036, 0.067] on a common background, 14/15 cells, all three algorithms positive
   (XGB +0.040, RF +0.048, MaxEnt +0.065); own-background +0.103, i.e. inflation of about +0.05 on this set too;
3. **B:** inflation survives (54/60, mean +0.041), and on the cleaned set the hybrid design also beats random on
   every evaluation background (3/3 on each of the four) — say plainly that this differs from the compiled set;
4. **C:** the criterion still rises to 1.0 (0.697 → 0.866); the common-background curve peaks at 0.3 (0.713) and
   is lowest at 1.0 (0.656), so on this set the criterion moves away from the best setting;
5. **D and F** hold, with the fragile-fringe caveat (see Limitations);
6. **E:** E235's primary contrast is +0.046/+0.046/+0.060 and still INCONCLUSIVE; 10/30 km ≈ 0; 15/30 km
   +0.019/+0.018/+0.037, NOT SPECIFIC — no volcano-specific effect in either set.

Cite `Table~\ref{tab:e236compare}` (Table S9) and `E236` in the experiment list.
**Mechanical, done by me when I insert your text:** the Table S9 entry in the Supplementary Materials list (line
618), a `\label` for the new subsection, and E236 added to the code list in Data Availability (line 644).
(E235 is described in §3.9 rather than in §2.6, so E236 follows the same pattern — it is described here, in the
results, and not added to the E217–E224 list.)

### D. Section 4.1 — "Hold the evaluation background fixed" (line 535)

**Now (line 537):** "The central methodological finding is that a background-design comparison is not identified
unless the evaluation background is held fixed …" — keep, but the sentence that follows must be true on both sets.

**Must now:** the robust finding is the **inflation** (+0.04–0.05 in both sets); whether a real gain remains is
**data-dependent** (−0.014 vs +0.051). The parenthetical "60/60 real seed×algorithm cells" needs its set named
(compiled records; 54/60 on the cleaned sites).

### E. Conclusions (line 589)

**Now (lines 593–596):** "When all designs are evaluated on a common background, changing the background design
contributes $-0.014$ AUC and adding distance to rivers contributes $+0.042$. On a common background the change of
design adds nothing. …"

**Must now:** the two-set statement, in two or three sentences. "On a common background the change of design adds
nothing" cannot stand.

### F. Limitations (line 573)

Add, in your words:
- the cleaned set has 161 sites, so its detection floor is larger than the ±0.03 measured at n = 378;
- the cleaned set's contingent fringe holds only 5, 15 and 5 sites, so its density ratios are reported but fragile;
- the cleaned-set contrast rests on five seeds (plus the ten-seed extension, if you accept its result);
- the cleaning rule is ours; a different rule could keep or drop other rows (name the rule, do not defend it).

---

## 3. Do not let these survive (they are now false or unsupported)

| Sentence (v0.3) | Why it must go |
|---|---|
| Abstract L54: "changing the background design contributes $-0.014$ AUC" | true only on the compiled records |
| Abstract L57: "The gain of the final, hybrid design is therefore an evaluation artefact" | false on the cleaned set |
| L141: "the part of this rise … is an evaluation artefact, not evidence that such designs produce better models" | too broad |
| L371: "does not survive a common evaluation background" | false on the cleaned set |
| L595: "On a common background the change of design adds nothing" | false on the cleaned set |
| L594–595: "$-0.014$ … and $+0.042$" unattributed | numbers now have a set |

**Still true, keep:** TGB adds nothing (both sets); the criterion has no interior optimum (both sets);
seed averaging helps but disjoint ensembles agree at 0.70–0.86 (both sets); no volcano-specific effect (E235);
the synthetic mechanism (E222).

---

## 4. Rules that apply to this text (R1 + the line rules)

- **No review history:** no "reviewer", "revision", "previous version", "we now", "this paper previously".
  R1 asked for a paper that stands alone. (38 such hits were removed in session 5; do not reintroduce them.)
- **"Section", not "§".**
- **Declare the estimand** with every AUC: own or common background, n, number of seeds.
- **No absolute quantifier** without the fraction beside it ("always", "never", "all").
- **Do not count converging channels as strength** (F9). The two sets are one comparison, not two results.
- **Abstract ≤ 300 words.** It is 305 now; the K1 rewrite is the place to cut.
- **Every number you write must be one of the numbers above, or I check it against source before it goes in.**

---

## 5. Mechanics

1. You write in the chat or in a file; rough English is fine.
2. I insert verbatim, compile, and check every number against `summary_{full,clean}.json`.
3. Every change is logged in `PI_PROSE_LOG_R2.md`; anything I change beyond your text is told to you first.
4. Then: re-check only the changed numbers (G1), a second independent read of the changed parts, and the
   response-letter paragraphs (you write the replies; I fill "Where in v0.3").

---

## H. One clause of the AI declaration must change (told now, not done silently)

**Now (line 654):** "In this version, new passages were written by the first author in Indonesian and translated
into English with Claude; the abstract, the conclusions and a number of corrections were drafted with Claude at
the first author's request."

**Why:** you have chosen to write the K1 passages directly in English, so "in Indonesian and translated" is no
longer true for them.

**Proposed (your yes/no):** "In this version, new passages were written by the first author, some in Indonesian and
translated into English with Claude and some directly in English; the abstract, the conclusions and a number of
corrections were drafted with Claude at the first author's request."

This is also why the editor email (drafted, ready to send) says the same thing. See `AGENDA_PI_20261026.md` §AI.

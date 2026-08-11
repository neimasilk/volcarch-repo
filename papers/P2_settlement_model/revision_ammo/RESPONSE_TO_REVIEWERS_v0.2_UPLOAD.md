# Response to Reviewers — JCAA #280, revision v0.2

## Cover note to the editor

Dear Dr Verhagen,

Thank you for the opportunity to revise, and please pass our thanks to both reviewers. This revision
is not the one we expected to write, and we want to state why at the outset rather than let you
discover it in §4.

Reviewer 1 asked us to benchmark against Maximum Entropy, and called it essential. We did. **The
benchmark refuted our own central claim.** The manuscript reported that pseudo-absence realism drives
spatial transfer, evidenced by an AUC ladder rising from 0.659 (E007) to 0.751/0.768 (E013). That
ladder is an **evaluation artefact**: each background design was scored against the negatives it had
selected for itself. Held to a common evaluation background, redesigning the background contributes
**−0.014 AUC**, while adding a single hydrological feature contributes **+0.042 AUC**. The gain we
reported does not survive the comparison Reviewer 1 asked for.

We have therefore not revised the paper's argument — we have replaced it. The same empirical material
now supports a different and, we believe, more useful proposition: **background designs in
presence-background archaeological modelling cannot be compared unless the evaluation background is
held fixed, and a selection criterion computed on a design's own background has no interior optimum.**
Seven new experiments (E217–E223) establish this, including validation against synthetic ground truth
where the true intensity surface is known.

We also report, unprompted, five defects we found in our own work while preparing this revision: an
incomplete volcano inventory (INT-1), a mislabelled result file (INT-4), a published correlation that
does not reproduce, a null-model comparison in the submitted version whose margins were quoted
against a single best run rather than the seed average, and seven overstated claims in our own
internal revision documents that we caught and corrected before they reached this manuscript. All
five are itemised under "Further disclosures" below. We would rather you have all of it.

We recognise that this is an unusual revision, and that a paper reaching the opposite conclusion from
the submitted version may in your judgement warrant fresh review. We are content with whatever process
you consider appropriate.

---

## Summary of changes

| # | Change | Driven by |
|---|---|---|
| 1 | Central claim withdrawn and replaced: the reported ladder is an evaluation artefact | R1-D (MaxEnt), our own E217 |
| 2 | Title changed; "tautology-free" removed | R1-B, R2 |
| 3 | New results section: E217–E223, including synthetic ground truth | R1-D |
| 4 | New §1 taxonomy of the four absence mechanisms; scope restricted to suitability | R2-A |
| 5 | Two new tables: covariate inclusion matrix and analytical roles | R2-D, R2-E |
| 6 | Archaeological framing expanded; heritage-management section added | R1-E, R1-F, R2-B |
| 7 | Abstract rewritten; jargon defined at first use; TRI/TWI defined in text | R1-G, R1-I |
| 8 | ENM/MaxEnt literature engaged; novelty claim moderated | R1-A |
| 9 | Volcano inventory corrected (7 → 13 centres in bounds); Test 1 recomputed | INT-1, self-reported |
| 10 | Seed-ensembling protocol added (k ≥ 7); single-seed maps withdrawn | E221 |
| 11 | Taphonomic interpretation withdrawn; terrain-matched control reported | R2-F, R2-G |
| 12 | Six further ENM references added (Lobo 2008; Barve 2011; Hijmans 2012; Warren & Seifert 2011; Radosavljevic & Anderson 2014; Guillera-Arroita 2015); §1.3 concedes the prior art before claiming the delta | R1-A |
| 13 | Maximum Entropy implementation and settings named (maxnet via `elapid`; linear/hinge/product features, β = 1.5, cloglog); bootstrap replicate count stated | R1-D, R2-D |
| 14 | Sampling deviation of E217–E224 disclosed in the methods (§2.6): benchmark backgrounds are lattice-drawn, so only within-benchmark contrasts are interpreted | self-reported |
| 15 | Supplementary tables S1–S6 supplied as one generated document, built from raw result files by script | R2-D |

---

# Reviewer 1

### R1-A — "The empirical finding is not entirely novel; ecological niche modelling has established this, and archaeological examples are missing."

**Accepted, and the situation has changed.** The reviewer was right that our original finding was not
novel; it has turned out not to be a finding at all.

We have taken the criticism as applying with equal force to the *replacement* finding, and rewritten
§1.3 to concede the prior art explicitly before claiming anything. It now states that the phenomenon
is established: **Lobo, Jiménez-Valverde & Real (2008)** showed AUC is not comparable between models
whose background differs; **Barve et al. (2011)** formalised the availability domain as the accessible
area *M* and showed its delineation conditions both fitting and evaluation; **Hijmans (2012)** showed
that the spatial arrangement of background relative to presences inflates cross-validated AUC and
proposed a null-model calibration. §1.3 further notes that criteria computed inside a single
background are known to be unreliable guides to transferability (**Warren & Seifert 2011**;
**Radosavljevic & Anderson 2014**) and that discrimination metrics answer a narrower question than
applications ask of them (**Guillera-Arroita et al. 2015**). On the archaeological side it engages
Yaworsky et al. (2020), Banks et al. (2006), Franklin (2009), Howey (2016), Verhagen & Whitley (2012)
and Noviello et al. (2018).

Against that background we claim three things and no more: (i) we *execute* the comparison this
literature implies but rarely runs end to end, on one dataset with identical folds; (ii) we
characterise the criterion a modeller actually optimises and show it has **no interior optimum**, and
we are not aware of a comparable dose-response measurement of reported against true performance along
a background-design dial; (iii) we give the effect a size in an archaeological setting, where the
inflation turns out to be the same order as the gain routinely reported as a result. The novelty claim
is therefore not "background design matters" — it is the mechanism, its dose-response, and its
magnitude.

### R1-B — "The 'tautology-free' claim exceeds the evidence and needs definition and support."

**Accepted in full.** Both reviewers flagged it and the manuscript's own Table 4 already said
CONDITIONAL PASS with T1–T2 in the grey zone. The term is removed from the title, abstract and
conclusions. §2.4 now defines exactly what the surviving tautology diagnostics (Tests 1 and 3 of the
earlier suite) can and cannot establish: they
can detect correlation between predictions and visibility proxies; they cannot demonstrate the
absence of tautology, which is not an identifiable property of observational data. We also retire the
"temporal split" label for Test 4 — the split is an accessibility proxy for discovery order, and
calling it temporal was an overstatement of what the data support (see INT-4 below).

### R1-C — "Structure obscures the research question behind methodological complexity."

**Accepted.** The E007–E013 iteration narrative is collapsed into a single progression table with two
paragraphs of interpretation; per-iteration detail moves to the supplement (§2.3, Table 2). This is
easier in v0.2 than it would have been in v0.1, because the iterations are no longer the argument —
they are the object being examined.

### R1-D — "Benchmark against MaxEnt, or justify not using it. Essential."

**Done, and it changed the paper.** We implemented a maxnet-equivalent MaxEnt and ran it under
*identical* spatial-block CV folds and *identical* background designs as E007–E013 (experiment E217).

Two results:

1. **MaxEnt does not reproduce the monotonic ladder.** Under a fixed common evaluation background, the
   full background redesign (random → hybrid) yields **−0.0142 AUC** averaged across MaxEnt, XGBoost
   and RandomForest, while adding one hydrological feature yields **+0.0424 AUC**, positive in 12/12
   paired comparisons.
2. **The ladder only exists when each design is scored on its own background.** In E218, the hybrid
   design beats the random design in 3/3 algorithms when evaluated on the hybrid background, and in
   **0/3** on each of the uniform, target-group and stratified evaluation backgrounds. The inflation
   from own-background evaluation is positive in **60/60** seed × algorithm cells (mean **+0.037**,
   range +0.005…+0.084) — the same magnitude as the entire reported E007→E013 gain.

We then tested the published number directly. Equivalence confidence intervals reject the published
+0.092 ladder gain in **12/12** algorithm × evaluation-background cells, and a bootstrap that resamples presence blocks rather than points (30 replicates) rejects it for all three algorithms, with CI upper bounds of +0.008, +0.025
and +0.026. MaxEnt regularisation β from 0.5 to 4.0 changes nothing (−0.020 to −0.022 throughout).

We are grateful for this request. It cost the paper its headline and gave it a better one.

### R1-E — "The archaeological side is underdeveloped; the road/accessibility motive appears only at the end."

**Accepted.** The survey-bias rationale for the road rasters moves to §2.1, where the
covariates are first introduced, and is stated as the design's core commitment rather than a technical
aside. The East Java archaeological background is expanded (§1.4), and the new covariate-role table
(Table 1, R2-E) makes the accessibility channel explicit from the start. This item and R2-B are
answered by the same revision: the archaeological content of this design lives in the background, not
in the label set.

### R1-F — "Put the results in the context of current and future heritage management in East Java."

**Accepted.** A new discussion subsection (§4.4) covers the Cagar Budaya framework (UU
11/2010), BPCB Jawa Timur survey practice, and how a suitability surface would enter permitting and
rescue prioritisation. This connects to a result we can now state responsibly: priority maps are
unstable under random seed alone (28.1%–47.4% of top-decile cells turn over between seeds), so a
single-seed map is not a defensible basis for a survey permit. We report a **robust core** (cells
selected by every seed) and a **contingent fringe** (Figure 5), with observed site densities in the
robust core 1.9×, 4.3× and 5.6× the fringe for RandomForest, XGBoost and MaxEnt respectively.

### R1-G — "Jargon undefined; the abstract is too technical."

**Accepted.** "Tautology suite", "conditional pass" and "null-model ceiling" are
either defined at first use or removed. TGB, DKNS, MVR, TRI and TWI are spelled out at first use. The
abstract is rewritten without per-iteration AUC values and carries a single headline number (+0.042).

### R1-H — "Citations are used descriptively rather than to support specific claims."

**Accepted.** A pass over §1 attaches every citation to a specific proposition.

### R1-I — "Define TRI and TWI in the text, not only in the figures."

**Done.** Both are defined at first use in §2.1.

---

# Reviewer 2

### R2-A — "The research question is ambiguous: suitability, site prediction, burial detection and survey-bias correction are different questions."

**Accepted; this was the most consequential comment we received.** §1.2 now opens with
an explicit taxonomy of the four reasons a site may be absent from the record — environmentally
unsuitable, not surveyed, buried, destroyed — and states that **this paper models suitability only**.
The other three are the interpretive frame, not the output.

The reframing has made this easier: with the taphonomic interpretation withdrawn (R2-F), the paper now
asks one question — *can competing background designs be compared on the evidence usually reported?* —
and answers it.

### R2-B — "What makes the approach specifically archaeological? Sites function mainly as spatial observations."

**Accepted, and we now answer it directly rather than defensively.** The archaeological content is not
in the label set; it is in the **background design** — a claim about where a survey could plausibly
have detected a site had one been present. That is what the target-group and hybrid designs encode,
and it is why they are archaeological rather than generic ML choices.

This also sharpens the paper's finding: because the archaeological content lives in the background,
and because the background is also what the model was evaluated against, the very thing that made the
design archaeological is what contaminated its evaluation. §2.4 and §4.2 state this.

### R2-C — "A two-stage design would be stronger: model known settlements, then find suitable-but-absent areas and test which mechanism explains the absence."

**Agreed in principle; out of scope for this version, and we say so plainly.** We built the first stage
of this design (E219) and can report the mechanics, but the second stage tests *why sites are absent* —
a question that requires the site-prediction claim we have now withdrawn. Attributing absence to burial
versus survey effort with the evidence available to us would reproduce exactly the error this revision
exists to correct. §4.6 (Limitations) declares the two-stage design future work and specifies what it
would require: burial-depth data (not currently available at survey resolution) and a
survey-effort record independent of road distance.

### R2-D — "Cannot tell which variables are in and out per experiment; the model is not clearly reproducible."

**Accepted.** A per-experiment covariate inclusion matrix is added (Table 2, §2.3),
built by reading the analysis scripts rather than the prose. Alongside it: presences n = 378 with valid
covariates inside 111–115°E, pseudo-absence ratio 1:5 throughout, deterministic 5-fold spatial block CV
at 0.45°, 2 km presence buffer, and the exact background parameters per experiment.

Building that table surfaced a defect of exactly the kind the reviewer was worried about — see INT-4
under "Further disclosures". We report it because the reviewer's concern was justified.

### R2-E — "Separate variables by analytical role: suitability, accessibility/survey effort, preservation."

**Accepted, and adopted as an organising principle.** New Table 1 (§2.1) assigns every
variable a single role: elevation, slope, TWI, TRI, aspect and river distance are **suitability**
predictors; road distance is **accessibility/survey effort** and is deliberately *never* a training
feature; volcano distance is a **taphonomic diagnostic** applied post hoc and never a predictor; clay
and silt (preservation) were tested in E009 and dropped when they reduced performance.

We also state a consequence the reviewer implies but does not spell out: road distance carries **four
roles** in this design — it defines the target-group background, it defines the hybrid pool, it defines
the Test 4 holdout split, and it is a tautology proxy in Tests 1 and 3. The submitted manuscript
mentioned this coupling once, in the limitations. Given E217, that placement was wrong: the variable
that builds the background is the variable that drives the reported number, and it now appears in §2.1.

### R2-F — "Elevation and slope may drive low suitability in rugged terrain regardless of volcanism; compare volcanic uplands with environmentally similar non-volcanic uplands."

**Accepted, tested, and the taphonomic interpretation is withdrawn.** We ran a coarsened-exact-matching
control (E219, part C): raster cells matched on elevation × slope × TRI × TWI, comparing volcanic
uplands with East Java's non-volcanic uplands (Southern Mountains karst, Kendeng limestone hills), 90
matched strata.

Result: matched mean suitability is **0.2249 (volcanic) vs 0.1702 (non-volcanic)**, and observed site
density is **0.01377 vs 0.00048 sites/km²**. The direction is consistent with a volcanic-specific
effect — but the non-volcanic arm contains **only 2 known sites**, so we present this as *consistency*,
not validation, and we do not build an argument on it.

More decisively: the taphonomic claim is withdrawn regardless of this control, because the model
comparison that supported it does not survive R1's MaxEnt benchmark. §3.9 and §4.3 state that
low predicted suitability near volcanoes is not evidence of buried sites, in the reviewer's own terms.

### R2-G — "Low suitability ≠ buried site; high suitability ≠ site exists."

**Accepted without reservation.** This is now stated in the discussion in these words, and it is a
constraint the revised paper takes seriously: with the site-prediction claim withdrawn, the output is a
suitability surface with declared limits, not a discovery map.

### R2-H — "Figure 1 is too simple; labels overflow in Figures 1 and 4; Figure 5 must explain how importance is computed and interpreted."

**Accepted, and the figure set has changed with the paper's argument. We state the changes explicitly
rather than quietly, because the reframing (R1-D) means the figures you asked us to repair are no
longer the figures the paper needs.**

- **v0.1 Figure 1 (interdisciplinary framework) — removed.** It illustrated the old argument. The
  pipeline it depicted is now carried by the covariate-inclusion table (Table 2) and §2.2 prose: which
  covariates enter as features, which enter only through the background, and what each analytical role
  is.
- **v0.1 Figure 4 (AUC/TSS progression) — retained, relabelled Figure 2.** Labels are corrected.
- **v0.1 Figure 5 (feature importance) — removed.** Feature importance across the old E007–E013
  pipeline was part of the argument we have replaced; with the taphonomic interpretation withdrawn, the
  ranking of elevation and slope is no longer a finding, so we do not expand its caption.
- **New figures carry the reframed argument.** Figure 3 shows reported vs common-background AUC across
  the design ladder together with the histogram of own-background inflation (E218/E222); Figure 4 shows
  the dose-response of the reported metric against the design dial, real and synthetic; Figure 5 shows
  the robust core / contingent fringe priority map; Figure 6 shows the seed-stabilisation curve. The
  study-area map (Figure 1) now labels all 13 canonical volcanic centres inside the paper's stated
  bounds — the inventory correction (INT-1) made visible.

---

# Further disclosures — issues we found ourselves

None of the following was raised by the reviewers. We report them because the paper's subject is the
honest reporting of model performance.

**1. INT-1 — incomplete volcano inventory (corrected).** The analysis code hard-coded **7** volcanic
centres (Kelud, Semeru, Arjuno-Welirang, Bromo, Lamongan, Raung, Ijen). The canonical inventory
contains **13** inside the paper's own stated bounds of 111–115°E, adding Lawu, Wilis, Kawi-Butak,
Penanggungan, Iyang-Argapura and Baluran — and Kawi-Butak and Penanggungan sit inside the
Malang–Mojokerto site concentration, so the omission distorted the distance field precisely where the
sites are. Volcano distance is not a training feature, so model performance is unaffected. Test 1 has
been recomputed on the canonical inventory: ρ = **−0.281** (previously reported −0.163). The verdict is
unchanged — |ρ| remains below the 0.5 threshold.

**2. INT-4 — a mislabelled result file (corrected).** Our Test 4 script contains two branches: a
discovery-year split, used only if enough sites carry known discovery dates, and an accessibility
fallback. The fallback ran, but the output template printed the temporal labels regardless, so the
stored result file described a "Split year: 2000, pre-2000 n=333, post-2000 n=45" split that never
happened. We verified the actual split by resampling the road-distance raster at all 378 site
locations: exactly **333 sites ≤ 1 km and 45 sites > 1 km**. The manuscript's Test 4 text describes the
accessibility split correctly, and AUC = 0.755 is unaffected; only the archived artefact was wrong. It
now carries a correction notice and the script records which branch ran.

**3. A published number that does not reproduce.** Test 1's volcano-distance correlation was published
as ρ = −0.163. A five-seed re-run on the *same* 7-volcano inventory gives **−0.243**. The published
value came from a single model instance. We report this not only as a correction but as evidence for
the protocol we now recommend: the instability we document in priority maps (28.1%–47.4% top-decile
turnover between seeds) also affects the manuscript's own diagnostics. Single-seed values should not be
reported, including by us. The corrected diagnostics in v0.2 are seed-ensembled; the E007–E013 ladder
is reproduced as the original published pipeline, because it is the object under examination rather
than a result we are asserting. The number of seeds required for Jaccard ≥ 0.90 in our stability
analysis is 4–7 depending on algorithm, and 7–9 for ≥ 0.95, so we recommend k ≥ 7.

**4. A null-model margin quoted against the wrong estimator (corrected).** The submitted version's null-model table reported the gap between our model and each null as +0.268, +0.187 and +0.122 AUC. Those margins were computed against E013's **best run** (0.768) while the table's headline value, and the value we quoted in the text, was the **seed-average** (0.751). Each margin was therefore overstated by 0.017. Table 4 now reports gaps against the seed average (+0.251, +0.170, +0.105) and says so in the caption. We found this while checking that the manuscript did not mix estimators in the very section where it criticises mixing evaluation backgrounds; it did.

**5. Seven overstated claims in our own revision drafts, caught before submission.** In preparing this
revision we wrote internal documents that themselves overstated the new findings — "the selection rule
picks the worst design in 100% of cases" (it picks a design costing +0.194 against the best, but never
the worst), "the reported number moves ~10× faster than the truth" (≈2×), "always inflated" (95.3%,
343/360), "monotone to the end of the dial" (one dip on the real data), "2–5.6× density" (1.9–5.6×).
Each was caught by a blind re-derivation of every headline number from the raw per-run outputs, run as
a pre-submission gate. We mention this because it is the same failure mode the paper is about, it
recurred in our own correction of it, and only a mechanical check caught it.

**6. An explanation we proposed, tested, and had to withdraw.** Our synthetic experiments found that
target-group background did not improve truth-anchored recovery over a random background — an
uncomfortable result, since target-group background is the standard bias correction and our simulation
gave it exactly the condition its theory requires. We proposed an explanation: the model's feature set
does not contain road distance, so the survey-bias factor is not representable in feature space and
target-group correction has nothing to cancel. It was a tidy account and it nearly entered this
manuscript as a "predicted null".

We pre-registered a test with both decision branches written down, then ran it (E224): the same
synthetic worlds with road distance added to the feature set. **It made no difference.** Target-group
background remained slightly worse than random: −0.022 in top-decile map agreement with road distance
in the features, −0.025 without, with 30% of the 30 paired comparisons positive in both conditions.
We therefore report the null as **unexplained** rather than predicted, and note the limitation of our
own test: road distance correlates +0.49 with river distance, so the model had partial access to the
bias signal even in the control condition, and a clean test would need a survey-effort surface
orthogonal to terrain by construction. That is future work.

**7. What we did not do.** Additional synthetic bias regimes (non-stationary bias, bias correlated with
non-road covariates) and replication in a second real region are declared future work. Our synthetic
evidence covers four regimes with n ≈ 300–500 observed presences; it is not a general refutation of
target-group background, and we do not claim one. The bootstrap can exclude effects of about +0.03 or
larger at n = 378; smaller effects remain undetermined.

---

## Note on the APC waiver

We requested an APC waiver on 2026-04-06 and it was acknowledged on 2026-04-07 but not resolved. We
would be grateful for a decision at whatever point in the process is convenient; it does not affect the
revision itself.

---

Yours sincerely,

**Mukhlis Amien** (corresponding author) — amien@ubhinus.ac.id — ORCID 0000-0002-1848-167X  
**Go Frendi Gunawan**  
Universitas Bhinneka Nusantara, Malang, Indonesia

11 August 2026

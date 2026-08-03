# Line 01 — SPATIAL (Predictive Modelling & Site Distribution)

> **Question:** Where were the settlements, and can a model find them — honestly?

**Recommended model:** Opus (dense statistical reasoning; this line is where the project's own
refutation lives). **Effort:** high.

---

## Scope

GIS/ML predictive modelling of archaeological site distribution on Java, and the geostatistics of
site/temple/inscription patterns. Reviewer community: **computational archaeology** (JCAA, ArchCalc,
Internet Archaeology). Toolchain: xgboost, scikit-learn, MaxEnt, geopandas, rasterio.

**In scope:** suitability models, background/pseudo-absence design, spatial CV, autocorrelation,
priority maps, site–temple distance statistics, satellite/SAR prospection.
**Out of scope:** *why* a site is missing (→ [02_taphonomy](../02_taphonomy/)), what texts say
(→ [04_language_text](../04_language_text/)), whether the civilization existed (→ [06_thesis](../06_thesis/)).

---

## The one thing to know before touching anything here

**This line refuted its own flagship claim, and that refutation is now the paper.**

P2's published headline (a +0.092 AUC gain from "hybrid" background design) is an **evaluation
artefact**: the model was scored against the background distribution it was tuned on. E217 found it,
E218/E218b confirmed it and named the mechanism, E222 tested it against synthetic ground truth,
E223 gave it CIs. Reported AUC and true generalisation move in **opposite directions**.

Then a hard review (`09_REVIEW_ATAS_BABAK2.md`) found that **two of our replacement headline claims
were themselves overstated** (K1–K3). Applying them on 2026-08-03 under SIG G1 — a blind re-derivation
of all 61 numbers from the per-run files — turned up **three more** (K5–K7). All are applied and live in
**`review_package_20260727/10_SET_KLAIM_TERKOREKSI.md`**, which is now the authoritative claim set.

| | Overstated claim | Measured truth |
|---|---|---|
| **K1** | "the manuscript's selection rule picks the worst design 60/60" | True only on a dial extended to `hard_frac=1.0`. On the grid the manuscript actually used (`{0.0, 0.15, 0.30}`): cost **+0.0000** median synthetic, **+0.0044** real. What survives: the criterion has **no interior optimum** — the manuscript stopped at 0.30 only because its grid stopped. |
| **K2** | "reported number moves ~10× faster than truth" | **2.01×** synthetic (endpoint) / 2.12× (per-run OLS slope — name the estimator). Real data: the two move in **opposite directions**. |
| **K3** | "always inflated" | **343/360 = 95.3%** (min −0.031, median +0.187). |
| **K4** | *(constructive)* the TGB null was reported as a surprise | `road_dist` is **not a model feature**, so the model cannot express survey bias and TGB has nothing to cancel. Turns a weakness into a tested condition. Confirmation run pre-registered as **E224**. |
| **K5** | "the rule picks the **worst** configuration 100% of the time" (doc 08 §3) | **False.** It picks hybrid(1.0), which is **never** the truth-worst (that is hybrid(0.0), 50/60). Correct wording: it costs **+0.194 against the best available design**. |
| **K6** | "the reported criterion rises **monotonically** to the end of the dial" | Synthetic yes; real data has one dip (0.0→0.1, −0.0071). Correct wording: **its maximum always sits at the edge of whatever grid is swept.** |
| **K7** | "robust-core site density 2–5.6× the fringe" | **1.93×** / 4.34× / 5.62× (rf / xgb / maxent). The low end is 1.9, not 2. |

Plus **G1c**, found by the same sweep: the published Test-1 correlation **ρ = −0.163 does not reproduce**
— a 5-seed re-run on the same 7-volcano inventory gives −0.243. That is the seed instability of this
line's own D1 finding appearing inside the manuscript's tautology diagnostic. Disclose it.

**Do not draft manuscript v0.2 from `08_HANDOFF_BABAK2.md` §3.** It calls itself final; three of its
five claims are now withdrawn. K1–K7 are an application of the SIG rule to our own work: *never answer
a valid critique by rewording.*

---

## Papers

| Paper | Folder | Status |
|---|---|---|
| **P2** Settlement model | `papers/P2_settlement_model/` | 🔥 **R&R, JCAA #280, deadline 2026-08-20.** First R&R in 14 months. R2 is the gate (framing "Poor"). Direction chosen: **Jalur A** — reframe around the artefact finding. APC £593 waiver still undecided. |
| **P17** Two Javas | `papers/P17_two_javas/` | Under review — ArchCalc (CNR, Diamond OA) submission **365**, double-blind. WAIT. Best odds in the portfolio. |
| **P11** Volcanic informedness | `papers/P11_volcanic_informedness/` | Rejected 2× (Cornell *Indonesia* scope; *Archipel* editorial). Core finding survives: temple–settlement gap 6.78 km, 80.6% <10 km, p<1e-6, inventory-independent. Retarget **SPAFA Journal** → Wacana → PCI Archaeology. Gated on: apply `revision_ammo/CANONICAL_INVENTORY_CORRECTIONS_20260610.md` (4-number abstract swap) + pass SIG. |

**77 experiments** are assigned to this line (66 primary). Authoritative list:
`docs/EXPERIMENT_INDEX.md` §"By Line of Inquiry" — regenerate with
`python tools/scan_experiments.py`.
*(D2 / Mini-NusaRC moved to [02_taphonomy](../02_taphonomy/): it is a radiocarbon database built for
H-TOM testing, not a site-model dataset.)*

**Key sub-folders:** `papers/P2_settlement_model/review_package_20260727/` — 9 documents;
**`09_REVIEW_ATAS_BABAK2.md` is the most recent and corrects 07 and 08.** Three rounds of work
happened on one day and later rounds correct earlier ones — read 09 or you will use wrong numbers.
Plan: `revision_ammo/JCAA_R1_RESPONSE_PLAN_20260727.md`.
**DO NOT USE:** `revision_ammo/anticipated_critiques.md` (stale, header-flagged).

---

## Experiments

**Refutation suite (2026-07-27, all pre-registered in `DESIGN.md`):**
`E217_maxent_benchmark` · `E218_evaluation_artefact` · `E219_map_divergence` ·
`E220_wrong_direction_selection` · `E221_seed_ensemble_stability` ·
`E222_synthetic_ground_truth` · `E223_statistical_robustness`

**Model lineage:** `E004`–`E006`, `E007`–`E013` (v1→v7, AUC 0.659→0.768), `E014` (temporal),
`E015` (SHAP), `E016` (zones)
**Distribution & nulls:** `E019`, `E100`, `E103`, `E104`, `E108` (demographic null), `E109`
(survey–burial confound), `E122` (gap sensitivity), `E126`, `E129`, `E152`, `E153`, `E175`, `E179`
**Spatial statistics:** `E184` (autocorrelation), `E185`, `E187`, `E183`
**Cascade (DOWNGRADED — over-parameterised, keep as pedagogy only):** `E110`, `E115`, `E176`, `E182`
**Prospection / remote sensing:** `E189`–`E194`, `E202`, `E209` (AUC 0.844 — a Hindu-Buddhist site
detector, **not** pre-Hindu), `E210`
**Priority maps & fieldwork:** `E059`, `E080`, `E139`, `E166`, `E167`, `E171`

Canonical volcano file: **`data/processed/dashboard/volcanoes_java_full.csv`** (30 volcanoes).
Never use the old 7-volcano `volcanoes.csv` — see [02_taphonomy](../02_taphonomy/) for the defect.

---

## Line rules

1. **SIG G1 is live here.** Every headline number in v0.2 gets re-derived blind from raw
   `results/`, including E217–E223. Partially done.
2. **Declare the estimand.** This line's whole lesson is that a metric without a declared
   availability domain is meaningless. State background design, benchmark, and selection rule
   explicitly in every claim.
3. **≥7 seeds for any published map or AUC** (E221: k\*=7 for XGB at J≥0.9). Single-seed results are
   not reportable — 31–45% of top-decile cells turn over on seed alone.
4. **No absolute quantifiers** ("always", "never", "all") without the fraction next to them. K3 is
   why.
5. **Nothing goes to the editor without the PI.** The Verhagen disclosure email is **HELD** by PI
   instruction: `docs/correspondence/EMAIL_VERHAGEN_JCAA_DISCLOSURE_DRAFT_20260727.md`.

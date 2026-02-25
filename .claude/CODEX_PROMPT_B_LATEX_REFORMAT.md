# CODEX PROMPT B: Reformat Paper 2 to MDPI Remote Sensing Template + Text Fixes

**Context:** VOLCARCH project, Paper 2. Current manuscript is at `papers/P2_settlement_model/submission_remote_sensing_v0.2.tex` using plain `\documentclass{article}`. This needs to be reformatted to MDPI Remote Sensing journal template and several text issues need to be fixed.

**Rules:**
- Do NOT push to GitHub.
- Do NOT change any experimental results or numbers (except the headline AUC fix below).
- Output the new file as `papers/P2_settlement_model/submission_remote_sensing_v0.3.tex`.
- Update `papers/P2_settlement_model/CANONICAL.md` to point to v0.3.
- After writing the .tex, compile with:
  ```
  "C:\Users\Mukhlis Amien\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode -halt-on-error papers/P2_settlement_model/submission_remote_sensing_v0.3.tex
  ```
  Run twice for references. If pdflatex fails on MDPI-specific packages, fall back to a clean article class that mimics MDPI structure (two-column is optional; single column is fine for initial submission).

---

## TASK 1: MDPI Template Structure

Remote Sensing (MDPI) uses a specific structure. Since the actual `mdpi.cls` may not be installed, create a LaTeX file that follows MDPI conventions as closely as possible using standard packages:

### Required elements:
1. **Article metadata block** (before `\begin{document}`):
   - Title
   - Author with affiliation: `Mukhlis Amien` — `Program Studi Informatika, Universitas Bhayangkara Surabaya, Surabaya 60231, Indonesia` — email: `amien@ubhinus.ac.id`
   - ORCID if available (skip if not)
   - Received/Accepted dates: leave as placeholder `Received: ; Accepted: ; Published: `

2. **Section structure** (MDPI standard):
   - Abstract (with keywords below)
   - 1. Introduction
   - 2. Materials and Methods
   - 3. Results
   - 4. Discussion
   - 5. Conclusions
   - Supplementary Materials
   - Author Contributions
   - Funding
   - Data Availability Statement
   - Conflicts of Interest
   - References

3. **References:** Convert the current `\begin{thebibliography}` to a `.bib` file (`papers/P2_settlement_model/references.bib`) and use `\bibliographystyle{mdpi}` or `unsrt`. The .bib entries should have proper BibTeX format. Current references in the .tex:
   - phillips2009, roberts2017, valavi2019, allouche2006, breiman2001, chen2016, verhagen2012, castiello2021, wang2023, comer2023, newhall1982, folch2012, mastin2016, mastin2022

---

## TASK 2: Text Fixes (apply during reformat)

### Fix 2a: Headline AUC — use seed-averaged, not single-seed

**Current (wrong):** Abstract and conclusions report single-seed AUC 0.768 as the headline.

**Correct:** The paper should use **seed-averaged AUC 0.751 (95% CI: 0.745–0.756)** as the primary result. The single-seed best (0.768) can be mentioned as "best single-run" but not as the headline.

Apply this change in:
- Abstract: change "achieved the best single-run result (XGBoost AUC 0.768)" → "achieved seed-averaged XGBoost AUC 0.751 (95% CI: 0.745–0.756; best single-run 0.768)"
- Section 3 (Results): keep the single-run 0.768 in the progression table, but add a sentence: "Across 20 alternate random seeds, E013 yielded a mean XGBoost AUC of 0.751 (95% bootstrap CI: 0.745–0.756), confirming that the result is stable but near-threshold."
- Section 5 (Conclusions): change "single-run AUC 0.768" → "seed-averaged AUC 0.751 (best single-run 0.768)"

### Fix 2b: Hard-frac discrepancy explanation

Add to Section 2 (Materials and Methods), after the E013 sweep description:

> "Note that the actual proportion of environmentally dissimilar pseudo-absences (zdist $\geq$ 2.0) in the best E013 configuration was 0.62, exceeding the target of 0.30. This occurs because the TGB candidate pool, constrained to road-accessible locations, is inherently more environmentally dissimilar from archaeological site environments than unconstrained random sampling would produce. The hard-fraction parameter controls only the intentionally selected hard negatives; additional candidates with high environmental distance enter through core sampling. This pool composition effect should be considered when interpreting the absolute AUC values."

### Fix 2c: Tautology test nuance

In the Discussion section (currently subsection "For Geology Readers"), add after the existing text:

> "We note that the negative Spearman $\rho$ is a necessary but not sufficient condition for tautology-free status. A model could still indirectly encode survey-access patterns that correlate with volcanic proximity without using volcano distance as a direct predictor. A stronger test would compare model predictions within surveyed versus unsurveyed volcanic zones, which requires spatially explicit survey-effort data not currently available for East Java. We flag this as a priority for future work."

### Fix 2d: Add new figure references

If Codex Prompt A has already generated the figures, add references to:
- `fig10_study_area_map.png` → as Figure 1 (move current fig1 to Figure 2)
- `fig11_suitability_map_static.png` → as new figure in Results section
- `fig12_feature_importance.png` → as new figure in Results section

If the figures don't exist yet, add placeholder `\includegraphics` with TODO comments.

---

## TASK 3: Create BibTeX file

Create `papers/P2_settlement_model/references.bib` with all references from the current .tex, properly formatted. Example entry:

```bibtex
@article{phillips2009,
  author  = {Phillips, Steven J. and Dud{\'\i}k, Miroslav and Elith, Jane and Graham, Catherine H. and Lehmann, Anthony and Leathwick, John and Ferrier, Simon},
  title   = {Sample selection bias and presence-only distribution models: implications for background and pseudo-absence data},
  journal = {Ecological Applications},
  year    = {2009},
  volume  = {19},
  number  = {1},
  pages   = {181--197},
  doi     = {10.1890/07-2153.1}
}
```

---

## Deliverables

1. `papers/P2_settlement_model/submission_remote_sensing_v0.3.tex` — reformatted manuscript
2. `papers/P2_settlement_model/references.bib` — BibTeX references
3. `papers/P2_settlement_model/submission_remote_sensing_v0.3.pdf` — compiled PDF (if compilation succeeds)
4. Updated `papers/P2_settlement_model/CANONICAL.md`
5. Append entry to `docs/JOURNAL.md` documenting the reformat

**Do NOT push to GitHub.**

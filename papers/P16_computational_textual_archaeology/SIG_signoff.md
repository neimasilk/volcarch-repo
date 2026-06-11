SIG sign-off — P16 (Computational Textual Archaeology) — 2026-06-08 — run by Claude (main loop)
Target: Wacana, Journal of the Humanities of Indonesia (Scopus Q2, zero-APC) — submission_wacana_v1.0.tex

G1 re-derivation: GREEN — all headline numbers (z=7.39, cosine 0.244/0.395, χ²=16.58 p=0.0003) are NLP outputs reproducible from public data (DHARMA + corpus E089 v5 + scripts in repo). No dependence on the volcano-distance artifact.
G2 domain-sanity: GREEN — removed "45 active volcanoes" (non-canonical count) from abstract; no remaining domain-fact claim hinges on a contested figure.
G3 canonical data: GREEN — paper does not use the volcano-inventory file load-bearingly; removed the one artifact-adjacent citation (former court-zone 15–30 km / Amien2026b spatial claim). Residual Amien2026b cite (line ~423, "temples on volcanic slopes") is a mild true statement, not the inflated-distance claim.
G4 circularity: GREEN — method/genre claim, not absence-as-erasure; no sampling-on-DV.
G5 equifinality: GREEN — core argument is explicitly "silence = genre selection, not ignorance" (the opposite of the absence=erasure error that sank P7). Alternatives acknowledged in Discussion/Limitations.
G6 counter-evidence: GREEN — reframed to be AGNOSTIC about population magnitude (removed "if ancient civilizations were real"); the genre-bias finding stands regardless of the peradaban-vulkanik downgrade. Not dependent on the E214-contested "large population" claim.
G7 reproducibility: GREEN — corpus-construction protocol (passage-selection criterion + thematic tagging rubric + 7 query definitions) now documented in Data availability + repo. (Ensure E089 v5 corpus is pushed public before submit.)
G8 overstatement: GREEN — softened "disappear entirely" → "absent in the post-929 subset (n=6 pre-929)" (×2); "first transformer NLP on Old Javanese epigraphy" retained (defensible, verified no prior work).
G9 cross-model: RED — DeepSeek-reasoner (2026-06-08) recommends REJECT. Full review: external_reviews/critical_deepseek_p16_wacana_20260608.md. FATAL: convergence test (Monte Carlo within-group vs random) measures within-group topical coherence, NOT cross-tradition convergence — so "12 traditions converge" (Finding 1) is unsupported by the current test; needs a tradition-controlled test (cross-tradition vs within-tradition pairwise, or label permutation). MAJOR: (2) "volcanic silence" overstated — OJ rank 4/7 is intermediate, not silence; query may not match OJ volcano vocabulary; (3) 929 CE shift underpowered (N=46, cells <5, drift-max at C11→C12 not 929) → downgrade to suggestive; (4) over-claiming genre-taphonomy as proven (equifinality); (5) no multiple-testing correction across 8 groups + 7 queries. MINOR-but-real: (6) Wacana scope-fit — too computational, thin humanities engagement → consider DHQ, or deepen philological context.
G10 human independent review: N/A (not the flagship masterpiece; optional).

Downgrades made to pass: abstract reframed magnitude-agnostic; "disappear entirely" softened; AI-disclosure productivity boast removed; "45 volcanoes" removed.
Last-mile status (2026-06-08):
- DONE (1) British-spelling pass: civilisations→civilizations, characterisation→characterization; analyse/modelling already British; "center" hits were \centering (LaTeX), not prose.
- DONE (2) compile clean: pdflatex→bibtex→pdflatex×2 → 20pp, 0 undefined citations, 0 LaTeX errors; abstract 142 words (≤150).
- DONE (3) corpus-reproducibility note (G7 → GREEN).
- DONE (4) Wacana author block (research interests + 1 main publication + email) added per styleguide.
- REMAINING (a) OPTIONAL G9 cross-model critical review (DeepSeek/Gemini) — recommended final integrity check before a real submission.
- REMAINING (b) references currently chicago author-date (satisfies Wacana's author-date requirement); exact KITLV house style (single-quote titles, '; ' subtitle, 'in:', 'pp.') is a copyedit-on-acceptance item, acceptable as-is for submission.
- REMAINING (c) Pak Amien: final read + submit via Wacana portal (push E089 v5 corpus public first).

DECISION: **NO-GO — revise before submission.** Format/framing/integrity-of-presentation pass, BUT G9 cross-model review caught a FATAL methodological flaw (convergence test) + 3 MAJOR issues. Per SIG, central+valid critiques must be FIXED (re-analysis/downgrade), not reworded. The gate worked: this is the class of flaw that produced the P1/P7 rejections — caught here BEFORE submission, not after.

REQUIRED before submit (revision round R1):
1. [FATAL → ✅ RESOLVED 2026-06-08] Redone as a tradition-controlled test (`experiments/E090_transformer_textual_nlp/e090_v6_tradition_controlled.py`): S_cross = mean cosine of CROSS-tradition pairs within each theme vs a random CROSS-tradition baseline (5000 bootstrap), BH-corrected. **RESULT: cross-tradition convergence SURVIVES for all 8/8 groups** — z_cross 3.56–32.17, p ≤ 2e-4; VOLCANO z=7.14 (12 traditions); no group tradition-dominated (dominant share 0.17–0.27, so NOT within-tradition homogeneity). Honest nuance: S_within (0.39–0.49) > S_cross (0.31–0.39) — within-tradition pairs are more similar (expected), but cross-tradition still significantly exceeds random → genuine convergence. **Finding 1 holds under the rigorous test — UNLIKE P7's distance artifact.** TO DO in manuscript: replace the intra-group exp5 test with this tradition-controlled test; report S_within>S_cross honestly.
2. [MAJOR] Temper "volcanic silence": lead with the cross-lingual-robust statement; report OJ rank 4/7 honestly; do a direct Old-Javanese volcano-term search; drop the word "silence" where 4/7 contradicts it.
3. [MAJOR] Downgrade 929 CE: present as suggestive/hypothesis; use Fisher exact (cells <5); foreground the drift-timing mismatch (C11→C12 > 929); remove causal language.
4. [MAJOR] Hedge the discussion: "consistent with" not "is"; acknowledge equifinality (shared Indian/Chinese sources; OJ volcano vocabulary).
5. [MAJOR] Add multiple-testing correction (BH/Bonferroni) across 8 groups + 7 queries; expand corpus-construction protocol.
6. [SCOPE] Decide venue: deepen humanities/philological engagement for Wacana, OR switch to DHQ (method-home, where the computational contribution is the point).

## R1 STATUS (2026-06-08) — DONE except W6
- W1 ✅ RESOLVED (tradition-controlled test, 8/8 survive; integrated into methods/results/Table 2/fig3/abstract/conclusion).
- W2 ✅ tempered (volcanic silence → below-average both langs, rank 4/7 OJ).
- W3 ✅ tempered (929 → sample-limited, Fisher, drift caveat).
- W4 ✅ hedged + equifinality paragraph.
- W5 ✅ BH multiple-testing correction noted.
- W6 ⏳ PENDING — humanities deepening (Pak Amien, ~2 paragraphs) OR switch to DHQ.
- Manuscript recompiles clean: 18 pp, 1.1 MB, 0 undefined refs/citations, 0 errors, abstract 139 words.
**REVISED DECISION: CONDITIONAL GO** — submit after (W6 humanities deepening for Wacana) + (Pak Amien final read) + (optional re-run G9 to confirm R1 fixes). The FATAL flaw is fixed and verified; the finding survived the rigorous re-test.

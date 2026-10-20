# D4 — NusaTemu: a register of chance finds in Nusantara

**NusaTemu** (*Nusantara* + *temu*, "find") is a dataset and analysis of objects found **by chance**: coins in a field,
statues in a well, structures under a building site. It draws on Indonesian news, colonial archaeological reports and
heritage-office records. Each find is extracted with NLP, checked against its source, and joined into one event.

**English name:** Nusantara Chance-Find Register. **Short name in files and commits:** `nusatemu`.

**Status:** 🟡 **WORKFLOW WRITTEN, NOT STARTED.** Phase 0 (design) is on paper; nothing has been harvested or run.
- Created 2026-10-20 on the PI's decision: "ide nomer satu silahkan di bikinkan workflow, kasih nama, dll.. untuk
  pelan2 kita eksekusi, untuk ide lainnya parkir dulu".
- Idea: **I-155** (`docs/IDEA_REGISTRY.md`).
- **Lines:** 05_archival_nlp (primary: NLP on texts) · 01_spatial (a discovery-process surface for site models) ·
  02_taphonomy (taphonomy of records).

## Why

The archaeological record we can see is not a sample of where people lived. It is a sample of where someone dug and
then reported what they found. In Indonesia much of what reaches the record comes from chance finds made while:
- building houses, mining sand, ploughing, or running infrastructure projects;
- and, earlier, building colonial railways, roads and plantations.

Three news items checked on 2026-10-20 show it:
- the MRT Jakarta finds (4,000+ objects, found because a tunnel was dug);
- the Lasem coin hoards (sand digging, eggplant planting);
- the Tutur coin hoard on the Bromo slope (a farmer's field, ~10 cm deep).

England has the Portable Antiquities Scheme for such finds. As far as a quick search on 2026-10-20 could tell,
**Indonesia has no systematic register of chance finds, and nobody has modelled the discovery process.** That is to be
confirmed by a proper literature check in Phase 0.

## Questions

1. **What is found by chance, where, by whom, during which activity, at what depth, and how often?** (descriptive)
2. **Does the discovery process, more than past settlement, shape where finds turn up?** For example, do finds track
   modern construction intensity and population?
3. **How has the discovery process changed from the colonial period (OV reports, 1900s–1940s) to today (news,
   2000s–2026)?**
4. **Can a chance-find intensity surface serve as the survey-effort covariate that P2 called future work?** P2 needs
   "a survey-effort surface orthogonal to terrain by construction".

These are questions about the **record**, not about the volcanic-burial thesis. This keeps NusaTemu outside the K9
stop list (no new computational test of the burial thesis without partner data), and it fits the K6 direction
"taphonomy of records" (decision due by 2026-11-15).

## Files

| File | What |
|---|---|
| `WORKFLOW.md` | Phases 0–5: steps, inputs, outputs, gates, pace |
| `CODEBOOK.md` | Record schema v0.1 and taxonomies (object, discovery activity, setting, response); event and retelling rules |
| `ETHICS.md` | Looting risk and coarse public coordinates; no finder names; copyright; legal context (UU 11/2010 Pasal 23) |
| `SOURCES.md` | Source tiers, first query list (Indonesian), access notes |

## Rules that always apply

- **P2 first.** Until P2 is uploaded (≤ 16 Nov 2026), NusaTemu takes no PI time. Claude may only do work that needs
  none.
- **G13:** the drafting slot is P23's. A NusaTemu manuscript waits for it; data work does not.
- Every analytic phase is **pre-registered** (DESIGN.md before the run) and gets its own `experiments/ENNN_*` number
  when it starts.
- **Zero cost of any kind** (G15). LLM extraction uses the cheap route already used in E225 (≤ $10 a phase), and only
  with the PI's yes.
- Never fabricate records. An extracted field must be backed by a **verbatim quote** from its source, as E225's guard
  requires.

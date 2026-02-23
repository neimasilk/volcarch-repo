# JOURNAL — Research Log

**Rule: APPEND ONLY. Never delete entries. Never edit past entries (add corrections as new entries).**

---

## 2026-02-23 | Project Genesis

**Type:** DECISION
**Author:** Amien + Claude

**Context:**
The VOLCARCH project originated from a casual observation about the Dwarapala statues of Singosari. Comparing a modern color photo with a historical B/W photo revealed that the statues were found with approximately half their 370 cm height buried underground in the 19th century, after ~510 years of volcanic sedimentation.

**Key insight:**
If volcanic activity buries artifacts at ~3.6 mm/year in the Malang basin, then remains from the Kanjuruhan era (~760 CE) could be 3.5–5 m underground, and pre-Hindu remains could be 6+ meters deep. This means the absence of archaeological evidence in volcanic Java is not evidence of absence — it is evidence of burial.

**Corollary (the "Kutai insight"):**
The oldest known kingdom in Indonesia (Kutai, ~400 CE) is in Kalimantan — a region with zero active volcanoes. Its Yupa inscriptions were found near the surface. Kutai may not be the oldest civilization in Indonesia — merely the most visible, due to differential preservation conditions.

**Decision:** Launch a computational research line to model this bias and predict where buried sites may exist.

**Dwarapala seed data (preserve for future reference):**
- Statue height: 370 cm (seated), weight ~40 tons, monolithic andesite
- Built: ~1268 CE (Kertanegara era, Singosari Kingdom 1222–1293)
- Discovered: 1803 by Nicolaus Engelhard
- Condition at discovery: "separuh tubuh terpendam" (half body buried)
- Estimated burial: ~185 cm over ~510 years = ~3.6 mm/year
- Cross-validated: Kelud eruptions deposit 2–20 cm per event at Malang distance; ~20 eruptions in 510 years plausibly accounts for ~100 cm; remainder from Semeru, Arjuno, alluvial processes
- Sources: BPCB Jawa Timur (kebudayaan.kemdikbud.go.id), Detik Travel, GVP Smithsonian, Wearemania.net, MalangTimes

---

## 2026-02-23 | Repo Structure Decision

**Type:** DECISION
**Author:** Amien + Claude

**Decision:** Use 3-layer PRD structure + append-only journal.
- L1 (Constitution): core hypotheses, philosophy — rarely changes
- L2 (Strategy): current phase, active papers — changes per quarter
- L3 (Execution): active tasks, experiments — changes per week
- Journal: log everything, delete nothing

**Rationale:** Research is non-linear. Unlike software PRDs, research PRDs must accommodate failure, pivoting, and revisiting. The layered approach separates stable foundations from volatile execution details, allowing Claude Code to always understand context at the right level of abstraction.

---

*Add new entries below. Use format: `## YYYY-MM-DD | Title`*
*Types: DECISION, EXPERIMENT, RESULT, FAILURE, PIVOT, INSIGHT, TODO*

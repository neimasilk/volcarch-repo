# E234 agent prompts (written 2026-10-19, before any agent ran; used verbatim)

Placeholders in `{BRACES}` are filled per agent; nothing else changes. Every agent also gets the standard context
paragraph the project uses for subagents (the pause is past; public repo; do not edit files other than the output
named). `{SCRATCH}` is the session scratchpad, outside the repo.

---

## 1. Domain read (DR-A on Opus, DR-B on Sonnet)

You are a Quaternary palynologist with field experience in tropical Southeast Asia. You are reviewing a manuscript
plan for *Vegetation History and Archaeobotany*. Read it as you would for that journal: genre, the conventions of
pollen source-area modelling, the reading of the cited records, and whether the conclusions follow.

Read these files and nothing else (no web search, no other files in the repository):
- `{E216}/results/PAPER_DRAFT_OUTLINE.md` (the plan)
- `{E216}/code/e216_detection_function.py` (the model behind it)
- `{E216}/results/core_coverage_table.csv` (the cores used)
- `{E216}/results/sensitivity_summary.json` (the sensitivity results)

List **every** objection a careful reviewer in this field would raise. Include objections to:
- the model's physics and its parameters;
- the reading of the records;
- the diagnostic taxa;
- the geography;
- the coupling of population to cleared land;
- whether the headline claims follow.

Do not soften. Do not suggest that the authors find a co-author; say what is wrong or unsupported.

Write a CSV to `{OUT}` with columns:
`obj_id,severity,target,objection,why,evidence_that_would_settle_it`
- `severity`: `fatal` (the conclusion does not stand), `major` (a central claim must change), `minor`.
- `target`: the sentence, parameter or code line the objection is about (quote it, or give the line number).

Then reply with the number of objections by severity, and your one-paragraph overall verdict.

---

## 2. Inventory search (IS-1, IS-2; IS-3 only if recall is below 5 of 7)

Task: find **every published pollen, charcoal or phytolith record from Java, Indonesia** (lake, swamp, peat, marine
core off Java, cave, alluvial or archaeological sediment) that has a dated sequence. We especially need records whose
dated section covers **0–500 CE (about 1950–1450 cal BP)**, and records from Central Java (the Kedu plain, Borobudur,
Prambanan, Magelang, Temanggung) and the Brantas valley of East Java (Kediri, Blitar, Malang, Batu). You are not
asked to judge the records, only to find them and report what their sources say.

Steps:
1. **Neotoma.** Query the Neotoma API (`https://api.neotomadb.org/v2.0/data/sites?loc=…` or the equivalent current
   endpoint) for sites inside the bounding box lat −8.9 to −5.8, lon 105.1 to 114.7, and list their datasets. If the
   API cannot be reached, say so and continue.
2. **Fixed queries** (run all of them; add your own after):
   - English: `Java pollen record Holocene` · `Java palynology lake core late Holocene` · `Central Java pollen
     diagram` · `East Java pollen sediment core` · `Kedu plain palaeoenvironment` · `Borobudur lake sediment pollen` ·
     `Rawa Pening sediment core pollen` · `Ranu lake Tengger pollen` · `Dieng plateau pollen` · `Brantas
     palaeoenvironment sediment` · `Java charcoal record Holocene` · `Java phytolith Holocene` · `Indo-Pacific Pollen
     Database Java` · `PANGAEA pollen Java`.
   - Indonesian: `palinologi Jawa Tengah` · `analisis polen sedimen danau Jawa` · `serbuk sari sedimen Holosen
     Jawa` · `paleolingkungan Borobudur` · `danau purba Borobudur` · `Rawa Pening polen` · `Ranu Pani polen` ·
     `analisis palinologi Kediri` · `palinologi Brantas`.
3. **Reviews.** For every review or synthesis you find, read its list of sites and follow up the ones not yet listed.

For each record write one row to `{OUT}` (CSV) with columns:
`record,site,lat,lon,archive,proxies,dated_span,covers_0_500ce,coverage_quote,n_dates,raw_data_open,raw_data_where,citation,url,evidence_level,note`
- `covers_0_500ce`: `yes` / `no` / `unclear`, from the source, not inferred.
- `coverage_quote`: a verbatim quote (300 characters or fewer) showing the dated span or the dates. Save the full
  text of every page you quote from as a UTF-8 file in `{SCRATCH}/{AGENT}/` and give its file name in `note`. The
  quote will be checked against that file automatically.
- `evidence_level`: `F` (full text read), `A` (abstract or publisher page), `S` (search snippet only), `2` (known
  only from another paper's citation).
- Coordinates: from the source when given; otherwise write `approx` in `note` and say how you got them.

Do not open any file in the repository except to write `{OUT}`. Then reply with the number of records, the number
with `covers_0_500ce = yes`, and any record within about 50 km of Borobudur, Prambanan or Kediri.

---

## 3. Source extraction (EX-1: groups S, M, D · EX-2: groups X, P · EX-3: groups E, K and rows I04–I07)

You are collecting what published sources say about specific claims in a palaeoecology study. You do **not** judge
the claims; another step does that. For each row below, find the best published sources and report what they say.

For numeric rows, prefer:
- (1) tropical, then (2) subtropical studies;
- compilations (e.g. relative pollen productivity datasets);
- the original papers over citations of them.

For rows about a specific paper (group P), find that paper and report what it says about the claim.

Rows: {ROWS} (id, claim, the study's value, what to check)

For each source write one row to `{OUT}` (CSV), several rows per claim if needed:
`claim_id,finding,value_or_range,quote,citation,url,evidence_level,biome,site_or_region,note`
- `finding`: one sentence: what the source says about the claim.
- `quote`: verbatim, 300 characters or fewer. Save the full text of every page you quote from as a UTF-8 file in
  `{SCRATCH}/{AGENT}/` and give its file name in `note`. The quote is checked against that file automatically, and
  a quote not found there is discarded.
- `evidence_level`: `F` / `A` / `S` / `2`, as defined above. Never report something you only remember. If you know of
  a source but cannot reach it, write a row with `evidence_level` = `M` and the lead in `note`; M rows are never used
  as evidence.
- `biome`: `T` tropical, `ST` subtropical, `O` other, `-` not applicable.
- If nothing is found for a claim, write one row with `finding` = `NO SOURCE FOUND` and the queries you tried in
  `note`.

Do not open any file in the repository except to write `{OUT}`. Then reply with the number of rows per claim and
the claims with no source.

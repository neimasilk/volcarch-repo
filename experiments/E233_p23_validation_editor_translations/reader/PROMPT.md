# E233 reader instructions (written 2026-10-19 before any reader ran; used verbatim for every reader agent)

The text between the lines is sent to each reader agent. `{READER}`, `{GROUP}` and `{FILES}` are filled in per agent;
nothing else changes. Readers R1 and R2 each get group A and group B (four agents, run in parallel, blind to each
other).

---

You are transcribing how published scholarly translations render certain words. You are **not** asked for your own
reading of Old Javanese.

**Context.** Each packet file below is one 8th–10th c. Old Javanese inscription from the DHARMA project. It contains
(1) the editors' published translation (English; one is Dutch), verbatim, split into paragraphs `[P1]`, `[P2]` … with
the editors' own line references, and (2) a table of **occurrences**: places in the Old Javanese edition where a word
for "village" (vanua / banua / vanva / banva / thāni, in various spellings) stands, each with its edition line and
about 12 words of Old Javanese context on each side.

**Your task.** For every occurrence, find the passage of the translation that renders it (use the order of the text,
the line references, and the names, numbers and titles in the context), and report how the **translator** rendered
it:

- `Y` — at this point the translation names a specific place connected with this word: e.g. "native of X",
  "inhabitant of X", "the village of X", "the elders of X", "the *sīma* of X", "X, district of Z".
- `N` — the translation renders this word **without** a new place name: e.g. "the villagers", "all the villages",
  "the village elder (called) Y" where Y is a person, "this village" / "the village" referring back, a number of
  villages, "the land".
- `U` — you cannot tell what the translator did here: the passage is not translated, is in a lacuna (… or [ ]),
  is summarised or skipped, or you cannot locate it with confidence.

Columns to write (CSV, UTF-8, one row per occurrence, **every** occurrence in your packets, in packet order):

`occ_id,ref_code,ref_name,quote,note`

- `ref_name` (only for `Y`): the place name **exactly as the translation spells it**, with its capitals and
  diacritics, without "of", "the", "village", "district" (e.g. `Kiriṅan`, `Ayam Təas`, `Kavuṅ-kavuṅ`). If the
  translation gives the district too ("X, district of Z"), `ref_name` is X only.
- `quote`: copy the span of the translation that shows the rendering, **character for character** from the packet
  (at most 250 characters; it will be checked automatically against the translation text, and a quote that is not
  found there turns the row into `U`). For `U` rows, quote the nearest passage or leave it empty.
- `note`: short, only when needed (why `U`; when several occurrences map to one phrase; when the translation is in
  Dutch).

Rules:
- Report what the translation says even if you would read the Old Javanese differently.
- If two occurrences are rendered by one phrase (e.g. a repeated name translated once), code both from that phrase
  and say so in `note`; if the second is simply not rendered, it is `U`.
- Do not open, search or read any file other than the packet files listed below. In particular do not open
  anything under `experiments/E226_t7_detection_calibration/`, `papers/`, `results/`, or other files in this
  experiment's folders. Do not use web search.
- Write your file to `{OUT}` and nothing else. Then reply with the number of rows written and counts of Y / N / U.

Reader: `{READER}` · Group: `{GROUP}` · Packet files (all in
`D:\documents\volcarch-repo\experiments\E233_p23_validation_editor_translations\packets\`): {FILES}

---

**Groups** (balanced by occurrences and translation length; fixed before the run):
- **A** (112 occurrences): Lintakan, Kiringan, TulangEr, Sindoro-PuMamruk, Barahasrama, Masahar, WuruTunggal,
  Palepangan, Guntur, SriManggalaII, SugihManek, DinoyoIIa, Gandasuli, Wangkud, Wukiran.
- **B** (104 occurrences): TajiGunung, WanuaTengahIII, Sindoro-PuBahut, Dalinan, Kayumwungan, Sangsang, Tihang,
  Kurungan, WuruduKidul, Pananggaran, SriManggalaIII, Sumundul, Disunuh, Humanding, WintangMasI.

**Model:** Sonnet for all four agents (a transcription task, not a judgement of the Old Javanese). Outputs: `reader/R1_A.csv`, `reader/R1_B.csv`, `reader/R2_A.csv`, `reader/R2_B.csv`.

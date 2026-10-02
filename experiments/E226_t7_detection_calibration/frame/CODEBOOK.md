# E226 village-frame codebook (coders A and B, independent)

**Input:** `frame/coder_input.csv` — one row per occurrence of a village noun (vanua / banua / vanva / banva /
thāni family) in the *edition* text of an 8th–10th c. CE DHARMA inscription, with ±12 tokens of context.
DHARMA transliteration conventions: capital `A`, `I`, `U` mark independent vowels (so `Anak vanuA I X` =
*anak wanua i X*, "resident of the village of X"); `v` = *w*; `ṅ` = *ng*; `ḥ` final *h*; `-` inside a word
is a line-join artefact; `*`, `_`, `eccentric_ductus` are editorial noise.

**Question per row:** does this occurrence of the village noun name a **specific, individual village**?

## Code `village_named`

- **Y** — the noun is followed (directly or after a locative particle *i / ri / riṅ / iṅ / ni*) by a proper
  name of a place, e.g. `Anak vanuA I kasugihan` → *kasugihan*; `vanva I ḍisunuḥ` → *ḍisunuḥ*;
  `vanuA Iṁ kuniṁ` → *kuniṁ*; `vanuA sĭma I Ayam təAs` → *ayam təAs* (the sīma village).
- **N** — generic or no name: plural/collective (`anak vanuA kabaiḥ` "all villagers"), counting
  (`sapuluḥ vanuA` "ten villages"), `tuha banua` / `rāma` titles followed by a *person* (`tuha banua si mahi`),
  `vanuA` as "land/territory" without a name, `sapasug banuA`, a false lexical match (e.g. *baṅun* "build"),
  possessive references back to an already-named village with no new name (`vanuAnya`, `ikanaṁ vanuA`).
- **?** — a name may be present but the segmentation is genuinely unclear (broken text, lacuna, the token after
  the particle could be a title or a person). Use sparingly; explain in `note`.

## Code `name` (only when Y or ?)

- Write the village name **as it appears**, minimal normalisation: lowercase, keep diacritics, drop the
  particle, drop line-join hyphens (`bāku- l` → `bākul`), keep reduplication hyphens (`viru-viru`).
- Multi-word names are allowed when the context clearly makes them one name (`ayam təAs`, `ra tguḥ`); stop
  before `vatak/vatək/vatek` (the district), before a title (*pu, saṁ, si, ḍaṁ, rake, juru, parujar,
  maṁraṅkapi/Aṁraṅkəpi, kapuA, InaṁsəAn, tatra*), or before the next person.
- Do **not** merge spelling variants across rows yourself; the adjudicator does that.

## Code `watak`

If the village is followed by `vatak/vatək/vatek X`, write X (the *watak*/district). Otherwise blank.

## Code `role` (Y rows only)

- `sima` — the village being granted / made sīma, or whose land is the subject of the grant.
- `witness` — a resident of the village appears as witness/recipient of gifts (`Anak vanuA I X` in the
  witness list).
- `boundary` — a neighbouring village named in a boundary or *tpi siriṁ* list.
- `other` — anything else (explain).

## Rules

- Code every row; do not skip. Work from the context given; do not consult the other coder's file or any
  village list. You may use general knowledge of Old Javanese sīma formulae; you may not look up
  modern place names (that is a later step).
- Output `frame/coder_<A|B>.csv` with columns: `occ_id, village_named, name, watak, role, note`.

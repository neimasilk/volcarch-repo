# NusaTemu — ethics, law and data handling

**1. Looting risk (the main one).** A public map of where coins, statues and hoards turn up is a treasure map.
- **Public releases** (repo, Zenodo, figures) carry **coarse locations only**: kecamatan centroid, or a 5 km grid at
  most. Coordinates are never finer than the source already makes public.
- **Precise locations** (desa, coordinates) live only in a private file **outside this public repository**.
- **No live "new finds" feed.** Releases are periodic and lagged (proposal: finds older than 12 months).

**2. People.** Finders are private individuals. **No names** are stored, even when news prints them: only
`finder_role`. The same applies to officials quoted in news, except institutions (BPK Wilayah XI etc.).

**3. Copyright.** News text belongs to its publisher. The repo stores:
- URL, outlet, date;
- the extracted fields;
- **short verbatim quotes** (≤ 30 words each) needed for verification.

Full article texts, if cached for extraction, stay outside the repo (session or local cache) and are not redistributed.

**4. Legal context (Indonesia).** UU No. 11/2010 tentang Cagar Budaya, **Pasal 23**:
- (1) whoever finds an object, building, structure or site suspected to be cultural heritage must report it to the
  culture office, the police and/or the relevant agency **within 30 days**;
- (2) finds not reported can be taken over by the government;
- (3) the culture office studies a reported find.

Text checked 2026-10-20 at pasal.id and a JDIH copy of the law; the official gazette was not read. NusaTemu records
`response` (reported, surveyed, kept, sold) as data. It **does not judge** individual finders, and it never publishes a
record in a way that identifies a finder who kept or sold a find.

**5. Use of AI.** Extraction and coding use LLMs with a verbatim-quote guard. Every phase reports its AI use and its
audited error rates; papers carry the disclosure (`docs/AI_DISCLOSURE_TEMPLATE.md`). Prose is the PI's (G16; JCAA-type
policies forbid AI ghost-writing).

**6. Who benefits.** Offer the cleaned register (with precise locations) to the BPK offices and BRIN if they want it.
They are the institutions that act on finds. The public version serves research.

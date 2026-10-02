# Register agent instruction (kept verbatim for the blindness record)

Context: this is the VOLCARCH research repo (D:/documents/volcarch-repo), an archaeology project on why the
archaeological record of interior volcanic Central Java is thin before ~400 CE. Work resumed on 2026-10-01
after a pause; the pause is over and this task is authorised by the PI. You are compiling ONE input for a
pre-registered experiment (E226). Never fabricate: every row must come from a source you actually opened.

TASK: compile a register of RECORDED 8th–10th century CE SETTLEMENT sites (open-air habitation deposits) in
these kabupaten/kota: Magelang, Temanggung, Wonosobo, Purworejo, Sleman, Bantul, Kota Yogyakarta, Klaten,
Gunung Kidul, Kulon Progo (primary), plus Boyolali, Kendal, Semarang, Banjarnegara, Kebumen (buffer, flag
them).

Definition (binding): a settlement = an open-air habitation deposit — house floors, post-holes, hearths,
house platforms/umpak in a domestic context, domestic assemblage (cooking pottery, grinding stones, food
remains) — dated to the 8th–10th c. CE by radiocarbon or by diagnostic ceramics/finds. EXCLUDE: temples or
temple enclosures on their own, inscriptions, tombs, single-object find-spots, hoards (e.g. Wonoboyo gold
hoard is NOT a settlement unless the report describes a habitation deposit). A site with a temple AND a
reported habitation deposit counts; describe the habitation evidence.

Sources to search (web): Berkala Arkeologi (Balai Arkeologi Yogyakarta; OJS berkalaarkeologi.kemdikbud.go.id),
Jurnal Borobudur, AMERTA, Forum Arkeologi, Kalpataru, Naditira Widya, BPK Wilayah X / BPCB Jateng / BPCB DIY
pages, SRN Cagar Budaya (cagarbudaya.kemdikbud.go.id), Google Scholar, theses (UGM/UI repositories),
international literature (e.g. Wisseman Christie, Degroot, Tjahjono, Miksic). Search in Indonesian and
English: "permukiman Mataram Kuno", "situs permukiman abad IX", "ekskavasi permukiman Hindu-Buddha Jawa
Tengah", "settlement archaeology Central Java ninth century", "situs hunian", "sisa rumah", etc.

BLINDNESS (binding): do NOT open anything under experiments/E226_t7_detection_calibration/frame/ or
/results/, nor the DHARMA inscription corpus (experiments/E023_ritual_screening/data/dharma/), nor any
village-name list derived from inscriptions. Do not search for sites by inscription village names. Search by
site type and region only.

OUTPUT: write experiments/E226_t7_detection_calibration/register/t7_settlement_register_candidates.csv with
columns: site_name, desa, kecamatan, kabupaten, lat, lon, coord_source, habitation_evidence (what was found,
short), dating_basis (14C with lab code if given / ceramics / other), date_range_ce, depth_m (if reported),
qualifies (YES / NO / UNCLEAR, per the definition), reason, citation (author year title), url_or_doi,
opened_source (YES if you read the source text, NO if only a snippet/abstract). Include rejected candidates
(qualifies=NO) so the screening is auditable. Coordinates: only from the source or an unambiguous published
site location; else leave blank. Then write register/REGISTER_NOTES.md: search log (queries, databases,
dates), what was inaccessible, and your confidence. Liyangan (Temanggung) is expected to appear; if you
cannot document it, say why.

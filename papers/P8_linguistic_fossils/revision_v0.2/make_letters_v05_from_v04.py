"""Letters v0.4 -> v0.5 (2026-10-12): the statements that the article v0.4 changes, and nothing else."""
import sys
from pathlib import Path
D = Path(__file__).resolve().parent

def apply(src_name, out_name, edits):
    s = (D / src_name).read_text(encoding="utf-8")
    bad = 0
    for old, new in edits:
        k = s.count(old)
        if k != 1:
            bad += 1; print(f"MISMATCH ({k}) in {src_name}: {old[:90]}")
        s = s.replace(old, new)
    if bad:
        sys.exit("nothing written")
    (D / out_name).write_text(s, encoding="utf-8", newline="\n")
    print(f"{out_name}: {len(edits)} edits")

R = [
 # R1-1: the article's wording as it now stands (the journal closes up "protoform")
 ('The wording is now "cannot be assigned to a higher-level proto-form" (first sentence of Section 1);',
  'The wording is now "cannot be assigned to a higher-level protoform" (first sentence of Section 1);'),
 # R1-2: the page of the thesis, as in the article
 ("the substratum raised and left open (Bulbeck 1992), and the 62 percent", "the substratum raised and left open (Bulbeck 1992:512–13), and the 62 percent"),
 # R1-2: 'one fact' was too strong for the shared cognate sets (E231 P1)
 ("What distinguishes Makasar from its relatives in these data is one fact — more uncoded meanings (exact McNemar p = 0.006 against Bugis, 0.0001 against Sa'dan Toraja). Its lower share in the PMP set and its smaller share of common sets with each relative are that fact counted again, because on the meanings coded in both lists the shares are equal (70 against 69 of 110; 70 against 71 of 115).",
  "What distinguishes Makasar from its relatives in these data is mainly one fact — more uncoded meanings (exact McNemar p = 0.006 against Bugis, 0.0001 against Sa'dan Toraja). Its lower share in the PMP set is that fact counted again: on the meanings coded in both lists the shares are practically equal (70 against 69 of 110; 70 against 71 of 115). Its smaller share of common sets with each relative is that fact in part only: on the meanings coded in both lists the shares are 76.3 percent with Bugis and 70.1 with Sa'dan Toraja, against 81.0 between Bugis and Sa'dan Toraja (no test was made of that difference)."),
 # R1-2: the low look-alike rate is not particular to Makasar (E231 P9)
 ("and a Bugis or Sa'dan Toraja look-alike for 10 of 80 (chance 3.8 percent; 90 of 136 among coded forms). The screen therefore does not favour",
  "and a Bugis or Sa'dan Toraja look-alike for 10 of 80 (chance 3.8 percent; 90 of 136 among coded forms); the uncoded forms of Bugis and of Sa'dan Toraja resemble their relatives' forms at similar rates (8 of 62 and 8 of 45). The screen therefore does not favour"),
 # R1-5: the pair count uses the plain distance
 ("four look-alike pairs against 0.5 expected;", "four pairs within 0.34 against 0.5 expected;"),
 # R1-6: the list name as ABVD writes it
 ('''(list 226, "Tae' (S. Toraja)", from van der Veen 1940)''', '''(list 226, "Tae' (S.Toraja)", from van der Veen 1940)'''),
 # R1-7 and R1-13: 88.4 percent is a look-alike rate with a chance rate
 ("and 88.4 percent of its uncoded forms whose meaning the five Tolaki dialect lists cover recur in those lists.",
  "and 88.4 percent of its uncoded forms whose meaning the five Tolaki dialect lists cover have a look-alike in those lists (7.5 percent by chance)."),
 ("of its 86 uncoded forms whose meaning the five Tolaki dialect lists also cover, 88.4 percent recur in those lists;",
  "of its 86 uncoded forms whose meaning the five Tolaki dialect lists also cover, 88.4 percent have a look-alike in those lists (7.5 percent by chance);"),
 # R1-13a: six, not seven (E228 04_pbt_matched_entry_recount.py)
 ("of the 34 uncoded Tolaki forms that match an entry of the Proto-Bungku-Tolaki list, seven match an entry that ABVD does assign to a cognate set,",
  "of the 34 uncoded Tolaki forms that match an entry of the Proto-Bungku-Tolaki list, six match an entry that ABVD does assign to a cognate set,"),
 # R1-13b: what the released columns hold
 ("three out-of-sample scores, the nearest look-alikes)", "three out-of-sample scores, the nearest same-meaning forms with their distances)"),
 # R1-11: the last column of Table 7
 ("the PMP entry of the meaning and the nearest same-meaning form in another Sulawesi list by the screen. The text states",
  "the PMP entry of the meaning and the nearest same-meaning form in another ABVD list from the Sulawesi region, with its distance (above 0.34 it is not a look-alike by the screen). The text states"),
 # R2-21: 'One Hundred'
 ("'One Hundred' (uncoded in four lists, coded in Bugis) is a different case: its forms are not uniform across the lists, the text says only that it does not belong with the compounds, and we give no account of it; the forms are in the released file.",
  "'One Hundred' is likewise uncoded in four lists and coded in Bugis; the text prints its five forms as ABVD records them and gives no account of them."),
 # Part C, references: what concerned the reference list of the submitted text; the page range is now confirmed at the source
 ("Submitted: Ross 2005 given as an article in this journal (it is a book chapter); two entries in our bibliography file for works that do not exist (neither was cited); seven further entries with a wrong entry type, publisher or pages. Revised: every entry of the reference list was checked against the publication, its catalogue record or, for a few page ranges, the reference list of another work; one page range that could not be confirmed is omitted; Ross 2005 is no longer cited and the two entries are deleted.",
  "Submitted: Ross 2005 given as an article in this journal (it is a book chapter); six further entries of the reference list with a wrong entry type, publisher or pages. Revised: every entry of the reference list was checked against the publication, its catalogue record or, for a few page ranges, the reference list of another work; Ross 2005 is no longer cited; the list is set in the journal's style."),
]
apply("RESPONSE_TO_REVIEWERS_v0.4.md", "RESPONSE_TO_REVIEWERS_v0.5.md", R)

C = [
 ("We have tried to answer every one of them.\n\nAs we wrote in our note",
  "We have tried to answer every one of them. As the editors asked, particular care has gone into the readability of Section 3: its terms are explained in words, and the internal experiment codes are gone.\n\nAs we wrote in our note"),
 ("are withheld in it. It is 24 pages in the journal's template, with eight tables. A version with names is available on request.",
  "are withheld in it, and the acknowledgments are left out, as the instructions for contributors ask. It is 24 pages in the journal's template, with eight tables; the text, the tables and the reference list follow the template and the journal's style sheet. A version with names is available on request."),
]
apply("COVER_LETTER_v0.4.md", "COVER_LETTER_v0.5.md", C)

"""v0.3 -> v0.4 of the article source: house style of Oceanic Linguistics only (no number, claim or sentence of content changes).
Each replacement must match exactly the expected number of times, otherwise nothing is written."""
import re, sys
from pathlib import Path

D = Path(__file__).resolve().parent
src = (D / "P8_revision_v0.3.md").read_text(encoding="utf-8")

E = []  # (old, new, expected count, rule)
def e(old, new, n=1, rule=""):
    E.append((old, new, n, rule))

# ---- serial comma (template: "OL style is to use the serial comma")
SC = "serial comma"
e("more often carry a written glottal mark and more often denote actions.", "more often carry a written glottal mark, and more often denote actions.", 1, SC)
e("radical sound change and coinage,", "radical sound change, and coinage,", 1, SC)
e("Muna-Buton and Wotu-Wolio but not", "Muna-Buton, and Wotu-Wolio but not", 1, SC)
e("Wotu-Wolio (Wolio) and Bungku-Tolaki (Tolaki).", "Wotu-Wolio (Wolio), and Bungku-Tolaki (Tolaki).", 1, SC)
e("the data, the label, the inputs and the tests;", "the data, the label, the inputs, and the tests;", 1, SC)
e("dictionary of 1940) and Tolaki (674;", "dictionary of 1940), and Tolaki (674;", 1, SC)
e("leading and trailing punctuation and stray spaces were removed", "leading and trailing punctuation, and stray spaces were removed", 1, SC)
e("(hyphens, spaces and glottal marks included); the number", "(hyphens, spaces, and glottal marks included); the number", 1, SC)
e("*ma, me, mo, pa, ka, ta, na, po* or *aŋ*;", "*ma, me, mo, pa, ka, ta, na, po*, or *aŋ*;", 1, SC)
e("of an onset string, of a nasal input and of a hyphen-or-repetition input,", "of an onset string, of a nasal input, and of a hyphen-or-repetition input,", 1, SC)
e("and the precision, recall and F1 score of the uncoded class", "and the precision, recall, and F1 score of the uncoded class", 1, SC)
e("Sa'dan Toraja, Wolio, Muna and Tolaki.", "Sa'dan Toraja, Wolio, Muna, and Tolaki.", 1, SC)
e("hyphens, apostrophes and the glottal mark have been removed", "hyphens, apostrophes, and the glottal mark have been removed", 1, SC)
e("the look-alike screens, this test and the spelling tests below", "the look-alike screens, this test, and the spelling tests below", 1, SC)
e("the Muna digraphs *gh*, *bh* and *dh* were converted", "the Muna digraphs *gh*, *bh*, and *dh* were converted", 1, SC)
e("39.3, 25.9 and 34.8 percent of 201 meanings; for Bugis 48.3, 28.4 and 23.4; for Sa'dan Toraja 48.0, 33.0 and 19.0.",
  "39.3, 25.9, and 34.8 percent of 201 meanings; for Bugis 48.3, 28.4, and 23.4; for Sa'dan Toraja 48.0, 33.0, and 19.0.", 1, SC)
e("Makasar 38, Sundanese 35 and Yogyakarta Javanese 30.", "Makasar 38, Sundanese 35, and Yogyakarta Javanese 30.", 1, SC)
e("gives 41.4, 39.2 and 53.1 percent for the same pairs", "gives 41.4, 39.2, and 53.1 percent for the same pairs", 1, SC)
e("whether substratum, borrowing or innovation.", "whether substratum, borrowing, or innovation.", 1, SC)
e("a coding gap, a chance resemblance and a loan.", "a coding gap, a chance resemblance, and a loan.", 1, SC)
e("Precision, recall and F1 are those of the uncoded class.", "Precision, recall, and F1 are those of the uncoded class.", 1, SC)
e("0.668 (Sa'dan Toraja) and 0.854 (Tolaki),", "0.668 (Sa'dan Toraja), and 0.854 (Tolaki),", 1, SC)
e("25.9 percent of the forms ending in another consonant and 39.3 percent", "25.9 percent of the forms ending in another consonant, and 39.3 percent", 1, SC)
e("the vowel share (0.210) and consonant-letter clusters (0.198).", "the vowel share (0.210), and consonant-letter clusters (0.198).", 1, SC)
e("with their set number, the PMP entry of the meaning and the nearest same-meaning form", "with their set number, the PMP entry of the meaning, and the nearest same-meaning form", 1, SC)
e("'Twenty', 'to stand' and 'to hit')", "'Twenty', 'to stand', and 'to hit')", 1, SC)
e("four Wolio forms (two uncoded, two coded) and no Muna form.", "four Wolio forms (two uncoded, two coded), and no Muna form.", 1, SC)
e("and of *gh*, *bh* and *dh* in Muna to single symbols", "and of *gh*, *bh*, and *dh* in Muna to single symbols", 1, SC)
e("begin with *ma-*, *mo-* or *me-* in several lists", "begin with *ma-*, *mo-*, or *me-* in several lists", 1, SC)
e("are long, more often glottal-marked and more often action words (Table 5),", "are long, more often glottal-marked, and more often action words (table 5),", 1, SC + " + lower-case table")
e("reported by Mills (1975), Sirk (1989) and Bulbeck,", "reported by Mills (1975), Sirk (1989), and Bulbeck,", 1, SC)
e("in four lists, in four forms of a fifth and not at all in the sixth,", "in four lists, in four forms of a fifth, and not at all in the sixth,", 1, SC)
e("mixes inherited forms, small local sets and flagged loans,", "mixes inherited forms, small local sets, and flagged loans,", 1, SC)
e("the look-alike screens, the permutation test and the spelling tests had", "the look-alike screens, the permutation test, and the spelling tests had", 1, SC)
e("a correlated bundle of length, a written glottal mark and an action meaning;", "a correlated bundle of length, a written glottal mark, and an action meaning;", 1, SC)
e("classification, the look-alike screen and the permutation test;", "classification, the look-alike screen, and the permutation test;", 1, SC)

# ---- numbers (template: single-digit numbers spelled out, numerals for all others; CMS 9.6 alternative rule)
NU = "numbers"
e("lists eleven linguistic microgroups", "lists 11 linguistic microgroups", 1, NU)
e("occasionally; eleven forms in these six lists", "occasionally; 11 forms in these six lists", 1, NU)
e("and five of the eleven also carry", "and five of the 11 also carry", 1, NU)
e("Twenty-five inputs are computed for every form, seventeen from the written form and eight from the meaning.",
  "For every form 25 inputs are computed, 17 from the written form and 8 from the meaning.", 1, NU + " (sentence recast so that it does not open with a numeral)")
e("is repeated ten times with different divisions, giving fifty test sets.", "is repeated 10 times with different divisions, giving 50 test sets.", 1, NU)
e("over the ten repetition means and over the fifty test sets;", "over the 10 repetition means and over the 50 test sets;", 1, NU)
e("(which average the ten repetitions and so run", "(which average the 10 repetitions and so run", 1, NU)
e("one of the twelve is loan-flagged", "one of the 12 is loan-flagged", 1, NU)
e("the twelve PMP look-alikes are listed", "the 12 PMP look-alikes are listed", 1, NU)
e("over the ten repetition means is 0.007 and over the fifty test sets 0.027;", "over the 10 repetition means is 0.007 and over the 50 test sets 0.027;", 1, NU)
e("Note. Five folds, ten repetitions, fifty test sets.", "Note. Five folds, 10 repetitions, 50 test sets.", 1, NU)
e("one bundle, not ten separate findings.", "one bundle, not 10 separate findings.", 1, NU)
e("(five-fold cross-validation, ten repetitions) and averaged over the ten repetitions,", "(five-fold cross-validation, 10 repetitions) and averaged over the 10 repetitions,", 1, NU)
e("for two of those eleven meanings", "for two of those 11 meanings", 1, NU)
e("a mean over twenty meanings", "a mean over 20 meanings", 1, NU)
e("applied the model to sixteen further ABVD lists", "applied the model to 16 further ABVD lists", 1, NU)
e("the uncoded forms of those sixteen lists", "the uncoded forms of those 16 lists", 1, NU)
e("0.60 or more in ten of the sixteen)", "0.60 or more in 10 of the 16)", 1, NU)
e("and twelve resemble the PMP entry.", "and 12 resemble the PMP entry.", 1, NU)
e("and ten properties were tested without adjustment.", "and 10 properties were tested without adjustment.", 1, NU)
e("More than twenty p-values were computed", "More than 20 p-values were computed", 1, NU)
e("the twelve Tolaki look-alikes of PMP entries;", "the 12 Tolaki look-alikes of PMP entries;", 1, NU)

# ---- references to sections and tables in lower case inside a sentence (template: "table 2 is an example, not Table 2"; "in section 2")
LC = "lower-case cross-reference"
e("carry information about it (Section 3.5).", "carry information about it (section 3.5).", 1, LC)
e("ABVD's second Muna list (Section 2.2).", "ABVD's second Muna list (section 2.2).", 1, LC)
e("written before the run (Section 2.6).", "written before the run (section 2.6).", 1, LC)

# ---- prefixes closed up (template, "Hyphen": noninitial, preempt, reexamine ...)
HY = "prefix closed up"
e("was re-coded in six ways", "was recoded in six ways", 1, HY)
e("each time re-running the cross-validation", "each time rerunning the cross-validation", 1, HY)
e("under each re-coding of the glottal mark", "under each recoding of the glottal mark", 1, HY)
e("so that a pre-glottalized consonant becomes a geminate", "so that a preglottalized consonant becomes a geminate", 1, HY)
e("the re-derivation of every number", "the rederivation of every number", 1, HY)
e("prompted part of the re-derivation reported here", "prompted part of the rederivation reported here", 1, HY)
e("The analysis scripts, the pre-stated decision rules and the result files are in",
  "The analysis scripts, the decision rules written before the runs, and the result files are in", 1, HY + " + serial comma (reworded with the paper's own phrase)")

# ---- comma inside closing double quotes (template: "Double quotes are put outside ... periods, commas")
QU = "comma inside double quotes"
e('("nasal cluster", "reduplication", "prefix-like")', '("nasal cluster," "reduplication," "prefix-like")', 1, QU)
e('''ABVD's list "Tae' (S. Toraja)", entered by Blust''', '''ABVD's list "Tae' (S. Toraja)," entered by Blust''', 1, QU)

out = src
bad = 0
for old, new, n, rule in E:
    k = out.count(old)
    if k != n:
        bad += 1
        print(f"MISMATCH ({k} found, {n} expected): {old[:90]}")
    out = out.replace(old, new)
if bad:
    sys.exit(f"{bad} replacement(s) did not match; nothing written")

# =====================================================================================================================
# Stage 2 — corrections of content after the independent adversarial read and the reference check of 2026-10-12.
# Every one narrows, qualifies or documents a statement; none adds a claim. The `old` strings are the text AFTER stage 1.
# =====================================================================================================================
C = []
def c(old, new, why, n=1):
    C.append((old, new, n, why))

c("In the Austronesian Basic Vocabulary Database (ABVD), part of every Sulawesi word list carries no cognate-set assignment, often read as substrate.",
  "In the Austronesian Basic Vocabulary Database (ABVD), part of every Sulawesi word list examined here carries no cognate-set assignment; residues of this kind have been read as substrate.",
  "abstract: 'every Sulawesi word list' and 'often read as substrate' said more than the paper shows or cites")
c("Lexicostatistical counts have long shown Makasar, with Konjo and Selayar, as the most divergent part of the South Sulawesi group (Mills 1975:491; Sirk 1989:71); whether a substratum lies behind the divergence has been raised and left open (Bulbeck 1992);",
  "Lexicostatistical counts have long set Makasar apart from the rest of the South Sulawesi group (Mills 1975:491), together with Konjo and Selayar in the counts of Sirk (1989:71); whether a substratum lies behind the divergence has been raised and left open (Bulbeck 1992:512–13);",
  "Mills's counts have Makasar only (Konjo and Selayar are Sirk's); 'the most divergent part' is not either source's wording; page added for the thesis")
c("Muna (ABVD list 27; 219 forms; compiled from van den Berg)", "Muna (ABVD list 27; 219 forms; René van den Berg)",
  "ABVD names van den Berg as the author of the list, as it does the compilers of the other lists")
c('''ABVD's list "Tae' (S. Toraja)," entered by Blust''', '''ABVD's list "Tae' (S.Toraja)," entered by Blust''', "the list name as ABVD writes it")
c("Every count in this paper is therefore a count of forms, with one exception noted in section 2.6, where the Makasar comparison is made by meaning.",
  "Every count in this paper is therefore a count of forms unless it is said to be by meaning, as in the Makasar comparison of section 2.6 and in parts of sections 3.2 and 3.3.",
  "sections 3.2 and 3.3 also count by meaning")
c("are not cross-coded with the Tolaki list in this way (no form). The label is thus",
  "are not cross-coded with the Tolaki list in this way: no Tolaki form is coded only through them. The label is thus", "'(no form)' was cryptic")
c("and we test stricter versions of each in section 3.5.", "and we look at stricter or length-conditioned versions of them in section 3.5.",
  "no stricter version of the onset-string input is reported in 3.5, only length-conditioned ones")
c("Table 5 lists the inputs that matter with their observed values.", "Table 5 gives the observed values of 10 of the inputs.",
  "the ten properties of table 5 were chosen in the design, not by a rule of what 'matters'")
c("each prediction is attributed to the inputs, and the mean of the absolute attributions over all forms is an input's average contribution, shown in figure 1.",
  "each prediction of a model fitted to all 1,357 forms is attributed to the inputs, in units of the log-odds of the score, and the mean of the absolute attributions over all forms is an input's average contribution, shown in figure 1.",
  "unit and model of the SHAP values were not stated (E229 01_tables.py: TreeExplainer on the model fitted to all forms)")
c("it shares a cognate-set number with the PMP entry; it is coded but in another set; or it is uncoded. The same is done for",
  "it shares a cognate-set number with the PMP entry; it is coded but in another set; or it is uncoded. A meaning with more than one form is placed in the first class if any of its forms shares a set with the PMP entry, and in the third only if none of its forms is coded. The same is done for",
  "how a meaning with several forms is classed (E228 DESIGN section 3) was not said")
c("the statistic is the mean normalized distance over all such pairs, compared with",
  "the statistic is the mean normalized distance over all such pairs (here the plain distance, with no leading characters ignored), compared with",
  "the permutation test uses the plain distance, the screens the leading-character-tolerant one (E228 DESIGN section 6)")
c("The decision rules for the three-class count, the look-alike screens, this test, and the spelling tests below were written down before the runs, on the same day and by the same analyst; none was registered externally.",
  "The decision rules for the three-class count, the Tolaki screen, this test, and the spelling tests below were written down before the runs, on the same day and by the same analyst; none was registered externally. The Makasar screens, the sensitivity figures of the screen, and the paired tests of section 3.2 were added after the first results were known and had no such rule.",
  "only the Tolaki screen had a rule written before the run; the Makasar screens, the sensitivity and the McNemar tests are post hoc (E231)")
c("or uncoded. For Makasar the three shares are", "or uncoded; figure 3 draws the same shares. For Makasar the three shares are",
  "figure 3 was mentioned only in a table note")
c("What distinguishes Makasar from its two relatives in these data is one fact.", "What distinguishes Makasar from its two relatives in these data is mainly one fact.",
  "see next")
c("are the same fact counted again: on the meanings coded in both lists, Makasar is in the PMP set for 70 of 110 meanings against Bugis's 69, and for 70 of 115 against Sa'dan Toraja's 71. Those conditional figures",
  "are largely the same fact counted again. On the meanings coded in both lists, Makasar is in the PMP set for 70 of 110 meanings against Bugis's 69, and for 70 of 115 against Sa'dan Toraja's 71; for the cognate sets held in common a smaller gap remains on such meanings (76.3 percent with Bugis and 70.1 with Sa'dan Toraja, against 81.0 between Bugis and Sa'dan Toraja). Those conditional figures",
  "E231 P1: among meanings coded in both lists the pairwise shares are 87/114, 82/117 and 111/137, so a gap remains")
c("against 3.8 percent expected by chance and against 90 of 136 (66 percent) among the coded Makasar forms.",
  "against 3.8 percent expected by chance and against 90 of 136 (66 percent) among the coded Makasar forms; the uncoded forms of Bugis and of Sa'dan Toraja resemble their two relatives' forms at similar rates (8 of 62 and 8 of 45).",
  "E231 P9: the low rate is not particular to Makasar")
c("comparison of 62 Austronesian languages with a 200-item reconstructed PMP list", "comparison of 62 Austronesian languages outside Taiwan with a 200-item reconstructed PMP list",
  "the source's own restriction (Bellwood 1997:115)")
c("Makasar shares 45 percent with Bugis and 42 with Sa'dan, while Bugis and Sa'dan share 60.",
  "Makasar shares 45 percent with Bugis and 42 with Sa'dan Toraja, while Bugis and Sa'dan Toraja share 60.", "one name for the list")
c("88.4 percent recur in those lists.", "88.4 percent have a look-alike in those lists by the screen of section 2.6 (7.5 percent by chance).",
  "the 88.4 percent is a look-alike rate with a chance rate, not attested recurrence (E228 S3_lookalikes.json)")
c("and table 4 the held-out-list results. With the written form", "and table 4 the held-out-list results, whose two AUC columns figure 4 plots. With the written form",
  "figure 4 was mentioned only in a table note")
c("is 0.701 (a difference of −0.025).", "is 0.701 (a difference of −0.025, computed before rounding).", "0.727 − 0.701 reads as 0.026")
c("Table 5 gives, for each input that matters, its mean or share among uncoded and among coded forms,",
  "Table 5 gives, for 10 properties, the mean or share among uncoded and among coded forms,", "as above")
c("a Mantel-Haenszel odds ratio of being uncoded for indicators. These properties",
  "a Mantel-Haenszel odds ratio of being uncoded for indicators. An odds ratio of 2 means that, within a list, the odds of being uncoded are twice as high for forms with the property as for forms without it. These properties",
  "the odds ratio was nowhere explained to the readers the editors named")
c("30.4 against 23.7 percent as coded (1.55 [1.18, 2.04])", "30.4 against 23.7 percent as the script defines it (1.55 [1.18, 2.04])",
  "'as coded' meant 'as programmed' in a paper whose key term is coded")
c("(1.68 [1.12, 2.51] as coded; 1.61", "(1.68 [1.12, 2.51] as the script defines it; 1.61", "same")
c("Which of the two accounts for the sign was not tested. Statements about the vocabulary",
  "Which of the two accounts for the sign was not tested. The core-set input, likewise, points toward uncoded in the figure although table 5 shows no difference. Statements about the vocabulary",
  "second disagreement between figure 1 and table 5 (E229 README) was not mentioned")
c("printed as ABVD records them with their set number, the PMP entry of the meaning, and the nearest same-meaning form in another Sulawesi list by the screen. They are the forms the classifier scores most extremely, so they illustrate its inputs by construction; none has been assessed etymologically, and the nearest look-alike is a spelling match, not a cognate.",
  "printed as ABVD records them with their set number, the PMP entry of the meaning, and the nearest same-meaning form in another ABVD list from the Sulawesi region with its distance by the measure of section 2.6. They are the forms the classifier scores most extremely, so they illustrate its inputs by construction; none has been assessed etymologically, and the nearest form is a match of spelling, not a cognate; at a distance above 0.34 it is not a look-alike by the screen.",
  "two of the eight 'nearest look-alikes' of table 7 have a distance of 0.5 and are not look-alikes by the screen")
c("Note. Forms are given as ABVD records them and have not been assessed etymologically. \"Set\" is ABVD's cognate-set number; \"PMP (ABVD)\" is the Proto-Malayo-Polynesian entry for the meaning; the nearest look-alike is a spelling match by the screen of section 2.6, which ignores up to three leading characters, so a distance of 0.0 does not mean identity.",
  "Note. Forms are given as ABVD records them and have not been assessed etymologically. \"Set\" is ABVD's cognate-set number; \"PMP (ABVD)\" is the Proto-Malayo-Polynesian entry for the meaning; \"Score\" is the classifier's out-of-sample score. The last column gives the nearest form of the same meaning in another ABVD list located in the Sulawesi region (lists of the same language and the Bajo lists left out; list names as in ABVD) and its distance by the measure of section 2.6, which ignores up to three leading characters, so a distance of 0.00 does not mean identity; a distance above 0.34 is not a look-alike by the screen.",
  "same; the pool of lists was defined nowhere in the article")
c("they are uncoded in four lists and coded in Bugis. The 'One Hundred' forms are not uniform across the lists and do not belong with them.",
  "they are uncoded in four lists and coded in Bugis (the Muna list has no form for them). 'One Hundred' is likewise uncoded in four lists and coded in Bugis; its forms are Bugis *səratuu*, Sa'dan Toraja *saratuʔ*, Wolio *saatu*, Tolaki *aso'etu*, and Makasar *sibilaŋŋaŋ*, and we give no account of them.",
  "'do not belong with them' was a judgment of the kind the paper disclaims; the forms are printed instead")
c("By the rule written before the run (a loss between 0.010 and 0.020 counts as partial dependence), the result depends partly",
  "By the rule written before the run for the cross-validated AUC (a loss of 0.010 or less counts as no dependence, a loss above 0.020 as dependence, and a loss in between as partial dependence), the result depends partly",
  "the rule applies to the cross-validated AUC and has an upper bound that the held-out figure beside it exceeds (E228 DESIGN section 2)")
c("TABLE 8. THE AUC OF THE FULL MODEL UNDER SIX WRITINGS OF THE GLOTTAL STOP", "TABLE 8. THE AUC OF THE FULL MODEL WITH THE GLOTTAL MARK AS IN THE SOURCES AND UNDER SIX RECODINGS",
  "the table has seven rows")
c("Note. Δ is the change from the first row. Rows 2 and 7", "Note. Δ is the change from the first row, computed before rounding. Rows 2 and 7", "0.727 − 0.710 reads as 0.017")
c("its accuracy as a yes/no classifier is close to the trivial rule, and for a list it has not seen it is below the majority answer.",
  "its accuracy as a yes/no classifier is close to the trivial rule, and for five of the six lists, when held out, it is below the majority answer.",
  "Makasar is the exception (table 4)")
c("by a screen whose sensitivity even on coded forms is at most 38 percent.",
  "by a screen that finds 38 percent of the coded forms in the comparison with the PMP entry and 66 percent in the comparison with the relatives.",
  "38 percent is the measured rate for one comparison, not a ceiling (E231 P8, P9)")
c("reported by Mills (1975), Sirk (1989), and Bulbeck, Pasqua, and Di Lello (2000); they do not prove it",
  "reported by Mills (1975) and Sirk (1989); they do not prove it", "the third work cites a retention figure and reports no lexical count of its own")
c("The look-alike screen compares spellings, its 42 Bungku-Tolaki comparison lists represent 13 languages from two contributors, and its sensitivity is an upper bound.",
  "The look-alike screen compares spellings, its 42 Bungku-Tolaki comparison lists represent 13 languages from two contributors, and its sensitivity is an upper bound. The Makasar forms were compared with two related lists and the Tolaki forms with 43, so the two look-alike rates are not on the same footing, and ABVD's Proto-South Sulawesi list was not used.",
  "pool sizes of the two screens differ; ABVD list 861 exists and was not used for Makasar")
c("The three-class count, the look-alike screens, the permutation test, and the spelling tests had their decision rules written before they were run, on the same day and by the same analyst and without external registration; the further counts of section 3.5 and the per-list figures were made after the results were known.",
  "The three-class count, the Tolaki screen, the permutation test, and the spelling tests had their decision rules written before they were run, on the same day and by the same analyst and without external registration; the Makasar screens, the sensitivity figures, the paired tests, the counts by meaning of section 3.3, the further counts of section 3.5, and the per-list figures were made after the results were known.",
  "as in section 2.6")
c("the cell of table 6, and the nearest same-meaning look-alike in other Sulawesi lists and, for Tolaki, in Bungku-Tolaki lists);",
  "the cell of table 6, and the nearest same-meaning form, with its distance, in other Sulawesi lists and, for Tolaki, in Bungku-Tolaki lists);",
  "as in table 7: the released columns give the nearest form, which is a look-alike only at 0.34 or less")
c("Conversion of the written forms to a phonetic transcription (Anderson et al. 2018) could reduce",
  "In addition, conversion of the written forms to a phonetic transcription (Anderson et al. 2018) could reduce", "'Three directions follow' was followed by a fourth")

bad = 0
for old, new, n, why in C:
    k = out.count(old)
    if k != n:
        bad += 1
        print(f"STAGE 2 MISMATCH ({k} found, {n} expected): {old[:100]}")
    out = out.replace(old, new)
if bad:
    sys.exit(f"{bad} stage-2 replacement(s) did not match; nothing written")
print(f"stage 2: {len(C)} corrections of content applied")

# =====================================================================================================================
# Stage 3 — after the independent verification of stage 2 (same day): one count that was wrong since v0.2, and stage-2
# sentences that were still imprecise. The `old` strings are the text AFTER stage 2.
# =====================================================================================================================
n2 = len(C)
c("Of the 34 uncoded forms that match an entry of the Proto-Bungku-Tolaki list, seven match an entry that ABVD does assign to a cognate set; these seven are the clearest candidates",
  "Of the 34 uncoded forms that match an entry of the Proto-Bungku-Tolaki list, six match an entry that ABVD does assign to a cognate set (for one of the six, 'Eight', an uncoded entry matches equally well); these six are the clearest candidates",
  "the stored 7 is a count by meaning; by matched entry it is six (E228 04_pbt_matched_entry_recount.py)")
c("Seven of them match a coded Proto-Bungku-Tolaki entry by the screen, and 12 resemble the PMP entry.",
  "Six of them match a coded Proto-Bungku-Tolaki entry by the screen, and 12 resemble the PMP entry.", "same")
c("by a screen that finds 38 percent of the coded forms in the comparison with the PMP entry and 66 percent in the comparison with the relatives.",
  "by a screen that finds 38 percent of the forms that ABVD places in the PMP entry's set, and a relative's look-alike for 66 percent of the coded forms.",
  "38 percent = 33 of the 87 forms in the PMP entry's set, not of all coded forms; 66 percent = 90 of 136 coded forms")
c("What distinguishes Makasar from its two relatives in these data is mainly one fact. On the meanings shared with each relative, Makasar has more uncoded meanings than Bugis (exact McNemar test, p = 0.006) and than Sa'dan Toraja (p = 0.0001; both unadjusted). Its lower share in the PMP set and its smaller share of cognate sets held in common with each relative (41.4 and 39.2 percent of the meanings present in both lists, against 53.1 percent between Bugis and Sa'dan Toraja) are largely the same fact counted again. On the meanings coded in both lists, Makasar is in the PMP set for 70 of 110 meanings against Bugis's 69, and for 70 of 115 against Sa'dan Toraja's 71; for the cognate sets held in common a smaller gap remains on such meanings (76.3 percent with Bugis and 70.1 with Sa'dan Toraja, against 81.0 between Bugis and Sa'dan Toraja). Those conditional figures",
  "What distinguishes Makasar from its two relatives in these data is mainly one fact. On the meanings shared with each relative, Makasar has more uncoded meanings than Bugis (exact McNemar test, p = 0.006) and than Sa'dan Toraja (p = 0.0001; both unadjusted). Its lower share in the PMP set is the same fact counted again: on the meanings coded in both lists, Makasar is in the PMP set for 70 of 110 meanings against Bugis's 69, and for 70 of 115 against Sa'dan Toraja's 71. Its smaller share of cognate sets held in common with each relative (41.4 and 39.2 percent of the meanings present in both lists, against 53.1 percent between Bugis and Sa'dan Toraja) is that fact in part only: on the meanings coded in both lists the shares are 76.3 percent with Bugis and 70.1 with Sa'dan Toraja, against 81.0 between Bugis and Sa'dan Toraja, a difference on which no test was made. Those conditional figures",
  "'largely' held for the pair with Bugis (gap 11.7 -> 4.7 points) but not for Sa'dan Toraja (13.9 -> 10.9); E231 P1")
c("The same screen is applied to the uncoded Makasar forms, against the PMP entry and against the Bugis and Sa'dan Toraja forms of the same meaning.",
  "The same screen is applied to the uncoded forms of Makasar and of Tolaki against the PMP entry, and to the uncoded forms of each South Sulawesi list against the forms of the same meaning in the other two.",
  "the Tolaki screen against PMP and the Bugis and Sa'dan Toraja screens had no statement of method")
c("the number of look-alike pairs at the 0.34 threshold is counted beside it.", "the number of pairs within 0.34 by the same plain distance is counted beside it.",
  "'look-alike' is defined by the other distance two sentences earlier")
c("The decision rules for the three-class count, the Tolaki screen, this test, and the spelling tests below were written down before the runs, on the same day and by the same analyst; none was registered externally. The Makasar screens, the sensitivity figures of the screen, and the paired tests of section 3.2 were added after the first results were known and had no such rule.",
  "The decision rules for the three-class count, the Tolaki screen against its subgroup, this test, the spelling tests below, and the application to further lists of section 3.8 were written down before the runs, on the same day and by the same analyst; none was registered externally. The analyses that look further into those results had no such rule and were added after the first results were known: the screens against the PMP entry and against relatives' forms with their sensitivity figures, the paired tests and conditional shares of section 3.2, the relabeling of section 3.1, the counts by meaning of section 3.3, and the further counts of sections 3.5 to 3.7.",
  "the stage-2 enumeration left out post hoc items (Tolaki against PMP, the Bugis and Sa'dan Toraja screens, the relabeling, the conditional shares) and the pre-specified sixteen-list rule (E228 DESIGN section 7)")
c("Note. Wilson 95 percent intervals; the unit is the meaning. The column \"In the PMP entry's set\" is a lower bound on the share ABVD would count as retained, since an uncoded meaning cannot share a set with the PMP entry. Figure 3 draws the same shares.",
  "Note. Wilson 95 percent intervals; the unit is the meaning. The column \"In the PMP entry's set\" is a lower bound on the share ABVD would count as retained, since an uncoded meaning cannot share a set with the PMP entry.",
  "said in the text and in the caption already")
c("Majority = accuracy of the majority answer for that list. Figure 4 plots the two AUC columns.", "Majority = accuracy of the majority answer for that list.", "same")
c("which ignores up to three leading characters, so a distance of 0.00 does not mean identity; a distance above 0.34 is not a look-alike by the screen.",
  "which ignores up to three leading characters, so a distance of 0.00 does not mean identity; a distance above 0.34 is not a look-alike by the screen, and where several lists are equally near the first in ABVD's order is shown.",
  "the nearest form is often one of many ties (release README)")
c("Four pairs are look-alikes at the 0.34 threshold, against 0.52 expected (p = 0.0016).", "Four pairs are within 0.34 by the plain distance, against 0.52 expected (p = 0.0016).", "as in section 2.6")
c("with 472 look-alike pairs against 8.5 expected.", "with 472 pairs within 0.34 against 8.5 expected.", "same")
c("and about one thirtieth on look-alike pairs (comparing", "and about one thirtieth on pairs within 0.34 (comparing", "same")
c("The four pairs are Bugis *ma-kunru*", "The four uncoded pairs are Bugis *ma-kunru*", "after the sentence on the coded pairs")
c("The three-class count, the Tolaki screen, the permutation test, and the spelling tests had their decision rules written before they were run, on the same day and by the same analyst and without external registration; the Makasar screens, the sensitivity figures, the paired tests, the counts by meaning of section 3.3, the further counts of section 3.5, and the per-list figures were made after the results were known.",
  "The three-class count, the Tolaki screen against its subgroup, the permutation test, the spelling tests, and the application to further lists had their decision rules written before they were run, on the same day and by the same analyst and without external registration; the screens against the PMP entry and against relatives' forms, their sensitivity figures, the paired tests and conditional shares, the relabeling of section 3.1, the counts by meaning, and the further counts of sections 3.5 to 3.7 were made after the first results were known.",
  "as in section 2.6")
# house style missed in stage 1 (template: protoform, protolanguage closed up; the en-dash as minus sign; numerals for both
# numbers of one kind in a phrase, Chicago 9.7)
c("cannot be assigned to a higher-level proto-form:", "cannot be assigned to a higher-level protoform:", "template, 'Hyphen': protoform")
c("reconstruct at most to a low-level proto-language,", "reconstruct at most to a low-level protolanguage,", "same")
c("by the speakers of a low-level proto-language can be inherited", "by the speakers of a low-level protolanguage can be inherited", "same")
c("(Spearman's rho = −0.55)", "(Spearman's rho = –0.55)", "template: the en-dash as a minus sign")
c("(a difference of −0.025, computed before rounding)", "(a difference of –0.025, computed before rounding)", "same")
c("(−0.0003 with form and meaning inputs,", "(–0.0003 with form and meaning inputs,", "same")
c("and five of the 11 also carry", "and 5 of the 11 also carry", "Chicago 9.7: numerals for both numbers of one kind")
c("one of the 12 is loan-flagged", "1 of the 12 is loan-flagged", "same")
c("for two of those 11 meanings", "for 2 of those 11 meanings", "same")
bad = 0
for old, new, n, why in C[n2:]:
    k = out.count(old)
    if k != n:
        bad += 1
        print(f"STAGE 3 MISMATCH ({k} found, {n} expected): {old[:100]}")
    out = out.replace(old, new)
if bad:
    sys.exit(f"{bad} stage-3 replacement(s) did not match; nothing written")
print(f"stage 3: {len(C) - n2} further corrections applied")
(D / "CHANGES_v0.3_to_v0.4_content.tsv").write_text(
    "why\told\tnew\n" + "\n".join(f"{w}\t{o}\t{nw}" for o, nw, _, w in C), encoding="utf-8", newline="\n")

# ---- Figure 1: caption and position marker move to the place of first mention (Section 2.4), so that the figures appear in
#      numerical order (Chicago: a figure follows its first text reference); the text of Section 3.5 is unchanged.
fig1 = "[[FIGURE:1|FIGURE 1. MEAN ABSOLUTE CONTRIBUTION OF EACH INPUT TO THE PREDICTIONS OF THE FULL MODEL (SHAP VALUES), WITH THE DIRECTION OF EACH INPUT'S EFFECT ON THE SCORE]]"
assert out.count(fig1 + "\n\n") == 1
out = out.replace(fig1 + "\n\n", "")
anchor = "Software: Python 3.11, scikit-learn 1.8.0, xgboost 3.0.3, shap 0.50.0.\n\n"
assert out.count(anchor) == 1
out = out.replace(anchor, anchor + fig1 + "\n\n")

(D / "P8_revision_v0.4.md").write_text(out, encoding="utf-8", newline="\n")
print(f"written P8_revision_v0.4.md: {len(E)} replacements + Figure 1 marker moved")
from collections import Counter
print(dict(Counter(r.split(' +')[0].split(' (')[0] for *_, r in E)))

# ---- what is left: spelled-out numbers of two digits (expected: only at the start of a sentence or in a heading)
words = r"\b(ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b"
for ln, l in enumerate(out.splitlines(), 1):
    for m in re.finditer(words, l, re.I):
        print(f"  left L{ln}: …{l[max(0, m.start()-40):m.end()+30]}…")
for ln, l in enumerate(out.splitlines(), 1):
    for m in re.finditer(r"(?<![.?!:]\s)(?<!^)\b(Section|Table|Figure)s? \d", l):
        if not l.startswith("[[") :
            print(f"  capital left L{ln}: …{l[max(0, m.start()-40):m.end()+20]}…")

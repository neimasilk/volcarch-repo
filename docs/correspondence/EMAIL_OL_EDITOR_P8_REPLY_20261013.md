# Reply of the editor of *Oceanic Linguistics* (Sander Adelaar) to the P8 note of 7 October — record and what it changed

**Received 2026-10-13, 13:19 WIB** in the PI's university mailbox, subject "your queries". **From:** Sander Adelaar (editor),
directly to the PI. It answers the content questions of the note sent through the portal on 2026-10-07
(`EMAIL_OL_EDITOR_P8_NOTE_20261007.md`); the practical questions had been answered by the Managing Editor on 2026-10-08
(`EMAIL_OL_EDITOR_P8_REPLY_20261008.md`). Read in this repo on 2026-10-15, when the PI pasted it into the session
("baca handoff marik kita selesaikan P8, kabar terbaru: …"); the mailbox check of 2026-10-15 (about 09:20 WIB, last 4 days) shows it as
the only journal mail since 2026-10-08; nothing from arXiv or from the co-author. **Public repo:** paraphrase only; the original stays in the PI's mailbox.

## 1. What he said (paraphrase)

1. He thanks the authors for the detailed explanation and takes the view that they have found inaccuracies that need to be
   corrected — some with the reviewers' help, some by rereading their own work — and that the paper is probably moving in the
   right direction. **They should go ahead.**
2. **When resubmitting, add a note that points out in general terms some of the changes**, so that the reviewers know how the
   new version relates to the old one.
3. The journal **will consult the current reviewers again** and will treat the manuscript **anonymously**.
4. He asks the authors to **acknowledge their use of AI** for some of the results, as he believes they have already done.
5. He looks forward to the revision.

## 2. What it settles

- The open question of the note of 7 October (D7): the corrections of our own are handled **inside this revision** — the
  editor agrees. The conditional sentence of cover letter v0.5 ("not having heard otherwise …") is obsolete; decision (f) of
  `BACA_DULU_P8_20261012.pdf` and item 4 of `docs/HANDOFF_20261012.md` §1 are moot.
- **A further review round will happen** (same reviewers). So the R1 files must stay anonymised — they are. It also means a
  decision is not imminent: the June 2027 issue (VENUE.md addendum) is uncertain, as already noted on 2026-10-08.

## 3. What was done about it (2026-10-15, controller)

| His request | Already in the package? | Change |
|---|---|---|
| A note for the reviewers, in general terms, on how the new version relates to the old one | Partly: paragraph 2 of the response named the main changes; Part C lists the corrections; the tracked copy exists | **Response v0.6:** a headed section "How the revised version relates to the submitted one" at the top — what is the same, what the paper now claims, terms, where each submitted section, figure and table has gone, the tracked copy. Paragraph 2 of v0.5 is absorbed into it. |
| Acknowledge the use of AI | **Yes**: the article's unnumbered "Declaration on the use of artificial intelligence" names every stage, the analyses included; it **is** in the anonymised file (checked in the `.docx`, 2026-10-15) | Article unchanged. **Cover letter v0.6** says, in answer to him, where the declaration is and what it covers. |
| Go ahead | — | **Cover letter v0.6:** the conditional sentence replaced by thanks for his reply and a pointer to the note. |

Built with `build_letters.py` (v0.6); `check_letters.py` passes (numbers, section references, 38 reviewer points, anonymity);
`make_manifest.py --check`: the seven files not replaced are byte for byte those in the portal since 2026-10-12. Only the
portal rows 6 (Author Cover Letter) and 7 (Revision Summary) change.

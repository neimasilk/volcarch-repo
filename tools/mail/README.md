# tools/mail — Gmail and portal helpers (moved from session scratchpads 2026-10-02)

The PI's Gmail is a Workspace account where IMAP is refused, so mail goes through a real Chrome window driven by
Playwright. Credentials are read from the repo `.env` (`user_gmail_dan_drive`, `pass_gmail`) inside the
scripts, so they never appear in a command or a log. Browser profiles (`gmail_profile/`, `portal_profile/`) hold
login cookies and are **gitignored**; the first run in a new place logs in again (a phone approval may be needed).

| Script | What it does | Safety |
|---|---|---|
| `gmail_check.py` | helpers (`load_env`, `login_if_needed`) + inbox search listing | read-only |
| `gmail_fetch_thread.py QUERY MUST_CONTAIN [OUTDIR]` | opens one thread, saves each message body as text and every attachment; OUTDIR **outside the repo** for reviewer reports and decision letters (public repo) | read-only (marks the thread read) |
| `gmail_compose_send.py TO SUBJECT BODY.txt [--attach=FILE] [--send]` | composes a new message; prints body, recipients, subject, attachments; sends only with `--send`, then shows the top of Sent | inspect first, then send |
| `gmail_drafts_audit.py [--discard=SUBJ]` | lists Sent and Drafts; `--discard` removes only drafts whose subject also appears in Sent (leftover duplicates) | read-only without the flag |
| `portal_browser.py` | opens a headed Chrome with CDP on :9333 (ArchCalc + Zenodo tabs) for the **PI to log in himself**; stays open ≤3 h | Claude never handles portal passwords |
| `cdp.py URLSUB goto/shot/text/eval/click ARG` | one action in a tab of that logged-in browser | — |

**Lesson (2026-10-02):** an inspection run of `gmail_compose_send.py` leaves an autosaved draft in Gmail. After
every send, run `gmail_drafts_audit.py` and discard the duplicate.
**Lesson:** the ArchCalc portal's notification email can fail; follow a portal discussion with an email to the
editorial address (redazioneac@ispc.cnr.it).
**Lesson (2026-10-06):** the Playwright MCP browser is not always logged in; the persistent profile `gmail_profile/` of these
scripts logged in from `.env` without a phone prompt. A `subject:"…"` search can return only hidden rows — search the plain
phrase and pick the row by its visible text. PDF annotations of a reviewer: PyMuPDF `page.annots()` (text under a highlight via its quads).

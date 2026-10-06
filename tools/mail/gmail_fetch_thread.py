"""Fetch one Gmail thread (message bodies + attachments) for the PI, at the PI's request.

usage: python gmail_fetch_thread.py "SEARCH QUERY" "TEXT THAT THE RESULT ROW MUST CONTAIN" [OUTDIR]

Credentials are read from the repo .env inside the helper (never on the command line, never printed).
OUTDIR defaults to tools/mail/gmail_out/<timestamp>/ (gitignored). **Give an OUTDIR outside the repository
for anything that must not become public** (reviewer reports, decision letters with portal links):
the repository is public and reviewer/editor text is never recorded in it.

Opening a thread marks it as read. Nothing is sent, changed or deleted.

Lessons built in (2026-10-06): a `subject:"..."` query can return rows that are all hidden, so the plain
query is tried and the row is chosen by its visible text; attachments are fetched through the browser
context's own request client, using the `download_url` attribute of the attachment chip.
"""
import io
import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gmail_check import PROFILE, load_env, login_if_needed  # noqa: E402

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)
QUERY, MUST = sys.argv[1], sys.argv[2]
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "gmail_out" / time.strftime("%Y%m%d_%H%M%S")
OUT.mkdir(parents=True, exist_ok=True)

env = load_env(HERE.parents[1] / ".env")
with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(
        str(PROFILE), channel="chrome", headless=False, accept_downloads=True,
        args=["--disable-blink-features=AutomationControlled"], viewport={"width": 1400, "height": 900})
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    if not login_if_needed(page, env["user_gmail_dan_drive"], env["pass_gmail"]):
        ctx.close()
        sys.exit(2)
    page.goto("https://mail.google.com/mail/u/0/#search/" + urllib.parse.quote(QUERY, safe=""),
              wait_until="domcontentloaded")
    try:
        page.wait_for_selector("tr.zA", timeout=20000)
    except Exception:
        pass
    page.wait_for_timeout(4000)
    rows = page.locator("tr.zA")
    hit = [i for i in range(rows.count()) if rows.nth(i).is_visible() and MUST in rows.nth(i).inner_text()]
    print("visible rows containing the required text:", len(hit))
    if not hit:
        ctx.close()
        sys.exit(3)
    rows.nth(hit[0]).click()
    page.wait_for_timeout(6000)
    exp = page.locator('[aria-label="Expand all"]')
    if exp.count():
        exp.first.click()
        page.wait_for_timeout(3000)
    info = page.evaluate("""() => {
      const msgs = [...document.querySelectorAll('div.adn')].map(m => {
        const from = m.querySelector('span.gD'), date = m.querySelector('span.g3'), body = m.querySelector('div.a3s');
        return {from: from ? (from.getAttribute('email') || from.innerText) : '',
                date: date ? (date.getAttribute('title') || date.innerText) : '', body: body ? body.innerText : ''};
      });
      const atts = [...document.querySelectorAll('span[download_url]')].map(s => s.getAttribute('download_url'));
      return {subj: (document.querySelector('h2.hP') || {}).innerText || '', msgs, atts};
    }""")
    print("subject:", info["subj"], "| messages:", len(info["msgs"]), "| attachments:", len(info["atts"]))
    for i, m in enumerate(info["msgs"], 1):
        (OUT / f"message_{i}.txt").write_text(f"FROM: {m['from']}\nDATE: {m['date']}\n\n{m['body']}", encoding="utf-8")
    saved = []
    for a in info["atts"]:
        parts = a.split(":", 2)            # mime:filename:url
        if len(parts) != 3:
            continue
        data = ctx.request.get(parts[2]).body()
        name = re.sub(r'[\\/:*?"<>|]', "_", urllib.parse.unquote(parts[1])) or "attachment.bin"
        (OUT / name).write_bytes(data)
        saved.append({"file": name, "bytes": len(data)})
    json.dump({"subject": info["subj"], "n_messages": len(info["msgs"]), "attachments": saved},
              open(OUT / "index.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("saved to", OUT, "|", saved)
    ctx.close()

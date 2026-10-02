"""List Drafts and Sent (top rows) with recipients/subject/snippet. Read-only unless --discard SUBJ_SUBSTR.

--discard removes ONLY drafts whose subject contains the given substring AND whose subject also
appears in Sent today (i.e. leftover duplicates of messages that were sent).
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from gmail_check import load_env, login_if_needed, PROFILE  # noqa: E402

DISCARD = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--discard=")]
ROWS_JS = """() => [...document.querySelectorAll('tr.zA')].filter(r => r.offsetParent).slice(0, 15).map(r => ({
    who: [...r.querySelectorAll('[email]')].map(e => e.getAttribute('email')).join(','),
    subj: (r.querySelector('span.bog') || {}).innerText || '',
    snip: ((r.querySelector('span.y2') || {}).innerText || '').slice(0, 80),
    when: (r.querySelector('td.xW span') || {}).innerText || ''}))"""

env = load_env(Path(__file__).resolve().parents[2] / ".env")
with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(str(PROFILE), channel="chrome", headless=False,
        args=["--disable-blink-features=AutomationControlled"], viewport={"width": 1400, "height": 900})
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    login_if_needed(page, env["user_gmail_dan_drive"], env["pass_gmail"])
    page.goto("https://mail.google.com/mail/u/0/#sent", wait_until="domcontentloaded"); page.wait_for_timeout(6000)
    sent = page.evaluate(ROWS_JS)
    print("== SENT"); [print(r) for r in sent[:6]]
    page.goto("https://mail.google.com/mail/u/0/#drafts", wait_until="domcontentloaded"); page.wait_for_timeout(6000)
    drafts = page.evaluate(ROWS_JS)
    print("== DRAFTS"); [print(r) for r in drafts]
    sent_subj = {r["subj"] for r in sent}
    for sub in DISCARD:
        rows = page.locator("tr.zA:visible")
        for i in range(rows.count()):
            s = rows.nth(i).locator("span.bog").inner_text()
            if sub in s and s in sent_subj:
                rows.nth(i).locator('div[role="checkbox"]').click()
        page.wait_for_timeout(1000)
        btn = page.locator('div[role="button"]:visible', has_text="Discard drafts")
        if btn.count():
            btn.first.click(); page.wait_for_timeout(4000); print("discarded drafts matching:", sub)
        else:
            print("no discard button / nothing selected for:", sub)
    if DISCARD:
        page.goto("https://mail.google.com/mail/u/0/#drafts", wait_until="domcontentloaded"); page.wait_for_timeout(5000)
        print("== DRAFTS AFTER"); [print(r) for r in page.evaluate(ROWS_JS)]
    ctx.close()

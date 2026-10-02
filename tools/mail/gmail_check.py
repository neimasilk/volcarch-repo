"""Gmail check for the PI, at the PI's request (2026-10-01).
Reads credentials from the repo .env itself so they never appear in any command or log.
Mode 'list': search results only (does not open threads, so nothing is marked read).
"""
import json, sys, time, urllib.parse
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
PROFILE = HERE / "gmail_profile"
OUT = HERE / "gmail_out"; OUT.mkdir(exist_ok=True)

def load_env(p):
    env = {}
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t or t.startswith("#") or "=" not in t:
            continue
        k, v = t.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env

def login_if_needed(page, user, pw):
    page.goto("https://mail.google.com/mail/u/0/#inbox", wait_until="domcontentloaded")
    page.wait_for_timeout(5000)
    if "accounts.google.com" not in page.url:
        return True
    try:
        page.fill("#identifierId", user, timeout=15000)
        page.click("#identifierNext")
        page.wait_for_selector('input[name="Passwd"]', state="visible", timeout=30000)
        page.fill('input[name="Passwd"]', pw)
        page.click("#passwordNext")
    except Exception as e:
        print("LOGIN FORM PROBLEM:", type(e).__name__, str(e)[:200])
    for _ in range(75):  # up to ~150 s, e.g. for a phone approval
        page.wait_for_timeout(2000)
        if "mail.google.com/mail" in page.url and "accounts.google.com" not in page.url:
            return True
    body = page.inner_text("body")[:1200]
    print("STILL ON GOOGLE ACCOUNTS PAGE. Visible text:\n", body)
    return False

def scrape_rows(page):
    return page.evaluate("""() => {
      const rows = [...document.querySelectorAll('tr.zA')];
      return rows.filter(r => r.offsetParent !== null).map(r => {
        const s = r.querySelector('span[email]');
        const subj = r.querySelector('span.bog');
        const snip = r.querySelector('span.y2');
        const d = r.querySelector('td.xW span[title], td.xW span');
        return {
          from_name: s ? (s.getAttribute('name') || s.innerText) : '',
          from_email: s ? s.getAttribute('email') : '',
          subject: subj ? subj.innerText : '',
          snippet: snip ? snip.innerText.replace(/^\s*-\s*/, '') : '',
          date: d ? (d.getAttribute('title') || d.innerText) : '',
          unread: r.classList.contains('zE'),
          thread: (r.querySelector('[data-thread-id]') || {}).getAttribute ? (r.querySelector('[data-thread-id]').getAttribute('data-thread-id') || '') : '',
        };
      });
    }""")

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "list"
    query = sys.argv[2] if len(sys.argv) > 2 else "after:2026/08/12"
    env = load_env(Path(__file__).resolve().parents[2] / ".env")
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            str(PROFILE), channel="chrome", headless=False,
            args=["--disable-blink-features=AutomationControlled"], viewport={"width": 1400, "height": 900})
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        if not login_if_needed(page, env["user_gmail_dan_drive"], env["pass_gmail"]):
            ctx.close(); sys.exit(2)
        print("logged in; account domain:", env["user_gmail_dan_drive"].split("@")[-1])
        allrows = []
        for pg in range(1, 9):
            url = "https://mail.google.com/mail/u/0/#search/" + urllib.parse.quote(query, safe="") + (f"/p{pg}" if pg > 1 else "")
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(5000)
            rows = scrape_rows(page)
            if not rows:
                break
            if allrows and rows[0] == allrows[-len(rows)] if len(allrows) >= len(rows) else False:
                break
            allrows.extend(rows)
            if len(rows) < 50:
                break
        json.dump(allrows, open(OUT / "rows.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("rows:", len(allrows))
        ctx.close()

if __name__ == "__main__":
    main()

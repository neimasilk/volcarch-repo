"""Compose a new Gmail message from a text file; inspect by default, send only with --send.

usage: python gmail_compose_send.py TO SUBJECT BODY_FILE [--send]
Body lines are reflowed: lines inside a paragraph are joined; numbered items start new paragraphs.
"""
import re
import sys
import urllib.parse
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from gmail_check import load_env, login_if_needed, PROFILE  # noqa: E402

to, subject, body_file = sys.argv[1:4]
SEND = "--send" in sys.argv
ATTACH = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--attach=")]
raw = Path(body_file).read_text(encoding="utf-8").strip()
main, sig = re.split(r"\n\s*\n(?=Hormat saya|With )", raw, maxsplit=1)
paras = []
for block in re.split(r"\n\s*\n", main):
    for it in re.split(r"\n(?=\d+\.\s)", block):
        paras.append(" ".join(l.strip() for l in it.splitlines()))
# signature keeps its line breaks; blank lines inside it collapse
body = "\n\n".join(paras) + "\n\n" + "\n".join(l.strip() for l in sig.splitlines() if l.strip())
print("----- BODY -----\n" + body + "\n----------------")

env = load_env(Path(__file__).resolve().parents[2] / ".env")
url = "https://mail.google.com/mail/?view=cm&fs=1&" + urllib.parse.urlencode(
    {"to": to, "su": subject, "body": body}, quote_via=urllib.parse.quote)
with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(str(PROFILE), channel="chrome", headless=False,
        args=["--disable-blink-features=AutomationControlled"], viewport={"width": 1400, "height": 900})
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    if not login_if_needed(page, env["user_gmail_dan_drive"], env["pass_gmail"]):
        print("LOGIN FAILED"); ctx.close(); sys.exit(2)
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_timeout(7000)
    for f in ATTACH:
        page.locator('input[type="file"][name="Filedata"]').first.set_input_files(f)
        page.wait_for_timeout(6000)
    att = page.evaluate("() => [...document.querySelectorAll('div[role=\"dialog\"] [aria-label^=\"Attachment\"], .dL')].map(e => e.innerText.split(String.fromCharCode(10))[0]).filter(Boolean)")
    print("attachments seen:", att)
    btxt = page.locator('div[aria-label="Message Body"][contenteditable="true"]').last.inner_text()
    rcpt = page.evaluate("() => [...document.querySelectorAll('[email]')].filter(e=>e.offsetParent).map(e=>e.getAttribute('email'))")
    subj = page.locator('input[name="subjectbox"]').input_value()
    print("recipients:", sorted(set(rcpt)), "| subject:", subj, "| body chars:", len(btxt))
    ok = to in rcpt and subj == subject and len(btxt) > 0.9 * len(body) and (not ATTACH or len(att) >= len(ATTACH))
    page.screenshot(path=str(Path(__file__).parent / "compose_before_send.png"))
    if not ok:
        print("ABORT: checks failed"); ctx.close(); sys.exit(4)
    if SEND:
        page.locator('div[role="button"][aria-label^="Send"]:visible').last.click()
        page.wait_for_timeout(6000)
        page.goto("https://mail.google.com/mail/u/0/#sent", wait_until="domcontentloaded")
        page.wait_for_timeout(6000)
        top = page.evaluate("""() => [...document.querySelectorAll('tr.zA')].filter(r => r.offsetParent).slice(0,2)
            .map(r => ((r.querySelector('span.bog')||{}).innerText||'') + ' | ' + ((r.querySelector('td.xW span')||{}).innerText||''))""")
        print("sent top:", top)
    else:
        print("INSPECTION ONLY — not sent")
    ctx.close()

"""Reply inside an existing Gmail thread from a text file; inspect by default, send only with --send.

usage: python gmail_reply_thread.py QUERY MUST_CONTAIN EXPECT_TO BODY_FILE [--attach=FILE] [--send] [--shot=PNG]
QUERY selects the thread list (e.g. 'in:sent "Kemungkinan analisis fitolit"'); MUST_CONTAIN picks the row by its
visible subject. EXPECT_TO must be among the reply's recipients, or nothing is sent.
Written 2026-10-22 because gmail_compose_send.py can only start new threads, and a follow-up should sit under the
first message so the reader sees both.
Without --send the reply draft is discarded again, so an inspection leaves nothing in Drafts.
"""
import re
import sys
import urllib.parse
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from gmail_check import load_env, login_if_needed, PROFILE  # noqa: E402

query, must, expect_to, body_file = sys.argv[1:5]
SEND = "--send" in sys.argv
ATTACH = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--attach=")]
SHOT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--shot=")),
            str(Path(__file__).parent / "reply_before_send.png"))

# same reflow as gmail_compose_send.py: paragraph lines joined, signature keeps its line breaks
raw = Path(body_file).read_text(encoding="utf-8").strip()
main, sig = re.split(r"\n\s*\n(?=Hormat saya|With )", raw, maxsplit=1)
paras = []
for block in re.split(r"\n\s*\n", main):
    for it in re.split(r"\n(?=\d+\.\s)", block):
        paras.append(" ".join(l.strip() for l in it.splitlines()))
sig_lines = [l.strip() for l in sig.splitlines() if l.strip()]
body = "\n\n".join(paras) + "\n\n" + "\n".join(sig_lines)
print("----- BODY -----\n" + body + "\n----------------")

EDITOR = 'div[aria-label="Message Body"][contenteditable="true"]'

env = load_env(Path(__file__).resolve().parents[2] / ".env")
with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(str(PROFILE), channel="chrome", headless=False,
        args=["--disable-blink-features=AutomationControlled"], viewport={"width": 1400, "height": 900})
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    if not login_if_needed(page, env["user_gmail_dan_drive"], env["pass_gmail"]):
        print("LOGIN FAILED"); ctx.close(); sys.exit(2)
    page.goto("https://mail.google.com/mail/u/0/#search/" + urllib.parse.quote(query, safe=""),
              wait_until="domcontentloaded")
    page.wait_for_timeout(6000)
    rows = page.locator("tr.zA:visible", has_text=must)
    if rows.count() != 1:
        print("ABORT: rows matching the subject =", rows.count()); ctx.close(); sys.exit(3)
    rows.first.click()
    page.wait_for_timeout(5000)
    # an earlier unsent reply would be reopened here and a second one stacked on it
    if page.locator(EDITOR + ":visible").count():
        print("ABORT: the thread already holds an open draft; remove it first"); ctx.close(); sys.exit(5)
    n_msgs = page.locator("div.adn:visible").count()
    btn = page.locator('span[role="link"]:visible', has_text=re.compile(r"^Reply$"))
    if not btn.count():
        btn = page.locator('div[role="button"][aria-label="Reply"]:visible')
    btn.last.click()
    page.wait_for_timeout(3000)
    ed = page.locator(EDITOR + ":visible").last
    ed.click()
    for i, para in enumerate(paras):
        page.keyboard.insert_text(para)
        page.keyboard.press("Enter"); page.keyboard.press("Enter")
    for j, line in enumerate(sig_lines):
        page.keyboard.insert_text(line)
        if j < len(sig_lines) - 1:
            page.keyboard.press("Enter")
    for f in ATTACH:
        page.locator('input[type="file"][name="Filedata"]').last.set_input_files(f)
        page.wait_for_timeout(6000)
    page.wait_for_timeout(2000)
    box = page.locator("div.M9:visible, div.ip.iq:visible").last  # the reply's compose block
    rcpt = box.evaluate("b => [...b.querySelectorAll('[email]')].map(e => e.getAttribute('email'))")
    att = box.evaluate("b => [...b.querySelectorAll('[aria-label^=\"Attachment\"], .dL, .vI')].map(e => e.innerText.split(String.fromCharCode(10))[0]).filter(Boolean)")
    btxt = ed.inner_text()
    subj = page.locator("h2.hP").first.inner_text()
    print("thread subject:", subj, "| messages in thread:", n_msgs)
    print("reply recipients:", sorted(set(rcpt)), "| attachments seen:", att, "| body chars:", len(btxt), "of", len(body))
    print("----- EDITOR TEXT -----\n" + btxt + "\n-----------------------")
    page.screenshot(path=SHOT)
    others = sorted(set(r for r in rcpt if r and r != expect_to and r != env["user_gmail_dan_drive"]))
    ok = (expect_to in rcpt and not others and len(btxt) > 0.95 * len(body)
          and (not ATTACH or len(att) >= len(ATTACH)))
    if not ok:
        print("ABORT: checks failed (unexpected recipients:", others, ")")
    if ok and SEND:
        box.locator('div[role="button"][aria-label^="Send"]:visible').last.click()
        page.wait_for_timeout(7000)
        print("messages in thread after send:", page.locator("div.adn:visible").count(), "(before:", n_msgs, ")")
    else:
        page.locator('div[role="button"][aria-label^="Discard draft"]:visible').last.click()
        page.wait_for_timeout(3000)
        print("INSPECTION ONLY — reply draft discarded, nothing sent")
    ctx.close()
    sys.exit(0 if ok else 4)

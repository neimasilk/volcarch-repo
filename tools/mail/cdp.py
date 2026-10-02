"""Small helper: run one action against a tab of the PI's logged-in portal browser (CDP :9333).

usage: python cdp.py <url-substring> <action> [arg]
actions: goto URL | shot NAME | text | eval JS | click SELECTOR
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
sub, action = sys.argv[1], sys.argv[2]
arg = sys.argv[3] if len(sys.argv) > 3 else None
with sync_playwright() as p:
    b = p.chromium.connect_over_cdp("http://localhost:9333")
    pages = [pg for pg in b.contexts[0].pages if sub in pg.url]
    pg = pages[0]
    if action == "goto":
        pg.goto(arg, wait_until="domcontentloaded"); pg.wait_for_timeout(4000); print(pg.url, "|", pg.title())
    elif action == "shot":
        pg.screenshot(path=str(HERE / f"{arg}.png"), full_page=False); print("saved", arg)
    elif action == "text":
        print(pg.inner_text("body")[:6000])
    elif action == "eval":
        print(pg.evaluate(arg))
    elif action == "click":
        pg.locator(arg).first.click(); pg.wait_for_timeout(3000); print("clicked", pg.url)

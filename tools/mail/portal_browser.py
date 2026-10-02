"""Keep a headed Chrome open (CDP on :9333) for the PI to log in to ArchCalc and Zenodo.

Later scripts attach with chromium.connect_over_cdp("http://localhost:9333") and act inside the
PI's logged-in session. Closes itself after 3 hours.
"""
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PROFILE = Path(__file__).parent / "portal_profile"
with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(
        str(PROFILE), channel="chrome", headless=False,
        args=["--remote-debugging-port=9333", "--disable-blink-features=AutomationControlled"],
        viewport={"width": 1300, "height": 880})
    a = ctx.pages[0] if ctx.pages else ctx.new_page()
    a.goto("https://submission.archcalc.cnr.it/login", wait_until="domcontentloaded")
    z = ctx.new_page()
    z.goto("https://zenodo.org/login/", wait_until="domcontentloaded")
    print("browser ready", flush=True)
    t0 = time.time()
    while time.time() - t0 < 3 * 3600 and ctx.pages:
        time.sleep(20)
    ctx.close()

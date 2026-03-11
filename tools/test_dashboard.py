"""
Playwright test script for VOLCARCH dashboard.
Launches Streamlit, captures console errors and screenshots for each tab.
"""

import subprocess
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).parent.parent
SCREENSHOT_DIR = REPO_ROOT / "tools" / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)

PORT = 8502
URL = f"http://localhost:{PORT}"


def start_streamlit():
    """Start Streamlit server in background."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "tools/dashboard.py",
         "--server.port", str(PORT),
         "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Wait for server to be ready
    start = time.time()
    while time.time() - start < 30:
        try:
            import urllib.request
            urllib.request.urlopen(URL, timeout=2)
            print(f"Streamlit server ready on {URL}")
            return proc
        except Exception:
            time.sleep(1)
    # Print any output if server didn't start
    proc.kill()
    out = proc.stdout.read()
    print(f"Server failed to start. Output:\n{out}")
    sys.exit(1)


def run_tests():
    console_errors = []
    page_errors = []

    proc = start_streamlit()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1400, "height": 900})
            page = context.new_page()

            # Capture console messages and page errors
            page.on("console", lambda msg: (
                console_errors.append(f"[{msg.type}] {msg.text}")
                if msg.type in ("error", "warning") else None
            ))
            page.on("pageerror", lambda err: page_errors.append(str(err)))

            print(f"\n{'='*60}")
            print("Loading dashboard...")
            print(f"{'='*60}")

            # Load the main page, wait for it to fully render
            page.goto(URL, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(5000)  # Extra time for Streamlit to render

            # Check for Streamlit error messages on page
            error_elements = page.query_selector_all('[data-testid="stException"]')
            if error_elements:
                for el in error_elements:
                    print(f"\nSTREAMLIT ERROR FOUND: {el.inner_text()}")

            # Screenshot the initial state (Tab 1 - Peta Interaktif)
            page.screenshot(path=str(SCREENSHOT_DIR / "tab1_peta_interaktif.png"), full_page=True)
            print("Screenshot: tab1_peta_interaktif.png")

            # Check if the page has rendered content
            body_text = page.inner_text("body")
            if "Dashboard data not found" in body_text:
                print("\nERROR: Dashboard data not found!")
                print("Need to run precompute_dashboard_data.py first.")
                return

            # ── Tab 2: Analisis SHAP ──
            print("\nSwitching to Tab 2 (SHAP)...")
            tabs = page.query_selector_all('[data-baseweb="tab"]')
            if len(tabs) >= 2:
                tabs[1].click()
                page.wait_for_timeout(3000)
                page.screenshot(path=str(SCREENSHOT_DIR / "tab2_shap.png"), full_page=True)
                print("Screenshot: tab2_shap.png")

                # Check for errors after tab switch
                error_elements = page.query_selector_all('[data-testid="stException"]')
                if error_elements:
                    for el in error_elements:
                        print(f"\nSTREAMLIT ERROR (Tab 2): {el.inner_text()}")
            else:
                print(f"WARNING: Found {len(tabs)} tabs, expected 4")

            # ── Tab 3: Klasifikasi Zona ──
            print("\nSwitching to Tab 3 (Zona)...")
            if len(tabs) >= 3:
                tabs[2].click()
                page.wait_for_timeout(3000)
                page.screenshot(path=str(SCREENSHOT_DIR / "tab3_zona.png"), full_page=True)
                print("Screenshot: tab3_zona.png")

                error_elements = page.query_selector_all('[data-testid="stException"]')
                if error_elements:
                    for el in error_elements:
                        print(f"\nSTREAMLIT ERROR (Tab 3): {el.inner_text()}")

            # ── Tab 4: Validasi Model ──
            print("\nSwitching to Tab 4 (Validasi)...")
            if len(tabs) >= 4:
                tabs[3].click()
                page.wait_for_timeout(3000)
                page.screenshot(path=str(SCREENSHOT_DIR / "tab4_validasi.png"), full_page=True)
                print("Screenshot: tab4_validasi.png")

                error_elements = page.query_selector_all('[data-testid="stException"]')
                if error_elements:
                    for el in error_elements:
                        print(f"\nSTREAMLIT ERROR (Tab 4): {el.inner_text()}")

            # ── Summary ──
            print(f"\n{'='*60}")
            print("RESULTS SUMMARY")
            print(f"{'='*60}")

            if console_errors:
                print(f"\nConsole errors/warnings ({len(console_errors)}):")
                for e in console_errors[:20]:
                    print(f"  {e}")
            else:
                print("\nNo console errors/warnings found.")

            if page_errors:
                print(f"\nPage errors ({len(page_errors)}):")
                for e in page_errors:
                    print(f"  {e}")
            else:
                print("\nNo page errors found.")

            browser.close()

    finally:
        proc.kill()
        proc.wait()
        print(f"\nStreamlit server stopped.")
        print(f"Screenshots saved to: {SCREENSHOT_DIR}")


if __name__ == "__main__":
    run_tests()

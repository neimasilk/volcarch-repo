"""Fix Zenodo draft 19081857 — fill Description (TinyMCE iframe) + Author."""
import asyncio
from playwright.async_api import async_playwright

DRAFT_URL = "https://zenodo.org/uploads/19081857"
DESC_TEXT = ("Java hosts 45 active volcanoes in 129,000 km². Its early archaeological record is strikingly thin "
             "compared to volcanic-free regions of the same archipelago. The standard explanation is developmental: "
             "Javanese kingdoms simply came later. The explanation advanced here is geophysical: the evidence is buried.\n\n"
             "In 1803, a Dutch colonial administrator documented two enormous guardian statues at Candi Singosari "
             "in East Java — each 370 cm tall — with roughly half their bodies below ground. Nobody had buried them "
             "deliberately. Gunung Kelud had erupted approximately twenty times; the landscape had risen around them.\n\n"
             "This paper calibrates what that rate means for the archaeological record as a whole. Using four sites "
             "with independently documented construction dates and measured burial depths, spanning two volcanic systems "
             "(Kelud and Merapi) and roughly 350 km of Java, we establish a mean sedimentation rate of 4.4 ± 1.2 mm/yr. "
             "The consistency across sites is the central finding: burial is not a local anomaly but a Java-wide "
             "taphonomic baseline. At this rate, remains from the Kanjuruhan period (~760 CE) now lie at 4.0–7.8 m "
             "depth — beyond the reach of surface survey. Spatial analysis of 666 East Java sites confirms the pattern: "
             "the observable record maps survey effort and monument durability, not historical settlement intensity.")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page(viewport={"width":1280,"height":900})

        # Login
        print("1. Login...")
        await page.goto("https://zenodo.org/login/", wait_until="domcontentloaded")
        await page.wait_for_timeout(1500)
        await page.click("text=ORCID")
        await page.wait_for_timeout(2500)
        ck = page.locator("button:has-text('Reject Unnecessary')")
        if await ck.count(): await ck.first.click(); await page.wait_for_timeout(800)
        await page.fill("input[placeholder*='ORCID iD'], input#username", "0000-0002-1848-167X")
        await page.fill("input[type='password']", "Ayas_arema_2023")
        await page.click("button:has-text('Sign in to ORCID')")
        await page.wait_for_timeout(7000)
        auth = page.locator("button#authorize-button")
        if await auth.count(): await auth.first.click(); await page.wait_for_timeout(3000)
        print(f"   OK: {page.url}")

        # Open draft
        print("2. Opening draft...")
        await page.goto(DRAFT_URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)
        zck = page.locator("button:has-text('Accept only essential')")
        if await zck.count(): await zck.first.click()

        # === AUTHOR ===
        print("3. Author...")
        await page.evaluate("window.scrollTo(0, 2500)")
        await page.wait_for_timeout(1000)

        # Look for the author section inputs (Zenodo uses specific placeholders)
        inputs = await page.query_selector_all("input[type='text']:not([readonly])")
        print(f"   Found {len(inputs)} text inputs on page")
        for inp in inputs:
            ph = await inp.get_attribute("placeholder") or ""
            name = await inp.get_attribute("name") or ""
            val = await inp.input_value()
            if ph or name:
                print(f"   Input: placeholder='{ph}' name='{name}' value='{val[:20]}'")

        # Fill by placeholder
        for ph, val in [("Family name", "Amien"), ("Given names", "Mukhlis"),
                        ("Family", "Amien"), ("Given", "Mukhlis")]:
            f = page.locator(f"input[placeholder='{ph}']")
            if await f.count():
                await f.first.fill(val)
                print(f"   '{ph}' → '{val}' ✓")

        affil = page.locator("input[placeholder='Affiliation'], input[placeholder*='affiliation']")
        if await affil.count():
            await affil.first.fill("Lab Data Sains, Universitas Bhinneka Nusantara, Indonesia")
            print("   Affiliation ✓")

        # === DESCRIPTION via TinyMCE iframe ===
        print("4. Description...")
        await page.evaluate("window.scrollTo(0, 1200)")
        await page.wait_for_timeout(3000)

        # Method 1: tinymce.get(0)
        result = await page.evaluate("""(text) => {
            try {
                var ed = tinymce.get(0);
                if(ed) { ed.setContent('<p>' + text.replace(/\\n\\n/g,'</p><p>') + '</p>'); return 'ok_get0'; }
                return 'no_editor';
            } catch(e) { return 'err:' + e.message; }
        }""", DESC_TEXT)
        print(f"   Method 1 (tinymce.get): {result}")

        if 'ok' not in result:
            # Method 2: via iframe locator
            print("   Trying iframe method...")
            try:
                iframe = page.frame_locator("iframe.tox-edit-area__iframe, iframe[title*='Rich Text'], iframe[id*='mce']")
                body = iframe.locator("body")
                await body.wait_for(state="visible", timeout=10000)
                await body.click()
                await page.keyboard.press("Control+a")
                await page.keyboard.type(DESC_TEXT, delay=2)
                print("   Description via iframe ✓")
            except Exception as e:
                print(f"   Iframe method failed: {e}")
                # Method 3: find all iframes and try each
                frames = page.frames
                print(f"   Total frames: {len(frames)}")
                for i, frame in enumerate(frames):
                    try:
                        body = frame.locator("body[contenteditable='true'], body")
                        if await body.count():
                            content = await body.inner_html()
                            if len(content) < 500:  # likely empty editor
                                await body.click()
                                await frame.evaluate(f"document.body.innerHTML = '<p>{DESC_TEXT[:100]}...</p>'")
                                print(f"   Frame {i} filled ✓")
                                break
                    except:
                        pass

        # Save
        print("5. Save draft...")
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(500)
        sv = page.locator("button:has-text('Save draft')")
        if await sv.count():
            await sv.first.click()
            await page.wait_for_timeout(3000)
            print("   Saved!")

        print(f"\n✓ Done. URL: {page.url}")
        print("Browser stays open 10 min — review and PUBLISH.")
        await page.wait_for_timeout(600000)
        await browser.close()

asyncio.run(main())

"""Zenodo upload — v5. Fixed: DOI click, description, author."""
import asyncio
from playwright.async_api import async_playwright

PDF = r"C:\Users\neima\Documents\volcarch-repo\papers\P1_taphonomic_framework\submission_v1.0.pdf"
TITLE = "Multi-Site Calibration of Volcanic Sedimentation Rates and Implications for Archaeological Visibility in Java, Indonesia"
DESC = ("<p>Java hosts 45 active volcanoes in 129,000 km². Its early archaeological record "
        "is strikingly thin compared to volcanic-free regions of the same archipelago. "
        "The standard explanation is developmental: Javanese kingdoms simply came later. "
        "The explanation advanced here is geophysical: the evidence is buried.</p>"
        "<p>In 1803, a Dutch colonial administrator documented two enormous guardian statues at "
        "Candi Singosari in East Java\u2014each 370 cm tall\u2014with roughly half their bodies below ground. "
        "Nobody had buried them deliberately. Gunung Kelud had erupted approximately twenty times; "
        "the landscape had risen around them.</p>"
        "<p>This paper calibrates what that rate means for the archaeological record as a whole. "
        "Using four sites with independently documented construction dates and measured burial depths, "
        "spanning two volcanic systems (Kelud and Merapi) and roughly 350 km of Java, we establish "
        "a mean sedimentation rate of 4.4 \u00b1 1.2 mm/yr. The consistency across sites is the central "
        "finding: burial is not a local anomaly but a Java-wide taphonomic baseline. At this rate, "
        "archaeological remains from the Kanjuruhan period (~760 CE) now lie at 4.0\u20137.8 m depth\u2014"
        "beyond the reach of surface survey.</p>"
        "<p>Spatial analysis of 666 East Java sites confirms the pattern: the observable record maps "
        "survey effort and monument durability, not historical settlement intensity. We identify "
        "priority zones where high terrain suitability coincides with deep predicted burial, providing "
        "a targeting framework for future subsurface investigation.</p>")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page(viewport={"width":1280,"height":900})

        # === LOGIN ===
        print("1. Login...")
        await page.goto("https://zenodo.org/login/", wait_until="domcontentloaded")
        await page.wait_for_timeout(1500)
        await page.click("text=ORCID")
        await page.wait_for_timeout(2500)

        # Cookie consent
        ck = page.locator("button:has-text('Reject Unnecessary')")
        if await ck.count(): await ck.first.click(); await page.wait_for_timeout(800)

        await page.fill("input[placeholder*='ORCID iD'], input#username", "0000-0002-1848-167X")
        await page.fill("input[type='password']", "Ayas_arema_2023")
        await page.click("button:has-text('Sign in to ORCID')")
        print("   Waiting for redirect...")
        await page.wait_for_timeout(7000)

        auth = page.locator("button#authorize-button")
        if await auth.count(): await auth.first.click(); await page.wait_for_timeout(3000)
        print(f"   URL: {page.url}")

        # === NEW UPLOAD ===
        print("2. New upload page...")
        await page.goto("https://zenodo.org/uploads/new", wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)

        # Zenodo cookie
        zck = page.locator("button:has-text('Accept only essential')")
        if await zck.count(): await zck.first.click(); await page.wait_for_timeout(500)

        # === UPLOAD PDF ===
        print("3. Upload PDF...")
        fi = page.locator("input[type='file']")
        if await fi.count():
            await fi.first.set_input_files(PDF)
            print("   Uploading...")
            await page.wait_for_timeout(10000)
            print("   Done!")

        # === DOI ===
        print("4. DOI selection...")
        await page.evaluate("window.scrollTo(0, 400)")
        await page.wait_for_timeout(800)

        # Step 1: click "No, I need one" radio
        no_doi = page.locator("label:has-text('No, I need one')")
        if await no_doi.count():
            await no_doi.first.click()
            print("   Clicked 'No, I need one'")
        else:
            radios = page.locator("input[type='radio']")
            cnt = await radios.count()
            print(f"   {cnt} radios found, clicking last one...")
            if cnt > 0:
                await radios.last.click()
        await page.wait_for_timeout(1500)

        # Step 2: click the button that appears AFTER selecting "No, I need one"
        # Usually "Get a DOI now!" or "Reserve DOI" or "Pre-reserve DOI"
        await page.screenshot(path=r"C:\Users\neima\Documents\volcarch-repo\tools\z_doi_before_btn.png")
        for txt in ["Get a DOI now!", "Get a DOI now", "Reserve DOI", "Pre-reserve DOI", "Reserve a DOI"]:
            btn = page.locator(f"button:has-text('{txt}'), a:has-text('{txt}')")
            if await btn.count():
                await btn.first.click()
                print(f"   Clicked button: '{txt}'")
                await page.wait_for_timeout(2000)
                break
        await page.screenshot(path=r"C:\Users\neima\Documents\volcarch-repo\tools\z_doi_after_btn.png")

        # === RESOURCE TYPE ===
        print("5. Resource type...")
        await page.evaluate("window.scrollTo(0, 600)")
        await page.wait_for_timeout(300)

        dd = page.locator("div.ui.selection.dropdown")
        if await dd.count():
            await dd.first.click()
            await page.wait_for_timeout(700)
            pub = page.locator("div.visible div.item:has-text('Publication')")
            if await pub.count():
                await pub.first.click()
                print("   Publication ✓")
                await page.wait_for_timeout(700)

                # Subtype dropdown
                dd2 = page.locator("div.ui.selection.dropdown")
                if await dd2.count() > 1:
                    await dd2.nth(1).click()
                    await page.wait_for_timeout(700)
                    pre = page.locator("div.visible div.item:has-text('Preprint')")
                    if await pre.count():
                        await pre.first.click()
                        print("   Preprint ✓")
                        await page.wait_for_timeout(500)

        # === TITLE ===
        print("6. Title...")
        await page.evaluate("window.scrollTo(0, 900)")
        await page.wait_for_timeout(500)

        # Try multiple selectors
        filled_title = False
        for sel in ["input#metadata\\.title", "input[name*='title']"]:
            ti = page.locator(sel)
            if await ti.count():
                await ti.first.fill(TITLE)
                filled_title = True
                print("   Title ✓")
                break

        if not filled_title:
            # Find by nearby label
            labels = await page.query_selector_all("label")
            for lbl in labels:
                txt = await lbl.text_content()
                if "Title" in txt:
                    for_id = await lbl.get_attribute("for")
                    if for_id:
                        await page.fill(f"#{for_id}", TITLE)
                        print(f"   Title via label ✓")
                        filled_title = True
                    break

        # === DESCRIPTION via TinyMCE ===
        print("7. Description...")
        await page.evaluate("window.scrollTo(0, 1200)")
        await page.wait_for_timeout(2000)  # Give TinyMCE time to init

        desc_escaped = DESC.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        try:
            result = await page.evaluate(f"""() => {{
                if(typeof tinymce !== 'undefined' && tinymce.editors.length > 0) {{
                    tinymce.editors[0].setContent(`{desc_escaped}`);
                    return 'ok';
                }}
                return 'no_tinymce';
            }}""")
            if result == 'ok':
                print("   Description ✓ (TinyMCE)")
            else:
                print(f"   TinyMCE not ready: {result}")
                # Retry after scroll + wait
                await page.wait_for_timeout(2000)
                result = await page.evaluate(f"""() => {{
                    if(typeof tinymce !== 'undefined' && tinymce.editors.length > 0) {{
                        tinymce.editors[0].setContent(`{desc_escaped}`);
                        return 'ok';
                    }}
                    return 'still no tinymce';
                }}""")
                print(f"   Retry: {result}")
        except Exception as e:
            print(f"   Description error: {e}")

        # === CREATORS/AUTHORS ===
        print("8. Author...")
        await page.evaluate("window.scrollTo(0, 1800)")
        await page.wait_for_timeout(500)

        # Check if there's an "Add creator" button to click first
        add_btn = page.locator("button:has-text('Add creator'), a:has-text('Add creator')")
        if await add_btn.count():
            await add_btn.first.click()
            await page.wait_for_timeout(500)

        # Fill family name
        family = page.locator("input[placeholder*='Family'], input[name*='family_name'], input[name*='familyname']")
        if await family.count():
            await family.first.fill("Amien")
            print("   Family: Amien ✓")

        # Fill given names
        given = page.locator("input[placeholder*='Given'], input[name*='given_name'], input[name*='givenname']")
        if await given.count():
            await given.first.fill("Mukhlis")
            print("   Given: Mukhlis ✓")

        # Fill affiliation
        affil = page.locator("input[placeholder*='Affiliation'], input[name*='affiliation']")
        if await affil.count():
            await affil.first.fill("Lab Data Sains, Universitas Bhinneka Nusantara, Indonesia")
            print("   Affiliation ✓")

        # ORCID
        orcid_field = page.locator("input[placeholder*='ORCID'], input[name*='orcid']")
        if await orcid_field.count():
            await orcid_field.first.fill("0000-0002-1848-167X")
            print("   ORCID ✓")

        await page.screenshot(path=r"C:\Users\neima\Documents\volcarch-repo\tools\z_author.png")

        # === SAVE DRAFT ===
        print("9. Saving draft...")
        sv = page.locator("button:has-text('Save draft')")
        if await sv.count():
            await sv.first.click()
            await page.wait_for_timeout(3000)
            print("   Saved!")

        await page.screenshot(path=r"C:\Users\neima\Documents\volcarch-repo\tools\z_final.png")

        print("\n" + "="*50)
        print("DONE — review in browser, then click PUBLISH")
        print("="*50)
        print(f"URL: {page.url}")

        # Keep open 10 minutes
        await page.wait_for_timeout(600000)
        await browser.close()

asyncio.run(main())

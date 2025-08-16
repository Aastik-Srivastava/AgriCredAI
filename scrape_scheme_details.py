import json
from playwright.sync_api import sync_playwright

scheme_urls = [
    "https://www.myscheme.gov.in/schemes/pm-kisan",
    "https://www.myscheme.gov.in/schemes/kcc",
    "https://www.myscheme.gov.in/schemes/pmfby",
    "https://www.myscheme.gov.in/schemes/nai",  # maybe NAI
    "https://www.myscheme.gov.in/schemes/kcc",
    "https://www.myscheme.gov.in/schemes/isac",
    "https://www.myscheme.gov.in/schemes/pm-kusum",
    "https://www.myscheme.gov.in/schemes/kvps",
    "https://www.myscheme.gov.in/schemes/pmksypdmc",
    "https://www.myscheme.gov.in/schemes/rkvyshfshc",
    "https://www.myscheme.gov.in/schemes/ky-smsp",
    "https://www.myscheme.gov.in/schemes/cpis",
    "https://www.myscheme.gov.in/schemes/ami"
]

all_schemes = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    for url in scheme_urls:
        print(f"Scraping: {url}")
        page.goto(url, timeout=60000)
        page.wait_for_load_state("networkidle")

        # Use a more specific selector and .first
        title_locator = page.locator("h1.font-bold").first
        description_locator = page.locator(".scheme-detail-desc").first
        benefits_locator = page.locator("#benefits").first
        eligibility_locator = page.locator("#eligibility").first

        title = title_locator.inner_text() if title_locator.count() > 0 else "N/A"
        description = description_locator.inner_text() if description_locator.count() > 0 else "N/A"
        benefits = benefits_locator.inner_text() if benefits_locator.count() > 0 else "N/A"
        eligibility = eligibility_locator.inner_text() if eligibility_locator.count() > 0 else "N/A"

        all_schemes.append({
            "url": url,
            "title": title.strip(),
            "description": description.strip(),
            "benefits": benefits.strip(),
            "eligibility": eligibility.strip()
        })

    browser.close()

with open("myschemes_full.json", "w", encoding="utf-8") as f:
    json.dump(all_schemes, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(all_schemes)} schemes to myschemes_full.json")

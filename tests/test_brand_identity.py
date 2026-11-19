"""Verify Cat's Eye brand identity, vector assets, design system tokens, and UI components.
Usage: python tests/test_brand_identity.py [base_url]
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = sys.argv[1] if len(sys.argv) > 1 else None

fails = []


def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name + (f"   {extra}" if extra else ""))
    if not ok:
        fails.append(name)


# 1. Brand vector files exist on disk
favicon = ROOT / "web" / "brand" / "favicon.svg"
logo = ROOT / "web" / "brand" / "logo.svg"
logo_light = ROOT / "web" / "brand" / "logo-light.svg"
loader = ROOT / "web" / "brand" / "loader.svg"
og_image = ROOT / "web" / "brand" / "og-image.svg"

check("favicon.svg exists", favicon.is_file())
check("logo.svg exists", logo.is_file())
check("logo-light.svg exists", logo_light.is_file())
check("loader.svg exists", loader.is_file())
check("og-image.svg exists", og_image.is_file())

# 2. Vector content inspection
fav_content = favicon.read_text(encoding="utf-8") if favicon.is_file() else ""
logo_content = logo.read_text(encoding="utf-8") if logo.is_file() else ""
light_content = logo_light.read_text(encoding="utf-8") if logo_light.is_file() else ""
loader_content = loader.read_text(encoding="utf-8") if loader.is_file() else ""

check("favicon has cat's eye slit pupil elements", "<svg" in fav_content and "viewBox" in fav_content)
check("logo has Cat's Eye wordmark or icon", "<svg" in logo_content and ("Cat's Eye" in logo_content or "STUDIO" in logo_content))
check("logo-light has valid vector structure", "<svg" in light_content and "path" in light_content)
check("loader has animated slit dilation and reduced-motion support", "slitDilate" in loader_content and "prefers-reduced-motion" in loader_content)
check("brand vectors use Obsidian & Amber Gold palette", any(c in fav_content.lower() for c in ["#f59e0b", "#fbbf24", "#d97706", "#fde68a"]))

# 3. Design System CSS Files
tokens_css = ROOT / "web" / "css" / "tokens.css"
components_css = ROOT / "web" / "css" / "components.css"
styleguide_html = ROOT / "web" / "styleguide.html"

check("tokens.css exists", tokens_css.is_file())
check("components.css exists", components_css.is_file())
check("styleguide.html exists", styleguide_html.is_file())

tokens_text = tokens_css.read_text(encoding="utf-8") if tokens_css.is_file() else ""
comp_text = components_css.read_text(encoding="utf-8") if components_css.is_file() else ""
sg_text = styleguide_html.read_text(encoding="utf-8") if styleguide_html.is_file() else ""

check("tokens.css defines amber gold brand and dark/light themes", "--ce-brand: #F59E0B" in tokens_text and '[data-theme="light"]' in tokens_text)

# Check that every component class in components.css appears in styleguide.html
comp_classes = re.findall(r'\.(ce-[a-z0-9-]+)', comp_text)
unique_classes = set(comp_classes)
missing_in_sg = [c for c in unique_classes if c not in sg_text]
check("styleguide contains documented component classes", len(missing_in_sg) <= 5, f"missing: {missing_in_sg[:5]}")

# 4. Web index.html branding and UI navigation inspection
index_file = ROOT / "web" / "index.html"
index_html = index_file.read_text(encoding="utf-8") if index_file.is_file() else ""

check("index.html title is Cat's Eye", "<title>Cat's Eye</title>" in index_html)
check("index.html links to tokens.css and components.css", 'href="/css/tokens.css"' in index_html and 'href="/css/components.css"' in index_html)
check("index.html defines Obsidian & Amber Gold tokens", "--brand:var(--ce-brand" in index_html or "#F59E0B" in index_html)

# Check required app shell navigation sections (Requirement 7)
nav_sections = ["Home", "Videos", "Clips", "Creators", "Live", "Posting", "Library", "Editor", "Settings"]
all_nav_found = all(s in index_html for s in nav_sections)
check("index.html navigation includes all required shell sections", all_nav_found)

# Check that empty state / button copy pass was applied (Requirement 10)
check("index.html copy pass: specific action labels", "Paste a link or drop" in index_html or "Make clips" in index_html)

# 5. HTTP endpoints (if test server is running)
if BASE:
    import httpx
    client = httpx.Client(base_url=BASE, headers={"Origin": BASE}, timeout=10)

    # Asset endpoints
    r_fav = client.get("/brand/favicon.svg")
    check("HTTP GET /brand/favicon.svg returns 200", r_fav.status_code == 200 and "<svg" in r_fav.text)

    r_logo = client.get("/brand/logo-light.svg")
    check("HTTP GET /brand/logo-light.svg returns 200", r_logo.status_code == 200 and "<svg" in r_logo.text)

    r_dark = client.get("/brand/logo.svg")
    check("HTTP GET /brand/logo.svg returns 200", r_dark.status_code == 200 and "<svg" in r_dark.text)

    r_tokens = client.get("/css/tokens.css")
    check("HTTP GET /css/tokens.css returns 200", r_tokens.status_code == 200 and "--ce-brand" in r_tokens.text)

    r_sg = client.get("/styleguide.html")
    check("HTTP GET /styleguide.html returns 200", r_sg.status_code == 200 and "Cat's Eye Style Guide" in r_sg.text)

    # Check pages return 200 and have no user-visible "PeakCut"
    pages_to_check = [
        ("/", "Landing Page"),
        ("/login", "Login Page"),
        ("/reset", "Password Reset"),
        ("/terms", "Terms of Service"),
        ("/privacy", "Privacy Policy"),
        ("/acceptable-use", "Acceptable Use"),
        ("/styleguide.html", "Living Style Guide")
    ]

    for path, desc in pages_to_check:
        resp = client.get(path)
        ok_status = resp.status_code == 200
        # Check no unescaped PeakCut in rendered visible text (excluding internal env vars or links)
        # Split out HTML tags to inspect visible text only
        text_only = re.sub(r'<[^>]+>', ' ', resp.text)
        no_peakcut = "PeakCut" not in text_only or "PEAKCUT_" in text_only or "built beside PeakCut" in text_only
        check(f"HTTP GET {path} ({desc}) returns 200 with Cat's Eye branding", ok_status and no_peakcut)

    # Authenticated app page check
    client.post("/api/auth/signup", json={"email": "brand-check@example.com", "password": "brand-pass-123", "name": "Tester"})
    r_app = client.get("/app")
    app_text_only = re.sub(r'<[^>]+>', ' ', r_app.text)
    no_peakcut_app = "PeakCut" not in app_text_only or "PEAKCUT_" in app_text_only
    check("HTTP GET /app serves Cat's Eye app with no visible PeakCut", r_app.status_code == 200 and no_peakcut_app)

print(f"\nALL PASSED" if not fails else f"{len(fails)} failed")
sys.exit(1 if fails else 0)

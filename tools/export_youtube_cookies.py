#!/usr/bin/env python3
"""
Export YouTube cookies from Edge browser via Playwright CDP.

Edge must be running (or will be launched). Connects via Chrome DevTools Protocol
to read cookies from the logged-in session, then writes them in Netscape format
for yt-dlp consumption.

Usage:
    python tools/export_youtube_cookies.py

Output:
    config/youtube_cookies.txt (Netscape HTTP Cookie File)

Why this exists:
    Edge v127+ uses App-Bound Encryption (v20) for cookies, which prevents
    yt-dlp from reading them directly. This script bypasses that by connecting
    to Edge's own process via CDP, where cookies are already decrypted.
"""
import asyncio
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
COOKIE_FILE = PROJECT_ROOT / "config" / "youtube_cookies.txt"
EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
CDP_PORT = 9222


async def export_cookies():
    from playwright.async_api import async_playwright

    # Launch Edge with remote debugging if not already running on CDP port
    edge_proc = None
    try:
        import urllib.request
        urllib.request.urlopen(f"http://localhost:{CDP_PORT}/json/version", timeout=2)
    except Exception:
        user_data = str(Path.home() / "AppData/Local/Microsoft/Edge/User Data")
        edge_proc = subprocess.Popen([
            EDGE_PATH,
            f"--remote-debugging-port={CDP_PORT}",
            f"--user-data-dir={user_data}",
            "--no-first-run",
            "https://www.youtube.com",
        ])
        time.sleep(5)

    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(f"http://localhost:{CDP_PORT}")
        context = browser.contexts[0]

        cookies = await context.cookies([
            "https://www.youtube.com",
            "https://accounts.google.com",
            "https://google.com",
        ])

        lines = [
            "# Netscape HTTP Cookie File",
            f"# Exported via Playwright CDP from Edge ({len(cookies)} cookies)",
            "",
        ]
        for c in cookies:
            domain = c["domain"]
            include_sub = "TRUE" if domain.startswith(".") else "FALSE"
            path = c.get("path", "/")
            secure = "TRUE" if c.get("secure") else "FALSE"
            expires = max(0, int(c.get("expires", 0)))
            name = c["name"]
            value = c["value"]
            lines.append(f"{domain}\t{include_sub}\t{path}\t{secure}\t{expires}\t{name}\t{value}")

        COOKIE_FILE.write_text("\n".join(lines), encoding="utf-8")
        print(f"Exported {len(cookies)} cookies to {COOKIE_FILE}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(export_cookies())

#!/usr/bin/env python3
"""Run Playwright outside a notebook event loop on Windows."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from playwright.sync_api import sync_playwright


def collect_pages(index_url: str) -> dict[str, Any]:
    urls: set[str] = set()
    anchors: list[dict[str, str]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page()
        page.goto(index_url, wait_until="domcontentloaded", timeout=120000)
        try:
            page.wait_for_load_state("networkidle", timeout=30000)
        except Exception:
            pass
        page.wait_for_timeout(5000)
        anchor_data = page.locator("a").evaluate_all(
            """
            elements => elements.map(a => ({
                text: (a.innerText || a.textContent || "").trim(),
                href: a.href
            }))
            """
        )
        anchors.extend(anchor_data)
        browser.close()

    for item in anchor_data:
        href = item.get("href", "")
        if not href:
            continue
        href = href.split("#")[0].split("?")[0]
        if "files_html/" in href and href.lower().endswith(".html"):
            urls.add(href)

    return {"urls": sorted(urls), "anchors": anchors}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-url", default="https://lingconlab.ru/pinega/#!/fulltexts")
    parser.add_argument("--discover-url")
    args = parser.parse_args()

    if args.discover_url:
        print(json.dumps(collect_pages(args.discover_url), ensure_ascii=False))
        return

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page()
        page.goto(args.test_url, wait_until="domcontentloaded", timeout=120000)
        print("Chromium успешно запущен!")
        print("Заголовок страницы:", page.title())
        browser.close()


if __name__ == "__main__":
    main()

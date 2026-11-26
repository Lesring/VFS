"""验证 Google News RSS 与 Suggest 的解析质量（免 key 入口）。"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read().decode("utf-8", "ignore")


def test_news(q: str) -> None:
    print("=" * 60)
    print("News RSS:", q)
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl=en-US&gl=US&ceid=US:en"
    try:
        xml = fetch(url)
        root = ET.fromstring(xml)
        items = root.findall(".//item")
        print("items:", len(items))
        for it in items[:4]:
            title = (it.findtext("title") or "").strip()
            link = (it.findtext("link") or "").strip()
            desc = re.sub(r"<[^>]+>", "", it.findtext("description") or "")[:120]
            date = it.findtext("pubDate") or ""
            print(f"  - {title[:80]}")
            print(f"    {date} | {desc}")
            print(f"    {link[:80]}")
    except Exception as e:
        print("ERR", type(e).__name__, e)


def test_suggest(q: str) -> None:
    print("=" * 60)
    print("Suggest:", q)
    url = f"https://suggestqueries.google.com/complete/search?client=firefox&q={urllib.parse.quote(q)}"
    try:
        data = json.loads(fetch(url))
        print("query:", data[0])
        for s in data[1][:8]:
            print("  ·", s)
    except Exception as e:
        print("ERR", type(e).__name__, e)


def main() -> int:
    test_news("MU5101 China Eastern")
    test_news("Shanghai Pudong Airport plane spotting")
    test_suggest("Airbus A3")
    test_suggest("上海浦东机场 飞机")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

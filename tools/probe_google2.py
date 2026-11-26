"""检查 Google 返回的到底是真实搜索结果页，还是同意页/拦截页。"""
from __future__ import annotations

import re
import urllib.request

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"


def fetch(url: str) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    })
    with urllib.request.urlopen(req, timeout=8) as r:
        return r.status, r.read().decode("utf-8", "ignore")


def main() -> int:
    for url in [
        "https://www.google.com/search?q=MU5101+flight+destination&hl=en&num=10",
        "https://www.google.com/search?q=Airbus+A320&hl=zh-CN&num=10",
    ]:
        print("=" * 70)
        print(url)
        try:
            status, html = fetch(url)
        except Exception as e:
            print("ERR", e)
            continue
        print("status", status, "len", len(html))
        # 判断页面类型
        markers = {
            "consent_page": "consent" in html.lower() and "before you continue" in html.lower(),
            "captcha": "captcha" in html.lower() or "unusual traffic" in html.lower(),
            "sorry_page": "/sorry/" in html,
            "has_h3": html.count("<h3"),
            "has_div_g": len(re.findall(r'<div class="[^"]*\bMjjYud\b', html)) or len(re.findall(r"<div class=\"g\"", html)),
            "js_required": "enablejs" in html.lower() or "javascript" in html.lower()[:3000],
        }
        print(markers)
        # 提取 h3 文本看是否是搜索结果
        h3s = re.findall(r"<h3[^>]*>(.*?)</h3>", html, re.S)
        clean = []
        for h in h3s[:5]:
            t = re.sub(r"<[^>]+>", "", h).strip()
            if t:
                clean.append(t[:70])
        print("h3 样本:", clean)
        print("前 300 字符:", re.sub(r"\s+", " ", html[:300]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

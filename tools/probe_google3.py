"""测试 Google 地图（静态图/地理编码）在不带 key、或 key 无效时的返回，
以及 Google 新闻 RSS 等免 key 入口是否可用。
"""
from __future__ import annotations

import re
import urllib.error
import urllib.request

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"


def probe(name: str, url: str, binary: bool = False) -> None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
        with urllib.request.urlopen(req, timeout=8) as r:
            body = r.read(2000)
            if binary:
                print(f"OK   {name:22s} {r.status} {len(body)}B type={r.headers.get('content-type')}")
            else:
                txt = body.decode("utf-8", "ignore")
                print(f"OK   {name:22s} {r.status} {txt[:180]}")
    except urllib.error.HTTPError as e:
        try:
            body = e.read(300).decode("utf-8", "ignore")
        except Exception:
            body = ""
        print(f"HTTP {e.code} {name:22s} {body[:170]}")
    except Exception as e:
        print(f"FAIL {name:22s} {type(e).__name__}: {str(e)[:60]}")


def main() -> int:
    # 静态地图：不带 key 会返回一张"错误占位图"（仍是图片），说明端点通
    probe("staticmap_nokey", "https://maps.googleapis.com/maps/api/staticmap?center=31.14,121.80&zoom=12&size=300x200", binary=True)
    # 地理编码：不带 key
    probe("geocode_nokey", "https://maps.googleapis.com/maps/api/geocode/json?address=Shanghai+Pudong+Airport")
    # Google 新闻 RSS（免 key）
    probe("news_rss", "https://news.google.com/rss/search?q=MU5101")
    # Google 建议词（免 key，可用于实体纠错）
    probe("suggest", "https://suggestqueries.google.com/complete/search?client=firefox&q=Shanghai+Pudong+Air")
    # Google 自定义搜索 HTML 无 key
    probe("cse_html", "https://cse.google.com/cse?q=test")
    # 注意：OSM 作为 Google Maps 的免费替代
    probe("osm_nominatim", "https://nominatim.openstreetmap.org/search?q=Shanghai+Pudong+Airport&format=json&limit=1")
    probe("osm_tile", "https://tile.openstreetmap.org/10/857/418.png", binary=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""检查各数据源返回结构，确保解析器写对。"""
from __future__ import annotations

import json
import re
import urllib.request

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def fetch(url: str, timeout: int = 10) -> str:
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


section("1. Wikimedia REST search")
try:
    d = json.loads(fetch("https://api.wikimedia.org/core/v1/wikipedia/en/search/page?q=Airbus%20A320&limit=2"))
    print(json.dumps(d, ensure_ascii=False)[:900])
except Exception as e:
    print("ERR", e)

section("2. Wikimedia REST summary")
try:
    d = json.loads(fetch("https://api.wikimedia.org/core/v1/wikipedia/en/page/Airbus_A320"))
    for k in ("title", "description", "extract", "extract_html"):
        v = d.get(k)
        if v:
            print(f"{k}: {str(v)[:220]}")
except Exception as e:
    print("ERR", e)

section("3. Wikidata entity search")
try:
    d = json.loads(fetch("https://www.wikidata.org/w/api.php?action=wbsearchentities&search=China%20Eastern&language=en&format=json&limit=2"))
    print(json.dumps(d, ensure_ascii=False)[:700])
except Exception as e:
    print("ERR", e)

section("4. Bing 搜索结果结构")
try:
    html = fetch("https://cn.bing.com/search?q=MU5101%20flight%20destination&setlang=zh-cn")
    print("len", len(html))
    for m in re.finditer(r'<li class="b_algo".*?</li>', html, re.S)[:0] if False else []:
        pass
    items = re.findall(r'<li class="b_algo".*?</li>', html, re.S)
    print("b_algo count", len(items))
    if items:
        print(items[0][:1200])
except Exception as e:
    print("ERR", e)

section("5. 百度结果结构")
try:
    html = fetch("https://www.baidu.com/s?wd=MU5101%20%E8%88%AA%E7%8F%AD")
    print("len", len(html))
    print("has c-container:", html.count("c-container"))
    m = re.search(r'<h3[^>]*>.*?</h3>', html, re.S)
    print(m.group(0)[:400] if m else "no h3")
except Exception as e:
    print("ERR", e)

section("6. OpenSky states")
try:
    d = json.loads(fetch("https://opensky-network.org/api/states/all?lamin=30&lomin=120&lamax=32&lomax=122", 20))
    st = (d.get("states") or [])[:2]
    print("count", len(d.get("states") or []))
    print("sample", json.dumps(st, ensure_ascii=False)[:600])
except Exception as e:
    print("ERR", e)

section("7. adsbdb aircraft")
try:
    d = json.loads(fetch("https://api.adsbdb.com/v0/aircraft/B-1672"))
    print(json.dumps(d, ensure_ascii=False)[:700])
except Exception as e:
    print("ERR", e)

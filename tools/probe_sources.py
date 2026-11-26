"""探测各个联网数据源在当前网络环境下是否可用。用于选检索后端。"""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
import urllib.request

URLS = {
    "wikidata": "https://www.wikidata.org/w/api.php?action=wbsearchentities&search=Airbus%20A320&language=en&format=json&limit=1",
    "wikimedia_rest": "https://api.wikimedia.org/core/v1/wikipedia/en/search/page?q=Airbus&limit=1",
    "zh_wiki_rest": "https://zh.wikipedia.org/api/rest_v1/page/summary/Airbus_A320",
    "en_wiki_api": "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=Airbus&format=json&srlimit=1",
    "cn_bing": "https://cn.bing.com/search?q=Airbus+A320",
    "www_bing": "https://www.bing.com/search?q=Airbus+A320",
    "sogou": "https://www.sogou.com/web?query=Airbus+A320",
    "baidu": "https://www.baidu.com/s?wd=Airbus+A320",
    "planespotters": "https://api.planespotters.net/pub/photos/reg/B-1672",
    "adsbdb_aircraft": "https://api.adsbdb.com/v0/aircraft/B-1672",
    "adsbdb_callsign": "https://api.adsbdb.com/v0/callsign/CA981",
    "opensky": "https://opensky-network.org/api/states/all?lamin=45&lomin=5&lamax=48&lomax=10",
    "openflights_airports": "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat",
    "github_raw": "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat",
    "airframes": "https://www.airframes.org/reg/B-1672",
    "flightradar": "https://www.flightradar24.com/data/aircraft/b-1672",
}

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"


def probe(item: tuple[str, str]) -> tuple[str, bool, str]:
    name, url = item
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=8) as r:
            body = r.read(4000)
        return name, True, f"{r.status} {len(body)}B"
    except Exception as e:
        return name, False, f"{type(e).__name__}: {str(e)[:60]}"


def main() -> int:
    out = {}
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        for name, ok, detail in ex.map(probe, URLS.items()):
            out[name] = {"ok": ok, "detail": detail}
            print(f"{'OK  ' if ok else 'FAIL'} {name:22s} {detail}")
    good = [k for k, v in out.items() if v["ok"]]
    print(f"\n可用: {len(good)}/{len(out)} -> {good}")
    if "--json" in sys.argv:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

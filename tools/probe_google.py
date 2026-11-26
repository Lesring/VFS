"""探测 Google 相关端点的可达性。

关键：Windows 上 urllib 的 timeout 管不住 DNS 解析，被墙的域名会卡死很久。
所以先做带超时的 DNS 预检，再决定要不要真发 HTTP 请求。
"""
from __future__ import annotations

import concurrent.futures as cf
import socket
import time
import urllib.error
import urllib.request

URLS = {
    "google_search": "https://www.google.com/search?q=MU5101+flight",
    "google_hk": "https://www.google.com.hk/search?q=MU5101",
    "google_gstatic": "https://www.gstatic.com/generate_204",
    "customsearch_api": "https://www.googleapis.com/customsearch/v1?q=test",
    "maps_geocode": "https://maps.googleapis.com/maps/api/geocode/json?address=PVG",
    "maps_static": "https://maps.googleapis.com/maps/api/staticmap?center=31.14,121.80&zoom=14&size=400x300",
    "maps_tiles": "https://maps.googleapis.com/maps/vt?pb=!1m5!1m4!1i12!2i3423!3i1674!4i256",
    "places_v1": "https://places.googleapis.com/v1/places:searchText",
    "translate": "https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=zh&dt=t&q=airport",
    "bing_global": "https://www.bing.com/search?q=MU5101",
}

HOSTS = [
    "www.google.com", "www.google.com.hk", "www.gstatic.com", "www.googleapis.com",
    "maps.googleapis.com", "places.googleapis.com", "translate.googleapis.com",
    "www.bing.com",
]

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"


def dns_check(host: str, timeout: float = 3.0) -> tuple[str, bool, str]:
    """带超时的 DNS 解析。socket.getaddrinfo 本身不受 timeout 控制，所以扔进线程里等。"""
    def _resolve():
        try:
            infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
            ips = sorted({i[4][0] for i in infos})
            return True, ",".join(ips[:3])
        except Exception as e:
            return False, type(e).__name__

    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_resolve)
        try:
            ok, detail = fut.result(timeout=timeout)
        except cf.TimeoutError:
            return host, False, "DNS timeout"
    return host, ok, detail


def probe(item):
    name, url = item
    host = url.split("/")[2]
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=5) as r:
            body = r.read(1500)
        return name, True, f"{r.status} {len(body)}B {time.time()-t0:.1f}s"
    except urllib.error.HTTPError as e:
        # 401/403/400 = 域名通、只是没 key，这也算可达
        return name, True, f"HTTP {e.code}（域名可达，缺 key）{time.time()-t0:.1f}s"
    except Exception as e:
        return name, False, f"{type(e).__name__}: {str(e)[:50]} {time.time()-t0:.1f}s"


def main() -> int:
    print("=== DNS 预检 ===")
    dns_ok = {}
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for host, ok, detail in ex.map(dns_check, HOSTS):
            dns_ok[host] = ok
            print(f"{'OK  ' if ok else 'FAIL'} {host:26s} {detail}")

    print("\n=== HTTP 探测（仅对 DNS 通过的域名） ===")
    reachable = {}
    todo = [(n, u) for n, u in URLS.items() if dns_ok.get(u.split("/")[2])]
    skipped = [n for n, u in URLS.items() if not dns_ok.get(u.split("/")[2])]
    for n in skipped:
        print(f"SKIP {n:16s} DNS 不可达")

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for name, ok, detail in ex.map(probe, todo):
            reachable[name] = ok
            print(f"{'OK  ' if ok else 'FAIL'} {name:16s} {detail}")

    print("\n=== 结论 ===")
    print("Google 搜索可用:", bool(reachable.get("google_search") or reachable.get("google_hk")))
    print("Google Maps API 可用:", bool(reachable.get("maps_geocode") or reachable.get("maps_static")))
    print("Google 静态/瓦片地图可用:", bool(reachable.get("maps_static") or reachable.get("maps_tiles")))
    print("说明：缺 key 时返回 HTTP 4xx 属于“域名可达”，配上 key 即可用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

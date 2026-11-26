"""找一个真正可用的免 key 静态地图服务（Wikimedia 那个 403 了）。"""
from __future__ import annotations

import urllib.error
import urllib.request

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"

CANDIDATES = {
    "wikimedia_osm": "https://maps.wikimedia.org/img/osm-intl,12,40.08,116.58,600x400.png",
    "staticmap_py": "https://staticmap.openstreetmap.de/staticmap.php?center=40.08,116.58&zoom=12&size=600x400",
    "osm_tile_single": "https://tile.openstreetmap.org/10/843/389.png",
    "carto_light": "https://a.basemaps.cartocdn.com/light_all/10/843/389.png",
    "carto_dark": "https://a.basemaps.cartocdn.com/dark_all/10/843/389.png",
    "carto_voyager": "https://a.basemaps.cartocdn.com/rastertiles/voyager/10/843/389.png",
    "osm_de": "https://tile.openstreetmap.de/10/843/389.png",
    "esri_sat": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/10/389/843",
    "esri_street": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/10/389/843",
}


def probe(name, url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": "http://localhost/"})
        with urllib.request.urlopen(req, timeout=9) as r:
            body = r.read(3000)
            ct = r.headers.get("content-type", "")
            ok = r.status == 200 and "image" in ct
            return name, ok, f"{r.status} {ct} {len(body)}B"
    except urllib.error.HTTPError as e:
        return name, False, f"HTTP {e.code}"
    except Exception as e:
        return name, False, f"{type(e).__name__}: {str(e)[:50]}"


def main() -> int:
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for name, ok, detail in ex.map(lambda kv: probe(*kv), CANDIDATES.items()):
            print(f"{'OK  ' if ok else 'FAIL'} {name:18s} {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Google 系数据源适配。

实测结论（决定了这里的实现方式）：
- google.com/search 返回的是纯 JS 外壳（94KB、零个 <h3>），直接爬 HTML 拿不到结果。
  所以要"真搜索"必须用 Programmable Search Engine API（需 key + cx）。
- Google News RSS 与 Suggest 免 key 可用，是无需配置就能享受的谷歌系能力。
- Google Maps 全部需要 key（静态图/地理编码/瓦片），但域名可达。
"""

from __future__ import annotations

import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any

from ..utils import clip, squash
from .http import CachedClient

_TAG = re.compile(r"<[^>]+>")


def _clean(s: str) -> str:
    return squash(_TAG.sub(" ", s or ""))


# --------------------------------------------------------------------------- #
# 1. Programmable Search API（真正的谷歌搜索，需 key + cx）
# --------------------------------------------------------------------------- #


async def search(
    client: CachedClient,
    query: str,
    limit: int = 5,
    api_key: str = "",
    cx: str = "",
) -> list[dict[str, Any]]:
    """Google Programmable Search。没有 key/cx 时直接返回空（不报错）。"""
    if not api_key or not cx:
        return []
    data = await client.get_json(
        "https://www.googleapis.com/customsearch/v1",
        params={"key": api_key, "cx": cx, "q": query, "num": min(10, max(1, limit))},
    )
    if not isinstance(data, dict):
        return []
    out = []
    for it in (data.get("items") or [])[:limit]:
        out.append({
            "title": clip(_clean(it.get("title") or ""), 160),
            "url": it.get("link") or "",
            "snippet": clip(_clean(it.get("snippet") or ""), 420),
            "engine": "google",
        })
    return out


# --------------------------------------------------------------------------- #
# 2. Google News RSS（免 key）
# --------------------------------------------------------------------------- #


def _parse_rss(xml_text: str, limit: int) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out = []
    for it in root.findall(".//item")[:limit]:
        title = _clean(it.findtext("title") or "")
        if not title:
            continue
        desc = _clean(it.findtext("description") or "")
        out.append({
            "title": clip(title, 160),
            "url": (it.findtext("link") or "").strip(),
            "snippet": clip(desc, 400),
            "published": (it.findtext("pubDate") or "").strip(),
            "source": _clean(it.findtext("source") or "Google News"),
            "engine": "google_news",
        })
    return out


async def news(
    client: CachedClient,
    query: str,
    limit: int = 5,
    lang: str = "en",
) -> list[dict[str, Any]]:
    """Google News RSS 检索。航班动态/事故这类"近期事件"很适合用它。"""
    hl = "zh-CN" if lang.startswith("zh") else "en-US"
    gl = "CN" if lang.startswith("zh") else "US"
    ceid = f"{gl}:{'zh-Hans' if gl == 'CN' else 'en'}"
    url = (f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}"
           f"&hl={hl}&gl={gl}&ceid={ceid}")
    xml = await client.get_text(url)
    if not xml:
        return []
    return _parse_rss(xml, limit)


# --------------------------------------------------------------------------- #
# 3. Suggest（免 key）：实体名纠错与规范化
# --------------------------------------------------------------------------- #


async def suggest(client: CachedClient, query: str) -> list[str]:
    """谷歌搜索建议词。用于校验/补全机型、机场名等专有名词拼写。"""
    url = ("https://suggestqueries.google.com/complete/search"
           f"?client=firefox&q={urllib.parse.quote(query)}")
    text = await client.get_text(url)
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
            return [str(s) for s in data[1][:10] if str(s).strip()]
    except (json.JSONDecodeError, IndexError, TypeError):
        pass
    return []


# --------------------------------------------------------------------------- #
# 4. Google Maps：地理编码 / 静态图（需 key）
# --------------------------------------------------------------------------- #


async def geocode(
    client: CachedClient,
    address: str,
    api_key: str = "",
) -> dict[str, Any] | None:
    """Google Geocoding。返回结构化地址 + 经纬度，用于"把地名钉到坐标"。"""
    if not api_key:
        return None
    data = await client.get_json(
        "https://maps.googleapis.com/maps/api/geocode/json",
        params={"address": address, "key": api_key, "language": "zh-CN"},
    )
    if not isinstance(data, dict) or data.get("status") != "OK":
        return None
    results = data.get("results") or []
    if not results:
        return None
    r0 = results[0]
    loc = ((r0.get("geometry") or {}).get("location")) or {}
    components = {}
    for c in r0.get("address_components") or []:
        for t in c.get("types") or []:
            components.setdefault(t, c.get("long_name"))
    return {
        "formatted": r0.get("formatted_address") or "",
        "lat": loc.get("lat"),
        "lon": loc.get("lng"),
        "place_id": r0.get("place_id"),
        "city": components.get("locality") or components.get("administrative_area_level_2"),
        "country": components.get("country"),
        "source": "google_maps",
    }


async def reverse_geocode(
    client: CachedClient,
    lat: float,
    lon: float,
    api_key: str = "",
) -> dict[str, Any] | None:
    if not api_key:
        return None
    data = await client.get_json(
        "https://maps.googleapis.com/maps/api/geocode/json",
        params={"latlng": f"{lat},{lon}", "key": api_key, "language": "zh-CN"},
    )
    if not isinstance(data, dict) or data.get("status") != "OK":
        return None
    results = data.get("results") or []
    if not results:
        return None
    r0 = results[0]
    components = {}
    for c in r0.get("address_components") or []:
        for t in c.get("types") or []:
            components.setdefault(t, c.get("long_name"))
    return {
        "formatted": r0.get("formatted_address") or "",
        "city": components.get("locality") or components.get("administrative_area_level_2"),
        "country": components.get("country"),
        "source": "google_maps",
    }


def static_map_url(lat: float, lon: float, api_key: str, zoom: int = 13,
                   size: str = "600x400", label: str = "") -> str | None:
    """生成静态地图 URL，前端可直接 <img src> 展示。

    这是"谷歌地图"在本系统里最实用的形态：把推断出的坐标可视化成一张图，
    让人一眼看出位置对不对。
    """
    if not api_key or lat is None or lon is None:
        return None
    params = {
        "center": f"{lat},{lon}", "zoom": str(zoom), "size": size,
        "maptype": "satellite", "key": api_key,
    }
    if label:
        params["markers"] = f"color:red|label:{label[:1]}|{lat},{lon}"
    return "https://maps.googleapis.com/maps/api/staticmap?" + urllib.parse.urlencode(params)


def embed_map_url(query: str, api_key: str) -> str | None:
    """Maps Embed API：可在页面里嵌交互式地图（需 key 且该 key 需启用 Embed API）。"""
    if not api_key or not query:
        return None
    return ("https://www.google.com/maps/embed/v1/place?"
            + urllib.parse.urlencode({"q": query, "key": api_key, "zoom": "13"}))


# --------------------------------------------------------------------------- #
# 5. 免 key 的地图兜底：OpenStreetMap
# --------------------------------------------------------------------------- #


async def osm_geocode(client: CachedClient, query: str) -> dict[str, Any] | None:
    """Nominatim 地理编码。Google Maps 没配 key 时的替代。"""
    data = await client.get_json(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "json", "limit": 1, "addressdetails": 1},
        headers={"Accept": "application/json"},
    )
    if not isinstance(data, list) or not data:
        return None
    d = data[0]
    addr = d.get("address") or {}
    try:
        lat, lon = float(d.get("lat")), float(d.get("lon"))
    except (TypeError, ValueError):
        return None
    return {
        "formatted": d.get("display_name") or "",
        "lat": lat, "lon": lon,
        "city": addr.get("city") or addr.get("town") or addr.get("county"),
        "country": addr.get("country"),
        "source": "openstreetmap",
    }


TILE_TEMPLATES = [
    "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
    "https://tile.openstreetmap.de/{z}/{x}/{y}.png",
]


def deg2tile(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    import math

    n = 2 ** zoom
    lat_rad = math.radians(lat_deg)
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_origin(lat: float, lon: float, zoom: int, grid: int = 2,
                tile_size: int = 256) -> tuple[int, int]:
    """拼图左上角瓦片的 (x, y)，与 stitch_map 使用同一套换算，保证一致。"""
    import math

    n = 2 ** zoom
    nf = float(n)
    xf = (lon + 180.0) / 360.0 * nf * tile_size
    lat_rad = math.radians(lat)
    yf = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * nf * tile_size
    cw = grid * tile_size
    tx0 = int(math.floor((xf - cw / 2.0) / tile_size))
    ty0 = int(math.floor((yf - cw / 2.0) / tile_size))
    return tx0, ty0


def tile_urls(lat: float, lon: float, zoom: int = 12, grid: int = 2) -> list[str]:
    """返回一组 OSM 瓦片 URL（前端可自行拼接，不依赖静态图接口）。"""
    n = 2 ** zoom
    tx0, ty0 = tile_origin(lat, lon, zoom, grid)
    urls = []
    for dy in range(grid):
        for dx in range(grid):
            x, y = tx0 + dx, ty0 + dy
            if 0 <= x < n and 0 <= y < n:
                urls.append(TILE_TEMPLATES[0].format(z=zoom, x=x, y=y))
    return urls


def stitch_map(
    fetch_tile,
    lat: float,
    lon: float,
    zoom: int = 12,
    grid: int = 3,
    out_path: str = "",
) -> str | None:
    """服务端把 OSM 瓦片拼成一张静态图，并标出目标点。

    为什么要在服务端拼：
    - Wikimedia 静态图接口对无 Referer 请求返回 403，浏览器里加载不可靠；
    - 拼图完全可控，还能顺带画标记点，且瓦片源可换。

    fetch_tile 是一个可调用对象 (url) -> bytes | None，由调用方提供（复用缓存客户端）。
    """
    try:
        from io import BytesIO

        from PIL import Image, ImageDraw
    except ImportError:
        return None

    import math

    n = 2 ** zoom
    tile_size = 256
    cw = grid * tile_size

    # 与 tile_urls 共用同一套换算，保证标记位置和瓦片选择一致
    tx0, ty0 = tile_origin(lat, lon, zoom, grid, tile_size)

    nf = float(n)
    xf = (lon + 180.0) / 360.0 * nf * tile_size
    lat_rad = math.radians(lat)
    yf = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * nf * tile_size
    px = xf - tx0 * tile_size
    py = yf - ty0 * tile_size
    canvas = Image.new("RGB", (cw, cw), (16, 20, 26))
    got = 0

    for dy in range(grid):
        for dx in range(grid):
            tx, ty = tx0 + dx, ty0 + dy
            if not (0 <= tx < n and 0 <= ty < n):
                continue
            tile_url = TILE_TEMPLATES[0].format(z=zoom, x=tx, y=ty)
            data = fetch_tile(tile_url)
            if not data:
                continue
            try:
                img = Image.open(BytesIO(data)).convert("RGB")
                canvas.paste(img, (dx * tile_size, dy * tile_size))
                got += 1
            except Exception:
                continue

    if got == 0:
        return None

    draw = ImageDraw.Draw(canvas)
    r = 9
    draw.ellipse([px - r, py - r, px + r, py + r], fill=(220, 40, 40), outline=(255, 255, 255), width=3)
    draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(255, 255, 255))

    from pathlib import Path

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(p, format="PNG", optimize=True)
    return str(p)

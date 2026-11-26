"""联网核实编排。

流程：
1. 从当前槽位挑出"硬线索"（航班号 > 注册号 > 机场 > 机型+航司 > 城市/地标）。
2. 用结构化 API 做权威查询（adsbdb 查航线/机号，OpenSky 查该区域实时航班）。
3. 用网页搜索补上下文（航班动态、机场、地标、机型）。
4. 把每条结果打上来源与置信度，交给 FusionEngine 融合。
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from . import config as cfg_mod
from .entities import normalize_flight, normalize_registration
from .kb import flights as flights_kb
from .kb import wiki as wiki_kb
from .kb.http import CachedClient
from .models import Evidence, KnowledgeHit, SlotResult
from .utils import clip, squash


class RetrievalResult:
    def __init__(self) -> None:
        self.route: dict[str, Any] | None = None
        self.aircraft: dict[str, Any] | None = None
        self.knowledge: list[KnowledgeHit] = []
        self.searches: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.trace: list[str] = []
        self.stats: dict[str, Any] = {}
        # 地图可视化：推断出的坐标 + 可展示的地图 URL
        self.map: dict[str, Any] | None = None

    def add_search(self, query: str, results: list[dict[str, Any]], attempted: list[str],
                   notes: dict[str, str] | None = None) -> None:
        self.searches.append({"query": query, "attempted": attempted, "results": results[:6],
                              "n": len(results), "notes": notes or {}})


def _top(slots: dict[str, SlotResult], name: str) -> str | None:
    s = slots.get(name)
    if s and s.candidates and s.confidence >= 0.12:
        return s.candidates[0].value
    return None


def _top_details(slots: dict[str, SlotResult], name: str) -> dict[str, Any]:
    s = slots.get(name)
    if s and s.candidates:
        return s.candidates[0].details
    return {}


async def run_retrieval(
    slots: dict[str, SlotResult],
    client: CachedClient,
    airlines,
    airports,
    cfg: dict[str, Any],
    question: str = "",
) -> RetrievalResult:
    res = RetrievalResult()
    if not cfg.get("enabled", True):
        res.warnings.append("联网检索被配置关闭")
        return res

    backend = str(cfg.get("search_backend") or "auto")
    limit = int(cfg.get("max_search_results") or 5)

    flight_raw = _top(slots, "flight_number")
    reg_raw = _top(slots, "registration")
    atype = _top(slots, "aircraft_type")
    airline = _top(slots, "airline")
    airport = _top(slots, "airport")
    dest = _top(slots, "destination")
    origin = _top(slots, "origin")
    city = _top(slots, "city")
    landmark = _top(slots, "landmark")
    date = _top(slots, "date")

    tasks: list[Any] = []

    # ---------------- 1. 结构化权威查询 ----------------
    flight = normalize_flight(flight_raw) if flight_raw else None
    reg = normalize_registration(reg_raw) if reg_raw else None

    if flight:
        cs = flight.get("iata") or flight.get("icao")
        res.trace.append(f"检索：按航班号 {cs} 查询航线")
        res.route = await flights_kb.route_by_callsign(client, cs or "")
        if not res.route and flight.get("icao"):
            res.route = await flights_kb.route_by_callsign(client, flight["icao"])
        if not res.route and flight.get("iata") and flights_kb:
            alt = await flights_kb.route_by_callsign(client, flight["iata"])
            res.route = alt
        if res.route:
            o = res.route.get("origin") or {}
            d = res.route.get("destination") or {}
            res.trace.append(
                f"航线库命中：{(res.route.get('airline') or {}).get('name', '?')} "
                f"{o.get('name', '?')} → {d.get('name', '?')}"
            )
        else:
            res.warnings.append(f"航线库中未查到 {cs} 的固定航线（可能是临时/货运/拼接航班号）")
    elif flight_raw:
        res.warnings.append(f"航班号 {flight_raw} 格式无法归一，跳过航线查询")

    if reg:
        res.trace.append(f"检索：按注册号 {reg} 查询机型与机主")
        res.aircraft = await flights_kb.aircraft_by_registration(client, reg)
        if res.aircraft:
            res.trace.append(
                f"航空器库命中：{res.aircraft.get('type')} / {res.aircraft.get('owner') or '未知机主'}"
            )
        else:
            res.warnings.append(f"航空器库中未查到注册号 {reg}")

    # 机场/航线兜底
    if airports is not None and len(airports) > 0:
        for slot_name, val in (("airport", airport), ("destination", dest), ("origin", origin)):
            if not val:
                continue
            hits = airports.lookup(val)
            if hits:
                a = hits[0]
                res.knowledge.append(KnowledgeHit(
                    title=f"机场库匹配：{a.label()}", source="openflights",
                    snippet=f"{a.name}，{a.city}，{a.country}；IATA {a.iata} / ICAO {a.icao}；"
                            f"坐标 {a.lat},{a.lon}；时区 {a.tz}",
                    score=0.9, data={"slot": slot_name, **a.to_dict()},
                ))
                res.trace.append(f"机场库匹配：{val} → {a.label()}")

    # ---------------- 2. 并发网页/百科检索 ----------------
    queries: list[str] = []

    if flight_raw and airlines is not None and len(airlines) > 0:
        pass
    if flight and not res.route:
        fn = flight.get("iata") or flight.get("icao")
        queries.append(f"{fn} 航班 目的地 航线")
        queries.append(f"{fn} flight route destination")
    if atype:
        queries.append(f"{atype} 客机 航程 载客量")
    if airline and atype and not flight:
        queries.append(f"{airline} {atype} 机队 航线")
    if airport and not (dest or origin):
        queries.append(f"{airport} 机场 航线 航站楼")
    if landmark:
        queries.append(f"{landmark} 位置 城市 国家")
    if city:
        queries.append(f"{city} 机场 航空")
    date_str = date or ""
    if flight_raw and date_str:
        queries.append(f"{flight_raw} {date_str} 航班动态")

    queries = list(dict.fromkeys(q for q in queries if q.strip()))[:6]

    gcfg = cfg_mod.CONFIG.get("google", {}) if hasattr(cfg_mod, "CONFIG") else {}
    google_key = str(gcfg.get("api_key") or "")
    google_cx = str(gcfg.get("cx") or "")
    strategy = str(cfg.get("region_strategy") or "overseas_first")

    async def do_search(q: str) -> tuple[str, list[dict[str, Any]], list[str], dict[str, str]]:
        from .kb.search import web_search
        lang = "zh" if re.search(r"[\u4e00-\u9fff]", q) else "en"
        try:
            results, attempted, notes = await web_search(
                client, q, limit=limit, backend=backend, lang=lang,
                strategy=strategy, google_key=google_key, google_cx=google_cx,
            )
            return q, results, attempted, notes
        except Exception as e:
            return q, [], [f"error"], {"error": f"{type(e).__name__}: {e}"}

    if queries:
        search_out = await asyncio.gather(*(do_search(q) for q in queries))
        for q, results, attempted, notes in search_out:
            res.add_search(q, results, attempted, notes)
            for r in results[:limit]:
                res.knowledge.append(KnowledgeHit(
                    title=r.get("title") or "", url=r.get("url") or "",
                    snippet=r.get("snippet") or "", source=f"web:{r.get('engine')}",
                    score=0.55,
                ))
            if notes:
                for src, why in notes.items():
                    msg = f"{src} 跳过：{why}"
                    if msg not in res.warnings:
                        res.warnings.append(msg)

    # ---------------- 2.5 Google News RSS（免 key，抓近期动态） ----------------
    news_queries: list[str] = []
    if flight_raw:
        news_queries.append(f"{flight_raw} flight")
    if airline and atype:
        news_queries.append(f"{airline} {atype}")
    if airport:
        news_queries.append(f"{airport} airport")
    news_queries = list(dict.fromkeys(q for q in news_queries if q.strip()))[:2]

    if news_queries:
        from .kb import google as google_kb
        from .kb import health as health_mod

        async def do_news(q: str):
            try:
                lang = "zh" if re.search(r"[\u4e00-\u9fff]", q) else "en"
                return q, await google_kb.news(client, q, limit=3, lang=lang)
            except Exception:
                return q, []

        if health_mod.check("google"):
            for q, items in await asyncio.gather(*(do_news(q) for q in news_queries)):
                for it in items:
                    res.knowledge.append(KnowledgeHit(
                        title=it.get("title") or "",
                        url=it.get("url") or "",
                        snippet=(it.get("published", "") + " " + (it.get("snippet") or ""))[:400],
                        source="google_news",
                        score=0.5,
                    ))
                if items:
                    res.trace.append(f"Google News 命中 {len(items)} 条：{clip(q, 50)}")
            res.searches.append({
                "query": " | ".join(news_queries), "attempted": ["google_news"],
                "results": [], "n": 0, "note": "Google News RSS",
            })

    # ---------------- 3. 百科摘要佐证 ----------------
    wiki_queries: list[str] = []
    if atype:
        wiki_queries.append(atype)
    if airline:
        wiki_queries.append(airline)
    if airport:
        wiki_queries.append(airport)
    wiki_queries = list(dict.fromkeys(wiki_queries))[:2]

    async def do_wiki(q: str) -> list[dict[str, Any]]:
        try:
            return await wiki_kb.ask(client, q, limit=1, langs=("en",))
        except Exception:
            return []

    if wiki_queries:
        for q, hits in zip(wiki_queries, await asyncio.gather(*(do_wiki(q) for q in wiki_queries))):
            for h in hits:
                res.knowledge.append(KnowledgeHit(
                    title=h.get("title") or q,
                    url=h.get("url") or "",
                    snippet=(h.get("description") or "") + " " + (h.get("extract") or "")[:400],
                    source=h.get("source") or "wikipedia",
                    score=0.65,
                ))
                res.trace.append(f"百科佐证：{h.get('title')} — {clip(h.get('description') or h.get('extract') or '', 100)}")

    res.stats = dict(client.stats)
    return res


async def _stitch_local_map(client: CachedClient, lat: float, lon: float,
                            zoom: int, out_path: str) -> str | None:
    """服务端拼瓦片成一张带标记的图。瓦片走缓存客户端，失败就返回 None。"""
    from .kb import google as google_kb

    def fetch(url: str) -> bytes | None:
        try:
            import urllib.request

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "VideoForensics/1.0 (+map)",
                         "Referer": "http://localhost/"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                if r.status != 200:
                    return None
                return r.read()
        except Exception:
            return None

    try:
        return google_kb.stitch_map(fetch, lat, lon, zoom=zoom, grid=3, out_path=out_path)
    except Exception:
        return None


async def build_map(
    client: CachedClient,
    slots: dict[str, SlotResult],
    cfg: dict[str, Any] | None = None,
    job_id: str = "",
    output_dir: str = "",
) -> dict[str, Any] | None:
    """把推断出的位置可视化成地图。

    优先用 Google Maps（需 key）；没配 key 就用 OpenStreetMap 兜底，
    保证"没配置也能看到图"，配了 key 则自动升级为谷歌地图。

    注意：必须在槽位最终确定后调用，否则会定位到已被否决的候选上。
    """
    out_path = ""
    if output_dir:
        out_path = str(Path(output_dir) / "map.png")
    gcfg = (cfg if cfg is not None else cfg_mod.CONFIG.get("google", {})) or {}
    if not gcfg.get("maps_enabled", True):
        return None

    from .kb import google as google_kb
    from .kb import health as health_mod

    maps_key = str(gcfg.get("maps_api_key") or "")

    # 位置来源优先级：目的地/出发地机场坐标 > 画面坐标 > 城市名
    lat = lon = None
    label = ""
    for slot_name, nice in (("destination", "目的地"), ("origin", "出发地"),
                            ("airport", "所在机场"), ("coordinate", "画面坐标")):
        s = slots.get(slot_name)
        if not s or not s.candidates:
            continue
        c = s.candidates[0]
        la, lo = c.details.get("lat"), c.details.get("lon")
        if isinstance(la, (int, float)) and isinstance(lo, (int, float)):
            lat, lon = float(la), float(lo)
            label = f"{nice}：{clip(c.value, 70)}"
            break

    place_query = ""
    if lat is None:
        for slot_name in ("destination", "airport", "city", "landmark"):
            v = _top(slots, slot_name)
            if v:
                place_query = v
                label = clip(v, 70)
                break

    if lat is None and not place_query:
        return None

    zoom = int(gcfg.get("map_zoom") or 13)
    out: dict[str, Any] = {"label": label, "lat": lat, "lon": lon}

    # 1) Google 地理编码补全（有 key 时）
    google_failed = False
    if maps_key and health_mod.check("google_maps"):
        if lat is not None and lon is not None:
            geo = await google_kb.reverse_geocode(client, lat, lon, maps_key)
        else:
            geo = await google_kb.geocode(client, place_query, maps_key)
        if not geo:
            google_failed = True
        if geo:
            lat = geo.get("lat") if geo.get("lat") is not None else lat
            lon = geo.get("lon") if geo.get("lon") is not None else lon
            out.update({"lat": lat, "lon": lon,
                        "address": geo.get("formatted"),
                        "city": geo.get("city"), "country": geo.get("country")})
            out["provider"] = "google_maps"
            out["static_url"] = google_kb.static_map_url(lat, lon, maps_key, zoom=zoom,
                                                        label="●") if lat is not None else None
            out["embed_url"] = google_kb.embed_map_url(geo.get("formatted") or place_query, maps_key)
            out["link"] = (f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
                           if lat is not None else
                           f"https://www.google.com/maps/search/?api=1&query={place_query}")
            return out

    # 2) 无 key：OSM 兜底
    if lat is None and place_query and health_mod.check("nominatim"):
        geo = await google_kb.osm_geocode(client, place_query)
        if geo:
            lat, lon = geo.get("lat"), geo.get("lon")
            out.update({"lat": lat, "lon": lon, "address": geo.get("formatted"),
                        "city": geo.get("city"), "country": geo.get("country")})

    if lat is not None and lon is not None:
        out["provider"] = "openstreetmap"
        out["link"] = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map={zoom}/{lat}/{lon}"
        out["tiles"] = google_kb.tile_urls(lat, lon, zoom=zoom - 1, grid=3)

        # 服务端拼图：比依赖第三方静态图接口可靠（Wikimedia 那个会 403）
        if out_path:
            png = await _stitch_local_map(client, lat, lon, zoom - 1, out_path)
            if png and job_id:
                out["static_url"] = f"/api/map/{job_id}/map.png"
                out["static_local_path"] = png
        if google_failed:
            out["note"] = "Google Maps Key 无效或无权限，已改用 OpenStreetMap"
            out["degraded_reason"] = "google_key_invalid"
        else:
            out["note"] = "未配置 Google Maps Key，使用 OpenStreetMap 展示"
            out["degraded_reason"] = "no_google_key"
        return out

    if place_query:
        out["provider"] = "link_only"
        out["link"] = f"https://www.google.com/maps/search/?api=1&query={place_query}"
        out["note"] = "仅有地名，未解析出坐标"
        return out

    return None

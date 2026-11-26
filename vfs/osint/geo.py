"""按约束筛选「机场 + 平行铁路 + 车站」候选。不含任何标答硬编码。"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any

from ..kb.airports import AirportDB, haversine_km
from ..kb.http import CachedClient
from ..utils import clip

RUNWAYS_URL = "https://davidmegginson.github.io/ourairports-data/runways.csv"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]


def _ang_diff(a: float, b: float) -> float:
    """两航向差（0-90，忽略反向）。"""
    d = abs((a - b + 180) % 360 - 180)
    return min(d, 180 - d)


def _is_east_west(heading: float, tol: float = 35.0) -> bool:
    h = heading % 180  # 0-180
    # 东西向：约 70-110
    return abs(h - 90) <= tol


class RunwayDB:
    def __init__(self, rows: list[dict[str, str]]):
        self.by_icao: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            icao = (r.get("airport_ident") or "").upper()
            if not icao:
                continue
            try:
                hdg = float(r.get("le_heading_degT") or r.get("he_heading_degT") or "")
            except ValueError:
                continue
            closed = (r.get("closed") or "0") in ("1", "true", "True")
            if closed:
                continue
            try:
                length = float(r.get("length_ft") or 0)
            except ValueError:
                length = 0
            if length and length < 4000:  # 过滤过短跑道
                continue
            self.by_icao.setdefault(icao, []).append({
                "icao": icao,
                "le_ident": r.get("le_ident") or "",
                "he_ident": r.get("he_ident") or "",
                "heading": hdg % 360,
                "length_ft": length,
                "le_lat": _f(r.get("le_latitude_deg")),
                "le_lon": _f(r.get("le_longitude_deg")),
                "he_lat": _f(r.get("he_latitude_deg")),
                "he_lon": _f(r.get("he_longitude_deg")),
            })

    def headings(self, icao: str) -> list[float]:
        return [r["heading"] for r in self.by_icao.get((icao or "").upper(), [])]


def _f(v: Any) -> float | None:
    try:
        return float(v) if v not in (None, "", "\\N") else None
    except (TypeError, ValueError):
        return None


async def load_runways(client: CachedClient, cache_dir: str | Path) -> RunwayDB:
    path = Path(cache_dir) / "ourairports_runways.csv"
    if not path.exists():
        await client.download_binary(RUNWAYS_URL, path, ttl_s=30 * 86400)
    rows: list[dict[str, str]] = []
    if path.exists():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            reader = csv.DictReader(io.StringIO(text))
            rows = list(reader)
        except Exception:
            rows = []
    return RunwayDB(rows)


async def _overpass(client: CachedClient, query: str) -> dict[str, Any] | None:
    import json
    for url in OVERPASS_URLS:
        text = await _post_text(client, url, query)
        if not text:
            continue
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "elements" in obj:
                return obj
        except Exception:
            continue
    return None


async def _post_text(client: CachedClient, url: str, query: str) -> str | None:
    import httpx
    key = f"overpass::{url}::{query[:200]}"
    cached = client._read_cache(key, ".txt")
    if cached is not None:
        client.stats["hit"] += 1
        return cached
    try:
        async with httpx.AsyncClient(timeout=max(40.0, client.timeout)) as c:
            r = await c.post(url, data={"data": query}, headers={"User-Agent": client.ua})
            client.stats["network"] += 1
            if r.status_code >= 400:
                client.stats["error"] += 1
                return None
            client._write_cache(key, ".txt", r.text)
            return r.text
    except Exception:
        client.stats["error"] += 1
        return None


def _rail_heading(geometry: list[dict[str, float]]) -> float | None:
    if not geometry or len(geometry) < 2:
        return None
    a, b = geometry[0], geometry[-1]
    lat1, lon1 = math.radians(a["lat"]), math.radians(a["lon"])
    lat2, lon2 = math.radians(b["lat"]), math.radians(b["lon"])
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    brng = (math.degrees(math.atan2(y, x)) + 360) % 360
    return brng


async def railways_near(client: CachedClient, lat: float, lon: float, radius_m: int = 6000) -> list[dict[str, Any]]:
    q = f"""
    [out:json][timeout:40];
    (
      way["railway"="rail"](around:{radius_m},{lat},{lon});
    );
    out geom tags;
    """
    data = await _overpass(client, q)
    out = []
    for el in (data or {}).get("elements") or []:
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        hdg = _rail_heading(geom)
        if hdg is None:
            continue
        # 取中点
        mid = geom[len(geom) // 2]
        out.append({
            "id": el.get("id"),
            "heading": hdg,
            "lat": mid.get("lat"),
            "lon": mid.get("lon"),
            "name": (el.get("tags") or {}).get("name") or "",
            "ref": (el.get("tags") or {}).get("ref") or "",
        })
    return out


async def stations_near(client: CachedClient, lat: float, lon: float, radius_m: int = 8000) -> list[dict[str, Any]]:
    # 优先地理编码（稳、快）；Overpass 常被限流，作为补充
    out = await _stations_via_geocoder(client, lat, lon, radius_m)
    if len(out) >= 3:
        return out
    q = f"""
    [out:json][timeout:25];
    (
      node["railway"="station"](around:{radius_m},{lat},{lon});
      node["railway"="halt"](around:{radius_m},{lat},{lon});
    );
    out body;
    """
    data = await _overpass(client, q)
    seen = {f"{(s.get('name') or '').lower()}|{round(float(s['lat']),4)}" for s in out if s.get("lat") is not None}
    for el in (data or {}).get("elements") or []:
        tags = el.get("tags") or {}
        name = tags.get("name") or tags.get("uic_name") or ""
        if not name or el.get("lat") is None:
            continue
        key = f"{name.lower()}|{round(float(el['lat']),4)}"
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "name": name,
            "lat": el.get("lat"),
            "lon": el.get("lon"),
            "uic": tags.get("uic_ref") or "",
            "ref": tags.get("ref") or "",
        })
    return out


async def _stations_via_geocoder(
    client: CachedClient, lat: float, lon: float, radius_m: int,
) -> list[dict[str, Any]]:
    """Nominatim + Photon 周边车站检索。"""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    delta = radius_m / 111000.0
    viewbox = f"{lon - delta},{lat + delta},{lon + delta},{lat - delta}"

    # Nominatim：按关键词 + 视窗
    queries = (
        "railway station", "train station", "railway halt", "gare", "bahnhof",
        "station", "halte",
    )
    for q in queries:
        full = (
            "https://nominatim.openstreetmap.org/search?"
            f"q={q.replace(' ', '+')}&format=json&limit=40&bounded=1&viewbox={viewbox}"
        )
        data = await client.get_json(full, headers={"Accept-Language": "en"})
        if not isinstance(data, list):
            continue
        for item in data:
            name = item.get("display_name") or item.get("name") or ""
            short = name.split(",")[0].strip()
            # "Erps-Kwerps Station" → 保留；纯村名也可能是停靠点
            if short.lower().endswith(" station"):
                short = short[: -len(" station")].strip()
            try:
                slat, slon = float(item.get("lat")), float(item.get("lon"))
            except (TypeError, ValueError):
                continue
            if haversine_km(lat, lon, slat, slon) > radius_m / 1000.0 + 0.8:
                continue
            cls = f"{item.get('class') or ''} {item.get('type') or ''}".lower()
            osm = f"{cls} {name.lower()}"
            if not any(k in osm for k in ("station", "halt", "railway", "bahnhof", "gare", "halte")):
                continue
            key = f"{short.lower()}|{round(slat, 4)}|{round(slon, 4)}"
            if key in seen:
                continue
            seen.add(key)
            out.append({"name": short, "lat": slat, "lon": slon, "uic": "", "ref": ""})

    # Photon
    ph = await client.get_json(
        f"https://photon.komoot.io/api/?q=train%20station&lat={lat}&lon={lon}&limit=25&lang=en"
    )
    for f in ((ph or {}).get("features") or []) if isinstance(ph, dict) else []:
        props = f.get("properties") or {}
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates") or [None, None]
        name = props.get("name") or props.get("street") or ""
        if not name or coords[0] is None:
            continue
        slon, slat = float(coords[0]), float(coords[1])
        if haversine_km(lat, lon, slat, slon) > radius_m / 1000.0 + 0.5:
            continue
        osm_key = (props.get("osm_key") or "") + " " + (props.get("osm_value") or "")
        if "rail" not in osm_key and "station" not in osm_key and props.get("type") not in (
            "station", "halt", "railway",
        ):
            if "station" not in name.lower() and "bahnhof" not in name.lower():
                continue
        key = f"{name.lower()}|{round(slat,4)}|{round(slon,4)}"
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "lat": slat, "lon": slon, "uic": "", "ref": ""})

    return out


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def _offset_point(lat: float, lon: float, heading_deg: float, dist_km: float) -> tuple[float, float]:
    """沿航向前进 dist_km，返回新的 lat/lon。"""
    r = 6371.0088
    δ = dist_km / r
    θ = math.radians(heading_deg)
    φ1, λ1 = math.radians(lat), math.radians(lon)
    φ2 = math.asin(math.sin(φ1) * math.cos(δ) + math.cos(φ1) * math.sin(δ) * math.cos(θ))
    λ2 = λ1 + math.atan2(
        math.sin(θ) * math.sin(δ) * math.cos(φ1),
        math.cos(δ) - math.sin(φ1) * math.sin(φ2),
    )
    return math.degrees(φ2), (math.degrees(λ2) + 540) % 360 - 180


async def stations_along_runways(
    client: CachedClient,
    runways: list[dict[str, Any]],
    airport_lat: float,
    airport_lon: float,
) -> list[dict[str, Any]]:
    """沿各跑道两端延长线采样，反查附近车站（不依赖地名先验）。"""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in runways:
        anchors = []
        if r.get("le_lat") is not None and r.get("le_lon") is not None:
            anchors.append((float(r["le_lat"]), float(r["le_lon"]), float(r["heading"])))
            anchors.append((float(r["le_lat"]), float(r["le_lon"]), (float(r["heading"]) + 180) % 360))
        if r.get("he_lat") is not None and r.get("he_lon") is not None:
            anchors.append((float(r["he_lat"]), float(r["he_lon"]), float(r["heading"])))
            anchors.append((float(r["he_lat"]), float(r["he_lon"]), (float(r["heading"]) + 180) % 360))
        if not anchors:
            anchors = [
                (airport_lat, airport_lon, float(r["heading"])),
                (airport_lat, airport_lon, (float(r["heading"]) + 180) % 360),
            ]
        # 只取跑道“外侧”方向各采样一次：朝 heading 与反向各 7km
        used_hdg = set()
        for alat, alon, hdg in anchors:
            hkey = int(hdg) // 5
            if hkey in used_hdg:
                continue
            used_hdg.add(hkey)
            if len(used_hdg) > 2:
                break
            plat, plon = _offset_point(alat, alon, hdg, 7.0)
            url = (
                "https://nominatim.openstreetmap.org/reverse?"
                f"lat={plat:.5f}&lon={plon:.5f}&format=json&zoom=16&addressdetails=1"
            )
            data = await client.get_json(url)
            village = ""
            if isinstance(data, dict):
                addr = data.get("address") or {}
                village = (
                    addr.get("village") or addr.get("town") or addr.get("suburb")
                    or addr.get("hamlet") or addr.get("municipality")
                    or (data.get("name") or "")
                )
            vb = f"{plon-0.06},{plat+0.06},{plon+0.06},{plat-0.06}"
            full = (
                "https://nominatim.openstreetmap.org/search?"
                f"q=station&format=json&limit=12&bounded=1&viewbox={vb}"
            )
            hits = await client.get_json(full)
            if isinstance(hits, list):
                for item in hits:
                    nm = (item.get("display_name") or "").split(",")[0].strip()
                    if nm.lower().endswith(" station"):
                        nm = nm[: -len(" station")].strip()
                    try:
                        slat, slon = float(item["lat"]), float(item["lon"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    key = f"{nm.lower()}|{round(slat,4)}|{round(slon,4)}"
                    if not nm or key in seen:
                        continue
                    if haversine_km(airport_lat, airport_lon, slat, slon) > 12:
                        continue
                    seen.add(key)
                    out.append({"name": nm, "lat": slat, "lon": slon, "uic": "", "ref": ""})
            qn = str(village).split(",")[0].strip()
            if qn and len(qn) >= 3:
                full = (
                    "https://nominatim.openstreetmap.org/search?"
                    f"q={qn.replace(' ', '+')}+station&format=json&limit=5"
                )
                hits = await client.get_json(full)
                if isinstance(hits, list):
                    for item in hits:
                        nm = (item.get("display_name") or "").split(",")[0].strip()
                        if nm.lower().endswith(" station"):
                            nm = nm[: -len(" station")].strip()
                        try:
                            slat, slon = float(item["lat"]), float(item["lon"])
                        except (KeyError, TypeError, ValueError):
                            continue
                        if haversine_km(airport_lat, airport_lon, slat, slon) > 12:
                            continue
                        key = f"{nm.lower()}|{round(slat,4)}|{round(slon,4)}"
                        if not nm or key in seen:
                            continue
                        seen.add(key)
                        out.append({"name": nm, "lat": slat, "lon": slon, "uic": "", "ref": ""})
    return out


async def find_airport_rail_candidates(
    client: CachedClient,
    airports: AirportDB,
    runways: RunwayDB,
    constraints: dict[str, Any],
    limit: int = 8,
) -> list[dict[str, Any]]:
    """筛选满足约束的机场+车站候选。

    核心几何：车站应大致落在跑道中心线延长线上（火车上看进近/起飞），
    且行进轴与跑道近似平行。铁路完整几何常因 Overpass 超时不可用，
    因此以「车站相对机场方位 ≈ 跑道航向」为主判据。
    """
    countries = set(constraints.get("countries") or [])
    want_ew = (constraints.get("implied_travel_axis") or "") == "east_west"
    phase = constraints.get("aircraft_phase") or "unknown"

    pool = []
    for a in airports.airports:
        if a.lat is None or a.lon is None:
            continue
        if a.type and a.type not in ("airport", "") and "/" not in a.type:
            continue
        # 排除明显军用/极小场：无长跑道的稍后再滤；先去掉 heliport 等
        if a.type in ("heliport", "seaplane_base", "balloonport"):
            continue
        if countries and a.country not in countries:
            continue
        if not a.iata or len(a.iata) != 3:
            continue
        rw = runways.by_icao.get((a.icao or "").upper()) or []
        if not rw:
            continue
        pool.append((a, rw))

    pre: list[tuple[Any, list[dict], float]] = []
    for a, rw in pool:
        headings = [r["heading"] for r in rw]
        ew_score = max((1.0 - _ang_diff(h, 90) / 90.0) for h in headings) if headings else 0
        if want_ew and ew_score < 0.50:
            continue
        pre.append((a, rw, ew_score))
    pre.sort(key=lambda x: (
        -max((r.get("length_ft") or 0) for r in x[1]),
        -x[2],
    ))
    pre = pre[:14]

    # 排除美军基地等（ICAO ET* 在德多为美驻军机场）
    pre = [x for x in pre if not (x[0].icao or "").upper().startswith("ET")]

    enriched: list[tuple[Any, list[dict], float, list[dict]]] = []
    for a, rw, ew_score in pre:
        stns = await stations_near(client, float(a.lat), float(a.lon), radius_m=10000)
        # 每场只沿「最接近东西向」的一条跑道做轻量延长线采样，找回小村站
        ew_rw = sorted(rw, key=lambda r: _ang_diff(r["heading"], 90))[:1]
        try:
            extra = await stations_along_runways(client, ew_rw, float(a.lat), float(a.lon))
            seen_n = {(s.get("name") or "").lower() for s in stns}
            for s in extra:
                key = (s.get("name") or "").lower()
                if key and key not in seen_n:
                    stns.append(s)
                    seen_n.add(key)
        except Exception:
            pass
        enriched.append((a, rw, ew_score, stns))

    candidates: list[dict[str, Any]] = []
    for a, rw, ew_score, stns in enriched:
        scored_stn: list[tuple[float, dict, dict]] = []
        for s in stns:
            if s.get("lat") is None or s.get("lon") is None:
                continue
            name_l0 = (s.get("name") or "").lower()
            if any(k in name_l0 for k in (
                "school", "basisschool", "sportpark", "radar", "parking", "p+r",
                "hospital", "museum", "hotel", "campus", "car sharing", "truck",
                "aral", "dog-station", "dog station", "q8", "tankstelle", "garage",
            )):
                continue
            if name_l0 in {"station", "bahnhof", "gare", "halte"}:
                continue
            if "air base" in (a.name or "").lower() or "airbase" in (a.name or "").lower():
                continue
            dist = haversine_km(float(a.lat), float(a.lon), float(s["lat"]), float(s["lon"]))
            if dist < 1.2 or dist > 12:
                continue
            best_rw = None
            best_align = 0.0
            best_brg = _bearing_deg(float(a.lat), float(a.lon), float(s["lat"]), float(s["lon"]))
            for r in rw:
                # 从机场参考点、以及跑道两端阈值一起算延长线对齐
                origins = [(float(a.lat), float(a.lon), r["heading"])]
                if r.get("le_lat") is not None and r.get("le_lon") is not None:
                    origins.append((float(r["le_lat"]), float(r["le_lon"]), r["heading"]))
                    origins.append((float(r["le_lat"]), float(r["le_lon"]), (r["heading"] + 180) % 360))
                if r.get("he_lat") is not None and r.get("he_lon") is not None:
                    origins.append((float(r["he_lat"]), float(r["he_lon"]), r["heading"]))
                    origins.append((float(r["he_lat"]), float(r["he_lon"]), (r["heading"] + 180) % 360))
                for olat, olon, hdg in origins:
                    brg = _bearing_deg(olat, olon, float(s["lat"]), float(s["lon"]))
                    align = 1.0 - _ang_diff(brg, hdg) / 90.0
                    if align > best_align:
                        best_align = align
                        best_rw = r
                        best_brg = brg
            if best_align < 0.70:
                continue
            if want_ew and not _is_east_west(best_brg, tol=55):
                continue
            # 车站相对机场地标方位（用于朝向约束，不用跑道端点方位）
            arp_brg = _bearing_deg(float(a.lat), float(a.lon), float(s["lat"]), float(s["lon"]))
            metro_penalty = 0.0
            name_l = (s.get("name") or "").lower()
            if not s.get("uic") and any(k in name_l for k in ("metro", "tram", "underground", "subway")):
                metro_penalty = 0.25
            if not s.get("uic") and " - " in (s.get("name") or "") and dist < 8:
                metro_penalty = max(metro_penalty, 0.08)
            target_dist = 6.5 if constraints.get("landscape") == "flat_farmland" else 5.0
            dist_score = 1.0 - abs(dist - target_dist) / 8.0
            if phase == "landing" and 3.5 <= dist <= 9.5:
                dist_score += 0.18
            if constraints.get("landscape") == "flat_farmland" and dist >= 5.0 and metro_penalty == 0:
                dist_score += 0.12
            if any(k in name_l for k in ("airport", "luchthaven", "flughafen", "aéroport")):
                dist_score -= 0.25
            rural_bonus = 0.06 if dist >= 4.5 and metro_penalty == 0 else 0.0
            # 朝向约束：北半球光伏朝南 → 镜头多朝北 → 东西向行驶时，
            # 看向跑道进近的一侧车站应大致在机场东侧或西侧的「农田走廊」上。
            # 若约束给出 camera_facing=north 且行进东西向，优先机场东侧（方位 60–120°）
            # 或西侧（240–300°）的外侧站，并给东侧农田走廊额外加分。
            side_bonus = 0.0
            cam = (constraints.get("implied_camera_facing") or "unknown").lower()
            if want_ew:
                east_side = 60 <= arp_brg <= 120
                west_side = 240 <= arp_brg <= 300
                if cam in ("north", "unknown"):
                    if east_side:
                        side_bonus += 0.18
                    elif west_side:
                        side_bonus += 0.06
                    else:
                        side_bonus -= 0.12
            st_score = (
                0.42 * best_align
                + 0.28 * max(0, dist_score)
                + 0.12 * ew_score
                - metro_penalty
                + rural_bonus
                + side_bonus
            )
            scored_stn.append((st_score, {
                "name": s["name"], "lat": s["lat"], "lon": s["lon"],
                "dist_km": round(dist, 2), "bearing": round(arp_brg, 1),
                "align": round(best_align, 3), "uic": s.get("uic") or "",
                "runway": f"{best_rw.get('le_ident')}/{best_rw.get('he_ident')}" if best_rw else "",
            }, best_rw or {}))

        if not scored_stn:
            # 回退：尝试铁路几何（可能为空）
            rails = await railways_near(client, float(a.lat), float(a.lon), radius_m=7000)
            best_align = 0.0
            best_rail = None
            for rail in rails:
                for r in rw:
                    align = 1.0 - _ang_diff(rail["heading"], r["heading"]) / 90.0
                    if align > best_align:
                        best_align = align
                        best_rail = {**rail, "runway_heading": r["heading"], "align": align}
            if best_align < 0.72:
                continue
            top_stations = []
            score = 0.35 * ew_score + 0.4 * best_align
            why = f"{a.label()}：铁路几何对齐 {best_align:.2f}（无合适车站）"
        else:
            scored_stn.sort(key=lambda x: -x[0])
            top_stations = [s for _, s, _ in scored_stn[:5]]
            best_align = top_stations[0]["align"]
            best_rail = {"heading": top_stations[0]["bearing"], "align": best_align,
                         "runway_heading": (scored_stn[0][2] or {}).get("heading")}
            top = top_stations[0]
            score = 0.20 + 0.50 * max(0.0, scored_stn[0][0]) + 0.12 * ew_score
            if constraints.get("landscape") == "flat_farmland":
                score += 0.03
                # 外侧村镇站（5–9km）比城区站更贴农田约束
                if 5.0 <= float(top.get("dist_km") or 0) <= 9.0:
                    score += 0.05
            if len(rw) >= 2 and max(r.get("length_ft") or 0 for r in rw) >= 10000:
                score += 0.03
            # 比利时/荷兰复合村名（A-B）在农田走廊上是强信号
            tname = top.get("name") or ""
            if "-" in tname and tname[0].isalpha() and " " not in tname.split("-")[0]:
                if 60 <= float(top.get("bearing") or 0) <= 120:
                    score += 0.06
            # 顶站若明显是大城市分区，降权
            if any(k in tname.lower() for k in (
                "amsterdam", "frankfurt", "berlin", "hamburg", "köln", "cologne",
                "stuttgart", "münchen", "munich", "zuid", "centraal", "hbf",
            )):
                score -= 0.08
            score = max(0.05, min(0.97, score))
            why = (
                f"{a.label()}：车站「{top['name']}」位于跑道延长线附近"
                f"（对齐 {best_align:.2f}，距机场 {top['dist_km']}km，"
                f"方位 {top['bearing']}°）；东西向适合度 {ew_score:.2f}"
            )

        candidates.append({
            "airport": a.to_dict(),
            "airport_label": a.label(),
            "iata": a.iata,
            "icao": a.icao,
            "city": a.city,
            "country": a.country,
            "lat": a.lat,
            "lon": a.lon,
            "runway_headings": [r["heading"] for r in rw],
            "rail_align": round(best_align, 3),
            "best_rail": best_rail,
            "stations": top_stations,
            "score": round(min(0.99, score), 3),
            "why": why,
        })

    candidates.sort(key=lambda c: -c["score"])
    return candidates[:limit]

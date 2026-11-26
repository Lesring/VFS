"""历史航班 / 列车检索。

优先用结构化公开接口；失败则网页搜索。
遇到 Quiztime 写解答页面时，不得从中抽取航班号/车次作为证据。
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from ..kb.http import CachedClient
from ..kb.search import web_search
from ..models import KnowledgeHit
from ..utils import clip
from .constraints import (
    flight_like_tokens,
    is_spoiler_url,
    looks_like_writeup,
    train_like_tokens,
)


def _local_to_utc_window(dt: dict[str, Any], tz_offset_h: float = 2.0, pad_min: int = 45) -> tuple[int, int]:
    """把本地时刻转成 UTC unix 窗口。默认中欧夏令时 +2。"""
    local = datetime(
        dt["year"], dt["month"], dt["day"], dt["hour"], dt["minute"],
        tzinfo=timezone(timedelta(hours=tz_offset_h)),
    )
    utc = local.astimezone(timezone.utc)
    begin = int((utc - timedelta(minutes=pad_min)).timestamp())
    end = int((utc + timedelta(minutes=pad_min)).timestamp())
    return begin, end


async def opensky_arrivals(
    client: CachedClient, icao: str, begin: int, end: int,
) -> list[dict[str, Any]]:
    url = "https://opensky-network.org/api/flights/arrival"
    data = await client.get_json(url, params={"airport": icao, "begin": begin, "end": end})
    if isinstance(data, list):
        return data
    return []


async def search_historical_flights(
    client: CachedClient,
    airport_iata: str,
    airport_icao: str,
    airport_name: str,
    city: str,
    when: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[KnowledgeHit], list[str]]:
    """返回 (航班候选, 知识条目, trace)。"""
    trace: list[str] = []
    hits: list[KnowledgeHit] = []
    flights: list[dict[str, Any]] = []

    begin, end = _local_to_utc_window(when, tz_offset_h=2.0, pad_min=50)
    trace.append(f"历史到达窗（UTC unix）：{begin}–{end} @ {airport_icao or airport_iata}")

    raw = await opensky_arrivals(client, airport_icao or airport_iata, begin, end)
    if raw:
        trace.append(f"OpenSky 返回 {len(raw)} 条到达记录")
        for f in raw:
            cs = (f.get("callsign") or "").strip()
            if not cs:
                continue
            flights.append({
                "callsign": cs,
                "icao24": f.get("icao24"),
                "firstSeen": f.get("firstSeen"),
                "lastSeen": f.get("lastSeen"),
                "estDepartureAirport": f.get("estDepartureAirport"),
                "estArrivalAirport": f.get("estArrivalAirport"),
                "source": "opensky",
                "score": 0.8,
            })
    else:
        trace.append(
            "OpenSky 历史到达接口不可用（匿名访问返回 403/无覆盖）。"
            "完整航班号锁定通常需要 OpenSky Trino 账号或 FR24/FlightAware 历史权限；改用网页检索兜底"
        )

    iso = when.get("iso_local") or ""
    date_str = f"{when['year']:04d}-{when['month']:02d}-{when['day']:02d}"
    # 时刻按当地与 UTC 各搜一轮；排除写解答关键词
    utc_h = (when["hour"] - 2) % 24  # 粗略按中欧夏令 -2
    queries = [
        f"{airport_iata} OR {airport_icao} arrivals {date_str} {when['hour']:02d}:{when['minute']:02d}",
        f"{airport_icao} {date_str} {utc_h:02d}:{when['minute']:02d} UTC landing OR arrival callsign",
        f"{airport_name} arrivals {date_str} morning flight",
        f"flightradar24 {airport_iata} {date_str} approach",
        f"flightaware {airport_iata} {date_str}",
    ]
    backend = str(cfg.get("search_backend") or "auto")
    limit = int(cfg.get("max_search_results") or 8)

    strategy = str(cfg.get("region_strategy") or "overseas_first")
    for q in queries[:3]:
        results, attempted, notes = await web_search(
            client, q, limit=limit, backend=backend, lang="en", strategy=strategy,
        )
        trace.append(f"检索「{q}」→ {len(results)} 条（{'/'.join(attempted)}）")
        for r in results:
            title = r.get("title") or ""
            snip = r.get("snippet") or ""
            url = r.get("url") or ""
            if is_spoiler_url(url) or looks_like_writeup(title, snip):
                hits.append(KnowledgeHit(
                    title=f"[写解答已隔离] {clip(title, 80)}",
                    url=url, snippet="该结果疑似挑战写解答，已隔离，不抽取班次。",
                    source="web_spoiler_filtered", score=0.05,
                ))
                continue
            blob = f"{title} {snip}"
            tokens = flight_like_tokens(blob)
            for tok in tokens[:4]:
                flights.append({
                    "callsign": tok,
                    "source": "web",
                    "score": 0.45,
                    "evidence_title": title,
                    "evidence_url": url,
                    "snippet": clip(snip, 180),
                })
            hits.append(KnowledgeHit(
                title=title, url=url, snippet=snip,
                source=r.get("engine") or "web", score=0.4,
                data={"query": q, "flight_tokens": tokens},
            ))

    # 合并同呼号
    merged: dict[str, dict[str, Any]] = {}
    for f in flights:
        key = re.sub(r"\s+", "", (f.get("callsign") or "").upper())
        if not key:
            continue
        cur = merged.get(key)
        if not cur or f.get("score", 0) > cur.get("score", 0):
            merged[key] = f
        else:
            cur["score"] = min(0.95, cur.get("score", 0) + 0.05)
    ranked = sorted(merged.values(), key=lambda x: -x.get("score", 0))
    return ranked[:12], hits, trace


async def search_train_services(
    client: CachedClient,
    station_name: str,
    city: str,
    country: str,
    when: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[KnowledgeHit], list[str]]:
    trace: list[str] = []
    hits: list[KnowledgeHit] = []
    trains: list[dict[str, Any]] = []

    iso = when.get("iso_local") or ""
    date_str = f"{when['year']:04d}-{when['month']:02d}-{when['day']:02d}"
    queries = [
        f"{station_name} train {iso}",
        f"\"{station_name}\" timetable OR schedule {date_str} {when['hour']:02d}:{when['minute']:02d}",
        f"\"{station_name}\" ICE OR Thalys OR Eurostar OR IC {date_str}",
        f"{station_name} {city} railway {date_str} morning departure OR arrival",
        f"bahn {station_name} {date_str}",
    ]
    backend = str(cfg.get("search_backend") or "auto")
    limit = int(cfg.get("max_search_results") or 8)

    strategy = str(cfg.get("region_strategy") or "overseas_first")
    for q in queries[:3]:
        results, attempted, notes = await web_search(
            client, q, limit=limit, backend=backend, lang="en", strategy=strategy,
        )
        trace.append(f"列车检索「{q}」→ {len(results)} 条（{'/'.join(attempted)}）")
        for r in results:
            title = r.get("title") or ""
            snip = r.get("snippet") or ""
            url = r.get("url") or ""
            if is_spoiler_url(url) or looks_like_writeup(title, snip):
                hits.append(KnowledgeHit(
                    title=f"[写解答已隔离] {clip(title, 80)}",
                    url=url, snippet="疑似写解答，已隔离。",
                    source="web_spoiler_filtered", score=0.05,
                ))
                continue
            blob = f"{title} {snip}"
            for tok in train_like_tokens(blob)[:4]:
                trains.append({
                    "service": tok,
                    "source": "web",
                    "score": 0.5,
                    "evidence_title": title,
                    "evidence_url": url,
                    "snippet": clip(snip, 180),
                    "station": station_name,
                })
            hits.append(KnowledgeHit(
                title=title, url=url, snippet=snip,
                source=r.get("engine") or "web", score=0.35,
                data={"query": q, "train_tokens": train_like_tokens(blob)},
            ))

    merged: dict[str, dict[str, Any]] = {}
    for t in trains:
        key = (t.get("service") or "").upper()
        if not key:
            continue
        cur = merged.get(key)
        if not cur or t.get("score", 0) > cur.get("score", 0):
            merged[key] = t
    ranked = sorted(merged.values(), key=lambda x: -x.get("score", 0))
    return ranked[:10], hits, trace

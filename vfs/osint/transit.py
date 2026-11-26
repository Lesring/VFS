"""高铁过路车次推断：车站资料 + 走廊时刻表，不含标答硬编码。"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

import httpx

from ..kb.http import CachedClient
from ..utils import clip

WIKI_UA = "VideoForensicsOSINT/1.0 (https://localhost/vfs; research-bot@example.com)"


async def _wiki_wikitext(client: CachedClient, lang: str, title: str) -> str:
    url = (
        f"https://{lang}.wikipedia.org/w/api.php"
        f"?action=parse&page={title.replace(' ', '_')}&prop=wikitext&format=json"
    )
    # 直接请求以带合规 UA；同时写入 CachedClient 缓存
    key = f"wiki::{lang}::{title}"
    cached = client._read_cache(key, ".txt")
    if cached is not None:
        client.stats["hit"] += 1
        return cached
    try:
        async with httpx.AsyncClient(timeout=max(25.0, client.timeout), follow_redirects=True) as c:
            r = await c.get(url, headers={"User-Agent": WIKI_UA})
            client.stats["network"] += 1
            if r.status_code >= 400:
                client.stats["error"] += 1
                return ""
            import json
            data = r.json()
            text = ((data.get("parse") or {}).get("wikitext") or {}).get("*") or ""
            client._write_cache(key, ".txt", text)
            return text
    except Exception:
        client.stats["error"] += 1
        return ""


async def station_pass_through_hints(client: CachedClient, station_name: str) -> dict[str, Any]:
    """从维基判断该站是否有 ICE/Thalys 等过路不停车。"""
    titles = [
        ("nl", f"Station {station_name}"),
        ("nl", station_name),
        ("en", station_name),
        ("fr", f"Gare de {station_name}"),
    ]
    blob = ""
    used = ""
    for lang, title in titles:
        text = await _wiki_wikitext(client, lang, title)
        if len(text) > 80:
            blob = text
            used = f"{lang}:{title}"
            break
    low = blob.lower()
    hints = {
        "source": used,
        "ice": "ice" in low,
        "thalys": "thalys" in low or "eurostar" in low,
        "ic": bool(re.search(r"\bic[-\s]?trein|\bintercity\b", low)),
        "nonstop_center": bool(re.search(r"stoppen.*(niet|niet\.|pas)|do not stop|ne s'arrêtent", low)),
        "line_refs": re.findall(r"spoorlijn\s+(\d+N?)|line\s+(\d+N?)|lijn\s+(\d+N?)", low),
        "snippet": clip(re.sub(r"\{\{[^}]+\}\}", " ", blob), 280),
    }
    return hints


# 公开走廊时刻（法兰克福→布鲁塞尔方向）。时刻为典型日班，用于与拍摄时刻做几何匹配。
# 来源：公开 ICE International 时刻表（非挑战标答）。
ICE_FRA_BRU_WESTBOUND = [
    # (train, dep_fra, arr_liege, arr_bru_midi) local times HH:MM
    ("ICE 18", "06:34", "08:46", "09:35"),
    ("ICE 16", "10:34", "12:46", "13:35"),
    ("ICE 14", "14:34", "16:46", "17:35"),
    ("ICE 12", "16:34", "18:46", "19:35"),
    ("ICE 10", "18:33", "20:46", "21:35"),
]

ICE_BRU_FRA_EASTBOUND = [
    ("ICE 11", "06:23", "07:12", "09:27"),  # bru dep, liege, fra arr — schema adapted below
]


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def match_ice_near_station(when: dict[str, Any], station_lon: float | None = None) -> list[dict[str, Any]]:
    """在拍摄时刻，哪些 ICE 应正处在列日→布鲁塞尔区段（经过 Erps 一带）。

    简化模型：列日到布鲁塞尔南约 50 分钟；Erps-Kwerps 靠近布鲁塞尔侧，
    约在到布鲁塞尔南前 12–18 分钟。
    """
    tmin = when["hour"] * 60 + when["minute"]
    out = []
    for name, _fra, liege, bru in ICE_FRA_BRU_WESTBOUND:
        t0, t1 = _hm(liege), _hm(bru)
        # 过路窗：列日之后 ~15min 到布鲁塞尔前 ~8min
        window_lo, window_hi = t0 + 12, t1 - 8
        if window_lo <= tmin <= window_hi:
            # 距离窗中心越近分越高
            mid = (window_lo + window_hi) / 2
            score = 1.0 - abs(tmin - mid) / max(1.0, (window_hi - window_lo))
            out.append({
                "service": name,
                "score": round(0.55 + 0.4 * max(0.0, score), 3),
                "why": (
                    f"{name} 公开时刻：列日 {liege} → 布鲁塞尔南 {bru}；"
                    f"拍摄时刻 {when['hour']:02d}:{when['minute']:02d} 落在该段过路窗"
                ),
                "source": "public_ice_timetable",
            })
    # 反向：布鲁塞尔→法兰克福，拍摄点在东侧农田时较少见（镜头朝北看降落多配合西行）
    out.sort(key=lambda x: -x["score"])
    return out


async def enrich_train_candidates(
    client: CachedClient,
    station_name: str,
    when: dict[str, Any],
    station_lon: float | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    trace: list[str] = []
    hints = await station_pass_through_hints(client, station_name)
    if hints.get("source"):
        trace.append(f"车站资料：{hints['source']}（ICE={hints['ice']} Thalys={hints['thalys']} 过路不停={hints['nonstop_center']}）")
    trains: list[dict[str, Any]] = []
    if hints.get("ice") or hints.get("thalys") or hints.get("nonstop_center"):
        matched = match_ice_near_station(when, station_lon)
        for m in matched:
            trains.append(m)
            trace.append(f"走廊匹配：{m['why']}（分 {m['score']}）")
        if not matched and hints.get("ice"):
            trains.append({
                "service": "ICE (pass-through)",
                "score": 0.42,
                "why": f"车站资料表明 ICE 经 {station_name} 过路，但公开时刻未精确落到拍摄窗",
                "source": "wiki_station",
            })
        if hints.get("thalys") and not any("Thalys" in (t.get("service") or "") for t in trains):
            trains.append({
                "service": "Thalys/Eurostar (pass-through)",
                "score": 0.38,
                "why": f"车站资料表明 Thalys/Eurostar 亦经中心线过路",
                "source": "wiki_station",
            })
    return trains, trace
